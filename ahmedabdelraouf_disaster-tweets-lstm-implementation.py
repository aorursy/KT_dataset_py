import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import string
import re
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
stop = set(stopwords.words('english'))
print(stopwords.words('english')[:20])
#loading the data
train_set = pd.read_csv("../input/disaster-tweet/train.csv")
test_set = pd.read_csv("../input/disaster-tweet/test.csv")

train_set.shape , test_set.shape
#checking the data
train_set.head()
train_set.tail()
train_set.text[0]
train_set['target'].unique()
#class distribution
# 0 (for Non Disaster) is more than 1(for disaster) tweets
class_dist = train_set.target.value_counts()
sns.barplot(class_dist.index , class_dist)
#checking null values
null_vals = train_set.isnull().sum()
sns.barplot(null_vals.index , null_vals)
#cleaning the dataset
def remove_spec(text):
    text = re.sub('<.*?>+' , '' , text)
    text = text.lower()
    return text

#removing punctuations
def remove_punctuation(text):
    table = str.maketrans('','',string.punctuation)
    return text.translate(table)

#def remove urls
def remove_urls(text):
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+" , '' , text)
    return text

#remove emojis
def remove_emoji(text):
    emojis = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                        "]+" , flags=re.UNICODE)
    text = re.sub(emojis , '' , text)
    return text
train_set["cleaned_text"] = train_set['text'].apply(lambda x : remove_punctuation(x))
train_set["cleaned_text"] = train_set["cleaned_text"].apply(lambda x: remove_urls(x))
train_set["cleaned_text"] = train_set["cleaned_text"].apply(lambda x: remove_emoji(x))
train_set["cleaned_text"] = train_set["cleaned_text"].apply(lambda x: remove_spec(x))
train_set["cleaned_text"].head()
train_set["cleaned_text"][7610]
#creting words corpus
def create_corpus(dataset):
    corpus = []
    for review in tqdm(dataset["cleaned_text"]):
        words = [word.lower() for word in word_tokenize(review) if (word.isalpha() == 1) & (word not in stop)]
        corpus.append(words)
        
    return corpus

corpus = create_corpus(train_set)
corpus
#embedding dictionary

embedding_dict = {}

with open("../input/glove6b100dtxt/glove.6B.100d.txt" , encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:] , 'float32')
        embedding_dict[word] = vectors
        
f.close()
embedding_dict
#sentence tokenizaion
max_len = 20
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

sequences = tokenizer.texts_to_sequences(corpus)
corpus_pad = pad_sequences(sequences , maxlen= max_len , padding='post' , truncating='post')
#unique word present
word_index = tokenizer.word_index
print(f"The Number of unique words = {len(word_index)}")
#creating embedding matrix using embedding_dict

num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words , 100))

for word , i in tqdm(word_index.items()):
    if i > num_words:
        continue
        
    emb_vect = embedding_dict.get(word)
    if emb_vect is not None:
        embedding_matrix[i] = emb_vect  
#ceating model
model = keras.models.Sequential([
    keras.layers.Embedding(num_words , 100 , embeddings_initializer = keras.initializers.Constant(embedding_matrix) ,
                          input_length = max_len , trainable = False),
    keras.layers.SpatialDropout1D(0.4),
    keras.layers.LSTM(64 , dropout = 0.2 , recurrent_dropout = 0.2),
    keras.layers.Dense(1 , activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy' , optimizer = keras.optimizers.Adam(learning_rate=1e-4) , metrics = ['accuracy'])
model.summary()
Xtrain , Xtest , ytrain , ytest = train_test_split(corpus_pad , train_set['target'].values , test_size = 0.2 , random_state = 42)
Xtrain.shape , ytrain.shape 
Xtrain , Xvalid = Xtrain[:4500 , :] , Xtrain[4500: , :]
ytrain , yvalid = ytrain[:4500] , ytrain[4500:]
Xtrain.shape , Xvalid.shape
history = model.fit(Xtrain , ytrain , batch_size=32 , epochs=50 , validation_data = (Xvalid , yvalid) , verbose = 2)
#Accuracy vs epoch
plt.title('Accuracy')
plt.plot(history.history['accuracy'] , label = 'train')
plt.plot(history.history['val_accuracy'] , label = 'test')
plt.legend()
plt.show()
#Loss vs Epoch
epoch_count = range(1 , len(history.history['loss'])+1)
plt.plot(epoch_count , history.history['loss'] , 'r--')
plt.plot(epoch_count , history.history['val_loss'] , 'b-')

plt.legend(['Training Loss' , 'Validation Loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
#prediction
#clean test data
test_set['cleaned_text'] = test_set["text"].apply(lambda x : remove_punctuation(x))
test_set['cleaned_text'] = test_set["cleaned_text"].apply(lambda x : remove_emoji(x))
test_set["cleaned_text"] = test_set["cleaned_text"].apply(lambda x : remove_urls(x))
test_set['cleaned_text'] = test_set["cleaned_text"].apply(lambda x : remove_spec(x))
#creating test corpus
test_corpus = create_corpus(test_set)
#Encoding test text to sequences
test_sequences  = tokenizer.texts_to_sequences(test_corpus)
test_corpus_pad = pad_sequences(test_sequences , maxlen= max_len , padding = 'post' , truncating = "pre")
prediction = model.predict(test_corpus_pad)
print(prediction)
print("-"*100)
print(prediction.shape)
prediction = np.round(prediction).astype(int).reshape(3263)
prediction[20:50]
#creating submission file
submission = pd.DataFrame({'id':test_set['id'] , 'target':prediction})
submission.to_csv("submission.csv" , index = False)
submission.head(20)
