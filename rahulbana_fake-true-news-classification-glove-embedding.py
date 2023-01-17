!wget -c "http://nlp.stanford.edu/data/glove.6B.zip"
#!wget -c "http://nlp.stanford.edu/data/glove.42B.zip"
#!wget -c "http://nlp.stanford.edu/data/glove.840B.zip"
#!wget -c "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
!unzip glove.6B.zip
!ls
import nltk
nltk.download('stopwords')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from wordcloud import WordCloud

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM
from tensorflow.keras.layers import Embedding, Dropout, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import model_from_json

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


from bs4 import BeautifulSoup
import re,string,unicodedata

import pickle
import h5py
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.tokenize import word_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import os
for dirname, _, filenames in os.walk('/kaggle/input/fake-and-real-news-dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df_fake = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
df_true = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")
df_fake['label'] = 1
df_true['label'] = 0
df = pd.concat([df_fake, df_true])
df['text'] = df['text'] + " " + df['title']
del df['title']
del df['subject']
del df['date']
features, labels = df['text'].tolist(), df['label'].tolist()
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

#remove html tags
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


# Removing URL's
def remove_urls(text):
    return re.sub(r'http\S+', '', text)


#Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return " ".join(final_text)


def clean_data(feature_list):
    feature_list = list(map(lambda x: x.lower(), feature_list))
    feature_list = list(map(strip_html, feature_list))
    feature_list = list(map(decontracted, feature_list))
    feature_list = list(map(remove_between_square_brackets, feature_list))
    feature_list = list(map(remove_urls, feature_list))
    feature_list = list(map(remove_stopwords, feature_list))
    
    return feature_list
features = clean_data(features)
plt.figure(figsize = (20,20)) # Text that is Fake
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop).generate(" ".join(df[df.label == 1].text))
plt.imshow(wc , interpolation = 'bilinear')
plt.figure(figsize = (20,20)) # Text that is not Fake
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop).generate(" ".join(df[df.label == 0].text))
plt.imshow(wc , interpolation = 'bilinear')
#max length of each text 
MAX_SEQUENCE_LENGTH = 500

#only take 10000 words from all unique words(or only 10000 features max)
MAX_NUM_WORDS = 10000

#each word should be represented by 300 dimension
EMBEDDING_DIM = 200

#ratio of train/test will be 80/20
VALIDATION_SPLIT = 0.2

#path for glove embedding file
glove_txt_path = "glove.6B.200d.txt"
x_train,x_test,y_train,y_test = train_test_split(features, labels, stratify=labels, random_state = 42, test_size=VALIDATION_SPLIT, shuffle=True)
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train)
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
def create_tokens(tokenizer, data, max_seq_len):
    tokenized_data = tokenizer.texts_to_sequences(data)
    padded_tokenized_data = pad_sequences(tokenized_data, maxlen=max_seq_len)
    return padded_tokenized_data
    
X_train = create_tokens(tokenizer, x_train, MAX_SEQUENCE_LENGTH)
X_test = create_tokens(tokenizer, x_test, MAX_SEQUENCE_LENGTH)
y_test = np.array(y_test)
y_train = np.array(y_train)
glove_txt_file = open(glove_txt_path, "r", encoding="utf8")
embeddings_index = {}
for line in glove_txt_file:
    values = line.split()
    word = ''.join(values[:-EMBEDDING_DIM])
    coefs = np.asarray(values[-EMBEDDING_DIM:], dtype='float32')
    embeddings_index[word] = coefs
glove_txt_file.close()
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

print('create embedding matrix')
word_index = tokenizer.word_index
nb_words = min(MAX_NUM_WORDS, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NUM_WORDS: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
del embeddings_index
def create_model(embed_matrix, max_num_words, embed_dim, max_seq_len):
    model = Sequential()
    model.add(Embedding(max_num_words, output_dim=embed_dim, weights=[embed_matrix], input_length=max_seq_len, trainable=False))  

    model.add(Conv1D(filters=128, kernel_size=4, activation='relu'))
    model.add(MaxPooling1D(4))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(units = 128 , activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid')) #here activation function is sigmoid because we want only one output 0/1
    
    return model

#creating model instance
model = create_model(embedding_matrix, MAX_NUM_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)

#compile mode with optimizer = adam, loss = binary_crossentropy, metrics = accuracy
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 128
epochs = 10

#setting callback function for reducing learning rate
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

#setting callback functiob for sarly stopping if loss is not decreasing
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


history = model.fit(X_train, 
                    y_train, 
                    batch_size = batch_size , 
                    validation_data = (X_test,y_test) ,
                    epochs = epochs, 
                    shuffle=True,
                    callbacks = [learning_rate_reduction, es])
accr_train = model.evaluate(X_train,y_train)
print('Accuracy Train: {}'.format(accr_train[1]*100))
accr_test = model.evaluate(X_test,y_test)
print('Accuracy Test: {}'.format(accr_test[1]*100))
pred = model.predict_classes(X_test)
cf_matrix = confusion_matrix(y_test,pred)
sns.heatmap(cf_matrix, annot=True, fmt='g')
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
# serialize model to json
json_model = model.to_json()

#save the model architecture to JSON file
with open('fake_true_news_model.json', 'w') as json_file:
    json_file.write(json_model)

#saving the weights of the model
model.save_weights('fake_true_news_weights.h5')
!ls