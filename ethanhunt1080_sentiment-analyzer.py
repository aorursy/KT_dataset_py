# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import re



import nltk 

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



from wordcloud import WordCloud



from PIL import Image



from sklearn.model_selection import train_test_split



from keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



import gensim.models.keyedvectors as word2vec



from tensorflow.keras.layers import Embedding,Dense,LSTM,Bidirectional

from tensorflow.keras.layers import BatchNormalization,Dropout



from tensorflow.keras import Sequential



import pickle
cols = ["target", "ids", "date", "flag", "user", "text"]

encoding = 'latin'

df = pd.read_csv('/kaggle/input/sentiment140/training.1600000.processed.noemoticon.csv',encoding=encoding,names=cols)

df.head()
print('Length of dataset is:',len(df))
df.info()
#The polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)

#Let's look at the count

plt.figure(figsize=(16,8))

sns.countplot(df['target'])

plt.title('Distribution of tweets',fontsize = 15)

plt.show()
#Here we can see that only 0 and 4 are present.Let's substitute 0 to 0 and 4 to 1 for convenience sake

df['target'].replace({0:0,4:1},inplace = True)
#Removing links,special character,@usernames and stopwords.

#Also applying lemmatization to the words



pattern = '@\S+|https?:\S+|http?:\S|[^A-Za-z]+|com|net'



stop_words = stopwords.words('english')

lemma = WordNetLemmatizer()



def preprocess(text):

    text = re.sub(pattern, ' ', str(text).lower()).strip()

    tokens = []

    for token in text.split():

        if token not in stop_words:

            tokens.append(lemma.lemmatize(token))

    return ' '.join(tokens)
df.text = df.text.apply(lambda x: preprocess(x))

df.head()
#Positive words

pos = ' '.join(df[df['target'] == 1].text)







img_pos = np.array(Image.open('../input/mask-img/6ir5bE75T.jpg'))

plt.figure(figsize = (15,9))

pw = WordCloud(mask = img_pos).generate(pos)

plt.imshow(pw)
#Negative words

neg = ' '.join(df[df['target'] == 0].text)



img_neg = np.array(Image.open('../input/mask-img/angry-emoji_53876-25519.jpg'))

plt.figure(figsize = (15,9))

nw = WordCloud(mask = img_neg).generate(neg)

plt.imshow(nw)
test_size = 0.2



df_train,df_test = train_test_split(df,test_size = test_size)
df_train.head()
df_train_clean = df_train[['target','text']]

df_test_clean = df_test[['target','text']]
df_train_clean.head()
max_length = 20

trunc_type = 'post'

padd_type = 'post'





tokens = Tokenizer()

tokens.fit_on_texts(df_train_clean['text'])



training_seq = tokens.texts_to_sequences(df_train_clean['text'])

X_train = pad_sequences(training_seq,maxlen = max_length,padding=padd_type,truncating=trunc_type)



testing_seq =  tokens.texts_to_sequences(df_test_clean['text'])

X_test = pad_sequences(testing_seq,maxlen = max_length,padding=padd_type,truncating=trunc_type)

print('Shape of X_train is ',X_train.shape)

print('Shape of X_test is ',X_test.shape)
#Declaring target labels 

y_train = df_train_clean['target']

y_test = df_test_clean['target']
#Converting everything to numpy arrays



X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)
word2vec_dict = word2vec.KeyedVectors.load_word2vec_format('../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin',binary = True)





embeddings_index = dict()

for word in word2vec_dict.vocab:

    embeddings_index[word] = word2vec_dict.word_vec(word)

    

print('Length of word2vec dictionary is :',len(embeddings_index))
vocab_size = len(tokens.word_index) + 1



embed_size = 300

embedding_matrix = np.zeros((vocab_size,embed_size))



for word,tok in tokens.word_index.items():

    if word in embeddings_index.keys():

        embedding_vector = embeddings_index[word]

        embedding_matrix[tok] =  embedding_vector



print('Shape of Embedding Matrix is :',embedding_matrix.shape)
#Initialising Embedding Layer



embedding_layer = Embedding(vocab_size,embed_size,weights = [embedding_matrix],input_length = max_length,trainable = False)

model = Sequential()

model.add(embedding_layer)

model.add(Bidirectional(LSTM(64,return_sequences = True)))

model.add(BatchNormalization())

model.add(Bidirectional(LSTM(64)))

model.add(Dropout(0.4))

model.add(Dense(64,activation = 'relu'))

model.add(Dense(32,activation = 'relu'))

model.add(Dropout(0.4))

model.add(Dense(1,activation = 'sigmoid'))



model.summary()



model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

#Training

batch_size = 16384

num_epochs = 30



hist = model.fit(X_train,y_train,batch_size = batch_size,epochs = num_epochs,verbose = 1)
score = model.evaluate(X_test, y_test, batch_size=batch_size)

print()

print("Accuracy of model is :",round(score[1],2))

print("Loss of model is :",round(score[0],2))
acc = hist.history['accuracy']

loss = hist.history['loss']

epochs = range(len(acc))



plt.plot(epochs, acc, 'b', label='Training acc')

plt.title('Training accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()



plt.show()
plt.plot(epochs, loss, 'r', label='Training loss')

plt.title('Training loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()



plt.show()
def decode_sentiment(score):

    label = None

    if score <= 0.4:

        label = 'Negative'

    elif score >= 0.7:

        label = 'Positive'

    else:

        label = 'Neutral'

    

    return label

def predict(text):

    x_test = pad_sequences(tokens.texts_to_sequences([text]), maxlen=max_length,padding=padd_type,truncating=trunc_type)

    score = model.predict([x_test])[0]

    

    label = decode_sentiment(score)



    return {"label": label, "score": float(score)}  
predict('I love fishing')
predict('Economy is going down')
model.save('model.h5')

pickle.dump(tokens,open('tokens.pkl','wb'),protocol = 0)