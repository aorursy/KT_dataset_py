# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
# Load Data

df = pd.read_csv('../input/bitcointweets.csv', header = None)

pd.set_option('display.max_colwidth', -1)

df.head()
df = df[[1,7]]

df.columns = ['tweet','label']

df.head()
df.tail()
df.info()
df.shape
df.describe()
sns.countplot(df["label"])
df["text_length"] = df["tweet"].apply(len)
df[["label","text_length","tweet"]].head()
df["text_length"].describe()
df["text_length"].hist(bins=20)
g = sns.FacetGrid(df,col = "label")

g.map(plt.hist,"text_length")
sns.boxenplot(x="label",y="text_length",data=df, palette="rainbow")
#word cloud

from nltk.corpus import stopwords

from wordcloud import WordCloud
import re



def clean_text(s):

    s = re.sub(r'http\S+', '', s)

    s = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', s)

    s = re.sub(r'@\S+', '', s)

    s = re.sub('&amp', ' ', s)

    return s



df['clean_tweet'] = df['tweet'].apply(clean_text)
df.head()
text = df["clean_tweet"].to_string()

wordcloud = WordCloud(relative_scaling=0.5 , background_color='white',stopwords=set(stopwords.words('english'))).generate(text)

plt.figure(figsize=(12,12))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
X = df["clean_tweet"]

y = pd.get_dummies(df["label"].values)

y.head()
y = np.array(y)

num_classes = df["label"].nunique()

print(y)

print(np.argmax(y[0:5],axis=1))

num_classes
# setting seed to have identical result in future run for comparisons

seed = 42

np.random.seed(seed)

from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = seed)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from keras.preprocessing.text import Tokenizer

max_features = 20000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(X_train))

X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)
totalNumWords = [len(one_cleaned_tweet) for one_cleaned_tweet in X_train]

plt.hist(totalNumWords,bins=30)

plt.show()
X_train[7]
max(totalNumWords)
from keras.preprocessing import sequence

max_words = max(totalNumWords)

X_train = sequence.pad_sequences(X_train , maxlen = max_words)

X_test = sequence.pad_sequences(X_test , maxlen = max_words)

print(X_train.shape,X_test.shape)
X_train[7]
import keras.backend as K

from keras.models import Sequential

from keras.layers import Dense,Embedding,LSTM

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
batch_size = 128

epochs = 2
def get_model(max_features , embed_dim):

    np.random.seed(seed)

    K.clear_session()

    model = Sequential()

    model.add(Embedding(max_features , embed_dim , input_length=X_train.shape[1]))

    model.add(LSTM(100 , dropout=0.2 , recurrent_dropout=0.2))

    model.add(Dense(num_classes , activation='softmax'))

    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

    print(model.summary())

    return model

    
def model_train(model):

    #training the model

    model_history = model.fit(X_train , y_train , validation_data = (X_test , y_test), 

                              epochs = epochs ,batch_size= batch_size,verbose = 2)

    #plotting train history

    plot_model_history(model_history)

def plot_model_history(model_history):

    fig , axs = plt.subplots( 1 , 2 , figsize=(15,5))

    

    #Summarize history for accuracy

    

    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])

    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])

    

    axs[0].set_title("Model Accuracy")

    axs[0].set_ylabel("Accuracy")

    axs[0].set_xlabel("Epoch")

    

    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)

    

    axs[0].legend(['train', 'val'], loc='best')

    

    #Summarize history for loss

    

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    

    axs[1].set_title("Model Loss")

    axs[1].set_ylabel("Loss")

    axs[1].set_xlabel("Epoch")

    

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    

    axs[1].legend(['train', 'val'], loc='best')

    

    plt.show()

    
def model_evaluate():

    #predict classes with test set

    y_pred_test = model.predict_classes(X_test , batch_size = batch_size, verbose =0)

    print("Predicted ", y_pred_test[:50])

    print("True " , np.argmax(y_test[:50],axis = 1))

    print('Accuracy:\t{:0.1f}%'.format(accuracy_score(np.argmax(y_test,axis = 1),y_pred_test)*100))

    

    #Classification Report

    print("\n")

    print(classification_report(np.argmax(y_test, axis =1),y_pred_test))

    

    #Confusion Matrix

    confmat = confusion_matrix(np.argmax(y_test , axis = 1), y_pred_test)

    fig , ax = plt.subplots(figsize=(4,4))

    ax.matshow(confmat , cmap =plt.cm.Blues , alpha = 0.3)

    

    for i in range(confmat.shape[0]):

        for j in range(confmat.shape[1]):

            ax.text( x = j , y = i , s =confmat[i,j] , va = 'center' , ha = 'center')

    

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.tight_layout()
max_features = 20000

embed_dim =100

model = get_model(max_features,embed_dim)

model_train(model)
model_evaluate()