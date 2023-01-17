# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
dataset = '../input/bitcointweets.csv'

data = pd.read_csv(dataset, header = None)

data
data.rename(columns={0: 'Data',1 : 'tweet', 2 : 'username' , 3 : 'len' , 5 : 'token' , 6 : 'link' , 7 : 'label'})
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', -1)

data = data[[1,7]]

data.columns = ['tweet','label']

data.head()
data.tail()
data.info()
sns.countplot(data['label'])
data['text_length'] = data['tweet'].apply(len)

data[['label','text_length','tweet']].head()
data['text_length'].describe()
data['text_length'].hist(bins=50)
g = sns.FacetGrid(data,col='label')

g.map(plt.hist,'text_length')
from nltk.corpus import stopwords

from wordcloud import WordCloud

import re



def clean_text(s):

    s = re.sub(r'http\S+', '', s)

    s = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', s)

    s = re.sub(r'@\S+', '', s)

    s = re.sub('&amp', ' ', s)

    return s

data['clean_tweet'] = data['tweet'].apply(clean_text)



text = data['clean_tweet'].to_string().lower()    

wordcloud = WordCloud(

    collocations=False,

    relative_scaling=0.5,

    stopwords=set(stopwords.words('english'))).generate(text)



plt.figure(figsize=(12,12))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
seed = 101 # fix random seed for reproducibility

np.random.seed(seed)
# Encode Categorical Variable

X = data['clean_tweet']

y = pd.get_dummies(data['label']).values

num_classes = data['label'].nunique()

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 

                                                    test_size=0.2,

                                                    stratify=y,

                                                    random_state=seed)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Tokenize Text

from keras.preprocessing.text import Tokenizer

max_features = 20000

tokenizer = Tokenizer(num_words=max_features)

tokenizer.fit_on_texts(list(X_train))

X_train = tokenizer.texts_to_sequences(X_train)

X_test = tokenizer.texts_to_sequences(X_test)

totalNumWords = [len(one_comment) for one_comment in X_train]

plt.hist(totalNumWords,bins = 30)

plt.show()
from keras.preprocessing import sequence

max_words = 30

X_train = sequence.pad_sequences(X_train, maxlen=max_words)

X_test = sequence.pad_sequences(X_test, maxlen=max_words)

print(X_train.shape,X_test.shape)
import keras.backend as K

from keras.models import Sequential

from keras.layers import Dense,Embedding,Conv1D,MaxPooling1D,LSTM

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



batch_size = 128

epochs = 5
def get_model(max_features, embed_dim):

    np.random.seed(seed)

    K.clear_session()

    model = Sequential()

    model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))

    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

    model.add(MaxPooling1D(pool_size=2))    

    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    return model
def model_train(model):

    # train the model

    model_history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 

                          epochs=epochs, batch_size=batch_size, verbose=2)

    # plot train history

    plot_model_history(model_history)
def plot_model_history(model_history):

    fig, axs = plt.subplots(1,2,figsize=(15,5))

    # summarize history for accuracy

    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])

    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)

    axs[0].legend(['train', 'val'], loc='best')

    # summarize history for loss

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)

    axs[1].legend(['train', 'val'], loc='best')

    plt.show()
def model_evaluate(): 

    # predict class with test set

    y_pred_test =  model.predict_classes(X_test, batch_size=batch_size, verbose=0)

    print('Accuracy:\t{:0.1f}%'.format(accuracy_score(np.argmax(y_test,axis=1),y_pred_test)*100))

    

    #classification report

    print('\n')

    print(classification_report(np.argmax(y_test,axis=1), y_pred_test))



    #confusion matrix

    confmat = confusion_matrix(np.argmax(y_test,axis=1), y_pred_test)



    fig, ax = plt.subplots(figsize=(4, 4))

    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

    for i in range(confmat.shape[0]):

        for j in range(confmat.shape[1]):

            ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('Predicted label')

    plt.ylabel('True label')

    plt.tight_layout()
# train the model

max_features = 20000

embed_dim = 100

model = get_model(max_features, embed_dim)

model_train(model)
model_evaluate()