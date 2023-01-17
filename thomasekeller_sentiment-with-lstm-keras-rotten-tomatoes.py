# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
pd.set_option('display.max_colwidth', -1)


# Any results you write to the current directory are saved as output.
df=pd.read_table('../input/train.tsv')
df.head()
df['Phrase'][:10]
sns.countplot(df.Sentiment)
#text length
df['tlen']=df['Phrase'].apply(len)
df[['Sentiment','tlen','Phrase']].head()
df['tlen'].describe()
sns.distplot(df['tlen'])
g = sns.FacetGrid(df,col='Sentiment')
g.map(sns.distplot,'tlen')
sns.boxplot(x='Sentiment',y='tlen',data=df,palette='cubehelix')
sns.heatmap(df[['Sentiment','tlen']].corr(), annot = True, cmap='plasma')
from nltk.corpus import stopwords
from wordcloud import WordCloud

text = df['Phrase'].to_string()
#wordcloud = WordCloud(
#        relative_scaling=0.5,
#        stopwords=set(stopwords.words('english'))).generate(text)
wordcloud = WordCloud(
    relative_scaling=0.5,
).generate(text)
    




plt.figure(figsize=(12,12))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
# Encode Categorical Variable
from keras.utils import to_categorical
X = df['Phrase']
y = to_categorical(df['Sentiment'])
num_classes = df['Sentiment'].nunique()
y


# good practice to set a seed so you have identical results in future runs for comparison
seed = 42 
np.random.seed(seed)


# Spilt Train Test sets
#test_size is how much do you subset the training data into a validation set
#usually .2-.3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3,
                                                    stratify=y,
                                                    random_state=seed)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Tokenize Text
from keras.preprocessing.text import Tokenizer
max_features = 15000
#The tokenizer filters based on word frequency by default
#keeping the N most common words, here 15,000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
#The word distribution of phrases
totalNumWords = [len(one_comment) for one_comment in X_train]
sns.distplot(totalNumWords,bins = 30)
from keras.preprocessing import sequence
max_words = 30 #max(totalNumWords) 

X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape,X_test.shape)
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,CuDNNLSTM
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

batch_size = 256
epochs = 10
def get_model(max_features, embed_dim):
    np.random.seed(seed)
    K.clear_session()
    model = Sequential()
    model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
    model.add(LSTM(128, dropout=0.25, recurrent_dropout=0.25))
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
    return model_history
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
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    
    hist=model_history.history
    dat={'val':hist['acc']+hist['val_acc'],'epochs':list(range(1,len(hist['acc'])+1))*2,
         'unit':np.repeat(['acc','val_acc'],len(hist['acc'])),
         'labels':np.repeat(['acc','val_acc'],len(hist['acc']))}
    df=pd.DataFrame(dat)
    sns.tsplot(df,value='val',time='epochs',unit='unit',condition='labels',ax=axs[0])
    #sns.tsplot(model_history.history['acc'],model_history.history['val_acc'],ax=axs[0])
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    dat={'val':hist['loss']+hist['val_loss'],'epochs':list(range(1,len(hist['loss'])+1))*2,
         'unit':np.repeat(['loss','val_loss'],len(hist['loss'])),
         'labels':np.repeat(['loss','val_loss'],len(hist['loss']))}
    df=pd.DataFrame(dat)
    sns.tsplot(df,value='val',time='epochs',unit='unit',condition='labels',ax=axs[1])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
def model_evaluate(model): 
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
max_features = 15000
embed_dim = 100
model = get_model(max_features, embed_dim)
mod_hist=model_train(model)
hist=mod_hist.history
dat={'val':hist['acc']+hist['val_acc'],'epochs':list(range(len(hist['acc'])))*2,'unit':np.repeat(['acc','val_acc'],len(hist['acc'])),'labels':np.repeat(['acc','val_acc'],len(hist['acc']))}
print(dat)
df=pd.DataFrame(dat)
print(df)
#Overfitting past 2 epochs, so we do early stopping by setting epochs to 2
final=model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                          epochs=2, batch_size=batch_size, verbose=2)
# evaluate model with test set
model_evaluate(model)
