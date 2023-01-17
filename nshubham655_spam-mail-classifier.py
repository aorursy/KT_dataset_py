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
df=pd.read_csv('/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv')
df.head()
df.columns
df.loc[0]['text']
df.loc[3]['text']
df.isnull().sum()
df['label'].value_counts()
ham=df[df['label']=='ham']

ham.head()
spam=df[df['label']=='spam']

spam.head()
ham.shape,spam.shape
ham=ham.sample(spam.shape[0])

ham.shape
data=ham.append(spam)

data.shape
data.head()
data['length']=data['text'].apply(lambda x:len(x))

data.head()
import matplotlib.pyplot as plt

%matplotlib inline
plt.hist(data[data['label']=='ham']['length'],bins=100,alpha=0.7)

plt.hist(data[data['label']=='spam']['length'],bins=100,alpha=0.7)

plt.xlim(0,5000)

plt.show()
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
data.head()
X_train, X_test,y_train,y_test=train_test_split(data['text'],data['label'],test_size=0.2,random_state=0,shuffle=True,stratify=data['label'])
tfidf=TfidfVectorizer()

#X_train=tfidf.fit_transform(X_train)
X_train.shape
X_train
clf=Pipeline([('tfidf',TfidfVectorizer()),('clf',RandomForestClassifier(n_estimators=200,n_jobs=-1))])
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
cm
report =classification_report(y_test,y_pred)

print(report)
acc=accuracy_score(y_test,y_pred)
acc
clf=Pipeline([('tfidf',TfidfVectorizer()),('clf',SVC(C=1000))])
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
cm
report =classification_report(y_test,y_pred)

print(report)
acc=accuracy_score(y_test,y_pred)
acc
df.head()
df.shape
import spacy

nlp=spacy.load('en_core_web_sm')
def get_vector(x):

    doc=nlp(x)

    return doc.vector.reshape(-1,1)
vec=get_vector("hello this is shubham")

vec.shape
df['vector']=df['text'].apply(lambda x:get_vector(x))
X=np.concatenate(df['vector'],axis=1)

X=np.transpose(X)

X.shape
y=df['label_num']

y=y.values.reshape(len(y),1)

y.shape
import tensorflow as tf

tf.keras.backend.clear_session() 
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout,Conv1D,Flatten

from tensorflow.keras.utils import to_categorical
y_oh=to_categorical(y)

y_oh.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y_oh,test_size=0.2,random_state=0,shuffle=True)
model=Sequential([

    Dense(128,activation='relu',input_shape=([96])),

    

   #Conv1D(64,5,activation='relu'),

    #Flatten(),

    Dense(64,activation='relu'),

 

    Dense(128,activation='relu'),

    

    Dense(2,activation='sigmoid')

])
model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

history=model.fit(X_train,y_train,batch_size=8,epochs=20,validation_data=[X_test,y_test])
import matplotlib.image  as mpimg

import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc=history.history['accuracy']

val_acc=history.history['val_accuracy']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, 'r')

plt.plot(epochs, val_acc, 'b')

plt.title('Training and validation accuracy')

plt.xlabel("Epochs")

plt.ylabel("Accuracy")

plt.legend(["Accuracy", "Validation Accuracy"])



plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r')

plt.plot(epochs, val_loss, 'b')

plt.title('Training and validation loss')

plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend(["Loss", "Validation Loss"])



plt.figure()

y_pred=model.predict(X_test)

y_pred=np.argmax(y_pred,axis=1)

y_test=np.argmax(y_test,axis=1)

y_pred.shape, y_test.shape
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
report=classification_report(y_test,y_pred)

cm=confusion_matrix(y_test,y_pred)

acc=accuracy_score(y_test,y_pred)
print(report)
print(cm)
print(acc)