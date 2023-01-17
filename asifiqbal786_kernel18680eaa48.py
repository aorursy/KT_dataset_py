#!/usr/bin/env python

# coding: utf-8



# In[1]:





# Load libraries



import numpy as np

import pandas as pd

import itertools

from pandas import read_csv

from pandas.plotting import scatter_matrix

from matplotlib import pyplot



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier

#from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis





# In[2]:





#Read the data

df=pd.read_csv('E:\\news.csv')

#Get shape and head

print(df.shape)

df.head(20)





# In[3]:





# Get the labels

labels=df.label

labels.head()

#Group by FAKE vs REAL

print(df.groupby('label').size())





# In[4]:





# Split the dataset

x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)





# In[5]:





x_train.head()





# In[6]:





y_train.head()





# In[7]:







# Initialize a TfidfVectorizer

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)





# In[8]:





tfidf_train=tfidf_vectorizer.fit_transform(x_train) 

tfidf_test=tfidf_vectorizer.transform(x_test)





# In[9]:





#classifier

pac=PassiveAggressiveClassifier(max_iter=50)

knn=KNeighborsClassifier()

cart=DecisionTreeClassifier()

svm=SVC(gamma='auto')

clf = MultinomialNB()

rf = RandomForestClassifier(n_estimators=100)





# ## CONFUSION MATRIX PRINT



# In[10]:





import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

   

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')





# ## PAC CLASSIFIER



# In[11]:





#PAC: Predict on the test set and calculate accuracy

pac.fit(tfidf_train,y_train)



y_pred=pac.predict(tfidf_test)

score=accuracy_score(y_test,y_pred)



print(f'PAC Accuracy: {round(score*100,2)}%')





# In[12]:





cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])





# ## DECISSION TREE CLASSIFIER



# In[13]:





#Decision Tree - Predict on the test set and calculate accuracy

cart.fit(tfidf_train,y_train)

y_pred=cart.predict(tfidf_test)

score=accuracy_score(y_test,y_pred)

confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

print(f'Decission Tree Accuracy: {round(score*100,2)}%')





# In[14]:





print('Decission Tree Confusion Matrix')

cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])





# ## KNN CLASSIFIER



# In[15]:





#KNN - Predict on the test set and calculate accuracy

knn.fit(tfidf_train,y_train)

y_pred=knn.predict(tfidf_test)

score=accuracy_score(y_test,y_pred)

print(f'KNN Accuracy: {round(score*100,2)}%')





# In[16]:





print('KNN Confusion Matrix')

cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])





# ## SVC CLASSIFIER



# In[17]:





#SVC Predict on the test set and calculate accuracy

svm.fit(tfidf_train,y_train)

y_pred=svm.predict(tfidf_test)

score=accuracy_score(y_test,y_pred)

print(f'SVC Accuracy: {round(score*100,2)}%')





# In[18]:





print('SVC Confusion Matrix')

cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])





# ## NB CLASSIFIER



# In[19]:





#NaiveBase - Predict on the test set and calculate accuracy

clf.fit(tfidf_train,y_train)

y_pred=clf.predict(tfidf_test)

score=accuracy_score(y_test,y_pred)

print(f'NB Accuracy: {round(score*100,2)}%')





# In[20]:





print('NB Confusion Matrix')

cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])





# ## RANDOM FOREST CLASSIFIER



# In[21]:





#Random Forest - Predict on the test set and calculate accuracy

rf.fit(tfidf_train,y_train)

y_pred=rf.predict(tfidf_test)

score=accuracy_score(y_test,y_pred)

print(f'Random Forest Accuracy: {round(score*100,2)}%')





# In[22]:





print('Random Forest Confusion Matrix')

cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])





# In[ ]: