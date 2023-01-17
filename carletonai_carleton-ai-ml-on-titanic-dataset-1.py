# data structure manipulation libraries

import numpy as np

import pandas as pd



# data visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns 



# data splitting

from sklearn.model_selection import train_test_split

from sklearn import metrics



# machine learning algorithms

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



import os

print(os.listdir('../input'))
train = pd.read_csv('../input/train.csv')

train.head()
train.info()
train.describe()
# The number of empty values for each column

cols = list(train)

for i in cols:

    print(i, ' '*(20-len(i)), train[i].isnull().sum())
#Dropping irrelevant columns

print('Before: ',train.shape)



cols_to_drop = ['Ticket','Name','Cabin','PassengerId']

train = train.drop(cols_to_drop, axis=1)

#Cleaning data

print('After ',train.shape)
#fill empty embarkation values

train['Embarked']=train['Embarked'].fillna('C')



#transforming text data into numerical values

train['Sex']=train['Sex'].apply(lambda x:1 if x=='female' else 0)

train['Embarked']=train['Embarked'].map({'S':0, 'Q':1,'C':2}).astype(int)

train.head()
#fill empty ages with valid and sensical values

def fillAges(df):

    count = df['Age'].isnull().sum()

    avg = df['Age'].mean()

    std = df['Age'].std()

    random_age = np.random.randint(avg-std,avg+std,count)

    df['Age'][np.isnan(df['Age'])] = random_age

    return df



train = fillAges(train)

#seaborn correlation

plt.figure(figsize=(10,12))

plt.title('Heatmap (Correlation between columns)')

colormap = plt.cm.RdBu

sns.heatmap(train.astype('float').corr(), annot=True, cmap = colormap)

#splitting of data into train and test

y=train['Survived']



x=train.drop('Survived',axis=1)



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.175)
#svc (Support Vector Machine/Constraint)

svc = SVC()

svc.fit(x_train,y_train)

svc_score = round(svc.score(x_test,y_test)*100,2)

print(svc_score)



predictions=svc.predict(x_test)

cm = metrics.confusion_matrix(y_test,predictions)



plt.figure(figsize=(9,9))

sns.heatmap(cm, annot=True, fmt=".3f",linewidths=.5,square=True,cmap='Blues_r')

plt.ylabel=('Actual label')

plt.xlabel=('Predicted label')

all_sample_title = 'Accuracy Score: {0}'.format(svc_score)

plt.title(all_sample_title,size=15)
#lr (Logistic Regression)

lr = LogisticRegression()



lr.fit(x_train,y_train)



lr_score = round(lr.score(x_test,y_test)*100,2)

print(lr_score)



predictions=lr.predict(x_test)

cm = metrics.confusion_matrix(y_test,predictions)



plt.figure(figsize=(9,9))

sns.heatmap(cm, annot=True, fmt=".3f",linewidths=.5,square=True,cmap='Blues_r')

plt.ylabel=('Actual label')

plt.xlabel=('Predicted label')

all_sample_title = 'Accuracy Score: {0}'.format(lr_score)

plt.title(all_sample_title,size=15)