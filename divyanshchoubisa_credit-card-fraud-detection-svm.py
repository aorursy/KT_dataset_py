#Lets Import Important Libraries

import pandas as pd

import numpy as np

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

from imblearn.over_sampling import RandomOverSampler

#from imblearn.combine import SMOTETome - Class to perform over-sampling using SMOTE (Synthetic Minority Oversampling Technique).

warnings.simplefilter("ignore")
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

df.drop(['Time'],axis=1,inplace=True)

df.head()
from sklearn.preprocessing import StandardScaler

df['norm_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))

df.head(1)
df.drop(['Amount'],axis=1,inplace=True)
print("The missing values in different columns are: ")

print(df.isnull().sum())
show = df['Class'].value_counts()

show.plot(kind='bar',figsize=(6,6),color='y')
show
print('The percentage of no frauds is in the provided data is : ',show[0]/show.sum() * 100,'%')

print('The percentage of frauds in the provided data is: ',show[1]/show.sum() * 100,'%')
frauds = df[df['Class'] == 1]

non_frauds = df[df['Class'] == 0]

print(frauds.shape)

print(non_frauds.shape)
X = df.drop(['Class'],axis=1)

X[:5]
Y = df[['Class']]

Y[:5]
ros =  RandomOverSampler(sampling_strategy=0.5) #To perform Oversampling

ros
Xs, ys = ros.fit_sample(X, Y)

print(Xs.shape)

print(ys.shape)
ys['Class'].value_counts()
Xs = Xs.values

ys = ys.values

print(Xs[:1])

print(ys[:3])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size = 0.25, random_state = 16)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train_X = scaler.fit_transform(X_train)

test_X = scaler.transform(X_test)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')

classifier
classifier.fit(train_X, y_train)
y_pred = classifier.predict(test_X)

y_pred
from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
print("Accuracy of Model on test data:  ", accuracy_score(y_test, y_pred) *  100)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm,annot=True,fmt="d")