import numpy as np

import pandas as pd 

import os

import seaborn as sns

from matplotlib import pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')
df.head()
df.isnull().sum()
df.drop('Albumin_and_Globulin_Ratio',axis=1,inplace=True)
df.isnull().sum()
df.shape
df.describe()
outputDistribution=df['Dataset'].value_counts()

sns.barplot(outputDistribution.index,outputDistribution.values)

plt.ylabel('Count')

plt.xlabel('Output Classes')

plt.title('Class Counts')

plt.show()



print ("Ratio of Class 1 to Class 2:",outputDistribution.values[0]/outputDistribution.values[1])
# Converting strings to binary feature



def getGen(gender):

    if (gender=='Male'):

        return 0

    else:

        return 1



df['Sex']=df['Gender'].apply(getGen)

df.drop('Gender',axis=1,inplace=True)
sns.catplot(x='Dataset',y='Age',data=df,kind='box')

plt.show()
sns.catplot(x='Dataset',y='Age',data=df,kind='violin')

plt.show()
sns.catplot(x='Dataset',y='Total_Protiens',data=df,kind='box')

plt.show()
toRemove=((df['Total_Protiens']<=3) | (df['Total_Protiens']>=9))

toRemove.sum()
df1=df[~toRemove]

df.shape,df1.shape
sns.catplot(x='Dataset',y='Total_Protiens',data=df1,kind='box')

plt.show()
df1.head()
sns.catplot(x='Dataset',y='Albumin',data=df1,kind='box')

plt.show()
df2=df1[~(df1['Albumin']>5)]
df1.shape,df2.shape
sns.catplot(x='Dataset',y='Total_Bilirubin',data=df2,kind='boxen')

plt.show()
df2[df2['Total_Bilirubin'] > 40]
df3=df2[df2['Total_Bilirubin']<40]
sns.catplot(x='Dataset',y='Direct_Bilirubin',data=df3,kind='boxen')

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.ensemble import RandomForestClassifier
dfFinal=df3.copy()
Y=dfFinal['Dataset']

dfFinal.drop('Dataset',axis=1,inplace=True)

X=dfFinal.values
print(X.shape,Y.shape)

xTrain,xTest,yTrain,yTest=train_test_split(X,Y)
# Logistic Regression



lr_clf=LogisticRegression(max_iter=1000)

lr_clf.fit(xTrain,yTrain)

yPredicted_lr=lr_clf.predict(xTest)





testScore=lr_clf.score(xTest,yTest)

trainScore=lr_clf.score(xTrain,yTrain)



print ("train score:",trainScore)

print ("test score:",testScore)



print()

print (classification_report(yTest,yPredicted_lr))

print (confusion_matrix(yTest,yPredicted_lr))

print()
# SVM with RBF Kernel



svm_clf=svm.SVC()

svm_clf.fit(xTrain,yTrain)



yPredicted_svm=svm_clf.predict(xTest)



trainScore=svm_clf.score(xTrain,yTrain)

testScore=svm_clf.score(xTest,yTest)



print ("Train Score:",trainScore)

print ("Test Score:",testScore)



print()

print('Clasification Report:')

print (classification_report(yTest,yPredicted_svm))

print('Confusion Matrix:')

print (confusion_matrix(yTest,yPredicted_svm))
# Random  Forest Classifier



clf_rf=RandomForestClassifier()

clf_rf.fit(xTrain,yTrain)



trainScore=clf_rf.score(xTrain,yTrain)

testScore=clf_rf.score(xTest,yTest)

yPredicted_rf=clf_rf.predict(xTest)



print ("Train Score:",trainScore)

print ("Test Score:",testScore)



print()

print (classification_report(yTest,yPredicted_rf))

print (confusion_matrix(yTest,yPredicted_rf))