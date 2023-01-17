# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score,classification_report

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





#Set input path

path = "../input/"

import os

print(os.listdir(path))



# Any results you write to the current directory are saved as output.
#Read input CSV file

df = pd.read_csv(path + "UnivBank.csv")

df.head()
print("------------------------------")

print(df.info())

print("------------------------------")

print(" ")

df.describe().T

#No null values in the dataframe
#ID column does not have any relevance to data. Drop ID column

df.drop(columns=["ID"],inplace=True)
df["Personal Loan"].value_counts()

# Personal Loan is the value to be predicted is highly unbalanced.
#Feature experience has Negative Value. Experience cannot have negative value

#Changing all Negative experience to positive

df.loc[df["Experience"] < 0,'Experience'] = df.Experience.abs()
#creating a balanced dataframe for model.

#extract equal number of Y classified data 0 & 1

df_balanced = df[df["Personal Loan"] == 1]

df_pl0 = df[df["Personal Loan"] == 0].head(480)

df_balanced = df_balanced.append(df_pl0)

df_balanced.info()
plt.figure(figsize=(16,6))

corr = df.corr()

sns.heatmap(corr

           ,xticklabels=corr.columns

           ,yticklabels=corr.columns

           ,annot=True

           ,cmap="YlGnBu")

plt.title(" Balanced dataframe Correlation Heatmap")
plt.figure(figsize=(16,6))

df_balanced_corr = df_balanced.corr()

sns.heatmap(df_balanced_corr

           ,xticklabels=df_balanced_corr.columns

           ,yticklabels=df_balanced_corr.columns

           ,annot=True

           ,cmap="YlGnBu")

plt.title(" Balanced dataframe Correlation Heatmap")
df_balanced_corr = df_balanced.corr()['Personal Loan'] 

Model_features = df_balanced_corr[(abs(df_balanced_corr) > 0.1) & (abs(df_balanced_corr) != 1.0)].sort_values(ascending=False)

print("{} absolute correlated values greater than 0.1:\n{}".format(len(Model_features), Model_features))
y = df_balanced['Personal Loan']

x = df_balanced.drop(columns=['Personal Loan'])

train_x, test_x,train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=1)

print(train_x.shape, test_x.shape,train_y.shape, test_y.shape)
svmlinear = svm.SVC(kernel='linear',C=1.0,random_state=2)

svmlinear.fit(train_x,train_y)
print ("\nSVM Linear Classifier - Train Confusion Matrix\n\n",pd.crosstab(train_y,svmlinear.predict(train_x),rownames = ["Actual"],colnames = ["Predicted"]) ) 
print ("\nSVM Linear Classifier - Train accuracy:",round(accuracy_score(train_y,svmlinear.predict(train_x)),3))

print ("\nSVM Linear Classifier - Train Classification Report\n",classification_report(train_y,svmlinear.predict(train_x)))



print ("\n\nSVM Linear Classifier - Test Confusion Matrix\n\n",pd.crosstab(test_y,svmlinear.predict(test_x),rownames = ["Actuall"],colnames = ["Predicted"]))      

print ("\nSVM Linear Classifier - Test accuracy:",round(accuracy_score(test_y,svmlinear.predict(test_x)),3))

print ("\nSVM Linear Classifier - Test Classification Report\n",classification_report(test_y,svmlinear.predict(test_x)))