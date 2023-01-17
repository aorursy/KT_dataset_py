# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



#Import the Libraries for visualision

import matplotlib.pyplot as plt

import seaborn as sns



#Importing the Dataset

df = pd.read_csv("../input/heart.csv")



#Lets a quick look of our dataset

df.info()
#Lets check the statistical inference of the dataset

df.describe()
#Now lets chech our target varibale first

sns.countplot(x = 'target', data = df)

plt.show()
#Lets see the pair plot

p=sns.pairplot(df, hue = 'target')
#Now lets see the correlation by plotting heatmap

corr = df.corr()

colormap = sns.diverging_palette(220, 10, as_cmap = True)

plt.figure(figsize = (8,6))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30,

            cmap = colormap, linecolor='white')

plt.title('Correlation of df Features', y = 1.05, size=10)

#Lets look the correlation score

print (corr['target'].sort_values(ascending=False), '\n')
#Now lets build a ml model 

#At first we take our matix of features and target variable

x = df.iloc[:, 0 :13].values

y = df.iloc[:,13].values
#Spliting the dataset into traning and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.transform(x_test)



# Fitting the Logistic Regression on Traning set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0, solver = "lbfgs")

classifier.fit(x_train,y_train)
#Predicting Test set Result

y_pred = classifier.predict(x_test)
#Making the Confussion Matrix and Print Accuracy

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression :")

print("Accuracy = ", accuracy)

print(cm)

#Let see the ROC curve

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test, y_pred)

print('AUC: %.3f' % auc)



fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.plot(fpr, tpr, marker='.')

plt.show()