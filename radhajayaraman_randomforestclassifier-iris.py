import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/iris/Iris.csv')

data.head()
X=data.iloc[:,1:5]

y=data.iloc[:,5]
#Splitting data into training and test set

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
classifier=RandomForestClassifier(criterion='entropy')

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print("Accuracy for test set is:",accuracy_score(y_test,y_pred)*100,"%")
cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)
print("Accuracy for training set:",accuracy_score(y_train,classifier.predict(X_train))*100,"%")
feature_imp=pd.Series(classifier.feature_importances_,index=X.columns)

feature_imp.plot(kind='bar')

plt.title('Feature importance for the model')