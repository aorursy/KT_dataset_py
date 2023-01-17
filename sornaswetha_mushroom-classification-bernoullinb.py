# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Read the dataset

df=pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

print (df.shape)
# Target Variable

print(df["class"].value_counts())

sns.countplot(df["class"] ,palette="Reds" )

plt.show()
#Dependent and Target Features

X=df.drop('class',axis=1) #Predictors

y=df['class'] #Response



#get_dummies

X=pd.get_dummies(X,columns=X.columns,drop_first=True)

X.head()
#Now all our features binary-valued. Bernoulli Naive Bayes can be applied now.
#Split the train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
#BernoulliNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
nb = BernoulliNB()

nb.fit(X_train,y_train)

print( "Training Accuracy", accuracy_score(y_train , nb.predict(X_train)))

print("Testing Accuracy", accuracy_score(y_test , nb.predict(X_test)))

print("Confusion Matrix:\n" , confusion_matrix(y_test , nb.predict(X_test ) ))

print("Classification Report:\n", classification_report(y_test , nb.predict(X_test)))
#Hyper Parameter Tunning - alpha value

#By tunning the parameters we can improve our accuracy

nb = BernoulliNB( alpha=0.02, binarize=0.0, class_prior=None, fit_prior=True)

nb.fit(X_train,y_train)

print( "Training Accuracy", accuracy_score(y_train , nb.predict(X_train)))

print("Testing Accuracy", accuracy_score(y_test , nb.predict(X_test)))

print("Confusion Matrix:\n" , confusion_matrix(y_test , nb.predict(X_test)))

print("Classification Report:\n", classification_report(y_test , nb.predict(X_test)))