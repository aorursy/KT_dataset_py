import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Let's read the data & explore

df=pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')

df.head()
df.isnull().sum()
X=df['Message']

y=df['Category']
#Encoding our lables ham and spam as 0 and 1

y_encode=LabelEncoder().fit_transform(y)
#Making our texts into vectores

X_vector=TfidfVectorizer().fit_transform(X)
#Let's split our dataset for train-test purpose

X_train, X_test, y_train, y_test=train_test_split(X_vector, y_encode, test_size=0.2, random_state=42)
#Building the model

model=RandomForestClassifier()

model.fit(X_train, y_train)

print('In Sample Score: ', model.score(X_train, y_train))
#Cross validation scoring

val_score=cross_val_score(model, X_vector, y_encode, cv=5, scoring='accuracy')

print('validation score: ',val_score)
#Make prediction

y_pred=model.predict(X_test)
print('Out Sample Score: ', accuracy_score(y_test, y_pred))
#Visualizing the result

cm=confusion_matrix(y_test, y_pred)

cm=pd.DataFrame(cm, index=[i for i in range(2)], columns=[i for i in range(2)])

plt.figure(figsize=(5,5))

sns.heatmap(cm, cmap='Blues',linecolor='black',linewidths=1, annot=True, fmt='')