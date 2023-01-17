 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling 
# Load the dataset

df = pd.read_csv('../input/wines.csv', delimiter =";")

df.head()

#Important: 1 = red, 0 = white
pandas_profiling.ProfileReport(df) 
df.shape
df.isna().sum()
# Display a description of the dataset



df.describe()
#Correlation again

corr=df.corr()

corr
 %matplotlib inline
corr=df.corr()

plt.figure(figsize=(14,6))

sns.heatmap(corr,annot=True)
df.head()
df['type'].value_counts()
#Only high corr variables to "type" were chosen to build the model

X = df[["fixed acidity", "volatile acidity" , "chlorides", "sulphates", 'free sulfur dioxide', 'total sulfur dioxide']]

y = df['type']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

X_test  = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)
print('Accuracy = ',round(accuracy_score(y_test,y_pred),4) *100, '%')
# Confusion Matrix 

from sklearn.metrics import confusion_matrix



cm_lr = confusion_matrix(y_test, y_pred)

print(cm_lr)