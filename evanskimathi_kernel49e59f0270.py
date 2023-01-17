import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

#inline matplotlib
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#importing the data
dataset = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')

#selecting the most gender as most valuable feature
df_getdummy = pd.get_dummies(data=dataset, columns=['Gender'])
dataset.head(5)
#Declaring dummy data
x = df_getdummy.drop('Purchased',axis=1)
y = df_getdummy['Purchased']
#Splitting my data between training and test data
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test=train_test_split(x,y, test_size = 0.20, random_state = 0)
#Feature scaling
from sklearn.preprocessing import StandardScaler

sc =StandardScaler()

x_train = sc.fit_transform(x_train)

x_test =sc.fit_transform(x_test)
#Fitting logistic regression model onto our dataset
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(x_train, y_train)
#predicting our test set results
y_pred = classifier.predict(x_test)
#getting result of our classification model in matrix form
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
#Retrieving our regression result
accuracy_score(y_true=y_train, y_pred=classifier.predict(x_train))