

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

%matplotlib inline
df = pd.read_csv("../input/gems-price/Gems.csv")
df
cut = pd.get_dummies(df['cut'],drop_first=True)

color = pd.get_dummies(df['color'],drop_first=True)

clarity = pd.get_dummies(df['clarity'],drop_first=True)
df = pd.concat([df,cut,color,clarity],axis=1)
df = df.drop(['cut','color','clarity'],axis=1)
df
plt.title('Correlation Matrix')

sns.heatmap(df.corr())
lg = LinearRegression()
X_train,X_test,y_train,y_test = train_test_split(df.drop(['price'],axis=1),df['price'])
lg.fit(X_train,y_train)
pred = lg.predict(X_test)
pred = pd.Series(pred)
pred.index = y_test.index
predictions = pd.concat([pred,y_test],axis=1,ignore_index=True,)
predictions.columns = ['Predictions','Original Value']
predictions