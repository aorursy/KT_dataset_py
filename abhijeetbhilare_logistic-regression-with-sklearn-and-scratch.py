import numpy as np 

import pandas as pd 

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("/kaggle/input/stat-mods/insurance.csv")

df = df.dropna(how='all', axis='columns')

print(df.shape)

df.head()
import seaborn as sns

sns.regplot(x='age', y='bought_insurance', data=df, logistic=True)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,test_size=0.1)

model = LogisticRegression()

model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

model.predict_proba(X_test)

model.coef_
model.intercept_
import math

def sigmoid(x):

  return (1 / (1 + math.exp(-x)))
def prediction_function(age):

    z = model.coef_[0][0] * age - (model.intercept_[0])

    y = sigmoid(z)

    return y
age = 47

print(prediction_function(age))