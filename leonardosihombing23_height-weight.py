#input data

import pandas as pd

df = pd.read_csv('../input/weight-height/weight-height.csv')

df.head()
#analyze the data

df.info()

df.describe()

df.isnull().sum()
#convert gender to number

df['Gender'].replace('Female',0, inplace=True)

df['Gender'].replace('Male',1, inplace=True)

X=df.iloc[:, :-1].values

y=df.iloc[:, 2].values
#split data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
#choosing regression model

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
#predict test value

y_pred = lin_reg.predict(X_test)
#predict the weight

the_weight_prediction = lin_reg.predict([[1,74]])

print('The Weight Prediction =', the_weight_prediction)
# R square

import math



print('R square = ',lin_reg.score(X_train,y_train))

print('Correlation = ',math.sqrt(lin_reg.score(X_train,y_train)))