



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns



print(os.listdir("../input"))



df = pd.read_csv('../input/student-mat.csv')



df.head(6)



df.isnull().any()

for column in ['school','failures','paid','higher', 'address']:

    print(df[column].value_counts())

    df[column].unique()

   
df.corr()



f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(),annot=True,linewidth=0.5,fmt='.3f',ax=ax)

plt.show()

df.columns
x=df[['age','Medu','Fedu','studytime','failures','goout','Dalc','Walc','G1','G2','famrel']]

y=df['G3'].values

print(x.head())

print(y)
#train and test dataset creation

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

regression = linear_model.LinearRegression()

regression.fit(x_train,y_train)

predicted_Values = regression.predict(x_test)

print(predicted_Values)

print(y_test)



#checking accuracy of matrix 

print('score',regression.score(x_test,y_test)) 

mean_squared_error = metrics.mean_squared_error(y_test, predicted_Values) 

print('Root Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))  # 2 is to round off the value

print('R-squared (training) ', round(regression.score(x_train, y_train), 3))  

print('R-squared (testing) ', round(regression.score(x_test, y_test), 3)) 

print('Intercept: ', regression.intercept_)

print('Coefficient:', regression.coef_)