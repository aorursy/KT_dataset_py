import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn import metrics

from sklearn import preprocessing  # to normalisation



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/traffic_volume_1hr.csv')

df.head(9)
#null value

df.isnull().any()
#To check unique values 



df['hr'].value_counts()
df['day'].value_counts()
#To check the data type of coloumn

df.dtypes
#To check wheater column is contnious or categorical



for column in df.columns:

    print(column,len(df[column].unique()))
#To check correlation (by default it gives person correlation)

df.corr() #df.corr(method='spearman')
import matplotlib.pyplot as plt

#Scatter plot for continous Value

for column in ['month','day','hr']:

    

    x=df[column]    

    y=df['total_volume']

    plt.scatter(x, y)

    plt.xlabel('x')

    plt.ylabel('y')

    plt.show()
#For box plot (for categorical values)



import seaborn as sns



for column in ['month','day','hr']:

    

    sns.boxplot(x=column,y='total_volume',data=df)

    plt.show()
#To define x and y

x = df.loc[:, df.columns != 'total_volume']

y=df['total_volume'].values



#convert categorical values (either text or integer) 

x=pd.get_dummies(x,columns=['month','day','hr'])

print(x.columns)



#To Normalise the equation

x=preprocessing.normalize(x)

print(y)
#train and test dataset creation

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

regression = linear_model.LinearRegression()

regression.fit(x_train,y_train)

predicted_Values = regression.predict(x_test)





#checking accuracy of matrix 

print('score',regression.score(x_test,y_test)) # R square(to check accuracy of test data)

print('score',regression.score(x_train,y_train)) # to check accuracy of train data so that we can compare both accuracy to identify if it is underfit, overfit etc)

mean_squared_error = metrics.mean_squared_error(y_test, predicted_Values) 

print('Root Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))  # 2 is to round off the value

print('R-squared (training) ', round(regression.score(x_train, y_train), 3))  

print('R-squared (testing) ', round(regression.score(x_test, y_test), 3)) 

print('Intercept: ', regression.intercept_)

print('Coefficient:', regression.coef_) #higher the value of coeff higher will be relation of that data point with y