import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn import metrics

import os



print(os.listdir("../input"))



df = pd.read_csv('../input/bikeshare.csv')



df=df.drop(['datetime'], axis=1)





df.head(6)





df.isnull().any()

for column in ['season','holiday','workingday']:

    print(df[column].value_counts())

    df[column].unique()

   

    

    

    
x=df[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered']]

y=df['count'].values # .values = to get the numpy array and dataset dont return index value and column with selected colouwmn

print(x.head())

print(y) #. head is only used for dataframe and y is not a dataframe.

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
