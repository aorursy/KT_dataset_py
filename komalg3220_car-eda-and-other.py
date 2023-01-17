

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn import metrics

from sklearn import preprocessing  # to normalisation



print(os.listdir("../input"))



df = pd.read_csv('../input/clean_df.csv')



df.head(6)



df.drop('Unnamed: 0',axis=1,inplace=True) # To drop column

#null value

df.isnull().any()
#to check how manu null values are there



df.isnull().sum()
#To check unique values 



df['stroke'].value_counts()
df['horsepower-binned'].value_counts()
#To fill null value

df['stroke']=df['stroke'].fillna(df['stroke'].mode()[0])

df['horsepower-binned']=df['horsepower-binned'].fillna(df['horsepower-binned'].mode()[0])
#To check the data type of coloumn

df.dtypes
#To check wheater column is contnious or categorical



for column in df.columns:

    print(column,len(df[column].unique()))

#To check correlation (by default it gives person correlation)

df.corr() #df.corr(method='spearman')


import matplotlib.pyplot as plt

#Scatter plot for continous Value

for column in ['normalized-losses','make','curb-weight','wheel-base','length','width','curb-weight','height','engine-size','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','city-L/100km']:

    

    x=df[column]    

    y=df['price']

    plt.scatter(x, y)

    plt.xlabel('x')

    plt.ylabel('y')

    plt.show()

#For box plot (for categorical values)



import seaborn as sns



for column in ['horsepower-binned','diesel','gas','engine-type','num-of-cylinders','fuel-system','aspiration','num-of-doors','body-style','drive-wheels','engine-location','symboling']:

    

    sns.boxplot(x=column,y='price',data=df)

    plt.show()


#To define x and y

x = df.loc[:, df.columns != 'price'] #to  select multiple column except one data point may be that we want to predict

#x=df.loc[:, ~df.columns.isin(['price','symboling','stroke','compression-ratio','peak-rpm','gas'])] # to  select multiple column except all the data point that we dont need.

y=df['price'].values #.values = to get the numpy array and dataset dont return index value and column with selected column



#convert categorical values (either text or integer) 

#df = pd.get_dummies(df, columns=['type'])

x=pd.get_dummies(x,columns=['make','aspiration','num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system','horsepower-binned'])

print(x.columns)



#To Normalise the equation

x=preprocessing.normalize(x)

#print(x.head())

print(y)
#method 1 to find accuracy

#train and test dataset creation

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

regression = linear_model.LinearRegression()

regression.fit(x_train,y_train)

predicted_Values = regression.predict(x_test)

print(predicted_Values)

print(y_test)





#checking accuracy of matrix 

print('score',regression.score(x_test,y_test)) # R square(to check accuracy of test data)

print('score',regression.score(x_train,y_train)) # to check accuracy of train data so that we can compare both accuracy to identify if it is underfit, overfit etc)

mean_squared_error = metrics.mean_squared_error(y_test, predicted_Values) 

print('Root Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))  # 2 is to round off the value

print('R-squared (training) ', round(regression.score(x_train, y_train), 3))  

print('R-squared (testing) ', round(regression.score(x_test, y_test), 3)) 

print('Intercept: ', regression.intercept_)

print('Coefficient:', regression.coef_) #higher the value of coeff higher will be relation of that data point with y
#method 2 to find accuracy



from sklearn.model_selection import KFold # import KFold

from statistics import mean 



scores = []

cv = KFold(n_splits=2, random_state=20, shuffle=False)



for train_index, test_index in cv.split(x):

    x_train, x_test, y_train, y_test = x[train_index], x[test_index], y[train_index], y[test_index]

    regression = linear_model.LinearRegression()

    regression.fit(x_train,y_train)

    scores.append(regression.score(x_test, y_test))



mean(scores)