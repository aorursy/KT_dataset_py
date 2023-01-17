# importing the required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# importing the 'insurance' dataset

df = pd.read_csv("../input/insurance-premium-prediction/insurance.csv")

# printing the shape - rows & columns

df.shape
# printing first 5 rows of dataset

df.head()
# printing data types of all columns

df.dtypes
# printing all unique values in categorical variables

a = df['sex'].unique()

b = df['children'].unique()

c = df['smoker'].unique()

d = df['region'].unique()

print(a,'\n',b,'\n',c,'\n',d)
# checking for duplicate rows present

df.duplicated().sum()
# removing duplicate rows

df = df.drop_duplicates()
# checking for null values present

df.isnull().sum()
# printing data types of all columns

df.dtypes
df.describe(include='all')
# plotting frequecy distribution for different variables

hist = df.hist(figsize = (15,15),color='#EC7063')
# frequecny distribution of each value present in 'region' variable

df.region.value_counts().plot(kind="bar",color='#58D68D')
# pie-chart to plot frequency of smokers and non-smokers

df.smoker.value_counts().plot(kind="pie")
# mean expenses for smokers and non-smokers

df.groupby('smoker').expenses.agg(["mean"])
# mean expenses both male & female

df.groupby('sex').expenses.agg(["mean"])
# find corelation between numerical variables

df.corr()
# bar graphs to show trends in 'expenses' variable w.r.t other variables present

a = ['age','children','bmi','sex','smoker','region']

for i in a:

    x = df[i]

    y = df['expenses']

    plt.bar(x,y,color='#A569BD')

    plt.xlabel(i)

    plt.ylabel('expenses')

    plt.show()
# splitting dependent variable from independent variables

x = df.drop(columns=['expenses'])

y = df['expenses']

x.head()
# One Hot Encoding all the categorical variables 

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[( 'encoder', OneHotEncoder() , [1,4,5] )], remainder='passthrough')

x = np.array(ct.fit_transform(x))
# splitting the dataset into train & test part (with 25% as test)

from sklearn.model_selection import train_test_split

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.25 , random_state=1)
# Feature Scaling the variables

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(x_train)

x_train = sc.transform(x_train)

x_test = sc.transform(x_test)
# training the model with multiple linear regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)
print("Model intercept",regressor.intercept_,"Model co-efficent",regressor.coef_)
# making predictions on test dataset

y_pred = regressor.predict(x_test)
# printing 'Root Mean Squared Value' for train and test part of the dataset separately

from sklearn import metrics

print("RMSE")

print(np.sqrt(metrics.mean_squared_error(y_test,regressor.predict(x_test))))