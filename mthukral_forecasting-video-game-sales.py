# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.style.use('ggplot')
dataset = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")
dataset.info()

#total 16719 rows, 15 col

dataset.describe(include="all")
dataset.sample(10)
dataset.query('EU_Sales> NA_Sales')
#the maximum selling video game in EU

dataset.iloc[dataset['EU_Sales'].idxmax()]
#the maximum selling video game in NA

dataset.iloc[dataset['NA_Sales'].idxmax()]
#the maximum selling video game globally

dataset.iloc[dataset['Global_Sales'].idxmax()]
#the maximum selling video game in JP

dataset.iloc[dataset['JP_Sales'].idxmax()]
dataset.groupby('Year_of_Release')['EU_Sales'].sum().idxmax()

#2009 is the highest selling year



dataset.groupby('Genre')['EU_Sales'].sum().idxmax()

#action is the most popular genre in EU

dataset.groupby('Publisher')['EU_Sales'].sum().idxmax()

#Ninetendo is the highest selling publisher in EU

dataset.groupby('Platform')['EU_Sales'].sum().idxmax()

#PS2 is the highest selling platform

dataset.query('Year_of_Release == 2009.0').groupby('Genre').sum().idxmax()

#We can perform similar kind of analysis for other regions

#lets deal with missing data. if we choose global_sales as response variable, so we should drop rows with Nan in Global_sales. Also I am dropping other sales as I will be analysing only gobal sales

missing_data = dataset.isnull()

missing_data.head(5)
for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")   

#lets drop the na from Critic Score as 50% data is NA

dataset.dropna(subset=['Critic_Score'],inplace=True)

dataset.info()

 

dataset.info()

missing_data = dataset.isnull()

missing_data.head(5)

for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")  
dataset.head(10)

        

       
#name,platform,genre,publisher,developer,rating are categorical varaibles. Lets handle them one by one



pd.get_dummies(dataset['Name'])




#there are 5085 unique values.It is better to drop this column

dataset.drop('Name',axis =1,inplace=True)



#lets repeat it for other columns

pd.get_dummies(dataset['Publisher'])

pd.get_dummies(dataset['Developer'])



# both developer and publisher has a large nuber of unique values. It is better to drop these columns

dataset.drop(['Developer','Publisher'],axis =1,inplace=True)
dg = pd.get_dummies(dataset['Genre'],drop_first=True)

dg
dp =pd.get_dummies(dataset['Platform'],drop_first=True)

dp

# we will encode genre, rating and platform . Rating col has missing value. we will replace Nan with most common value .

dataset['Rating'].value_counts().idxmax()
#Replace Nan with "E"

dataset['Rating'].replace(np.nan, dataset['Rating'].value_counts().idxmax(), inplace=True)
dr = pd.get_dummies(dataset['Rating'],drop_first=True)

dr
dataset.describe(include='all')


missing_data = dataset.isnull()

missing_data.head(5)

for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")  
dataset.info()
#lets first drop these columns and then concatenate

# concatenate these columns with existing dataset

dataset.drop(['Platform','Genre','Rating'],axis=1, inplace=True)
dataset.info()
dataset = pd.concat([dataset,dp,dg,dr],axis=1)
dataset.info()

#lets check the missing values again

missing_data = dataset.isnull()

missing_data.head(5)

for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("") 
# user_count, user_score and year has missing values

#first lets convert the type of user score to float

dataset['User_Score'].replace('tbd', np.nan,inplace=True)

userscoredf = dataset['User_Score'].copy()
userscoredf.reset_index(drop=True,inplace=True)
userscoredf.dropna(inplace=True)
userscoredf.reset_index(drop=True,inplace=True)


userscoredf= pd.to_numeric(userscoredf)

userscoredf.dtype
count, bin_edges = np.histogram(userscoredf, 10)



# un-stacked histogram

userscoredf.plot(kind ='hist', 

          figsize=(10, 6),

          bins=10,

          alpha=0.6,

          xticks=bin_edges,

          color=['coral', 'darkslateblue', 'mediumseagreen']

         )



plt.title('Histogram of User Score')



plt.show()
# user score is right skewed. 

userscoredf.mean()

userscoredf.median()
dataset['User_Score']
dataset['User_Score'].replace(np.nan, dataset['User_Score'].astype("float").mean(axis=0) , inplace=True)
dataset['User_Score']= pd.to_numeric(dataset['User_Score'])
dataset.info()
# replace the missing values in year with most frequent year

dataset['Year_of_Release'].replace(np.nan, dataset['Year_of_Release'].value_counts().idxmax(), inplace=True)
missing_data = dataset.isnull()

for column in missing_data.columns.values.tolist():

    print(column)

    print (missing_data[column].value_counts())

    print("")  
#lets visualize the user count data

usercountdf = dataset['User_Count'].copy()

usercountdf
usercountdf.dropna(inplace=True)
usercountdf.shape
usercountdf.reset_index(drop=True,inplace=True)

usercountdf.idxmax()
count, bin_edges = np.histogram(usercountdf, 40)



# un-stacked histogram

usercountdf.plot(kind ='hist', 

          figsize=(10, 6),

          bins=40,

          alpha=0.6,

          xticks=bin_edges,

          color=['coral', 'darkslateblue', 'mediumseagreen']

         )



plt.title('Histogram of User Count')



plt.show()
# maximum number of user count is between 4 to 27. rather than taking the mean, we will use the most frequent value of user_count



usercountdf.mean()
usercountdf.value_counts()
dataset.User_Count.shape
dataset['User_Count'].replace(np.nan, dataset['User_Count'].value_counts().idxmax(),inplace=True)
dataset.describe()
# We willbe forecasting Global_sales, so lets drop other columns such as NA_Sales, JP_Sales





dataset.drop(['NA_Sales','EU_Sales','JP_Sales','Other_Sales'],axis =1,inplace=True)
dataset.reset_index(drop=True,inplace=True)
dataset.info()
#So now this is or final dataset

dataset

#Lets populate X and y arrays

y = dataset[['Global_Sales']].values

y.shape
y
dataset_new = dataset.drop('Global_Sales',axis=1)
dataset_new.describe()

dataset_new.info()

X = dataset_new.iloc[:,:].values

X.shape

X
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

X_train.shape

y_train.shape

X_test.shape

y_test.shape
#Model 1 Simple Linear Regression



# Fitting Simple Linear Regression to the Training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)



y_test

y_pred

from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)
from sklearn.metrics import r2_score

r2_score(y_test, y_pred) 
regressor.score(X_test,y_test)
#Model 2 Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)
from sklearn.metrics import r2_score

r2_score(y_test, y_pred) 
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train_scaled= sc_X.fit_transform(X_train)

y_train_scaled = sc_y.fit_transform(y_train)

X_test_scaled= sc_X.fit_transform(X_test)

y_test_scaled = sc_y.fit_transform(y_test)
from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train_scaled, y_train_scaled.ravel())
y_pred = regressor.predict(sc_X.fit_transform(X_test))

y_pred = sc_y.inverse_transform(y_pred)
y_pred

from sklearn.metrics import r2_score

r2_score(y_test, y_pred) 
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)