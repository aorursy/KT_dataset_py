# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
dataset = pd.read_csv('/kaggle/input/black-friday/train.csv')
dataset.head()
dataset.info()
dataset.describe()
dataset.shape
#Getting the idea of unique users

print('Total Users :',dataset.shape[0])

print('Unique Users:',dataset.shape[0] - len(set(dataset.User_ID)))
#Checking the distribution of the target variable

import matplotlib.pyplot as plt

import seaborn as sns

sns.distplot(dataset.Purchase,bins = 30)

#seems like a near gaussian distribution
#Checking the columns with dypes which are numeric

numeric_col = dataset.select_dtypes(include=[np.number]).columns

numeric_col
Categorical_features = dataset.select_dtypes(include = ['object']).columns

Categorical_features
#Univariate analysis

fig = plt.figure(figsize = (15,25))

for i, col in enumerate(numeric_col[1:-1]):

    fig.add_subplot(5,1,i+1)

    sns.countplot(dataset[col])

#More unmarried users then married users in the sale
dataset.corr()['Purchase'].sort_values(ascending = True)
plt.figure(figsize = (15,10))

sns.heatmap(dataset.corr(),annot = True)

#Numerical features doesnot  seem to be the best predictor of the purchase
#Univariate analysis

fig = plt.figure(figsize = (15,25))

for i, col in enumerate(Categorical_features[1:]):

    fig.add_subplot(5,1,i+1)

    sns.countplot(dataset[col])

#More purchase by male in the sale

#More purchase by 26-35 in the sale

#More purchase by new members in the city in the sale

#purchase by City B members are more in the sale
dataset.pivot_table(index = 'Occupation',values = 'Purchase',aggfunc = np.mean).plot(kind = 'bar',figsize  = (15,10))

#Dependency of the purchase seems to be less on occupation
dataset.pivot_table(index = 'Marital_Status',values = 'Purchase',aggfunc = np.mean).plot(kind = 'bar',figsize  = (10,8))

#No significant effect on the purchase
dataset.pivot_table(index = 'Product_Category_1',values = 'Purchase',aggfunc = np.mean).plot(kind = 'bar',figsize  = (10,8))

#There was more product category 1 in 1,5 and 8 but doesn't seem to be have more purhase despite being low on sales
dataset.pivot_table(index = 'Gender',values = 'Purchase',aggfunc = np.mean).plot(kind = 'bar',figsize  = (10,8))

dataset.pivot_table(index = 'Age',values = 'Purchase',aggfunc = np.mean).plot(kind = 'bar',figsize  = (10,8))

dataset.pivot_table(index = 'City_Category',values = 'Purchase',aggfunc = np.mean).plot(kind = 'bar',figsize  = (10,8))

#There was more average purchase by the C,there was more from the users from the city B in the count
dataset.pivot_table(index = 'Stay_In_Current_City_Years',values = 'Purchase',aggfunc = np.mean).plot(kind = 'bar',figsize  = (10,8))

train = pd.read_csv('/kaggle/input/black-friday/train.csv')

test = pd.read_csv('/kaggle/input/black-friday/test.csv')
train['source'] = 'train'

test['source']= 'test'
data = pd.concat([train,test],ignore_index = True,sort = False)
data.head()
(data.isnull().sum()/data.shape[0])*100
# Product_Category_3 is has 69.64 percent of the data, no sense in filling the missing value

data['Product_Category_2'].value_counts().sort_values(ascending = False)
data['Product_Category_2'].value_counts().sort_values(ascending = False)
data['Product_Category_2'].isnull().sum()
data["Product_Category_2"]= data["Product_Category_2"].fillna(-2.0).astype("float")

data.Product_Category_2.value_counts().sort_index()
data["Product_Category_3"]= data["Product_Category_3"].fillna(-2.0).astype("float")

data.Product_Category_2.value_counts().sort_index()
data.drop(['User_ID','Product_ID'],axis = 1,inplace = True)
data = pd.get_dummies(data)
data
train = data.loc[data['source_train']==1]

test = data.loc[data['source_test']==1]
train.drop(['source_train','source_test'],axis = 1,inplace = True)

test.drop(['source_train','source_test'],axis = 1,inplace = True)

X = train.drop('Purchase',axis = 1)

y = train.Purchase
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)
from sklearn import metrics
from sklearn.linear_model import LinearRegression

Regressor = LinearRegression()
Regressor.fit(X_train,y_train)
y_pred  = Regressor.predict(X_test)
from sklearn.metrics import mean_squared_error

rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))

rmse
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
from sklearn.linear_model import Ridge

RR = Ridge(alpha=0.05,normalize=True)

RR.fit(X_train,y_train)
y_pred  = RR.predict(X_test)
from sklearn.tree import DecisionTreeRegressor

DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)

DT.fit(X_train,y_train)
y_pred  = DT.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))

rmse
RF = DecisionTreeRegressor(max_depth=8, min_samples_leaf=150)
RF.fit(X_train,y_train)
y_pred  = RF.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))

rmse
r2_score(y_test,y_pred)

from sklearn.model_selection import cross_val_score

# function to get cross validation scores

def get_cv_scores(model):

    scores = cross_val_score(model,

                             X_train,

                             y_train,

                             cv=5,

                             scoring='r2')

    

    print('CV Mean: ', np.mean(scores))

    print('STD: ', np.std(scores))

    print('\n')
get_cv_scores(DT)