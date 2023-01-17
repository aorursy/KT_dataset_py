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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/blackfriday/BlackFriday.csv')
df.head()
df.describe()
df.corr()
sns.countplot(x = 'Gender',data = df)
sns.countplot(x = "Marital_Status",data = df)
plt.figure(figsize=(8,8))

sns.countplot(x = "Age", data = df, color= 'blue')
sns.set_style('dark')

plt.figure(figsize= (20,10))

sns.countplot(x='Age',hue = "Gender",data = df)
sns.set_style('dark')

plt.figure(figsize = (20,10))

sns.countplot('Occupation',data = df)
sns.countplot(x='City_Category',data = df)
plt.figure(figsize=(10,10))

sns.countplot(x='Stay_In_Current_City_Years',color = 'purple',data = df)
plt.figure(figsize = (20,10))

sns.barplot(x='Age', y='Purchase',hue = 'Gender',data= df)
plt.figure(figsize=(20,10))

sns.countplot(x = 'Product_Category_1',color = 'purple',data = df)
plt.figure(figsize=(20,10))

plt.xticks(rotation = 90)

sns.countplot(x = 'Product_Category_2',color = 'purple',data = df)
plt.figure(figsize = (20,10))

sns.heatmap(df.corr(),annot = True)

plt.xticks(rotation  = 60)
plt.figure(figsize=(20,10))

plt.xticks(rotation=90)

sns.countplot(x = 'Product_Category_3',color = 'blue',data = df)
#gender and purchase analysis

sns.barplot(x = 'Gender',y = 'Purchase',data = df)
#martial_status and purchase analysis

plt.figure(figsize = (10,8))

sns.barplot(x='Marital_Status',y='Purchase',hue = 'Gender' ,  data =df)
plt.figure(figsize=(20,10))

sns.barplot(x='Occupation',y='Purchase',color='blue',data = df)
plt.figure(figsize = (15,10))

sns.barplot(x='City_Category',y='Purchase',hue='Gender',data = df)
plt.figure(figsize=(10,8))

sns.barplot(x='Stay_In_Current_City_Years',y='Purchase',color = 'blue',data = df)
plt.figure(figsize=(10,8))

sns.barplot(x='Age',y='Purchase',data = df)
plt.figure(figsize=(20,10))

sns.barplot(x='Product_Category_1',y = 'Purchase', color = 'purple',

            data = df)
plt.figure(figsize=(20,10))

sns.set_style('dark')

sns.barplot(x='Product_Category_2',y = 'Purchase', color = 'purple',

            data = df)
plt.figure(figsize=(20,10))

sns.set_style('dark')

sns.barplot(x='Product_Category_3',y = 'Purchase', color = 'purple',

            data = df)
plt.figure(figsize = (20,8))

sns.barplot(x='City_Category', y='Purchase',hue = 'Age',data = df)
df.isnull().sum()

#here we can see the null values in product_category_2,product_category_3
#replacing the null vlaues using simpleImputer from scikit learn replace by mean.

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")

df.iloc[:,9:11] = imputer.fit_transform(df.iloc[:,9:11])
#check the null vlaues again

df.isnull().sum()
# Drop unwanted columns

df.drop(['User_ID','Product_ID'],axis = 1, inplace = True)
df.head()
#removing symbol('+') from the age and sttay in current city years

df['Age']=(df['Age'].str.strip('+'))

df['Stay_In_Current_City_Years']=(df['Stay_In_Current_City_Years'].str.strip('+').astype('float'))

#splitting the data into dependent and independent variables

X = df.iloc[:,0:9]

y = df.iloc[:, 9:]

#encoding the independnt variables

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder



ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[0,1,3])],remainder='passthrough')

X  = ct.fit_transform(X)

print(X)
#splitting training and test set

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# using linar regression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

LR = LinearRegression()

LR.fit(X_train,y_train)

y_pred = LR.predict(X_test)



print('R2 score for linear model: ',r2_score(y_test, y_pred))

print('mean absolute error: ',mean_absolute_error(y_test,y_pred))

# using ridge regressor

from sklearn.linear_model import Ridge

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

RR = Ridge(alpha=0.05)

RR.fit(X_train,y_train)

y_pred = RR.predict(X_test)



print('R2 score for Ridge model: ',r2_score(y_test, y_pred))

print('mean absolute error: ',mean_absolute_error(y_test,y_pred))



# using decession tree regressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

DT = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)

DT.fit(X_train,y_train)

y_pred = DT.predict(X_test)



print('R2 score for decission tree reg: ',r2_score(y_test, y_pred))

print('mean absolute error: ',mean_absolute_error(y_test,y_pred))

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

rf = RandomForestRegressor(n_estimators = 100, random_state =0)

rf.fit(X_train,np.ravel(y_train))



y_pred= rf.predict(X_test)

print('R2 score for random forest Reg model: ',r2_score(y_test,y_pred))

print('mean absolute error: ',mean_absolute_error(y_test,y_pred))
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=DT,X = X_train,y = y_train,cv =10)

print(accuracies.mean())