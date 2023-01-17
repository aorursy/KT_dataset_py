# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set()



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
bigmart_train = pd.read_csv('../input/bigmart-sales-data/Train.csv')

bigmart_test = pd.read_csv('../input/bigmart-sales-data/Test.csv')

bigmart_train.head()
#dataset shape

bigmart_train.shape
#dataset dtypes

bigmart_train.dtypes
#dataset properities

bigmart_train.info()
#Missing values Discovery

bigmart_train.isnull().sum()
#So, Missing values are present in one numerical column- Item_weight and one Categorical column- Outlet_Size

#Lets plot out the distribution of the data in numerical column



bigmart_train['Item_Weight'].hist()

plt.show()
bigmart_train['Item_Weight'].fillna(bigmart_train['Item_Weight'].mean(), inplace=True)
sns.countplot(x='Outlet_Size', data=bigmart_train)

plt.show()
bigmart_train['Outlet_Size'].fillna('Medium', inplace=True)
#rechecking missing values in Dataset

bigmart_train.isnull().sum()
#lets examine Outlet_Establishment_Year

bigmart_train['Outlet_Establishment_Year'].value_counts()
#Lets understand the distribution of Establishment Year

bigmart_train['Outlet_Establishment_Year'].hist()

plt.show()
sns.barplot(data=bigmart_train, x='Outlet_Establishment_Year', y='Item_Outlet_Sales')

plt.show()
#Binning values



# Specify the boundaries of the bins

bins = [-np.inf, 1990, 2000, 2010]

# Bin labels

labels = ['1980-1990', '1990-2000', '2000-2010']



bigmart_train['Outlet_Establishment_Year']=pd.cut(bigmart_train['Outlet_Establishment_Year'], 

                                                  bins=bins, labels=labels)

bigmart_train['Outlet_Establishment_Year'].value_counts().astype('object')
obj_cols= bigmart_train.columns[bigmart_train.dtypes == 'object']
for obj_cols in bigmart_train[obj_cols]:

   print(bigmart_train[obj_cols].value_counts())
#examine Fat_Content column

bigmart_train['Item_Fat_Content'].value_counts()
bigmart_train['Item_Fat_Content']= bigmart_train['Item_Fat_Content'].replace(

    {'LF':'Low Fat','low fat':'Low Fat', 'reg':'Regular'})



print(bigmart_train['Item_Fat_Content'].value_counts())
bigmart_train['Item_Identifier'].value_counts()
bigmart_train['Item_Type_Extracted']=bigmart_train['Item_Identifier'].apply(lambda x:x[0:2])

bigmart_train['Item_Type_Extracted'].value_counts()
# Create a dictionary that maps strings

mapping = {'FD':'Food', 'NC':'Non Consumable', 'DR':'Drinks'}

bigmart_train['Item_Type_Combined']=bigmart_train['Item_Type_Extracted'].map(mapping)
bigmart_train['Item_Type_Combined'].value_counts().astype('object')
#Drop columns

bigmart_train= bigmart_train.drop(['Item_Identifier', 'Item_Type','Item_Type_Extracted',

                                   'Outlet_Identifier'], axis='columns')
bigmart_train.head()
bigmart_train["Outlet_Establishment_Year"]= bigmart_train["Outlet_Establishment_Year"].astype('object')

bigmart_train.dtypes
bigmart_train.describe()
bigmart_train.describe(exclude='number')
bigmart_train.corr()
bigmart_train.var()
bigmart_train.hist()

plt.show()
sns.barplot(data=bigmart_train, x='Item_Type_Combined',y='Item_Outlet_Sales', hue="Outlet_Location_Type")

plt.show()
sns.barplot(data=bigmart_train, x='Outlet_Location_Type',y='Item_Outlet_Sales', hue="Outlet_Type")

plt.show()
sns.barplot(data=bigmart_train, x='Outlet_Size',y='Item_Outlet_Sales')

plt.show()
sns.barplot(data=bigmart_train, x='Outlet_Establishment_Year', y='Item_Outlet_Sales', hue="Outlet_Location_Type")

plt.legend(loc='best')

plt.show()
sns.barplot(data=bigmart_train, x= "Item_Type_Combined", y="Item_Outlet_Sales", hue="Item_Fat_Content")

plt.legend(loc='best')

plt.show()
sns.scatterplot(data=bigmart_train, x="Item_Visibility", y="Item_Outlet_Sales")

plt.show()
bigmart_train.boxplot()

plt.show()
#Encode categorical columns to Numerical values to make ML algorithm understands it better

#here we use OneHotEncoding



categorical_cols= bigmart_train.columns[bigmart_train.dtypes == 'object'].tolist()

categorical_cols
encoded_df= pd.get_dummies(bigmart_train[categorical_cols], drop_first=True)

encoded_df.head()
#concatenate encoded dataframe with train dataframe

df_encoded= pd.concat([bigmart_train, encoded_df], axis=1)

df_encoded.head()
#Drop columns from dataset

df_final= df_encoded.drop(categorical_cols, axis=1)

df_final.head()
print(df_final.shape)
from sklearn.preprocessing import StandardScaler



scaler= StandardScaler()



scaler.fit(df_final[['Item_MRP']])

scaler.fit(df_final[['Item_Weight']])

scaler.fit(df_final[['Item_Visibility']])



df_final['Item_MRP_scaled']=scaler.transform(df_final[['Item_MRP']])

df_final['Item_Weight_scaled']=scaler.transform(df_final[['Item_Weight']])

df_final['Item_Visibility_scaled']=scaler.transform(df_final[['Item_Visibility']])



df_final[['Item_MRP_scaled', 'Item_MRP', 'Item_Weight_scaled', 'Item_Weight', 

          'Item_Visibility_scaled', 'Item_Visibility']].head()
#drop the original columns and keep the scaled column in dataframe

df=df_final.drop(['Item_MRP', 'Item_Weight', 'Item_Visibility'], axis=1)

df.head()
#seperate the training and test set



y= df['Item_Outlet_Sales']

X= df.drop('Item_Outlet_Sales', axis=1)
#import necessary scikit learn libraries

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error, accuracy_score



#Instantiate Linear Regression model

lr= LinearRegression(normalize=True)
# Compute 5-fold cross-validation scores: cv_scores

cv_scores= cross_val_score(lr, X, y, cv=5)



# Print the 5-fold cross-validation scores

print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
#Lets divide the data-set into training and test-set

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
#Fit the linear regression model to training data

lr.fit(X_train, y_train)
predicted_Outlet_sales = lr.predict(X_test)

mean_squared_error(y_test, predicted_Outlet_sales)
# Compute and print the RMSE

rmse = np.sqrt(mean_squared_error(y_test,predicted_Outlet_sales))

print("RMSE: %f" % (rmse))
#training set accuracy

lr.score(X_train, y_train)
#test set accuracy

lr.score(X_test, y_test)
#review test dataframe

bigmart_test.head()
bigmart_test.shape
#Missing Value Discovery in Test Data

bigmart_test.isnull().sum()
#impute the missing values same like Train data

bigmart_test['Item_Weight'].fillna(bigmart_test['Item_Weight'].mean(), inplace=True)
bigmart_test['Outlet_Size'].fillna('Medium', inplace=True)
bigmart_test.isnull().sum()
#Binning values



# Specify the boundaries of the bins

bins = [-np.inf, 1990, 2000, 2010]

# Bin labels

labels = ['1980-1990', '1990-2000', '2000-2010']



bigmart_test['Outlet_Establishment_Year']=pd.cut(bigmart_test['Outlet_Establishment_Year'], 

                                                  bins=bins, labels=labels)

bigmart_test['Outlet_Establishment_Year'].value_counts().astype('object')
obj_cols= bigmart_test.columns[bigmart_test.dtypes == 'object']
bigmart_test['Item_Fat_Content']= bigmart_test['Item_Fat_Content'].replace(

    {'LF':'Low Fat','low fat':'Low Fat', 'reg':'Regular'})



print(bigmart_test['Item_Fat_Content'].value_counts())
bigmart_test['Item_Type_Extracted']=bigmart_test['Item_Identifier'].apply(lambda x:x[0:2])

bigmart_test['Item_Type_Extracted'].value_counts()
# Create a dictionary that maps strings

mapping = {'FD':'Food', 'NC':'Non Consumable', 'DR':'Drinks'}

bigmart_test['Item_Type_Combined']=bigmart_test['Item_Type_Extracted'].map(mapping)
bigmart_test['Item_Type_Combined'].value_counts().astype('object')
#Drop columns

bigmart_test= bigmart_test.drop(['Item_Identifier', 'Item_Type','Item_Type_Extracted',

                                 'Outlet_Identifier'], axis='columns')
bigmart_test.head()
bigmart_test["Outlet_Establishment_Year"]= bigmart_test["Outlet_Establishment_Year"].astype('object')
categorical_cols_test= bigmart_test.columns[bigmart_test.dtypes == 'object'].tolist()

categorical_cols_test
test_df= pd.get_dummies(bigmart_test[categorical_cols_test], drop_first=True)

test_df.head()
df_test= pd.concat([bigmart_test, test_df], axis=1)

df_test.head()
#Drop columns from dataset

test_final= df_test.drop(categorical_cols, axis=1)

test_final.head()
#Apply trained scaler to the test set

test_final['Item_MRP_scaled']=scaler.transform(test_final[['Item_MRP']])

test_final['Item_Weight_scaled']=scaler.transform(test_final[['Item_Weight']])

test_final['Item_Visibility_scaled']=scaler.transform(test_final[['Item_Visibility']])



test_final[['Item_MRP_scaled', 'Item_MRP', 'Item_Weight_scaled', 'Item_Weight', 

          'Item_Visibility_scaled', 'Item_Visibility']].head()
#drop the original columns and keep the scaled column in datafram

test_X=test_final.drop(['Item_MRP', 'Item_Weight', 'Item_Visibility'], axis=1)

test_X.head()
#predicting Outlet Sales for Test Set

predicted_Outlet_Sales= lr.predict(test_X)

predicted_Outlet_Sales
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor





lr= LinearRegression()

ridge= Ridge(alpha=0.05,solver='cholesky')

lasso= Lasso(alpha=0.01)

dt= DecisionTreeRegressor()

rf= RandomForestRegressor(n_estimators=100)

br= BaggingRegressor(max_samples=70)

abr= AdaBoostRegressor()

gbr= GradientBoostingRegressor()
regressors = [('Linear Regression', lr), ('Ridge', ridge), ('Lasso', lasso), ('Decision Tree Regressor', dt),

             ('RandomForest Classifier', rf), ('Bagging Regressor', br), ('Ada Boost', abr), ('Gradient Boost', gbr)]



# Iterate over the pre-defined list of regressors

for reg_name, reg in regressors:   

    # Fit clf to the training set

    reg.fit(X_train, y_train)    

    y_pred = reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)



    print('{:s} : {:.3f}'.format(reg_name, mse))

    print('{:s} : {:.3f}'.format(reg_name, rmse))