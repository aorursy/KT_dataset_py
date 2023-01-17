import pandas as pd
import numpy as np
import codecs as cd
import matplotlib.pyplot as plt

# the commonly used alias for seaborn is sns
import seaborn as sns
# set a seaborn style of your taste
sns.set(style="whitegrid", color_codes=True)
import pytz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# to evaluate the models
from sklearn.metrics import mean_squared_error

pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore')
Train=pd.read_csv('../input/BlackFriday.csv')#import train dataset
Train.info()
Train.describe()
# look for duplicates before data cleaning
print(Train.duplicated().sum())
Train.head()
# Validate all values in column is same by checking unique values in column
uniques = Train.apply(lambda x:x.nunique())
print(uniques)
print('Number of User Id labels: ', len(Train.User_ID.unique()))
print('Number of rows in the Dataset: ', len(Train))
# find categorical variables
categorical = [var for var in Train.columns if Train[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))
# let's visualise the percentage of missing values
for var in Train.columns:
    if Train[var].isnull().sum()>0:
        print(var, Train[var].isnull().mean())
#Replacing null values of 'Product_Category_2', 'Product_Category_3' with 0
Train['Product_Category_2'] = Train['Product_Category_2'].fillna(0).astype(int)
Train['Product_Category_3'] = Train['Product_Category_3'].fillna(0).astype(int)
# Create a derived metric combining gender and marital status
Train['combined_G_M'] = Train.apply(lambda x:'%s%s' % (x['Gender'],x['Marital_Status']),axis=1)
#replacing categoriacal value like combined_G_M into numerical 
Train=Train.replace({'combined_G_M':{'F0':0,'M0':1,'M1':2,'F1':3}})
#dropping Name as we have column title that would be neccesary for general analysis not the name
Train.drop(['Gender','Marital_Status'],axis=1, inplace=True)
Train.head()
Train.City_Category.unique()
#replacing categorical value City Category into numerical 
Train=Train.replace({'City_Category':{'A':0,'B':1,'C':2}})
Train["Stay_In_Current_City_Years"] = Train["Stay_In_Current_City_Years"].str.replace("+", "").astype("int")
Train.info()
# get whole set of dummy variables
Train = pd.get_dummies(data=Train, columns=['Age'])
# Let's separate into train and validation set
X_train, X_cv, y_train, y_cv = train_test_split(Train, Train.Purchase, test_size=0.2,
                                                    random_state=0)
X_train.shape, X_cv.shape
training_vars = [var for var in X_train.columns if var not in ['User_ID', 'Product_ID', 'Purchase']]

# fit scaler
scaler = StandardScaler() # create an instance
scaler.fit(Train[training_vars]) #  fit  the scaler to the train set for later use
rf_model = RandomForestRegressor()
rf_model.fit(X_train[training_vars], y_train)

pred = rf_model.predict(X_train[training_vars])
print('rf train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = rf_model.predict(X_cv[training_vars])
print('rf test mse: {}'.format(mean_squared_error(y_cv, pred)))
lin_model = Lasso(random_state=2909)
lin_model.fit(scaler.transform(X_train[training_vars]), y_train)

pred = lin_model.predict(scaler.transform(X_train[training_vars]))
print('linear train mse: {}'.format(mean_squared_error(y_train, pred)))
pred = lin_model.predict(scaler.transform(X_cv[training_vars]))
print('linear test mse: {}'.format(mean_squared_error(y_cv, pred)))