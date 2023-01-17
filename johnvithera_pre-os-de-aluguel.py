# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data =  pd.read_csv('/kaggle/input/singapore-airbnb/listings.csv')

data.head(10)
data.info()
plt.figure(figsize=(12,6))

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
del data['last_review']

del data['reviews_per_month']
plt.figure(figsize=(12,6))

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
from sklearn.preprocessing import LabelEncoder

preprocessing = LabelEncoder();

le = preprocessing                                            # Fit label encoder

le.fit(data['neighbourhood_group'])

data['neighbourhood_group']=le.transform(data['neighbourhood_group'])    # Transform labels to normalized encoding.



le = preprocessing

le.fit(data['neighbourhood'])

data['neighbourhood']=le.transform(data['neighbourhood'])



le = preprocessing

le.fit(data['room_type'])

data['room_type']=le.transform(data['room_type'])



data.sort_values(by='price',ascending=True,inplace=True)



data.head()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

lm = LinearRegression()



X = data[['host_id','neighbourhood_group','neighbourhood','latitude','longitude','room_type','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']]

y = data['price']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)



lm.fit(X_train,y_train)
predicts = lm.predict(X_test)

error_airbnb = pd.DataFrame({

        'Actual Values': np.array(y_test).flatten(),

        'Predicted Values': predicts.flatten()})

error_airbnb.head()
from sklearn.metrics import mean_absolute_error



mae = mean_absolute_error(predicts, y_test)

print("Mean Absolute Error:" , mae)
from sklearn.model_selection import train_test_split

X = pd.read_csv('/kaggle/input/singapore-airbnb/listings.csv')

X_test_full = pd.read_csv('/kaggle/input/singapore-airbnb/listings.csv')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['price'], inplace=True)

y = X.price              

X.drop(['price'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 20 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
import xgboost as xgb



my_model_1 = xgb.XGBClassifier(random_state=0).fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error

predictions_1 = my_model_1.predict(X_valid)

mae_1 = mean_absolute_error(predictions_1, y_valid)

print("Mean Absolute Error:" , mae_1)