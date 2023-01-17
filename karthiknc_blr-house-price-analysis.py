import numpy as np 

import pandas as pd

import re



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.linear_model import LogisticRegression,LinearRegression

from sklearn import metrics

from math import sqrt

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor
data=pd.read_csv('../input/bengaluru-house-price-data/Bengaluru_House_Data.csv')
data.head()
data.describe().T
data.info()
data.isnull().sum()
data['society'].shape
data['size'].unique()
data.corr()
sns.pairplot(data)
sns.distplot(data['price'])
data.select_dtypes(exclude=['object']).describe()
corr=data.corr()
sns.heatmap(corr)
from collections import Counter

Counter(data['total_sqft'])
data.shape
#preprocessing the total sqft cols as it has vivid entries

def preprocess_total_sqft(my_list):

    if len(my_list) == 1:

        

        try:

            return float(my_list[0])

        except:

            strings = ['Sq. Meter', 'Sq. Yards', 'Perch', 'Acres', 'Cents', 'Guntha', 'Grounds']

            split_list = re.split('(\d*.*\d)', my_list[0])[1:]

            area = float(split_list[0])

            type_of_area = split_list[1]

            

            if type_of_area == 'Sq. Meter':

                area_in_sqft = area * 10.7639

            elif type_of_area == 'Sq. Yards':

                area_in_sqft = area * 9.0

            elif type_of_area == 'Perch':

                area_in_sqft = area * 272.25

            elif type_of_area == 'Acres':

                area_in_sqft = area * 43560.0

            elif type_of_area == 'Cents':

                area_in_sqft = area * 435.61545

            elif type_of_area == 'Guntha':

                area_in_sqft = area * 1089.0

            elif type_of_area == 'Grounds':

                area_in_sqft = area * 2400.0

            return float(area_in_sqft)

        

    else:

        return (float(my_list[0]) + float(my_list[1]))/2.0
data['total_sqft'] = data.total_sqft.str.split('-').apply(preprocess_total_sqft)
#converting the categorical to numerical data - area_type

data.area_type.value_counts()
replace_area_type = {'Super built-up  Area': 0, 'Built-up  Area': 1, 'Plot  Area': 2, 'Carpet  Area': 3}

data['area_type'] = data.area_type.map(replace_area_type)
#converting the categorical to numerical data - availabilty

data.availability.value_counts()
def replace_availabilty(my_string):

    if my_string == 'Ready To Move':

        return 0

    elif my_string == 'Immediate Possession':

        return 1

    else:

        return 2
data['availability'] = data.availability.apply(replace_availabilty)
#converting NaN in location

data['location'].isnull().sum()
data['location'] = data['location'].fillna('No Location')
#converting the categorical to numerical data - size

Counter(data['size'])
le = LabelEncoder()

le.fit(data['size'].astype('str').append(data['size'].astype('str')))

data['size'] = le.transform(data['size'].astype('str'))
data.head()
#converting the NaNs to other - society

data['society'] = data['society'].fillna('Other')
le.fit(data['society'].append(data['society'].fillna('Other')))

data['society'] = le.transform(data['society'])
#converting the categorical to numerical data - location

Counter(data['location'])
le.fit(data['location'].append(data['location'].fillna('other')))

data['location']=le.transform(data['location'])
#converting NaNs in bath

data['bath'].isna().sum()
#missing values are filled by grouping the rows based on location and taking the mean of the column 'bath' in that location.

col_bath=data.groupby('location')['bath'].transform(lambda x: x.fillna(x.mean()))
col_bath.isna().sum()
col_bath[~col_bath.notnull()]
#col 1775 has nan even after transformation

col_bath = col_bath.fillna(col_bath.mean())
#finally its resolved

col_bath.isnull().sum()
data['bath']=col_bath
#missing values are filled by grouping the rows based on location and taking the mean of the column 'balcony' in that location.

data['balcony'].isnull().sum()
col_balcony=data.groupby('location')['balcony'].transform(lambda x: x.fillna(x.mean()))
col_balcony.isna().sum()
#col 45 has nan even after transformation

col_balcony = col_balcony.fillna(col_balcony.mean())
col_balcony.isnull().sum()
data['balcony']=col_balcony
data.head()
#preprocessing for building ML models

x=data.drop('price',axis=1)

y=data['price']
data.info()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)
#Linear Regression

lr=LinearRegression()
lr.fit(x_train,y_train)
lpred=lr.predict(x_test)

print(lpred)
lrrmse=np.sqrt(np.mean((y_test-lpred)**2))

lrrmse
#Decision Tree

dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
dtpred=dt.predict(x_test)

print(dtpred)
dtrmse=np.sqrt(np.mean((y_test-dtpred)**2))

dtrmse
#Random Forest

rf=RandomForestRegressor()
rf.fit(x_train,y_train)
rfpred=rf.predict(x_test)

print(rfpred)
rfrmse=np.sqrt(np.mean((y_test-rfpred)**2))

rfrmse