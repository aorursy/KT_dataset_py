#importing libraries

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



# reading our from our file

data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

data.head()
data.dtypes
data.columns
data.isna().sum()
fig,axes = plt.subplots(1,1,figsize=(15,5))

sns.heatmap(data.isna(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
data.drop('company',axis=1,inplace=True)
sns.heatmap(data.corr())

plt.show()
#dealing with  missing data

data['children'] = data['children'].fillna(data['children'].median(),axis=0)

data['country']=data['country'].fillna(value='PRT')

data.drop(['agent'],axis=1,inplace=True)
data.isnull().sum()
# transforming categorial data using label encoding



from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 





data['customer_type']= label_encoder.fit_transform(data['customer_type']) 

data['assigned_room_type'] = label_encoder.fit_transform(data['assigned_room_type'])

data['deposit_type'] = label_encoder.fit_transform(data['deposit_type'])

data['reservation_status'] = label_encoder.fit_transform(data['reservation_status'])

data['meal'] = label_encoder.fit_transform(data['meal'])

data['country'] = label_encoder.fit_transform(data['country'])

data['distribution_channel'] = label_encoder.fit_transform(data['distribution_channel'])

data['market_segment'] = label_encoder.fit_transform(data['market_segment'])

data['reserved_room_type'] = label_encoder.fit_transform(data['reserved_room_type'])

data['reservation_status_date'] = label_encoder.fit_transform(data['reservation_status_date'])

data['arrival_date_month'] = label_encoder.fit_transform(data['arrival_date_month'])

data['hotel'] = label_encoder.fit_transform(data['hotel'])



print('customer_type:', data['customer_type'].unique())

print('reservation_status', data['reservation_status'].unique())

print('deposit_type', data['deposit_type'].unique())

print('assigned_room_type', data['assigned_room_type'].unique())

print('meal', data['meal'].unique())

print('Country:',data['country'].unique())

print('Dist_Channel:',data['distribution_channel'].unique())

print('Market_seg:', data['market_segment'].unique())

print('reserved_room_type:', data['reserved_room_type'].unique())
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import LogisticRegression
x = data.drop(('is_canceled'),axis=1)

y = data['is_canceled']
#Linear Regression

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.20,random_state=42)

linear = LinearRegression()

linear.fit(xtrain,ytrain)

pred = linear.predict(xtest)
print("MAE",metrics.mean_absolute_error(ytest,pred))

print("MSE",metrics.mean_squared_error(ytest,pred))

print('RMSE',np.sqrt(metrics.mean_squared_error(ytest, pred)))

print('r2_score:', r2_score(ytest, pred))