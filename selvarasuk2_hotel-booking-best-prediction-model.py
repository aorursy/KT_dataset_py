import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''Read the data'''

df_hot = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

df_hot.head()
df_hot.isnull().sum()
df_hot['children'] = df_hot['children'].fillna(df_hot['children'].median(),axis=0)

df_hot['country']=df_hot['country'].fillna(df_hot['country'].mode().to_string() )
df_hot.drop('agent',axis=1,inplace=True)

df_hot.drop('company',axis=1,inplace=True)
df_hot.isnull().sum()
#df_hot.dtypes
df_hot_obj = [ 'hotel','stays_in_week_nights','adults','children','babies','distribution_channel','is_repeated_guest',

'previous_bookings_not_canceled','assigned_room_type','customer_type','adr','required_car_parking_spaces'

, 'arrival_date_month','meal','country','market_segment','reserved_room_type','reserved_room_type','reservation_status',

             'reservation_status_date','deposit_type','reserved_room_type','reserved_room_type'

             ,'reserved_room_type']

df_hot [ df_hot_obj]= df_hot [ df_hot_obj].astype('category')
df_hot.dtypes
fig,axes = plt.subplots(1,1,figsize=(18,17))

sns.heatmap(df_hot.corr(),annot=True)

plt.show()
from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()

features=['lead_time','total_of_special_requests','required_car_parking_spaces','booking_changes',

          'previous_cancellations','is_repeated_guest','adults','previous_bookings_not_canceled','days_in_waiting_list'

        , 'hotel',

'arrival_date_month',

'stays_in_week_nights',

'adults',

'children',

'babies',

'country',

'market_segment',

'distribution_channel',

'is_repeated_guest',

'previous_bookings_not_canceled',

'deposit_type',

'customer_type',

'adr',

'required_car_parking_spaces'

]



for i in features :

    df_hot[i] = label_encoder.fit_transform(df_hot[i])
from sklearn.model_selection import train_test_split 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestRegressor  
X=df_hot[features]

y=df_hot['is_canceled']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

LR=LogisticRegression(solver = 'lbfgs')

LR.fit(X_train,y_train)

y_pred = LR.predict(X_test)

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))

random_forest = RandomForestRegressor(n_estimators = 100, random_state = 42)

random_forest.fit(X_train,y_train)

y_pred_random_forest = random_forest.predict(X_test)

print(confusion_matrix(y_test,y_pred_random_forest.round()))

print(classification_report(y_test,y_pred_random_forest.round()))

print('Accuracy score of each models')

print('LogisticRegression:' ,accuracy_score(y_test, y_pred))

print('RandomForestRegressor:', accuracy_score(y_test, y_pred_random_forest.round()))