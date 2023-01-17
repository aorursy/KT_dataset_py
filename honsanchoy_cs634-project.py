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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

data.loc[data['meal'] == 'Undefined', 'meal'] = 'SC'

data.loc[data['agent'].isnull(), 'agent'] = 0

data.loc[data['company'].isnull(), 'company'] = 0

missing = [

    'country', 

    'children']

data = data.dropna(subset=missing)

data1 = data[data['adr'] > 0]

df_le = data1.copy()

le = LabelEncoder()



categoricals = [

    'arrival_date_month',

    'meal',

    'country',

    'market_segment',

    'distribution_channel',

    'reserved_room_type',

    'assigned_room_type',

    'deposit_type',

    'agent',

    'company',

    'customer_type',

    'reservation_status',

]



for col in categoricals:

    df_le[col] = le.fit_transform(df_le[col])



df_le = df_le.drop(['reservation_status','reservation_status_date'], axis=1)

df_le = df_le.drop(['arrival_date_year','assigned_room_type','market_segment'], axis=1)

df_le = df_le.drop(['booking_changes'], axis=1)

df_le = df_le.drop(['country'], axis=1)

new_columns = [i for i in df_le.columns]

data2 = data1[new_columns]

data_dummies=pd.get_dummies(data=data2, columns=['hotel', 'distribution_channel', 'deposit_type', 'customer_type', 'is_repeated_guest','arrival_date_month','meal','reserved_room_type'])



random_state=10

X = data_dummies.drop(['is_canceled'], axis=1).values

y = data_dummies[['is_canceled','hotel_City Hotel']]



# X = SelectKBest(chi2, k=k).fit_transform(X, y)

# X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)



y_train,train_hotel_name=y_train.is_canceled,y_train['hotel_City Hotel']

y_test,test_hotel_name=y_test.is_canceled,y_test['hotel_City Hotel']

rf = RandomForestClassifier(n_estimators=100, random_state=random_state)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

RF = accuracy_score(y_test, y_pred)

# print('Accuracy of Random Forest Classifier: {:.3f}'.format(RF))

result=pd.DataFrame()

result['Hotel Name']=test_hotel_name

result['Hotel Name']=np.where(result['Hotel Name']==1,'City Hotel','Resort Hotel')

result['Booking_Possibility']=y_pred

result.to_csv('result.csv')


result