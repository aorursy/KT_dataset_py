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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report
from imblearn.over_sampling import SMOTE

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
hotel_bookings_df= pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
hotel_bookings_df.shape
hotel_bookings_df.dtypes
hotel_bookings_df.isnull().sum()
hotel_bookings_df.drop(['company'],inplace=True,axis=1)
hotel_bookings_df['agent'].fillna(hotel_bookings_df['agent'].mode()[0],inplace=True)
hotel_bookings_df['country'].fillna(hotel_bookings_df['country'].mode()[0],inplace=True)
hotel_bookings_df['children'].fillna(hotel_bookings_df['children'].mean(),inplace=True)
categorical_columns=[]
continuous_columns=[]
for col in hotel_bookings_df.columns:
    if hotel_bookings_df[col].dtype!='object':
        continuous_columns.append(col)
    else:
        categorical_columns.append(col)
    
continuous_columns
plt.figure(figsize=(16,16))
for i, col in enumerate(['lead_time', 'total_of_special_requests', 'days_in_waiting_list','booking_changes', 'adults','children', 'babies','adr']):
    plt.subplot(4,4,i+1)
    sns.boxplot(hotel_bookings_df[col])
    plt.tight_layout()
hotel_bookings_df.loc[hotel_bookings_df.lead_time> 450,'lead_time']=450
hotel_bookings_df.loc[hotel_bookings_df.days_in_waiting_list> 125,'days_in_waiting_list']=125
hotel_bookings_df.loc[hotel_bookings_df.booking_changes> 10,'booking_changes']=10
hotel_bookings_df.loc[hotel_bookings_df.adults> 20,'adults']=20
hotel_bookings_df.loc[hotel_bookings_df.children> 4,'children']=4
hotel_bookings_df.loc[hotel_bookings_df.adr> 400,'adr']=400
hotel_bookings_df['hotel'].value_counts()
bool_dict = {1:'Yes',0:'No'}
hotel_bookings_df['is_canceled']=hotel_bookings_df['is_canceled'].map(bool_dict)
plt.figure(figsize=(12,6))
cancel_hotel_df=hotel_bookings_df.groupby(['hotel','is_canceled']).size().reset_index().rename(columns=({0:'count'}))
sns.barplot(x='hotel',y='count',data=cancel_hotel_df,hue='is_canceled')

plt.figure(figsize=(12,6))
month_booking_df = hotel_bookings_df['arrival_date_month'].value_counts(normalize=True).rename_axis('Month').reset_index(name='Percentage')
sns.barplot(x='Month',y='Percentage',data=month_booking_df)
plt.figure(figsize=(12,6))
Months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_booking_df = hotel_bookings_df.groupby(['arrival_date_month','is_canceled']).size().reset_index().rename(columns=({0:'count'})).sort_values(['count'],ascending=False)
sns.barplot(x='arrival_date_month',y='count',data=month_booking_df,hue='is_canceled',order=Months)

plt.figure(figsize=(12,6))
sns.countplot(data=hotel_bookings_df,x='arrival_date_year',hue='is_canceled')
plt.figure(figsize=(12,6))
country_booking_df = hotel_bookings_df['country'].value_counts(normalize=True).rename_axis('country').reset_index(name='Percentage')
country_booking_df=country_booking_df.head(15)
sns.barplot(x='country',y='Percentage',data=country_booking_df)

country_booking=hotel_bookings_df.groupby(['country']).size().reset_index().rename(columns=({0:'count'}))
country_booking_canceled=hotel_bookings_df.groupby(['country','is_canceled']).size().reset_index().rename(columns=({0:'count'}))
country_booking=country_booking.merge(country_booking_canceled,on='country')
country_booking['percentage_cancellation']=country_booking['count_y']/country_booking['count_x']
country_booking_top=country_booking[(country_booking['percentage_cancellation']>=0.4)&(country_booking['is_canceled']=='Yes')&((country_booking['count_x']>50))].copy()
country_booking_top.rename(columns=({'count_x':'Total_Booking'}),inplace=True)
print(list(country_booking_top.country))

booking_year_children=hotel_bookings_df.groupby(['babies','arrival_date_year']).size().reset_index().rename(columns=({0:'count'}))
booking_year_children
countries_with_booking=hotel_bookings_df.country.value_counts().reset_index(name="count").query("count > 50")
print(list(countries_with_booking['index']))

plt.figure(figsize=(20,6))
sns.FacetGrid(hotel_bookings_df[(hotel_bookings_df['days_in_waiting_list']>0)], hue = 'is_canceled',
             height = 6,xlim = (0,150)).map(sns.kdeplot, 'days_in_waiting_list', shade = True,bw=2).add_legend()
plt.figure(figsize=(20,6))
sns.FacetGrid(hotel_bookings_df[(hotel_bookings_df['lead_time']>0)], hue = 'is_canceled',
             height = 8,xlim = (0,500)) .map(sns.kdeplot, 'lead_time', shade = True,bw=2).add_legend()
plt.figure(figsize=(12,6))
sns.countplot(data=hotel_bookings_df,x='market_segment',hue='deposit_type')
plt.figure(figsize=(12,6))
sns.countplot(data=hotel_bookings_df,hue='is_canceled',x='market_segment')

hotel_bookings_df['arrival_date_month'] = hotel_bookings_df['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
hotel_bookings_df['hotel'] = hotel_bookings_df['hotel'].map({'Resort Hotel':0, 'City Hotel':1})
hotel_bookings_df["total_members"] = hotel_bookings_df["adults"] + hotel_bookings_df["children"] + hotel_bookings_df["babies"]
hotel_bookings_df["total_stay"] = hotel_bookings_df["stays_in_weekend_nights"]+ hotel_bookings_df["stays_in_week_nights"]
hotel_bookings_df.drop(columns = ['adults', 'babies', 'children', 'stays_in_weekend_nights', 'stays_in_week_nights'],inplace=True,axis=1)
le = LabelEncoder()
hotel_bookings_df['meal'] = le.fit_transform(hotel_bookings_df['meal'])
hotel_bookings_df['country'] = le.fit_transform(hotel_bookings_df['country'])
hotel_bookings_df['distribution_channel'] = le.fit_transform(hotel_bookings_df['distribution_channel'])
hotel_bookings_df['reserved_room_type'] = le.fit_transform(hotel_bookings_df['reserved_room_type'])
hotel_bookings_df['assigned_room_type'] = le.fit_transform(hotel_bookings_df['assigned_room_type'])
hotel_bookings_df['deposit_type'] = le.fit_transform(hotel_bookings_df['deposit_type'])
hotel_bookings_df['customer_type'] = le.fit_transform(hotel_bookings_df['customer_type'])
hotel_bookings_df['reservation_status'] = le.fit_transform(hotel_bookings_df['reservation_status'])
hotel_bookings_df['market_segment'] = le.fit_transform(hotel_bookings_df['market_segment'])
hotel_bookings_df['reservation_status_date'] = le.fit_transform(hotel_bookings_df['reservation_status_date'])
hotel_bookings_df['is_canceled'] = le.fit_transform(hotel_bookings_df['is_canceled'])
hotel_bookings_df=hotel_bookings_df[['hotel', 'lead_time', 'arrival_date_year',
       'arrival_date_month', 'arrival_date_week_number',
       'arrival_date_day_of_month', 'meal', 'country', 'market_segment',
       'distribution_channel', 'is_repeated_guest', 'previous_cancellations',
       'previous_bookings_not_canceled', 'reserved_room_type',
       'assigned_room_type', 'booking_changes', 'deposit_type', 'agent',
       'days_in_waiting_list', 'customer_type', 'adr',
       'required_car_parking_spaces', 'total_of_special_requests',
       'reservation_status', 'reservation_status_date', 'total_members',
       'total_stay', 'is_canceled']]
plt.figure(figsize=(12,12))
sns.heatmap(hotel_bookings_df.corr())
hotel_bookings_df.drop(['reservation_status'],inplace=True,axis=1)
hotel_bookings_df.drop(['arrival_date_year','arrival_date_day_of_month','assigned_room_type'],inplace=True,axis=1)
def evaluation_stats(model,X_train, X_test, y_train, y_test,algo,is_feature=True):
    print('Train Accuracy')
    if algo=='NN':
        print(confusion_matrix(y_train,model.predict_classes(X_train)))
        y_pred = model.predict_classes(X_test)
    else:
        print(confusion_matrix(y_train,model.predict(X_train)))
        y_pred = model.predict(X_test)
    print('Validation Accuracy')
    
    print(confusion_matrix(y_test,y_pred))
    print('Classification_report')
    print(classification_report(y_test,y_pred))
    if is_feature:
        plot_feature_importance(rf_model.feature_importances_,X.columns,algo)

def training(model,X_train, y_train):
    return model.fit(X_train, y_train)

def plot_feature_importance(importance,names,model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
hotel_bookings_df['is_canceled'].value_counts()
X = hotel_bookings_df.drop(["is_canceled"], axis=1)
y = hotel_bookings_df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 101)

sm = SMOTE(random_state=101)
X_res, y_res = sm.fit_resample(X_train, y_train)
rf_model = training(RandomForestClassifier(n_estimators=1000,max_depth=10),X_res, y_res)
evaluation_stats(rf_model,X_train, X_test, y_train, y_test,'RANDOM FOREST')

xbg_model = training(XGBClassifier(n_estimators=1000,max_depth=10),X_res, y_res)
evaluation_stats(xbg_model,X_train, X_test, y_train, y_test,'RANDOM FOREST')

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(12, input_dim=23, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_res, y_res, epochs=20, batch_size=64)
evaluation_stats(model,X_train, X_test, y_train, y_test,'NN',is_feature=False)


