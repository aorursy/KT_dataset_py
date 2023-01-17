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
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#from google.colab import files
import io
%matplotlib inline
sns.set_style('whitegrid')
#read data
data = pd.read_csv('https://raw.githubusercontent.com/kevinasyraf/find-it-2020-dac/master/hotel_bookings.csv')
data.head()
#lets get the whole overview of data
data.info()
#check for missing data
data.isnull().sum()
#lets fill null data with zero
data = data.drop('company', axis = 1)
data = data.fillna({
    'children' : 0,
    'agent' : 0,
    'country': 'Unknown',
})
#again check for null data
any(data.isna().sum())
# find no guest data
zero_guests = list(data.loc[data["adults"]
                   + data["children"]
                   + data["babies"]==0].index)
data.drop(data.index[zero_guests], inplace=True)
data.shape
#find outliers
sns.boxplot(data=data, x = 'lead_time')
plt.show()

sns.boxplot(data=data, x = 'adr')
plt.show()
#remove outliers using linear and non linear techniques
IQR_lt = data['lead_time'].quantile(0.75) -  data['lead_time'].quantile(0.25)
RUB = data['lead_time'].quantile(0.75) + 1.5*IQR_lt

data_no_outlier = data[data['lead_time'] <= RUB]
IQR_adr = data['adr'].quantile(0.75) -  data['adr'].quantile(0.25)
RUB = data['adr'].quantile(0.75) + 1.5*IQR_adr

data_no_outlier = data_no_outlier[data_no_outlier['adr'] <= RUB]
#lets see country wise data
data_country = pd.DataFrame(data.loc[data['is_canceled'] != 1]['country'].value_counts())
data_country.index.name = 'country'
data_country.rename(columns={"country": "Number of Guests"}, inplace=True)
total_guests = data_country["Number of Guests"].sum()
data_country["Guests in %"] = round(data_country["Number of Guests"] / total_guests * 100, 2)
data_country.head(10)
#lets see the guest map
import plotly.express as px
guest_map = px.choropleth(data_country,
                    locations=data_country.index,
                    color=data_country["Guests in %"], 
                    hover_name=data_country.index, 
                    color_continuous_scale=px.colors.sequential.Viridis,
                    title="Guest from countries")
guest_map.show()
#lets see month wise data of hotels
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
guest_data = data[data['is_canceled'] == 0].copy()
guests_monthly = guest_data[['hotel', 'arrival_date_year', 'arrival_date_month', 'adults', 'children', 'babies']].sort_values('arrival_date_year')
guests_monthly['total visitors'] = guests_monthly['adults'] + guests_monthly['children'] + guests_monthly['babies']
guests_monthly = guests_monthly.astype({'total visitors' : int})
guests_monthly = guests_monthly.drop(['adults', 'children', 'babies'], axis=1)
guests_monthly.head()
guests_monthly['arrival_date_month'] = pd.Categorical(guests_monthly['arrival_date_month'], categories=months, ordered=True)
guests_monthly = guests_monthly.groupby(['hotel', 'arrival_date_year', 'arrival_date_month'], as_index = False).sum()

f, ax = plt.subplots(3,1,figsize=(15,15))
sns.lineplot(x = 'arrival_date_month', y="total visitors", hue="hotel", data=guests_monthly[guests_monthly['arrival_date_year'] == 2015],  ci="sd", ax=ax[0])
sns.lineplot(x = 'arrival_date_month', y="total visitors", hue="hotel", data=guests_monthly[guests_monthly['arrival_date_year'] == 2016],  ci="sd", ax=ax[1])
sns.lineplot(x = 'arrival_date_month', y="total visitors", hue="hotel", data=guests_monthly[guests_monthly['arrival_date_year'] == 2017],  ci="sd", ax=ax[2])

ax[0].set(title="Visitors come in whole 2015")
ax[0].set(xlabel="Month", ylabel="Total Visitor")
ax[0].set(ylim = (0,5000))

ax[1].set(title="Visitors come in whole2016")
ax[1].set(xlabel="Month", ylabel="Total Visitor")
ax[1].set(ylim = (0,5000))

ax[2].set(title="Visitors come in whole 2017")
ax[2].set(xlabel="Month", ylabel="Total Visitor")
ax[2].set(ylim = (0,5000))

plt.show()
rh = data_no_outlier[(data_no_outlier['hotel'] == 'Resort Hotel') & (data_no_outlier['is_canceled'] == 0)]
ch = data_no_outlier[(data_no_outlier['hotel'] != 'Resort Hotel') & (data_no_outlier['is_canceled'] == 0)]
rh['adr_pp'] = rh['adr'] / (rh['adults'] + rh['children'])
ch['adr_pp'] = ch['adr'] / (ch['adults'] + ch['children'])
full_data_guests = data.copy()
full_data_guests = full_data_guests.loc[full_data_guests['is_canceled'] == 0]
full_data_guests['adr_pp'] = full_data_guests['adr'] / (full_data_guests['adults'] + full_data_guests['children'])
room_prices = full_data_guests[['hotel', 'reserved_room_type', 'adr_pp']].sort_values("reserved_room_type")
plt.figure(figsize=(10,5))
sns.barplot(x='reserved_room_type', y='adr_pp', hue='hotel', data=room_prices, hue_order=['City Hotel', 'Resort Hotel'], palette='pastel')
plt.title('Hotel room prices data', fontsize=16)
plt.xlabel('Tipe Kamar', fontsize = 16)
plt.ylabel('Euro (€)', fontsize = 16)
plt.show()
sns.countplot(x = 'reserved_room_type', data = data.sort_values('reserved_room_type'), hue='hotel')
#cancelled booking
data_canceled = data[data['is_canceled'] == 1].sort_values('market_segment')
data_not_canceled = data[data['is_canceled'] == 0].sort_values('market_segment')
f, ax = plt.subplots(1,2,figsize=(20,5))
sns.countplot(data=data_canceled, x= 'market_segment', hue='hotel', ax =ax[0])
sns.countplot(data=data_not_canceled, x= 'market_segment', hue='hotel', ax =ax[1])
ax[0].set(title='Market Segment basis cancellation of hotel booking')
ax[1].set(title='Market Segment basis cancellation of hotel booking')
plt.show()
#heatmap of whole dataset
plt.figure(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True)
#lets see adult and children data
adult_only = data[(data['adults'] != 0) & (data['children'] == 0) & (data['babies'] == 0)].sort_values('reserved_room_type')
adult_child = data[(data['adults'] != 0) & (data['children'] != 0) | (data['babies'] != 0)].sort_values('reserved_room_type')
percentage = [(len(adult_only)/(len(adult_only) + len(adult_child)))*100, (len(adult_child)/(len(adult_only) + len(adult_child)))*100]
labels = 'adults', 'children'

f, ax = plt.subplots(figsize=(7,7))
ax.pie(percentage, labels = labels, autopct='%1.1f%%' , startangle = 180)
ax.axis('equal')

ax.set_title('Percentage of adults and children came', fontsize=16)
plt.show()
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier, plot_importance, DMatrix, train
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
dtrain = pd.read_csv('https://raw.githubusercontent.com/kevinasyraf/find-it-2020-dac/master/hotel_bookings.csv')
nan_replacements = {"children:": 0.0,"country": "Unknown", "agent": 0, "company": 0}
dtrain = dtrain.fillna(nan_replacements)

# "meal" contains values "Undefined", which is equal to SC.
dtrain["meal"].replace("Undefined", "SC", inplace=True)
dtrain=dtrain.drop(['company'],axis=1)
dtrain=dtrain.dropna(axis=0)
dtrain.isna().sum()
#label encoding
# hotel
dtrain['hotel']=dtrain['hotel'].map({'Resort Hotel':0,'City Hotel':1})
dtrain['hotel'].unique()
# arrival_date_month
dtrain['arrival_date_month'] = dtrain['arrival_date_month'].map({'July':7,'August':8,'September':9,'October':10
                                                                ,'November':11,'December':12,'January':1,'February':2,'March':3,
                                                                'April':4,'May':5,'June':6})
dtrain['arrival_date_month'].unique()
label_encoder = LabelEncoder()
dtrain['meal']=label_encoder.fit_transform(dtrain['meal'])
dtrain['meal'].unique()
dtrain.head()
# Gathering which feature is more important.....using corr() function
correlation=dtrain.corr()['is_canceled']
correlation.abs().sort_values(ascending=False)
cols=['arrival_date_day_of_month','children',
     'arrival_date_week_number','stays_in_week_nights','reservation_status']
dtrain=dtrain.drop(cols,axis=1)
dtrain.head(5)
y=dtrain['is_canceled'].values
x=dtrain.drop(['is_canceled'],axis=1).values
#dataset split.
train_size=0.80
test_size=0.20
seed=5

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=train_size,test_size=test_size,random_state=seed)
ensembles=[]
ensembles.append(('scaledRFC',Pipeline([('scale',StandardScaler()),('rf',RandomForestClassifier(n_estimators=10))])))

results=[]
names=[]
for name,model in ensembles:
    fold = KFold(n_splits=10,random_state=5)
    result = cross_val_score(model,x_train,y_train,cv=fold,scoring='accuracy')
    results.append(result)
    names.append(name)
    msg="%s : %f (%f)"%(name,result.mean(),result.std())
    print(msg)
# Random Forest Classifier Tuning
from sklearn.model_selection import GridSearchCV

scaler=StandardScaler().fit(x_train)
rescaledx=scaler.transform(x_train)

n_estimators=[10,20,30,40,50]

param_grid=dict(n_estimators=n_estimators)

model=RandomForestClassifier()

fold=KFold(n_splits=10,random_state=0)

grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy',cv=fold)
grid_result=grid.fit(rescaledx,y_train)

print("Best: %f using %s "%(grid_result.best_score_,grid_result.best_params_))
from sklearn.metrics import confusion_matrix

scaler=StandardScaler().fit(x_train)
scaler_x=scaler.transform(x_train)
model=RandomForestClassifier(n_estimators=40)
model.fit(scaler_x,y_train)

#Transform the validation test set data
scaledx_test=scaler.transform(x_test)
y_pred=model.predict(scaledx_test)

accuracy_mean=accuracy_score(y_test,y_pred)
accuracy_matric=confusion_matrix(y_test,y_pred)
print(accuracy_mean)
print(accuracy_matric)
y_pred = model.predict(scaler.transform(x))
print(accuracy_score(y, y_pred))
