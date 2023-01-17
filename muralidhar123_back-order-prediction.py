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

pd.set_option('display.max_columns', None)

df = pd.read_csv('/kaggle/input/back-order/Training_Dataset_v2.csv')

df

#droppping last row

df=df.drop(df.iloc[-1:].index,inplace=False)


df.isnull().sum()
df['lead_time'].value_counts().unique()
df.tail()
df['national_inv'].unique()
df.columns
#replacing nan values in dateset

df['national_inv']=df['national_inv'].fillna(df['national_inv'].mean())

df['lead_time'] = df['lead_time'].fillna(df['lead_time'].mean())

df['in_transit_qty']=df['in_transit_qty'].fillna(df['in_transit_qty'].mean())

df['forecast_3_month']=df['forecast_3_month'].fillna(df['forecast_3_month'].mean())

df['forecast_6_month']=df['forecast_6_month'].fillna(df['forecast_6_month'].mean())

df['forecast_9_month']=df['forecast_9_month'].fillna(df['forecast_9_month'].mean())

df['sales_1_month']=df['sales_1_month'].fillna(df['sales_1_month'].mean())

df['sales_3_month']=df['sales_3_month'].fillna(df['sales_3_month'].mean())

df['sales_6_month']=df['sales_6_month'].fillna(df['sales_6_month'].mean())

df['sales_9_month']=df['sales_9_month'].fillna(df['sales_9_month'].mean())

df['min_bank']=df['min_bank'].fillna(df['min_bank'].mean())

df['potential_issue']=df['potential_issue'].replace({'Yes':1,'No':0})

df['potential_issue']=df['potential_issue'].fillna(df['potential_issue'].mean())

df['pieces_past_due']=df['pieces_past_due'].fillna(df['pieces_past_due'].mean())

df['perf_6_month_avg']=df['perf_6_month_avg'].fillna(df['perf_6_month_avg'].mean())

df['perf_12_month_avg']=df['perf_12_month_avg'].fillna(df['perf_12_month_avg'].mean())

df['local_bo_qty']=df['local_bo_qty'].fillna(df['local_bo_qty'].mean())

#replaceing yes and no into 1 and 0

df['deck_risk']=df['deck_risk'].replace({'Yes':1,'No':0})

df['deck_risk']=df['deck_risk'].fillna(df['deck_risk'].mean())

df['oe_constraint']=df['oe_constraint'].replace({'Yes':1,'No':0})

df['oe_constraint']=df['oe_constraint'].fillna(df['oe_constraint'].mean())

df['ppap_risk']=df['ppap_risk'].replace({'Yes':1,'No':0})

df['ppap_risk']=df['ppap_risk'].fillna(df['ppap_risk'].mean())

df['stop_auto_buy']=df['stop_auto_buy'].replace({'Yes':1,'No':0})

df['stop_auto_buy']=df['stop_auto_buy'].fillna(df['stop_auto_buy'].mean())

df['rev_stop']=df['rev_stop'].replace({'Yes':1,'No':0})

df['rev_stop']=df['rev_stop'].fillna(df['rev_stop'].mean())

df['went_on_backorder']=df['went_on_backorder'].replace({'Yes':1,'No':0})

df['went_on_backorder']=df['went_on_backorder'].fillna(df['went_on_backorder'].mean())
df.isnull().sum()
df.head()


#finding outliers in dataset

fig,ax = plt.subplots(figsize=(35,15),facecolor='white')

sns.boxplot(data=df,ax=ax,width=0.6,fliersize=4)


df.describe()
#playing with outliers

q=df['national_inv'].quantile(0.85)

data_cleaned=df[df['national_inv']<q]

q=df['lead_time'].quantile(0.98)

data_cleaned=df[df['lead_time']<q]

q=df['in_transit_qty'].quantile(0.95)

data_cleaned=df[df['in_transit_qty']<q]

q=df['forecast_3_month'].quantile(0.93)

data_cleaned=df[df['forecast_3_month']<q]

q=df['forecast_6_month'].quantile(0.90)

data_cleaned=df[df['forecast_6_month']<q]

q=df['forecast_9_month'].quantile(0.88)

data_cleaned=df[df['forecast_9_month']<q]

q=df['sales_1_month'].quantile(0.96)

data_cleaned=df[df['sales_1_month']<q]

q=df['sales_3_month'].quantile(0.93)

data_cleaned=df[df['sales_3_month']<q]

q=df['sales_1_month'].quantile(0.90)

data_cleaned=df[df['sales_6_month']<q]

q=df['sales_9_month'].quantile(0.88)

data_cleaned=df[df['sales_9_month']<q]

q=df['min_bank'].quantile(0.97)

data_cleaned=df[df['min_bank']<q]

q=df['pieces_past_due'].quantile(0.98)

data_cleaned=df[df['pieces_past_due']<q]


#correlation features

cormat = df.corr()

corr_feature = cormat.index

plt.figure(figsize=(35,40))

g=sns.heatmap(df[corr_feature].corr(),annot=True,cmap="YlGnBu")
X = df.drop(labels='went_on_backorder',axis=1)
X.dtypes
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled=scaler.fit_transform(X)


y = df['went_on_backorder']

y.head()
X_scaled


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.70,random_state=152)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()

rfc=rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy_score(y_pred,y_test)
from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(rfc,X_test,y_test,cmap=plt.cm.Blues,normalize=None)

disp.ax_.set_title('Confusion Matrix')

plt.show()
import pickle

pickle.dump(rf,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1043852,7.0,8.000000,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0,0.0,0.10,0.13,0.0,1,0,0,0,0]]))