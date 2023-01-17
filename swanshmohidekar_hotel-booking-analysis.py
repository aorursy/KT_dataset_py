# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/hotel-booking-demand/hotel_bookings.csv")
df
df.isnull().sum()
df['reserved_room_type'] = df['reserved_room_type'].map({'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'P':8,'L':9})
df['arrival_date_year'].value_counts()
df['deposit_type'] = df['deposit_type'].map({'No Deposit':0,'Non Refund':1,'Refundable':2})
df['hotel'] = df['hotel'].map({'Resort Hotel':0,'City Hotel':1})
df['arrival_date_month'] = df['arrival_date_month'].map({'January':0,'February':1,'March': 2,'Ápril':3,'May':4,'June':5,'July':6,'August':7,'September':8,'October':9,'November':10,'December':11})
df['total'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
df
df.info()
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


plt.figure(figsize=(15,10))
sns.countplot(x='adults', data = df,
              order=pd.value_counts(df['adults']).index, palette='YlOrBr_r')
plt.title('Number of Adults logging in', weight='bold')
plt.xlabel('Count of Adults', fontsize=10)
plt.ylabel('Number', fontsize=10)
sns.boxplot(x='is_canceled',y='lead_time',data=df)
df['deposit_type'].value_counts()
s = sns.countplot(x='deposit_type', hue='is_canceled', data=df)
s.set_title("Hotels")
plt.show(s)
s = sns.countplot(x='adults', hue='is_canceled', data=df)
s.set_title("Hotels")
plt.show(s)
s = sns.countplot(x='hotel', hue='is_canceled', data=df)
s.set_title("Hotels")
plt.show(s)
s = sns.countplot(x='distribution_channel', hue='is_canceled', data=df)
s.set_title("Hotels")
plt.show(s)
data = df[['hotel','lead_time','stays_in_week_nights','stays_in_weekend_nights','adults','reserved_room_type','adr'
                          ,'is_canceled']]
y = data["is_canceled"]
x = data.drop(labels = ["is_canceled"],axis = 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  cross_val_score,GridSearchCV

model = RandomForestClassifier(random_state=15)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# calculating the classification accuracies
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))