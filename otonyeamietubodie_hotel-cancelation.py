import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly_express as px

import matplotlib.image as mpimg

from tabulate import tabulate

import missingno as msno 

from IPython.display import display_html

from PIL import Image

import gc

import cv2

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')
data.head(5)
data.describe(include='all')
data.info()
def missing_values(df):

    mv = df.isnull().sum().sort_values(ascending=False)

    percentage = round(df.isnull().sum().sort_values(ascending=False) / len(df) * 100, 2)

    return pd.concat([mv, percentage], axis=1, keys=['Total', 'Percentage'])
missing_values(data)
data = data.drop(['company'], axis=1)
data = data.drop(['agent'], axis=1)
def percentage_count(df, feature):

    percentage = pd.DataFrame(round(df.loc[:, feature].value_counts(dropna=False, normalize=True)* 100, 2))

    total = pd.DataFrame(df.loc[:, feature].value_counts(dropna=False))

    total.columns = ['Total']

    total.columns = ['Percentage']

    return pd.concat([total, percentage], axis=1)
percentage_count(data, 'country')
data[data['country'].isnull()]
nan_replacements = {"children:": 0.0,"country": "Unknown"}

data = data.fillna(nan_replacements)
data[data['children'].isnull()]
zero_guests = list(data.loc[data["adults"]

                   + data["children"]

                   + data["babies"]==0].index)

data.drop(data.index[zero_guests], inplace=True)
f, ax = plt.subplots(figsize=(12, 5))

sns.countplot('is_canceled', data=data)
dataset = data.copy()

cancelled = data[data.is_canceled==1]

not_cancelled = data[data.is_canceled==0]
f, ax = plt.subplots(figsize=(12, 5))

sns.countplot('customer_type', data=data)
from plotly.offline import init_notebook_mode,iplot

import plotly.graph_objects as go

from plotly.subplots import make_subplots



trace1 = go.Histogram(

    x=cancelled.customer_type,

    opacity=0.75,

    name='cancelled')



trace2 = go.Histogram(

    x=not_cancelled.customer_type,

    opacity=0.75,

    name='did not cancelled')



data = [trace1, trace2]

layout = go.Layout(barmode='stack',

                   title='Cancelled and did not according to customer type ',

                   xaxis=dict(title='Gender'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
f, ax = plt.subplots(figsize=(12, 5))

sns.countplot('hotel', data=dataset)
from plotly.offline import init_notebook_mode,iplot

import plotly.graph_objects as go

from plotly.subplots import make_subplots



trace1 = go.Histogram(

    x=cancelled.hotel,

    opacity=0.75,

    name='cancelled')



trace2 = go.Histogram(

    x=not_cancelled.hotel,

    opacity=0.75,

    name='did not cancelled')



data = [trace1, trace2]

layout = go.Layout(barmode='stack',

                   title='Cancelled and did not according to hotel',

                   xaxis=dict(title='Hotel'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
datas = dataset.copy()

fig = plt.figure(figsize=(15, 8))

ax=sns.kdeplot(datas.loc[(datas['is_canceled'] == 0),'adr'] , color='red',shade=True,label='did not cancel')

ax=sns.kdeplot(datas.loc[(datas['is_canceled'] == 1),'adr'] , color='green',shade=True, label='cancelled')

plt.title('price Distribution cancelled vs Non cancelled', fontsize = 25, pad = 40)

plt.ylabel("Frequency of customers cancelled", fontsize = 15, labelpad = 20)

plt.xlabel("Adr", fontsize = 15, labelpad = 20)
# order by month:

ordered_months = ["January", "February", "March", "April", "May", "June", 

          "July", "August", "September", "October", "November", "December"]

datas["arrival_date_month"] = pd.Categorical(datas["arrival_date_month"], categories=ordered_months, ordered=True)



plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="adr", hue="hotel", data=datas, 

            hue_order = ["City Hotel", "Resort Hotel"], ci="sd", size="hotel", sizes=(2.5, 2.5))

plt.title("Room price per night and person over the year", fontsize=16)

plt.xlabel("Month", fontsize=16)

plt.xticks(rotation=45)

plt.ylabel("Price [EUR]", fontsize=16)

plt.show()
datas[datas.adr > 500]
f, ax = plt.subplots(figsize=(12, 5))

sns.countplot('assigned_room_type', data=dataset)
plt.subplots(figsize = (15,8))

ax = sns.barplot(x = "assigned_room_type", 

                 y = "adr", 

                 data=datas, 

                 linewidth=5



                )



plt.ylabel("prices", fontsize = 15, )

plt.xlabel("assigned_room_type",fontsize = 15);
# order by month:

ordered_months = ["January", "February", "March", "April", "May", "June", 

          "July", "August", "September", "October", "November", "December"]

datas["arrival_date_month"] = pd.Categorical(datas["arrival_date_month"], categories=ordered_months, ordered=True)



plt.figure(figsize=(12, 8))

sns.lineplot(x = "arrival_date_month", y="adr", hue="assigned_room_type", data=datas, 

            hue_order = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"], ci="sd", size="assigned_room_type", sizes=(2.5, 2.5))

plt.title("Assigned room type price per night and person over the year", fontsize=16)

plt.xlabel("Month", fontsize=16)

plt.xticks(rotation=45)

plt.ylabel("Price [EUR]", fontsize=16)

plt.show()
#Below is a heatmap of the correlation of the normal data:

correlation_matrix = dataset.corr()

fig = plt.figure(figsize=(20,8))

sns.heatmap(correlation_matrix, vmax=0.8, square=True)
correlation_matrix['is_canceled'].sort_values(ascending=False)
trace1 = go.Histogram(

    x=cancelled.deposit_type,

    opacity=0.75,

    name='cancelled')



trace2 = go.Histogram(

    x=not_cancelled.deposit_type,

    opacity=0.75,

    name='did not cancelled')





data = [trace1, trace2]

layout = go.Layout(barmode='stack',

                   title='Cancelled and did not according to deposit type',

                   xaxis=dict(title='Deposite type'),

                   yaxis=dict( title='Count'),

                   paper_bgcolor='beige',

                   plot_bgcolor='beige'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
datas['hotel'] = datas['hotel'].map({'Resort Hotel':0, 'City Hotel':1})

datas['hotel'].unique()
datas =datas.drop(['children'], axis=1)
datas['arrival_date_month'].unique()
datas['arrival_date_month'] = datas['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,

                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
datas['arrival_date_month'].unique()
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
datas['customer_type']= label_encoder.fit_transform(datas['customer_type']) 

datas['assigned_room_type'] = label_encoder.fit_transform(datas['assigned_room_type'])

datas['deposit_type'] = label_encoder.fit_transform(datas['deposit_type'])

datas['reservation_status'] = label_encoder.fit_transform(datas['reservation_status'])

datas['meal'] = label_encoder.fit_transform(datas['meal'])

datas['country'] = label_encoder.fit_transform(datas['country'])

datas['distribution_channel'] = label_encoder.fit_transform(datas['distribution_channel'])

datas['market_segment'] = label_encoder.fit_transform(datas['market_segment'])

datas['reserved_room_type'] = label_encoder.fit_transform(datas['reserved_room_type'])

datas['reservation_status_date'] = label_encoder.fit_transform(datas['reservation_status_date'])
print('customer_type:', datas['customer_type'].unique())

print('reservation_status', datas['reservation_status'].unique())

print('deposit_type', datas['deposit_type'].unique())

print('assigned_room_type', datas['assigned_room_type'].unique())

print('meal', datas['meal'].unique())

print('Country:',datas['country'].unique())

print('Dist_Channel:',datas['distribution_channel'].unique())

print('Market_seg:', datas['market_segment'].unique())

print('reserved_room_type:', datas['reserved_room_type'].unique())
y = datas['is_canceled']

X = datas.drop(['is_canceled'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
training_score = cross_val_score(classifier, X_train, y_train, cv=10)
training_score
training_score.mean()
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}







grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)

# We automatically get the logistic regression with the best parameters.

log_reg = grid_log_reg.best_estimator_
log_reg
log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=10)
log_reg_score
y_pred = log_reg.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_test)

cm
from sklearn.metrics import classification_report
print('Logistic Regression:')

print(classification_report(y_test, y_pred))