import pandas as pd

df=pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')

df.head()
df.describe()
#checking NA's in dataframe

df.isna().sum()
df=df.drop(['company'],axis=1)



for col in ['country','agent','children']:

    df[col].fillna(df[col].mode()[0],inplace=True)
df.isna().sum()
df.info()
from sklearn import preprocessing 

dff=df.apply(preprocessing.LabelEncoder().fit_transform)

dff
from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor 

from sklearn.ensemble import AdaBoostRegressor
X = dff.drop(['is_canceled'], axis = 1)

y = dff['is_canceled']
dff['is_canceled'].unique()
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
logreg = LogisticRegression(solver = 'lbfgs')

# fit the model with data

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)



print('Mean Absolute Error_logreg:', metrics.mean_absolute_error(y_test, y_pred).round(3))  

print('Mean Squared Error_logreg:', metrics.mean_squared_error(y_test, y_pred).round(3))  

print('Root Mean Squared Error_logreg:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)).round(3))

print('r2_score_logreg:', r2_score(y_test, y_pred).round(3))

import matplotlib.pyplot as plt

con=confusion_matrix(y_test, y_pred)

print(con)

acc=accuracy_score(y_test, y_pred)

print(acc)



import seaborn as sns

ax= plt.subplot()

sns.heatmap(con, annot=True, ax = ax,fmt='g',cmap='gist_rainbow'); 
import pandas as pd  

import numpy as np  

import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline

pre = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

pre

import chart_studio.plotly as py

#import chart_studio.graph_objs as go

from plotly import graph_objs as go

from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)
df['country'].iplot(kind='hist', xTitle='month',

                  yTitle='count', title='Arrivals in a year',colors='Blue')
df.groupby(['arrival_date_month','arrival_date_year'])['children', 'babies','adults'].sum().plot.bar(figsize=(15,5))

plt.figure(figsize=(15,5))

sns.lineplot(x= 'arrival_date_month', y = 'lead_time', data = df)
df_ct=dff['customer_type'].unique()
df['customer_type'].unique()
explode = (0, 0.1, 0, 0)



labels = ['Transient', 'Contract', 'Transient-Party', 'Group']

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

fig1, ax1 = plt.subplots()

ax1.pie(df_ct, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')

plt.tight_layout()

plt.show()
