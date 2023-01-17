import numpy as np

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/online_ads_dummy_data.csv')
data.head()
data.info()
#Converting Timestamp column to 'datetime' Dtype

import datetime

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

#Converting Male column to 'int' Dtype

data['Male'] = data['Male'].apply(lambda x: int(x))

data.info()
data.isnull().sum()
data.head()
def get_hour(x):

    return x.hour



days = dict(enumerate(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']))



def get_day(x):

    return days[x.weekday()]



months = dict(enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']))



def get_month(x):

    return months[x.month-1]
data['Month'] = data['Timestamp'].apply(lambda x:get_month(x))

data['Day'] = data['Timestamp'].apply(lambda x:get_day(x))

data['Hour'] = data['Timestamp'].apply(lambda x:get_hour(x))
data.head(4)
sns.pairplot(data, hue='Clicked on Ad')
data.corr()[data.corr()>0.3]
sns.heatmap(data.corr(),cmap=sns.color_palette("BuGn_r"))
data.groupby('Clicked on Ad').mean()
import plotly.express as px



fig = px.scatter(data, 

                y="Daily Time Spent on Site", 

                x="Daily Internet Usage", 

                title='How daily internet usage affects daily time spent on site.',

                color='Clicked on Ad', 

                trendline='ols'

                )

fig.show()
px.scatter(data, 

              x = 'Area Income',

              y = 'Daily Time Spent on Site',

              color='Clicked on Ad', 

              trendline='ols',

              title = 'How Area income effects the daily amount of time that someone spends on the site.'

             )
px.scatter(data, 

              x = 'Area Income',

              y = 'Daily Internet Usage',

              color='Clicked on Ad', 

              trendline='ols',

              title = 'How Area income effects the daily amount of time that someone spends on the site.'

             )
by_hour = data[['Age','Hour']].groupby('Hour').mean()

sns.barplot(data = by_hour,

            x = by_hour.index,

            y='Age',

            color='green',

           ).set_title('Average age of site visitor per hour')
px.scatter(data, x='Hour',

           y='Age', 

           color='Clicked on Ad',

           size_max=5, 

           trendline='lowess', 

           title = 'Age of visitors to the site during each hour of the day.')
month_dummies = pd.get_dummies(data['Month'],drop_first=True)
day_dummies = pd.get_dummies(data['Day'],drop_first=True)
hour_dummies = pd.get_dummies(data['Hour'],drop_first=True)
age_dummies = pd.get_dummies(data['Age']>30,drop_first=True,prefix='>30')
data.head(3)
categories = pd.concat([month_dummies,day_dummies,hour_dummies,age_dummies, data[['Male','Clicked on Ad']]],axis=1)

categories
from sklearn.model_selection import train_test_split
X = categories.drop('Clicked on Ad',axis=1)

y = data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()

lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
error = lm.predict(X_test)-y_test
false_negative = 0

correct = 1

false_positive = 0



for i in error:

    if i == -1:

        false_negative+=1

    elif i==0:

        correct+=1

    else:

        false_positive+=1

        

print(f'false negative predictions:\t {false_negative} out of {len(error)} predictions.\n'

     +f'\nfalse negative prediction percentage:  {"{:.2f}".format((100*false_negative)/len(error))}%\n')

print(f'\ncorrect predictions:\t {correct} out of {len(error)} predictions.\n'

     +f'\ncorrect prediction percentage:  {"{:.2f}".format((100*correct)/len(error))}%\n')

print(f'\nfalse positive predictions:\t {false_positive} out of {len(error)} predictions.\n'

     +f'\nfalse positive predictions:  {"{:.2f}".format((100*false_positive)/len(error))}%')

      