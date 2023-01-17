

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport





import plotly.express as px    ## for visualization

import matplotlib.pyplot as plt  ## for visualization

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test =  pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
train_profile = ProfileReport(train, title='Pandas Profiling Report', html={'style':{'full_width':True}})

train_profile

train.isna().sum()
# dropped 2 columns

train = train.drop(["Id", 'Province_State'], axis = 1)
train_grp_date = train.groupby(['Date'])

   

train_grp_country = train.groupby(['Country_Region'])
train_grp_date.head()


#plt.bar(train_grp_date.mean(), height = train_grp_date['ConfirmedCases'])



plt.figure(figsize = (25,5))

plt.xticks(rotation = 90)

p1 = plt.bar(train_grp_date.mean().index, train_grp_date.mean()['ConfirmedCases'])

p2 = plt.bar(train_grp_date.mean().index, train_grp_date.mean()['Fatalities'])



plt.legend((p1[0], p2[0]), ("Confirmed_cases", 'Fatalities'))
plt.figure(figsize = (28,5))

#plt.bar(train_grp_country.mean().index, train_grp_country.mean())

p1 = plt.bar(train_grp_country.mean().index, train_grp_country.mean()['ConfirmedCases'])

p2 = plt.bar(train_grp_country.mean().index, train_grp_country.mean()['Fatalities'])

plt.xticks(rotation = 90)

plt.legend((p1[0],p2[0]), ("Confirmed_cases", 'Fatalities'))
train_grp_country_1 = train.groupby(['Country_Region'])['ConfirmedCases','Fatalities'].sum().sort_values('ConfirmedCases',ascending= False).reset_index().head(10)
train_grp_country_1 =train_grp_country_1.set_index(train_grp_country_1['Country_Region'])



train_grp_country_1.drop('Country_Region', axis =1)
plt.figure(figsize = (28,5))

#plt.bar(train_grp_country.mean().index, train_grp_country.mean())

p1 = plt.bar(train_grp_country_1.index, train_grp_country_1['ConfirmedCases'])

p2 = plt.bar(train_grp_country_1.index, train_grp_country_1['Fatalities'])

plt.xticks(rotation = 90)

plt.legend((p1[0],p2[0]), ("Confirmed_cases", 'Fatalities'))
train_grp_country.mean()
top10 = train_grp_country_1.head(10)



fig = px.bar(top10, x=top10.index, y='ConfirmedCases', labels={'x':'Country'},

             color="ConfirmedCases", color_continuous_scale=px.colors.sequential.Brwnyl)

fig.update_layout(title_text='Confirmed COVID-19 cases by country')

fig.show()
df_by_date = train.groupby(['Country_Region','Date'])['ConfirmedCases'].sum().sort_values().reset_index()



def country_conf_plot(country):

    fig = px.bar(df_by_date.loc[(df_by_date['Country_Region'] == country) &(df_by_date.Date >= '2020-03-01')].sort_values('ConfirmedCases',ascending = False), 

             x='Date', y='ConfirmedCases', color="ConfirmedCases", color_continuous_scale=px.colors.sequential.BuGn)

    fig.update_layout(title_text='Confirmed COVID-19 cases per day')

    fig.show()
country_conf_plot('Italy')
country_conf_plot('Spain')

country_conf_plot('Germany')

country_conf_plot('Iran')

country_conf_plot('China')



country_conf_plot('India')
country_conf_plot('US')
top10 = train_grp_country.sum().sort_values('ConfirmedCases', ascending = False).head(10)

train_grp_country.head()
#top10 = train_grp_country_1



plt.figure(figsize = (15,5))

plt.pie(top10['ConfirmedCases'], labels =top10.index , radius = 0.9, frame = True)

#train_grp_country.mean()['ConfirmedCases'])

plt.show()
test = test.drop(['ForecastId','Province_State'], axis = 1)
test.shape
train.head(), train.shape
test.head(), test.shape
# converting the dtypes to datetime format

train["Date"] = pd.to_datetime(train["Date"])

test['Date'] = pd.to_datetime(test['Date'])



train = train.set_index(train['Date'])

test = test.set_index(test['Date'])
def create_features(df,label=None):

    """

    Creates time series features from datetime index.

    """

    #df = df.copy()

    df['Date'] = df.index

    df['hour'] = df['Date'].dt.hour

    df['dayofweek'] = df['Date'].dt.dayofweek

    df['quarter'] = df['Date'].dt.quarter

    df['month'] = df['Date'].dt.month

    df['year'] = df['Date'].dt.year

    df['dayofyear'] = df['Date'].dt.dayofyear

    df['dayofmonth'] = df['Date'].dt.day

    df['weekofyear'] = df['Date'].dt.weekofyear

    

    #X = df[['hour','dayofweek','quarter','month','year','dayofyear','dayofmonth','weekofyear']]

   

    return df
train = create_features(train)

test = create_features(test)
train.head()
train.dtypes
# labelencodinng the columns 



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



def encoding_col(df):

    for col in df.columns:

        if df.dtypes[col] == object:

            df[col] =le.fit_transform(df[col])

            

            #le.fit(df[col].astype(str))

            #df[c] = le.transform(df[c].astype(str))

                      

    return df

    
encoding_col(train)

encoding_col(test)

x_train= train[['Country_Region','month', 'dayofyear', 'dayofmonth' , 'weekofyear']]

y1 = train[['ConfirmedCases']]

y2 =train[['Fatalities']]

x_test = test[['Country_Region', 'month', 'dayofyear', 'dayofmonth' , 'weekofyear']]
x_train.dtypes, x_test.dtypes


x_train = x_train.astype(float)

x_test = x_test.astype(float)
x_train.dtypes, x_test.dtypes
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
xg_reg = XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)





xg_reg.fit(x_train,y1)



ConfirmedCases = xg_reg.predict(x_test)


xg_reg.fit(x_train,y2)



Fatalities = xg_reg.predict(x_test)