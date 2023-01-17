# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
train.describe()

test.describe()
import matplotlib.pyplot as plt

confirmed_total_date = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date = train.groupby(['Date']).agg({'Fatalities':['sum']})

total_date = confirmed_total_date.join(fatalities_total_date)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date.plot(ax=ax1)

ax1.set_title("Global confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date.plot(ax=ax2, color='red')

ax2.set_title("Global deceased cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
confirmed_total_date_noChina = train[train['Country/Region']!='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_noChina = train[train['Country/Region']!='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_noChina = confirmed_total_date_noChina.join(fatalities_total_date_noChina)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_noChina.plot(ax=ax1)

ax1.set_title("Global confirmed cases excluding China", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_noChina.plot(ax=ax2, color='red')

ax2.set_title("Global deceased cases excluding China", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
confirmed_total_date_China = train[train['Country/Region']=='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_China = train[train['Country/Region']=='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_China = confirmed_total_date_China.join(fatalities_total_date_China)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_China.plot(ax=ax1)

ax1.set_title("Confirmed cases in China", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_China.plot(ax=ax2, color='red')

ax2.set_title("Deceased cases in China", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
import plotly.express as px 
tcc=train.groupby(["Country/Region","Date"]).sum().reset_index()

fig=px.choropleth(tcc,locations="Country/Region",color="ConfirmedCases",hover_name="Country/Region",\

                 locationmode="country names")

fig.update_layout(title={'text':"Confirmed cases Country-wise",\

                         'x':0.475,'y':0.9,'xanchor':'center','yanchor':'top'})



fig.show()
formated_gdf = train.groupby(['Date', 'Country/Region'])['ConfirmedCases', 'Fatalities'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['ConfirmedCases'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 

                     color="ConfirmedCases", size='size', hover_name="Country/Region", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="Date", 

                     title='Spread Over Time of COVID-19', color_continuous_scale="portland")

fig.update(layout_coloraxis_showscale=True)

fig.show()
tff=train.groupby(["Country/Region","Date"]).sum().reset_index()

fig=px.choropleth(tff,locations="Country/Region",color="Fatalities",hover_name="Country/Region",\

                 locationmode="country names")

fig.update_layout(title={'text':"Country-wise Fatalities",\

                         'x':0.475,'y':0.9,'xanchor':'center','yanchor':'top'})



fig.show()
formated_gdf = train.groupby(['Date', 'Country/Region'])['ConfirmedCases', 'Fatalities'].max()

formated_gdf = formated_gdf.reset_index()

formated_gdf['Date'] = pd.to_datetime(formated_gdf['Date'])

formated_gdf['Date'] = formated_gdf['Date'].dt.strftime('%m/%d/%Y')

formated_gdf['size'] = formated_gdf['Fatalities'].pow(0.3)



fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 

                     color="Fatalities", size='size', hover_name="Country/Region", 

                     range_color= [0, 1500], 

                     projection="natural earth", animation_frame="Date", 

                     title='Deaths Over Time due to COVID-19', color_continuous_scale="reds")

fig.update(layout_coloraxis_showscale=True)

fig.show()
train.isnull().sum()
test.isnull().sum()
# Handling missing Province/State

train[['Province/State']] = train[['Province/State']].fillna('')

test[['Province/State']] = test[['Province/State']].fillna('')



def impute_missing_province_state(data):

    if data[1] == '':

        data[1] = data[2]

    return data



train = train.apply(impute_missing_province_state, axis = 1)

test = test.apply(impute_missing_province_state, axis = 1)



# print(train)
# Delete Useless Column

del(train['Id'])
# LabelEncoder for Country/Region & Province/State

from sklearn.preprocessing import LabelEncoder

lb=LabelEncoder()

lb_transformer = lb.fit(train['Country/Region'])

train['Country/Region'] = lb_transformer.transform(train['Country/Region'])

test['Country/Region'] = lb_transformer.transform(test['Country/Region'])



lb_transformer = lb.fit(train['Province/State'])

train['Province/State'] = lb_transformer.transform(train['Province/State'])

test['Province/State'] = lb_transformer.transform(test['Province/State'])



train, test
# Convert date to string

from datetime import datetime

train["Date"] = train["Date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))

#train["Date"] = train["Date"].apply(lambda x: x.timestamp())

#train["Date"]  = train["Date"].astype(int)
# Splitting datas.

splitting = "2020-03-18"

Train = train[train['Date'] < splitting]

validation = train[train['Date'] >= splitting]
train_df = pd.DataFrame(Train)

validation_df = pd.DataFrame(validation)



train_df["Date"] = train_df["Date"].apply(lambda x: x.timestamp())

train_df["Date"]  = train_df["Date"].astype(int)





validation_df["Date"] = validation_df["Date"].apply(lambda x: x.timestamp())

validation_df["Date"]  = validation_df["Date"].astype(int)
train_df.info(), validation_df.info()
Xtr = train_df[['Province/State','Country/Region','Lat','Long','Date']].to_numpy()

Ytr = train_df['ConfirmedCases'].to_numpy()

Ztr = train_df['Fatalities'].to_numpy()



Xval = validation_df[['Province/State', 'Country/Region','Lat','Long','Date']].to_numpy()

Yval = validation_df['ConfirmedCases'].to_numpy()

Zval = validation_df['Fatalities'].to_numpy()
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(

    bootstrap = True, 

    max_features = 'auto', 

    n_estimators = 150, 

    random_state = None, 

    n_jobs = 1, 

    verbose = 0, 

    max_depth = None, 

    max_leaf_nodes = None)
# Confirmed Cases

model.fit(Xtr,Ytr)

pred1 = model.predict(Xval)

pred1 = pd.DataFrame(pred1)

pred1.columns = ["ConfirmedCases_prediction"]



pred1
type(pred1)
pred1["ConfirmedCases_prediction"] = pred1["ConfirmedCases_prediction"].apply(lambda x : int(x) if x > 0 else 0 ) 
pred1, Yval
# Fatalities

Xtr_fatality = train_df[['Province/State', 'Country/Region','Lat','Long','Date', 'ConfirmedCases']].to_numpy()

# Ytr = train_df['ConfirmedCases'].to_numpy()

Ztr_fatality = train_df['Fatalities'].to_numpy()



Xval_fatality = train_df[['Province/State', 'Country/Region','Lat','Long','Date', 'ConfirmedCases']].to_numpy()

# Ytr = train_df['ConfirmedCases'].to_numpy()

Zval_fatality = train_df['Fatalities'].to_numpy()
model_fatality2 = RandomForestRegressor(

    bootstrap = True, 

    max_features = 'auto', 

    n_estimators = 150, 

    random_state = None, 

    n_jobs = 1, 

    verbose = 0, 

    max_depth = None, 

    max_leaf_nodes = None)
model_fatality2.fit(Xtr_fatality, Ztr_fatality)

pred2 = model_fatality2.predict(Xval_fatality)

pred2 = pd.DataFrame(pred2)

pred2.columns = ["Fatalities_prediction"]



pred2["Fatalities_prediction"] = pred2["Fatalities_prediction"].apply(lambda x : int(x) if x > 0 else 0 ) 



pred2, Zval_fatality
test["Date"] = test["Date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))

test["Date"] = test["Date"].apply(lambda x: x.timestamp())

test["Date"]  = test["Date"].astype(int)
test_dataset = test[['Province/State','Country/Region','Lat','Long','Date']].to_numpy()
test_pred = model.predict(test_dataset)
test_pred = pd.DataFrame(test_pred)

test_pred.columns = ['ConfirmedCases']

test_pred["ConfirmedCases"] = test_pred["ConfirmedCases"].apply(lambda x : int(x) if x > 0 else 0 ) 

test_pred
test['ConfirmedCases'] = test_pred['ConfirmedCases']

test_dataset_fatality = test[['Province/State','Country/Region','Lat','Long','Date', 'ConfirmedCases']].to_numpy()
test_pred_ft = pd.DataFrame(model_fatality2.predict(test_dataset_fatality))

test_pred_ft.columns = ['Fatalities']

test_pred_ft
sub = pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')
for i in range(12212):

    sub.iloc[i,1] = test_pred.iloc[i,0]

    sub.iloc[i,2] = test_pred_ft.iloc[i,0] 



sub['ConfirmedCases'] = test_pred['ConfirmedCases']

sub['Fatalities'] = test_pred_ft['Fatalities'].astype(int)



sub
sub.to_csv('submission.csv',index = False)