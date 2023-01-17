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
import plotly.express as px

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import plotly

plotly.offline.init_notebook_mode() # For not show up chart error



import matplotlib.pyplot as plt

import matplotlib.animation as animation

from IPython.display import HTML

%matplotlib inline



from tqdm import tqdm



def RMSLE(pred,actual):

    return np.sqrt(np.mean(np.power((np.log(pred+1)-np.log(actual+1)),2)))
pd.set_option('mode.chained_assignment', None)

test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

train['Province_State'].fillna('', inplace=True)

test['Province_State'].fillna('', inplace=True)

train['Date'] =  pd.to_datetime(train['Date'])

test['Date'] =  pd.to_datetime(test['Date'])

train = train.sort_values(['Country_Region','Province_State','Date'])

test = test.sort_values(['Country_Region','Province_State','Date'])
# Fix error in train data



train[['ConfirmedCases', 'Fatalities']] = train.groupby(['Country_Region', 'Province_State'])[['ConfirmedCases', 'Fatalities']].transform('cummax') 

# from sklearn.linear_model import LinearRegression, BayesianRidge

# from sklearn.preprocessing import PolynomialFeatures

# from sklearn.pipeline import make_pipeline



# feature_day = [1,20,50,100,200,500,1000]

# def CreateInput(data):

#     feature = []

#     for day in feature_day:

#         #Get information in train data

#         data.loc[:,'Number day from ' + str(day) + ' case'] = 0

#         if (train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].count() > 0):

#             fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        

#         else:

#             fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].min()       

#         for i in range(0, len(data)):

#             if (data['Date'].iloc[i] > fromday):

#                 day_denta = data['Date'].iloc[i] - fromday

#                 data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 

#         feature = feature + ['Number day from ' + str(day) + ' case']

    

#     return data[feature]

# pred_data_all = pd.DataFrame()

# with tqdm(total=len(train['Country_Region'].unique())) as pbar:

#     for country in train['Country_Region'].unique():

#     #for country in ['Japan']:

#         for province in train[(train['Country_Region'] == country)]['Province_State'].unique():

#             df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]

#             df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

#             X_train = CreateInput(df_train)

#             y_train_confirmed = df_train['ConfirmedCases'].ravel()

#             y_train_fatalities = df_train['Fatalities'].ravel()

#             X_pred = CreateInput(df_test)



#             # Define feature to use by X_pred

#             feature_use = X_pred.columns[0]

#             for i in range(X_pred.shape[1] - 1,0,-1):

#                 if (X_pred.iloc[0,i] > 0):

#                     feature_use = X_pred.columns[i]

#                     break

#             idx = X_train[X_train[feature_use] == 0].shape[0]          

#             adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)

#             adjusted_y_train_confirmed = y_train_confirmed[idx:]

#             adjusted_y_train_fatalities = y_train_fatalities[idx:] #.values.reshape(-1, 1)

              

#             adjusted_X_pred = X_pred[feature_use].values.reshape(-1, 1)



#             model = make_pipeline(PolynomialFeatures(2), BayesianRidge())

#             model.fit(adjusted_X_train,adjusted_y_train_confirmed)                

#             y_hat_confirmed = model.predict(adjusted_X_pred)



#             model.fit(adjusted_X_train,adjusted_y_train_fatalities)                

#             y_hat_fatalities = model.predict(adjusted_X_pred)



#             pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

#             pred_data['ConfirmedCases_hat'] = y_hat_confirmed

#             pred_data['Fatalities_hat'] = y_hat_fatalities

#             pred_data_all = pred_data_all.append(pred_data)

#         pbar.update(1)

    

# df_val = pd.merge(pred_data_all,train[['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']],on=['Date','Country_Region','Province_State'], how='left')

# df_val.loc[df_val['Fatalities_hat'] < 0,'Fatalities_hat'] = 0

# df_val.loc[df_val['ConfirmedCases_hat'] < 0,'ConfirmedCases_hat'] = 0



# df_val_1 = df_val.copy()
# RMSLE(df_val[(df_val['ConfirmedCases'].isnull() == False)]['ConfirmedCases'].values,df_val[(df_val['ConfirmedCases'].isnull() == False)]['ConfirmedCases_hat'].values)
# RMSLE(df_val[(df_val['Fatalities'].isnull() == False)]['Fatalities'].values,df_val[(df_val['Fatalities'].isnull() == False)]['Fatalities_hat'].values)
# val_score = []

# for country in df_val['Country_Region'].unique():

#     df_val_country = df_val[(df_val['Country_Region'] == country) & (df_val['Fatalities'].isnull() == False)]

#     val_score.append([country, RMSLE(df_val_country['ConfirmedCases'].values,df_val_country['ConfirmedCases_hat'].values),RMSLE(df_val_country['Fatalities'].values,df_val_country['Fatalities_hat'].values)])

    

# df_val_score = pd.DataFrame(val_score) 

# df_val_score.columns = ['Country','ConfirmedCases_Scored','Fatalities_Scored']

# df_val_score.sort_values('ConfirmedCases_Scored', ascending = False)
# country = "Vietnam"

# df_val = df_val_1

# df_val[df_val['Country_Region'] == country].groupby(['Date','Country_Region']).sum().reset_index()
# country = "Vietnam"

# df_val = df_val_1

# df_country = df_val[df_val['Country_Region'] == country].groupby(['Date','Country_Region']).sum().reset_index()

# df_train = train[(train['Country_Region'].isin(df_country['Country_Region'].unique())) & (train['ConfirmedCases'] > 0)].groupby(['Date']).sum().reset_index()



# idx = df_country[((df_country['ConfirmedCases'].isnull() == False) & (df_country['ConfirmedCases'] > 0))].shape[0]

# fig = px.line(df_country, x="Date", y="ConfirmedCases_hat", title='Forecast Total Cases of ' + df_country['Country_Region'].values[0])

# fig.add_scatter(x=df_train['Date'], y=df_train['ConfirmedCases'], mode='lines', name="Actual train", showlegend=True)

# fig.add_scatter(x=df_country['Date'][0:idx], y=df_country['ConfirmedCases'][0:idx], mode='lines', name="Actual test", showlegend=True)

# fig.show()



# fig = px.line(df_country, x="Date", y="Fatalities_hat", title='Forecast Total Fatalities of ' + df_country['Country_Region'].values[0])

# fig.add_scatter(x=df_train['Date'], y=df_train['Fatalities'], mode='lines', name="Actual train", showlegend=True)

# fig.add_scatter(x=df_country['Date'][0:idx], y=df_country['Fatalities'][0:idx], mode='lines', name="Actual test", showlegend=True)



# fig.show()
# df_total = df_val.groupby(['Date']).sum().reset_index()

# df_train = train[(train['Country_Region'].isin(df_val['Country_Region'].unique())) & (train['ConfirmedCases'] > 0)].groupby(['Date']).sum().reset_index()



# idx = df_total[((df_total['ConfirmedCases'].isnull() == False) & (df_total['ConfirmedCases'] > 0))].shape[0]

# fig = px.line(df_total, x="Date", y="ConfirmedCases_hat", title='Total Cases of World Forecast')

# fig.add_scatter(x=df_train['Date'], y=df_train['ConfirmedCases'], mode='lines', name="Actual train", showlegend=True)

# fig.add_scatter(x=df_total['Date'][0:idx], y=df_total['ConfirmedCases'][0:idx], mode='lines', name="Actual test", showlegend=True)

# fig.show()



# fig = px.line(df_total, x="Date", y="Fatalities_hat", title='Total Fatalities of World Forecast')

# fig.add_scatter(x=df_train['Date'], y=df_train['Fatalities'], mode='lines', name="Actual train", showlegend=True)

# fig.add_scatter(x=df_total['Date'][0:idx], y=df_total['Fatalities'][0:idx], mode='lines', name="Actual test", showlegend=True)

# fig.show()
# df_now = train.groupby(['Date','Country_Region']).sum().sort_values(['Country_Region','Date']).reset_index()

# df_now['New Cases'] = df_now['ConfirmedCases'].diff()

# df_now['New Fatalities'] = df_now['Fatalities'].diff()

# df_now = df_now.groupby('Country_Region').apply(lambda group: group.iloc[-1:]).reset_index(drop = True)



# fig = go.Figure()

# for country in df_now.sort_values('ConfirmedCases', ascending=False).head(5)['Country_Region'].values:

#     df_country = df_val[df_val['Country_Region'] == country].groupby(['Date','Country_Region']).sum().reset_index()

#     idx = df_country[((df_country['ConfirmedCases'].isnull() == False) & (df_country['ConfirmedCases'] > 0))].shape[0]

#     fig.add_trace(go.Scatter(x=df_country['Date'][0:idx],y= df_country['ConfirmedCases'][0:idx], name = country))

#     fig.add_trace(go.Scatter(x=df_country['Date'],y= df_country['ConfirmedCases_hat'], name = country + ' forecast'))

# fig.update_layout(title_text='Top 5 ConfirmedCases forecast')

# fig.show()



# fig = go.Figure()

# for country in df_now.sort_values('Fatalities', ascending=False).head(5)['Country_Region'].values:

#     df_country = df_val[df_val['Country_Region'] == country].groupby(['Date','Country_Region']).sum().reset_index()

#     idx = df_country[((df_country['Fatalities'].isnull() == False) & (df_country['Fatalities'] > 0))].shape[0]

#     fig.add_trace(go.Scatter(x=df_country['Date'][0:idx],y= df_country['Fatalities'][0:idx], name = country))

#     fig.add_trace(go.Scatter(x=df_country['Date'],y= df_country['Fatalities_hat'], name = country + ' forecast'))

# fig.update_layout(title_text='Top 5 Fatalities forecast')

#fig.show()
# df_now = df_now.sort_values('ConfirmedCases', ascending = False)

# fig = make_subplots(rows = 2, cols = 2)

# fig.add_bar(x=df_now['Country_Region'].head(10), y = df_now['ConfirmedCases'].head(10), row=1, col=1, name = 'Total cases')

# df_now = df_now.sort_values('Fatalities', ascending=False)

# fig.add_bar(x=df_now['Country_Region'].head(10), y = df_now['Fatalities'].head(10), row=1, col=2, name = 'Total Fatalities')





# df_now = df_now.sort_values('New Cases', ascending=False)

# fig.add_bar(x=df_now['Country_Region'].head(10), y = df_now['New Cases'].head(10), row=2, col=1, name = 'New Cases')

# df_now = df_now.sort_values('New Fatalities', ascending=False)

# fig.add_bar(x=df_now['Country_Region'].head(10), y = df_now['New Fatalities'].head(10), row=2, col=2, name = 'New Fatalities')



# fig.update_layout({'title_text':'Top 10 Country','legend_orientation':'h','legend_y':1.1,'legend_yanchor':'auto'})
# def update(frame):    

#     fdata = df_country[df_country['Date'] <= frame].groupby('Country_Region').apply(lambda group: group.iloc[-1:])

#     fdata = fdata.sort_values(by = 'ConfirmedCases', ascending = False).head(20)

#     fdata = fdata.sort_values(by = 'ConfirmedCases', ascending = True)

#     xdata = fdata['Country_Region']

#     ydata = fdata['ConfirmedCases']

#     ax.clear()

#     time_unit_displayed = 'Top confirmed case forecast - ' + frame.strftime("%Y-%m-%d")

#     ax.text(0, 1.06, 'History and forecast animation: from ' + frDate.strftime("%Y-%m-%d") + " to " + toDate.strftime("%Y-%m-%d"), transform = ax.transAxes, color = '#666666',

#             size = 12, ha = 'left', va = 'center', weight = 'bold')    

#     ax.text(0, 1.02, time_unit_displayed, transform = ax.transAxes, color = '#666666',

#             size = 10, ha = 'left', va = 'center', weight = 'bold')

#     #colors = list(map(lambda x:mapcolor(x),fdata['Change']))

#     colors = plt.get_cmap('PuRd')(np.linspace(0.15, 0.85, fdata.shape[0]))

#     ax.barh(xdata, ydata, color = colors, tick_label = fdata['Country_Region'])

#     num_of_elements = len(xdata)

#     dx = float(fdata['ConfirmedCases'].max()) / 200

#     for i, (name, value, change) in enumerate(zip(fdata['Country_Region'],fdata['ConfirmedCases'],fdata['Fatalities'])):

#         #ax.text(amount + dx, i, " ({:.2f}%)".format(change), size = 8, ha = 'left', va = 'center', color = '#666666')

#         ax.text(value, i, name, size=8, weight=600, ha='left', va='bottom', color = '#666666')

#         ax.text(value, i-.25, f'{value:,.0f}',  size=8, ha='left',  va='center', color = '#666666')

    

# df_country = train.groupby(['Date','Country_Region']).sum().reset_index()

# df_future = df_val_1[df_val_1['ConfirmedCases'].isnull() == True].groupby(['Date','Country_Region']).sum()

# df_future = df_future.reset_index()[['Date','Country_Region','ConfirmedCases_hat','Fatalities_hat']]

# df_future.columns = ['Date','Country_Region','ConfirmedCases','Fatalities']



# df_country = pd.concat([df_country,df_future], ignore_index=True, sort=True)



# fig, ax = plt.subplots(figsize=(15, 8))

# frDate =  df_country['Date'].min()

# toDate =  df_country['Date'].max()

# animator = animation.FuncAnimation(fig, update, frames=pd.date_range(frDate, toDate).tolist(), interval=500)

# HTML(animator.to_jshtml())
# animator.save('confirm_animation.gif', writer='imagemagick', fps=2)

# from IPython.display import Image, display

# display(Image(url='confirm_animation.gif'))
# import warnings

# from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



# feature_day = [1,20,50,100,200,500,1000]



# def CreateInput(data):

#     feature = []

#     for day in feature_day:

#         #Get information in train data

#         data.loc[:,'Number day from ' + str(day) + ' case'] = 0

#         if (train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].count() > 0):

#             fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        

#         else:

#             fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].min()       

#         for i in range(0, len(data)):

#             if (data['Date'].iloc[i] > fromday):

#                 day_denta = data['Date'].iloc[i] - fromday

#                 data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 

#         feature = feature + ['Number day from ' + str(day) + ' case']

    

#     return data[feature]

# pred_data_all = pd.DataFrame()

# with tqdm(total=len(train['Country_Region'].unique())) as pbar:

#     for country in train['Country_Region'].unique():

#     #for country in ['Vietnam']:

#         for province in train[(train['Country_Region'] == country)]['Province_State'].unique():

#             with warnings.catch_warnings():

#                 warnings.filterwarnings("ignore")

#                 df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]

#                 df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

#                 X_train = CreateInput(df_train)

#                 y_train_confirmed = df_train['ConfirmedCases'].ravel()

#                 y_train_fatalities = df_train['Fatalities'].ravel()

#                 X_pred = CreateInput(df_test)



#                 # Define feature to use by X_pred

#                 feature_use = X_pred.columns[0]

#                 for i in range(X_pred.shape[1] - 1,0,-1):

#                     if (X_pred.iloc[0,i] > 0):

#                         feature_use = X_pred.columns[i]

#                         break

#                 idx = X_train[X_train[feature_use] == 0].shape[0]   



#                 adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)

#                 adjusted_y_train_confirmed = y_train_confirmed[idx:]

#                 adjusted_y_train_fatalities = y_train_fatalities[idx:] #.values.reshape(-1, 1)



#                 pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

#                 max_train_date = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].max()

#                 min_test_date = pred_data['Date'].min()

#                 #The number of day forcast

#                 #pred_data[pred_data['Date'] > max_train_date].shape[0]

#                 #model = SimpleExpSmoothing(adjusted_y_train_confirmed).fit()

#                 #model = Holt(adjusted_y_train_confirmed).fit()

#                 #model = Holt(adjusted_y_train_confirmed, exponential=True).fit()

#                 #model = Holt(adjusted_y_train_confirmed, exponential=True, damped=True).fit()



#                 model = ExponentialSmoothing(adjusted_y_train_confirmed, trend = 'additive').fit()

#                 y_hat_confirmed = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])

#                 y_train_confirmed = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['ConfirmedCases'].values

#                 y_hat_confirmed = np.concatenate((y_train_confirmed,y_hat_confirmed), axis = 0)



#                 #model = Holt(adjusted_y_train_fatalities).fit()



#                 model = ExponentialSmoothing(adjusted_y_train_fatalities, trend = 'additive').fit()

#                 y_hat_fatalities = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])

#                 y_train_fatalities = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['Fatalities'].values

#                 y_hat_fatalities = np.concatenate((y_train_fatalities,y_hat_fatalities), axis = 0)





#                 pred_data['ConfirmedCases_hat'] =  y_hat_confirmed

#                 pred_data['Fatalities_hat'] = y_hat_fatalities

#                 pred_data_all = pred_data_all.append(pred_data)

#         pbar.update(1)



# df_val = pd.merge(pred_data_all,train[['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']],on=['Date','Country_Region','Province_State'], how='left')

# df_val.loc[df_val['Fatalities_hat'] < 0,'Fatalities_hat'] = 0

# df_val.loc[df_val['ConfirmedCases_hat'] < 0,'ConfirmedCases_hat'] = 0

# df_val_2 = df_val.copy()
# country = "Vietnam"

# df_country = df_val[df_val['Country_Region'] == country].groupby(['Date','Country_Region']).sum().reset_index()

# idx = df_country[((df_country['ConfirmedCases'].isnull() == False) & (df_country['ConfirmedCases'] > 0))].shape[0]

# fig = px.line(df_country, x="Date", y="ConfirmedCases_hat", title='Total Cases of ' + df_country['Country_Region'].values[0])

# fig.add_scatter(x=df_country['Date'][0:idx], y=df_country['ConfirmedCases'][0:idx], mode='lines', name="Actual", showlegend=False)

# fig.show()



# fig = px.line(df_country, x="Date", y="Fatalities_hat", title='Total Fatalities of ' + df_country['Country_Region'].values[0])

# fig.add_scatter(x=df_country['Date'][0:idx], y=df_country['Fatalities'][0:idx], mode='lines', name="Actual", showlegend=False)

# fig.show()
# df_total = df_val.groupby(['Date']).sum().reset_index()



# idx = df_total[((df_total['ConfirmedCases'].isnull() == False) & (df_total['ConfirmedCases'] > 0))].shape[0]

# fig = px.line(df_total, x="Date", y="ConfirmedCases_hat", title='Total Cases of World')

# fig.add_scatter(x=df_total['Date'][0:idx], y=df_total['ConfirmedCases'][0:idx], mode='lines', name="Actual", showlegend=False)

# fig.show()



# fig = px.line(df_total, x="Date", y="Fatalities_hat", title='Total Fatalities of World')

# fig.add_scatter(x=df_total['Date'][0:idx], y=df_total['Fatalities'][0:idx], mode='lines', name="Actual", showlegend=False)

# fig.show()
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.arima_model import ARIMA



feature_day = [1,20,50,100,200,500,1000]

def CreateInput(data):

    feature = []

    for day in feature_day:

        #Get information in train data

        data.loc[:,'Number day from ' + str(day) + ' case'] = 0

        if (train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].count() > 0):

            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['ConfirmedCases'] < day)]['Date'].max()        

        else:

            fromday = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].min()       

        for i in range(0, len(data)):

            if (data['Date'].iloc[i] > fromday):

                day_denta = data['Date'].iloc[i] - fromday

                data['Number day from ' + str(day) + ' case'].iloc[i] = day_denta.days 

        feature = feature + ['Number day from ' + str(day) + ' case']

    

    return data[feature]

pred_data_all = pd.DataFrame()

with tqdm(total=len(train['Country_Region'].unique())) as pbar:

    for country in train['Country_Region'].unique():

    #for country in ['Vietnam']:

        for province in train[(train['Country_Region'] == country)]['Province_State'].unique():

            with warnings.catch_warnings():

                warnings.filterwarnings("ignore")

                df_train = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]

                df_test = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

                X_train = CreateInput(df_train)

                y_train_confirmed = df_train['ConfirmedCases'].ravel()

                y_train_fatalities = df_train['Fatalities'].ravel()

                X_pred = CreateInput(df_test)



                feature_use = X_pred.columns[0]

                for i in range(X_pred.shape[1] - 1,0,-1):

                    if (X_pred.iloc[0,i] > 0):

                        feature_use = X_pred.columns[i]

                        break

                idx = X_train[X_train[feature_use] == 0].shape[0] 



                adjusted_X_train = X_train[idx:][feature_use].values.reshape(-1, 1)

                adjusted_y_train_confirmed = y_train_confirmed[idx:]

                adjusted_y_train_fatalities = y_train_fatalities[idx:] #.values.reshape(-1, 1)

                idx = X_pred[X_pred[feature_use] == 0].shape[0]    

                adjusted_X_pred = X_pred[idx:][feature_use].values.reshape(-1, 1)



                pred_data = test[(test['Country_Region'] == country) & (test['Province_State'] == province)]

                max_train_date = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]['Date'].max()

                min_test_date = pred_data['Date'].min()

                model = SARIMAX(adjusted_y_train_confirmed, order=(1,1,0), 

                                #seasonal_order=(1,1,0,12),

                                measurement_error=True).fit(disp=False)

                y_hat_confirmed = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])

                y_train_confirmed = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['ConfirmedCases'].values

                y_hat_confirmed = np.concatenate((y_train_confirmed,y_hat_confirmed), axis = 0)

                model = SARIMAX(adjusted_y_train_fatalities, order=(1,1,0), 

                                #seasonal_order=(1,1,0,12),

                                measurement_error=True).fit(disp=False)

                y_hat_fatalities = model.forecast(pred_data[pred_data['Date'] > max_train_date].shape[0])

                y_train_fatalities = train[(train['Country_Region'] == country) & (train['Province_State'] == province) & (train['Date'] >=  min_test_date)]['Fatalities'].values

                y_hat_fatalities = np.concatenate((y_train_fatalities,y_hat_fatalities), axis = 0)

                pred_data['ConfirmedCases_hat'] =  y_hat_confirmed

                pred_data['Fatalities_hat'] = y_hat_fatalities

                pred_data_all = pred_data_all.append(pred_data)

        pbar.update(1)

df_val = pd.merge(pred_data_all,train[['Date','Country_Region','Province_State','ConfirmedCases','Fatalities']],on=['Date','Country_Region','Province_State'], how='left')

df_val.loc[df_val['Fatalities_hat'] < 0,'Fatalities_hat'] = 0

df_val.loc[df_val['ConfirmedCases_hat'] < 0,'ConfirmedCases_hat'] = 0

df_val_3 = df_val.copy()
# country = "Vietnam"

# df_country = df_val[df_val['Country_Region'] == country].groupby(['Date','Country_Region']).sum().reset_index()

# idx = df_country[((df_country['ConfirmedCases'].isnull() == False) & (df_country['ConfirmedCases'] > 0))].shape[0]

# fig = px.line(df_country, x="Date", y="ConfirmedCases_hat", title='Total Cases of ' + df_country['Country_Region'].values[0] + ' (SARIMA)')

# fig.add_scatter(x=df_country['Date'][0:idx], y=df_country['ConfirmedCases'][0:idx], mode='lines', name="Actual", showlegend=False)

# fig.show()



# fig = px.line(df_country, x="Date", y="Fatalities_hat", title='Total Fatalities of ' + df_country['Country_Region'].values[0] + ' (SARIMA)')

# fig.add_scatter(x=df_country['Date'][0:idx], y=df_country['Fatalities'][0:idx], mode='lines', name="Actual", showlegend=False)

# fig.show()
# df_total = df_val.groupby(['Date']).sum().reset_index()



# idx = df_total[((df_total['ConfirmedCases'].isnull() == False) & (df_total['ConfirmedCases'] > 0))].shape[0]

# fig = px.line(df_total, x="Date", y="ConfirmedCases_hat", title='Total Cases of World - SARIMA')

# fig.add_scatter(x=df_total['Date'][0:idx], y=df_total['ConfirmedCases'][0:idx], mode='lines', name="Actual", showlegend=False)

# fig.show()



# fig = px.line(df_total, x="Date", y="Fatalities_hat", title='Total Fatalities of World - SARIMA')

# fig.add_scatter(x=df_total['Date'][0:idx], y=df_total['Fatalities'][0:idx], mode='lines', name="Actual", showlegend=False)

# fig.show()
#[df_val_1.shape,df_val_2.shape,df_val_3.shape]
# method_list = ['Poly Bayesian Ridge','Exponential Smoothing','SARIMA']

# method_val = [df_val_1,df_val_2,df_val_3]

# for i in range(0,3):

#     df_val = method_val[i]

#     method_score = [method_list[i]] + [RMSLE(df_val[(df_val['ConfirmedCases'].isnull() == False)]['ConfirmedCases'].values,df_val[(df_val['ConfirmedCases'].isnull() == False)]['ConfirmedCases_hat'].values)] + [RMSLE(df_val[(df_val['Fatalities'].isnull() == False)]['Fatalities'].values,df_val[(df_val['Fatalities'].isnull() == False)]['Fatalities_hat'].values)]

#     print (method_score)
df_val = df_val_3

submission = df_val[['ForecastId','ConfirmedCases_hat','Fatalities_hat']]

submission.columns = ['ForecastId','ConfirmedCases','Fatalities']

submission = submission.round({'ConfirmedCases': 0, 'Fatalities': 0})

submission.to_csv('submission.csv', index=False)

submission
# import requests

# from bs4 import BeautifulSoup



# req = requests.get('https://www.worldometers.info/coronavirus/')

# soup = BeautifulSoup(req.text, "lxml")



# df_country = soup.find('div',attrs={"id" : "nav-tabContent"}).find('table',attrs={"id" : "main_table_countries_today"}).find_all('tr')

# arrCountry = []

# for i in range(1,len(df_country)-1):

#     tmp = df_country[i].find_all('td')

#     if (tmp[0].string.find('<a') == -1):

#         country = [tmp[0].string]

#     else:

#         country = [tmp[0].a.string] # Country

#     for j in range(1,7):

#         if (str(tmp[j].string) == 'None' or str(tmp[j].string) == ' '):

#             country = country + [0]

#         else:

#             country = country + [float(tmp[j].string.replace(',','').replace('+',''))]

#     arrCountry.append(country)

# df_worldinfor = pd.DataFrame(arrCountry)

# df_worldinfor.columns = ['Country','Total Cases','Cases','Total Deaths','Deaths','Total Recovers','Active Case']

# for i in range(0,len(df_worldinfor)):

#     df_worldinfor['Country'].iloc[i] = df_worldinfor['Country'].iloc[i].strip()
# fig = px.bar(df_worldinfor.sort_values('Total Cases', ascending=False)[:10][::-1], 

#              x='Total Cases', y='Country',

#              title='Total Cases Worldwide', text='Total Cases', orientation='h')

# fig.show()



# fig = px.bar(df_worldinfor.sort_values('Cases', ascending=False)[:10][::-1], 

#              x='Cases', y='Country',

#              title='New Cases Worldwide', text='Cases', orientation='h')

# fig.show()



# fig = px.bar(df_worldinfor.sort_values('Active Case', ascending=False)[:10][::-1], 

#              x='Active Case', y='Country',

#              title='Active Cases Worldwide', text='Active Case', orientation='h')

# fig.show()
#df_worldinfor[df_worldinfor['Country'] == 'Vietnam']