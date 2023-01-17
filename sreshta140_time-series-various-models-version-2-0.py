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
train = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/train.csv')
meal = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/meal_info.csv')
center = pd.read_csv('/kaggle/input/av-genpact-hack-dec2018/fulfilment_center_info.csv')
meal
train.head()
data = train.merge(meal, on='meal_id')
data
data = data.merge(center, on='center_id')
train_data = data
df=data.copy()
data.nunique()
corr = data.corr()
import seaborn as sns
sns.heatmap(corr)
ts_tot_orders = data.groupby(['week'])['num_orders'].sum()
ts_tot_orders = pd.DataFrame(ts_tot_orders)
ts_tot_orders
import plotly.graph_objs as go
import plotly.offline as pyoff
plot_data = [
    go.Scatter(
        x=ts_tot_orders.index,
        y=ts_tot_orders['num_orders'],
        name='Time Series for num_orders',
        marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",http://localhost:8888/notebooks/Kaggle_for_timepass/hackathon/Sigma-thon-master/Sigma-thon-master/eda1.ipynb#
    )
]
plot_layout = go.Layout(
        title='Total orders per week',
        yaxis_title='Total orders',
        xaxis_title='Week',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
center_id = data.groupby(['center_id'])['num_orders'].sum()
center_id = pd.DataFrame(center_id)
center_id=center_id.reset_index()
import plotly.express as px
fig = px.bar(center_id, x="center_id", y="num_orders", color='center_id')
fig.update_layout({
'plot_bgcolor': 'rgba(1, 1, 1, 1)',
'paper_bgcolor': 'rgba(1, 1, 1, 1)',
})

fig.show()
meal_id = df.groupby(['category', 'cuisine'])['num_orders'].sum()
meal_id = pd.DataFrame(meal_id)
meal_id=meal_id.reset_index()
meal_id
meal_id['meal'] = meal_id.apply(lambda x : x['category']+', '+x['cuisine'],axis=1)
meal_id
import plotly.express as px
fig = px.bar(meal_id, x="meal", y="num_orders", color='meal')
fig.update_layout({
'plot_bgcolor': 'rgba(1, 1, 1, 1)',
'paper_bgcolor': 'rgba(1, 1, 1, 1)',
})

fig.show()
cat_var = ['center_type',
 'category',
 'cuisine']
import plotly.graph_objs as go
import plotly.offline as pyoff
for i in cat_var:
    grp=df.groupby([i])
    grp=pd.DataFrame(grp)
    lis=grp[0]
    x=0
    plot_data=[]
    for j in lis:
        print(i)
        print(j)
        data = df[df[i]==j]
        data = pd.DataFrame(data)
        tot_orders = data.groupby(['week'])['num_orders'].sum()
        tot_orders = pd.DataFrame(tot_orders)
       
        plot_data.append(go.Scatter(
                x=tot_orders.index,
                y=tot_orders['num_orders'],
                name=str(j),
                #marker = dict(color = colors[x%12])
                #x_axis="OTI",
                #y_axis="time",
            ))
        
        x+=1
    plot_layout = go.Layout(
            title='Total orders per week for '+str(i),
            yaxis_title='Total orders',
            xaxis_title='Week',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)
center_type = df.groupby(['center_type'])['num_orders'].sum()
center_type = pd.DataFrame(center_type)
center_type
center_type=center_type.reset_index()
import plotly.express as px
fig = px.bar(center_type, x="center_type", y="num_orders", color='center_type')
fig.update_layout({
'plot_bgcolor': 'rgba(1, 1, 1, 1)',
'paper_bgcolor': 'rgba(1, 1, 1, 1)',
})

fig.show()
category = df.groupby(['category'])['num_orders'].sum()
category = pd.DataFrame(category)
category = category.reset_index()
import plotly.express as px
fig = px.bar(category, x="category", y="num_orders", color='category')
fig.update_layout({
'plot_bgcolor': 'rgba(1, 1, 1, 1)',
'paper_bgcolor': 'rgba(1, 1, 1, 1)',
})
fig.show()
cuisine = df.groupby(['cuisine'])['num_orders'].sum()
cuisine = pd.DataFrame(cuisine)
cuisine = cuisine.reset_index()
import plotly.express as px
fig = px.bar(cuisine, x="cuisine", y="num_orders", color='cuisine')
fig.update_layout({
'plot_bgcolor': 'rgba(1, 1, 1, 1)',
'paper_bgcolor': 'rgba(1, 1, 1, 1)',
})
fig.show()
cat_ct=df.groupby(['category', 'center_type'])['num_orders'].sum()
cat_ct = cat_ct.unstack().fillna(0)
cat_ct
# Visualize this data in bar plot
import matplotlib.pyplot as plt
ax = (cat_ct).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
cat_cu=df.groupby(['category', 'cuisine'])['num_orders'].sum()
cat_cu = cat_cu.unstack().fillna(0)
cat_cu
# Visualize this data in bar plot
ax = (cat_cu).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
ct_cu=df.groupby(['center_type', 'cuisine'])['num_orders'].sum()
ct_cu = ct_cu.unstack().fillna(0)
ct_cu
# Visualize this data in bar plot
ax = (ct_cu).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
center_id = 55
meal_id = 1993

# here we are putting user input
# here the prediction related to that center id and that particular meal id.
train_df = train_data[train_data['center_id']==center_id]
train_df = train_df[train_df['meal_id']==meal_id]
period = len(train_df)
train_df['Date'] = pd.date_range('2015-01-01', periods=period, freq='W')
train_df['Day'] = train_df['Date'].dt.day
train_df['Month'] = train_df['Date'].dt.month
train_df['Year'] = train_df['Date'].dt.year
train_df['Quarter'] = train_df['Date'].dt.quarter
train_df.head()
colors=['#b84949', '#ff6f00', '#ffbb00', '#9dff00', '#329906', '#439c55', '#67c79e', '#00a1db', '#002254', '#5313c2', '#c40fdb', '#e354aa']
import plotly.graph_objs as go
import plotly.offline as pyoff
for i in cat_var:
    grp=train_df.groupby([i])
    grp=pd.DataFrame(grp)
    lis=grp[0]
    x=0
    for j in lis:
        print(i)
        print(j)
        data = train_df[train_df[i]==j]
        data = pd.DataFrame(data)
        tot_orders = data.groupby(['week'])['num_orders'].sum()
        tot_orders = pd.DataFrame(tot_orders)
        plot_data = [
            go.Scatter(
                x=tot_orders.index,
                y=tot_orders['num_orders'],
                name='Time Series for num_orders for '+str(j),
                marker = dict(color = colors[x%12])
                #x_axis="OTI",
                #y_axis="time",
            )
        ]
        plot_layout = go.Layout(
                title='Total orders per week for '+str(j),
                yaxis_title='Total orders',
                xaxis_title='Week',
                plot_bgcolor='rgba(0,0,0,0)'
            )
        fig = go.Figure(data=plot_data, layout=plot_layout)
        x+=1
        pyoff.iplot(fig)
xb_data = train_df.drop(columns=['id','center_id','meal_id','category','cuisine','center_type'])

xb_data = xb_data.set_index(['Date'])
x_train = xb_data.drop(columns='num_orders')
y_train = xb_data['num_orders']
y_train = np.log1p(y_train)
split_size = period-15
X_train = x_train.iloc[:split_size,:]
X_test = x_train.iloc[split_size:,:]
Y_train =  y_train.iloc[:split_size]
Y_test = y_train.iloc[split_size:]
import matplotlib.pyplot as plt
plt.figure(figsize=(20,5))
plt.plot(Y_train, label='training_data')
plt.plot(Y_test, label='Validation_test')
plt.legend(loc='best')
from xgboost import XGBRegressor
model_2 = XGBRegressor(
 learning_rate = 0.01,
 eval_metric ='rmse',
    n_estimators = 50000,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 1,
    gamma = 0.5
  
  
 )
#model.fit(X_train, y_train)
model_2.fit(X_train, Y_train, eval_metric='rmse', 
          eval_set=[(X_test, Y_test)], early_stopping_rounds=500, verbose=100)
a = (model_2.get_booster().best_iteration)
a
xgb_model = XGBRegressor(
     
     learning_rate = 0.01,
   
    n_estimators = a,
    max_depth = 5,
    subsample = 0.8,
    colsample_bytree = 1,
    gamma = 0.5)
xgb_model.fit(X_train, Y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_preds = np.exp(xgb_preds)
train_df.tail()
xgb_preds = pd.DataFrame(xgb_preds)
xgb_preds.index = Y_test.index
Y_train = np.exp(Y_train)
Y_test = np.exp(Y_test)
plt.figure(figsize=(20,5))
plt.plot(Y_train, label='training_data')
plt.plot(Y_test, label='Validation_test')
plt.plot(xgb_preds, color='cyan', label='xgb_preds')
plt.legend(loc='best')
from lightgbm import LGBMRegressor
lgb_fit_params={"early_stopping_rounds":500, 
            "eval_metric" : 'rmse', 
            "eval_set" : [(X_test,Y_test)],
            'eval_names': ['valid'],
            'verbose':100
           }

lgb_params = {'boosting_type': 'gbdt',
 'objective': 'regression',
 'metric': 'rmse',
 'verbose': 0,
 'bagging_fraction': 0.8,
 'bagging_freq': 1,
 'lambda_l1': 0.01,
 'lambda_l2': 0.01,
 'learning_rate': 0.001,
 'max_bin': 255,
 'max_depth': 6,
 'min_data_in_bin': 1,
 'min_data_in_leaf': 1,
 'num_leaves': 31}

Y_train = np.log1p(Y_train)
Y_test = np.log1p(Y_test)
clf_lgb = LGBMRegressor(n_estimators=10000, **lgb_params, random_state=123456789, n_jobs=-1)
clf_lgb.fit(X_train, Y_train, **lgb_fit_params)
lgb_model = LGBMRegressor(bagging_fraction=0.8, bagging_freq=1, lambda_l1=0.01,
              lambda_l2=0.01, learning_rate=0.01, max_bin=255, max_depth=6,
              metric='rmse', min_data_in_bin=1, min_data_in_leaf=1,
              n_estimators=10000, objective='regression',
              random_state=123456789, verbose=0)
lgb_model.fit(X_train,Y_train)

lgm_preds = lgb_model.predict(X_test)
lgm_preds = np.exp(lgm_preds)
lgm_preds = pd.DataFrame(lgm_preds)
lgm_preds.index = Y_test.index
Y_train = np.exp(Y_train)
Y_test = np.exp(Y_test)
plt.figure(figsize=(20,5))
plt.plot(Y_train)
plt.plot(Y_test, label='Original')
plt.plot(xgb_preds, color='cyan', label="xgb_prediction")
plt.plot(lgm_preds, color='red', label='light_lgm_prediction')
plt.legend(loc='best')
from catboost import CatBoostRegressor
Y_train = np.log1p(Y_train)
Y_test = np.log1p(Y_test)

cat_model=CatBoostRegressor()
cat_model.fit(X_train, Y_train)
cat_preds = cat_model.predict(X_test)
cat_preds = np.exp(cat_preds)
cat_preds = pd.DataFrame(cat_preds)
cat_preds.index = Y_test.index
Y_train = np.exp(Y_train)
Y_test = np.exp(Y_test)
plt.figure(figsize=(20,5))
plt.plot(Y_train)
plt.plot(Y_test, label='Original')
plt.plot(xgb_preds, color='cyan', label="xgb_prediction")
plt.plot(lgm_preds, color='red', label='light_lgm_prediction')
plt.plot(cat_preds, color='green', label='cat_prediction')
plt.legend(loc='best')
prophet_data = train_df[['Date','num_orders']]
prophet_data.index = xb_data.index
prophet_data = prophet_data.iloc[:split_size,:]
prophet_data =prophet_data.rename(columns={'Date':'ds',
                             'num_orders':'y'})
prophet_data.head()
from fbprophet import Prophet
m = Prophet(growth='linear',
            seasonality_mode='multiplicative',
#            changepoint_prior_scale = 30,
           seasonality_prior_scale = 35,
           holidays_prior_scale = 10,
           daily_seasonality = True,
           weekly_seasonality = False,
           yearly_seasonality= False,
           ).add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=30
            
            ).add_seasonality(
                name='weekly',
                period=7,
                fourier_order=55
            ).add_seasonality(
                name='yearly',
                period=365.25,
                fourier_order=20
            )
        
m.fit(prophet_data)
future = m.make_future_dataframe(periods=15, freq='W')
forecast = m.predict(future)
# forecast['yhat'] = np.exp(forecast['yhat'])
# forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
# forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig2 = m.plot_components(forecast)

from fbprophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

fig = plot_plotly(m, forecast)  # This returns a plotly Figure
py.iplot(fig)
prophet_preds = forecast['yhat'].iloc[split_size:]
prophet_preds.index = Y_test.index
plt.figure(figsize=(20,5))
plt.plot(Y_train)
plt.plot(Y_test, label='Original')
plt.plot(xgb_preds, color='cyan', label="xgb_prediction")
plt.plot(lgm_preds, color='red', label='light_lgm_prediction')
plt.plot(prophet_preds, color='green', label='prophet_prediction')
plt.plot(cat_preds, color='blue', label='cat_prediction')
plt.legend(loc='best')
Y_train1=pd.DataFrame(Y_train)
Y_train1
original=pd.DataFrame(Y_test)
xgb_preds1=pd.DataFrame(xgb_preds)
lgm_preds1=pd.DataFrame(lgm_preds)
prophet_preds1=pd.DataFrame(prophet_preds)
cat_preds1=pd.DataFrame(cat_preds)
cat_preds1
import plotly.graph_objs as go
import plotly.offline as pyoff
plot_data = [
    go.Scatter(
        x=Y_train1.index,
        y=Y_train1['num_orders'],
        name='Time Series for num_orders',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=original.index,
        y=original['num_orders'],
        name='Original',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=xgb_preds1.index,
        y=xgb_preds1[0],
        name='xgb_prediction',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=lgm_preds1.index,
        y=lgm_preds1[0],
        name='light_lgm_prediction',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=prophet_preds1.index,
        y=prophet_preds1['yhat'],
        name='prophet_prediction',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=cat_preds1.index,
        y=cat_preds1[0],
        name='cat_prediction',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    )
    
    
]
plot_layout = go.Layout(
        title='Total orders per week',
        yaxis_title='Total orders',
        xaxis_title='Week',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
plt.figure(figsize=(20,5))
# plt.plot(Y_train)
plt.plot(Y_test, label='Original')
plt.plot(xgb_preds, color='cyan', label="xgb_prediction")
plt.plot(lgm_preds, color='red', label='light_lgm_prediction')
plt.plot(prophet_preds, color='green', label='prophet_prediction')
plt.plot(cat_preds, color='blue', label='cat_prediction')
plt.legend(loc='best')
import plotly.graph_objs as go
import plotly.offline as pyoff
plot_data = [
    go.Scatter(
        x=original.index,
        y=original['num_orders'],
        name='Original',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=xgb_preds1.index,
        y=xgb_preds1[0],
        name='xgb_prediction',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=lgm_preds1.index,
        y=lgm_preds1[0],
        name='light_lgm_prediction',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=prophet_preds1.index,
        y=prophet_preds1['yhat'],
        name='prophet_prediction',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=cat_preds1.index,
        y=cat_preds1[0],
        name='cat_prediction',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    )
    
    
]
plot_layout = go.Layout(
        title='Total orders per week',
        yaxis_title='Total orders',
        xaxis_title='Week',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
a = np.array(prophet_preds)
b = np.array(lgm_preds)
c = np.array(xgb_preds)
d = np.array(cat_preds)
final_preds =  (b*0.8)+ (d*0.2) 
final_preds = (final_preds*0.4) + (a*0.6)
final_preds[6]
final_preds = pd.DataFrame(final_preds[6])
final_preds.index = Y_test.index
final_preds = pd.DataFrame(final_preds)
final_preds.index = Y_test.index
plt.figure(figsize=(20,5))
# plt.plot(Y_train)
plt.plot(Y_test, label='Original')
plt.plot(xgb_preds, color='cyan', label="xgb_prediction")
plt.plot(lgm_preds, color='orange', label='light_lgm_prediction')
plt.plot(prophet_preds, color='green', label='prophet_prediction')
plt.plot(final_preds, color='red',linestyle='--', label='final_prediction')
plt.plot(cat_preds, color='blue', label='cat_prediction')
plt.legend(loc='best')
final_preds1=pd.DataFrame(final_preds)
final_preds1
import plotly.graph_objs as go
import plotly.offline as pyoff
plot_data = [
    go.Scatter(
        x=Y_train1.index,
        y=Y_train1['num_orders'],
        name='Time Series for num_orders',
        #marker = dict(color = 'Blue')
        #x_axis="OTI",
        #y_axis="time",
    ),
    go.Scatter(
        x=final_preds1.index,
        y=final_preds1[0],
        name='final_prediction',
        marker = dict(color = 'Red')
        #x_axis="OTI",
        #y_axis="time",
    ),
    
]
plot_layout = go.Layout(
        title='Total orders per week',
        yaxis_title='Total orders',
        xaxis_title='Week',
        plot_bgcolor='rgba(0,0,0,0)'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)
from sklearn.metrics import mean_squared_error
print(mean_squared_error(Y_test, final_preds, squared=False))