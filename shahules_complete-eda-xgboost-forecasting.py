from statsmodels.graphics.tsaplots import plot_pacf

from statsmodels.graphics.tsaplots import plot_acf

from sklearn.preprocessing import LabelEncoder

from plotly.offline import init_notebook_mode

from datetime import datetime, timedelta

init_notebook_mode(connected=False)

from keras.models import Sequential

import plotly.graph_objects as go

import matplotlib.pyplot as plt

from fbprophet import Prophet

from sklearn import metrics

import plotly.express as px

import plotly.offline as py

from keras import layers

import lightgbm as lgb

import seaborn as sns

import pandas as pd

import numpy as np

import gc

import os





print(os.listdir("../input/novel-corona-virus-2019-dataset/"))

path="../input/novel-corona-virus-2019-dataset/"
df_one = pd.read_csv(path+'time_series_covid_19_recovered.csv')
df_one.head()
df_two=pd.read_csv(path+"time_series_covid_19_deaths.csv")

df_two=pd.melt(df_two,id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name=['Date'],value_name='Deadths')

df_two.rename({'Province/State':"State","Country/Region":'Country'},axis=1,inplace=True)

df_two['Date']=pd.to_datetime(df_two['Date'])

def melt_and_merge(agg=True):



    df_one = pd.read_csv(path+'time_series_covid_19_recovered.csv')

    df_one=pd.melt(df_one,id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name=['Date'],value_name='Recovered')

    df_one.rename({'Province/State':"State","Country/Region":'Country'},axis=1,inplace=True)

    df_one['Date']=pd.to_datetime(df_one['Date'])



    df_two=pd.read_csv(path+"time_series_covid_19_deaths.csv")

    df_two=pd.melt(df_two,id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name=['Date'],value_name='Deaths')

    df_two.rename({'Province/State':"State","Country/Region":'Country'},axis=1,inplace=True)

    df_two['Date']=pd.to_datetime(df_two['Date'])



    df_three=pd.read_csv(path+"time_series_covid_19_confirmed.csv")

    df_three=pd.melt(df_three,id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],var_name=['Date'],value_name='Confirmed')

    df_three.rename({'Province/State':"State","Country/Region":'Country'},axis=1,inplace=True)

    df_three['Date']=pd.to_datetime(df_three['Date'])

    

    if (agg):

        col={"Lat":np.mean,"Long":np.mean,"Recovered":sum}

        df_one=df_one.groupby(['Country',"Date"],as_index=False).agg(col)

        

        col={"Lat":np.mean,"Long":np.mean,"Deaths":sum}

        df_two=df_two.groupby(['Country',"Date"],as_index=False).agg(col)

        

        col={"Lat":np.mean,"Long":np.mean,"Confirmed":sum}

        df_three=df_three.groupby(['Country',"Date"],as_index=False).agg(col)



    else:

        df_one['State'].fillna(df_one['Country'],inplace=True)

        df_two['State'].fillna(df_two['Country'],inplace=True)

        df_three['State'].fillna(df_three['Country'],inplace=True)

    

    

    print("The shape of three datasets are equal :",(df_three.shape[0]==df_one.shape[0]==df_two.shape[0]))

    

    merge=pd.merge(df_one,df_two)

    merge=pd.merge(merge,df_three)

    

    return merge
data=melt_and_merge(True)
data.head()
print("There are {} rows are {} columns in our data".format(data.shape[0],data.shape[1]))
print("The data starts from the date {} and ends in {}".format(data.Date.min().date(),data.Date.max().date()))

print("So we have {} of data".format(data.Date.max().date()-data.Date.min().date()))

print("From {} countries".format(data.Country.nunique()))
x=data.groupby(['Country'],as_index=False)['Deaths'].last().sort_values(by="Deaths",ascending=False)

fig=px.pie(x,"Country","Deaths")

fig.update_layout(title="Global Covid-19 Deaths")
df_four=pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv",usecols=['age','sex','province','country','wuhan(0)_not_wuhan(1)',

                                                                                              "latitude",'longitude'])

df_four.head()
df_four.sex.fillna('Unknown',inplace=True)

df_four.sex=df_four['sex'].map({"Female":"female","Male":"male","male":"male",'female':'female',"Unknown":"Unknown"})

sex= df_four.sex.value_counts()[1:]

fig=px.pie(sex,sex.index,sex)

fig.update_layout(title="Male vs Female infected Globally")
most_effected=data.groupby(['Country'],as_index=False)['Deaths'].last().sort_values(by="Deaths",ascending=False)[:10]

fig=px.bar(most_effected,x="Country",y="Deaths",title="Most affected countries by Number of Deaths")

fig.show()
most_effected=data.groupby(['Country'],as_index=False)['Confirmed'].last().sort_values(by="Confirmed",ascending=False)[:10]

fig=px.bar(most_effected,x="Country",y="Confirmed",title="Most affected countries by Number of Confirmed Cases")

fig.show()
most_affected=data.groupby(['Country'],as_index=False)['Recovered'].last().sort_values(by="Recovered",ascending=False)[:10]

fig=px.bar(most_affected,x="Country",y="Recovered",title="Most affected countries by Number of Recovered Cases")

fig.show()
fig = go.Figure()

for country in ["China","Italy","Iran","Spain"]:

   



    fig.add_trace(go.Scatter(

        x=data[data['Country']==country]['Date'],

        y=data[data['Country']==country]['Confirmed'],

        name = country, # Style name/legend entry with html tags

        connectgaps=True # override default to connect the gaps

    ))

fig.update_layout(title="Timeseries plot of number of Confirmed Cases") 

fig.update_traces(mode='markers+lines', marker_line_width=2.5, marker_size=3)

fig.show()
fig = go.Figure()

for country in ['China',"Italy","Iran","Spain"]:

   



    fig.add_trace(go.Scatter(

        x=data[data['Country']==country]['Date'],

        y=data[data['Country']==country]['Deaths'],

        name = country, # Style name/legend entry with html tags

        connectgaps=True # override default to connect the gaps

    ))

fig.update_layout(title="Timeseries plot of number of deaths")    

fig.update_traces(mode='markers+lines', marker_line_width=2.5, marker_size=3)



fig.show()
fig = go.Figure()

for country in ['China',"Italy","Iran","Spain"]:

   



    fig.add_trace(go.Scatter(

        x=data[data['Country']==country]['Date'],

        y=data[data['Country']==country]['Confirmed'],

        name = country, # Style name/legend entry with html tags

        connectgaps=True # override default to connect the gaps

    ))

fig.update_layout(title="Timeseries plot of number of Recovered Cases")  

fig.update_traces(mode='markers+lines', marker_line_width=2.5, marker_size=3)

fig.show()
fig = go.Figure()

for country in ['China',"Italy","Iran"]:

   

    lag_1=data[data['Country']==country]['Confirmed'].shift(1)

    change=(data[data['Country']==country]['Confirmed']-lag_1).fillna(0)

    fig.add_trace(go.Scatter(

        x=data[data['Country']==country]['Date'],

        y=change,

        name = country, # Style name/legend entry with html tags

        connectgaps=True # override default to connect the gaps

    ))

fig.update_layout(title="Timeseries plot of number of Confirmed cases each day")    

fig.update_traces(mode='markers+lines', marker_line_width=2.5, marker_size=3)



fig.show()
fig = go.Figure()

for country in ['China',"Italy","Iran"]:

   

    lag_1=data[data['Country']==country]['Deaths'].shift(1)

    change=(data[data['Country']==country]['Deaths']-lag_1).fillna(0)

    fig.add_trace(go.Scatter(

        x=data[data['Country']==country]['Date'],

        y=change,

        name = country, # Style name/legend entry with html tags

        connectgaps=True # override default to connect the gaps

    ))

fig.update_layout(title="Timeseries plot of number of Deaths each day")   

fig.update_traces(mode='markers+lines', marker_line_width=2.5, marker_size=3)



fig.show()
locations=pd.read_csv("../input/plotlycountrycodes/plotly_countries_and_codes.csv",usecols=['COUNTRY','CODE'])

df=pd.merge(data,locations,left_on=['Country'],right_on=['COUNTRY'],how="left")

df_country=df.groupby(['Country',"CODE"],as_index=False)[['Recovered',"Deaths","Confirmed"]].last()
fig = px.choropleth(df_country, locations="CODE",

                    color="Deaths", # lifeExp is a column of gapminder

                    hover_name="Country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(title="Global COVID-19 Deaths")

fig.show()
fig = px.choropleth(df_country, locations="CODE",

                    color="Confirmed", # lifeExp is a column of gapminder

                    hover_name="Country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(title="Global COVID-19 Confirmed Cases")

fig.show()
fig = px.choropleth(df_country, locations="CODE",

                    color="Recovered", # lifeExp is a column of gapminder

                    hover_name="Country", # column to add to hover information

                    color_continuous_scale=px.colors.sequential.Plasma)

fig.update_layout(title="Global COVID-19 Recovered Cases")

fig.show()
china=data[data['Country']=="China"]

fig = go.Figure()

for i in ["Confirmed","Recovered","Deaths"]:

   



    fig.add_trace(go.Scatter(

        y=china[i],

        x=china['Date'],

        name = i, # Style name/legend entry with html tags

        connectgaps=True # override default to connect the gaps

    ))

fig.update_layout(title="Timeseries plot of China ") 

fig.update_traces(mode='markers+lines', marker_line_width=2.5, marker_size=3)



fig.show()
df=melt_and_merge(False)
china=df[df['Country']=="China"]

fig = go.Figure()

states=china.State.unique().tolist()

states.remove('Hubei')

for country in states:

   



    fig.add_trace(go.Scatter(

        x=china[china['State']==country]['Date'],

        y=china[china['State']==country]['Confirmed'],

        name = country, # Style name/legend entry with html tags

        connectgaps=True # override default to connect the gaps

    ))

fig.update_layout(title="Timeseries plot of number of Confirmed Cases in Provinces except Hubei")  

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.update_traces(mode='lines', marker_line_width=2.5, marker_size=3)



fig.show()

china=df[df['Country']=="China"]

fig = go.Figure()

states=china.State.unique().tolist()

states.remove('Hubei')

for country in states:

   



    fig.add_trace(go.Scatter(

        x=china[china['State']==country]['Date'],

        y=china[china['State']==country]['Deaths'],

        name = country, # Style name/legend entry with html tags

        connectgaps=True # override default to connect the gaps

    ))

fig.update_layout(title="Timeseries plot of number of Deaths Cases in Provinces except Hubei")

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
china=df[df['Country']=="China"].groupby(['State'],as_index=False)[[ 'Lat', 'Long', 'Date', 'Recovered', 'Deaths',"Confirmed"]].last()
fig=px.bar(china,x="State",y="Confirmed")

fig.update_layout(title="Confirmed Cases in Provinces of China")

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
Hubei=df[df['State']=="Hubei"]

Hubei.loc[:,'lag_1']=Hubei['Confirmed'].shift(1)

Hubei.loc[:,'Daily']=(Hubei['Confirmed']-Hubei['lag_1']).fillna(0).values

fig=px.bar(Hubei,x="Date",y="Daily")

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.update_layout(title="Daily Confirmed Cases in Hubei province")

fig.show()
Hubei=df[df['State']=="Hubei"]

Hubei.loc[:,'lag_1']=Hubei['Recovered'].shift(1)

Hubei.loc[:,'Daily']=(Hubei['Recovered']-Hubei['lag_1']).fillna(0)

fig=px.bar(Hubei,x="Date",y="Daily")

fig.update_layout(title="Daily recovered in Hubei province")

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()

Hubei=df[df['State']=="Hubei"]

Hubei.loc[:,'lag_1']=Hubei['Deaths'].shift(1)

Hubei.loc[:,'Daily']=(Hubei['Deaths']-Hubei['lag_1']).fillna(0)

fig=px.bar(Hubei,x="Date",y="Daily",)

fig.update_layout(title="Daily Deaths in Hubei province")

fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')

fig.show()
restof_world=data[data['Country']!="China"].groupby(['Date'],as_index=False)[['Confirmed',"Recovered","Deaths"]].agg(sum)

china=data[data['Country']=="China"]


fig = go.Figure()

fig.add_trace(go.Bar(x=china['Date'],

                y=china['Confirmed'],

                name='China',

                marker_color='rgb(255, 0, 0)'

                ))

fig.add_trace(go.Bar(x=restof_world['Date'],

                y=restof_world['Confirmed'],

                name='Rest of world',

                marker_color='rgb(0, 0, 255)'

                ))



fig.update_layout(

    title='Global Confirmed Cases, China and Rest of World',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Confirmed Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')



fig.show()





fig = go.Figure()

fig.add_trace(go.Bar(x=china['Date'],

                y=china['Deaths'],

                name='China',

                marker_color='rgb(255, 0, 0)'

                ))

fig.add_trace(go.Bar(x=restof_world['Date'],

                y=restof_world['Deaths'],

                name='Rest of world',

                marker_color='rgb(0, 0, 255)'

                ))



fig.update_layout(

    title='Global Deaths China and Rest of World',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Death Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(250, 242, 242,0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')



fig.show()





fig = go.Figure()

fig.add_trace(go.Bar(x=china['Date'],

                y=china['Recovered'],

                name='China',

                marker_color='rgb(255, 0, 0)'

                ))

fig.add_trace(go.Bar(x=restof_world['Date'],

                y=restof_world['Recovered'],

                name='Rest of world',

                marker_color='rgb(0, 0, 255)'

                ))



fig.update_layout(

    title='Global Recovered Cases, China and Rest of World',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Recovered Cases',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.update_layout(plot_bgcolor='rgb(250, 242, 242)')



fig.show()



from plotly.subplots import make_subplots



fig = make_subplots(

    rows=2, cols=2,

    subplot_titles=("Italy","Korea, South","Spain", "India",))



countries=["Italy","Korea, South","Spain", "India",]



    

country=data[data['Country']==countries[0]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Confirmed'],

                    marker=dict(color=country['Confirmed'], coloraxis="coloraxis")),

              1, 1)

    

country=data[data['Country']==countries[1]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Confirmed'],

                    marker=dict(color=country['Confirmed'], coloraxis="coloraxis")),

              1,2 )

    

country=data[data['Country']==countries[2]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Confirmed'],

                    marker=dict(color=country['Confirmed'], coloraxis="coloraxis")),

              2, 1)

    

country=data[data['Country']==countries[3]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Confirmed'],

                    marker=dict(color=country['Confirmed'], coloraxis="coloraxis")),

              2,2 )

fig.update_layout(title="Confirmed cases in Italy,S.Korea,Spain and India")



fig.show()   
fig = make_subplots(

    rows=2, cols=2,

    subplot_titles=("Italy","Korea, South","Spain", "India",))



countries=["Italy","Korea, South","Spain", "India",]



    

country=data[data['Country']==countries[0]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Deaths'],

                    marker=dict(color=country['Deaths'], coloraxis="coloraxis")),

              1, 1)

    

country=data[data['Country']==countries[1]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Deaths'],

                    marker=dict(color=country['Deaths'], coloraxis="coloraxis")),

              1,2 )

    

country=data[data['Country']==countries[2]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Deaths'],

                    marker=dict(color=country['Deaths'], coloraxis="coloraxis")),

              2, 1)

    

country=data[data['Country']==countries[3]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Deaths'],

                    marker=dict(color=country['Deaths'], coloraxis="coloraxis")),

              2,2 )

fig.update_layout(title="Deaths in Italy,S.Korea,Spain and India")



fig.show()   
fig = make_subplots(

    rows=2, cols=2,

    subplot_titles=("Italy","Korea, South","Spain", "India",))



countries=["Italy","Korea, South","Spain", "India",]



    

country=data[data['Country']==countries[0]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Recovered'],

                    marker=dict(color=country['Recovered'], coloraxis="coloraxis")),

              1, 1)

    

country=data[data['Country']==countries[1]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Recovered'],

                    marker=dict(color=country['Recovered'], coloraxis="coloraxis")),

              1,2 )

    

country=data[data['Country']==countries[2]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Recovered'],

                    marker=dict(color=country['Recovered'], coloraxis="coloraxis")),

              2, 1)

    

country=data[data['Country']==countries[3]]

fig.add_trace(go.Scatter(x=country['Date'], y=country['Recovered'],

                    marker=dict(color=country['Recovered'], coloraxis="coloraxis")),

              2,2 )

fig.update_layout(title="Recovered in Italy,S.Korea,Spain and India")



fig.show()   
data=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")
data.isna().sum()

data.fillna("Unknown",inplace=True)

data=data[['ObservationDate',"Province/State","Country/Region","Confirmed","Deaths","Recovered"]]



data.rename({'ObservationDate':"ds","Province/State":"State","Country/Region":"Country"},axis=1,inplace=True)

data['ds']=pd.to_datetime(data['ds'])
data.head()




def train_test_split(df,test_days):

    df=data.copy()

    max_date=df.ds.max()-timedelta(test_days)

    

    for col in ["State","Country"]:

        lb=LabelEncoder()

        df[col]=lb.fit_transform(df[col])

    

    train = df[df['ds'] < max_date]

    #y_train = df[df['ds'] < max_date] [['Confirmed',"Deaths","Recovered"]]

    

    test = df[df['ds'] > max_date]

    #y_test = df[df['ds'] > max_date] [['Confirmed',"Deaths","Recovered"]]

    

    return train,test



train,test= train_test_split(data,7)
def train_predict(train,test):

    targets=['Confirmed',"Deaths","Recovered"]

    predictions=pd.DataFrame()

    for col in targets:

        

        trainX=train[['ds',"State","Country"]+[col]]

        X_test=test[['ds','State', 'Country']]

        

        m= Prophet()

        trainX.rename({col:"y"},axis=1,inplace=True)

        m.add_regressor("State")

        m.add_regressor("Country")

        m.fit(trainX)

        

        future=m.predict(X_test)

        

        predictions[col]=future['yhat']

        

    return predictions

sub=train_predict(train,test)

sub['ds']=test['ds'].values
sub.head()
plot_acf(data[(data['Country']=="Mainland China") & (data['State']=="Hubei")][['Recovered']])
plot_pacf(data[(data['Country']=="Mainland China") & (data['State']=="Hubei")][['Recovered']])
def simple_fe(df):

    

    df['year']=df['ds'].dt.year

    df['month']=df['ds'].dt.month

    df['day']=df['ds'].dt.day

    

    ##lag features

    df.loc[:,'rec_lag_2']=df.groupby(['Country','State'])['Recovered'].transform(lambda x: x.shift(1))

    df.loc[:,'conf_lag_2'] = df.groupby(['Country'])['Confirmed'].transform(lambda x: x.shift(1))

    df.loc[:,'deaths_lag_2'] =df.groupby(['Country'])['Deaths'].transform(lambda x: x.shift(1))

    

    ##rolling mean

    df['rec_rollmean_7']=df.groupby(['Country','State'])['Recovered'].transform(lambda x: x.rolling(7).mean())

    df['conf_rollmean_7'] = df.groupby(['Country'])['Confirmed'].transform(lambda x: x.rolling(7).mean())

    df['deaths_rollmean_7'] =df.groupby(['Country'])['Deaths'].transform(lambda x: x.rolling(7).mean())

    

    ##rolling std

    df['rec_rollstd_7']=df.groupby(['Country','State'])['Recovered'].transform(lambda x: x.rolling(7).std())

    df['conf_rollstd_7'] = df.groupby(['Country'])['Confirmed'].transform(lambda x: x.rolling(7).std())

    df['deaths_rollstd_7'] =df.groupby(['Country'])['Deaths'].transform(lambda x: x.rolling(7).std())

    

    #df.drop(['ds'],axis=1,inplace=True)

    df.fillna(0,inplace=True)

    

    return df

    
data= simple_fe(data)
def run_lgb(data,target):

    

    features=['year', 'month','State', 'Country','Recovered',

               'day', 'rec_lag_2', 'conf_lag_2', 'deaths_lag_2',

               'rec_rollmean_7', 'conf_rollmean_7', 'deaths_rollmean_7',

               'rec_rollstd_7', 'conf_rollstd_7', 'deaths_rollstd_7']

     

    train,test=train_test_split(data,7)

    x_train=train[features]

    y_train=train[target]

    print(x_train.shape)

    x_val=test[features]

    y_val=test[target]

    print(x_val.shape)



    # define random hyperparammeters

    params = {

        'boosting_type': 'gbdt',

        'metric': 'rmse',

        'objective': 'regression',

        'n_jobs': -1,

        'seed': 236,

        'learning_rate': 0.1,

        'bagging_fraction': 0.75,

        'bagging_freq': 10, 

        'colsample_bytree': 0.75}



    train_set = lgb.Dataset(x_train[features], y_train,categorical_feature=['State',"Country",'year','month','day'])

    val_set = lgb.Dataset(x_val[features], y_val,categorical_feature=['State',"Country",'year','month','day'])



    del x_train, y_train



    model = lgb.train(params, train_set, num_boost_round = 500, early_stopping_rounds = 50, valid_sets = [train_set, val_set],

                      verbose_eval = 100,)

    val_pred = model.predict(x_val[features])

    val_score = np.sqrt(metrics.mean_squared_error(val_pred, y_val))

    print(val_score)



    #y_pred = model.predict(x_val)

    #test[targets] = y_pred.values

    return val_pred
sub=pd.DataFrame()

sub['ds']=test['ds'].values

targets=['Confirmed', 'Deaths', 'Recovered']

for target in targets:

    

        sub[target]=run_lgb(data,target)
sub.head()