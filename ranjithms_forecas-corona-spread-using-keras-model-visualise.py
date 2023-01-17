import numpy as np # linear algebra

import pandas as pd 

import seaborn as sns

from matplotlib import pyplot as plt

import matplotlib as mpl



import plotly.express as px

import plotly.graph_objects as go

import matplotlib.colors as mcolors



import operator 

import random



from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization, Flatten

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing import text, sequence

from tensorflow.keras.callbacks import Callback

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model

from tensorflow.keras.losses import mean_squared_error as mse_loss



from tensorflow.keras import optimizers

from tensorflow.keras.optimizers import RMSprop, Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
data=pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
data.head()
data.describe()
def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            print("min for this col: ",mn)

            print("max for this col: ",mx)

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",df[col].dtype)

            print("******************************")

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist
train=data

b=len(train)

print("total examples:",b)

for col in train.columns:

  a=train[col].isna().sum()

  c=(a/b)*100

  print(col, "has", a ,"NaN's with",c,"percentage")
fig, ax = plt.subplots(figsize=(50,25))

# use a ranked correlation to catch nonlinearities

corr = train[[col for col in train.columns]].corr(method='spearman')

_ = sns.heatmap(corr, annot=True,

                xticklabels=corr.columns.values,

                yticklabels=corr.columns.values)
confirmed=data

dates = confirmed.keys()

dates
confirmed_df=data

cols = confirmed_df.keys()

confirmed=data.loc[:, cols[4]:cols[-1]]

dates = confirmed.keys()



world_cases = []



china_cases = [] 

italy_cases = []

us_cases = [] 

india_cases= []

spain_cases = [] 



for i in dates:

    confirmed_sum = confirmed[i].sum()

   

    world_cases.append(confirmed_sum)

    



    # case studies 

    china_cases.append(confirmed_df[confirmed_df['Country/Region']=='China'][i].sum())

    italy_cases.append(confirmed_df[confirmed_df['Country/Region']=='Italy'][i].sum())

    india_cases.append(confirmed_df[confirmed_df['Country/Region']=='India'][i].sum())

    us_cases.append(confirmed_df[confirmed_df['Country/Region']=='US'][i].sum())

    spain_cases.append(confirmed_df[confirmed_df['Country/Region']=='Spain'][i].sum())
def daily_increase(data):

    d = [] 

    for i in range(len(data)):

        if i == 0:

            d.append(data[0])

        else:

            d.append(data[i]-data[i-1])

    return d 



world_daily_increase = daily_increase(world_cases)

china_daily_increase = daily_increase(china_cases)

italy_daily_increase = daily_increase(italy_cases)

us_daily_increase = daily_increase(us_cases)

spain_daily_increase = daily_increase(spain_cases)

india_daily_increase = daily_increase(india_cases)
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

days_in_future = 10

future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-10]
type(world_cases)
x = []

for sublist in adjusted_dates:

    for item in sublist:

        x.append(item)



#x=adjusted_dates.tolist()

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=world_cases,mode='lines+markers',name='Confirmed cases'))

fig.update_layout(title='Coronavirus Cases Over Time',xaxis_title="Days Since 1/22/2020",

    yaxis_title="Number of cases",)

fig.update_yaxes(nticks=20)

fig.update_xaxes(nticks=36)

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(y=world_daily_increase,name='Confirmed cases'))

fig.update_layout(title='Coronavirus Cases Increasing Over Time',xaxis_title="Days Since 1/22/2020",

    yaxis_title="Number of cases",)

fig.update_yaxes(nticks=20)

fig.update_xaxes(nticks=36)

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(y=china_daily_increase,name='Confirmed cases'))

fig.update_layout(title='Coronavirus Cases Increasing Over Time in China',xaxis_title="Days Since 1/22/2020",

    yaxis_title="Number of cases",)

fig.update_yaxes(nticks=20)

fig.update_xaxes(nticks=36)

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(y=italy_daily_increase,name='Confirmed cases'))

fig.update_layout(title='Coronavirus Cases Increasing Over Time in Italy',xaxis_title="Days Since 1/22/2020",

    yaxis_title="Number of cases",)

fig.update_yaxes(nticks=20)

fig.update_xaxes(nticks=36)

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(y=us_daily_increase,name='Confirmed cases'))

fig.update_layout(title='Coronavirus Cases Increasing Over Time in US',xaxis_title="Days Since 1/22/2020",

    yaxis_title="Number of cases",)

fig.update_yaxes(nticks=20)

fig.update_xaxes(nticks=36)

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(y=spain_daily_increase,name='Confirmed cases'))

fig.update_layout(title='Coronavirus Cases Increasing Over Time in Spain',xaxis_title="Days Since 1/22/2020",

    yaxis_title="Number of cases",)

fig.update_yaxes(nticks=20)

fig.update_xaxes(nticks=36)

fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(y=india_daily_increase,name='Confirmed cases'))

fig.update_layout(title='Coronavirus Cases Increasing Over Time in India',xaxis_title="Days Since 1/22/2020",

    yaxis_title="Number of cases",)

fig.update_yaxes(nticks=20)

fig.update_xaxes(nticks=36)

fig.show()
#x=adjusted_dates.tolist()

fig = go.Figure()





fig.add_trace(go.Scatter(x=x, y=india_cases,mode='lines+markers',name="India's cases"))





fig.update_layout(title='Coronavirus Cases Over Time In India',xaxis_title="Days Since 1/22/2020",

    yaxis_title="Number of cases",)

fig.update_yaxes(nticks=20)

fig.update_xaxes(nticks=36)

fig.show()
#x=adjusted_dates.tolist()

fig = go.Figure()



#fig.add_trace(go.Scatter(x=x, y=world_cases,mode='lines+markers',name='Totla cases'))

fig.add_trace(go.Scatter(x=x, y=china_cases,mode='lines+markers',name="China's cases"))

fig.add_trace(go.Scatter(x=x, y=india_cases,mode='lines+markers',name="India's cases"))

fig.add_trace(go.Scatter(x=x, y=us_cases,mode='lines+markers',name="USA's cases"))

fig.add_trace(go.Scatter(x=x, y=italy_cases,mode='lines+markers',name="Italy's cases"))

fig.add_trace(go.Scatter(x=x, y=spain_cases,mode='lines+markers',name="Spain's cases"))



fig.update_layout(title='Coronavirus Cases Over Time',xaxis_title="Days Since 1/22/2020",

    yaxis_title="Number of cases",)

fig.update_yaxes(nticks=20)

fig.update_xaxes(nticks=36)

fig.show()
#x=adjusted_dates.tolist()

fig = go.Figure()



fig.add_trace(go.Scatter(x=x, y=world_cases,mode='lines+markers',name='Total cases'))

fig.add_trace(go.Scatter(x=x, y=china_cases,mode='lines+markers',name="China's cases"))

fig.add_trace(go.Scatter(x=x, y=india_cases,mode='lines+markers',name="India's cases"))

fig.add_trace(go.Scatter(x=x, y=us_cases,mode='lines+markers',name="USA's cases"))

fig.add_trace(go.Scatter(x=x, y=italy_cases,mode='lines+markers',name="Italy's cases"))

fig.add_trace(go.Scatter(x=x, y=spain_cases,mode='lines+markers',name="Spain's cases"))



fig.update_layout(title='Coronavirus Cases Over Time',xaxis_title="Days Since 1/22/2020",

    yaxis_title="Number of cases",)

fig.update_yaxes(nticks=20)

fig.update_xaxes(nticks=36)

fig.show()
unique_countries =  list(confirmed_df['Country/Region'].unique())

country_confirmed_cases = []

latest_confirmed = confirmed_df[dates[-1]]

no_cases = []

for i in unique_countries:

    cases = latest_confirmed[confirmed_df['Country/Region']==i].sum()

    if cases > 0:

        country_confirmed_cases.append(cases)

    else:

        no_cases.append(i)

        

for i in no_cases:

    unique_countries.remove(i)

    

# sort countries by the number of confirmed cases

unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1), reverse=True)]

for i in range(len(unique_countries)):

    country_confirmed_cases[i] = latest_confirmed[confirmed_df['Country/Region']==unique_countries[i]].sum()
# number of cases per country/region

print('Confirmed Cases by Countries/Regions:')

for i in range(len(unique_countries)):

    print(f'{unique_countries[i]}: {country_confirmed_cases[i]} cases')
visual_unique_countries = [] 

visual_confirmed_cases = []

others = np.sum(country_confirmed_cases[10:])



for i in range(len(country_confirmed_cases[:10])):

    visual_unique_countries.append(unique_countries[i])

    visual_confirmed_cases.append(country_confirmed_cases[i])

    

visual_unique_countries.append('Others')

visual_confirmed_cases.append(others)
plt.figure(figsize=(16, 9))

plt.barh(visual_unique_countries, visual_confirmed_cases)

plt.title('# of Covid-19 Confirmed Cases in Countries/Regions', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
full_table = confirmed_df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Confirmed")



full_table['Date'] = pd.to_datetime(full_table['Date'])

full_table.head()
len(full_table)
full_table.head()
data=full_table.drop(['Lat','Long'],axis=1)
data['days']=(data['Date']-pd.to_datetime("2020-01-22")).dt.days
data.tail()
data['Confirmed'].max()
from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()

le2 = LabelEncoder()
categoricals=["Province/State", 	"Country/Region"]

numericals=["days"]
data.fillna(value="no",inplace=True)

data.tail()
data["Province/State"]=le1.fit_transform(data["Province/State"])

data["Country/Region"]=le2.fit_transform(data["Country/Region"])
data.tail()
from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler()

scaler2 = StandardScaler()
data[["Confirmed"]]=scaler1.fit_transform(data[["Confirmed"]].to_numpy())

data[["days"]]=scaler2.fit_transform(data[["days"]].to_numpy())
data=data.drop("Date",axis=1)

data.head()
data, NAlist = reduce_mem_usage(data)

data.head()
val=data[data.days>61]

train=data[data.days<62]
train.head()
train.describe()
def model(dense_dim_1=16, dense_dim_2=8, dense_dim_3=8, dense_dim_4=4, 

dropout1=0.4, dropout2=0.3, dropout3=0.3, dropout4=0.4, lr=0.0005,pre_model=None):



    #Inputs

    state = Input(shape=[1], name="Province/State")

    country = Input(shape=[1], name="Country/Region")

    #conf = Input(shape=[1], name="Confirmed")

    days = Input(shape=[1], name="days")

    

   

    #Embeddings layers

    emb_state = Embedding(77, 4)(state)

    emb_country = Embedding(175, 4)(country)

    



    concat_emb = concatenate([

           Flatten() (emb_state)

         , Flatten() (emb_country)

         

    ])

    

    categ = Dropout(dropout1)(Dense(dense_dim_1,activation='relu') (concat_emb))

    categ = BatchNormalization()(categ)

    categ = Dropout(dropout2)(Dense(dense_dim_2,activation='relu') (categ))

    

    #main layer

    main_l = concatenate([

          categ

        , days

        

    ])

    

    main_l = Dropout(dropout3)(Dense(dense_dim_3,activation='relu') (main_l))

    main_l = BatchNormalization()(main_l)

    main_l = Dropout(dropout4)(Dense(dense_dim_4,activation='relu') (main_l))

    

    #output

    output = Dense(1) (main_l)



    model = Model([ state,

                    country, 

                    days], output)



    model.compile(optimizer = Adam(lr=lr),

                  loss= mse_loss,

                  metrics=[root_mean_squared_error,"accuracy"])

    return model



def root_mean_squared_error(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))
def get_keras_data(df, num_cols, cat_cols):

    cols = num_cols + cat_cols

    X = {col: np.array(df[col]) for col in cols}

    return X
Y_train=train.Confirmed

X_train=train.drop('Confirmed',axis=1)

Y_valid=val.Confirmed

X_valid=val.drop("Confirmed",axis=1)
X_t = get_keras_data(X_train, numericals, categoricals)

X_v = get_keras_data(X_valid, numericals, categoricals)
keras_model=model(lr=0.001)
early_stopping = EarlyStopping(patience=105, verbose=2)

model_checkpoint = ModelCheckpoint("corona.hdf5",

                                       save_best_only=True, verbose=2, monitor='val_root_mean_squared_error', mode='min')

hist = keras_model.fit(X_t, Y_train, batch_size=128, epochs=150,

                             validation_split=0.1,

                            callbacks=[early_stopping, model_checkpoint])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['loss'])
Modl=load_model("corona.hdf5",custom_objects={'root_mean_squared_error': root_mean_squared_error})
sub=pd.read_csv("/kaggle/input/submission(1).csv")

sub=sub.melt(id_vars=["Province/State", "Country/Region"], var_name="Date", value_name="Confirmed")

sub['Date'] = pd.to_datetime(sub['Date'])

sub.head()
sub["Province/State"].fillna(value="no",inplace=True)



sub["Province/State"]=le1.transform(sub["Province/State"])

sub["Country/Region"]=le2.transform(sub["Country/Region"])
sub['days']=(sub['Date']-pd.to_datetime("2020-01-22")).dt.days
sub[["days"]]=scaler2.transform(sub[["days"]].to_numpy())
sub.head()
len(sub)
for_prediction=get_keras_data(sub,numericals,categoricals)

result=Modl.predict(for_prediction, batch_size=1715)
sub['Confirmed']=result
sub['Confirmed']=scaler1.inverse_transform(sub['Confirmed'])
sub.head()
sub.Confirmed.max()
sub.describe()

sub['Confirmed']=(sub.Confirmed.astype(np.int64))
sub=sub.drop("days",axis=1)
sub.head()
pt=sub.set_index(["Province/State", "Country/Region", 'Date']).unstack('Date').reset_index()

pt.columns.name=None

pt.reset_index()
sub1=pd.read_csv('/kaggle/input/submission(1).csv')
pt.columns=sub1.columns
pt['Province/State']=le1.inverse_transform(pt['Province/State'])
pt['Country/Region']=le2.inverse_transform(pt['Country/Region'])
pt['Province/State'].value_counts()
pt=pt.replace("no","")
len(pt)
pt.head()
pt['Province/State'].value_counts()
pt.to_csv("submit.csv")