import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns; sns.set()

# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import missingno as msno

# Configure visualisations

%matplotlib inline

mpl.style.use( 'ggplot' )

plt.style.use('fivethirtyeight')

sns.set(context="notebook", palette="dark", style = 'whitegrid' , color_codes=True)



import warnings

import datetime

warnings.filterwarnings('ignore')
filename = "/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv"

df = pd.read_csv(filename)

df
df.info()
df.columns
countries=['China', 'India', 'US']

y=df.loc[df['Country/Region']=='Italy'].iloc[0,4:]

s = pd.DataFrame({'Italy':y})

for c in countries:    

    #pyplot.plot(range(y.shape[0]),y,'r--')

    s[c] = df.loc[df['Country/Region']==c].iloc[0,4:]

#pyplot.plot(range(y.shape[0]),y,'g-')

plt.plot(range(y.shape[0]), s)
s
for r in df['Country/Region']:

    if r != 'China':

        plt.plot(range(len(df.columns)-4), df.loc[df['Country/Region']==r].iloc[0,4:])

#         pyplot.legend()
df[df['Country/Region'].str.match('US')]
df[df['Country/Region'].str.match('China')]
df[df['Country/Region'].str.match('India')]
india=df.loc[df['Country/Region']=='India'].iloc[0,4:]
india = pd.DataFrame({'India':india})

india
x=df.loc[df['Country/Region']=='China']
x.head()
china=x.iloc[:,4:].sum(axis = 0, skipna = True)
china= pd.DataFrame({'China':china})
china
countries=['China','US']
countries_3=india
for c in countries:

    countries_3[c] = df.loc[df['Country/Region']==c].iloc[0,4:]
countries_3
countries_3.drop(['China'],axis=1)
countries_3['China']=china
countries_3
# from sklearn.preprocessing import MinMaxScaler 

# scaler=MinMaxScaler()

# countries_3_norm = scaler.fit_transform(countries_3)
# countries_3_norm=pd.DataFrame(countries_3_norm)

# countries_3_norm.columns=['India','China','US']
# countries_3_norm
# countries_3_norm_va=countries_3_norm.values
# countries_3_norm_va.shape
# train_va=train.values

# test_va=test.values
india1=countries_3.iloc[:,0]
india1=pd.DataFrame(india1)
india1
from sklearn.preprocessing import MinMaxScaler 

scaler1=MinMaxScaler()

india1 = scaler1.fit_transform(india1)



# # india1=india1/14352
india1=np.array(india1)
india1.shape
india1
lag = 5

#assuming target column is last one

X=[ ]

Y = [ ]

for x in range(lag, len(india1)):

    X.append(india1[x-lag:x,:])
X= np.array(X)

X.shape
count=5

Y=[]

# for i in range(0,87,6):

while(count<87):

    Y.append(india1[count])

    count=count+1

Y= np.array(Y)

Y.shape
Y
X
target=[]

for i in range(0,82):

    target.append(Y[i][0])
target
data=X
data.shape
target= np.array(target)

target.shape
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(data,target,test_size=0.2,random_state=42)
xtrain.shape
ytrain.shape
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



model = Sequential()  

model.add(LSTM((1),input_shape=(5,1),return_sequences=True))

model.add(LSTM((1),return_sequences=False))

# model.add(Dense(100))

model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])

model.summary()
model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])

model.fit(xtrain, ytrain, epochs=50, batch_size=1,validation_data=(xtest, ytest))
predict = model.predict(xtest)
predict
ytest
predict.shape
ytest.shape
plt.plot(range(17),predict,c='r')

plt.plot(range(17),ytest,c='g')

plt.show()
countries_3
countries_3_update=countries_3.reset_index(drop=False)
countries_3_update
countries_3_update=countries_3_update.rename(columns={"index": "Dates"})
gspc="/kaggle/input/stock-history/GSPC.csv"

gspc = pd.read_csv(gspc)



nifty="/kaggle/input/stock-history/NSEI.csv"

nifty = pd.read_csv(nifty)



shanghai_composite="/kaggle/input/stock-history/000001.SS (2).csv"

shanghai_composite= pd.read_csv(shanghai_composite)
countries_3_update
import plotly.express as ax

fig = ax.bar(countries_3_update, x='Dates', y='India', title='India: Confirmed Cases')

fig.show()
fig = ax.bar(countries_3_update, x='Dates', y='US', title='US: Confirmed Cases')

fig.show()
fig = ax.bar(countries_3_update, x='Dates', y='China', title='China: Confirmed Cases')

fig.show()
plt.figure(figsize=(7,7))

for i, col in enumerate(countries_3.columns):

    countries_3[col].plot()

plt.title('Covid-19 cases')

plt.xticks(rotation=70)

plt.legend(countries_3.columns)
gspc.head()
gspc.loc[gspc['Date']=='22-01-20'].index.values
gspc.iloc[257,:]
gspc_plots=gspc.iloc[257:,:].reset_index(drop=True)

gspc_plots=gspc_plots.set_index('Date')
gspc_plots
gspc_plots=gspc_plots.drop(['High','Low','Close','Adj Close','Volume'], axis=1)
gspc_plots
nifty.head()
nifty.loc[nifty['Date']=='22-01-20'].index.values
nifty.iloc[243,:]
nifty_plots=nifty.iloc[243:,:].reset_index(drop=True)

nifty_plots=nifty_plots.set_index('Date')
nifty_plots=nifty_plots.drop(['High','Low','Close','Adj Close','Volume'], axis=1)
nifty_plots
shanghai_composite.head()
shanghai_composite.loc[shanghai_composite['Date']=='22-01-20'].index.values
shanghai_composite.iloc[242,:]
shanghai_composite_plots=shanghai_composite.iloc[242:,:].reset_index(drop=True)

shanghai_composite_plots=shanghai_composite_plots.set_index('Date')
shanghai_composite_plots=shanghai_composite_plots.drop(['High','Low','Close','Adj Close','Volume'], axis=1)
shanghai_composite_plots
stock_3=gspc_plots

stock_3=stock_3.rename(columns={"Open": "GSPC"})

stock_3['NIFTY']=nifty_plots['Open']

stock_3['SHANGHAI COMPOSITE']=shanghai_composite_plots['Open']
stock_3
stock_3=stock_3.reset_index(drop=False)
stock_3
demo_stock=stock_3
demo_stock=demo_stock.iloc[:,1:]
demo_stock
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()



Normalized_stocks=scaler.fit_transform(demo_stock)

Normalized_stocks=pd.DataFrame(Normalized_stocks)

Normalized_stocks['Date']=stock_3['Date']
Normalized_stocks
Normalized_stocks=Normalized_stocks.rename(columns={0:"GSPC",1:'NIFTY',2:'SHANGHAI'})

Normalized_stocks
Normalized_stocks=Normalized_stocks.set_index('Date')
Normalized_stocks
plt.figure(figsize=(10,7))

for i, col in enumerate(Normalized_stocks.columns):

    Normalized_stocks[col].plot()

plt.title('Price Evolution Comparison During COVID-19 outbreak')

plt.xticks(rotation=70)

plt.legend(Normalized_stocks.columns)
countries_3_update.head()
Normalized_stocks.head()
# plt.figure(figsize=(10,10))

# plt.xticks(rotate=90) without axes 



#using axes object

fig,ax=plt.subplots(figsize=(13,8))

ax.plot(countries_3_update.Dates,demo_stock['SHANGHAI COMPOSITE'],marker='o',color="green")

ax.set_xlabel("Dates")

ax.set_ylabel("Shanghai composite",color="green")

#rotating x axes ticks

# fig.autofmt_xdate() #or use for loop if not date 

for tick in ax.get_xticklabels():

    tick.set_rotation(90)



#in order to get two y axes on same plot  we use twinx()



ax2=ax.twinx()

ax2.plot(countries_3_update.Dates,countries_3_update.China)

# plt.figure(figsize=(10,10))

# plt.xticks(rotate=90) without axes 



#using axes object

fig,ax=plt.subplots(figsize=(13,8))

ax.plot(countries_3_update.Dates,demo_stock['NIFTY'],marker='o',color="green")

ax.set_xlabel("Dates")

ax.set_ylabel("Nifty",color="green")

#rotating x axes ticks

# fig.autofmt_xdate() #or use for loop if not date 

for tick in ax.get_xticklabels():

    tick.set_rotation(90)



#in order to get two y axes on same plot  we use twinx()



ax2=ax.twinx()

ax2.plot(countries_3_update.Dates,countries_3_update.India)
# plt.figure(figsize=(10,10))

# plt.xticks(rotate=90) without axes 



#using axes object

fig,ax=plt.subplots(figsize=(13,8))

ax.plot(countries_3_update.Dates,demo_stock['GSPC'],marker='o',color="green")

ax.set_xlabel("Dates")

ax.set_ylabel("DOW JONES",color="green")

#rotating x axes ticks

# fig.autofmt_xdate() #or use for loop if not date 

for tick in ax.get_xticklabels():

    tick.set_rotation(90)



#in order to get two y axes on same plot  we use twinx()



ax2=ax.twinx()

ax2.plot(countries_3_update.Dates,countries_3_update.US)
gspc_plots_predict=gspc.iloc[257:,:].reset_index(drop=True)

# gspc_plots=gspc_plots.set_index('Date')
gspc_plots_predict
import seaborn as sns

plt.figure(1 , figsize = (7, 6))

cor = sns.heatmap(gspc_plots_predict.corr(), annot = True)
x = gspc_plots_predict.loc[:,'High':'Volume']

y = gspc_plots_predict.loc[:,'Open']
y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state = 42)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)
input_22_april=[[2815.10,2775.95,2799.31,2799.31,5049660000]]

predict_next = regressor.predict(input_22_april)
predict_next