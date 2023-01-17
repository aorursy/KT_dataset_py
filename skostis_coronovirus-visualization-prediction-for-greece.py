import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sb

import datetime

# color pallette

cnf = '#393e46' # confirmed - grey

dth = '#ff2e63' # death - red

rec = '#21bf73' # recovered - cyan

act = '#fe9801' # active case - yellow



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')
population=pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv')

data=pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')
treat_numeric=[]

treat_str=[]

for y in population.columns:

    if(population[y].dtype == np.float64 or population[y].dtype == np.int64):

          treat_numeric.append(y)

    else:

          treat_str.append(y)



cols_to_float=['Pop. Density (per sq. mi.)','Coastline (coast/area ratio)', 'Net migration', 'Infant mortality (per 1000 births)', 'Literacy (%)', 'Phones (per 1000)', 'Arable (%)',

 'Crops (%)', 'Other (%)', 'Birthrate', 'Deathrate', 'Agriculture','Climate', 'Industry', 'Service']



for i in cols_to_float:

    population[i]= population[i].str.replace(',','.', case = False) 

    population[i]=pd.to_numeric(population[i])



data=data.groupby(['Country/Region','Date']).sum()

data.reset_index(level='Date',inplace=True)

data=data[data['Confirmed']!=0]

data['date']=pd.to_datetime(data['Date'])

del data['Date']

data.sort_values(by=['Country/Region','date'],inplace=True)

data.reset_index(level=0,inplace=True)
data=data.astype({'Country/Region': str})

population['Country']=population['Country'].astype('str')

data['match']=data['Country/Region'].eq(data['Country/Region'].shift())

data['Country/Region']=data['Country/Region'].str.strip(' ').str.lower()

population['Country']=population['Country'].str.strip(' ').str.lower()

data.loc[data['Country/Region']=='us']='united states'

data.loc[data['Country/Region']=='taiwan*']='taiwan'

data.loc[data['Country/Region']=='congo (kinshasa)']='congo, dem. rep.'

data['days_since_first']=0

for i in data.index[1:]:

    if data.at[i,'match']==True:

        data.loc[i,'days_since_first']=(data.at[i-1,'days_since_first']+1)

        data.loc[i,'increase_rate']=(((data.at[i,'Confirmed']-data.at[i-1,'Confirmed'])/data.at[i-1,'Confirmed'])*100)

        data.loc[i,'previous_Confirmed']=data.at[i-1,'Confirmed']

    else:

        data.set_value(i,'days_since_first',0)
data=data[data['increase_rate']>=0]

data['previous_Confirmed'].astype(int)

del data['match']

data.set_index('Country/Region',inplace=True)

population.set_index('Country',inplace=True)
result = pd.merge(data, population,left_index=True, right_index=True)
result['death_rate']=(result['Deaths']/result['Confirmed'])*100

cols=['Lat','Long','Confirmed','Deaths','Recovered','death_rate']                             

for i in cols:

    result[i]=pd.to_numeric(result[i])
result.info()
viz=result.copy()

viz.reset_index(inplace=True)
viz['conf/1m_popul']=viz['Confirmed']/(viz['Population']/1000000)
plt.figure(figsize=(30, 7))

mcount=viz.groupby('index').tail(1)

mcount = mcount.reset_index().sort_values(by='conf/1m_popul',ascending=False)

mcount=mcount[mcount['conf/1m_popul']>100]

g=sb.barplot(data = mcount, x = 'index',y='conf/1m_popul');

g.set_title('Countries confirmed per 1m population')

g.set_ylabel('conf/1m_popul')

g.set_xlabel('Country');

plt.xticks(rotation=90)
plt.figure(figsize=(30, 7))

mcount=viz.groupby('index').mean()

mcount = mcount.reset_index().sort_values(by='increase_rate',ascending=False)

mcount=mcount[mcount['increase_rate']>30]

g=sb.barplot(data = mcount, x = 'index',y='increase_rate');

g.set_title('Countries mean increase rate')

g.set_ylabel('Mean Increase rate %')

g.set_xlabel('Country');

plt.xticks(rotation=90)
mcount=viz.groupby('index').mean()

mcount = mcount.reset_index()

dcount=viz.groupby('index').tail(1)

dcount = dcount.reset_index()



g = sb.jointplot("GDP ($ per capita)", "increase_rate", data=mcount,

                  kind="reg", truncate=True,

                  xlim=(0, 60000), ylim=(0, 150),

                  color="m", height=7)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Increase rate compared to GDP')



g = sb.jointplot("GDP ($ per capita)", "conf/1m_popul", data=mcount,

                  kind="reg", truncate=True,

                  xlim=(0, 60000), ylim=(0, 1500),

                  color="m", height=7)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Confirmed cases per 1m population compared to GDP')
mcount=viz.groupby('index').tail(1)

mcount = mcount.reset_index().sort_values(by='index',ascending=False)

mcount
plt.figure(figsize=(30, 7))

mcount=viz.groupby('index').tail(1)

mcount = mcount.reset_index().sort_values(by='index',ascending=False)





pal = sb.cubehelix_palette(len(mcount['days_since_first'].unique()), rot=-.5, dark=.3)



# Show each distribution with both violins and points

sb.swarmplot(x="days_since_first", y='conf/1m_popul',

              palette=["r", "c", "y"], data=mcount)
plt.figure(figsize=(30, 7))

mcount=viz.groupby('days_since_first').mean()

mcount = mcount.reset_index().sort_values(by='days_since_first',ascending=False)

g=sb.barplot(data = mcount, x = 'days_since_first',y='increase_rate');

g.set_title('Increase rate after x days of infection')

g.set_ylabel('Mean Increase rate %')

g.set_xlabel('Days since first victim');

plt.xticks(rotation=90)
viz.groupby('index').max()
mcount=viz.groupby('index').mean()

mcount = mcount.reset_index()

dcount=viz.groupby('index').tail(1)

dcount = dcount.reset_index()

g = sb.jointplot("Pop. Density (per sq. mi.)", "increase_rate", data=mcount,

                  kind="reg", truncate=True,

                  xlim=(0, 17000), ylim=(0, 200),

                  color="y", height=7)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Increase rate compared to Pop. Density')



g = sb.jointplot("Industry", "increase_rate", data=mcount,

                  kind="reg", truncate=True,

                  xlim=(0, 1), ylim=(0, 150),

                  color="y", height=7)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Increase rate compared to Industry')



g = sb.jointplot("Pop. Density (per sq. mi.)", "death_rate", data=dcount,

                  kind="reg", truncate=True,

                  xlim=(0, 17000), ylim=(0, 10),

                  color="r", height=7)

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Confirmed cases compared to Pop. Density')
plt.figure(figsize=(30, 7))

mcount=viz.groupby('days_since_first').mean()

mcount = mcount.reset_index().sort_values(by='days_since_first',ascending=False)

g=sb.barplot(data = mcount, x = 'days_since_first',y='death_rate');

g.set_title('death_rate after x days of infection')

g.set_ylabel('death_rate %')

g.set_xlabel('Days since first victim');

plt.xticks(rotation=90)
mcount=viz.groupby('index').tail(1)

mcount = mcount.reset_index().sort_values(by='death_rate',ascending=False)

mcount
plt.figure(figsize=(30, 7))

mcount=viz.groupby('index').tail(1)

mcount = mcount.reset_index().sort_values(by='death_rate',ascending=False)

g=sb.barplot(data = mcount, x = 'index',y='death_rate');

g.set_title('Countries mean death rate')

g.set_ylabel('Mean death rate %')

g.set_xlabel('Country');

g.set(ylim=(0, 40))

g.set(xlim=(0, 40))



plt.xticks(rotation=90)
viz.columns
X=result.copy()

del X['Deaths']

del X['date']

del X['increase_rate']

del X['Recovered']

del X['death_rate']





X=pd.get_dummies(X)
corr = X.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error
y = X.Confirmed     

corr = X.corr()['Confirmed']

corr=abs(corr)

col_to_keep = corr[corr>0.03]

for i in X.columns:

    if not i in col_to_keep:

        X.drop(i, axis=1,inplace=True)



X.drop(['Confirmed'], axis=1, inplace=True)





# Break off validation set from training data



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, test_size=0.1, random_state=1)



X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.9, test_size=0.1, random_state=1)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,

                           colsample_bytree=1, max_depth=7)
eval_set = [(X_val.as_matrix(), y_val.as_matrix())]

xgb.fit(X_train.as_matrix(),y_train.as_matrix(), early_stopping_rounds=50, eval_metric="mae", eval_set=eval_set, verbose=True)
from sklearn.metrics import explained_variance_score

predictions = xgb.predict(X_test.as_matrix())

gr=X.copy()

gr=gr.reset_index()

gr=gr[gr['index']=='greece']

del gr['index']

#gr.drop(['Confirmed'], axis=1, inplace=True)

a=data.copy()

a=a.reset_index()

#a['date']=pd.to_datetime(a['date'])

#a=a['Country/Region'=='greece']
gr
future_gr=gr.copy()

future_gr=future_gr[X_test.columns]
future_gr.columns,X_test.columns
future_gr.reset_index(inplace=True)

del future_gr['index']



future_gr.iloc[0]=gr.iloc[-1]

#future_gr['date']=last_date

for i in future_gr.index[1:]:

    future_gr.loc[i,'days_since_first']=(future_gr.at[i-1,'days_since_first']+1)





predictions=[]

for i in future_gr.index:

    pred=int(xgb.predict(future_gr.iloc[i]))

    future_gr.loc[i+1,'previous_Confirmed']=pred

    predictions.append(pred)
future_gr['predicted_infections']=future_gr['previous_Confirmed'].shift(-1)

future_gr=future_gr[{'days_since_first', 'previous_Confirmed','predicted_infections'}]
future_gr['increase_rate']=0
future_gr['increase_rate']=((future_gr['predicted_infections']-future_gr['previous_Confirmed'])/future_gr['previous_Confirmed'])*100
future_gr['date']=''

future_gr.loc[0,'date']='2020-03-22'

future_gr['date']=pd.to_datetime(future_gr['date'])

for i in future_gr.index[1:]:

    future_gr.loc[i,'date']=(future_gr.at[i-1,'date'] + datetime.timedelta(days=1))
del future_gr['previous_Confirmed']

future_gr
plt.figure(figsize = [16, 8])

sb.set(style="whitegrid")

sb.lineplot(x='date',y='increase_rate',data=future_gr[:-2], palette="tab10", linewidth=2.5).set_title('Prediction of increase rate in Greece')

plt.xticks(rotation=90);

plt.figure(figsize = [16, 8])

sb.set(style="whitegrid")

sb.lineplot(x='date',y='predicted_infections',data=future_gr[:-2], palette="tab10", linewidth=2.5).set_title('Prediction of number of infections in Greece')



plt.xticks(rotation=90);



plt.show()