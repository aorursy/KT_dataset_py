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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sns

import pickle

from sklearn.preprocessing import MinMaxScaler,StandardScaler
def plot_series(country_df,province=None):

    if province!=None:

        country_df = country_df[country_df['Province_State']==province]

    plt.plot(country_df.index, country_df[['ConfirmedCases']], 'r')

    plt.title('ConfirmedCases')

    plt.ylabel('Count')

    plt.show()
population = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")

population.head()
df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

df.head()
df.shape
all_countries_list = df['Country_Region'].unique().tolist()
india = df[df['Country_Region']=='India']

plot_series(india)
usa = df[df['Country_Region']=='US']

plot_series(usa,'Alabama')
show_cum = df.groupby(by='Country_Region')[['ConfirmedCases','Fatalities']].max().reset_index()

plt.figure(figsize=(20,10))

sns.barplot(x='ConfirmedCases',y='Country_Region',data=show_cum[show_cum['ConfirmedCases'] != 0].sort_values(by='ConfirmedCases',ascending=False).head(30))
show_cum = df.groupby(by='Country_Region')[['ConfirmedCases','Fatalities']].max().reset_index()

plt.figure(figsize=(20,10))

sns.barplot(x='ConfirmedCases',y='Country_Region',data=show_cum[show_cum['Fatalities'] != 0].sort_values(by='Fatalities',ascending=False).head(30))
def preprocess(df,ple=None,cle=None,scaler=None):

    df['Province_State'] = df['Province_State'].replace(np.NaN,df['Province_State'].mode()[0])

    df['Country_Region'] = df['Country_Region'].replace(np.NaN,df['Country_Region'].mode()[0])

    df['Date'] = df['Date'].replace(np.NaN,'2020-01-21')

    df = df.replace(np.NaN,-1)

    df = df.replace('N.A.',-1)

    df['month'] = pd.DatetimeIndex(df['Date']).month

    df['day'] = pd.DatetimeIndex(df['Date']).day

    df['year'] = pd.DatetimeIndex(df['Date']).year

    df['Days'] = df['Date'].apply(lambda x: datetime.strptime(str(x),'%Y-%m-%d')) - datetime.strptime('2020-01-21', '%Y-%m-%d')

    df['Days'] = df['Days'].apply(lambda x: x.days)

    

    if 'Id' not in df.columns.tolist():

        df['Id'] = df['ForecastId']

        df.drop(columns=['ForecastId'],inplace=True,axis=1)

    df.drop(columns=['Id','Date'],inplace=True,axis=1)

    

    df['Province_State'] = df['Province_State'].replace(np.NaN,'')

    if ple==None and cle==None:

        from sklearn.preprocessing import LabelEncoder

        ple = LabelEncoder()

        ple.fit(df['Province_State'])

        pickle.dump(ple,open('ple.pkl','wb'))

        p_df = ple.transform(df['Province_State'])

        df['Province_State'] = p_df

        

        cle = LabelEncoder()

        cle.fit(df['Country_Region'])

        pickle.dump(cle,open('cle.pkl','wb'))

        c_df = cle.transform(df['Country_Region'])

        df['Country_Region'] = c_df

    else:

        df['Province_State'] = ple.transform(df['Province_State'])

        df['Country_Region'] = cle.transform(df['Country_Region'])

    if scaler==None:

        sc = MinMaxScaler()

        sc.fit(df)

        pickle.dump(sc,open('covid_scaler.pkl','wb'))

        df = pd.DataFrame(sc.transform(df),columns=df.columns.tolist())

    else:

        df = pd.DataFrame(scaler.transform(df),columns=df.columns.tolist())

    return df
# y = df[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log(x))

# y.replace([np.inf, -np.inf], 0, inplace=True)

y1 = df['ConfirmedCases']

y2 = df['Fatalities']

X = df.drop(columns=['ConfirmedCases','Fatalities'],axis=1)
X.head()
y1 = y1.replace(np.NaN,-1)

y2 = y2.replace(np.NaN,-1)
X = preprocess(X)

X.head()
from sklearn.ensemble import RandomForestRegressor

model1 = RandomForestRegressor()

model1.fit(X, y1)



model2 = RandomForestRegressor()

model2.fit(X, y2)
ple = pickle.load(open('ple.pkl','rb'))

cle = pickle.load(open('cle.pkl','rb'))
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

test.head()
test = preprocess(test,ple,cle,pickle.load(open('covid_scaler.pkl','rb')))

test.head()
test.shape
pred1 = model1.predict(test)

pred2 = model2.predict(test)

submission = pd.DataFrame()

submission['ForecastId'] = np.arange(start=1,stop=len(pred1)+1,step=1)

submission['ConfirmedCases'] = pred1

submission['Fatalities'] = pred2
submission.to_csv('submission.csv',index=False)