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
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

import pandas as pd

from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
path = "/kaggle/input/covid19-global-forecasting-week-5/"

cpc = ['County','Province_State','Country_Region']
#Load the data

dftrain = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

dftest = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")

dfsub = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/submission.csv")

dftr = pd.read_csv(path+"train.csv" , parse_dates = ['Date'])

dftt = pd.read_csv(path+"test.csv" , parse_dates = ['Date'])
trmindate = dftr['Date'].min()

dftr['ndays']= (dftr['Date'] - trmindate).dt.days

dftt['ndays'] = (dftt['Date'] - trmindate).dt.days
#dftr[(dftr['Date'] == "2020-04-27") & (dftr['Country_Region'] == 'Afghanistan')]

# df_train['Date'] = pd.to_datetime(df_train['Date'], infer_datetime_format=True)

# df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)

# df_train['Date'] = df_train['Date'].apply(lambda s: time.mktime(s.timetuple()))

# df_test['Date'] = df_test['Date'].apply(lambda s: time.mktime(s.timetuple()))

# min_timestamp = np.min(df_train['Date'])

# df_train['Date'] = df_train['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)

# df_test['Date'] = df_test['Date'].apply(lambda s: (s - min_timestamp) / 86400.0)

# df_train.tail()
dftr[dftr['Country_Region'] == 'Afghanistan']['Weight'].value_counts()
dftr.isnull().sum()
dftr[cpc] = dftr[cpc].fillna('')

dftt[cpc] = dftt[cpc].fillna('')
dftt.isnull().sum()
dftr['Date'].describe()
dftr['Loc'] = dftr['Country_Region'] + ' ' + dftr['Province_State'] + ' ' +  dftr['County']

dftr['Loc'] = dftr['Loc'].str.strip()

dftr['Loc'].value_counts()

dftt['Loc'] = dftt['Country_Region'] + ' ' + dftt['Province_State'] + ' ' +  dftt['County']

dftt['Loc'] = dftt['Loc'].str.strip()

dftt['Loc'].value_counts()
# dftr['Datei'] = dftr.Date.dt.strftime("%m%d")

# dftr['Datei'] = dftr['Datei'].astype(int)

# dftt['Datei'] = dftt.Date.dt.strftime("%m%d")

# dftt['Datei'] = dftt['Datei'].astype(int)
dftr.shape, dftt.shape
train_columns = ['Id','Loc','Population','Weight','Date','ndays','Target','TargetValue']

test_columns = ['ForecastId','Loc','Population','Weight','Date','ndays','Target']
dftr1= dftr[train_columns]

dftt1 = dftt[test_columns]
le = preprocessing.LabelEncoder()

dftr1.Loc = le.fit_transform(dftr1.Loc)

dftt1.Loc = le.fit_transform(dftt1.Loc)

dftrc = dftr1[dftr1.Target == 'ConfirmedCases']

dftrc = dftrc.drop('Target', axis=1)

dftrc = dftrc.rename(columns={'TargetValue':'ConfirmedCases'})



dftrf = dftr1[dftr1.Target == 'Fatalities']

dftrf = dftrf.drop('Target',axis=1)

dftrf = dftrf.rename(columns={'TargetValue':'Fatalities'})



dfttc = dftt1[dftt1.Target == 'ConfirmedCases']

dfttc = dfttc.drop('Target', axis=1)



dfttf = dftt1[dftt1.Target == 'Fatalities']

dfttf = dfttf.drop('Target',axis=1)
Xtrainc = dftrc[['Loc','Population','Weight','ndays']]

yc = dftrc.ConfirmedCases

Xtrainf = dftrf[['Loc','Population','Weight','ndays']]

yf = dftrf.Fatalities
XttFCid = dfttc['ForecastId']

XttFFid = dfttf['ForecastId']
XttC = dfttc[['Loc','Population','Weight','ndays']]

XttF = dfttf[['Loc','Population','Weight','ndays']]
xoutc = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': []})

xoutf = pd.DataFrame({'ForecastId': [], 'Fatalities': []})

l_alpha = 0.05

m_alpha = 0.50

u_alpha = 0.95
l_modelc = GradientBoostingRegressor(loss="quantile", alpha=l_alpha,max_depth = 6,

                                     learning_rate=0.2, random_state=47,n_estimators=100)

m_modelc = GradientBoostingRegressor(loss="quantile", alpha=m_alpha, max_depth = 6,

                                     learning_rate=0.2, random_state=47,n_estimators=100)

u_modelc = GradientBoostingRegressor(loss="quantile", alpha=u_alpha,max_depth = 6, 

                                     learning_rate=0.2, random_state=47,n_estimators=100)

        

l_modelc.fit(Xtrainc, yc)

m_modelc.fit(Xtrainc, yc)

u_modelc.fit(Xtrainc, yc)



l_predict = l_modelc.predict(XttC)

m_predict = m_modelc.predict(XttC)

u_predict = u_modelc.predict(XttC)

l_data = pd.DataFrame({'ForecastId': [str(i)+'_0.05' for i in XttFCid.values.tolist()], 'ConfirmedCases': l_predict})

l_data['ConfirmedCases'] = 1.0

xoutc = xoutc.append(l_data)

m_data = pd.DataFrame({'ForecastId': [str(i)+'_0.5' for i in XttFCid.values.tolist()], 'ConfirmedCases': m_predict})

xoutc = xoutc.append(m_data)

u_data = pd.DataFrame({'ForecastId': [str(i)+'_0.95' for i in XttFCid.values.tolist()], 'ConfirmedCases': u_predict})

xoutc = xoutc.append(u_data)



xoutc.reset_index(inplace=True)

xoutc = xoutc.rename(columns={'ConfirmedCases':'PValue'})
#l_data.head()

#u_data['ConfirmedCases'].value_counts()

# cols = ['T1','T2','T3','T4']

# df[df[cols] < 0] = -5

#l_data[l_data['ConfirmedCases'] == 0.0] =1.0

#l_data['ConfirmedCases'] = 1.0

l_data.head()
l_modelf = GradientBoostingRegressor(loss="quantile", alpha=l_alpha,max_depth = 6,

                                     learning_rate=0.1, random_state=47,n_estimators=100)

m_modelf = GradientBoostingRegressor(loss="quantile", alpha=m_alpha, max_depth = 6,

                                     learning_rate=0.1, random_state=47,n_estimators=100)

u_modelf = GradientBoostingRegressor(loss="quantile", alpha=u_alpha, max_depth = 6,

                                     learning_rate=0.1, random_state=47,n_estimators=100)

l_modelf.fit(Xtrainf, yf)

m_modelf.fit(Xtrainf, yf)

u_modelf.fit(Xtrainf, yf)



l_predictf = l_modelf.predict(XttF)

m_predictf = m_modelf.predict(XttF)

u_predictf = u_modelf.predict(XttF)

l_dataf = pd.DataFrame({'ForecastId': [str(i)+'_0.05' for i in XttFFid.values.tolist()], 'Fatalities': l_predictf})

num = l_dataf._get_numeric_data()

num[num<0] = 0

xoutf = xoutf.append(l_dataf)

m_dataf = pd.DataFrame({'ForecastId': [str(i)+'_0.5' for i in XttFFid.values.tolist()], 'Fatalities': m_predictf})

xoutf = xoutf.append(m_dataf)

u_dataf = pd.DataFrame({'ForecastId': [str(i)+'_0.95' for i in XttFFid.values.tolist()], 'Fatalities': u_predictf})

xoutf = xoutf.append(u_dataf)



xoutf.reset_index(inplace=True)

xoutf = xoutf.rename(columns={'Fatalities':'PValue'})
l_dataf['Fatalities'].value_counts()

#df.loc[df.Weight == "155", "Name"] = "John"

#num = l_dataf._get_numeric_data()

#num[num<0] = 0

#xoutf.head()
dfs = pd.concat([xoutc, xoutf], ignore_index=True)

dfs = dfs.drop('index', axis=1)

dfs = dfs.rename(columns={'ForecastId':'ForecastId_Quantile'})

dfsss = dfsub.merge(dfs, on='ForecastId_Quantile', how='inner')

dfsss = dfsss.drop("TargetValue", axis=1)

dfsss = dfsss.rename(columns = ({'PValue':'TargetValue'}))

dfsss['TargetValue'] = dfsss['TargetValue'].astype(int)
submit =dfsss.to_csv('submission.csv', index=False)