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
cwd = os.getcwd()

print(cwd)
Train =pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

Test=pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

Useful = pd.read_csv("../input/covid19-useful-features-by-country/Countries_usefulFeatures.csv")
Train
Test
Useful
Train = pd.merge(Train,Useful[['Country_Region','Mean_Age','Lockdown_Date','Date_FirstConfirmedCase','Country_Code']],on= 'Country_Region', how='left')

Train
Train['Country_Code'] = Train.iloc[:,9].str.slice(0, -1)
Capacity = pd.read_csv("../input/hospital-bed/hospital_beds_global_v1.csv")

Capacity
Train = pd.merge(Train,Capacity[['Country_Code','beds']],on= 'Country_Code', how='left')

Train
Rule = pd.read_csv("../input/roflaw/Rule of Law.csv")

Rule
Train = pd.merge(Train,Rule[['Country_Region','Rule of Law']],on= 'Country_Region', how='left')

Train
Train["UniqueRegion"]=Train.Country_Region

Train.UniqueRegion[Train.Province_State.isna()==False]=Train.Province_State+" , "+Train.Country_Region

Train[Train.Province_State.isna()==False]

Train
Train['Date'] = pd.to_datetime(Train.Date,format='%Y/%m/%d')

Train['Lockdown_Date'] = pd.to_datetime(Train.Lockdown_Date,format='%Y/%m/%d')

Train['Date_FirstConfirmedCase'] = pd.to_datetime(Train.Date_FirstConfirmedCase,format='%Y/%m/%d')
Train['DaysSinceLockdown'] =  (Train['Date'] - Train['Lockdown_Date']).dt.days

Train.loc[Train['Date'] < Train['Lockdown_Date'], 'BFAFLockdown'] = 'Before'

Train.loc[Train['Date'] >= Train['Lockdown_Date'], 'BFAFLockdown'] = 'After'

Train['Infected'] = Train['Date'] >= Train['Date_FirstConfirmedCase']

Train
Train['AfterLockdown'] = 0 

Train.AfterLockdown[Train['BFAFLockdown'] == 'After'] = 1

Train
import statsmodels

from statsmodels.tsa.stattools import adfuller

from matplotlib import pyplot
series = Train.ConfirmedCases[Train.UniqueRegion == 'Thailand']

series.plot()

pyplot.show()
series = Train.Fatalities[Train.UniqueRegion == 'Thailand']

series.plot()

pyplot.show()
diffseries = series.diff()

diffseries.plot()

pyplot.show()
seconddiffseries = diffseries.diff()

seconddiffseries.plot()

pyplot.show()
Train['FirstDiffCF'] = Train.groupby(['UniqueRegion']).ConfirmedCases.diff()

Train['2ndDiffCF'] = Train.groupby(['UniqueRegion']).FirstDiffCF.diff()

Train
Train['FirstDiffFT'] = Train.groupby(['UniqueRegion']).Fatalities.diff()

Train['2ndDiffFT'] = Train.groupby(['UniqueRegion']).FirstDiffFT.diff()

Train
!pip install --upgrade pip
!pip install pmdarima
!pip install --upgrade pmdarima
import pmdarima as pm

from pmdarima import model_selection

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
RMSE_ARIMA = []

obs = []

for country in Train.UniqueRegion.unique():

    Data = Train[(Train['UniqueRegion'] == country) & (Train['Infected'] == True)]

    if not Data.empty:

        train, test = model_selection.train_test_split(Data, train_size=0.8, test_size = 0.2)

        modl = pm.auto_arima(train['ConfirmedCases'], start_p=0, start_q=0, start_P=0, start_Q=0,

                        max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,

                        stepwise=True, suppress_warnings=True, max_d=3,

                        error_action='ignore')

    # Create predictions for the future, evaluate on test

        preds, conf_int = modl.predict(n_periods=test.shape[0], return_conf_int=True)

        A = np.sqrt(mean_squared_error(test['ConfirmedCases'], preds))

        RMSE_ARIMA.append(A)

        B  = test.shape[0]

        obs.append(B)
sum(RMSE_ARIMA)/sum(obs)
RMSE_ARIMA = []

obs = []

for country in Train.UniqueRegion.unique():

    Data = Train[(Train['UniqueRegion'] == country) & (Train['Fatalities'] > 0)]

    if len(Data) > 1:

        train, test = model_selection.train_test_split(Data, train_size=0.8, test_size = 0.2)

        modl = pm.auto_arima(train['Fatalities'], start_p=0, start_q=0, start_P=0, start_Q=0,

                        max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,

                        stepwise=True, suppress_warnings=True, max_d=3,

                        error_action='ignore')

    # Create predictions for the future, evaluate on test

        preds, conf_int = modl.predict(n_periods=test.shape[0], return_conf_int=True)

        A = np.sqrt(mean_squared_error(test['Fatalities'], preds))

        RMSE_ARIMA.append(A)

        B  = test.shape[0]

        obs.append(B)
sum(RMSE_ARIMA)/sum(obs)
RMSE_ARIMAX = []

obs = []

for country in Train.UniqueRegion.unique():

    Data = Train[(Train['UniqueRegion'] == country) & (Train['Infected'] == True)]

    Data = Data.dropna(axis = 0, subset = ['DaysSinceLockdown'])

    if not Data.empty:

        train, test = model_selection.train_test_split(Data, train_size=0.8, test_size = 0.2)

        exo_train = train['DaysSinceLockdown']

        exo_test = test['DaysSinceLockdown']

        exo_train = exo_train.values.reshape(-1,1)

        exo_test = exo_test.values.reshape(-1,1)



        modl = pm.auto_arima(train['ConfirmedCases'], exogenous = exo_train ,start_p=0, start_q=0, start_P=0, start_Q=0,

                        max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,

                        stepwise=True, suppress_warnings=True, max_d=3,

                        error_action='ignore')

    # Create predictions for the future, evaluate on test

        preds = modl.predict(n_periods=test.shape[0],exogenous = exo_test)

        A = np.sqrt(mean_squared_error(test['ConfirmedCases'], preds))

        RMSE_ARIMAX.append(A)

        B  = test.shape[0]

        obs.append(B)
sum(RMSE_ARIMAX)/sum(obs)
RMSE_ARIMAX = []

obs = []

for country in Train.UniqueRegion.unique():

    Data = Train[(Train['UniqueRegion'] == country) & (Train['Fatalities'] > 0)]

    Data = Data.dropna(axis = 0, subset = ['2ndDiffCF'])

    if len(Data) > 5:

        train, test = model_selection.train_test_split(Data, train_size=0.8, test_size = 0.2)

        exo_train = train['2ndDiffCF']

        exo_test = test['2ndDiffCF']

        exo_train = exo_train.values.reshape(-1,1)

        exo_test = exo_test.values.reshape(-1,1)



        modl = pm.auto_arima(train['Fatalities'], exogenous = exo_train ,start_p=0, start_q=0, start_P=0, start_Q=0,

                        max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,

                        stepwise=True, suppress_warnings=True, max_d=3,

                        error_action='ignore')

    # Create predictions for the future, evaluate on test

        preds = modl.predict(n_periods=test.shape[0],exogenous = exo_test)

        A = np.sqrt(mean_squared_error(test['Fatalities'], preds))

        RMSE_ARIMAX.append(A)

        B  = test.shape[0]

        obs.append(B)
sum(RMSE_ARIMAX)/sum(obs)
Train['logConfirmedCases'] = np.log(Train.ConfirmedCases)

Train
Train['logFatalities'] = np.log(Train.Fatalities)

Train
RMSE_ARIMA = []

obs = []

for country in Train.UniqueRegion.unique():

    Data = Train[(Train['UniqueRegion'] == country) & (Train['Infected'] == True) & (Train['logConfirmedCases'] != np.inf) & (Train['logConfirmedCases'] != -np.inf)]

    if not Data.empty:

        train, test = model_selection.train_test_split(Data, train_size=0.8, test_size = 0.2)

        modl = pm.auto_arima(train['logConfirmedCases'], start_p=0, start_q=0, start_P=0, start_Q=0,

                        max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,

                        stepwise=True, suppress_warnings=True, max_d=3,

                        error_action='ignore')

    # Create predictions for the future, evaluate on test

        preds, conf_int = modl.predict(n_periods=test.shape[0], return_conf_int=True)

        A = np.sqrt(mean_squared_error(np.exp(test['logConfirmedCases']), np.exp(preds)))

        RMSE_ARIMA.append(A)

        B  = test.shape[0]

        obs.append(B)
sum(RMSE_ARIMA)/sum(obs)
RMSE_ARIMA = []

obs = []

for country in Train.UniqueRegion.unique():

    Data = Train[(Train['UniqueRegion'] == country) & (Train['logFatalities'] != np.inf) & (Train['logFatalities'] != -np.inf)]

    if len(Data) > 1:

        train, test = model_selection.train_test_split(Data, train_size=0.8, test_size = 0.2)

        modl = pm.auto_arima(train['logFatalities'], start_p=0, start_q=0, start_P=0, start_Q=0,

                        max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,

                        stepwise=True, suppress_warnings=True, max_d=3,

                        error_action='ignore')

    # Create predictions for the future, evaluate on test

        preds, conf_int = modl.predict(n_periods=test.shape[0], return_conf_int=True)

        A = np.sqrt(mean_squared_error(np.exp(test['logFatalities']), np.exp(preds)))

        RMSE_ARIMA.append(A)

        B  = test.shape[0]

        obs.append(B)
sum(RMSE_ARIMA)/sum(obs)
RMSE_ARIMAX = []

obs = []

for country in Train.UniqueRegion.unique():

    Data = Train[(Train['UniqueRegion'] == country) & (Train['Infected'] == True) & (Train['logConfirmedCases'] != np.inf) & (Train['logConfirmedCases'] != -np.inf)]

    Data = Data.dropna(axis = 0, subset = ['DaysSinceLockdown'])

    if not Data.empty:

        train, test = model_selection.train_test_split(Data, train_size=0.8, test_size = 0.2)

        exo_train = train['DaysSinceLockdown']

        exo_test = test['DaysSinceLockdown']

        exo_train = exo_train.values.reshape(-1,1)

        exo_test = exo_test.values.reshape(-1,1)



        modl = pm.auto_arima(train['logConfirmedCases'], exogenous = exo_train ,start_p=0, start_q=0, start_P=0, start_Q=0,

                        max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,

                        stepwise=True, suppress_warnings=True, max_d=3,

                        error_action='ignore')

    # Create predictions for the future, evaluate on test

        preds = modl.predict(n_periods=test.shape[0],exogenous = exo_test)

        A = np.sqrt(mean_squared_error(np.exp(test['logConfirmedCases']), np.exp(preds)))

        RMSE_ARIMAX.append(A)

        B  = test.shape[0]

        obs.append(B)
sum(RMSE_ARIMAX)/sum(obs)
!pip install linearmodels
from linearmodels.panel import RandomEffects

from linearmodels import PanelOLS
Train['lag2ndDiffCF'] = Train['2ndDiffCF'].shift(1)

Train['laglag2ndDiffCF'] = Train['lag2ndDiffCF'].shift(1)

Train['lag2ndDiffFT'] = Train['2ndDiffFT'].shift(1)

Train['laglag2ndDiffFT'] = Train['lag2ndDiffFT'].shift(1)

Train
TrainPanel = Train

TrainPanel = TrainPanel.set_index(["UniqueRegion","Date"])

TrainPanel
DataCC = TrainPanel.dropna(subset=['Infected','2ndDiffCF', 'DaysSinceLockdown','lag2ndDiffCF','laglag2ndDiffCF','Rule of Law','AfterLockdown'])

DataCC = DataCC[['DaysSinceLockdown','Infected','2ndDiffCF','lag2ndDiffCF','laglag2ndDiffCF','Rule of Law','AfterLockdown']]

DataCC
DataCC.info()
CCFixed = PanelOLS(DataCC[['2ndDiffCF']], DataCC[['lag2ndDiffCF','laglag2ndDiffCF','DaysSinceLockdown','Infected']], entity_effects=True)

CCFixedfit = CCFixed.fit(cov_type='clustered', cluster_entity=True)

CCFixedfit
CCRandom = RandomEffects(DataCC[['2ndDiffCF']], DataCC[['DaysSinceLockdown','Infected','lag2ndDiffCF','laglag2ndDiffCF','Rule of Law']])

CCRandomfit = CCRandom.fit(cov_type='robust')

CCRandomfit
DataFT = TrainPanel.dropna(subset=['2ndDiffFT','Mean_Age','2ndDiffCF','beds'])

DataFT = DataFT[['2ndDiffFT','Mean_Age','2ndDiffCF','beds']]

DataFT
FTFixed = PanelOLS(DataFT[['2ndDiffFT']], DataFT[['2ndDiffCF']], entity_effects=True)

FTFixedfit = FTFixed.fit()

FTFixedfit
FTRandom = RandomEffects(DataFT[['2ndDiffFT']], DataFT[['Mean_Age','2ndDiffCF','beds']])

FTRandomfit = FTRandom.fit()

FTRandomfit
Test
Test['Date'] = pd.to_datetime(Test.Date,format='%Y/%m/%d')

Test["UniqueRegion"]=Test.Country_Region

Test.UniqueRegion[Test.Province_State.isna()==False]=Test.Province_State+" , "+Test.Country_Region

Test[Test.Province_State.isna()==False]

Test
TrainCopy = Train[Train.Date < '2020-04-02']

TrainCopy
Train_dates=list(TrainCopy.Date.unique())

Test_dates=list(Test.Date.unique())
# Dates in train only

only_train_dates=set(Train_dates)-set(Test_dates)

print("Only train dates : ",len(only_train_dates))

#dates in train and test

intersection_dates=set(Test_dates)&set(Train_dates)

print("Intersection dates : ",len(intersection_dates))

#dates in only test

only_test_dates=set(Test_dates)-set(Train_dates)

print("Only Test dates : ",len(only_test_dates))
print(f" Periodes to predict ahead : {len(Test)/len(Test.UniqueRegion.unique())}")
Test = pd.merge(Test,Useful[['Country_Region','Lockdown_Date']],on= 'Country_Region', how='left')

Test['Lockdown_Date'] = pd.to_datetime(Test.Lockdown_Date,format='%Y/%m/%d')
Test['DaysSinceLockdown'] =  (Test['Date'] - Test['Lockdown_Date']).dt.days

TrainCopy.DaysSinceLockdown[Train['DaysSinceLockdown'].isna() == True] = 0

Test.DaysSinceLockdown[Test['DaysSinceLockdown'].isna() == True] = 0

TrainCopy.info()
TrainCopy = TrainCopy[TrainCopy.UniqueRegion != 'Burundi']

TrainCopy = TrainCopy[TrainCopy.UniqueRegion != 'Sierra Leone']
CCPred = pd.DataFrame(columns=["UniqueRegion","Date","ConfirmedCases"])

for country in TrainCopy.UniqueRegion.unique():

    Data = TrainCopy[(TrainCopy['UniqueRegion'] == country) & (TrainCopy['Infected'] == True)]

    if not Data.empty:

        exo = Data['DaysSinceLockdown']

        exo = exo.values.reshape(-1,1)

        exo_test = Test.DaysSinceLockdown[Test['UniqueRegion'] == country]

        exo_test = exo_test.values.reshape(-1,1)

        modl = pm.auto_arima(Data['ConfirmedCases'], exogenous = exo ,start_p=0, start_q=0, start_P=0, start_Q=0,

                        max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,

                        stepwise=True, suppress_warnings=True, max_d=5, max_D=5,

                        error_action='ignore')

    # Create predictions for the future 43 periods

        preds = modl.predict(n_periods= 43,exogenous = exo_test)

        UniqueRegion = [country]*43

        preds = preds.tolist()

        Date = Test.Date.unique()

        df_temp = pd.DataFrame(list(zip(UniqueRegion, Date, preds)), columns =['UniqueRegion', 'Date','ConfirmedCases']) 

        CCPred = pd.concat([CCPred,df_temp])
CCPred
Test = pd.merge(Test,CCPred[['UniqueRegion','Date','ConfirmedCases']],on= ['UniqueRegion','Date'], how='left')

Test
CCPred['PredFirstDiffCF'] = CCPred.groupby(['UniqueRegion']).ConfirmedCases.diff()

CCPred['Pred2ndDiffCF'] = CCPred.groupby(['UniqueRegion']).PredFirstDiffCF.diff()

CCPred
Temp = Train[(Train.Date > '2020-04-01') & (Train.Date < '2020-04-04')]

Temp 
CCPred = pd.merge(CCPred,Temp[['UniqueRegion','Date','2ndDiffCF']],on= ['UniqueRegion','Date'], how='left')

CCPred
CCPred['Pred2ndDiffCF'] = CCPred['Pred2ndDiffCF'].fillna(0)

CCPred['2ndDiffCF'] = CCPred['2ndDiffCF'].fillna(0)

CCPred['SecondDiffCC'] = CCPred['Pred2ndDiffCF'] + CCPred['2ndDiffCF']

CCPred = CCPred.drop(['Pred2ndDiffCF','2ndDiffCF'], axis = 1)

CCPred
TrainCopy = TrainCopy[TrainCopy.UniqueRegion != 'Bahamas']

TrainCopy = TrainCopy[TrainCopy.UniqueRegion != 'MS Zaandam']

TrainCopy = TrainCopy[TrainCopy.UniqueRegion != 'Sint Maarten , Netherlands']

TrainCopy = TrainCopy[TrainCopy.UniqueRegion != 'Senegal']

TrainCopy = TrainCopy[TrainCopy.UniqueRegion != 'Isle of Man , United Kingdom']
FTPred = pd.DataFrame(columns=["UniqueRegion","Date","Fatalities"])

for country in TrainCopy.UniqueRegion.unique():

    Data = TrainCopy[(TrainCopy['UniqueRegion'] == country) & (TrainCopy['Fatalities'] > 0)]

    Data = Data.dropna(axis = 0, subset = ['2ndDiffCF'])

    if len(Data) > 0:

        exo = Data['2ndDiffCF']

        exo = exo.values.reshape(-1,1)

        exo_test = CCPred.SecondDiffCC[CCPred['UniqueRegion'] == country]

        exo_test = exo_test.values.reshape(-1,1)

        modl = pm.auto_arima(Data['Fatalities'], exogenous = exo ,start_p=0, start_q=0, start_P=0, start_Q=0,

                        max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,

                        stepwise=True, suppress_warnings=True, max_d=5, max_D=5,

                        error_action='ignore')

    # Create predictions for the future 43 periods

        preds = modl.predict(n_periods= 43,exogenous = exo_test)

        UniqueRegion = [country]*43

        preds = preds.tolist()

        Date = Test.Date.unique()

        df_temp = pd.DataFrame(list(zip(UniqueRegion, Date, preds)), columns =['UniqueRegion', 'Date','Fatalities']) 

        FTPred = pd.concat([FTPred,df_temp])
FTPred
Prediction = pd.merge(CCPred,FTPred[['UniqueRegion','Date','Fatalities']],on= ['UniqueRegion','Date'], how='left')

Prediction
Test = pd.merge(Test,FTPred[['UniqueRegion','Date','Fatalities']],on= ['UniqueRegion','Date'], how='left')

Test
Test = pd.merge(Test,Train[['UniqueRegion','Date','Fatalities','ConfirmedCases']],on= ['UniqueRegion','Date'], how='left')

Test
Test[['ConfirmedCases_x', 'Fatalities_x', 'Fatalities_y', 'ConfirmedCases_y']] = Test[['ConfirmedCases_x', 'Fatalities_x', 'Fatalities_y', 'ConfirmedCases_y']].fillna(0)

Test.ConfirmedCases_x[Test.ConfirmedCases_x.isna()==True] = Test.ConfirmedCases_y

Test.Fatalities_x[Test.Fatalities_x.isna()==True] = Test.Fatalities_y

Test.info()
Test
Test = Test.drop(['Province_State','Country_Region','Date','UniqueRegion','Lockdown_Date','DaysSinceLockdown','Fatalities_y','ConfirmedCases_y'], axis = 1)

Test
Test.columns = ['ForecastId','ConfirmedCases','Fatalities']

Test
Test.to_csv("submission.csv", index=None)