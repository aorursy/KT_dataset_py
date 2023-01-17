# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Train=pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')
Train['Lat'].fillna(12.5,inplace=True)

Train['Long'].fillna(-70.0,inplace=True)
Train['Province/State'].fillna("NA",inplace=True)
Lat_longs=Train[['Province/State', 'Country/Region','Lat','Long']].groupby(['Country/Region','Province/State']).mean()

plt.scatter(Lat_longs['Lat'],Lat_longs['Long'],linewidths=.001)
from sklearn.cluster import KMeans

clusters=np.arange(2,21,1)

wss=[]

for i in clusters:

    cl=KMeans(n_clusters=i,max_iter=200)

    cl.fit(Lat_longs)

    wss.append(cl.inertia_)
plt.plot(wss)
cl_4=KMeans(n_clusters=4,max_iter=2000)

cl_4.fit(Lat_longs)

Lat_longs['Clusters']=cl_4.predict(Lat_longs)+1

plt.scatter('Lat','Long',linewidths=.001,c='Clusters',data=Lat_longs)
Train['Clusters']=cl_4.predict(Train[['Lat','Long']])+1

Train['Date']=pd.to_datetime(Train['Date'])

Regions=Lat_longs.index

l=len(Regions)

l
First_ConfirmedCases=[]

First_Fatalities=[]

for i in range(l):

    temp_dataset=Train[(Train['Country/Region']==Regions[i][0]) & (Train['Province/State']==Regions[i][1])]

    First_ConfirmedCases.append(temp_dataset['Date'][temp_dataset['ConfirmedCases']>0].min())

    First_Fatalities.append(temp_dataset['Date'][temp_dataset['Fatalities']>0].min())

Country=[]

State=[]

for i in range(l):

    Country.append(Regions[i][0])

    State.append(Regions[i][1])

#pd.Series(First_Fatalities)

Dates_cases_Fatalities=pd.DataFrame({'Country/Region':Country,

                                     'Province/State':State,

                                     'First_ConfirmedCases':First_ConfirmedCases,

                                     'First_Fatalities':First_Fatalities})

#Train['First_ConfirmedCases'].fillna(Train['Date'].max(),inplace=True)

#Train['First_Fatalities'].fillna(Train['Date'].max(),inplace=True)

Train=Train.merge(Dates_cases_Fatalities)
Train['Date_After_First_Fatalities']=Train['Date']-Train['First_Fatalities']

Train['Date_After_First_ConfirmedCases']=Train['Date']-Train['First_ConfirmedCases']

Train['Date_After_First_Fatalities']=Train['Date_After_First_Fatalities'].dt.days

Train['Date_After_First_ConfirmedCases']=Train['Date_After_First_ConfirmedCases'].dt.days

Train['Date_After_First_Fatalities'][Train['Date_After_First_Fatalities']<0]=0

Train['Date_After_First_ConfirmedCases'][Train['Date_After_First_ConfirmedCases']<0]=0
Train.sort_values(by='Date',inplace=True)

Train_7=Train.iloc[0*284:(59-7)*284]

Train_6=Train.iloc[1*284:(59-6)*284]

Train_5=Train.iloc[2*284:(59-5)*284]

Train_4=Train.iloc[3*284:(59-4)*284]

Train_3=Train.iloc[4*284:(59-3)*284]

Train_2=Train.iloc[5*284:(59-2)*284]

Train_1=Train.iloc[6*284:(59-1)*284]

Train_0=Train.iloc[7*284:(59-0)*284]

Train_7.sort_values(by=['Province/State','Country/Region'],inplace=True)

Train_6.sort_values(by=['Province/State','Country/Region'],inplace=True)

Train_5.sort_values(by=['Province/State','Country/Region'],inplace=True)

Train_4.sort_values(by=['Province/State','Country/Region'],inplace=True)

Train_3.sort_values(by=['Province/State','Country/Region'],inplace=True)

Train_2.sort_values(by=['Province/State','Country/Region'],inplace=True)

Train_1.sort_values(by=['Province/State','Country/Region'],inplace=True)

Train_0.sort_values(by=['Province/State','Country/Region'],inplace=True)
Train_0['ConfirmedCases_lag_1']=np.array(Train_1['ConfirmedCases'])

Train_0['Fatalities_lag_1']=np.array(Train_1['Fatalities'])



Train_0['ConfirmedCases_lag_2']=np.array(Train_2['ConfirmedCases'])

Train_0['Fatalities_lag_2']=np.array(Train_2['Fatalities'])



Train_0['ConfirmedCases_lag_3']=np.array(Train_3['ConfirmedCases'])

Train_0['Fatalities_lag_3']=np.array(Train_3['Fatalities'])



Train_0['ConfirmedCases_lag_4']=np.array(Train_4['ConfirmedCases'])

Train_0['Fatalities_lag_4']=np.array(Train_4['Fatalities'])





Train_0['ConfirmedCases_lag_5']=np.array(Train_5['ConfirmedCases'])

Train_0['Fatalities_lag_5']=np.array(Train_5['Fatalities'])



Train_0['ConfirmedCases_lag_6']=np.array(Train_6['ConfirmedCases'])

Train_0['Fatalities_lag_6']=np.array(Train_6['Fatalities'])



Train_0['ConfirmedCases_lag_7']=np.array(Train_7['ConfirmedCases'])

Train_0['Fatalities_lag_7']=np.array(Train_7['Fatalities'])

Train_0.sort_values('Province/State',inplace=True)

Train_0.sort_values('Country/Region',inplace=True)
Final_data=Train_0[['Date', 'Province/State', 'Country/Region','ConfirmedCases', 'Fatalities', 'Clusters',

       'Date_After_First_ConfirmedCases', 'Date_After_First_Fatalities',

                    'ConfirmedCases_lag_1',

       'Fatalities_lag_1', 'ConfirmedCases_lag_2', 'Fatalities_lag_2',

       'ConfirmedCases_lag_3', 'Fatalities_lag_3', 'ConfirmedCases_lag_4',

       'Fatalities_lag_4', 'ConfirmedCases_lag_5', 'Fatalities_lag_5',

       'ConfirmedCases_lag_6', 'Fatalities_lag_6', 'ConfirmedCases_lag_7',

       'Fatalities_lag_7']]

Final_data.fillna(0,inplace=True)
D_1=Final_data[['Date', 'Province/State', 'Country/Region','Clusters','Date_After_First_ConfirmedCases', 'Date_After_First_Fatalities']]

D_2=Final_data.drop(['Date', 'Province/State', 'Country/Region','Clusters','Date_After_First_ConfirmedCases', 'Date_After_First_Fatalities'],axis=1)

D_2=np.log(D_2+0.8)

D_2[D_1.columns]=D_1

D_2['ConfirmedCases_Difference_1']=D_2['ConfirmedCases']-D_2['ConfirmedCases_lag_1']

D_2['ConfirmedCases_Difference_2']=D_2['ConfirmedCases_lag_1']-D_2['ConfirmedCases_lag_2']

D_2['ConfirmedCases_Difference_3']=D_2['ConfirmedCases_lag_2']-D_2['ConfirmedCases_lag_3']

D_2['ConfirmedCases_Difference_4']=D_2['ConfirmedCases_lag_3']-D_2['ConfirmedCases_lag_4']

D_2['ConfirmedCases_Difference_5']=D_2['ConfirmedCases_lag_4']-D_2['ConfirmedCases_lag_5']

D_2['ConfirmedCases_Difference_6']=D_2['ConfirmedCases_lag_5']-D_2['ConfirmedCases_lag_6']

D_2['ConfirmedCases_Difference_7']=D_2['ConfirmedCases_lag_6']-D_2['ConfirmedCases_lag_7']



D_2['Fatalities_Difference_1']=D_2['Fatalities']-D_2['Fatalities_lag_1']

D_2['Fatalities_Difference_2']=D_2['Fatalities_lag_1']-D_2['Fatalities_lag_2']

D_2['Fatalities_Difference_3']=D_2['Fatalities_lag_2']-D_2['Fatalities_lag_3']

D_2['Fatalities_Difference_4']=D_2['Fatalities_lag_3']-D_2['Fatalities_lag_4']

D_2['Fatalities_Difference_5']=D_2['Fatalities_lag_4']-D_2['Fatalities_lag_5']

D_2['Fatalities_Difference_6']=D_2['Fatalities_lag_5']-D_2['Fatalities_lag_6']

D_2['Fatalities_Difference_7']=D_2['Fatalities_lag_6']-D_2['Fatalities_lag_7']

#Final_data=D_2

#del D_1,D_2
D_2[['2','3','4']]=pd.get_dummies(D_2['Clusters'].astype(str),drop_first=True)

D_2.sort_values(['Date','Country/Region','Province/State'],inplace=True)
Covariates_target=D_2[['2', '3', '4',

     'Date_After_First_ConfirmedCases','Date_After_First_Fatalities', 

     'ConfirmedCases_Difference_7','Fatalities_Difference_7',

    'ConfirmedCases_Difference_6','Fatalities_Difference_6',

    'ConfirmedCases_Difference_5','Fatalities_Difference_5',

    'ConfirmedCases_Difference_4','Fatalities_Difference_4',

    'ConfirmedCases_Difference_3','Fatalities_Difference_3',

    'ConfirmedCases_Difference_2','Fatalities_Difference_2',

    'ConfirmedCases_Difference_1','Fatalities_Difference_1']]

INFO=D_2[['Date', 'Province/State', 'Country/Region']]

Target=Covariates_target[['ConfirmedCases_Difference_1','Fatalities_Difference_1']]

Covariates=Covariates_target.drop(['ConfirmedCases_Difference_1','Fatalities_Difference_1'],axis=1)

Target_1=Covariates_target['ConfirmedCases_Difference_1']

Target_2=Covariates_target['Fatalities_Difference_1']

from sklearn.model_selection import train_test_split

Covariates_Train,Covariates_Test,Target_1_Train,Target_1_Test,Target_2_Train,Target_2_Test= train_test_split(Covariates,Target_1,Target_2, test_size=0.33, random_state=42)
from xgboost import XGBRegressor

model = XGBRegressor(learning_rate=0.1,gamma=0.001,n_estimators=200,reg_lambda=.9,

                      booster='gbtree',max_depth=2,subsample=1,rate_drop=0,

                     sampling_method="gradient_based")

eval_set = [(Covariates_Test.values, Target_1_Test.values)]

model.fit(Covariates_Train.values, Target_1_Train.values, early_stopping_rounds=5, eval_set=eval_set, verbose=True)
model_1 = XGBRegressor(learning_rate=0.1,gamma=0.001,n_estimators=77,reg_lambda=1,

                      booster='gbtree',max_depth=2,subsample=1,rate_drop=0,

                     sampling_method="gradient_based")

model_1.fit(Covariates_Train.values, Target_1_Train.values, verbose=True)
model = XGBRegressor(learning_rate=0.01,gamma=0.001,n_estimators=5000,reg_lambda=1,

                      booster='gbtree',max_depth=6,subsample=1,rate_drop=0,

                     sampling_method="gradient_based")

eval_set = [(Covariates_Test.values, Target_2_Test.values)]

model.fit(Covariates_Train.values, Target_2_Train.values, early_stopping_rounds=5, eval_set=eval_set, verbose=True)
model_2 = XGBRegressor(learning_rate=0.01,gamma=0.001,n_estimators=436,reg_lambda=1,

                      booster='gbtree',max_depth=6,subsample=1,rate_drop=0,

                     sampling_method="gradient_based")

model_2.fit(Covariates.values, Target_2.values,verbose=True)
Test_to_submit=pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

Test_to_submit['Lat'].fillna(12.5,inplace=True)

Test_to_submit['Long'].fillna(-70.0,inplace=True)

Test_to_submit.fillna("NA",inplace=True)

Test_to_submit['Clusters']=cl_4.predict(Test_to_submit[['Lat','Long']])+1

Test_to_submit['Clusters']=Test_to_submit['Clusters'].astype(str)

dummy_temp=pd.get_dummies(Test_to_submit['Clusters'],drop_first=True)

Test_to_submit[dummy_temp.columns]=dummy_temp

del dummy_temp

Test_to_submit
Test_to_submit=Test_to_submit.merge(Dates_cases_Fatalities)

Test_to_submit['Date']=pd.to_datetime(Test_to_submit['Date'])

Test_to_submit['Date_After_First_Fatalities']=Test_to_submit['Date']-Test_to_submit['First_Fatalities']

Test_to_submit['Date_After_First_ConfirmedCases']=Test_to_submit['Date']-Test_to_submit['First_ConfirmedCases']

Test_to_submit['Date_After_First_Fatalities']=Test_to_submit['Date_After_First_Fatalities'].dt.days

Test_to_submit['Date_After_First_ConfirmedCases']=Test_to_submit['Date_After_First_ConfirmedCases'].dt.days
Test_to_submit.sort_values(by='Date',inplace=True)
Test_to_submit=Test_to_submit[['ForecastId','Date','2', '3', '4', 'Date_After_First_ConfirmedCases','Date_After_First_Fatalities']]
temp=Covariates_target[INFO['Date']==INFO['Date'].max()]

temp.drop(['2', '3', '4',

           'Date_After_First_ConfirmedCases','Date_After_First_Fatalities', 

           'ConfirmedCases_Difference_7','Fatalities_Difference_7'],axis=1,inplace=True)

temp.columns=['ConfirmedCases_Difference_7', 'Fatalities_Difference_7',

       'ConfirmedCases_Difference_6', 'Fatalities_Difference_6',

       'ConfirmedCases_Difference_5', 'Fatalities_Difference_5',

       'ConfirmedCases_Difference_4', 'Fatalities_Difference_4',

       'ConfirmedCases_Difference_3', 'Fatalities_Difference_3',

       'ConfirmedCases_Difference_2', 'Fatalities_Difference_2']

temp.index=np.arange(0,284)

Test_Dates=Test_to_submit['Date'].unique()

l=len(Test_Dates)
i=0

ConfirmedCases_Forecasted=[]

Fatalities_Forecasted=[]





for i in range(l):

    fixed_part=Test_to_submit[Test_to_submit['Date']==Test_Dates[i]].drop(['ForecastId', 'Date'],axis=1)

    fixed_part.index=np.arange(0,284)

    cov=fixed_part

    cov[temp.columns]=temp





    pred_1=list(model_1.predict(cov.values))

    temp['ConfirmedCases_Difference_1']=pred_1

    ConfirmedCases_Forecasted=ConfirmedCases_Forecasted+pred_1



    pred_2=list(model_2.predict(cov.values))

    temp['Fatalities_Difference_1']=pred_2

    Fatalities_Forecasted=Fatalities_Forecasted+pred_2



    temp.drop(['ConfirmedCases_Difference_7', 'Fatalities_Difference_7'],axis=1,inplace=True)

    temp.columns=['ConfirmedCases_Difference_7', 'Fatalities_Difference_7',

           'ConfirmedCases_Difference_6', 'Fatalities_Difference_6',

           'ConfirmedCases_Difference_5', 'Fatalities_Difference_5',

           'ConfirmedCases_Difference_4', 'Fatalities_Difference_4',

           'ConfirmedCases_Difference_3', 'Fatalities_Difference_3',

           'ConfirmedCases_Difference_2', 'Fatalities_Difference_2']

    temp.index=np.arange(0,284)
Forecasted=pd.DataFrame({'ConfirmedCases_Forecasted':ConfirmedCases_Forecasted,'Fatalities_Forecasted':Fatalities_Forecasted})
Last_Days_Records=D_2[D_2['Date']==D_2['Date'].max()][['ConfirmedCases', 'Fatalities']]

Last_Days_ConfirmedCases=Last_Days_Records['ConfirmedCases']

Last_Days_Fatalities=Last_Days_Records['Fatalities']
ConfirmedCases_Forecasted=np.array(Forecasted['ConfirmedCases_Forecasted'])

Fatalities_Forecasted=np.array(Forecasted['Fatalities_Forecasted'])

Last_Days_ConfirmedCases=np.array(Last_Days_ConfirmedCases)

Last_Days_Fatalities=np.array(Last_Days_Fatalities)

i=0

ConfirmedCases=np.array([])

Fatalities=np.array([])

for i in range(43):

    Last_Days_ConfirmedCases=ConfirmedCases_Forecasted[(i*284):((i+1)*284)]+Last_Days_ConfirmedCases

    ConfirmedCases=np.concatenate((ConfirmedCases,Last_Days_ConfirmedCases))



    Last_Days_Fatalities=Fatalities_Forecasted[(i*284):((i+1)*284)]+Last_Days_Fatalities

    Fatalities=np.concatenate((Fatalities,Last_Days_Fatalities))

ConfirmedCases=np.exp(ConfirmedCases)-0.8

Fatalities=np.exp(Fatalities)-0.8

Test_to_submit['ConfirmedCases']=np.round(ConfirmedCases)

Test_to_submit['Fatalities']=np.round(Fatalities)
Test_to_submit.sort_values('ForecastId',inplace=True)
Test_to_submit[['ForecastId','ConfirmedCases','Fatalities']].to_csv('submission.csv',index=False)