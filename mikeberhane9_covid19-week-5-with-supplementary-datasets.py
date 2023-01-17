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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor 

import warnings

warnings.filterwarnings('ignore')



df=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv', index_col='Id')

dtest=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv', index_col='ForecastId')



# dividing the datasets into ConfirmedCases and Fatalities

#dcc=df.loc[df['Target']=='ConfirmedCases']

#dfat=df.loc[df['Target']=='Fatalities']

#tdcc=dtest.loc[dtest['Target']=='ConfirmedCases']

#tdfat=dtest.loc[dtest['Target']=='Fatalities']



#Drop Target column 

#dcc.drop('Target', axis=1, inplace=True)

#dfat.drop('Target', axis=1, inplace=True)

#tdcc.drop('Target', axis=1, inplace=True)

#tdfat.drop('Target', axis=1, inplace=True)



#create y and drop TargetValue from dcc and dfat

y=df.TargetValue

#yfat=dfat.TargetValue

df.drop('TargetValue', axis=1, inplace=True)

#dfat.drop('TargetValue', axis=1, inplace=True)

df['check']=1

dtest['check']=2

combo=pd.concat([df,dtest])

#dfat['check']=1

#tdfat['check']=2

#combo_fat=pd.concat([dfat,tdfat])



#Add democracy indices for each country from Democracy dataset

democracy=pd.read_csv('/kaggle/input/covid19-week-5-supplementary-datasets/Democracy.csv')

demo2018=democracy.loc[democracy['time']==2018]

demo2019=democracy.loc[democracy['time']==2019]



demo2018.drop(['geo','time','Electoral pluralism index (EIU)','Political participation index(EIU)',

              'Political culture index (EIU)','Change in democracy index (EIU)'], axis=1, inplace=True)

demo2018=demo2018.rename(columns={"name": "Country_Region", "Democracy index (EIU)": "di18", 

                         "Government index (EIU)": "gi18","Civil liberties index (EIU)": "cli18"})

demo2019.drop(['geo','time','Electoral pluralism index (EIU)','Political participation index(EIU)',

              'Political culture index (EIU)','Change in democracy index (EIU)'], axis=1, inplace=True)

demo2019=demo2019.rename(columns={"name": "Country_Region", "Democracy index (EIU)": "di19", 

                         "Government index (EIU)": "gi19","Civil liberties index (EIU)": "cli19"})

combo=combo.merge(demo2018, how='left', on=['Country_Region'])

combo=combo.merge(demo2019, how='left', on=['Country_Region'])

#combo_fat=combo_fat.merge(demo2018, how='left', on=['Country_Region'])

#combo_fat=combo_fat.merge(demo2019, how='left', on=['Country_Region'])



#Add economy indicators for each country from 2019 economy dataset

economy=pd.read_excel('/kaggle/input/covid19-week-5-supplementary-datasets/economy.xlsx')



econ_GDP_ConstPr=economy.loc[(economy['Subject Descriptor']=='Gross domestic product, constant prices')]

econ_GDP_CurrPr=economy.loc[economy['Subject Descriptor']=='Gross domestic product, current prices']

econ_GDP_Capita=economy.loc[economy['Subject Descriptor']=='Gross domestic product per capita, constant prices']

econ_Inf_ConstPr=economy.loc[economy['Subject Descriptor']=='Inflation, average consumer prices']

econ_Inf_eoP=economy.loc[economy['Subject Descriptor']=='Inflation, end of period consumer prices']

econ_unemp=economy.loc[economy['Subject Descriptor']=='Unemployment rate']



econ_GDP_ConstPr.drop(['Subject Descriptor','Units'], axis=1, inplace=True)

econ_GDP_ConstPr=econ_GDP_ConstPr.rename(columns={"Country": "Country_Region", 2019: "GDP_ConstPr19"})

econ_GDP_CurrPr.drop(['Subject Descriptor','Units'], axis=1, inplace=True)

econ_GDP_CurrPr=econ_GDP_CurrPr.rename(columns={"Country": "Country_Region", 2019: "GDP_CurrPr19"})

econ_GDP_Capita.drop(['Subject Descriptor','Units'], axis=1, inplace=True)

econ_GDP_Capita=econ_GDP_Capita.rename(columns={"Country": "Country_Region", 2019: "GDP_Capita19"})

econ_Inf_ConstPr.drop(['Subject Descriptor','Units'], axis=1, inplace=True)

econ_Inf_ConstPr=econ_Inf_ConstPr.rename(columns={"Country": "Country_Region", 2019: "Inf_ConstPr19"})

econ_Inf_eoP.drop(['Subject Descriptor','Units'], axis=1, inplace=True)

econ_Inf_eoP=econ_Inf_eoP.rename(columns={"Country": "Country_Region", 2019: "Inf_eoP19"})

econ_unemp.drop(['Subject Descriptor','Units'], axis=1, inplace=True)

econ_unemp=econ_unemp.rename(columns={"Country": "Country_Region", 2019: "unemp19"})



combo=combo.merge(econ_GDP_ConstPr, how='left', on=['Country_Region'])

combo=combo.merge(econ_GDP_CurrPr, how='left', on=['Country_Region'])

combo=combo.merge(econ_GDP_Capita, how='left', on=['Country_Region'])

combo=combo.merge(econ_Inf_ConstPr, how='left', on=['Country_Region'])

combo=combo.merge(econ_Inf_eoP, how='left', on=['Country_Region'])

combo=combo.merge(econ_unemp, how='left', on=['Country_Region'])



#combo_fat=combo_fat.merge(econ_GDP_ConstPr, how='left', on=['Country_Region'])

#combo_fat=combo_fat.merge(econ_GDP_CurrPr, how='left', on=['Country_Region'])

#combo_fat=combo_fat.merge(econ_GDP_Capita, how='left', on=['Country_Region'])

#combo_fat=combo_fat.merge(econ_Inf_ConstPr, how='left', on=['Country_Region'])

#combo_fat=combo_fat.merge(econ_Inf_eoP, how='left', on=['Country_Region'])

#combo_fat=combo_fat.merge(econ_unemp, how='left', on=['Country_Region'])



#Add population density for each country 

density=pd.read_excel('/kaggle/input/covid19-week-5-supplementary-datasets/population_density.xlsx')

combo=combo.merge(density, how='left', on=['Country_Region'])

#combo_fat=combo_fat.merge(density, how='left', on=['Country_Region'])



#Add Sanitation Services for each country from WHO 2017 WASH report (the latest from the data repository) 

San=pd.read_csv('/kaggle/input/covid19-week-5-supplementary-datasets/San_service.csv')

combo=combo.merge(San, how='left', on=['Country_Region'])

#combo_fat=combo_fat.merge(San, how='left', on=['Country_Region'])



#Add Mortality due to unsafe WASH services for each country from WHO 2016 WASH report (the latest from the data repository) 

Mort=pd.read_csv('/kaggle/input/covid19-week-5-supplementary-datasets/Mort_unsafe.csv')

combo=combo.merge(Mort, how='left', on=['Country_Region'])

#combo_fat=combo_fat.merge(Mort, how='left', on=['Country_Region'])



#Impute Mort by median()

combo['Mor_unsafeWASH16']=combo['Mor_unsafeWASH16'].fillna(combo['Mor_unsafeWASH16'].median())

#combo_fat['Mor_unsafeWASH16']=combo_fat['Mor_unsafeWASH16'].fillna(combo_fat['Mor_unsafeWASH16'].median())

#Impute demo2018 and demo2019 indicators by min()

combo['di18']=combo['di18'].fillna(combo['di18'].min())

#combo_fat['di18']=combo_fat['di18'].fillna(combo_fat['di18'].min())

combo['di19']=combo['di19'].fillna(combo['di19'].min())

#combo_fat['di19']=combo_fat['di19'].fillna(combo_fat['di19'].min())

combo['gi18']=combo['gi18'].fillna(combo['gi18'].min())

#combo_fat['gi18']=combo_fat['gi18'].fillna(combo_fat['gi18'].min())

combo['gi19']=combo['gi19'].fillna(combo['gi19'].min())

#combo_fat['gi19']=combo_fat['gi19'].fillna(combo_fat['gi19'].min())

combo['cli18']=combo['cli18'].fillna(combo['cli18'].min())

#combo_fat['cli18']=combo_fat['cli18'].fillna(combo_fat['cli18'].min())

combo['cli19']=combo['cli19'].fillna(combo['cli19'].min())

#combo_fat['cli19']=combo_fat['cli19'].fillna(combo_fat['cli19'].min())

#Impute San by max() - cruise ships

combo['Basic_San_service17']=combo['Basic_San_service17'].fillna(combo['Basic_San_service17'].max())

#combo_fat['Basic_San_service17']=combo_fat['Basic_San_service17'].fillna(combo_fat['Basic_San_service17'].max())

combo['Safe_San_service17']=combo['Safe_San_service17'].fillna(combo['Safe_San_service17'].max())

#combo_fat['Safe_San_service17']=combo_fat['Safe_San_service17'].fillna(combo_fat['Safe_San_service17'].max())

#Impute econ by 0

combo['GDP_ConstPr19']=combo['GDP_ConstPr19'].fillna(0)

#combo_fat['GDP_ConstPr19']=combo_fat['GDP_ConstPr19'].fillna(0)

combo['GDP_CurrPr19']=combo['GDP_CurrPr19'].fillna(0)

#combo_fat['GDP_CurrPr19']=combo_fat['GDP_CurrPr19'].fillna(0)

combo['GDP_Capita19']=combo['GDP_Capita19'].fillna(0)

#combo_fat['GDP_Capita19']=combo_fat['GDP_Capita19'].fillna(0)

combo['Inf_ConstPr19']=combo['Inf_ConstPr19'].fillna(0)

#combo_fat['Inf_ConstPr19']=combo_fat['Inf_ConstPr19'].fillna(0)

combo['Inf_eoP19']=combo['Inf_eoP19'].fillna(0)

#combo_fat['Inf_eoP19']=combo_fat['Inf_eoP19'].fillna(0)

combo['unemp19']=combo['unemp19'].fillna(0)

#combo_fat['unemp19']=combo_fat['unemp19'].fillna(0)

#Impute density by max()

combo['density_19']=combo['density_19'].fillna(combo['density_19'].max())

#combo_fat['density_19']=combo_fat['density_19'].fillna(combo_cc['density_19'].max())

combo['density_20']=combo['density_20'].fillna(combo['density_20'].max())

#combo_fat['density_20']=combo_fat['density_20'].fillna(combo_cc['density_20'].max())



#Impute County and Province_State by 'A'

combo['County']=combo['County'].fillna('A')

#combo_fat['County']=combo_fat['County'].fillna('A')

combo['Province_State']=combo['Province_State'].fillna('A')

#combo_fat['Province_State']=combo_fat['Province_State'].fillna('A')



#Managing Date by taking only the MM-DD

def date_split(date):

    d=date.str.split('-', n=1, expand=True)

    return d[1]

combo['MM_DD']= date_split(combo['Date'])

#combo_fat['MM_DD']= date_split(combo_fat['Date'])



#Managing Location

combo['Location']=combo['County'] + combo['Province_State'] + combo['Country_Region']

#combo_fat['Location']=combo_fat['County'] + combo_fat['Province_State'] + combo_fat['Country_Region']



#Label Encode Date

le=LabelEncoder()

combo['Target']=le.fit_transform(combo['Target'])

combo['MM_DD']=le.fit_transform(combo['MM_DD'])

#combo_fat['MM_DD']=le.fit_transform(combo_fat['MM_DD'])

combo['Location']=le.fit_transform(combo['Location'])

#combo_fat['Location']=le.fit_transform(combo_fat['Location'])



#Drop repeated columns like Date

combo.drop(['County','Province_State','Country_Region','Date'], axis=1, inplace=True)

#combo_fat.drop(['County','Province_State','Country_Region','Date'], axis=1, inplace=True)



#Divide combo to train and test dataframes and drop 'check' column

df_cc=combo[combo['check']==1]

dtest_cc=combo[combo['check']==2]

#df_fat=combo_fat[combo_fat['check']==1]

#dtest_fat=combo_fat[combo_fat['check']==2]



#Separate 'Weight' in separate df and drop it from the originals

#w_cc=df_cc['Weight']

#wtest_cc=dtest_cc['Weight']

#w_fat=df_fat['Weight']

#wtest_fat=dtest_fat['Weight']

df_cc.drop('check', axis=1, inplace=True)

dtest_cc.drop('check', axis=1, inplace=True)

#df_fat.drop(['Weight', 'check'], axis=1, inplace=True)

#dtest_fat.drop(['Weight', 'check'], axis=1, inplace=True)
df_cc.head()
dtest_cc.head()
for i in range(2):

    X_train1, X_valid1, y_train1, y_valid1=train_test_split(df_cc, y, train_size=0.8, 

                                                            test_size=0.2, random_state=8)

    et=ExtraTreesRegressor(n_estimators=10, random_state=14)

    p=et.fit(X_train1, y_train1).predict(X_valid1)

    print(i,'----', mean_absolute_error(y_valid1, p))

    
pf1=et.fit(df_cc,y).predict(dtest_cc)

pf_out=pd.DataFrame({'Id': dtest_cc.index, 'TargetValue':pf1})



q=pf_out.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

a=pf_out.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

z=pf_out.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()



q.columns=['Id', '0.05']

a.columns=['Id', '0.5']

z.columns=['Id', '0.95']

q=pd.concat([q,a['0.5'],z['0.95']], 1)
s={}

for i in range(len(q)):

    s[str(i+1)+'_'+'0.05']=q['0.05'][i]

    s[str(i+1)+'_'+'0.5']=q['0.5'][i]

    s[str(i+1)+'_'+'0.95']=q['0.95'][i]

submission=pd.DataFrame(s.items(), columns=['ForecastId_Quantile', 'TargetValue'])

submission.head()
submission.to_csv('submission.csv', index=False)