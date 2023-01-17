# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import date



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
PATH_WEEK2 = './week2'



df_Train = pd.read_csv(f'/kaggle/input/covid19-global-forecasting-week-3/train.csv')

df_test = pd.read_csv(f'/kaggle/input/covid19-global-forecasting-week-3/test.csv')



df_Population = pd.read_csv(f'/kaggle/input/population-by-country-2020/population_by_country_2020.csv')



df_gdp = pd.read_excel(f'/kaggle/input/global-economic-monitor/gem-excel-zip-9-97-mb-/GDP at market prices, constant 2010 LCU, millions, seas. adj..xlsx')
df_Train['Date_Since'] = pd.to_datetime(date(2019, 12, 17)); df_Train['Date_Since']

df_Train['Days_Since'] = pd.to_datetime(df_Train['Date']) - df_Train['Date_Since']



df_Train['Days_Since'] = df_Train['Days_Since'].dt.days

df_Train = df_Train.drop(columns=['Date_Since'])
df_test['Date_Since'] = pd.to_datetime(date(2019, 12, 17)); df_test['Date_Since']

df_test['Days_Since'] = pd.to_datetime(df_test['Date']) - df_test['Date_Since']



df_test['Days_Since'] = df_test['Days_Since'].dt.days

df_test = df_test.drop(columns=['Date_Since'])
# create new column with gdp

# with the key of Country

# we want to omit the countries that are not in gdp dataset

# can we join?





# select unique countries where gdp == Nan

df_Train.rename(columns={'Country_Region':'Country'}, inplace=True)

df_test.rename(columns={'Country_Region':'Country'}, inplace=True)



df_Train.rename(columns={'Province_State':'State'}, inplace=True)

df_test.rename(columns={'Province_State':'State'}, inplace=True)
df_Train.loc[: , ['Country', 'ConfirmedCases', 'Fatalities']].groupby(['Country']).max().sort_values(by='ConfirmedCases', ascending=False).reset_index()[:15].style.background_gradient(cmap='rainbow')
df_Population.columns
df_Population.rename(columns={'Country (or dependency)':'Country'}, inplace=True)
train_countries = df_Train.Country.unique().tolist()

pop_countries = df_Population.Country.unique().tolist()



for country in train_countries:

    if country not in pop_countries:

        print (country)
renameCountryNames = {

    "Congo (Brazzaville)": "Congo",

    "Congo (Kinshasa)": "Congo",

    "Cote d'Ivoire": "Côte d'Ivoire",

    "Czechia": "Czech Republic (Czechia)",

    "Korea, South": "South Korea",

    "Saint Kitts and Nevis": "Saint Kitts & Nevis",

    "Saint Vincent and the Grenadines": "St. Vincent & Grenadines",

    "Taiwan*": "Taiwan",

    "US": "United States"

}
#df_Train.loc[df_Train.Country in renameCountryNames.keys(), 'Country'] = df_Train.loc[df_Train.Country in renameCountryNames.keys(), 'Country'].map(country_map)

df_Train.replace({'Country': renameCountryNames}, inplace=True)

df_test.replace({'Country': renameCountryNames}, inplace=True)
df_Population.loc[df_Population['Med. Age']=='N.A.', 'Med. Age'] = df_Population.loc[df_Population['Med. Age']!='N.A.', 'Med. Age'].mode()[0]

df_Population.loc[df_Population['Urban Pop %']=='N.A.', 'Urban Pop %'] = df_Population.loc[df_Population['Urban Pop %']!='N.A.', 'Urban Pop %'].mode()[0]

df_Population.loc[df_Population['Fert. Rate']=='N.A.', 'Fert. Rate'] = df_Population.loc[df_Population['Fert. Rate']!='N.A.', 'Fert. Rate'].mode()[0]

df_Population.loc[:, 'Migrants (net)'] = df_Population.loc[:, 'Migrants (net)'].fillna(0)

df_Population['Yearly Change'] = df_Population['Yearly Change'].str.rstrip('%')

df_Population['World Share'] = df_Population['World Share'].str.rstrip('%')

df_Population['Urban Pop %'] = df_Population['Urban Pop %'].str.rstrip('%')

df_Population = df_Population.astype({"Net Change": int,"Density (P/Km²)": int,"Population (2020)": int,"Land Area (Km²)": int,"Yearly Change": float,"Urban Pop %": int,"Fert. Rate": float,"Med. Age": int,"World Share": float, "Migrants (net)": float,})



# As the Country value "Diamond Princess" is a CRUISE, we replace the population 

df_Population = df_Population.append(pd.Series(['Diamond Princess', 3500, 0, 0, 0, 0, 0.0, 1, 30, 0, 0.0], index=df_Population.columns ), ignore_index=True)
df_Train = df_Train.merge(df_Population, how='left', left_on='Country', right_on='Country')

df_test = df_test.merge(df_Population, how='left', left_on='Country', right_on='Country')
year18 = df_gdp[df_gdp['Unnamed: 0'] == 2018.0].drop(columns=['Unnamed: 0'])



pd.set_option("display.max_rows", 100, "display.max_columns", 100)



year18 = year18.T.reset_index()

year18.columns = year18.columns.astype(str)

year18.columns

year18 = year18.rename(columns={'index': 'Country', '29': 'GDP'})



df_Train = df_Train.merge(year18, how='left', on=['Country'])

df_Train['GDP'] = df_Train['GDP'].fillna(df_Train['GDP'].median())



df_test = df_test.merge(year18, how='left', on=['Country'])

df_test['GDP'] = df_test['GDP'].fillna(df_test['GDP'].median())



# if gdp == Nan fill with median

# if X_Train.loc[X_Train['Country'] == country]['GDP'].isnull().values.any():

# X_Train = X_Train.loc[X_Train['Country'] == country].drop(columns=['GDP'])

# X_Test = X_Test.loc[X_Test['Country'] == country].drop(columns=['GDP'])







df_Train['Date'] = pd.to_datetime(df_Train['Date'], infer_datetime_format=True)

df_test['Date'] = pd.to_datetime(df_test['Date'], infer_datetime_format=True)
MIN_TEST_DATE = df_test.Date.min()
df_train = df_Train.loc[df_Train.Date < MIN_TEST_DATE, :]
y1_Train = df_train.iloc[:, -2]

y1_Train.head()
y2_Train = df_train.iloc[:, -1]

y2_Train.head()
EMPTY_VAL = "EMPTY_VAL"



def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
#X_Train = df_train.loc[:, ['State', 'Country', 'Date']]

X_Train = df_train.copy()



X_Train['State'].fillna(EMPTY_VAL, inplace=True)

X_Train['State'] = X_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_Train['year'] = X_Train['Date'].dt.year

X_Train['month'] = X_Train['Date'].dt.month

X_Train['week'] = X_Train['Date'].dt.week

X_Train['day'] = X_Train['Date'].dt.day

X_Train['dayofweek'] = X_Train['Date'].dt.dayofweek



X_Train.loc[:, 'Date'] = X_Train.Date.dt.strftime("%m%d")

X_Train["Date"]  = X_Train["Date"].astype(int)



#X_Train.drop(columns=['Date'], axis=1, inplace=True)



# X_Train.head()
#X_Test = df_test.loc[:, ['State', 'Country', 'Date']]

X_Test = df_test.copy()



X_Test['State'].fillna(EMPTY_VAL, inplace=True)

X_Test['State'] = X_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



X_Test['year'] = X_Test['Date'].dt.year

X_Test['month'] = X_Test['Date'].dt.month

X_Test['week'] = X_Test['Date'].dt.week

X_Test['day'] = X_Test['Date'].dt.day

X_Test['dayofweek'] = X_Test['Date'].dt.dayofweek



X_Test.loc[:, 'Date'] = X_Test.Date.dt.strftime("%m%d")

X_Test["Date"]  = X_Test["Date"].astype(int)



#X_Test.drop(columns=['Date'], axis=1, inplace=True)



# X_Test.head()
from sklearn import preprocessing



le = preprocessing.LabelEncoder()
X_Train.Country = le.fit_transform(X_Train.Country)

X_Train['State'] = le.fit_transform(X_Train['State'])



# X_Train.head()
X_Test.Country = le.fit_transform(X_Test.Country)

X_Test['State'] = le.fit_transform(X_Test['State'])



# X_Test.head()
from warnings import filterwarnings

filterwarnings('ignore')
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn import metrics
from fastai.tabular import * 
procs = [FillMissing, Categorify, Normalize]

# get out of bad variable name



# df_case.shape

X_Train = X_Train.rename(columns={'Id': 'ForecastId'})
df_case = X_Train.copy()

df_case = df_case.drop(columns=['Fatalities'])



valid_idx = range(len(df_case)-4000, len(df_case))



dep_var = 'ConfirmedCases'



cat_names = ['State', 'Country', 'year', 'month', 'week', 'day', 'dayofweek']



data_cases = TabularDataBunch.from_df(path='.',df=df_case,

                                      dep_var=dep_var, 

                                      valid_idx=valid_idx,

                                      procs=procs,

                                      cat_names=cat_names,

                                      test_df=X_Test)

learn_c = tabular_learner(

    data_cases, layers=[200,50], emb_szs={'native-country': 10}, metrics=mse)

# learn.fit_one_cycle(1, 1e-2)

learn_c.fit(5, 1e-1)



preds_c, _ = learn_c.get_preds(ds_type=DatasetType.Test)
df_fatal = X_Train.copy()

df_fatal = df_fatal.drop(columns=['ConfirmedCases'])



valid_idx = range(len(df_fatal)-4000, len(df_fatal))



dep_var = 'Fatalities'



cat_names = ['State', 'Country', 'year', 'month', 'week', 'day', 'dayofweek']



data_fatal = TabularDataBunch.from_df(path='.',df=df_fatal,

                                      dep_var=dep_var, 

                                      valid_idx=valid_idx,

                                      procs=procs,

                                      cat_names=cat_names,

                                      test_df=X_Test)

learn_f = tabular_learner(

    data_fatal, layers=[200,50], emb_szs={'native-country': 10}, metrics=mse)

# learn.fit_one_cycle(1, 1e-2)

learn_f.fit(5, 1e-1)



preds_f, _ = learn_c.get_preds(ds_type=DatasetType.Test)
df_out = pd.DataFrame({'ForecastId': X_Test['ForecastId'], 'ConfirmedCases': preds_c,

                   'Fatalities': preds_f})



# X_Test['ForecastId'].shape, preds_c.shape, preds_f.shape

data_cases.train_ds
X_Test.tail()
preds
df_out.to_csv('submission.csv', index=False)