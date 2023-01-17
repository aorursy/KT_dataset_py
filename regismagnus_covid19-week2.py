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

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer



from sklearn.ensemble import RandomForestRegressor



path_train = '../input/covid19-global-forecasting-week-2/train.csv'

path_test = '../input/covid19-global-forecasting-week-2/test.csv'



covid_train = pd.read_csv(path_train)

covid_test = pd.read_csv(path_test, index_col='ForecastId')



print("Train len: %d, Test len: %d" %(len(covid_train), len(covid_test)))

print("Last register Date: ", covid_train['Date'].iloc[len(covid_train)-1])
def inDay(col):

    '''get new register in day of Confirmed Cases or Fatalities'''

    inDay=[]

    for i in range(len(covid_train)):

        if i==0 or covid_train['Country_Region'][i]!=covid_train['Country_Region'][i-1] or (not pd.isna(covid_train['Province_State'][i]) and not pd.isna(covid_train['Province_State'][i-1]) and covid_train['Province_State'][i]!=covid_train['Province_State'][i-1]) or pd.isna(covid_train['Province_State'][i])!=pd.isna(covid_train['Province_State'][i-1]) :

            inDay.append(covid_train[col][i])

        else:

            inDay.append(covid_train[col][i]-covid_train[col][i-1])

    return pd.DataFrame(inDay)
covid_train['ConfirmedCasesInDay'] = inDay('ConfirmedCases')

covid_train['FatalitiesInDay'] = inDay('Fatalities')
covid_train['Date'] = pd.to_datetime(covid_train['Date'])
covid_train.describe()
covid_train[covid_train['FatalitiesInDay']<0].head()
plt.figure(figsize=(14,6))

plt.title('Confirmed Cases timeline')

sns.lineplot(y=covid_train['ConfirmedCases'], x=covid_train['Date'])
plt.figure(figsize=(14,6))

plt.title('Fatalities timeline')

sns.lineplot(y=covid_train['Fatalities'], x=covid_train['Date'])
plt.figure(figsize=(14,6))

plt.title('Compare Confirmed Cases and Fatalities in day')

pl = sns.lineplot(y=covid_train['ConfirmedCases'], x=covid_train['Date'])

pl2 = pl.twinx()

sns.lineplot(y=covid_train['FatalitiesInDay'], x=covid_train['Date'], ax=pl2, color="r")
plt.figure(figsize=(14,6))

plt.title('Fatalities in Day')

sns.lineplot(y=covid_train['FatalitiesInDay'], x=covid_train['Date'])
plt.figure(figsize=(14,6))

plt.title('Confirmed Cases in Day')

sns.lineplot(y=covid_train['ConfirmedCasesInDay'], x=covid_train['Date'])
'''

Create new database merge data by Country_Region

'''

occurence_per_country=[]

for country in covid_train['Country_Region'].unique():

    country_data = covid_train[covid_train['Country_Region']==country]

    register_date = country_data[country_data['ConfirmedCases']>0]

    if len(register_date)>0:

        date_cc = register_date.iloc[0]['Date']

    else:

        data_cc = np.nan

        

    register_date = country_data[country_data['Fatalities']>0]

    if len(register_date):

        date_ft = register_date.iloc[0]['Date']

    else:

        date_ft = np.nan

    #get all max of province state and sum to have the total in country

    max_cc=0

    max_ft=0

    for province_state in country_data['Province_State'].unique():

        if not pd.isna(province_state):

            province_state_data=country_data[country_data['Province_State']==province_state]

        else:

            province_state_data=country_data[pd.isna(country_data['Province_State'])]

        max_cc+=max(province_state_data['ConfirmedCases'])

        max_ft+=max(province_state_data['Fatalities'])

            

    country_data_new = [country,

                        date_cc, date_ft,

                        max_cc, max_ft, 

                        max(country_data['ConfirmedCasesInDay']), max(country_data['FatalitiesInDay'])]

    occurence_per_country.append(country_data_new)

occurence_per_country = pd.DataFrame(occurence_per_country)
occurence_per_country.columns = ['Country_Region', 'FirstRegisterConfirmedCase', 'FirstRegisterFatalities',

                                'ConfirmedCasesMax', 'FatalitiesMax', 'ConfirmedCasesInDayMax', 'FatalitiesInDayMax']
occurence_per_country.head()
occurence_per_country.describe()
occurence_per_country[occurence_per_country['FatalitiesMax']==max(occurence_per_country['FatalitiesMax'])]
occurence_per_country[occurence_per_country['FatalitiesMax']==min(occurence_per_country['FatalitiesMax'])].head()
largest_fatalities = occurence_per_country.nlargest(10, 'FatalitiesMax')
largest_fatalities.head()
plt.figure(figsize=(14,6))

plt.title('Top 10 Countries with more Fatalities')

sns.barplot(y=largest_fatalities['FatalitiesMax'], x=largest_fatalities['Country_Region'])
largest_confirmed_cases = occurence_per_country.nlargest(10, 'ConfirmedCasesMax')
largest_confirmed_cases.head()
plt.figure(figsize=(14,6))

plt.title("Top 10 countries with more Confirmed Cases")

sns.barplot(y=largest_confirmed_cases['ConfirmedCasesMax'], x=largest_confirmed_cases['Country_Region'])
X_features = ['Date', 'Country_Region', 'Province_State']

X = covid_train[X_features]



y_cc = covid_train['ConfirmedCases']

y_ft = covid_train['Fatalities']



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, ['Country_Region', 'Province_State'])

    ])
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



cc_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', RandomForestRegressor(n_estimators=100, random_state=0))

                             ])



ft_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', RandomForestRegressor(n_estimators=100, random_state=0))

                             ])



X_train, X_valid, y_train, y_valid = train_test_split(X, y_cc, random_state=0)

cc_pipeline.fit(X_train, y_train)

preds_cc = cc_pipeline.predict(X_valid).astype(int)



print('MAE to ConfirmedCases:', mean_absolute_error(y_valid, preds_cc))



X_train, X_valid, y_train, y_valid = train_test_split(X, y_ft, random_state=0)

ft_pipeline.fit(X_train, y_train)

preds_ft = ft_pipeline.predict(X_valid).astype(int)



print('MAE to Fatalities:', mean_absolute_error(y_valid, preds_ft))
#predict tests

test_preds_cc = cc_pipeline.predict(covid_test[X_features])

test_preds_ft = ft_pipeline.predict(covid_test[X_features])
submission = pd.DataFrame({'ForecastId': covid_test.index,'ConfirmedCases':test_preds_cc,'Fatalities':test_preds_ft})

filename = 'submission.csv'



submission.to_csv(filename,index=False)