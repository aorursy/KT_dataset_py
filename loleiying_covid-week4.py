import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from datetime import date

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")

submit = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")

c2 = pd.read_csv("/kaggle/input/covid19-forecasting-metadata/region_metadata.csv")
# Steps 1 to 6

def dataTrans(df):

    # 1. Combine the Country_Region and Province_State columns into country_province.

    df.Province_State[df['Province_State'].isnull()] = '' # change the null to empty string

    df['country_province'] = df.apply(lambda x: x.Country_Region+'-' if x.Province_State == '' else x.Country_Region+'-'+x.Province_State, axis = 1)

    # 3. Chagnge the datatype of Date to datetime

    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)    

    # 4. Log transform the cases and deaths.

    #df['log_ConfirmedCases'] = df['ConfirmedCases'].apply(lambda x: np.log(x) if x > 0 else 0)

    #df['log_Fatalities'] = df['Fatalities'].apply(lambda x: np.log(x) if x > 0 else 0)

    

    ##################################################################################################

    # 2. Calculate the cumulative cases and fatalities for each country_province for the train data

    # 5.  Add column for prevCases and prevDeath

    cumCases = pd.Series()

    cumDeath = pd.Series()

    prevCases = pd.Series()

    prevDeath = pd.Series()

    

    for region in df.country_province.unique():

        cases = df.ConfirmedCases[df.country_province==region].cumsum() # cumulative sum of log cases

        death = df.Fatalities[df.country_province==region].cumsum() # cumulative sum of log deaths

        cumCases = pd.concat([cumCases,cases])

        cumDeath = pd.concat([cumDeath,death])

        prevCases = pd.concat([prevCases,pd.Series([0]),cases.iloc[:len(cases)-1]])

        prevDeath = pd.concat([prevDeath,pd.Series([0]),death.iloc[:len(death)-1]])

        

        # 7.  Add a column for the days since the first cases

        date_firstCase = df.iloc[df.ConfirmedCases.to_numpy().nonzero()[0][0]].Date # date of the first cases

        df["DaySinceFirstCase"] = (df.Date - date_firstCase).dt.days # calculates the day since the first case

        df["DaySinceFirstCase"] = df['DaySinceFirstCase'].apply(lambda x: x if x>0 else 0) # change the negative values to zero



    prevCases = prevCases.reset_index(drop=True)

    prevDeath = prevDeath.reset_index(drop=True)

    df_trans = pd.concat([df,cumCases,cumDeath,prevCases,prevDeath], axis=1)

    df_trans = df_trans.rename(columns={0:'log_cumCases', 1:'log_cumDeath', 2:'log_prevCases', 3:'log_prevDeath'})

    log_transform = lambda x: np.log(x) if x>0 else 0

    df_trans['log_cumCases'] = df_trans['log_cumCases'].apply(log_transform)

    df_trans['log_cumDeath'] = df_trans['log_cumDeath'].apply(log_transform)

    df_trans['log_prevCases'] = df_trans['log_prevCases'].apply(log_transform)

    df_trans['log_prevDeath'] = df_trans['log_prevDeath'].apply(log_transform)

    return df_trans

###############################################################################################################################################



df_train = dataTrans(train)

df_train.head()
len(df_train), len(train)
# add north america and south america in the continents

northamerica = ['Antigua and Barbuda','Bahamas','Barbados','Belize','Canada','Costa Rica','Cuba', 'Dominica', \

                'Dominican Republic','El Salvador', 'Grenada', 'Guatemala','Haiti','Honduras', 'Jamaica', \

                'Mexico', 'Nicaragua', 'Panama','Saint Kitts and Nevis', 'Saint Lucia','Saint Vincent and the Grenadines',\

                'Trinidad and Tobago', 'US']

southamerica = ['Argentina','Bolivia', 'Brazil','Chile', 'Colombia', 'Ecuador','Guyana','Paraguay','Peru',\

                'Suriname','Uruguay', 'Venezuela']

#print(len(c2[c2.continent == "Americas"].Country_Region.unique()) == (len(northamerica)+len(southamerica)))

for country in c2.Country_Region.unique():

    #print(country)

    if country in northamerica:

        c2.loc[c2.Country_Region==country,['continent']] = 'North_America'

    elif country in southamerica:

        c2.loc[c2.Country_Region==country,['continent']] = 'South_America'

print(c2.continent.unique())



# change the continents to codes

c2.continent = pd.Categorical(c2.continent)

c2['continent_code'] = c2.continent.cat.codes 



# add a column for country_province

c2.Province_State[c2['Province_State'].isnull()] = '' # change the null to empty string

c2['country_province'] = c2.apply(lambda x: x.Country_Region+'-' if x.Province_State == '' else x.Country_Region+'-'+x.Province_State, axis = 1)

c2.columns
len(c2.country_province.unique()), len(df_train.country_province.unique())
# merge columns from the additional datasets to the training set and testing set

#weather.rename(columns={'country+province':'country_province'}, inplace=True)



df_train = pd.merge(df_train, c2[['country_province','lat', 'lon', 'continent_code','population', 'area', 'density']], on='country_province', how='left')

df_train.columns
len(df_train), len(train)
# making the testdata the same

def dataTrans_test(df):

    # 1. Combine the Country_Region and Province_State columns into country_province.

    df.Province_State[df['Province_State'].isnull()] = '' # change the null to empty string

    df['country_province'] = df.apply(lambda x: x.Country_Region+'-' if x.Province_State == '' else x.Country_Region+'-'+x.Province_State, axis = 1)

    # 3. Chagnge the datatype of Date to datetime

    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)    

    return df



df_test = dataTrans_test(test)

df_test = pd.merge(df_test, c2[['country_province','lat', 'lon', 'continent_code','population', 'area', 'density']], on='country_province', how='left')

df_test = df_test.merge(df_train[['Date', 'ConfirmedCases',

                                   'Fatalities', 'country_province', 

                                   'DaySinceFirstCase', 'log_cumCases', 'log_cumDeath',

                                   'log_prevCases', 'log_prevDeath']], on=['country_province','Date'],how='left')



# calculate the DaySinceFirstCase column

s = df_test.groupby('country_province').DaySinceFirstCase.cumcount()

s1 = (df_test.DaySinceFirstCase - s).groupby(df_test.country_province).transform('first')

df_test['DaySinceFirstCase'] = s1 + s



df_test.columns

df_test.head()
columns = df_train.columns.tolist()

columns = ['ForecastId'] + columns[1:]

columns

df_test = df_test.loc[:,columns]

df_test.head()
len(df_train), len(train)
len(df_test), len(test)
def modelTraining(model, df_train, df_test, predictors, targets):

    # train the models separately for each region

    for region in df_train.country_province.unique():

        firstIndex = df_test[(df_test.country_province == region)].index[0]

        lastIndex = df_test[(df_test.country_province == region)].index[-1]

        lastValidIndex = df_test[(df_test.country_province == region)].index[-1]

        lastValideDate = df_test[(df_test.country_province == region)].Date.min()

        while (lastValidIndex < lastIndex):

            # assign the latest cumulated cases and deaths to the next day as previous cases and deaths

            data = df_train.loc[(df_train.Date == lastValidDate) & (df_train.country_province == region) , 'log_cumCases':'log_cumDeath']

            df_test.loc[lastValidIndex+1,'log_prevCases':'log_prevDeath'] = data[0], data[1]



            # train the models separately for log_cumCases and log_cumDeath

            for target in targets:

                X_train = df_train[df_train.country_province == region][predictors]

                y_train = df_train[df_train.country_province == region][target]



                X_test = df_test.loc[lastValidIndex+1, predictors].values.reshape(1,-1)

                

                model.fit(X_train, y_train)

                

                df_test.loc[lastValidIndex+1, target] = model.predict(X_test)[0]

                

            # update the last valid index where the cumulative cases is not null

            lastValidIndex = df_test[(df_test.country_province == region) & (df_test.log_cumCases.notnull())].index[-1]

            

    # change the predicted outcome from log scale to linear scale

    df_test['cumCases'] = np.exp(df_test['log_cumCases'])

    df_test['cumDeaths'] = np.exp(df_test['log_cumDeath'])

    # calculate the daily new cases

    df_test['ConfirmedCases_predicted'] = df_test.groupby('country_province').cumCases.diff()

    df_test['Fatalities_predicted'] = df_test.groupby('country_province').cumDeaths.diff()

    # fill the nan in daily cases and deaths with predicted values

    df_test['ConfirmedCases'] = df_test['ConfirmedCases'].fillna(df_test['ConfirmedCases_predicted'])

    df_test['Fatalities'] = df_test['Fatalities'].fillna(df_test['Fatalities_predicted'])

    

    return df_test



from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression

from sklearn import metrics



rfr = RandomForestRegressor(n_estimators=100)

ridge = Ridge(alpha=1.0)

linear = LinearRegression()



predictors =['DaySinceFirstCase','log_prevCases','log_prevDeath', 'lat', 'lon', 'continent_code', 'population', 'area', 'density']

targets = ['log_cumCases','log_cumDeath']



# subset df_train into train and test into 7:3

dates = np.sort(df_train['Date'].unique())

training = df_train[df_train['Date'] <= df_test.Date.min()]

testing = df_train[df_train['Date'] > dates[int(len(dates)*0.7)]]

validation = testing.copy()

testing.loc[:,'log_cumCases':'log_prevDeath'] = np.nan



#for model,model_name in zip([rfr, ridge, linear],['Random Forest Regression', 'Ridge Regression', 'Linear Regression']):

    #result = modelTraining(model, training, df_test, predictors, targets)

    #mse = np.mean((result['ConfirmedCases'] - validation['ConfirmedCases'])**2 + (result['Fatalities'] - validation['Fatalities'])**2)

    #print('Model:',model_name)

    #print('Mean Square Error:', mse)

result_linear = modelTraining(linear, training, df_test, predictors, targets)
cols = ['ForecastId', 'ConfirmedCases_predicted', 'Fatalities_predicted']

submission = result_linear.loc[:,cols]

submission = submission.rename(columns={'ConfirmedCases_predicted':'ConfirmedCases', 'Fatalities_predicted':'Fatalities'})

submission.describe()

submission = submission.to_csv('submission.csv',index=False)