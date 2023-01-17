import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import datetime as dt



from sklearn import preprocessing



from warnings import filterwarnings

filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

PATH_WEEK2 = '/kaggle/input/covid19-global-forecasting-week-2'

train = pd.read_csv(f'{PATH_WEEK2}/train.csv')

test = pd.read_csv(f'{PATH_WEEK2}/test.csv')

train.head()
PATH_POPULATION = '/kaggle/input/population-by-country-2020'

population = pd.read_csv(f'{PATH_POPULATION}/population_by_country_2020.csv')

population.head()
train['Province_State'].fillna(train['Country_Region'],inplace = True)

test['Province_State'].fillna(test['Country_Region'],inplace = True)



train.rename(columns={'Country_Region':'Country'}, inplace=True)

test.rename(columns={'Country_Region':'Country'}, inplace=True)



train.rename(columns={'Province_State':'State'}, inplace=True)

test.rename(columns={'Province_State':'State'}, inplace=True)

train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)

train['year'] = train['Date'].dt.year

train['month'] = train['Date'].dt.month

train['week'] = train['Date'].dt.week

train['day'] = train['Date'].dt.day

train['dayofweek'] = train['Date'].dt.dayofweek

train.loc[:, 'Date'] = train.Date.dt.strftime("%m%d")

train["Date"]  = train["Date"].astype(int)

train['Date'] -= 122 #because first_date = 122





test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)

test['year'] = test['Date'].dt.year

test['month'] = test['Date'].dt.month

test['week'] = test['Date'].dt.week

test['day'] = test['Date'].dt.day

test['dayofweek'] = test['Date'].dt.dayofweek

test.loc[:, 'Date'] = test.Date.dt.strftime("%m%d")

test["Date"]  = test["Date"].astype(int)

test['Date'] -= 122 #because first_date = 122

test['Date'] -= 122
#merge_key

population.rename(columns={'Country (or dependency)':'Country'}, inplace=True)



master_countries = train.Country.unique().tolist()

branch_countries = df_Population.Country.unique().tolist()



print("branch")

for country in master_countries:

    if country not in branch_countries:

        print (country)

#print("\nmaster")

#for country in master_countries:

 #   if country not in master_countries:

  #      print (country)
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

train.replace({'Country': renameCountryNames}, inplace=True)

test.replace({'Country': renameCountryNames}, inplace=True)
population.loc[population['Med. Age']=='N.A.', 'Med. Age'] = population.loc[population['Med. Age']!='N.A.', 'Med. Age'].mode()[0]

population.loc[population['Urban Pop %']=='N.A.', 'Urban Pop %'] = population.loc[population['Urban Pop %']!='N.A.', 'Urban Pop %'].mode()[0]

population.loc[population['Fert. Rate']=='N.A.', 'Fert. Rate'] = population.loc[population['Fert. Rate']!='N.A.', 'Fert. Rate'].mode()[0]

population.loc[:, 'Migrants (net)'] = population.loc[:, 'Migrants (net)'].fillna(0)

population['Yearly Change'] = population['Yearly Change'].str.rstrip('%')

population['World Share'] = population['World Share'].str.rstrip('%')

population['Urban Pop %'] = population['Urban Pop %'].str.rstrip('%')

population = population.astype({"Net Change": int,"Density (P/Km²)": int,"Population (2020)": int,"Land Area (Km²)": int,"Yearly Change": float,"Urban Pop %": int,"Fert. Rate": float,"Med. Age": int,"World Share": float, "Migrants (net)": float,})



# As the Country value "Diamond Princess" is a CRUISE, we replace the population 

population = population.append(pd.Series(['Diamond Princess', 3500, 0, 0, 0, 0, 0.0, 1, 30, 0, 0.0], index= population.columns ), ignore_index=True)
train = train.merge(population, how='left', left_on='Country', right_on='Country')

test = test.merge(population, how='left', left_on='Country', right_on='Country')
train.columns, train.shape
le = preprocessing.LabelEncoder()



train.Country = le.fit_transform(train.Country)

train.State = le.fit_transform(train.State)

test.Country = le.fit_transform(test.Country)

test.State = le.fit_transform(test.State)
val = train[train["Date"] > 196].reset_index(drop = True)

train = train[train["Date"] <= 196].reset_index(drop = True)



x_train = train.drop(['Id', 'ConfirmedCases','Fatalities'], axis = 1)

y_train_1 = train["ConfirmedCases"]

y_train_2 = train["Fatalities"]

x_val = val.drop(['Id', 'ConfirmedCases','Fatalities'], axis = 1)

y_val_1 = val["ConfirmedCases"]

y_val_2 = val["Fatalities"]

train.shape , val.shape
from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor
def evaluate(predict,actual):                                           

    return np.sqrt(np.mean(np.square(np.log(predict+1) - np.log(actual+1))))
xgb_model_1= XGBRegressor(n_estimators = 4000)

xgb_model_1.fit(x_train,y_train_1)

y_pred_1 = xgb_model_1.predict(x_val)

xgb_model_2= XGBRegressor(n_estimators = 5000)

xgb_model_2.fit(x_train,y_train_2)

y_pred_2 = xgb_model_2.predict(x_val)



evaluate(y_pred_1,y_val_1), evaluate(y_pred_2,y_val_2),evaluate(y_pred_1,y_val_1) + evaluate(y_pred_2,y_val_2)
random_forest_1=RandomForestRegressor(bootstrap=True, 

            max_depth=25, max_features='auto', max_leaf_nodes=None,

            min_samples_leaf=1, min_samples_split=15,

            min_weight_fraction_leaf=0.0, n_estimators=150, 

            random_state=0, verbose=0, warm_start=False)

random_forest_1.fit(x_train,y_train_1)

random_forest_1 = xgb_model_1.predict(x_val)



random_forest_2=RandomForestRegressor(bootstrap=True, 

            max_depth=25, max_features='auto', max_leaf_nodes=None,

            min_samples_leaf=1, min_samples_split=15,

            min_weight_fraction_leaf=0.0, n_estimators=150, 

            random_state=0, verbose=0, warm_start=False)

random_forest_2.fit(x_train,y_train_2)

y_pred_2 = random_forest_2.predict(x_val)

evaluate(y_pred_1,y_val_1), evaluate(y_pred_2,y_val_2),evaluate(y_pred_1,y_val_1)+evaluate(y_pred_2,y_val_2)
x_test = test.drop(['ForecastId',"ConfirmedCases","Fatalities"],axis = 1)

ans_1 =xgb_model_1.predict(x_test)

ans_2 = xgb_model_2.predict(x_test)

submit = pd.DataFrame()

submit["ForecastId"] = test.ForecastId

submit["ConfirmedCases"] = ans_1

submit["Fatalities"] = ans_2

submit.to_csv("submission.csv",index = False)