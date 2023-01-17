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
PATH_WEEK2 = '/kaggle/input/covid19-global-forecasting-week-2'

df_Train = pd.read_csv(f'{PATH_WEEK2}/train.csv')
df_test = pd.read_csv(f'{PATH_WEEK2}/test.csv')
PATH_POPULATION = '/kaggle/input/population-by-country-2020'

df_Population = pd.read_csv(f'{PATH_POPULATION}/population_by_country_2020.csv')
df_Train.iloc[np.r_[0:5, -6:-1], :]
df_test.iloc[np.r_[0:5, -6:-1], :]
df_Population.iloc[np.r_[0:5, -6:-1], :]
df_Train.rename(columns={'Country_Region':'Country'}, inplace=True)
df_test.rename(columns={'Country_Region':'Country'}, inplace=True)

df_Train.rename(columns={'Province_State':'State'}, inplace=True)
df_test.rename(columns={'Province_State':'State'}, inplace=True)
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
df_Train.loc[: , ['Country', 'ConfirmedCases', 'Fatalities']].groupby(['Country']).max().sort_values(by='ConfirmedCases', ascending=False).reset_index()[:15].style.background_gradient(cmap='rainbow')
df_Train['Date'] = pd.to_datetime(df_Train['Date'], infer_datetime_format=True)
df_plot = df_Train.loc[: , ['Date', 'Country', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country']).max().reset_index()

df_plot.loc[:, 'Date'] = df_plot.Date.dt.strftime("%Y-%m-%d")
df_plot.loc[:, 'Size'] = np.power(df_plot["ConfirmedCases"]+1,0.3)-1 #np.where(df_plot['Country'].isin(['China', 'Italy']), df_plot['ConfirmedCases'], df_plot['ConfirmedCases']*300)
fig = px.scatter_geo(df_plot,
                     locations="Country",
                     locationmode = "country names",
                     hover_name="Country",
                     color="ConfirmedCases",
                     animation_frame="Date", 
                     size='Size',
                     #projection="natural earth",
                     title="Rise of Coronavirus Confirmed Cases")
fig.show()
top_10_countries = df_Train.groupby('Country')['Date', 'ConfirmedCases', 'Fatalities'].max().sort_values(by='ConfirmedCases', ascending=False).reset_index().loc[:, 'Country'][:10]
df_plot = df_Train.loc[df_Train.Country.isin(top_10_countries), ['Date', 'Country', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country']).max().reset_index()

fig = px.line(df_plot, x="Date", y="ConfirmedCases", color='Country')
fig.update_layout(title='No.of Confirmed Cases per Day for Top 10 Countries',
                   xaxis_title='Date',
                   yaxis_title='No.of Confirmed Cases')
fig.show()
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
df_test.tail()
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
df_Population.describe()
df_Population[df_Population['Population (2020)'] <= 5000]
df_Train = df_Train.merge(df_Population, how='left', left_on='Country', right_on='Country')
df_test = df_test.merge(df_Population, how='left', left_on='Country', right_on='Country')
df_Train.info()
df_test.info()
df_Population.info()
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

X_Train.head()
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

X_Test.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X_Train.Country = le.fit_transform(X_Train.Country)
X_Train['State'] = le.fit_transform(X_Train['State'])

X_Train.head()
X_Test.Country = le.fit_transform(X_Test.Country)
X_Test['State'] = le.fit_transform(X_Test['State'])

X_Test.head()
df_train.head()
df_train.loc[df_train.Country == 'Afghanistan', :]
df_test.tail()
X_Train.head()
X_Train.iloc[3990:4020]
from warnings import filterwarnings
filterwarnings('ignore')
'''
from sklearn.model_selection import GridSearchCV
import time
param_grid = {'n_estimators': [1000]}
#param_grid = {'nthread':[4], 'objective':['reg:linear'], 'learning_rate': [.03, 0.05], 'max_depth': [5, 6], 'min_child_weight': [4], 'silent': [1], 'subsample': [0.7], 'colsample_bytree': [0.7], 'n_estimators': [500, 1000]}

def gridSearchCV(model, X_Train, y_Train, param_grid, cv=10, scoring='neg_mean_squared_error'):
    start = time.time()
    
    grid_cv = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    grid_cv.fit(X_Train, y_Train)
    
    print (f'{type(model).__name__} Hyper Paramter Tuning took a Time: {time.time() - start}')
    print (f'Best {scoring}: {grid_cv.best_score_}')
    print ("Best Hyper Parameters:\n{}".format(grid_cv.best_params_))
    
    return grid_cv.best_estimator_
'''
'''
from xgboost import XGBRegressor

model = XGBRegressor()

model1 = gridSearchCV(model, X_Train, y1_Train, param_grid, 10, 'neg_mean_squared_error')
model2 = gridSearchCV(model, X_Train, y2_Train, param_grid, 10, 'neg_mean_squared_error')
'''
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
def applyMLA(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    time2 = end_time-start_time

    predictions = model.predict(X_test)
    RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    
    return [RMSE_test, time2]
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
from xgboost import XGBRegressor
#from sklearn.ensemble import RandomForestRegressor

countries = X_Train.Country.unique().tolist()

#models_C = {}
#models_F = {}

df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

for country in countries:
    states = X_Train.loc[X_Train.Country == country, :].State.unique().tolist()
    #print(country, states)
    # check whether string is nan or not
    for state in states:
        X_Train_CS = X_Train.loc[(X_Train.Country == country) & (X_Train.State == state), :]
        
        y1_Train_CS = X_Train_CS.loc[:, 'ConfirmedCases']
        y2_Train_CS = X_Train_CS.loc[:, 'Fatalities']
        #y1_Train_CS_log = np.log1p(X_Train_CS.loc[:, 'ConfirmedCases'])
        #y2_Train_CS_log = np.log1p(X_Train_CS.loc[:, 'Fatalities'])
        
        X_Train_CS.drop(columns=['Id', 'ConfirmedCases', 'Fatalities'], axis=1, inplace=True)
        
        X_Train_CS_PCA = X_Train_CS#applyPCA(18, X_Train_CS)
        
        #X_Train_CS.Country = le.fit_transform(X_Train_CS.Country)
        #X_Train_CS['State'] = le.fit_transform(X_Train_CS['State'])
        
        X_Test_CS = X_Test.loc[(X_Test.Country == country) & (X_Test.State == state), :]
        
        X_Test_CS_Id = X_Test_CS.loc[:, 'ForecastId']
        X_Test_CS.drop(columns=['ForecastId'], axis=1, inplace=True)
        
        X_Test_CS_PCA = X_Test_CS#applyPCA(18, X_Test_CS)
        
        #X_Test_CS.Country = le.fit_transform(X_Test_CS.Country)
        #X_Test_CS['State'] = le.fit_transform(X_Test_CS['State'])
        
        #models_C[country] = gridSearchCV(model, X_Train_CS, y1_Train_CS, param_grid, 10, 'neg_mean_squared_error')
        #models_F[country] = gridSearchCV(model, X_Train_CS, y2_Train_CS, param_grid, 10, 'neg_mean_squared_error')
        
        model1 = XGBRegressor(n_estimators=1250)
        #model1 = RandomForestRegressor(bootstrap=True, max_depth=80, max_features=3, min_samples_leaf=5, min_samples_split=12, n_estimators=100)
        model1.fit(X_Train_CS_PCA, y1_Train_CS)
        y1_pred = model1.predict(X_Test_CS_PCA)
        #model1.fit(X_Train_CS_PCA, y1_Train_CS_log)
        #y1_pred = np.expm1(model1.predict(X_Test_CS_PCA))
        
        model2 = XGBRegressor(n_estimators=1000)
        #model2 = RandomForestRegressor(bootstrap=True, max_depth=80, max_features=3, min_samples_leaf=5, min_samples_split=12, n_estimators=100)
        model2.fit(X_Train_CS_PCA, y2_Train_CS)
        y2_pred = model2.predict(X_Test_CS_PCA)
        #model2.fit(X_Train_CS_PCA, y2_Train_CS_log)
        #y2_pred = np.expm1(model2.predict(X_Test_CS_PCA))
        
        df = pd.DataFrame({'ForecastId': X_Test_CS_Id, 'ConfirmedCases': y1_pred, 'Fatalities': y2_pred})
        df_out = pd.concat([df_out, df], axis=0)
    # Done for state loop
# Done for country Loop
df_out.ForecastId = df_out.ForecastId.astype('int')
df_out[3990:4020]
df_out.tail()
df_out.to_csv('submission.csv', index=False)
