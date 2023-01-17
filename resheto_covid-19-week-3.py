import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import plotly.express as px

import plotly.offline as py

import plotly.graph_objects as go

py.init_notebook_mode()

DT=1

optimize_model=False 



optimize_model_2=False



Make_submission=True 

#n_estimators=450 #200 #400 #500  #1500

#max_depth=4 #2 #4 #12  #8





#{'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 800}



max_depth=7

min_child_weight=2

n_estimators=600

learning_rate=0.1





# {'max_depth': 4, 'min_child_weight': 5, 'n_estimators': 500}



max_depth_2=5

min_child_weight_2=5

n_estimators_2=600

learning_rate_2=0.1
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

train.head()
have_states=train[train['Province_State'].notna()].groupby(['Country_Region'], sort=False)['Province_State'].nunique()

print(have_states)
def add_location(df_old):

    df=df_old.copy()

    df['Date']=pd.to_datetime(df['Date'])

    df['Country_Region']=df['Country_Region'].fillna('')

    df['Province_State']=df['Province_State'].fillna('')

    df['location']=df['Province_State'].astype('str')+" "+df['Country_Region'].astype('str')

    

    df['Island']=False 

    df.loc[df['Province_State'].str.contains("Islan"),'Island']=True

    df.loc[df['Province_State'].isin(['French Polynesia',

       'Guadeloupe', 'Martinique', 'Mayotte', 'New Caledonia', 'Reunion',

       'Saint Barthelemy','Anguilla', 'Bermuda','Isle of Man', 'Montserrat','Aruba',

       'Curacao']),'Island']=True 

    

    df.loc[df['Country_Region'].isin(['Diamond Princess', 'MS Zaandam']),'Island']=True 

    

    return df
train=add_location(train)
train[(train['Province_State']!="") & (train['Island']==False)]['Province_State'].unique()
max_cases_old=train[train['Date']<'2020-04-01'].groupby(['location'], sort=False)['ConfirmedCases'].max()

max_cases=train.groupby(['location'], sort=False)['ConfirmedCases'].max()

print("Now: {}\r\nSeven Days ago: {}".format(len(max_cases[max_cases<5]),len(max_cases_old[max_cases_old<5])))
max_cases[max_cases<5]
train.set_index('location',inplace=True)



train['day_of_year']=train['Date'].dt.dayofyear

train['day_of_week']=train['Date'].dt.dayofweek





first_day=train[(train['ConfirmedCases']>0)].groupby(['location'], sort=False)['day_of_year'].min()

first_day.rename('first_day',inplace=True)



day_ten=train[(train['ConfirmedCases']>10)].groupby(['location'], sort=False)['day_of_year'].min()

day_ten.rename('day_ten',inplace=True)
def add_days_passed(df_old,first_day,day_ten):

    df=df_old.copy()

    df=pd.concat([df,first_day],axis=1,join='inner')

    

    df['days_passed']=df['day_of_year']-df['first_day']

    df.drop(columns=['first_day'],inplace=True)

    df.loc[df['days_passed']<0,'days_passed']=-1

    

    df=df.merge(day_ten,left_index=True,right_index=True,how="outer")

    

    df['days_passed_10']=df['day_of_year']-df['day_ten']

    df.loc[df['day_ten'].isna(),'days_passed_10']=-1

    df.loc[df['days_passed_10']<0,'days_passed_10']=-1

    df.drop(columns=['day_ten'],inplace=True)

   

    df['location']=df.index

    

    df.loc[df['location']=='Hubei China','days_passed']+=35

    df.loc[df['location']=='Hubei China','days_passed_10']+=22

    

    df.set_index('Id',inplace=True)

    df['Id']=df.index

    return df

train=add_days_passed(train,first_day,day_ten)



train.head()
country_stat=pd.read_csv('../input/countryinfo/covid19countryinfo.csv')

country_stat = country_stat[country_stat['region'].isnull()] 



us_stat=pd.read_csv('../input/covid19-state-data/COVID19_state.csv')

us_stat.rename(columns={'State':'location','Population':'pop','Pop Density':'density','Smoking Rate':'smokers'},inplace=True)

us_stat['location']+=" US"

us_stat.set_index('location',inplace=True)



def add_country_stat(old_df,country_stat,us_stat):

    df=old_df.copy()

    df=df.merge(country_stat[['country','pop','medianage','sex65plus','lung','smokers','density']],left_on=['Country_Region'],right_on=['country'],how='left')

    df.drop(columns=['country'],inplace=True)

    

    df['pop']=df['pop'].fillna(1000)

    df['pop']=df['pop'].apply(lambda x: int(str(x).replace(',', '')))

    #df['gdp2019']=df['gdp2019'].fillna(0)

    #df['gdp2019']=df['gdp2019'].apply(lambda x: int(str(x).replace(',', '')))

    #df['gdp2019']=df['gdp2019']/df['pop']

    

    

    df['density']=df['density'].fillna(0)

    df['medianage']=df['medianage'].fillna(0)

    #df['sexratio']=df['sexratio'].fillna(1)

    df['sex65plus']=df['sex65plus'].fillna(1)

    df['lung']=df['lung'].fillna(24)

    df['smokers']=df['smokers'].fillna(24)

    #df['lung']=df['lung']*df['pop']

    

    df.set_index('location',inplace=True)

    df.update(us_stat[['pop','density','smokers']])

    

    df['location']=df.index

    df.set_index('Id',inplace=True)

    df['Id']=df.index

    

    

    

    return df

    



train=add_country_stat(train,country_stat,us_stat)
country_stat.info()
us_stat.info()
weather_info=pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv")

weather_info['dt']=pd.to_datetime(weather_info['dt'])

weather_info=weather_info[weather_info['dt']>'2000-12-30']

weather_info['month']=weather_info['dt'].dt.month

weather_info.drop(weather_info[weather_info['Country'].isin(

    ['Denmark', 'France', 'Netherlands','United Kingdom'])].index,axis=0,inplace=True)



weather_info.replace(

   ['Denmark (Europe)', 'France (Europe)', 'Netherlands (Europe)', 'United Kingdom (Europe)'],

   ['Denmark', 'France', 'Netherlands', 'United Kingdom'],inplace=True)



weather_info.replace({

    'Antigua And Barbuda':'Antigua and Barbuda',

    'Bosnia And Herzegovina':'Bosnia and Herzegovina',

    'Congo (Democratic Republic Of The)':'Congo (Kinshasa)',

    'Congo':'Congo (Brazzaville)',

    'Palestina':'West Bank and Gaza',

    'Cape Verde':'Cabo Verde',

    "Côte D'Ivoire":"Cote d'Ivoire",

    'Trinidad And Tobago':'Trinidad and Tobago',

    'Saint Kitts And Nevis':'Saint Kitts and Nevis',

    'Czech Republic':'Czechia',

    'Swaziland':'Eswatini',

    'Guinea Bissau':'Guinea-Bissau',

    'South Korea':'Korea, South', 

    'Macedonia':'North Macedonia',

    'Saint Vincent And The Grenadines':'Saint Vincent and the Grenadines',

    'Taiwan':'Taiwan*', 

    'Timor Leste':'Timor-Leste',

    'United States':'US'

},inplace=True)

weather_country=weather_info.groupby(['Country','month'])['AverageTemperature'].mean()





state_weather_info=pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv")

state_weather_info.replace({'United States':'US','Georgia (State)':'Georgia','District Of Columbia':'District of Columbia'},inplace=True)

state_weather_info['dt']=pd.to_datetime(state_weather_info['dt'])

state_weather_info=state_weather_info[state_weather_info['dt']>'2000-12-30']

state_weather_info['month']=state_weather_info['dt'].dt.month

weather_state=state_weather_info[state_weather_info['Country'].isin(have_states.index)].groupby(['Country','State','month'])['AverageTemperature'].mean()
def add_temperature(old_df,weather_country,weather_state):

    df=old_df.copy()

    df['Month']=df['Date'].dt.month

    df=df.merge(weather_country,how="left",left_on=['Country_Region','Month'],right_index=True)

    df=df.merge(weather_state,how="left",left_on=['Country_Region','Province_State','Month'],right_index=True)

    df.loc[df['AverageTemperature_y'].notnull(),'AverageTemperature_x']=df['AverageTemperature_y']

    df.drop(columns=['AverageTemperature_y','Month'],inplace=True)

    df.rename(columns={'AverageTemperature_x':'AverageTemperature'},inplace=True)

    

    return df



train=add_temperature(train,weather_country,weather_state)

    
train[train['AverageTemperature'].isnull()]['Country_Region'].unique()
train.head()
border_info=pd.read_csv("../input/country-borders/border_info.csv")

#border_info.drop(columns=["country_code","country_border_code"],inplace=True)

border_info.replace({'United States of America':'US',

                    'United Kingdom of Great Britain and Northern Ireland':'United Kingdom',

                    'Bolivia (Plurinational State Of)':'Bolivia',

                    'Brunei Darussalam':'Brunei',

                    'Gambia (the)':'Gambia',

                     'Congo (the Democratic Republic of the)':'Congo (Kinshasa)',

                     'Cote d’Ivoire':"Cote d'Ivoire",

                     "Iran (Islamic Republic of)":'Iran',

                     "Korea (the Republic of)":'Korea, South',

                    "Lao People's Democratic Republic":'Laos',

                     "Moldova (the Republic of)":'Moldova',

                     'Myanmar':'Burma',

                     'Palestine, State of':'West Bank and Gaza',

                     "Russian Federation":'Russia',

                    "Syrian Arab Republic":'Syria',

                     "Taiwan (Province of China)":'Taiwan*',

                    "Tanzania (the United Republic of)":'Tanzania',

                     "Venezuela (Bolivarian Republic of)":'Venezuela',

                     "Viet Nam":'Vietnam'},inplace=True)

border_info=border_info.fillna("")

#border_info.to_csv("border_info.csv")

#set(border_info['country_name'].unique()).difference(set(train['Country_Region'].unique()))

set(train['Country_Region'].unique()).difference(set(border_info['country_name'].unique()))
from itertools import product as it_product

def expand_grid(data_dict):

  rows = it_product(*data_dict.values())

  return pd.DataFrame.from_records(rows, columns=data_dict.keys())
skel=expand_grid({'Index':border_info.index,'Date':train['Date'].unique()})



country_info=train.groupby(['Date','Country_Region'])['ConfirmedCases'].sum()



skel=skel.merge(border_info, how='inner', left_on=['Index'],right_index=True)

skel=skel.merge(country_info, how='inner', 

                left_on=['Date','country_border_name'],right_on=['Date','Country_Region'])
from datetime import timedelta

skel['Date']=skel['Date']+timedelta(days=DT)

border_cases=skel.groupby(['country_name','Date'])['ConfirmedCases'].sum()

len(skel['country_name'].unique())
train=train.merge(border_cases, how='left', left_on=['Country_Region','Date'],right_on=['country_name','Date'])

train['ConfirmedCases_y']=train['ConfirmedCases_y'].fillna(0)

train.rename(columns={'ConfirmedCases_y':'ConfirmedCases_neighbors','ConfirmedCases_x':'ConfirmedCases'},inplace=True)
big_train = pd.concat([train,pd.get_dummies(train['location'], prefix='loc')],axis=1)

big_train['ConfirmedCases_neighbors']=np.log1p(big_train['ConfirmedCases_neighbors'])

big_train.reset_index(inplace=True)

big_train.drop(columns=["Id"],inplace=True)
big_train.shape
def df_add_deltas(df_old):

    df=df_old.copy()

    df=df.sort_values(by=['location', 'Date'])

    df['d_ConfirmedCases'] = df.groupby(['location'])['ConfirmedCases'].diff()

    df['d_Fatalities'] = df.groupby(['location'])['Fatalities'].diff()

    df.loc[df['d_Fatalities']<0,'d_Fatalities']=0

    df.loc[df['d_ConfirmedCases']<0,'d_ConfirmedCases']=0

    

    df['prev_ConfirmedCases']=df['ConfirmedCases']-df['d_ConfirmedCases']

    df['prev_Fatalities']=df['Fatalities']-df['d_Fatalities']

    

    #df['prev_ConfirmedCases']=np.log1p(df['prev_ConfirmedCases'])

    #df['prev_Fatalities']=np.log1p(df['prev_Fatalities'])

    

    df['growth_ConfirmedCases']=df['d_ConfirmedCases']/(df['prev_ConfirmedCases']+1)

    df['growth_Fatalities']=df['d_Fatalities']/(df['prev_Fatalities']+1)

    

    df['growth_ConfirmedCases']=np.log1p(df['growth_ConfirmedCases'])

    df['growth_Fatalities']=np.log1p(df['growth_Fatalities'])

    

    df.drop(columns=['prev_ConfirmedCases','prev_Fatalities'], inplace=True)

    

    

    first_day_stat=df[df['Date']=='2020-01-22']

    df.drop(df[df['Date']=='2020-01-22'].index, inplace=True)

    

    return df,first_day_stat

    
big_train,first_day_stat=df_add_deltas(big_train)
big_train.reset_index(inplace=True,drop=True)
X=big_train.drop(columns=['Province_State','Country_Region','Date','ConfirmedCases','Fatalities','location',

                          'd_ConfirmedCases','d_Fatalities','growth_ConfirmedCases','growth_Fatalities'])



y=big_train['d_ConfirmedCases']

y_2=big_train['d_Fatalities']
max_day=X['day_of_year'].max()

mask_train=X['day_of_year']<max_day-DT+1

mask_test=X['day_of_year']>=max_day-DT+1
X_train=X[mask_train]

X_test=X[mask_test]





y_train=y[mask_train]

y_test=y[mask_test]



y_train_2=y_2[mask_train]

y_test_2=y_2[mask_test]
X_test['day_of_year'].nunique()
import seaborn as sns

from matplotlib import pyplot as plt



corr = big_train[['d_ConfirmedCases','d_Fatalities','days_passed','ConfirmedCases_neighbors','pop',

                  'medianage','sex65plus','lung','smokers','density','Island','growth_ConfirmedCases',

                  'growth_Fatalities','AverageTemperature','days_passed_10'#,'prev_ConfirmedCases','prev_Fatalities'

                 ]].corr("spearman")

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(12,12))

    ax = sns.heatmap(corr, annot=True,cmap="YlGnBu",vmax=.3, square=True, linewidths=.3)

plt.show()
len(big_train[(big_train['growth_ConfirmedCases']==0)]['growth_ConfirmedCases'])
len(big_train[(big_train['growth_ConfirmedCases']>0)]['growth_ConfirmedCases'])
big_train[(big_train['growth_ConfirmedCases']>0) & (big_train['growth_ConfirmedCases']<=1)]['growth_ConfirmedCases'].hist(bins=50)
fig1 = px.scatter(big_train[(big_train['growth_ConfirmedCases']<2) & (big_train['Country_Region']=='Italy')],x='Date',y='growth_ConfirmedCases')

fig1.show()
fig1 = px.scatter(big_train[(big_train['Country_Region']=='Italy')],x='Date',y='d_ConfirmedCases')

fig1.show()
X_train.drop(columns=['day_of_year'],inplace=True)  #including day of year makes things worse RMLSE goes up from 0.49 to 0.7

X_test.drop(columns=['day_of_year'],inplace=True)   #including day of year makes things worse RMLSE goes up from 0.49 to 0.7



X_train.drop(columns=['day_of_week'],inplace=True)  #including day of week makes things worse RMLSE goes up from 0.49 to 0.57

X_test.drop(columns=['day_of_week'],inplace=True)   #including day of week makes things worse RMLSE goes up from 0.49 to 0.57



X.drop(columns=['day_of_year'],inplace=True)  

X.drop(columns=['day_of_week'],inplace=True)   



X.drop(columns=['index'],inplace=True)   

X_train.drop(columns=['index'],inplace=True)

X_test.drop(columns=['index'],inplace=True)
# Best: -1.094395 using {'max_depth': 6, 'n_estimators': 400}



import xgboost as xgb

from sklearn.model_selection import GridSearchCV



if optimize_model:



    model = xgb.XGBRegressor(random_state=42)

    n_estimators_grid = [400,600,800,1000]

    max_depth_grid = [6]

    param_grid = dict(max_depth=max_depth_grid, n_estimators=n_estimators_grid)

    grid_search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=[(X[mask_train].index,X[mask_test].index)], verbose=1)

    grid_result = grid_search.fit(X,np.log1p(y))

    # summarize results

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    print(grid_result.cv_results_)
#Best: -1.039208 using {'max_depth': 6, 'n_estimators': 200}





if optimize_model:



    model = xgb.XGBRegressor(random_state=42)

    n_estimators_grid = [100,200,300,400]

    max_depth_grid = [6]

    param_grid = dict(max_depth=max_depth_grid, n_estimators=n_estimators_grid)

    grid_search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=[(X[mask_train].index,X[mask_test].index)], verbose=1)

    grid_result = grid_search.fit(X,np.log1p(y))

    # summarize results

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    print(grid_result.cv_results_)
#Best: -0.968577 using {'max_depth': 7, 'min_child_weight': 5, 'n_estimators': 200}



#Best: -0.944948 using {'max_depth': 6, 'min_child_weight': 3, 'n_estimators': 200}



if optimize_model:



    model = xgb.XGBRegressor(random_state=42)



    max_depth_grid = [5,6,7]

    min_child_weight_grid =[1,3,5,7]

    n_estimators_grid=[200]

    

    param_grid = dict(max_depth=max_depth_grid, n_estimators=n_estimators_grid,min_child_weight=min_child_weight_grid)

    grid_search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=[(X[mask_train].index,X[mask_test].index)], verbose=1)

    grid_result = grid_search.fit(X,np.log1p(y))

    # summarize results

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    print(grid_result.cv_results_)
# Best: -0.964721 using {'max_depth': 7, 'min_child_weight': 6, 'n_estimators': 200



# Best: -0.944948 using {'max_depth': 6, 'min_child_weight': 3, 'n_estimators': 200}



if optimize_model:



    model = xgb.XGBRegressor(random_state=42)



    max_depth_grid = [7,8]

    min_child_weight_grid =[2,3,4]

    n_estimators_grid=[200]

    

    param_grid = dict(max_depth=max_depth_grid, n_estimators=n_estimators_grid,min_child_weight=min_child_weight_grid)

    grid_search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=[(X[mask_train].index,X[mask_test].index)], verbose=1)

    grid_result = grid_search.fit(X,np.log1p(y))

    # summarize results

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    print(grid_result.cv_results_)
#Best: -1.039208 using {'max_depth': 6, 'n_estimators': 200}



#Best: -0.939250 using {'max_depth': 6, 'min_child_weight': 3, 'n_estimators': 800}





if optimize_model:



    model = xgb.XGBRegressor(random_state=42,learning_rate=0.1)

    n_estimators_grid = [200,400,600,800]

    max_depth_grid = [7]

    min_child_weight_grid =[2]

    param_grid = dict(max_depth=max_depth_grid, n_estimators=n_estimators_grid, min_child_weight=min_child_weight_grid)

    grid_search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=[(X[mask_train].index,X[mask_test].index)], verbose=1)

    grid_result = grid_search.fit(X,np.log1p(y))

    # summarize results

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    print(grid_result.cv_results_)
#Best: -0.944889 using {'max_depth': 8, 'min_child_weight': 7, 'n_estimators': 800}



#Best: -0.926787 using {'max_depth': 5, 'min_child_weight': 4, 'n_estimators': 800}



if optimize_model:



    model = xgb.XGBRegressor(random_state=42,learning_rate=0.1)

    n_estimators_grid = [600]

    max_depth_grid = [6,7,8]

    min_child_weight_grid =[1,2,3]

    param_grid = dict(max_depth=max_depth_grid, n_estimators=n_estimators_grid, min_child_weight=min_child_weight_grid)

    grid_search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=[(X[mask_train].index,X[mask_test].index)], verbose=1)

    grid_result = grid_search.fit(X,np.log1p(y))

    # summarize results

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    print(grid_result.cv_results_)
X['d_Confirmed']=np.log1p(y)



#Best: -0.514411 using {'max_depth': 5, 'min_child_weight': 5, 'n_estimators': 400}



if optimize_model_2:



    model = xgb.XGBRegressor(random_state=42,learning_rate=0.1)

    n_estimators_grid = [400,600,800,1000]

    max_depth_grid = [5]

    min_child_weight_grid = [5] 

    param_grid = dict(max_depth=max_depth_grid, n_estimators=n_estimators_grid,min_child_weight=min_child_weight_grid)

    grid_search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=[(X[mask_train].index,X[mask_test].index)], verbose=1)

    grid_result = grid_search.fit(X,np.log1p(y_2))

    # summarize results

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    print(grid_result.cv_results_)
#Best: -0.507334 using {'max_depth': 4, 'min_child_weight': 5, 'n_estimators': 400}



if optimize_model_2:



    model = xgb.XGBRegressor(random_state=42,learning_rate=0.1)

    n_estimators_grid = [600]

    max_depth_grid = [4,5]

    min_child_weight_grid = [1,3,5,7] 

    param_grid = dict(max_depth=max_depth_grid, n_estimators=n_estimators_grid,min_child_weight=min_child_weight_grid)

    grid_search = GridSearchCV(model, param_grid, scoring="neg_root_mean_squared_error", n_jobs=-1, cv=[(X[mask_train].index,X[mask_test].index)], verbose=1)

    grid_result = grid_search.fit(X,np.log1p(y_2))

    # summarize results

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    print(grid_result.cv_results_)
reg = xgb.XGBRegressor(n_estimators=n_estimators,

    max_depth=max_depth,

    min_child_weight=min_child_weight,

    learning_rate=learning_rate,

    random_state=42)

reg_2 = xgb.XGBRegressor(n_estimators=n_estimators_2,

    max_depth=max_depth_2,

    min_child_weight=min_child_weight_2,

    learning_rate=learning_rate_2,random_state=42)
reg.fit(X_train,np.log1p(y_train))
plot = xgb.plot_importance(reg, max_num_features=10)
y_pred = reg.predict(X_test)
from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_pred,np.log1p(y_test)))
X_train_2=X_train.copy()

X_train_2['d_confirmed']=np.log1p(y_train)  

X_test_2=X_test.copy()

X_test_2['d_confirmed']=y_pred
reg_2.fit(X_train_2,np.log1p(y_train_2))
plot = xgb.plot_importance(reg_2, max_num_features=10)
y_pred_2 = reg_2.predict(X_test_2)
np.sqrt(mean_squared_error(y_pred_2,np.log1p(y_test_2)))
test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")

test.rename(columns={'ForecastId':'Id'},inplace=True)

test=add_location(test)



test.set_index('location',inplace=True)



test['day_of_year']=test['Date'].dt.dayofyear

test['day_of_week']=test['Date'].dt.dayofweek

test=add_days_passed(test,first_day,day_ten)

test=add_country_stat(test,country_stat,us_stat)

test=add_temperature(test,weather_country,weather_state)
days_to_predict=test['Date'].unique()

days_to_predict.sort()
test.head()
big_train=big_train.drop(columns=["index"])

known=big_train['Date'].unique()

print(known)
if Make_submission==True:

    results=[]

    full_results=[]



    for d in days_to_predict:

        print("Predicting {}".format(d))

        if d in known:

            print("Data Known")

        

            X=big_train.drop(columns=['Province_State','Country_Region','ConfirmedCases','Fatalities','location','Date',

                                  'day_of_year','day_of_week','d_ConfirmedCases','d_Fatalities','growth_Fatalities',

                                      'growth_ConfirmedCases'])



            y=big_train['d_ConfirmedCases']

            y_2=big_train['d_Fatalities']

        

            mask_train=big_train['Date']<d

            mask_val=big_train['Date']==d

        

            X_train=X[mask_train]

            y_train=y[mask_train]

            y_train_2=y_2[mask_train]

        

            X_val=X[mask_val]

            y_val=y[mask_val]

            y_val_2=y_2[mask_val]

        

            reg = xgb.XGBRegressor(n_estimators=n_estimators,

                                   max_depth=max_depth,

                                   min_child_weight=min_child_weight,

                                   learning_rate=learning_rate,random_state=42)

            reg_2 = xgb.XGBRegressor(n_estimators=n_estimators_2,

                                   max_depth=max_depth_2,

                                   min_child_weight=min_child_weight_2,

                                   learning_rate=learning_rate_2,random_state=42)

        

            reg.fit(X_train,np.log1p(y_train))

        

            y_pred = reg.predict(X_val)

            print("MSLE {}".format(mean_squared_error(y_pred,np.log1p(y_val))))

        

            X_train_2=X_train.copy()

            X_train_2['d_ConfirmedCases']=np.log1p(y_train)  #0.4412899060661785 <- without , with - 0.4463  

            X_val_2=X_val.copy()

            X_val_2['d_ConfirmedCases']=y_pred

        

            reg_2.fit(X_train_2,np.log1p(y_train_2))

        

            y_pred_2 = reg_2.predict(X_val_2)

        

            print("MSLE {}".format(mean_squared_error(y_pred_2,np.log1p(y_val_2))))

        

        #result=X_test[['']]

        elif d-np.timedelta64(86400000000000,'ns') in known:

            print("Data Known")

        

            X=big_train.drop(columns=['Province_State','Country_Region','ConfirmedCases','Fatalities','location','Date',

                                  'day_of_year','day_of_week','d_ConfirmedCases','d_Fatalities','growth_Fatalities',

                                      'growth_ConfirmedCases'])



            y=big_train['d_ConfirmedCases']

            y_2=big_train['d_Fatalities']

        

            mask_train=big_train['Date']<d

        

            X_train=X[mask_train]

            y_train=y[mask_train]

            y_train_2=y_2[mask_train]

        

        

            reg = xgb.XGBRegressor(n_estimators=n_estimators,

                                   max_depth=max_depth,

                                   min_child_weight=min_child_weight,

                                   learning_rate=learning_rate,random_state=42)

            reg_2 = xgb.XGBRegressor(n_estimators=n_estimators_2,

                                   max_depth=max_depth_2,

                                   min_child_weight=min_child_weight_2,

                                   learning_rate=learning_rate_2,random_state=42)

        

            reg.fit(X_train,np.log1p(y_train))

        

            X_train_2=X_train.copy()

            X_train_2['d_ConfirmedCases']=np.log1p(y_train)  #0.4412899060661785 <- without , with - 0.4463  

            

            reg_2.fit(X_train_2,np.log1p(y_train_2))

        

        

        

        X_test=test[test['Date']==d]

    

        day=X_test['day_of_year'].iloc[0]

    

        country_info=big_train[big_train['day_of_year']==day-1].groupby(['Country_Region'])['ConfirmedCases'].sum()

    

        border_cases=border_info.merge(country_info, how='inner', 

                left_on=['country_border_name'],right_on=['Country_Region'])

    

        border_cases=border_cases.groupby(['country_name'])['ConfirmedCases'].sum()

        border_cases=border_cases.rename('ConfirmedCases_neighbors')

    

        X_test=X_test.merge(border_cases, how='left', left_on=['Country_Region'],right_on=['country_name'])

        X_test['ConfirmedCases_neighbors']=X_test['ConfirmedCases_neighbors'].fillna(0)

    

        X_test = pd.concat([X_test,pd.get_dummies(X_test['location'], prefix='loc')],axis=1)

        X_test['ConfirmedCases_neighbors']=np.log1p(X_test['ConfirmedCases_neighbors'])

        

        #X_test=X_test.merge(big_train[big_train['day_of_year']==day-1][['location','ConfirmedCases','Fatalities']], how='left', 

        #         left_on=['location'],right_on=['location'])

        #X_test.rename(columns={'ConfirmedCases':'prev_ConfirmedCases','Fatalities':'prev_Fatalities'},inplace=True)

        

        #X_test['prev_ConfirmedCases']=np.log1p(X_test['prev_ConfirmedCases'])

        #X_test['prev_Fatalities']=np.log1p(X_test['prev_Fatalities'])

        

    

        X_test.set_index('Id',inplace=True)

    

    #print(X_test.head(5))

    

        y_test=reg.predict(X_test.drop(columns=['Province_State','Country_Region','location','Date','day_of_year','day_of_week']))

    

    #print(y_test)

    

        X_test['d_ConfirmedCases']=y_test

    

        y_test=reg_2.predict(X_test.drop(columns=['Province_State','Country_Region','location','Date',

                                            'day_of_year','day_of_week']))

    

        X_test['d_Fatalities']=y_test

    

    #print(X_test.shape)

    

        X_test['Id']=X_test.index

    

        X_test=X_test.merge(big_train[big_train['day_of_year']==day-1][['location','ConfirmedCases','Fatalities']], how='left', 

                 left_on=['location'],right_on=['location'])

    

    #print(X_test.head(5))

    

    #X_test.set_index('Id',inplace=True)

    

    #print(X_test.shape)

    

        X_test.set_index('Id',inplace=True)

    

        #print(X_test.head(5))

        

        X_test['d_ConfirmedCases']=np.expm1(X_test['d_ConfirmedCases'])

        X_test['d_Fatalities']=np.expm1(X_test['d_Fatalities'])

    

        X_test['ConfirmedCases']+=X_test['d_ConfirmedCases']

        X_test['Fatalities']+=X_test['d_Fatalities']

       

    

    

        results.append(X_test[['ConfirmedCases','Fatalities']])

        full_results.append(X_test)

    

        if not d in known: #Needed to correctly get data on neighbors         

            big_train=pd.concat([big_train,X_test])

    

    

    

    

    

    

    

    

        

        

        

        
if Make_submission==True:

    submission=pd.read_csv("../input/covid19-global-forecasting-week-3/submission.csv")

    submission.drop(columns=['ConfirmedCases','Fatalities'],inplace=True)

    submission=submission.merge(pd.concat(results),left_on=['ForecastId'],right_index=True).clip(lower=0)

    submission.to_csv('submission.csv',index=False)
full_res=pd.concat(full_results)




trained=big_train[(big_train['loc_ Ukraine']==1) & (big_train['Date']<'2020-04-02')][['Date','ConfirmedCases','Fatalities']]

prediction=full_res[full_res['loc_ Ukraine']==1][['Date','ConfirmedCases','Fatalities']]

fig1 = px.scatter(trained, x="Date", y="ConfirmedCases")



fig1.add_trace(go.Scatter(

        x=prediction["Date"],

        y=prediction["ConfirmedCases"],

        mode="lines",

        line=go.scatter.Line(color="red"),

        showlegend=False))



fig1.show()
trained=big_train[(big_train['loc_ Switzerland']==1) & (big_train['Date']<'2020-04-02')][['Date','ConfirmedCases','Fatalities']]

prediction=full_res[full_res['loc_ Switzerland']==1][['Date','ConfirmedCases','Fatalities']]

fig1 = px.scatter(trained, x="Date", y="ConfirmedCases")



fig1.add_trace(go.Scatter(

        x=prediction["Date"],

        y=prediction["ConfirmedCases"],

        mode="lines",

        line=go.scatter.Line(color="red"),

        showlegend=False))



fig1.show()
trained=big_train[(big_train['loc_ Italy']==1) & (big_train['Date']<'2020-04-02')][['Date','ConfirmedCases','Fatalities']]

prediction=full_res[full_res['loc_ Italy']==1][['Date','ConfirmedCases','Fatalities']]

fig1 = px.scatter(trained, x="Date", y="ConfirmedCases")



fig1.add_trace(go.Scatter(

        x=prediction["Date"],

        y=prediction["ConfirmedCases"],

        mode="lines",

        line=go.scatter.Line(color="red"),

        showlegend=False))



fig1.show()
trained=big_train[(big_train['loc_ France']==1) & (big_train['Date']<'2020-04-02')][['Date','ConfirmedCases','Fatalities']]

prediction=full_res[full_res['loc_ France']==1][['Date','ConfirmedCases','Fatalities']]

fig1 = px.scatter(trained, x="Date", y="ConfirmedCases")



fig1.add_trace(go.Scatter(

        x=prediction["Date"],

        y=prediction["ConfirmedCases"],

        mode="lines",

        line=go.scatter.Line(color="red"),

        showlegend=False))



fig1.show()