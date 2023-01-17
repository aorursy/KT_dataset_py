

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train =pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

df_test =pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

df_ss =pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')



# df_geo = pd.read_csv('/kaggle/input/corona-virus-report/covid_19_clean_complete.csv')

print("Any missing sample in training set:",df_train.isnull().values.any())

print("Any missing sample in test set:",df_test.isnull().values.any(), "\n")
print("Check the Country list in train and testing data: ",(df_test.Country_Region.unique() == df_train.Country_Region.unique()).all())

# print("Check the Country list in train and clean_  data: ",(df_geo['Country/Region'].unique() == df_train.Country_Region.unique()).all())

# countries_geo = df_geo['Country/Region'].unique()

# states_geo = df_geo['Province/State'].unique()

# # df_train.Country_Region.unique() ==df_geo['Country/Region'].unique()

# # print("Check the Province list in train and clean_  data: ",(df_geo['Province/State'].unique() == df_train.Province_State.unique()).all())

# for i, country in enumerate(df_train.Country_Region.unique()):

#     if (country not in countries_geo):

#         print(i, country)

# for i, country in enumerate(countries_geo):

#     if 'Korea' in country:

#         print(country)

        

# for i, state in enumerate(df_train.Province_State.unique()):

#     if (state not in states_geo):

#         print(i, state)

import numpy as np # linear algebra

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

# pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier, GradientBoostingRegressor

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor  

from sklearn.preprocessing import LabelEncoder

renames = {'Province_State':'state','Country_Region':'country','ConfirmedCases':'cc','Fatalities':'deaths','Date':'date','ForecastId':'id','Id':'id'}

df_train.rename(columns=renames,inplace=True)

df_test.rename(columns=renames,inplace=True)

# df_train.head()
# Fill the empty states in countries

df_train.state.fillna(df_train.country,inplace=True)

df_test.state.fillna(df_test.country,inplace=True)



def date_convert(df):

    date = pd.DatetimeIndex(df['date'])

    df['year'] = date.year

    df['month'] = date.month

    df['dayofmonth'] = date.day

    df['weekofyear'] = date.weekofyear

    df['dayofweek'] = date.dayofweek

    df['dayofyear'] = date.dayofyear

date_convert(df_train)

date_convert(df_test)

df_train.columns
countries_train = df_train[df_train['country']!=df_train['state']]['country'].unique().tolist()

print(countries_train)

states_train = df_train['state'].unique().tolist()

for country in countries_train:

    if country in states_train:

        print(country)
countries_test = df_test[df_test['country']!=df_test['state']]['country'].unique().tolist()

print(countries_test)

states_test = df_test['state'].unique().tolist()

for country in countries_test:

    if country in states_test:

        print(country)
t = (df_train['country']==country) &(df_train['state']!=country)

cntr = df_train[df_train['state']==country][['cc']].reset_index().copy()

cntr['cum'] = 0.0

cntr[['cum','cc']]

for i,state in enumerate(df_train[t]['state'].unique().tolist()):

    print(i,df_train[df_train['state'] == state]['cc'].max())

    cntr['cum_{}'.format(i)] = df_train[df_train['state'] == state]['cc'].values

# a = df_train[t].groupby(by=['state','dayofyear'])['cc'].sum(axis=1).plot()

# df_train[(df_train['country']==country) &(df_train['state']==country)].plot(x='dayofyear',y='cc')

cntr[['cum_{}'.format(i) for i,state in enumerate(df_train[t]['state'].unique().tolist())]].sum(axis=1)

# cntr['cc','cum']

cntr.tail()
# Label the countries and states

label_countries = LabelEncoder()

label_states = LabelEncoder()



for i in ['state', 'country']:

    df_train['label_{}'.format(i)] = label_countries.fit_transform(df_train[i])

    df_test['label_{}'.format(i)] = label_countries.transform(df_test[i])

# set the first day of first case



day_first_case_state = df_train[df_train.cc>0].groupby('state').first()['dayofyear']

day_first_case_country = df_train[df_train.cc>0].groupby('country').first()['dayofyear']

# df_train['day1_state'] = np.zeros(df_train.state.size)

# df_train['day1_country'] = np.zeros(df_train.country.size)



# for day in day_first_case_state.keys():

#     cnt = df_train[df_train['state']==day].loc[:,'day1_state'].size

#     df_train[df_train['state']==day].loc[:,'day1_state'] = np.ones(cnt) * day_first_case_state[day]

# for day in day_first_case_country.keys():

#     cnt = df_train[df_train['state']==day].loc[:,'day1_country'].size

#     df_train[df_train['state']==day].loc[:,'day1_country'] = np.ones(cnt) * day_first_case_country[day] 



df_train['day1_state'] = day_first_case_state[df_train['state']].values

df_train['day1_country'] = day_first_case_country[df_train['country']].values



df_test['day1_state'] = day_first_case_state[df_test['state']].values

df_test['day1_country'] = day_first_case_country[df_test['country']].values



print("set the log1p for cc and deaths")

df_train['log_cc'] = np.log1p(df_train.cc)

df_train['log_deaths'] = np.log1p(df_train.deaths)



df_train['cc_day'] = 0.

df_train['deaths_day'] = 0.





for i, c_state in enumerate(df_train.state.unique()[:]):

    t = (df_train.state == c_state)

#     df_train[t].apply(,axis=1)

    isize = df_train[t].cc.size

#     print(t.size, isize)

#     df_train[t].loc[:,'cc_day'] = df_train[t].cc.diff().fillna(0)

    cnt = 0

    for i, ind in enumerate(df_train[t].index):

        if(df_train.loc[ind,'cc'] < cnt):

            df_train.loc[ind,'cc'] = df_train.loc[ind-1,'cc']

        cnt = df_train.loc[ind,'cc']

    cnt = 0

    for i, ind in enumerate(df_train[t].index):

        if(df_train.loc[ind,'deaths'] < cnt):

            df_train.loc[ind,'deaths'] = df_train.loc[ind-1,'deaths']

        cnt = df_train.loc[ind,'deaths']

        

    df_train.loc[t,'cc_day'] = df_train[t].cc.diff().fillna(0).values

    df_train.loc[t,'deaths_day'] = df_train[t].deaths.diff().fillna(0).values

#     print(c_state,df_train.loc[t,['cc','deaths_day']].min().values)

#Set lag of every day diff

df_train.loc[df_train['cc_day']<0 , ['cc_day']] = 0

df_train.loc[(df_train['deaths_day']<0) , ['deaths_day']] = 0

print("Set lag of every day diff ", df_train['cc_day'].min(),df_train['deaths_day'].min())

df_train['log_cc_day'] = np.log1p(df_train['cc_day'])

df_train['log_deaths_day'] = np.log1p(df_train['deaths_day'])

print("Check Inf :",df_train['log_cc_day'].notnull().values.all(),df_train['log_deaths_day'].notnull().values.all(),df_train['log_cc_day'].isna().any())

df_train['log_cc_day'].replace([np.inf, -np.inf], 0,inplace=True)

df_train['log_deaths_day'].replace([np.inf, -np.inf], 0,inplace=True)

print("Check Inf after replace :",df_train['log_cc_day'].notnull().values.any(),df_train['log_deaths_day'].notnull().values.any(),df_train['log_cc_day'].isna().any())



# print(df_train.is





df_train['log_cc_diff'] = 0.

df_train['log_deaths_diff'] = 0.

# get log_diff

print("get log_diff")

for i, c_state in enumerate(df_train.state.unique()[:]):

    t = (df_train.state == c_state)

    df_train.loc[t,'log_cc_diff'] = (df_train[t]['log_cc'] - df_train[t]['log_cc'].shift()) #.fillna(0)

    df_train.loc[t,'log_deaths_diff'] = (df_train[t]['log_deaths'] - df_train[t]['log_deaths'].shift()) #.fillna(0)



df_train.loc[df_train['log_cc_diff']<0 , ['log_cc_diff']] = 0

df_train.loc[(df_train['log_deaths_diff']<0) , ['log_deaths_diff']] = 0

print("Check the negative vals ",df_train['log_cc_diff'].min(),df_train['log_deaths_diff'].min())

df_train['log_cc_diff'] = np.log1p(df_train['log_cc_diff'])

df_train['log_deaths_diff'] = np.log1p(df_train['log_deaths_diff'])



df_train['log_cc_diff'].fillna(0,inplace=True)

df_train['log_deaths_diff'].fillna(0,inplace=True)

df_train['log_cc_diff'].replace([np.inf, -np.inf], 0,inplace=True)

df_train['log_deaths_diff'].replace([np.inf, -np.inf], 0,inplace=True)

df_train[df_train['state']=='France'].plot(x='dayofyear',y='cc')
t = df_train.country=='Sweden'

df_train[t][['log_cc_diff','log_deaths_diff']].fillna(0)

df_train[['log_cc_diff','log_deaths_diff']].isna().any()
# df_train[t].plot(x='dayofyear',y=['log_cc_diff','log_deaths_diff'])

plt.plot(df_train[t]['dayofyear'],df_train[t]['log_cc'].ewm(5).mean())

plt.plot(df_train[t]['dayofyear'],df_train[t]['log_cc'].rolling(5).mean())

plt.plot(df_train[t]['dayofyear'],df_train[t]['log_cc'])

# pd.(df_train[t]['cc'],5)

# df_train[t]['cc'].ewm(5).mean()
df_corr = df_train.drop(['year','id','country','state','month','weekofyear'

                         ,'date','deaths','cc','log_cc','cc_day','log_deaths'

                         ,'log_cc_day', 'deaths_day', 'log_deaths_day'

                        ],axis=1) .corr()

import seaborn as sns

sns.heatmap(df_corr, annot = False)
df_train.country.unique()
cols_drop = ['year','id','country','state','month','weekofyear','date','day1_state','day1_country','dayofmonth']

cols_pred = ['cc','deaths']

# cols_drop += cols_pred

cols_pred_day = ['cc_day','deaths_day']

cols_pred_log = ['log_cc','log_deaths']

cols_pred_log_day = ['log_cc_day','log_deaths_day']

cols_pred_log_diff = ['log_cc_diff','log_deaths_diff']



col_drop = cols_drop+cols_pred + cols_pred_log + cols_pred_day + cols_pred_log_day + cols_pred_log_diff



X = df_train[df_train.cc>0].drop(col_drop,axis=1)

y = df_train[df_train.cc>0][cols_pred]

y_log = df_train[df_train.cc>0][cols_pred_log]

y_day = df_train[df_train.cc>0][cols_pred_day]

y_dif = df_train[df_train.cc>0][cols_pred_log_diff + cols_pred_log]



X_test = df_test.drop(cols_drop,axis=1)

# y_test = df_test[['cc','deaths']]



day_end_train = df_train.dayofyear.max()

day_start_test = df_test.dayofyear.min()

print('Testing starting from day: {} , {}'.format(day_start_test, day_end_train))



X_train = X[X.dayofyear< day_start_test]

y_train = y[X.dayofyear< day_start_test]

y_train_log = y_log[X.dayofyear< day_start_test]

y_train_day = y_day[X.dayofyear< day_start_test]

y_train_dif = y_dif[X.dayofyear< day_start_test]



X_valid = X[X.dayofyear>= day_start_test]

y_valid = y[X.dayofyear>= day_start_test]

# check the same columns in train and test data

(X_train.columns == X_test.columns).all()

# print(X_train.columns.size,X_test.columns.size)

# print(X_train.columns,X_test.columns)

# ‘neg_mean_squared_log_error’

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_log_error

from sklearn.svm import SVC, SVR

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA



model_dtr = DecisionTreeRegressor(random_state = 0) 

model_rfr = RandomForestRegressor(n_estimators=110, random_state=0)



# Multiply by -1 since sklearn calculates *negative* MAE

def get_score(model,X_tr, y_tr,n_est=50,n_cv=5):

    pipeline = Pipeline(steps=[

        ('preprocessor', SimpleImputer()),

        ('model', model)

])



    scores = -1 * cross_val_score(pipeline, X_tr, y_tr,

                              cv=n_cv,

#                               scoring='neg_mean_squared_log_error')

                              scoring='neg_mean_absolute_error')

    return scores.mean()





print("Average NMSLE score RandForestRegre :", get_score(model_rfr,X_train, y_train))

print("Average NMSLE score DecisTreeRegres:", get_score(model_dtr,X_train, y_train))

print("Average NMSLE score DecisTreeClass:", get_score(DecisionTreeClassifier(random_state=0),X_train, y_train))

# print("Average NMSLE score GBC:", get_score(GradientBoostingClassifier(random_state=0),X_train, y_train))

# print("Average NMSLE score GBR:", get_score(GradientBoostingRegressor(random_state=0),X_train, y_train))

# # print("Average NMSLE score:", get_score(model_dtr,X_train, y_train))

# print("Average NMSLE score:", get_score(model_rfr,X_train, y_train))



# results = {i: get_score(model_rfr,X_train, y_train,n_est=i,n_cv=5) for i in range(50,450,50)} # Your code here

# for i in [50,100,500,5000]:

#     print(i," Average NMSLE score:", get_score(model_rfr,X_train, y_train_dif,n_est=i,n_cv=5))

    
# for i in range(50,500,50):

#     print(i," Num of Est MSE score:", get_score(model_rfr,X_train, y_train,n_est=i,n_cv=5))

    

# 50  Num of Est MSE score: 0.31316054949126604

# 100  Num of Est MSE score: 0.31269114946680254

# 150  Num of Est MSE score: 0.31290359325866357

# 200  Num of Est MSE score: 0.3127087613582483

# 250  Num of Est MSE score: 0.3131997694520491

# 300  Num of Est MSE score: 0.31319708740038366

# 350  Num of Est MSE score: 0.3127921127791045

# 400  Num of Est MSE score: 0.31244946171626103

# 450  Num of Est MSE score: 0.3128081193774369
# for i in [2,3,4,5]:

#     print(i," Average NMSLE score:", get_score(model_rfr,X_train, y_train,n_est=350,n_cv=i))

# for i in [6,8,10]:

#     print(i," Average NMSLE score:", get_score(model_rfr,X_train, y_train,n_est=350,n_cv=i))

    

# set the number of CV

# 2  Average NMSLE score: 0.013467870703916035

# 3  Average NMSLE score: 0.008843653417933703

# 4  Average NMSLE score: 0.006452329486737336

# 5  Average NMSLE score: 0.006024922098354479

# 6  Average NMSLE score: 0.007329168527967208

# 8  Average NMSLE score: 0.006077414852762886

# 10  Average NMSLE score: 0.005814285650045135
# X_train.shape, y_train.shape

# sss  = SVR()

# sss.fit(X_train,y_train_log['log_cc'])
# y_predict_log = sss.predict(X_valid)

# y_pred = np.exp(y_predict_log) - 1

# # y_pred = (y_pred).astype(int)

# # score =  mean_squared_log_error(y_predict, y_valid)

# score_log =  mean_squared_log_error(y_pred, y_valid['cc'])

# print("MAE result: ", score_log)
# get_score(model_rfr,X_train, y_train,n_est=350,n_cv=10)

steps = [('scaler', StandardScaler()), 

         ('SVM', SVC()),        

#          ('model', RandomForestRegressor(n_estimators=400, random_state=0)

          ]

pipeline = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

    ('model', DecisionTreeRegressor(random_state = 0))

#     ('model', RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0))

])



pipeline0 = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

#     ('model', DecisionTreeRegressor(random_state = 0)),

    ('gbm', GradientBoostingRegressor(n_estimators=300, learning_rate = 0.01,alpha=0.9,random_state = 0))

])

pipeline1 = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

#     ('model', DecisionTreeRegressor(random_state = 0)),

    ('gbm', GradientBoostingRegressor(n_estimators=300, learning_rate = 0.01,alpha=0.9,random_state = 0))

])



pipeline_log = Pipeline(steps=[

    ('scaler', StandardScaler()),

    ('preprocessor', SimpleImputer()),

    ('model', RandomForestRegressor(n_estimators=100, max_depth=2, random_state=0))

#     ('gdm', GradientBoostingRegressor(n_estimators=300, learning_rate = 0.01,alpha=0.9,random_state = 0))

])

pipeline_svm = Pipeline(steps=steps)





# scores = -1 * cross_val_score(pipeline, X_train, y_train,

#                           cv=10,

# #                               scoring='neg_mean_squared_log_error')

#                           scoring='neg_mean_absolute_error')

parameters_svm = {'SVM__C':[0.001,0.1,10], 'SVM__gamma':[0.1,0.01]}

# parameters_log = {'model__n_estimators': [100,300,450], 'model__max_depth': [None,3]} #, 'model__min_samples_split': [1, 2, 3]}

parameters_log = {'model__n_estimators': [450], 'model__max_depth': [None]} #, 'model__min_samples_split': [1, 2, 3]}

parameters = {'model__max_depth': [None, 1, 2, 3],'model__max_features':[None,'log2','auto']} #, 'model__min_samples_split': [1, 2, 3]}

# parameters = {'model__n_estimators': [100,300,500, 700, 1000]}

grid = GridSearchCV(pipeline_log, 

                    param_grid=parameters_log, 

                    scoring='r2', 

#                     scoring='neg_mean_squared_log_error',

                    verbose=True,

                    cv=5)

# grid = GridSearchCV(pipeline_svm, 

#                     param_grid=parameters_svm, 

# #                     scoring='r2', 

#                     scoring='neg_mean_squared_log_error',

#                     verbose=True,

#                     cv=5)

grid0 = GridSearchCV(pipeline0, 

                    param_grid=parameters, 

#                     scoring='r2', 

                    scoring='neg_mean_squared_log_error',

                    cv=5)

grid1 = GridSearchCV(pipeline1, 

                    param_grid=parameters, 

#                     scoring='r2', 

                    scoring='neg_mean_squared_log_error',

                    cv=5)

grid2 = GridSearchCV(pipeline_log, 

                    param_grid=parameters_log, 

                    scoring='r2', 

#                     scoring='neg_mean_squared_log_error',

                    verbose=True,

                    cv=5)



pipeline.fit(X_train,y_train)



# grid0.fit(X_train,y_train['cc'])

# grid1.fit(X_train,y_train['deaths'])



grid.fit(X_train,y_train_log)



grid2.fit(X_train,y_train_dif)



# grid.fit(X_train,y_train['cc'])



# pipeline.fit(X_train,y_train_day['cc_day'])



y_predict = pipeline.predict(X_valid)



# y_predict_cc = grid0.predict(X_valid)

# y_predict_cc = np.expm1(y_predict_cc)

# y_predict_deaths = grid1.predict(X_valid)

# y_predict_deaths = np.expm1(y_predict_deaths)



# y_predict_log = pipeline_log.predict(X_valid)

y_predict_log = grid.predict(X_valid)



y_predict_dif = grid2.predict(X_valid)



y_pred = np.exp(y_predict_log) - 1

# y_pred_dif = np.exp(y_predict_dif) - 1

# y_pred = (y_predict_log)



# y_pred = np.round(y_pred).astype(int)

# score_cc =  mean_squared_log_error(y_predict_cc, y_valid['cc'])

# score_deaths =  mean_squared_log_error(y_predict_deaths, y_valid['deaths'])

score =  mean_squared_log_error(y_predict, y_valid)

score_log =  mean_squared_log_error(y_pred, y_valid)

print("MAE result: ",score, score_log ) #, score_cc, score_deaths)
y_predict_dif
X_train.dayofyear.max(),X_valid.dayofyear.min(),y_train.loc[X_train.dayofyear == X_train.dayofyear.max(),'cc']
y_predicted_diff = y_valid.copy()

# y_predicted_diff

y_pred_diff_res = y_valid.copy()

y_pred_diff_res['log_cc_pred_diff'] = y_predict_dif[:,0]

y_pred_diff_res['log_dd_pred_diff'] = y_predict_dif[:,1]

y_pred_diff_res['log_cc_pred'] = y_predict_dif[:,2]

y_pred_diff_res['log_dd_pred'] = y_predict_dif[:,3]

y_pred_diff_res['log_cc_cumsum'] = 0.0

y_pred_diff_res['log_dd_cumsum'] = 0.0



for i, c_state in enumerate(X_valid.label_state.unique()[:]):

    t = (X_valid.label_state == c_state)

    y_pred_diff_res.loc[t,'log_cc_cumsum'] = y_pred_diff_res[t]['log_cc_pred_diff'].cumsum() #.fillna(0)

    y_pred_diff_res.loc[t,'log_dd_cumsum'] = y_pred_diff_res[t]['log_dd_pred_diff'].cumsum() #.fillna(0)

    

#     tt = (X_train.label_state == c_state) & (X_train.dayofyear==X_train.dayofyear.max())

#     cc = np.log1p(y_train.loc[tt,'cc'])

#     dd = np.log1p(y_train.loc[tt,'deaths'])

#     y_pred_diff_res.loc[t,'log_cc_cumsum'] +=cc

#     y_pred_diff_res.loc[t,'log_dd_cumsum'] +=dd

#     print(c_state,cc.values, dd.values)

    

y_pred_diff_res['log_cc_cumsum'] += (y_pred_diff_res['log_cc_pred'])

y_pred_diff_res['log_dd_cumsum'] += (y_pred_diff_res['log_dd_pred'])

y_pred_diff_res['cc_cumsum'] = np.expm1(y_pred_diff_res['log_cc_cumsum'])

y_pred_diff_res['dd_cumsum'] = np.expm1(y_pred_diff_res['log_dd_cumsum']) 



y_pred_diff_res.head()

score =  mean_squared_log_error(y_pred_diff_res[['cc_cumsum','dd_cumsum']], y_valid)

print("MAE result: ",score)
y_pred_diff_res.head()
t_tr = X_train.label_state == 261

t_v = X_valid.label_state == 261

f, ax = plt.subplots()

# y_train[t_tr].plot(ax=ax)

# y_valid[t_v].plot(y='deaths',ax=ax)

y_valid[t_v].plot(y=['cc','deaths'],ax=ax)

# y_pred_diff_res[t_v].plot(y=['dd_cumsum'],ax=ax)

y_pred_diff_res[t_v].plot(y=['cc_cumsum','dd_cumsum'],ax=ax)

# df_chk = pd.DataFrame({'cc_log':y_pred[:,0],'cc_pl':y_predict[:,0],'cc_cc':y_predict_cc,'cc_valid':y_valid['cc'],'deaths_log':y_pred[:,1]})

# df_chk

# # y_valid[X_valid['state']=='Jamaica']

# # X_valid

# for i, state in enumerate(X_valid.label_state.unique()[:]):

#     t = X_valid.label_state == state

#     vals = df_chk.loc[t,'cc_log'].values

#     df_chk.loc[t,'cc_log'] = [vals[:j+1].max() for j in range(vals.size)]

#     vals = df_chk.loc[t,'deaths_log'].values

#     df_chk.loc[t,'deaths_log'] = [vals[:j+1].max() for j in range(vals.size)]

# #     print([vals[:j+1].max() for j in range(vals.size)])

# score_log =  mean_squared_log_error(df_chk[['cc_log','deaths_log']], y_valid)

# score_log1 =  mean_squared_log_error(np.round(df_chk[['cc_log','deaths_log']]).astype('int'), y_valid)

# print("MAE result: ",score, score_log, score_log1, score_deaths)

# df_chk
grid2.best_params_
# grid.fit(X,y_log)

grid2.fit(X,y_dif)

# my_pipeline.fit(X,y)

y_test_predict = pipeline.predict(X_test) 

# y_test_predict_log = pipeline_log.predict(X_test) 

y_test_predict_log = grid.predict(X_test) 

y_test_predict_log = np.expm1(y_test_predict_log)

# y_test_predict_log = np.round(y_test_predict_log).astype('int')
y_test_predict_dif = grid2.predict(X_test) 



y_pred_diff_res = X_test[['label_state']].copy()

y_pred_diff_res['log_cc_pred_diff'] = y_test_predict_dif[:,0]

y_pred_diff_res['log_dd_pred_diff'] = y_test_predict_dif[:,1]

y_pred_diff_res['log_cc_pred'] = y_test_predict_dif[:,2]

y_pred_diff_res['log_dd_pred'] = y_test_predict_dif[:,3]

y_pred_diff_res['log_cc_cumsum'] = 0.0

y_pred_diff_res['log_dd_cumsum'] = 0.0

for i, c_state in enumerate(X_test.label_state.unique()[:]):

    t = (X_test.label_state == c_state)

    y_pred_diff_res.loc[t,'log_cc_cumsum'] = y_pred_diff_res[t]['log_cc_pred_diff'].cumsum() #.fillna(0)

    y_pred_diff_res.loc[t,'log_dd_cumsum'] = y_pred_diff_res[t]['log_dd_pred_diff'].cumsum() #.fillna(0)

y_pred_diff_res['log_cc_cumsum'] += (y_pred_diff_res['log_cc_pred'])

y_pred_diff_res['log_dd_cumsum'] += (y_pred_diff_res['log_dd_pred'])

y_pred_diff_res['cc_cumsum'] = np.expm1(y_pred_diff_res['log_cc_cumsum'])

y_pred_diff_res['dd_cumsum'] = np.expm1(y_pred_diff_res['log_dd_cumsum']) 

df_chk = pd.DataFrame({'st':y_test_predict[:,0],'cc_log':y_test_predict_log[:,0],'deaths_log':y_test_predict_log[:,1]})

for i, state in enumerate(X_valid.label_state.unique()[:]):

    t = X_test.label_state == state

    vals = df_chk.loc[t,'cc_log'].values

    df_chk.loc[t,'cc_log'] = [vals[:j+1].max() for j in range(vals.size)]

    vals = df_chk.loc[t,'deaths_log'].values

    df_chk.loc[t,'deaths_log'] = [vals[:j+1].max() for j in range(vals.size)]

df_chk['cc_log'] = df_chk['cc_log'].apply(lambda x: np.round(x).astype('int'))

df_chk['deaths_log'] = df_chk['deaths_log'].apply(lambda x: np.round(x).astype('int'))

# y_valid.values[:,0].size

df_chk
# df_ss[['ConfirmedCases','Fatalities']] = y_test_predict_log



df_ss[['ConfirmedCases','Fatalities']] = df_chk[['cc_log','deaths_log']]



# df_ss[['ConfirmedCases','Fatalities']] = y_pred_diff_res[['cc_cumsum','dd_cumsum']]

df_ss.to_csv('submission.csv',index=False)

# # preds = my_pipeline.predict(X_test)

# test_preds0 = my_pipeline0.predict(X_test)

# test_preds1 = my_pipeline1.predict(X_test)



# t0 = np.round(test_preds0).astype(int)

# t1 = np.round(test_preds1).astype(int)
df_ss
#check for inconsistencies in daily new cases, cumulative count should only increase or remain equal

df_train[df_train['cc_day'] < 0].sort_values('cc_day')
