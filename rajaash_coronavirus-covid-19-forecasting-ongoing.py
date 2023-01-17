import pandas as pd

import matplotlib as mat

import matplotlib.pyplot as plt

from PIL import Image

from wordcloud import WordCloud

import xgboost as xgb

from xgboost import plot_importance

from sklearn.model_selection import cross_val_score, GridSearchCV,cross_val_predict

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error,  r2_score

import numpy as np

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"
mat.rcParams.update({'figure.figsize':(20,15),'font.size':14})
covid19_train = pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

covid19_test = pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')
covid19_train.rename(columns={'Id':'ForecastId'},inplace=True)
covid19_train['Date-All'] = covid19_train['Date'].str.replace('-','').astype(int)

covid19_test['Date-All'] = covid19_test['Date'].str.replace('-','').astype(int)
covid19_train['Date'] = pd.to_datetime(covid19_train['Date'])

covid19_test['Date'] = pd.to_datetime(covid19_test['Date'])
covid19_gdf = covid19_train.groupby(['Date','Country_Region'])['ConfirmedCases'].sum().reset_index()

covid19_gdf['date'] = pd.to_datetime(covid19_gdf['Date'])

covid19_gdf['date'] = covid19_gdf['date'].dt.strftime('%m/%d/%Y')
fig = px.scatter_geo(covid19_gdf.fillna(0), locations="Country_Region", locationmode='country names', 

                     color="ConfirmedCases", size="ConfirmedCases", hover_name="Country_Region", 

                     projection="natural earth", animation_frame="date", 

                     title='Coronavirus Spread', color_continuous_scale="OrRd")

fig.update(layout_coloraxis_showscale=False)

fig.show()
totalCountryCases = covid19_train.drop_duplicates(['Province_State', 'Country_Region'],keep='last').groupby(['Country_Region'])[['ConfirmedCases']].sum().sort_values('ConfirmedCases',ascending=False)

totalCountryCases.head(20).plot(kind='bar',color='r')

plt.grid()

plt.show()
mask=np.array(Image.open("../input/coronavirusimage/coronavirus.png"))

wc = WordCloud(background_color="black",colormap=plt.cm.OrRd,collocations = False,mask=mask).generate_from_frequencies(totalCountryCases.to_dict()['ConfirmedCases'])

plt.imshow(wc,interpolation="bilinear")

plt.axis("off")

wc.to_file('Coronavirus-spread.png')

plt.show()

totalCountryFatalities = covid19_train.drop_duplicates(['Province_State', 'Country_Region'],keep='last').groupby(['Country_Region'])[['Fatalities']].sum().sort_values('Fatalities',ascending=False)



totalCountryFatalities.head(20).plot(kind='bar',color='r')

plt.grid()

plt.show()
str(round((totalCountryFatalities['Fatalities'].sum()/totalCountryCases['ConfirmedCases'].sum())*100,1)) + '% Deaths out of total of ' + str(totalCountryCases['ConfirmedCases'].sum()/1e3) + 'K infection'
totalDailyCases = covid19_train.groupby(['Date'])[['ConfirmedCases']].sum().sort_values('ConfirmedCases',ascending=False)
totalDailyCases.plot(grid=True,linestyle='', marker='.',color='r',markersize=20)

plt.show()
totalDailyCases['prop'] = round((totalDailyCases['ConfirmedCases'].cumsum()/totalDailyCases['ConfirmedCases'].sum())*100,2)
'80% of the infections were in last '+ str(len(totalDailyCases.loc[totalDailyCases['prop'] <= 80])) + ' days'
'A proportion of ' + str(round(len(totalDailyCases.loc[totalDailyCases['prop'] <= 80])/(len(totalDailyCases)+len(pd.date_range('17-11-2019','21-01-2020')))*100)) + '% from last days had the highest spread'
dependant_vars = ['ConfirmedCases', 'Fatalities']

y_train_CC = covid19_train.loc[:,[dependant_vars[0]]].values

y_train_fa = covid19_train.loc[:,[dependant_vars[1]]].values
X = pd.concat([covid19_train.drop(dependant_vars,1), covid19_test],sort=False).set_index('ForecastId')
X = pd.get_dummies(X,drop_first=True)
X_train = X.iloc[:len(covid19_train),:].drop('Date',1).copy()

X_test = X.iloc[len(covid19_train):,:].drop('Date',1).copy()
gsc = GridSearchCV(

        estimator=RandomForestRegressor(),

        param_grid={

            'max_depth': range(3,7),

            'n_estimators': (10, 50, 100, 1000),

        },

        cv=5, 

    scoring='neg_mean_squared_error', verbose=0, n_jobs=3)

    

grid_result = gsc.fit(X_train, y_train_CC)

best_params = grid_result.best_params_
best_params
kf = KFold(n_splits = 10, shuffle=True)
reg = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],                               

                            random_state=False, verbose=False)
rmse = []
for i in range(10):

    offset = next(kf.split(X_train),None)

    x_train = X_train.iloc[offset[0]]

    x_test = X_train.iloc[offset[1]]

    Y_train = y_train_CC[offset[0]]

    Y_test = y_train_CC[offset[1]]



    reg.fit(x_train, Y_train.reshape(-1))

    ypred = reg.predict(x_test)

    rmse.append(np.sqrt(mean_squared_error(Y_test, ypred)))
X_train
rmse
feat_importances = pd.Series(reg.feature_importances_, index=X_train.columns)

feat_importances.nlargest(4).plot(kind='barh')
X['yPred'] = reg.predict(X.drop(['Date'],1))

X['yPred'] = X['yPred'].apply(round)

X.loc[X['yPred'] < 0, 'yPred'] = 0

X.loc[:covid19_train.ForecastId.max(),'ConfirmedCases'] = y_train_CC
np.sqrt(mean_squared_error(y_true=X.loc[~X['ConfirmedCases'].isna()]['ConfirmedCases'],

                   y_pred=X.loc[~X['ConfirmedCases'].isna()]['yPred']))
r2_score(X.loc[~X['ConfirmedCases'].isna()]['ConfirmedCases'],

                               X.loc[~X['ConfirmedCases'].isna()]['yPred'])
X.set_index('Date').resample('D')[['ConfirmedCases']].sum().replace(0,np.nan).join(X.set_index('Date').resample('D')[['yPred']].sum()).plot(grid=True)
X.drop(['yPred','ConfirmedCases'],1,inplace=True)
kf = KFold(n_splits = 10, shuffle=True)
reg = xgb.XGBRegressor(colsample_bytree=0.85, max_depth=10, n_estimators = 500, learning_rate = 0.01, gamma = 4,seed=42)
rmse = []
for i in range(10):

    offset = next(kf.split(X_train),None)

    x_train = X_train.iloc[offset[0]]

    x_test = X_train.iloc[offset[1]]

    Y_train = y_train_CC[offset[0]]

    Y_test = y_train_CC[offset[1]]



    reg.fit(x_train, Y_train,eval_set=[(x_train, Y_train), 

                  (x_test, Y_test)],

        eval_metric="rmse",verbose=False)

    eval_results = reg.evals_result()

    rmse.append(np.mean(eval_results['validation_0']['rmse']))

    
rmse
fig, ax = plt.subplots(1,1,figsize=(20,50))

_ = plot_importance(reg,ax=ax)

plt.show()
X['yPred'] = reg.predict(X.drop(['Date'],1))

X['yPred'] = X['yPred'].apply(round)

X.loc[X['yPred'] < 0, 'yPred'] = 0

X.loc[:covid19_train.ForecastId.max(),'ConfirmedCases'] = y_train_CC
np.sqrt(mean_squared_error(y_true=X.loc[~X['ConfirmedCases'].isna()]['ConfirmedCases'],

                   y_pred=X.loc[~X['ConfirmedCases'].isna()]['yPred']))
r2_score(X.loc[~X['ConfirmedCases'].isna()]['ConfirmedCases'],

                               X.loc[~X['ConfirmedCases'].isna()]['yPred'])
X.set_index('Date').resample('D')[['ConfirmedCases']].sum().replace(0,np.nan).join(X.set_index('Date').resample('D')[['yPred']].sum()).plot(grid=True)
X.drop(['yPred','ConfirmedCases'],1,inplace=True)