import time

start = time.time()

import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from textblob import TextBlob



#plotting

import plotly.express as px

import plotly.graph_objects as go

import seaborn as sns

import matplotlib.pyplot as plt



#statistics & econometrics

import statsmodels.tsa.api as smt

import statsmodels.api as sm



#model fiiting and selection

from sklearn.metrics import mean_squared_error

from sklearn.metrics import make_scorer

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import TimeSeriesSplit

from sklearn.linear_model import Lasso, Ridge

from sklearn.ensemble import RandomForestRegressor

from xgboost.sklearn import XGBRegressor
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/stocknews/Combined_News_DJIA.csv",low_memory=False,

                    parse_dates=[0])



full_stock = pd.read_csv("../input/stocknews/DJIA_table.csv",low_memory=False,

                    parse_dates=[0])



#add the closing stock value to the df - this will be the y variable

df["Close"]=full_stock.Close



#show how the dataset looks like

df.head(5)
#drop the label column

df = df.drop(["Label"], axis=1)
#check for NAN

df.isnull().sum()
df = df.replace(np.nan, ' ', regex=True)



#sanity check

df.isnull().sum().sum()
df = df.replace('b\"|b\'|\\\\|\\\"', '', regex=True)

df.head(2)
Anakin = SentimentIntensityAnalyzer()



Anakin.polarity_scores(" ")
def detect_subjectivity(text):

    return TextBlob(text).sentiment.subjectivity



detect_subjectivity(" ") #should return 0
#get the headline columns' names

cols = []

for i in range(1,26):

    col = ("Top{}".format(i))

    cols.append(col)
start_vect=time.time()

print("ANAKIN: 'Intializing the process..'")



#get the name of the headline columns

cols = []

for i in range(1,26):

    col = ("Top{}".format(i))

    cols.append(col)





for col in cols:

    df[col] = df[col].astype(str) # Make sure data is treated as a string

    df[col+'_comp']= df[col].apply(lambda x:Anakin.polarity_scores(x)['compound'])

    df[col+'_sub'] = df[col].apply(detect_subjectivity)

    print("{} Done".format(col))

    

print("VADER: Vaderization completed after %0.2f Minutes"%((time.time() - start_vect)/60))
#the text isn't required anymore

df = df.drop(cols,axis=1)

df.head(5)
comp_cols = []

for col in cols:

    comp_col = col + "_comp"

    comp_cols.append(comp_col)



w = np.arange(1,26,1).tolist()

w.reverse()



weighted_comp = []

max_comp = []

min_comp = []

for i in range(0,len(df)):

    a = df.loc[i,comp_cols].tolist()

    weighted_comp.append(np.average(a, weights=w))

    max_comp.append(max(a))

    min_comp.append(min(a))



df['compound_mean'] = weighted_comp

df['compound_max'] = max_comp

df['compound_min'] = min_comp





sub_cols = []

for col in cols:

    sub_col = col + "_sub"

    sub_cols.append(sub_col)





weighted_sub = []

max_sub = []

min_sub = []

for i in range(0,len(df)):

    a = df.loc[i,sub_cols].tolist()

    weighted_sub.append(np.average(a, weights=w))

    max_sub.append(max(a))

    min_sub.append(min(a))



df['subjectivity_mean'] = weighted_sub

df['subjectivity_max'] = max_sub

df['subjectivity_min'] = min_sub



to_drop = sub_cols+comp_cols

df = df.drop(to_drop, axis=1)
df.head(5)
fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=df.Date, y=df.Close,

                    mode='lines'))

title = []

title.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Development of stock values from Aug, 2008 to Jun, 2016',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig1.update_layout(xaxis_title='Date',

                   yaxis_title='Closing stock value (in $)',

                  annotations=title)

fig1.show()
#function for quick plotting and testing of stationarity

def stationary_plot(y, lags=None, figsize=(12, 7), style='bmh'):

    """

        Plot time series, its ACF and PACF, calculate Dickey–Fuller test

        

        y - timeseries

        lags - how many lags to include in ACF, PACF calculation

    """

    if not isinstance(y, pd.Series):

        y = pd.Series(y)

        

    with plt.style.context(style):    

        fig = plt.figure(figsize=figsize)

        layout = (2, 2)

        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)

        acf_ax = plt.subplot2grid(layout, (1, 0))

        pacf_ax = plt.subplot2grid(layout, (1, 1))

        

        y.plot(ax=ts_ax)

        p_value = sm.tsa.stattools.adfuller(y)[1]

        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))

        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)

        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)

        plt.tight_layout()
stationary_plot(df.Close)
diff = df.Close - df.Close.shift(7)

stationary_plot(diff[7:])
diff2 = diff - diff.shift(1)

stationary_plot(diff2[7+1:], lags=60)
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=df.Date, y=df.compound_mean,

                    mode='lines',

                    name='Mean'))

fig2.add_trace(go.Scatter(x=df.Date, y=df.compound_max,

                    mode='lines',

                    name='Maximum'))

fig2.add_trace(go.Scatter(x=df.Date, y=df.compound_min,

                    mode='lines',

                    name='Minimum'))

title = []

title.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Development of sentiment compound score',

                               font=dict(family='Arial',

                                       size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig2.update_layout(xaxis_title='Date',

                   yaxis_title='Compound score',

                  annotations=title)

fig2.show()
compm_hist = px.histogram(df, x="compound_mean")

compm_hist.show()
fig3 = go.Figure()

fig3.add_trace(go.Scatter(x=df.Date, y=df.subjectivity_mean,

                    mode='lines',

                    name='Mean'))

fig3.add_trace(go.Scatter(x=df.Date, y=df.subjectivity_min,

                    mode='lines',

                    name='Min'))

fig3.add_trace(go.Scatter(x=df.Date, y=df.subjectivity_max,

                    mode='lines',

                    name='Max'))

title = []

title.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Development of subjectivity score',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig3.update_layout(xaxis_title='Date',

                   yaxis_title='Subjectivity score',

                  annotations=title)

fig3.show()
subm_hist = px.histogram(df, x="subjectivity_mean")

subm_hist.show()
df.describe()
def unique_ratio (col):

    return len(np.unique(col))/len(col)



cols = ['Close', 'compound_mean', 'compound_max', 'compound_min', 'subjectivity_mean', 'subjectivity_max', 'subjectivity_min']



ur = []

var = []

for col in cols:

    ur.append(unique_ratio(df[col]))

    var.append(np.var(df[col]))

    

feature_sel = pd.DataFrame({'Column': cols, 

              'Unique': ur,

              'Variance': var})

feature_sel
sel_fig = go.Figure(data=go.Scatter(

    x=feature_sel.Column,

    y=feature_sel.Unique,

    mode='markers',

    marker=dict(size=(feature_sel.Unique*100)),

))

sel_fig.update_layout(title='Ratio of unique values', 

                      yaxis_title='Unique ratio')

sel_fig.show()
drop = ['subjectivity_min', 'subjectivity_max']

clean_df = df.drop(drop,axis=1)
lag_df = clean_df.copy()

lag_df.head(3)
to_lag = list(lag_df.columns)

to_lag_4 = to_lag[1]

to_lag_1 = to_lag[2:len(to_lag)]
#lagging text features two days back

for col in to_lag_1:

    for i in range(1,3):

        new_name = col + ('_lag_{}'.format(i))

        lag_df[new_name] = lag_df[col].shift(i)

    

#lagging closing values 4 days back

for i in range(1, 5):

    new_name = to_lag_4 + ('_lag_{}'.format(i))

    lag_df[new_name] = lag_df[to_lag_4].shift(i)
#Show many rows need to be removed

lag_df.head(10) 
lag_df = lag_df.drop(lag_df.index[[np.arange(0,4)]])

lag_df = lag_df.reset_index(drop=True)



#sanity check for NaNs

lag_df.isnull().sum().sum()
lag_df.head(5)
# for time-series cross-validation set 10 folds 

tscv = TimeSeriesSplit(n_splits=10)
def mape(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

scorer = make_scorer(mean_squared_error)

scaler = StandardScaler()   
def ts_train_test_split(X, y, test_size):

    """

        Perform train-test split with respect to time series structure

    """

    

    # get the index after which test set starts

    test_index = int(len(X)*(1-test_size))

    

    X_train = X.iloc[:test_index]

    y_train = y.iloc[:test_index]

    X_test = X.iloc[test_index:]

    y_test = y.iloc[test_index:]

    

    return X_train, X_test, y_train, y_test
X = lag_df.drop(['Close'],axis=1)

X.index = X["Date"]

X = X.drop(['Date'],axis=1)

y = lag_df.Close



X_train, X_test, y_train, y_test = ts_train_test_split(X, y, test_size = 0.2)



#sanity check

(len(X_train)+len(X_test))==len(X)
#function for plotting coeficients of models (lasso and XGBoost)

def plotCoef(model,train_x):

    """

        Plots sorted coefficient values of the model

    """

    

    coefs = pd.DataFrame(model.coef_, train_x.columns)

    coefs.columns = ["coef"]

    coefs["abs"] = coefs.coef.apply(np.abs)

    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

    

    plt.figure(figsize=(15, 7))

    coefs.coef.plot(kind='bar')

    plt.grid(True, axis='y')

    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');
econ_cols = list(X_train.columns)

econ_cols = econ_cols[12:17]

X_train_e = X_train[econ_cols]

X_test_e = X_test[econ_cols]

y_train_e = y_train

y_test_e = y_test
econ_perf = pd.DataFrame(columns=['Model','MSE', 'SD'])

econ_perf
ridge_param = {'model__alpha': list(np.arange(0.001,1,0.001))}

ridge = Ridge(max_iter=5000)

pipe = Pipeline([

    ('scale', scaler),

    ('model', ridge)])

search_ridge = GridSearchCV(estimator=pipe,

                          param_grid = ridge_param,

                          scoring = scorer,

                          cv = tscv,

                          n_jobs=4

                         )

search_ridge.fit(X_train_e, y_train_e)
ridge_e = search_ridge.best_estimator_



#get cv results of the best model + confidence intervals

from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(ridge_e, X_train_e, y_train_e, cv=tscv, scoring=scorer)

econ_perf = econ_perf.append({'Model':'Ridge', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

ridge_e
plotCoef(ridge_e['model'], X_train_e)
coefs = ridge_e['model'].coef_

ridge_coefs = pd.DataFrame({'Coef': coefs,

                           'Name': list(X_train_e.columns)})

ridge_coefs["abs"] = ridge_coefs.Coef.apply(np.abs)

ridge_coefs = ridge_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

ridge_coefs
econ_perf
rf_param = {'model__n_estimators': [10, 100, 300],

            'model__max_depth': [10, 20, 30, 40],

            'model__min_samples_split': [2, 5, 10],

            'model__min_samples_leaf': [1, 2, 3],

            'model__max_features': ["auto", 'sqrt']}
rf = RandomForestRegressor()

pipe = Pipeline([

    ('scale', scaler),

    ('model', rf)])

gridsearch_rf = GridSearchCV(estimator=pipe,

                          param_grid = rf_param,

                          scoring = scorer,

                          cv = tscv,

                          n_jobs=4,

                          verbose=3

                         )
gridsearch_rf.fit(X_train_e, y_train_e)
rf_e = gridsearch_rf.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(rf_e, X_train_e, y_train_e, cv=tscv, scoring=scorer)

econ_perf = econ_perf.append({'Model':'RF', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)
xgb_param = {'model__lambda': list(np.arange(0.1,3, 0.1)), #L2 regularisation

             'model__alpha': list(np.arange(0.1,3, 0.1)),  #L1 regularisation

            }
xgb = XGBRegressor(booster='gblinear', feature_selector='shuffle', objective='reg:squarederror')



pipe = Pipeline([

    ('scale', scaler),

    ('model', xgb)])

gridsearch_xgb = GridSearchCV(estimator=pipe,

                          param_grid = xgb_param,

                          scoring = scorer,

                          cv = tscv,

                          n_jobs=4,

                          verbose=3

                         )
gridsearch_xgb.fit(X_train_e, y_train_e)
xgb_e = gridsearch_xgb.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(xgb_e, X_train_e, y_train_e, cv=tscv, scoring=scorer)

econ_perf = econ_perf.append({'Model':'XGB', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

xgb_e
print(econ_perf)
econ_fig = px.scatter(econ_perf, x="Model", y='MSE', color='Model', error_y="SD")

econ_fig.update_layout(title_text="Performance of models trained on lags of y")

econ_fig.show()
X_train_n = X_train.drop(econ_cols, axis=1)

X_test_n = X_test.drop(econ_cols, axis=1)

y_train_n = y_train

y_test_n = y_test
nlp_perf = pd.DataFrame(columns=['Model','MSE', 'SD'])

nlp_perf
ridge_param = {'model__alpha': list(np.arange(1,10,0.1))}

ridge = Ridge(max_iter=5000)

pipe = Pipeline([

    ('scale', scaler),

    ('model', ridge)

])

search_ridge = GridSearchCV(estimator=pipe,

                          param_grid = ridge_param,

                          scoring = scorer,

                          cv = tscv,

                          n_jobs=4

                         )

search_ridge.fit(X_train_n, y_train_n)
ridge_n = search_ridge.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(ridge_n, X_train_n, y_train_n, cv=tscv, scoring=scorer)

nlp_perf = nlp_perf.append({'Model':'Ridge', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

ridge_n
plotCoef(ridge_n['model'], X_train_n)



coefs = ridge_n['model'].coef_

ridge_coefs = pd.DataFrame({'Coef': coefs,

                           'Name': list(X_train_n.columns)})

ridge_coefs["abs"] = ridge_coefs.Coef.apply(np.abs)

ridge_coefs = ridge_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

ridge_coefs
mape(y_test, ridge_n.predict(X_test_n))
rf_param = {'model__n_estimators': [10, 100, 300],

            'model__max_depth': [10, 20, 30, 40],

            'model__min_samples_split': [2, 5, 10],

            'model__min_samples_leaf': [1, 2, 3],

            'model__max_features': ["auto", 'sqrt']}

rf = RandomForestRegressor()

pipe = Pipeline([

    ('scale', scaler),

    ('model', rf)])

gridsearch_rf = GridSearchCV(estimator=pipe,

                          param_grid = rf_param,

                          scoring = scorer,

                          cv = tscv,

                          n_jobs=4,

                          verbose=3

                         )

gridsearch_rf.fit(X_train_n, y_train_n)
rf_n = gridsearch_rf.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(rf_n, X_train_n, y_train_n, cv=tscv, scoring=scorer)

nlp_perf = nlp_perf.append({'Model':'RF', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)
xgb_param = {'model__lambda': list(np.arange(1,10, 1)), #L2 regularisation

             'model__alpha': list(np.arange(1,10, 1)),  #L1 regularisation

            }

xgb = XGBRegressor(booster='gblinear', feature_selector='shuffle', objective='reg:squarederror')



pipe = Pipeline([

    ('scale', scaler),

    ('model', xgb)])

gridsearch_xgb = GridSearchCV(estimator=pipe,

                          param_grid = xgb_param,

                          scoring = scorer,

                          cv = tscv,

                          n_jobs=4,

                          verbose=3

                         )

gridsearch_xgb.fit(X_train_n, y_train_n)
xgb_n = gridsearch_xgb.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(xgb_n, X_train_n, y_train_n, cv=tscv, scoring=scorer)

nlp_perf = nlp_perf.append({'Model':'XGB', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

xgb_n
print(nlp_perf)
nlp_fig = px.scatter(nlp_perf, x="Model", y='MSE', color='Model', error_y="SD")

#nlp_fig.update_layout(title_text="Performance of models trained on NLP features",

nlp_fig.show()
en_perf = pd.DataFrame(columns=['Model','MSE', 'SD'])

en_perf
ridge_param = {'model__alpha': list(np.arange(0.1,1,0.01))}

ridge = Ridge(max_iter=5000)

pipe = Pipeline([

    ('scale', scaler),

    ('model', ridge)])

search_ridge = GridSearchCV(estimator=pipe,

                          param_grid = ridge_param,

                          scoring = scorer,

                          cv = tscv,

                          n_jobs=4,

                          verbose=3

                         )

search_ridge.fit(X_train, y_train)
ridge_en = search_ridge.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(ridge_en, X_train, y_train, cv=tscv, scoring=scorer)

en_perf = en_perf.append({'Model':'Ridge', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

ridge_en
coefs = ridge_en['model'].coef_

ridge_coefs = pd.DataFrame({'Coef': coefs,

                           'Name': list(X_train.columns)})

ridge_coefs["abs"] = ridge_coefs.Coef.apply(np.abs)

ridge_coefs = ridge_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

ridge_coefs
plotCoef(ridge_en['model'], X_train)
rf_param = {'model__n_estimators': [10, 100, 300],

            'model__max_depth': [10, 20, 30, 40],

            'model__min_samples_split': [2, 5, 10],

            'model__min_samples_leaf': [1, 2, 3],

            'model__max_features': ["auto", 'sqrt']}

rf = RandomForestRegressor()

pipe = Pipeline([

    ('scale', scaler),

    ('model', rf)])

gridsearch_rf = GridSearchCV(estimator=pipe,

                          param_grid = rf_param,

                          scoring = scorer,

                          cv = tscv,

                          n_jobs=4,

                          verbose=3

                         )

gridsearch_rf.fit(X_train, y_train)
rf_en = gridsearch_rf.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(rf_en, X_train, y_train, cv=tscv, scoring=scorer)

en_perf = en_perf.append({'Model':'RF', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

rf_en
xgb_param = {'model__lambda': list(np.arange(1,10, 1)), #L2 regularisation

             'model__alpha': list(np.arange(1,10, 1)),  #L1 regularisation

            }

xgb = XGBRegressor(booster='gblinear', feature_selector='shuffle', objective='reg:squarederror')



pipe = Pipeline([

    ('scale', scaler),

    ('model', xgb)])

gridsearch_xgb = GridSearchCV(estimator=pipe,

                          param_grid = xgb_param,

                          scoring = scorer,

                          cv = tscv,

                          n_jobs=4,

                          verbose=3

                         )

gridsearch_xgb.fit(X_train, y_train)
xgb_en = gridsearch_xgb.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(xgb_en, X_train, y_train, cv=tscv, scoring=scorer)

en_perf = en_perf.append({'Model':'XGB', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

xgb_en
from sklearn.model_selection import cross_val_predict



X_train_stack = pd.DataFrame(pd.DataFrame(columns=['econ_r', 'nlp_r']))

X_train_stack['econ_r'] = cross_val_predict(ridge_e, X_train_e, y_train, cv=10)

X_train_stack['nlp_r'] = cross_val_predict(ridge_n, X_train_n, y_train, cv=10)



X_test_stack = pd.DataFrame(pd.DataFrame(columns=['econ_r', 'nlp_r']))

X_test_stack['econ_r'] = ridge_e.predict(X_test_e)

X_test_stack['nlp_r'] = ridge_n.predict(X_test_n)



X_train_stack.to_csv("Stack_train.csv")

X_test_stack.to_csv("Stack_test.csv")



from sklearn.linear_model import ElasticNetCV

stack = ElasticNetCV(cv=tscv)

stack.fit(X_train_stack, y_train)

cv_score = cross_val_score(stack, X_train_stack, y_train, cv=tscv, scoring=scorer)

stack_performance = {'Model':'XGB', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}

stack_performance



mape(y_test, stack.predict(X_test_stack))
coefs = stack.coef_

ridge_coefs = pd.DataFrame({'Coef': coefs,

                           'Name': list(X_train_stack.columns)})

ridge_coefs["abs"] = ridge_coefs.Coef.apply(np.abs)

ridge_coefs = ridge_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

print(ridge_coefs)

plotCoef(stack, X_train_stack)
prediction_compare = pd.DataFrame(pd.DataFrame(columns=['y_true', 'econ_r', 'econ_rf', 'econ_x', 'nlp_r', 'nlp_rf', 'nlp_x', 'comb_r', 'comb_rf', 'comb_x', 'stack']))

prediction_compare['y_true'] = y_test

prediction_compare['econ_r'] = ridge_e.predict(X_test_e)

prediction_compare['econ_rf'] = rf_e.predict(X_test_e)

prediction_compare['econ_x'] = xgb_e.predict(X_test_e)

prediction_compare['nlp_r'] = ridge_n.predict(X_test_n)

prediction_compare['nlp_rf'] = rf_n.predict(X_test_n)

prediction_compare['nlp_x'] = xgb_n.predict(X_test_n)

prediction_compare['comb_r'] = ridge_en.predict(X_test)

prediction_compare['comb_rf'] = rf_en.predict(X_test)

prediction_compare['comb_x'] = xgb_en.predict(X_test)

prediction_compare['stack'] = stack.predict(X_test_stack)



prediction_compare.sample(3)
econ_perf.to_csv("econ_perf.csv")

nlp_perf.to_csv("nlp_perf.csv")

en_perf.to_csv("en_perf.csv")

prediction_compare.to_csv('compare_predictions.csv')

X_test.to_csv('X_test.csv')