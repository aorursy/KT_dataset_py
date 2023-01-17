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

from xgboost.sklearn import XGBRegressor, XGBClassifier
df = pd.read_csv("../input/stocknews/Combined_News_DJIA.csv",low_memory=False,

                    parse_dates=[0])

#use date as index

df.index = df.Date

df = df.drop(["Date"], axis=1)



full_stock = pd.read_csv("../input/stocknews/DJIA_table.csv",low_memory=False,

                    parse_dates=[0])

full_stock.index = full_stock.Date

full_stock = full_stock.drop(["Date"], axis=1)



#calculate the difference between opening and closing stock value

full_stock['Diff'] = full_stock.Close - full_stock.Open

fl_cols = list(full_stock.columns)

fl_cols = fl_cols[0:6]

full_stock = full_stock.drop(fl_cols, axis=1)



#merge the headlines together into one text

headlines = []

for row in range(0,len(df.index)):

   headlines.append(' '.join(str(x) for x in df.iloc[row,2:27]))



df['Headlines'] = headlines



#add the difference between opening and closing stock value to the df - this will be the y variable

df = df.merge(full_stock, left_index=True, right_index=True)



#drop the Label column and Top1-Top25

drop_it = df.columns

drop_it = drop_it[0:26]

df = df.drop(drop_it, axis=1)



#show how the dataset looks like

df.head(5)
df = df.replace(np.nan, ' ', regex=True)



#sanity check

df.isnull().sum().sum()
df = df.replace('b\"|b\'|\\\\|\\\"|\'', '', regex=True)

df = df.replace('[0-9]', '', regex=True)

df.head(5)
df.shape
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
X = df.copy()

X = X.drop(['Diff'],axis=1)

y = df.Diff
mean = np.mean(y)

sd = np.std(y)

y = (y-mean)/sd
X_train, X_test, y_train, y_test = ts_train_test_split(X, y, test_size = 0.1)



#remove first 7 rows of training set to acocunt for rows that will be removed later due to lagging

X_train = X_train.drop(X_train.index[[np.arange(0,7)]])

y_train = y_train[7:len(y_train)]





#save the train-test indeces

test_idx = y_test.index
Anakin = SentimentIntensityAnalyzer()



Anakin.polarity_scores(" ")
def detect_subjectivity(text):

    return TextBlob(text).sentiment.subjectivity



detect_subjectivity(" ") #should return 0
start_vect=time.time()

print("ANAKIN: 'Intializing the process..'")

col="Headlines"





df[col] = df[col].astype(str) # Make sure data is treated as a string

df[col+'_pos']= df[col].apply(lambda x:Anakin.polarity_scores(x)['pos'])

df[col+'_neg']= df[col].apply(lambda x:Anakin.polarity_scores(x)['neg'])

df[col+'_comp']= df[col].apply(lambda x:Anakin.polarity_scores(x)['compound'])

df[col+'_sub'] = df[col].apply(detect_subjectivity)

    

print("VADER: Vaderization completed after %0.2f Minutes"%((time.time() - start_vect)/60))
from sklearn.feature_extraction.text import CountVectorizer



ngrammer = CountVectorizer(ngram_range=(1, 2), lowercase=True)

n_grams_train = ngrammer.fit_transform(X_train.Headlines)

n_grams_test = ngrammer.transform(X_test.Headlines)
n_grams_train.shape
#the text isn't required anymore

df = df.drop(col,axis=1)

df.head(5)
fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=df.index, y=df.Diff,

                    mode='lines'))

title = []

title.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Aug, 2008 - Jun, 2016',

                              font=dict(family='Arial',

                                        size=30,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig1.update_layout(xaxis_title='Date',

                   yaxis_title='Difference between opening and closing value (in $)',

                  annotations=title)

fig1.show()
pairplot = sns.pairplot(df)
df.describe()
def unique_ratio (col):

    return len(np.unique(col))/len(col)



cols = list(df.columns)

cols = cols[1:len(cols)]



ur = []

var = []

for col in cols:

    ur.append(unique_ratio(df[col]))

    var.append(np.var(df[col]))

    

feature_sel = pd.DataFrame({'Column': cols, 

              'Unique': ur,

              'Variance': var})

feature_sel
uniq_fig = go.Figure(data=go.Scatter(

    x=feature_sel.Column,

    y=feature_sel.Unique,

    mode='markers'

))

uniq_fig.update_layout( yaxis_title='Unique ratio')

uniq_fig.show()
var_fig = go.Figure(data=go.Scatter(

    x=feature_sel.Column,

    y=feature_sel.Variance,

    mode='markers'

))

var_fig.update_layout( yaxis_title='Variance')

var_fig.show()
drop = ['Headlines_pos']

clean_df = df.copy()

clean_df = clean_df.drop(drop, axis=1)
lag_df = clean_df.copy()

lag_df.head(3)
to_lag = list(lag_df.columns)

to_lag_7 = to_lag[0]

to_lag_3 = to_lag[1:len(to_lag)]
#lagging text features two days back

for col in to_lag_3:

    for i in range(1, 4):

        new_name = col + ('_lag_{}'.format(i))

        lag_df[new_name] = lag_df[col].shift(i)

    

#lagging closing values 7 days back

for i in range(1, 8):

    new_name = to_lag_7 + ('_lag_{}'.format(i))

    lag_df[new_name] = lag_df[to_lag_7].shift(i)
lag_df.head(10)
lag_df = lag_df.drop(lag_df.index[[np.arange(0,7)]])



#sanity check for NaNs

lag_df.isnull().sum().sum()
lag_df.head(10)
lag_df["Day"] = lag_df.index.dayofweek

lag_df["Month"] = lag_df.index.month

lag_df["Year"] = lag_df.index.year

lag_df["Quarter"] = lag_df.index.quarter
# for time-series cross-validation set 5 folds 

tscv = TimeSeriesSplit(n_splits=5)
lag_df = lag_df.drop(['Diff'], axis=1)

print(min(test_idx))

X_train = lag_df[lag_df.index < min(test_idx)]

X_test = lag_df[lag_df.index >= min(test_idx)]
#classifier 1 - already prepared from before

#n_grams_train

#n_grams_test

X_train.columns

#classifier 2 - the same as X_train

X_train_c2 = X_train

X_test_c2 = X_test



#Econometric

drop_e = list(X_train.columns)

drop_e = drop_e[0:12]

X_train_e = X_train.drop(drop_e, axis=1)

X_test_e = X_test.drop(drop_e, axis=1)



#NLP

X_train_n = X_train[drop_e]

X_test_n = X_test[drop_e]
y_train_dir = []

for i in range(0,len(y_train)):

    if y_train[i]<0: y_train_dir.append(0)

    else: y_train_dir.append(1)

        

y_test_dir = []

for i in range(0,len(y_test)):

    if y_test[i]<0: y_test_dir.append(0)

    else: y_test_dir.append(1)
from sklearn.metrics import balanced_accuracy_score

scorer_class = make_scorer(balanced_accuracy_score)
class_perf = pd.DataFrame(columns=['Model','Acc', 'SD'])
from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()



nb_param = {'alpha': list(np.arange(0,1,0.01))}

search_nb = GridSearchCV(estimator=NB,

                          param_grid = nb_param,

                          scoring = scorer_class,

                          cv = tscv,

                          n_jobs=4,

                          verbose=2

                         )

search_nb.fit(X=n_grams_train, y=y_train_dir)
nb_c1 = search_nb.best_estimator_



#get cv results of the best model + confidence intervals

from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(nb_c1, n_grams_train, y_train_dir, cv=tscv, scoring=scorer_class)

class_perf = class_perf.append({'Model':'NB_c1', 'Acc':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

nb_c1
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(class_weight='balanced') #using l2



lr_param = {'C': list(np.arange(0.1,1,0.1))}

search_lr = GridSearchCV(estimator=lr,

                          param_grid = lr_param,

                          scoring = scorer_class,

                          cv = tscv,

                          n_jobs=4,

                          verbose=2

                         )

search_lr.fit(n_grams_train, y_train_dir)
lr_c1 = search_lr.best_estimator_



#get cv results of the best model + confidence intervals

from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(lr_c1, n_grams_train, y_train_dir, cv=tscv, scoring=scorer_class)

class_perf = class_perf.append({'Model':'Logistic Regression_c1', 'Acc':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

lr_c1
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.naive_bayes import GaussianNB

nb_c2 = GaussianNB()



nb_c2.fit(X=X_train, y=y_train_dir)



#get cv results of the best model + confidence intervals

from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(nb_c2, X_train, y_train_dir, cv=tscv, scoring=scorer_class)

class_perf = class_perf.append({'Model':'NB_c2', 'Acc':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

nb_c2
lr = LogisticRegression(class_weight='balanced') #using l2



lr_param = {'C': list(np.arange(0.1,1,0.1))}

search_lr = GridSearchCV(estimator=lr,

                          param_grid = lr_param,

                          scoring = scorer_class,

                          cv = tscv,

                          n_jobs=4,

                          verbose=2

                         )

search_lr.fit(X_train, y_train_dir)



lr_c2 = search_lr.best_estimator_



#get cv results of the best model + confidence intervals

from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(lr_c2, X_train, y_train_dir, cv=tscv, scoring=scorer_class)

class_perf = class_perf.append({'Model':'Logistic Regression_c2', 'Acc':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

lr_c2
lr_c2
print(class_perf)
from sklearn.model_selection import cross_val_predict

stack = LogisticRegression(class_weight='balanced')



stack_train = pd.DataFrame(pd.DataFrame(columns=['nb_c2', 'lr_c2']))

stack_train['nb_c2'] = nb_c2.predict(X_train)

stack_train['lr_c2'] = lr_c2.predict(X_train)



stack_test = pd.DataFrame(pd.DataFrame(columns=['nb_c2', 'lr_c2']))

stack_test['nb_c2'] = nb_c2.predict(X_test)

stack_test['lr_c2'] = lr_c2.predict(X_test)





stack.fit(stack_train, y_train_dir)



pred_dir_train = cross_val_predict(stack, stack_train, y_train_dir)

pred_dir_test = stack.predict(stack_test)



#add the stack to the classifier performance table

cv_score = cross_val_score(stack, stack_train, y_train_dir, cv=tscv, scoring=scorer_class)

class_perf = class_perf.append({'Model':'Stack', 'Acc':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)
class_perf
X_train_n.loc[:,'Direction'] = pred_dir_train

X_test_n.loc[:,'Direction'] = pred_dir_test
def mape(y_true, y_pred): 

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

scorer = make_scorer(mean_squared_error)

scaler = StandardScaler()
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
econ_perf = pd.DataFrame(columns=['Model','MSE', 'SD'])
ridge_param = {'model__alpha': list(np.arange(1,10,0.1))}

ridge = Ridge(max_iter=5000)

pipe = Pipeline([

    ('scale', scaler),

    ('model', ridge)])

search_ridge = GridSearchCV(estimator=pipe,

                          param_grid = ridge_param,

                          scoring = scorer,

                          cv = tscv,

                          n_jobs=4,

                          verbose=2

                         )

search_ridge.fit(X_train_e, y_train)
ridge_e = search_ridge.best_estimator_



#get cv results of the best model + confidence intervals

from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(ridge_e, X_train_e, y_train, cv=tscv, scoring=scorer)

econ_perf = econ_perf.append({'Model':'Ridge', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

ridge_e
plotCoef(ridge_e['model'], X_train_e)
coefs = ridge_e['model'].coef_

ridge_coefs = pd.DataFrame({'Coef': coefs,

                           'Name': list(X_train_e.columns)})

ridge_coefs["abs"] = ridge_coefs.Coef.apply(np.abs)

ridge_coefs = ridge_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

ridge_coefs
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

gridsearch_rf.fit(X_train_e, y_train)
rf_e = gridsearch_rf.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(rf_e, X_train_e, y_train, cv=tscv, scoring=scorer)

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

gridsearch_xgb.fit(X_train_e, y_train)
xgb_e = gridsearch_xgb.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(xgb_e, X_train_e, y_train, cv=tscv, scoring=scorer)

econ_perf = econ_perf.append({'Model':'XGB', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

xgb_e
plotCoef(xgb_e['model'], X_train_e)



coefs = xgb_e['model'].coef_

xgb_coefs = pd.DataFrame({'Coef': coefs,

                           'Name': list(X_train_e.columns)})

xgb_coefs["abs"] = xgb_coefs.Coef.apply(np.abs)

xgb_coefs = xgb_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

xgb_coefs
print(econ_perf)

econ_fig = px.scatter(econ_perf, x="Model", y='MSE', color='Model', error_y="SD")

econ_fig.show()
nlp_perf = pd.DataFrame(columns=['Model','MSE', 'SD'])
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

search_ridge.fit(X_train_n, y_train)
ridge_n = search_ridge.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(ridge_n, X_train_n, y_train, cv=tscv, scoring=scorer)

nlp_perf = nlp_perf.append({'Model':'Ridge', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

ridge_n
plotCoef(ridge_n['model'], X_train_n)



coefs = ridge_n['model'].coef_

ridge_coefs = pd.DataFrame({'Coef': coefs,

                           'Name': list(X_train_n.columns)})

ridge_coefs["abs"] = ridge_coefs.Coef.apply(np.abs)

ridge_coefs = ridge_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

ridge_coefs
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

gridsearch_rf.fit(X_train_n, y_train)
rf_n = gridsearch_rf.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(rf_n, X_train_n, y_train, cv=tscv, scoring=scorer)

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

gridsearch_xgb.fit(X_train_n, y_train)
xgb_n = gridsearch_xgb.best_estimator_



#get cv results of the best model + confidence intervals

cv_score = cross_val_score(xgb_n, X_train_n, y_train, cv=tscv, scoring=scorer)

nlp_perf = nlp_perf.append({'Model':'XGB', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)

xgb_n
plotCoef(xgb_n['model'], X_train_n)



coefs = xgb_n['model'].coef_

xgb_coefs = pd.DataFrame({'Coef': coefs,

                           'Name': list(X_train_n.columns)})

xgb_coefs["abs"] = xgb_coefs.Coef.apply(np.abs)

xgb_coefs = xgb_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

xgb_coefs
print(nlp_perf)



nlp_fig = px.scatter(nlp_perf, x="Model", y='MSE', color='Model', error_y="SD")

nlp_fig.show()
prediction_compare = pd.DataFrame(pd.DataFrame(columns=['y_true', 'econ_r', 'econ_rf', 'econ_x', 'nlp_r', 'nlp_rf', 'nlp_x']))

prediction_compare['y_true'] = y_test

prediction_compare['econ_r'] = ridge_e.predict(X_test_e)

prediction_compare['econ_rf'] = rf_e.predict(X_test_e)

prediction_compare['econ_x'] = xgb_e.predict(X_test_e)

prediction_compare['nlp_r'] = ridge_n.predict(X_test_n)

prediction_compare['nlp_rf'] = rf_n.predict(X_test_n)

prediction_compare['nlp_x'] = xgb_n.predict(X_test_n)



prediction_compare.sample(5)
class_perf.to_csv('class_perf.csv')

econ_perf.to_csv("econ_perf.csv")

nlp_perf.to_csv("nlp_perf.csv")

prediction_compare.to_csv('compare_predictions.csv')



X_test = pd.DataFrame(data=X_test[1:,1:], 

                      index=X_test[1:,0],

                      columns=X_test[0,1:])

X_test.to_csv("X_test.csv")