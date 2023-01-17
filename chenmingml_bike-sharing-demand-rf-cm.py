#load data, review the field and data type 
import pandas as pd
import numpy as np
df_train=pd.read_csv("../input/train.csv", header=0)
df_train.head()
df_train.dtypes
df_train.describe()
df_train.shape
df_train.count()
#key is the time and need to separate the time, the datetime includes: month, day, year, hour (24h)
df_train['hour']=pd.DatetimeIndex(df_train.datetime).hour
df_train['day']=pd.DatetimeIndex(df_train.datetime).dayofweek
df_train['month']=pd.DatetimeIndex(df_train.datetime).month
#other method
#df_train['dt']=pd.to_datetime(df_train['datetime'])
#df_train['day_of_week']=df_train['dt'].apply(lambda x:x.dayofweek)
#df_train['day_of_month]=df_train['dt'].apply(lambda x:x.day)
df_train.head()
#refine the columns 
#df=df_train.drop(['datetime','casual','registered'], axis=1, inplace=True)
df_train=df_train[['season', 'holiday','workingday','weather','temp','atemp','humidity','windspeed','count','month','day','hour']]
df_train.head()
df_train_target=df_train['count'].values
df_train_target.shape
df_train_data=df_train.drop(['count'],axis=1).values
df_train_data.shape
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.learning_curve import learning_curve
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import explained_variance_score
#train, text/cross validation
cv=cross_validation.ShuffleSplit(len(df_train_data), n_iter=3, test_size=0.2, random_state=101)
#No.1, Ridge regression
for train, test in cv:
    svc1=linear_model.Ridge().fit(df_train_data[train], df_train_target[train])
    print ("train score: {0:.3f}, test score:{1:.3f}\n".format(svc1.score(df_train_data[train],df_train_target[train]),svc1.score(df_train_data[test], df_train_target[test]))) 
#No.2, SVM, SVR(kernel='rbf', C=10, gamma=0.001)
for train, test in cv:
    svc2=svm.SVR(kernel='rbf', C=10, gamma=0.001).fit(df_train_data[train], df_train_target[train])
    print ("train score:{0:.3f}, test score:{1:.3f}\n".format(svc2.score(df_train_data[train],df_train_target[train]),svc2.score(df_train_data[test], df_train_target[test])))
#No.3, Random Forest (n_estimators=100)
for train, test in cv:
    svc3=RandomForestRegressor(n_estimators=100).fit(df_train_data[train],df_train_target[train])
    print ("train score:{0:.3f}, test score:{1:.3f}\n".format(svc3.score(df_train_data[train], df_train_target[train]), svc3.score(df_train_data[test], df_train_target[test])))
# svc3>>svc2>svc1, randomforest is the best!
#Grid search to get optimization
X=df_train_data
y=df_train_target
X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y, test_size=0.2, random_state=102)
tuned_parameters=[{'n_estimators':[10,100,500,600]}]
scores=['r2']
for score in scores:
    print (score)
    clf=GridSearchCV(RandomForestRegressor(),tuned_parameters, cv=5,scoring=score)
clf.fit(X_train, y_train)
print ('found the best:')
print (clf.best_estimator_)
print ("scores are:")
print("")
for params, mean_score, scores in clf.grid_scores_:
    print ("%0.3f(+/-%0.03f) for %r" %(mean_score, scores.std()/2, params))
print("")
#Best n_estimators:600
rf = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=600, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
rf.fit(X_train, y_train)
print ("train score:{0:.3f}\n".format(rf.score(X_train, y_train)))
#prediction with Randomforest
#import test dataset
df_test=pd.read_csv("../input/test.csv", header=0)
df_test.head()
df_test.describe()
df_test.count()
df_test['hour']=pd.DatetimeIndex(df_test.datetime).hour
df_test['day']=pd.DatetimeIndex(df_test.datetime).dayofweek
df_test['month']=pd.DatetimeIndex(df_test.datetime).month
df_test.head()
df_submission=pd.DataFrame(df_test['datetime'], columns=['datetime'])
df_submission.head()
df_test=df_test[['season', 'holiday','workingday','weather','temp','atemp','humidity','windspeed','month','day','hour']]
df_test.shape
df_test_target = rf.predict(df_test)
df_test_target.shape
df_submission['count']=df_test_target
df_submission.head()
#df_submission.to_csv('bike_share_result.csv',index=False)
