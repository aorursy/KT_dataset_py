import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

import pandas_profiling as pf



from scipy.stats import uniform, randint

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

from sklearn.model_selection import cross_val_score, cross_val_predict,GridSearchCV, StratifiedKFold, KFold, RandomizedSearchCV, train_test_split



import xgboost as xgb

import imblearn
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz')

data.head()
pf.ProfileReport(data)
data.holiday.unique()
data.weather_main.unique()
data.weather_description.unique()
data.info()
data.date_time = pd.to_datetime(data.date_time)
data['weekday'] = data.date_time.dt.dayofweek
data.weekday.unique()
data['hour'] = data.date_time.dt.hour
data['month'] = data.date_time.dt.month
data.columns
data.drop('date_time',1, inplace=True)
data.loc[data['holiday']=='None']
data.holiday.unique()
data_holiday = data.holiday

data_holiday.value_counts()
data['holiday'] = data['holiday'].apply(lambda x: 'None' if x=='None' else 'Holiday')
data['holiday'] = data['holiday'].apply(lambda x: 0 if x=='None' else 1)
data.holiday.value_counts()


sns.distplot(data['holiday'])
le1 = LabelEncoder()

data['weather_main'] = le1.fit_transform(data['weather_main'])
data.weather_description.value_counts()
le2 = LabelEncoder()

data['weather_description'] = le1.fit_transform(data['weather_description'])
data.head()
data[['holiday','weather_main','weather_description', 'weekday', 'hour','month']] = data[['holiday','weather_main','weather_description', 'weekday', 'hour','month']].astype('category')
data.columns
X = data.drop('traffic_volume',1)

y = data.traffic_volume
X = pd.get_dummies(X)

X.head()
X.columns
data1 = data.drop('traffic_volume',1)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=1, max_depth=10)

model.fit(data1,y)

#Plot the feature importance to see the important features

features = data1.columns

importances = model.feature_importances_

indices = np.argsort(importances)[-9:] #Top 10 features

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
model = RandomForestRegressor(random_state=1, max_depth=10)

model.fit(X,y)

#Plot the feature importance to see the important features

features = X.columns

importances = model.feature_importances_

indices = np.argsort(importances)[-15:] #Top 10 features

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = sc_X.fit_transform(X)

sc_y = StandardScaler()

y = sc_y.fit_transform(y.values.reshape(-1,1))
from sklearn.decomposition import PCA

pca = PCA(n_components=4)

pca_result = pca.fit_transform(X)
#visualize using the explained_variance_ratio_



plt.plot(range(4),pca.explained_variance_ratio_)

plt.plot(range(4), np.cumsum(pca.explained_variance_ratio_))

plt.title('Component-wise and cumulative Explained variance')

plt.legend()
pca_result.shape
from sklearn.decomposition import FastICA

ICA = FastICA(n_components=3, random_state=12)

ICA_data = ICA.fit_transform(X)



plt.figure(figsize=(12,8))

plt.title('ICA Components')

plt.scatter(ICA_data[:,0], ICA_data[:,1])

plt.scatter(ICA_data[:,1], ICA_data[:,2])

plt.scatter(ICA_data[:,2], ICA_data[:,0])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
regressor.score(X_test,y_test)
np.sqrt(mean_squared_error(y_test,y_pred))
(sc_y.inverse_transform(y_test[:10]),sc_y.inverse_transform(y_pred[:10]))
from sklearn.linear_model import Ridge

regressor = Ridge()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

regressor.score(X_test,y_test),mean_squared_error(y_test,y_pred)
sc_y.inverse_transform(y_test[:10]),sc_y.inverse_transform(y_pred[:10])
from sklearn import neighbors

for k in range(3,10):

    n_neighbors=k

    regressor=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform',n_jobs=-1)

    regressor.fit(X_train,y_train)

    y_pred = regressor.predict(X_test)

    print(regressor.score(X_test,y_test),mean_squared_error(y_test,y_pred))
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(max_depth=20)

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

regressor.score(X_test,y_test),mean_squared_error(y_test,y_pred)
(sc_y.inverse_transform(y_test[:10]),sc_y.inverse_transform(y_pred[:10]))
from sklearn.svm import SVR

regressor=SVR()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

regressor.score(X_test,y_test),mean_squared_error(y_test,y_pred)
from sklearn.ensemble import RandomForestRegressor

estim = np.arange(100,1000,100)

for k in estim:

    regressor=RandomForestRegressor(n_estimators=k)

    regressor.fit(X_train,y_train)

    y_pred = regressor.predict(X_test)

    print(f'{k} estimators - {regressor.score(X_test,y_test),mean_squared_error(y_test,y_pred)}')
regressor=RandomForestRegressor(n_estimators=500,n_jobs=-1)

regressor.fit(X_train,y_train.ravel())

y_pred = regressor.predict(X_test)

regressor.score(X_test,y_test),mean_squared_error(y_test,y_pred)
(sc_y.inverse_transform(y_test[:10]),sc_y.inverse_transform(y_pred[:10]))
xgb_model = xgb.XGBRegressor(objective="reg:linear",  n_estimators=1000, random_state=42,n_jobs=-1)

xgb_model.fit(X_train,y_train,early_stopping_rounds=10,eval_set=[(X_test, y_test)])

#xgb_model.fit(X_train,y_train,eval_set=[(X_test, y_test)])

y_pred = xgb_model.predict(X_test)

xgb_model.score(X_test,y_test), mean_squared_error(y_test, y_pred)
xgb.plot_importance(xgb_model,max_num_features=10)
# converts the target tree to a graphviz instance

xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration)
def display_scores(scores):

    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
def report_best_scores(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
params = {

    "colsample_bytree": uniform(0.7, 0.3),

    "gamma": uniform(0, 0.5),

    "learning_rate": uniform(0.03, 0.3), # default 0.1 

    "max_depth": randint(2, 6), # default 3

    "n_estimators": randint(100, 150), # default 100

    "subsample": uniform(0.6, 0.4)

}



search = RandomizedSearchCV(xgb_model, param_distributions=params, random_state=42, n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True)



search.fit(X_train,y_train)



report_best_scores(search.cv_results_, 1)
xgb_model = xgb.XGBRegressor(objective="reg:linear",random_state=42,n_jobs=-1,colsample_bytree= 0.7467983561008608, gamma= 0.02904180608409973, learning_rate= 0.28985284373248055, max_depth= 5, n_estimators= 1000, subsample= 0.8832290311184181)

xgb_model.fit(X_train,y_train,early_stopping_rounds=10,eval_set=[(X_test, y_test)])

# xgb_model.fit(X_train,y_train,eval_set=[(X_test, y_test)])

y_pred = xgb_model.predict(X_test)

xgb_model.score(X_test,y_test), mean_squared_error(y_test, y_pred)
(sc_y.inverse_transform(y_test[:10]),sc_y.inverse_transform(y_pred[:10]))
print("best score: {0}, best iteration: {1}, best ntree limit {2}".format(xgb_model.best_score, xgb_model.best_iteration, xgb_model.best_ntree_limit))
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42,colsample_bytree= 0.9078671076075817, gamma= 0.17416830222659868, learning_rate=0.3109944455567122, max_depth= 5, subsample= 0.7671784126862315)



scores = cross_val_score(xgb_model, X, y, scoring="neg_mean_squared_error", cv=10,n_jobs=-1)



display_scores(np.sqrt(-scores))
y_pred = cross_val_predict(xgb_model,X,y,cv=10,n_jobs=-1)
sc_y.inverse_transform(y_pred[:10]),sc_y.inverse_transform(y[:10])