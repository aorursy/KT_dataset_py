import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import r2_score, median_absolute_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score, learning_curve, GridSearchCV, train_test_split, learning_curve
from sklearn.feature_selection import RFECV
fdata = pd.read_csv('../input/Financial Distress.csv')
fdata.head(n= 10)
f, ax = plt.subplots(figsize= (12,12))
fdata.groupby(['Company'])['Financial Distress'].mean().plot()
plt.show()
correlation = fdata.drop(labels= ['Time', 'Company'], axis =1).corr()
f, ax = plt.subplots(figsize= (25,25))
sns.heatmap(correlation)
plt.xticks(fontsize = 23)
plt.yticks(fontsize = 23)
plt.show()

correlation['Financial Distress'].plot()
plt.show()
f, ax = plt.subplots(figsize= (25,25))
sns.barplot(x = correlation['Financial Distress'], y = correlation.index)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 23)
plt.ylabel('FEATURES', fontsize= 25)
plt.xlabel('MEAN FINANCIAL DISTRESS', fontsize= 25)
plt.title('FEATURE CORRELATION', fontsize =30)
plt.show()
Y = fdata['Financial Distress']
X = fdata.drop(['Company', 'Time','Financial Distress'], axis = 1)
X.head()
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=20, test_size= 0.25)
svreg = svm.SVR(epsilon = 0.1, C = 80)
svreg.fit(X_train,Y_train)
svreg.score(X_train, Y_train)
cv = ShuffleSplit(n_splits = 3, test_size=0.25, random_state=0)
median_absolute_error(svreg.predict(X_test), Y_test)
mse(svreg.predict(X_test), Y_test)
gbr = GradientBoostingRegressor()
gbr.fit(X_train, Y_train)
selector = RFECV(gbr, cv= cv)
selector.fit(X_train, Y_train)
f, ax = plt.subplots(figsize= (25,25))
sns.barplot(x = (1/selector.ranking_), y = X_train.columns)
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 23)
plt.xlabel('1/rank', fontsize= 25)
plt.ylabel('FEATURES', fontsize= 25)
plt.title('FEATURE RANKING', fontsize =30)
plt.show()
sr = pd.DataFrame(selector.ranking_)
rd = (pd.DataFrame(X_train.columns),sr)
pd.concat(rd, axis = 1 )
my_params= { 
             'learning_rate' : [0.1, 0.01],
             'max_depth' : [6, 4, 5],
            'n_estimators' : [150, 160, 140]
             
}

gbrgs = GridSearchCV(gbr, my_params, cv =cv)
gbrgs.fit(X_train, Y_train)
gbrgs.best_params_
gbrgs.best_score_
gbrgs.best_estimator_.score(X_train, Y_train)
gbr1 = GradientBoostingRegressor(learning_rate = 0.01, max_depth = 7, n_estimators = 210)
gbr1.fit(X_train, Y_train)
gbr1.score(X_train, Y_train)
median_absolute_error(gbr1.predict(X_test),Y_test)
mse(gbr1.predict(X_test),Y_test)
tra_sizes, tra_scores, test_scores = learning_curve(gbr1, X, Y,cv = cv)
tra_smean = np.mean(tra_scores, axis=1)
test_smean = np.mean(test_scores, axis=1)
plt.figure(figsize=(12,10))
plt.title("Learning Curve Of Gradient Boosting -Regressor Model", fontsize = 25)
plt.xlabel("Training examples", fontsize= 25)
plt.ylabel("Score", fontsize = 25)
plt.plot(tra_sizes[1:5], tra_smean[1:5], 'o-', color = 'r', label = "training set")
plt.plot(tra_sizes[1:5], test_smean[1:5], 'o-', color = 'g', label = "cv set")
plt.legend(fontsize = 20)
plt.show()
plt.figure(figsize=(12,10))
plt.title("Prediction Curve", fontsize = 25)
plt.xlabel("Training examples",fontsize = 25)
plt.ylabel("Prediction", fontsize = 25)
plt.plot(X_test[2:20].reset_index().index, Y_test[2:20], 'o-', color = 'r', label = 'real')
plt.plot(X_test[2:20].reset_index().index, gbr1.predict(X_test)[2:20] , 'o-', color = 'g', label = "predicted")
plt.legend(fontsize = 25)
plt.show()


