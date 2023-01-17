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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows',500)
cc=pd.read_csv('/kaggle/input/concrete-compressive-strength-data-set/compresive_strength_concrete.csv')

cc.head()
cc.shape
cc.info()
for i in cc.columns:

    sns.distplot(cc[i])

    plt.show()
cc.describe()
cc.isnull().sum()
q1=cc.quantile(0.25)

q3=cc.quantile(0.75)

IQR=q3-q1

cwo=((cc.iloc[:] <(q1-1.5*IQR))|(cc.iloc[:]>(q3+1.5*IQR))).sum(axis=0)

opdf=pd.DataFrame(cwo,index=cc.columns,columns=['No. of Outliers'])

opdf['Percentage Outliers']=round(opdf['No. of Outliers']*100/len(cc),2)

opdf
rwo=(((cc[:]<(q1-1.5*IQR))|(cc[:]>(q3+1.5*IQR))).sum(axis=1))

ro005=(((rwo/len(cc.columns))<0.05).sum())*100/len(cc)

ro01=(((rwo/len(cc.columns))<0.1).sum())*100/len(cc)

ro015=(((rwo/len(cc.columns))<0.15).sum())*100/len(cc)

ro02=(((rwo/len(cc.columns))<0.2).sum())*100/len(cc)

ro025=(((rwo/len(cc.columns))<0.25).sum())*100/len(cc)

ro03=(((rwo/len(cc.columns))<0.30).sum())*100/len(cc)

ro035=(((rwo/len(cc.columns))<=0.35).sum())*100/len(cc)

ro04=(((rwo/len(cc.columns))<=0.4).sum())*100/len(cc)

ro045=(((rwo/len(cc.columns))<=0.45).sum())*100/len(cc)

ro05=(((rwo/len(cc.columns))<=0.50).sum())*100/len(cc)

ro055=(((rwo/len(cc.columns))<0.55).sum())*100/len(cc)

ro06=(((rwo/len(cc.columns))<0.6+0).sum())*100/len(cc)

ro=pd.DataFrame(np.round([ro005,ro01,ro015,ro02,ro025,ro03,ro035,ro04,ro045,ro05,ro055,ro06],2),

             index=['5%','10%','15%','20%','25%','30%','35%','40%','45%','50%','55%','60%'],

            columns=['% Data'])

ro.index.name='% Outlier'

ro
from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
imp = IterativeImputer()

imp.fit(cc)

cc=pd.DataFrame(imp.transform(cc),columns=cc.columns)
cc.isnull().sum()
g=cc.groupby('Age (day)')

g1=g.get_group(1)

g3=g.get_group(3)

g7=g.get_group(7)

g14=g.get_group(14)

g28=g.get_group(28)

pd.DataFrame(round(g28.iloc[:,-1].sort_values()).unique(),columns=['Comp Strength @ 28 days'])
cp = cc.corr()

mask = np.zeros_like(cp)

mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(8,8))

with sns.axes_style("white"):

    sns.heatmap(cp,annot=True,linewidth=2,mask = mask,cmap="coolwarm")

plt.title("Correlation Plot")

plt.show()
import statsmodels.api as sm

X=cc.iloc[:,:8]

Y=cc.iloc[:,8]
ls=sm.OLS(Y,sm.add_constant(X))

results=ls.fit()

results.summary()
ls=sm.OLS(Y,X)

results=ls.fit()

results.summary()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, Y, random_state=150, test_size=0.3 )
lr=LinearRegression()

lr.fit(X_train,y_train)

print('Score: ',lr.score(X_train,y_train))

y_pred_lrtr=lr.predict(X_train)

y_pred_lrte=lr.predict(X_test)

from sklearn.metrics import r2_score

print('Train R2 score: ',r2_score(y_train,y_pred_lrtr))

print('Test R2 score: ',r2_score(y_test,y_pred_lrte))
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree = 2)

X_polytr = pf.fit_transform(X_train)

lr.fit(X_polytr,y_train)

y_pred_lr2tr = lr.predict(X_polytr)

print("Training R2 - degree 2 polynomial: ",r2_score(y_train, y_pred_lr2tr ))

X_polyte = pf.fit_transform(X_test)

y_pred_lr2te= lr.predict(X_polyte)

print("Test R2 - degree 2 polynomial: ",r2_score(y_test,y_pred_lr2te))
pf = PolynomialFeatures(degree = 3)

X_polytr = pf.fit_transform(X_train)

lr.fit(X_polytr,y_train)

y_pred_lr2tr = lr.predict(X_polytr)

print("Training R2 - degree 2 polynomial: ",r2_score(y_train, y_pred_lr2tr ))

X_polyte = pf.fit_transform(X_test)

y_pred_lr2te= lr.predict(X_polyte)

print("Test R2 - degree 2 polynomial: ",r2_score(y_test,y_pred_lr2te))
pf = PolynomialFeatures(degree = 4)

X_polytr = pf.fit_transform(X_train)

lr.fit(X_polytr,y_train)

y_pred_lr2tr = lr.predict(X_polytr)

print("Training R2 - degree 2 polynomial: ",r2_score(y_train, y_pred_lr2tr ))

X_polyte = pf.fit_transform(X_test)

y_pred_lr2te= lr.predict(X_polyte)

print("Test R2 - degree 2 polynomial: ",r2_score(y_test,y_pred_lr2te))
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()

dt.fit(X_train,y_train)

dt.score(X_train,y_train)

y_pred_dttr=dt.predict(X_train)

y_pred_dtte=dt.predict(X_test)

print('Train R2 score: ',r2_score(y_train,y_pred_dttr))

print('Test R2 score: ',r2_score(y_test,y_pred_dtte))
from sklearn.model_selection import GridSearchCV

param_grid = {'max_depth': np.arange(3, 8),

             'criterion' : ['mse','mae'],

             'max_leaf_nodes': [5,10,20,100],

             'min_samples_split': [2, 5, 10, 20]}



grid_tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv = 5, scoring= 'r2')

grid_tree.fit(X_train, y_train)

print(grid_tree.best_estimator_)

print(np.abs(grid_tree.best_score_))
dtpr=DecisionTreeRegressor(criterion='mse', max_depth=7, max_features=None,

                      max_leaf_nodes=100, min_impurity_decrease=0.0,

                      min_impurity_split=None, min_samples_leaf=1,

                      min_samples_split=10, min_weight_fraction_leaf=0.0,

                      presort=False, random_state=None, splitter='best')

dtpr.fit(X_train,y_train)

dtpr.score(X_train,y_train)

y_pred_dtprtr=dtpr.predict(X_train)

y_pred_dtprte=dtpr.predict(X_test)

print('Train R2 score: ',r2_score(y_train,y_pred_dtprtr))

print('Test R2 score: ',r2_score(y_test,y_pred_dtprte))
param_grid = {'max_depth': np.arange(3, 6),

             'criterion' : ['mse','mae'],

             'max_leaf_nodes': [100,105, 90,95],

             'min_samples_split': [6,7,8,9,10],

             'max_features':[2,3,4,5,6]}



grid_tree = GridSearchCV(DecisionTreeRegressor(), param_grid, cv = 5, scoring= 'r2')

grid_tree.fit(X_train, y_train)

print(grid_tree.best_estimator_)

print(np.abs(grid_tree.best_score_))
dtpr=DecisionTreeRegressor(criterion='mae', max_depth=5, max_features=6,

                      max_leaf_nodes=95, min_impurity_decrease=0.0,

                      min_impurity_split=None, min_samples_leaf=1,

                      min_samples_split=8, min_weight_fraction_leaf=0.0,

                      presort=False, random_state=None, splitter='best')

dtpr.fit(X_train,y_train)

dtpr.score(X_train,y_train)

y_pred_dtprtr=dtpr.predict(X_train)

y_pred_dtprte=dtpr.predict(X_test)

print('Train R2 score: ',r2_score(y_train,y_pred_dtprtr))

print('Test R2 score: ',r2_score(y_test,y_pred_dtprte))
from sklearn.ensemble import AdaBoostRegressor

abr = AdaBoostRegressor(random_state=0, n_estimators=100)

abr.fit(X_train, y_train)

abr.feature_importances_  

abr.fit(X_train,y_train)

abr.score(X_train,y_train)

y_pred_abrtr=abr.predict(X_train)

y_pred_abrte=abr.predict(X_test)

print('Train R2 score: ',r2_score(y_train,y_pred_abrtr))

print('Test R2 score: ',r2_score(y_test,y_pred_abrte))
from sklearn.ensemble import RandomForestRegressor

rfr=RandomForestRegressor()

rfr.fit(X_train,y_train)

rfr.score(X_train,y_train)

y_pred_rfrtr=rfr.predict(X_train)

y_pred_rfrte=rfr.predict(X_test)

print('Train R2 score: ',r2_score(y_train,y_pred_rfrtr))

print('Test R2 score: ',r2_score(y_test,y_pred_rfrte))
param_grid = {'max_depth': np.arange(3, 8),

             'criterion' : ['mse','mae'],

             'max_leaf_nodes': [100,105, 90,95],

             'min_samples_split': [6,7,8,9,10],

             'max_features':['auto','sqrt','log2']}



grid_tree = GridSearchCV(RandomForestRegressor(), param_grid, cv = 5, scoring= 'r2')

grid_tree.fit(X_train, y_train)

print(grid_tree.best_estimator_)

print(np.abs(grid_tree.best_score_))
rfr=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=7,

                      max_features='auto', max_leaf_nodes=90,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=2, min_samples_split=7,

                      min_weight_fraction_leaf=0.0, n_estimators=100,

                      n_jobs=None, oob_score=False, random_state=None,

                      verbose=0, warm_start=False)

rfr.fit(X_train,y_train)

rfr.score(X_train,y_train)

y_pred_rfrtr=rfr.predict(X_train)

y_pred_rfrte=rfr.predict(X_test)

print('Train R2 score: ',r2_score(y_train,y_pred_rfrtr))

print('Test R2 score: ',r2_score(y_test,y_pred_rfrte))
from sklearn.ensemble import GradientBoostingRegressor

gb=GradientBoostingRegressor()

gb.fit(X_train,y_train)

gb.score(X_train,y_train)

y_pred_gbtr=gb.predict(X_train)

y_pred_gbte=gb.predict(X_test)

print('Train R2 score: ',r2_score(y_train,y_pred_gbtr))

print('Test R2 score: ',r2_score(y_test,y_pred_gbte))
param_grid = {'n_estimators': [230],

              'max_depth': range(10,31,2), 

              'min_samples_split': range(50,501,10), 

              'learning_rate':[0.2]}

clf = GridSearchCV(GradientBoostingRegressor(random_state=1), 

                   param_grid = param_grid, scoring='r2', 

                   cv=5).fit(X_train, y_train)

print(clf.best_estimator_) 

print("R Squared:",clf.best_score_)
gb=GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,

                          learning_rate=0.2, loss='ls', max_depth=14,

                          max_features=None, max_leaf_nodes=None,

                          min_impurity_decrease=0.0, min_impurity_split=None,

                          min_samples_leaf=1, min_samples_split=150,

                          min_weight_fraction_leaf=0.0, n_estimators=230,

                          n_iter_no_change=None, presort='auto', random_state=1,

                          subsample=1.0, tol=0.0001, validation_fraction=0.1,

                          verbose=0, warm_start=False)

gb.fit(X_train,y_train)

gb.score(X_train,y_train)

y_pred_gbtr=gb.predict(X_train)

y_pred_gbte=gb.predict(X_test)

print('Train R2 score: ',r2_score(y_train,y_pred_gbtr))

print('Test R2 score: ',r2_score(y_test,y_pred_gbte))
from xgboost import XGBRegressor



xgb=XGBRegressor()

xgb.fit(X_train,y_train)

print('Model Score: ', xgb.score(X_train,y_train))

y_pred_xgbtr=xgb.predict(X_train)

y_pred_xgbte=xgb.predict(X_test)

print('Train R2-Score: ', r2_score(y_train,y_pred_xgbtr))

print('Test R2-Score: ', r2_score(y_test,y_pred_xgbte))
xgb=XGBRegressor(base_score=0.7, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=0.65, colsample_bytree=1, gamma=0.3,

             importance_type='weight', learning_rate=0.2, max_delta_step=150,

             max_depth=4, min_child_weight=0.5, missing=None, n_estimators=200,

             n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,

             reg_alpha=0.001, reg_lambda=1, scale_pos_weight=1, seed=None,

             silent=None, subsample=1, verbosity=1)

xgb.fit(X_train,y_train)

print('Model Score: ', xgb.score(X_train,y_train))

y_pred_xgbtr=xgb.predict(X_train)

y_pred_xgbte=xgb.predict(X_test)

print('Train R2-Score: ', r2_score(y_train,y_pred_xgbtr))

print('Test R2-Score: ', r2_score(y_test,y_pred_xgbte))
import shap
explainer = shap.TreeExplainer(xgb)

shap_values = explainer.shap_values(X_train)
for i in X_train.columns:

    shap.dependence_plot(i,shap_values, X_train)
shap.summary_plot(shap_values, X_train)