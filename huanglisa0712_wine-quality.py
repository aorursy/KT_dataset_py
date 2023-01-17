# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

# for correlation matrix

from pandas.plotting import scatter_matrix

from matplotlib import cm 







from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler



from sklearn.ensemble import RandomForestClassifier







#k-fold validation



from sklearn.model_selection import GridSearchCV





from sklearn.metrics import confusion_matrix,classification_report, auc,roc_curve, r2_score

from collections import OrderedDict 

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
"""

<a id='data-at-a-glance'></a>

# Data at a glance

In the following table, we give a peek at the data to have a general understanding of the data. 

* All of the features are numeric and the dependent variable( wine quality ) is 

"""
"""

<a id='XGBoost-review' ></a>

# A Brief Review on XGBoost

XGBoost stands for eXtreme Gradient Boosting and was first proposed by [Tianqi Chen](https://arxiv.org/abs/1603.02754). 

XGBoost is a model, dealing with supervised learning problems, where we predict the target variable with features. 



### Boosting Algorithm 

Ensemble learning (or boosting) technique combines weak (base) learners into a strong learners. Though the base learners tends to have limited power, we can use ensemble learning to form more accurate model. *We can consider XGBoost is a special case of Newton Boosting. We can considered it as the interpretation of Newton method is functional space. It is numerical optimization algorithms in function space. *



Given a finite training set $\{ x_i, y_i \},i=1...n$, we hope to solve the objective function:

$$\hat{\Phi}(F)=\sum^{n}_{i=1}L(y_i,F(x_i))$$



### Numerical Optimization in Function Space



Assume the true risk R(f) is known to us, and we iterative risk minimization procedures in function space. We are minimizing

$$R(f(x))=E[L(Y,f(x))|X=x] , \forall x \in \mathcal{X}$$ 

at each iteration 



#### Construct $F(x)$

Practically, We view boosting algorithms as numerical optimization in function space. Boosting algorithms implements a sequantial process, generates weak learners and combines them into strong learners. We can consider the weak learners as a set of basis functions. Boosting algorithm sequentially add base models to improves the fitting. 



$$f^{(m)}(x)=f^{(m-1)}(x)+f_m(x)$$

$f^{(m-1)}$ is current estimate. We take the "step" $f_m$ in function space, and get $f^{(m)}$. Therefore, we can view f(x) as the combination of initial guess and all successive "steps" taken previously in function space. 



$$F(x)\equiv f^{(M)}(x)\equiv \sum_{m=0}^{M}f_m(x)$$



, where $f(x)$ is the prediction, M is the number of weak learners and $f_0(x)$



forward stagewise additive modeling (FSAM)$\hat{f}(x)=\hat{f}^{(M)}(x)=\sum_{m=1}^{M}\hat{\theta}_m \hat{c}_m(x_i)$







### System Features:

* Parallelization : The parallelisatio happens during the tree construction and enables distributed training and predicting across nodes. 

* Cache Optimization : XGBoost keeps all of the immediate calculation 

* Distributed Computing

* Out-of-Core Computing



### Model Features

* Gradient Boosting

* Stochastic Gradient Boosting

* Regularizaed Gradient Boosting



### Algorithm Features

* Sparse Aware

* Block Structure

* Continued Training



### CART algorithm 

"""
df = pd.read_csv(os.path.join(dirname,filename))

df.head()
df.isna().sum()
df.describe()


fig = plt.figure()

key = df.keys()

f, axes = plt.subplots(3,4,figsize = (20,25)) # rows, columns

for j in range(4): 

    for i in range(3):

        sns.distplot(df[key[4*i+j]],ax = axes[i,j]) 



plt.show()

df['quality'] = df['quality'].values.astype(np.int)

df['good_bad_wine'] = df['quality']>=6.5

df['good_bad_wine'] = df['good_bad_wine'].values.astype(np.int)

df.head()

import plotly.express as px

def hide_current_axis(*args, **kwds):

    plt.gca().set_visible(False)

    



    

plt.figure(figsize = (60,60))

cmap = cm.get_cmap('gnuplot')

plt.figure(figsize = (10,10))

grid = sns.PairGrid(df,hue = 'good_bad_wine')

grid = grid.map_lower(sns.scatterplot, alpha=0.3, edgecolor='none')



grid.map_diag(plt.hist)

grid.map_upper(hide_current_axis)



plt.figure(figsize = (10,10))

sns.heatmap(df.corr(),annot = True,cmap='PuBuGn')

plt.show()
x = df[df.columns[0:11]].values

y = df['good_bad_wine'].values.astype(np.int)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

print('X train size: ', x_train.shape)

print('X test size: ', x_test.shape)

print('y train size: ', y_train.shape)

print('y test size: ', y_test.shape)
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA



my_pipeline = Pipeline([('pca', PCA()),  ('xgbrg', XGBClassifier(random_state = 2))])

tuned_parameters = {'pca__n_components': [5, 10,None],'xgbrg__gamma':[0,0.5,1],'xgbrg__reg_alpha':[0,0.5,1], 'xgbrg__reg_lambda':[0,0.5,1],"xgbrg__learning_rate": [0.1, 0.5, 1]}



xgb = GridSearchCV(my_pipeline, cv=5,param_grid = tuned_parameters,  scoring='roc_auc')

xgb.fit(x_train, y_train)

print('The best model is: ', xgb.best_params_)

prediction = xgb.predict_proba(x_test)[:,1]

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,prediction)



plt.subplots(figsize=(6,6))

plt.plot(false_positive_rate, true_positive_rate)

plt.xlim(0,1)

plt.ylim(0,1)

plt.xlabel('False positive rate', fontsize = 14)

plt.ylabel('True positive rate', fontsize = 14)

plt.show()
print("AUC is: ", auc(false_positive_rate, true_positive_rate))
from xgboost import XGBRegressor

x = df[df.columns[0:11]].values

y = df['quality'].values.astype(np.int)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)

print('X train size: ', x_train.shape)

print('X test size: ', x_test.shape)

print('y train size: ', y_train.shape)

print('y test size: ', y_test.shape)

reg = XGBRegressor(random_state = 2)

reg.fit(x_train, y_train)
tuned_parameters = {'gamma':[0,0.5,1], 'reg_lambda': [1,5,10], 'reg_alpha':[0,1,5], 'subsample': [0.6,0.8,1.0]}



reg = GridSearchCV(XGBRegressor(random_state = 2, n_estimators=500), tuned_parameters, cv=5, scoring='r2')

reg.fit(x_train, y_train)
print('The best model is: ', reg.best_params_)
"""

prediction = reg.predict(x_test)

plt.plot(y_test, prediction, linestyle='', marker='o')

plt.xlabel('true values', fontsize = 16)

plt.ylabel('predicted values', fontsize = 16)

plt.show()



"""
prediction = reg.predict(x_test)

print('The r2_score on the test set is: ',r2_score(y_test, prediction))
x = df[df.columns[0:11]].values

y = df['good_bad_wine'].values.astype(np.int)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=42)



"""

tuned_parameters = {'n_estimators':[500], 'max_depth': [2,3,5,7], 'max_features': [0.5,0.7,0.9],'n_jobs':[-1],'min_samples_leaf':[1,5,10]} 

#,'random_state':[14]

clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='roc_auc')

clf.fit(X_train, y_train)

"""
"""

# Generate the "OOB error rate" vs. "n_estimators" plot.



for label, clf_err in error_rate.items():

    xs, ys = zip(*clf_err)

    plt.plot(xs, ys, label=label)



plt.xlim(min_estimators, max_estimators)

plt.xlabel("n_estimators")

plt.ylabel("OOB error rate")

plt.legend(loc="upper right")

plt.show()

"""
rf = RandomForestClassifier(n_estimators = 100,oob_score = True)



max_features = ['sqrt','log2',None]

n_estimators = [int(x) for x in np.linspace(start = 100,stop = 250,num = 16 )]



param_grid = {'max_features':max_features,'n_estimators' : n_estimators }

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(x_train,y_train)


