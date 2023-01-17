# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

import warnings

import xgboost as xgb

import lightgbm as lgb

from scipy.stats import skew

from scipy import stats

from scipy.stats.stats import pearsonr

from scipy.stats import norm

from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve,train_test_split

from sklearn.preprocessing import StandardScaler,MinMaxScaler, Normalizer, RobustScaler,LabelEncoder

from collections import Counter

warnings.filterwarnings('ignore')

sns.set(style='white', context='notebook', palette='deep')
train = pd.read_csv("/kaggle/input/glass-quality-prediction/Train.csv")

test = pd.read_csv("/kaggle/input/glass-quality-prediction/Test.csv")

train.head()

test.head()
train.describe()
print(train.isnull().mean())#checking null value
sns.boxplot(x="class",y="max_luminosity", data=train)#outliers

sns.boxplot(x="class",y="thickness", data=train)#outliers

sns.boxplot(x="class",y="ymin", data=train)#outliers

from scipy import stats#removing using zscore

import numpy as np

z = np.abs(stats.zscore(train))

print(train.shape)

train = train[(z < 3).all(axis=1)]

print(train.shape)
sns.boxplot(x="class",y="max_luminosity", data=train)#outliers

print(train["max_luminosity"].skew())#before

train["max_luminosity"] = train["max_luminosity"].map(lambda i: np.log(i) if i > 0 else 0) 

print(train["max_luminosity"].skew())#after

'''

sns.boxplot(x="class",y="thickness", data=train)#outliers

print(train["thickness"].skew())

train["thickness"] = train["thickness"].map(lambda i: np.log(i) if i > 0 else 0) 

print(train["thickness"].skew())'''
fig, axs = plt.subplots(nrows=1, figsize=(13, 9))

sns.heatmap(train.corr(),

            annot=True, square=True, cmap='YlGnBu', linewidths=2, linecolor='black',

            annot_kws={'size':12})
dataset = pd.concat([train,test],sort=False,ignore_index=True)#combine so easily drops columns

#always drop columns from test and train data both ore it will produce very poor results

dataset.drop(["xmax",'ymax','pixel_area',"x_component_1","x_component_4","x_component_5",

             "grade_A_Component_1"],axis=1,inplace=True)

# i remove few other columns too as their value is never changing alsways the same throughout data



train = dataset[:len(train)]

y_train = train["class"].values

x_train = train.drop("class", axis=1)

test = dataset[len(train):].drop("class",axis=1)
#always apply scalling as i do

#never to scale target data..... or predictions

scaler = MinMaxScaler()

scaler.fit(x_train)# this should remain same for train and test both

scaled_train = scaler.transform(x_train)

scaled_test = scaler.transform(test)

print(scaled_train)
X_train,X_test,y_train,y_test = train_test_split(scaled_train,y_train,random_state=1)

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier,StackingClassifier,RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.decomposition import PCA

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline, make_pipeline
lr = LogisticRegression(C=1)

mlp = MLPClassifier(hidden_layer_sizes=[100, 100],alpha=.1,learning_rate="constant",)

xgb = XGBClassifier(gamma = .0001, learning_rate = .1, max_depth = 3, n_estimators = 100)              

gboost = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 4, n_estimators = 100)

bayes = GaussianNB()

rfc = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,

                       criterion='gini', max_depth=12, max_features='auto',

                       max_leaf_nodes=None, max_samples=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=2,

                       min_weight_fraction_leaf=0.0, n_estimators=100,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)
#use grid search to find best parameters for every model

grid_param = { 'hidden_layer_sizes': [[100,100]], 

              'alpha': [.0001,.001,.01,.1,1] }

grid_search = GridSearchCV(MLPClassifier(), grid_param,cv=5)

grid_search.fit(X_train,y_train)

y_pred = grid_search.predict(X_test)

print(grid_search.best_params_)#it will show the best parameters

#u can also use pipeline and robust feature on classifier for more functionality
clf_isotonic = CalibratedClassifierCV(rfc, cv=2, method='isotonic')

clf_isotonic.fit(X_train,y_train)

y_rfc = clf_isotonic.predict_proba(X_test)

print(clf_isotonic.score(X_train, y_train))

def get_models():

	models = dict()

	models['lr'] = lr

	models['mlp'] = mlp

	models['xgb'] = xgb

	models['gboost'] = gboost

	models['bayes'] = bayes

	return models





# get a stacking ensemble of models

def get_stacking():

	# define the base models

	level0 = list()

	level0.append(('lr', lr))

	level0.append(('mlp', mlp))

	level0.append(('xgb', xgb))

	level0.append(('gboost', gboost))

	level0.append(('bayes', bayes))

	# define meta learner model

	level1 = LogisticRegression()

	# define the stacking ensemble

	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

	return model



 

# evaluate a given model using cross-validation

def evaluate_model(model):

	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

	scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

	return scores

 

# get the models to evaluate

models = get_models()

models['staking'] = get_stacking()

# evaluate the models and store results

results, names = list(), list()

for name, model in models.items():

	scores = evaluate_model(model)

	results.append(scores)

	names.append(name)

# plot model performance for comparison

plt.boxplot(results, labels=names, showmeans=True)

plt.show()
def get_stacking():

	# define the base models

	level0 = list()

	level0.append(('lr', lr))

	level0.append(('mlp', mlp))

	level0.append(('xgb', xgb))

	level0.append(('gboost', gboost))

	level0.append(('rfc', rfc))

    

	# define meta learner model

	level1  = LogisticRegression()

	# define the stacking ensemble

	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

	return model

model = get_stacking()

model.fit(X_train,y_train)

y_pred = model.predict_proba(X_test)

print(model.score(X_train, y_train))

print(model.score(X_test, y_test))



output = pd.DataFrame({1:y_pred[:,0], 2:y_pred[:,1]})

output.to_excel("submitted.xlsx",index=False)