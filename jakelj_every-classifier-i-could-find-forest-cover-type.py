# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np 

import pandas as pd 

from sklearn.tree import ExtraTreeClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm.classes import OneClassSVM

from sklearn.neural_network.multilayer_perceptron import MLPClassifier

from sklearn.neighbors.classification import RadiusNeighborsClassifier

from sklearn.neighbors.classification import KNeighborsClassifier

from sklearn.linear_model.stochastic_gradient import SGDClassifier

from sklearn.linear_model.ridge import RidgeClassifierCV

from sklearn.linear_model.ridge import RidgeClassifier

from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    

from sklearn.gaussian_process.gpc import GaussianProcessClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import GaussianNB

from sklearn.semi_supervised import LabelPropagation

from sklearn.semi_supervised import LabelSpreading

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegressionCV

from sklearn.naive_bayes import MultinomialNB  

from sklearn.neighbors import NearestCentroid

from sklearn.svm import NuSVC

from sklearn.linear_model import Perceptron

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.svm import SVC

from sklearn.mixture import GaussianMixture

from sklearn.mixture import BayesianGaussianMixture

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.metrics import make_scorer, accuracy_score

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

#import h2o

#from h2o.automl import H2OAutoML



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings("ignore")







#h2o.init()



# Any results you write to the current directory are saved as output.



train = pd.read_csv('../input/learn-together/train.csv')

test = pd.read_csv('../input/learn-together/test.csv')

ID = test['Id']





train = train.drop(['Soil_Type7','Soil_Type15','Id'], axis =1)

test = test.drop(['Soil_Type7','Soil_Type15','Id'], axis =1)



#train_auto = h2o.H2OFrame(train)

#test_auto = h2o.H2OFrame(test)







#x_auto = list(train.columns)

#y_auto = 'Cover_Type'

#train_auto['Cover_Type'] = train_auto['Cover_Type'].asfactor()







y = train.Cover_Type

X = train.drop(['Cover_Type'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=1)
### This took forever to run you will just have to trust me that the results are correct, or you could uncomment this and run it 

#aml_ti = H2OAutoML(nfolds = 10)

#aml_ti.train(x = x_auto, y = y_auto, training_frame = train_auto)

          

#check the leaderboard

#lb_ti = aml_ti.leaderboard

#lb_ti

'''

-                                                       mpce      logloss   rmse     mse

- StackedEnsemble_AllModels_AutoML_20190912_165033	    0.12123	  0.340173	0.31537	 0.0994582

- StackedEnsemble_BestOfFamily_AutoML_20190912_165033	0.126124  0.347805	0.319726 0.102225

- DRF_1_AutoML_20190912_165033	                        0.13545	  0.418555	0.36234	 0.13129

- XGBoost_1_AutoML_20190912_165033	                    0.13631	  0.363222	0.332949 0.110855

- GBM_1_AutoML_20190912_165033	                        0.140476  0.378649	0.340086 0.115659

- XGBoost_2_AutoML_20190912_165033	                    0.145238  0.391099	0.348184 0.121232

- GBM_2_AutoML_20190912_165033	                        0.168585  0.481562	0.393332 0.15471

- XGBoost_3_AutoML_20190912_165033	                    0.174471  0.453391	0.381124 0.145256

- GBM_3_AutoML_20190912_165033	                        0.219114  1.20778	0.689493 0.475401

- GLM_grid_1_AutoML_20190912_165033_model_1	            0.394709  1.64125	0.801555 0.64249

'''
models = ['ExtraTreeClassifier','DecisionTreeClassifier','MLPClassifier','KNeighborsClassifier',

         'SGDClassifier', 'RidgeClassifier', 'PassiveAggressiveClassifier','AdaBoostClassifier', 'GradientBoostingClassifier', 

          'BaggingClassifier', 'ExtraTreesClassifier', 'RandomForestClassifier', 'BernoulliNB',

          'CalibratedClassifierCV', 'GaussianNB', 'LinearDiscriminantAnalysis', 'LinearSVC', 'LogisticRegression',

          'LogisticRegressionCV', 'NearestCentroid', 'Perceptron', 'QuadraticDiscriminantAnalysis', 'SVC', 'LGBMClassifier', 'XGBClassifier','CatBoostClassifier' 

         ]







# models that didn't work : 'GaussianMixture','BayesianGaussianMixture', neareastcentroid, nuSVC, oneclassSVM, LabelPropagation', 'LabelSpreading'



results_dict = {'auto_ml_stacked': 0.8787}  # I am assuming that mean_per_class_error is similar to accuracy, this may not be the case. I did 1 - mpce

results_dict2 = {}




for model in models:

    

    k_fold = KFold( n_splits=10, shuffle=True, random_state=0)



    if model == 'CatBoostClassifier':

        clf = CatBoostClassifier(logging_level='Silent')

    else:

        clf = eval(model+'()')

        scoring = 'accuracy'



    cv_results = cross_val_score(clf, x_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)

    results_dict[model] = pd.Series(cv_results).mean()

    results_dict2[model] = cv_results

    msg = "%s: %f (%f)" % (model, cv_results.mean(), cv_results.std())

    print(msg)

    

  
import matplotlib.pyplot as plt

from collections import OrderedDict

import seaborn as sns

D1 = dict(OrderedDict(sorted(results_dict.items(), key = lambda t: t[1])))
labels = list(D1.keys())

values = list(D1.values())

plt.subplots(figsize=(30,10))

sns.barplot(x = labels, y =  values)

plt.xticks(rotation= 90,  fontsize=15)

plt.hlines(0.51,0,30.47, linestyles = 'dotted')

plt.ylabel('Accuracy')





df = pd.DataFrame(results_dict2)



df.boxplot(figsize=(30,10), rot=90, fontsize=15)

plt.hlines(0.7,0,30.47, linestyles = 'dotted', color = 'g')

plt.hlines(0.5,0,30.47, linestyles = 'dotted', color = 'r')