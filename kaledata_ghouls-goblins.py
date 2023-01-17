# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt  # Matlab-style plotting

# plotly



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head(10)
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(train.shape))

print("The test data size before dropping Id feature is : {} ".format(test.shape))



#Save the 'Id' column

train_ID = train['id']

test_ID = test['id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("id", axis = 1, inplace = True)

test.drop("id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(test.shape))
corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True, cbar=True, annot=True)
missing = train.isnull().sum()
train_dummy = pd.get_dummies(pd.read_csv('../input/train.csv'))

train_dummy.head()
corrs = train_dummy.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

corrs = corrs[corrs['level_0'] != corrs['level_1']]

corrs.tail(20)
ghost_num = {"type":     {"Ghoul": 1, "Goblin": 2, "Ghost": 3} }

train.replace(ghost_num, inplace=True)

train.head()
ntrain = train.shape[0]

ntest = test.shape[0]

y = train.type.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['type'], axis=1, inplace=True)





print("all_data size is : {}".format(all_data.shape))
all_data.drop(['color'], axis=1, inplace=True)

all_data["bone_soul"] = all_data["bone_length"]*all_data["has_soul"]

all_data["hair_soul"] = all_data["hair_length"]*all_data["has_soul"]

all_data["flesh_soul"] = all_data["rotting_flesh"]*all_data["has_soul"]

all_data["bone_hair"] = all_data["bone_length"]*all_data["hair_length"]

all_data["flesh_hair"] = all_data["rotting_flesh"]*all_data["hair_length"]

all_data_simple = pd.DataFrame()

all_data_simple["bone_hair"] = all_data["bone_hair"]

all_data_simple["rotting_flesh"] = all_data["rotting_flesh"]

all_data_simple["bone_soul"] = all_data["bone_soul"]

all_data_simple["hair_soul"] = all_data["hair_soul"]

all_data_simple.head()
train[train['bone_length']>0.8]
#all_data = pd.get_dummies(all_data)

X = all_data_simple[:ntrain]
from sklearn.model_selection import KFold, cross_val_score, train_test_split





train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(train_X, train_y)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    

    score = clf.score(val_X, val_y)

    print("Score: {:.4%}".format(score))

    

    

    

print("="*30)
from sklearn.model_selection import GridSearchCV

from sklearn import metrics



accuracy_scorer = metrics.make_scorer(metrics.accuracy_score)





params = {'n_estimators':[10, 20, 50, 100], 'criterion':['gini', 'entropy'], 'max_depth':[None, 5, 10, 25, 50]}

rf = RandomForestClassifier(random_state = 0)

clf = GridSearchCV(rf, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)

clf.fit(train_X, train_y)

print('Best score: {}'.format(clf.best_score_))

print('Best parameters: {}'.format(clf.best_params_))
rf_best = RandomForestClassifier(criterion= 'entropy', max_depth= 5, n_estimators= 50)
params = {'n_estimators':[10, 25, 50, 100], 'max_samples':[1, 3, 5, 10]}

bag = BaggingClassifier(random_state = 0)

clf = GridSearchCV(bag, param_grid = params, scoring = accuracy_scorer, cv = 5, n_jobs = -1)

clf.fit(train_X, train_y)

print('Best score: {}'.format(clf.best_score_))

print('Best parameters: {}'.format(clf.best_params_))
bag_best = BaggingClassifier(max_samples = 5, n_estimators = 100, random_state = 0)
import shap 

explainer = shap.TreeExplainer(classifiers[1], train_X)

shap_values = explainer.shap_values(train_X)



shap.summary_plot(shap_values, train_X)

# For example for feature 33 low values have a negative impact on model predictions (zero is more likely), 

#and high values have a positive impace (ones are more likely). Feature 217 has an opposite effect: 

#low values have a positive impact and high values have a negative impact.

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], train_X.iloc[:,1:10])

##

top_cols = train_X.columns[np.argsort(shap_values.std(0))[::-1]][:10]

for col in top_cols:

    shap.dependence_plot(col, shap_values, train_X)
from sklearn.ensemble import VotingClassifier

ensemble=VotingClassifier(estimators=[('4', classifiers[4]), ('3', classifiers[3]), ('5', classifiers[5])],

                       voting='soft', weights=[1,1,1]).fit(train_X,train_y)

print('The accuracy for DecisionTree and Random Forest is:',ensemble.score(val_X,val_y))
voting_clf = VotingClassifier(estimators=[('rf', rf_best), ('bag', bag_best)]

                              , voting='hard')

voting_clf.fit(train_X, train_y)

print('The accuracy for DecisionTree and Random Forest is:',voting_clf.score(val_X,val_y))
ghost_cat = {"type":     {1: "Ghoul", 2: "Goblin", 3: "Ghost"} }
test = all_data_simple[ntrain:]
sub = pd.DataFrame()

sub['id'] = test_ID

sub['type'] = voting_clf.predict(test)

sub.replace(ghost_cat, inplace=True)

sub.to_csv('subvoting_clf.csv',index=False)



sub.head()