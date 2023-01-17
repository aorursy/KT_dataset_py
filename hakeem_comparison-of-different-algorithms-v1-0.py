import os

%matplotlib inline

import matplotlib.pyplot as plt  

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import gc

import scipy as sp

import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

#Importing most common alogorithms 

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC#we will not be using SVM due tot he huge training time required on our dataset.

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.multiclass import OneVsRestClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis



from sklearn import model_selection #Cross-validation multiple scoring function

import warnings

warnings.simplefilter("ignore")

from tqdm import tqdm
tqdm.pandas()
%%time

# Only load those columns 

keep_cols = ['event_id', 'game_session', 'installation_id', 'event_count', 'event_code', 'title', 'game_time', 'type', 'world']



train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', usecols=keep_cols)

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv', usecols=keep_cols)

train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')
test_assess = test[test.type == 'Assessment'].copy()

test_labels = submission.copy()

test_labels['title'] = test_labels.installation_id.progress_apply(

    lambda install_id: test_assess[test_assess.installation_id == install_id].iloc[-1].title

)
df = train

group1 = df.drop(columns=['event_id', 'event_code']).groupby(

    ['game_session', 'installation_id', 'title', 'type', 'world']

).max().reset_index()



group2 = pd.get_dummies(

    df[['game_session', 'installation_id', 'event_code']], 

    columns=['event_code']

).groupby(['game_session', 'installation_id']).sum().reset_index()
def group_and_reduce(df):

    """

    Author: https://www.kaggle.com/xhlulu/

    Source: https://www.kaggle.com/xhlulu/ds-bowl-2019-simple-lgbm-using-aggregated-data

    """

    # group1 and group2 are intermediary "game session" groups,

    # which are reduced to one record by game session. group1 takes

    # the max value of game_time (final game time in a session) and 

    # of event_count (total number of events happened in the session).

    # group2 takes the total number of event_code of each type

    group1 = df.drop(columns=['event_id', 'event_code']).groupby(

        ['game_session', 'installation_id', 'title', 'type', 'world']

    ).max().reset_index()



    group2 = pd.get_dummies(

        df[['game_session', 'installation_id', 'event_code']], 

        columns=['event_code']

    ).groupby(['game_session', 'installation_id']).sum().reset_index()



    # group3, group4, group5 are grouped by installation_id 

    # and reduced using summation and other summary stats

    group3 = pd.get_dummies(

        group1.drop(columns=['game_session', 'event_count', 'game_time']),

        columns=['title', 'type', 'world']

    ).groupby(['installation_id']).sum()



    group4 = group1[

        ['installation_id', 'event_count', 'game_time']

    ].groupby(

        ['installation_id']

    ).agg([np.sum, np.mean, np.std, np.min, np.max, np.median, sp.stats.skew])



    group5 = group2.drop(

        columns=['game_session']

    ).groupby(

        ['installation_id']

    ).agg([np.sum, np.mean, np.std, np.min, np.max, np.median])



    return group3.join(group4).join(group5)
%%time

train_small = group_and_reduce(train)

test_small = group_and_reduce(test)



print(train_small.shape)

train_small.head()
final_train = pd.get_dummies(train_labels.set_index('installation_id').drop(

    columns=['num_correct', 'num_incorrect', 'accuracy', 'game_session']

).join(train_small), columns=['title'])



# Experimental:

final_train = final_train.reset_index().groupby('installation_id').apply(lambda x: x.iloc[-1])



print(final_train.shape)

final_train.head()
final_train = final_train.drop(columns='installation_id')
final_test = pd.get_dummies(test_labels.set_index('installation_id').join(test_small), columns=['title'])



print(final_test.shape)

final_test.head()
final_train_ = final_train.dropna()

X = final_train_.drop(columns='accuracy_group').values

y = final_train_['accuracy_group'].values
seed = 42

np.random.seed(seed)

# prepare models

models = []

models.append(('LR', LogisticRegression(multi_class='auto',n_jobs=-1)))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('QDA',QuadraticDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier(7, n_jobs=-1)))

models.append(('CART', DecisionTreeClassifier(max_depth=10)))

models.append(('NB', GaussianNB()))

#models.append(('Gaussian Process Classifier', GaussianProcessClassifier()))

#models.append(('SVM_linear', SVC(kernel="linear", C=0.025)))#we will not be using SVM due to the huge training time required for this dataset

#models.append(('SVM_',SVC(gamma=2, C=1)))

models.append(('RandomForest',RandomForestClassifier( n_estimators=100, n_jobs=-1)))

models.append(('MLP',MLPClassifier(alpha=0.0001)))

models.append(('ADABoost',AdaBoostClassifier()))

models.append(('One-Vs-Rest LR', OneVsRestClassifier(LogisticRegression(multi_class='auto',n_jobs=-1),n_jobs=-1)))

models.append(('One-Vs-Rest LDA', OneVsRestClassifier(OneVsRestClassifier(LinearDiscriminantAnalysis(),n_jobs=-1))))

models.append(('One-Vs-Rest QDA',OneVsRestClassifier(QuadraticDiscriminantAnalysis())))

models.append(('One-Vs-Rest KNN', OneVsRestClassifier(KNeighborsClassifier(7, n_jobs=-1))))

models.append(('One-Vs-Rest CART', OneVsRestClassifier(DecisionTreeClassifier(max_depth=10),n_jobs=-1)))

models.append(('One-Vs-Rest NB', OneVsRestClassifier(GaussianNB(),n_jobs=-1)))

#models.append(('One-Vs-Rest Gaussian Process Classifier', OneVsRestClassifier(GaussianProcessClassifier(),n_jobs=-1)))

#models.append(('One-Vs-Rest SVM_linear', SVC(kernel="linear", C=0.025)))#we will not be using SVM due to the huge training time required for this dataset

#models.append(('One-Vs-Rest SVM_',SVC(gamma=2, C=1)))

models.append(('One-Vs-Rest RandomForest',OneVsRestClassifier(RandomForestClassifier( n_estimators=100, n_jobs=-1),n_jobs=-1)))

models.append(('One-Vs-Rest MLP',OneVsRestClassifier(MLPClassifier(alpha=0.0001),n_jobs=-1)))

models.append(('One-Vs-Rest ADABoost',OneVsRestClassifier(AdaBoostClassifier(),n_jobs=-1)))



# evaluate each model in turn



results = []

scoring = ['accuracy',

          'precision_weighted',

          'recall_weighted',

          'f1_weighted']

names = []



sk = model_selection.StratifiedKFold(n_splits=10, random_state=42)

for name, model in models:

    cv_results = model_selection.cross_validate(model, X, y, cv=sk, scoring=scoring) 

    results.append(cv_results)

    names.append(name)

    msg ='-------------------------------------------------------------------------------------------------------------\n'

    msg = "Model : %s \n" % (name)

    msg = msg +'\n'

    msg =  msg + "Accuracy :  %f (%f)\n" % (cv_results['test_accuracy'].mean(),cv_results['test_accuracy'].std())

    msg =  msg + "Precision score :  %f (%f)\n" % (cv_results['test_precision_weighted'].mean(),cv_results['test_precision_weighted'].std())

    msg =  msg + "Recall score :  %f (%f)\n" % (cv_results['test_recall_weighted'].mean(),cv_results['test_recall_weighted'].std())

    msg =  msg + "F1 score :  %f (%f)\n" % (cv_results['test_f1_weighted'].mean(),cv_results['test_f1_weighted'].std())

    msg = msg + '------------------------------------------------------------------------------------------------------------\n'

    print(msg)
Accuracy = []

Precision = []

Recall = []

F1 = []

for idx,scores in enumerate(results):

    Accuracy.append(scores['test_accuracy'])

    Precision.append(scores['test_precision_weighted'])

    Recall.append(scores['test_recall_weighted'])

    F1.append(scores['test_f1_weighted'])
fig = plt.figure(figsize=(30,30))

fig.suptitle('Algorithms Comparison')

ax = fig.add_subplot(221)

plt.boxplot(Accuracy)

plt.title('Accuracy score')

ax.set_xticklabels(names)

ax = fig.add_subplot(222)

plt.boxplot(Precision)

plt.title('Precision Score')

ax.set_xticklabels(names)

ax = fig.add_subplot(223)

plt.boxplot(Recall)

ax.set_xticklabels(names)

plt.title('Recall score')

ax = fig.add_subplot(224)

plt.title('F1 score')

plt.boxplot(F1)

ax.set_xticklabels(names)



plt.show()
final_model =  DecisionTreeClassifier(max_depth=10)



final_model.fit(X,y)
X_test = final_test.drop(columns=['accuracy_group'])

test_pred = final_model.predict(X_test.fillna(0))



final_test['accuracy_group'] = test_pred

final_test[['accuracy_group']].to_csv('submission.csv')