import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



seed = 7



%matplotlib inline



dataset = pd.read_csv("../input/creditcardfraud/creditcard.csv")



print('This dataset contains ',dataset.shape[0],'rows')

print('This dataset contains ',dataset.shape[1],'columns')
dataset.head()
dataset.info()
# Check if NA values are present

dataset.isnull().sum().sum()
# Change the type of the Class column

dataset.Class = dataset.Class.astype('bool')



# Get the count of each Class

dataset.groupby('Class').size()
dataset.describe()
import seaborn as sns



frauds = dataset[dataset.Class==True]

genuines = dataset[dataset.Class==False]
# 'Time' visualization

sns.distplot(dataset.Time,

             bins=80, color = 'red', 

             hist_kws={'edgecolor':'green'},

             kde_kws={'linewidth': 3})

plt.title('Density plot and Histogram of Time (in seconds)')

plt.show()
# 'Time' visualization for frauds

sns.distplot(frauds.Time,

             bins=80, color = 'darkred', 

             hist_kws={'edgecolor':'yellow'},

             kde_kws={'linewidth': 2})

plt.title('Density plot and Histogram of Time for Frauds (in seconds)')

plt.show()
# 'Amount' visualization

# According to the Introduction, we assume the currency is Euro

sns.distplot(dataset.Amount, 

             bins=80, color = 'maroon', 

             hist_kws={'edgecolor':'red'},

             kde_kws={'linewidth': 2})

plt.title('Density plot and Histogram of Amount (in €)')

plt.show()
# Ratio of Frauds vs. Amount

amounts = np.linspace(0,5000,1001)

ratios = np.array([])



for amount in amounts:

    

    nbGenuine = len(genuines[genuines.Amount > amount])

    nbFrauds = len(frauds[frauds.Amount > amount])

    ratio = 100*nbFrauds/nbGenuine

    

    ratios = np.append(ratios,ratio)



plt.plot(amounts,ratios,'r-')

plt.title('Ratio #Fraud/#Genuine vs. Amount')

plt.xlabel('Amount (in €)')

plt.ylabel('Ratio (in %)')

plt.show()
from pandas.plotting import scatter_matrix



fig = plt.figure(figsize=(7,7))

ax = fig.add_subplot(111)

cax = ax.matshow(dataset.corr(), vmin=-2, vmax=2, interpolation='none')

fig.colorbar(cax)

plt.show()
# We randomly select 492 genuine transactions

genuines_sub = genuines.sample(492, random_state=seed)



# dataset_sub is the dataset composed of 492 frauds and of 492 genuine transactions

dataset_sub = frauds.append(genuines_sub, ignore_index=True)



# We drop the 'Time' column

dataset_sub = dataset_sub.drop('Time',axis=1)



print('This sub dataset contains ',dataset_sub.shape[0],'rows')

print('This sub dataset contains ',dataset_sub.shape[1],'columns')
from sklearn.model_selection import train_test_split



# Predictors

X = dataset_sub.drop('Class',axis=1)



# Response

y = dataset_sub.Class



# Split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = seed)



# Proportion of fraud in train set and test set

print('Proportion of fraud in train:',y_train[y_train == True].shape[0]/X_train.shape[0])

print('Proportion of fraud in test:',y_test[y_test == True].shape[0]/X_test.shape[0])
from sklearn.metrics import recall_score, precision_recall_curve, average_precision_score, confusion_matrix, precision_score



scoring = 'average_precision'
from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

import xgboost



models = []

models.append(('LR',LogisticRegression(random_state=seed)))

models.append(('LDA',LinearDiscriminantAnalysis()))

models.append(('QDA',QuadraticDiscriminantAnalysis()))

models.append(('SVM',SVC(random_state=seed,gamma='scale')))



ensembles = []

ensembles.append(('RF', RandomForestClassifier(random_state=seed,n_estimators=100)))

ensembles.append(('ADA', AdaBoostClassifier(random_state=seed)))

ensembles.append(('GBM', GradientBoostingClassifier(random_state=seed)))

ensembles.append(('XGB', XGBClassifier(random_state=seed)))
# Models evaluation function

def get_score_models(model,X_train,X_test,y_train,y_test):

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    

    # All our models implement the 'decision_function' method

    # It is not the case of all our ensembles

    y_score = model.decision_function(X_test)

    

    compare(y_test,y_pred,y_score)

    

# Ensembles evaluation function

def get_score_ensembles(ensemble,X_train,X_test,y_train,y_test):

    ensemble.fit(X_train,y_train)

    y_pred = ensemble.predict(X_test)

    

    # All our ensembles implement the 'predict_proba' method

    # It is not the case of all our models

    y_score = ensemble.predict_proba(X_test)[:,1]

    

    compare(y_test,y_pred,y_score)



# Print metrics and graph function

def compare(y_test,y_pred,y_score):

    print('Confusion matrix:')

    print(confusion_matrix(y_test,y_pred))

    

    print('Recall:',recall_score(y_test,y_pred))

    print('Precision:',precision_score(y_test,y_pred))

    print('Area under the curve:',average_precision_score(y_test,y_score))

    

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, alpha=0.4, color='b', where='post')

    plt.fill_between(recall, precision, alpha=0.2, color='b', step='post')

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.ylim([0, 1.05])

    plt.xlim([0, 1])

    plt.title('Precision-Recall curve')

    plt.show()
# Evaluation of each model

for name,model in models:

    print('----------',name,'----------')

    get_score_models(model,X_train,X_test,y_train,y_test)
# Evaluation of each ensemble method

for name,ensemble in ensembles:

    print('----------',name,'----------')

    get_score_ensembles(ensemble,X_train,X_test,y_train,y_test)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



num_folds = 10

kfold = KFold(n_splits=num_folds,random_state=seed)



names = []

results_recall = []

results_aupcr = []

models_score = {}

ensembles_score = {}



# Function cross validating and printing Recall and AUPRC results

def cross_validation(name,classifier,classifiers_score,results_recall,results_aupcr):

    cv_results_recall = cross_val_score(model,X_train,y_train,cv=kfold,scoring='recall')

    cv_results_auprc = cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)



    models_score[name] = [cv_results_recall.mean()]

    models_score[name].append(cv_results_recall.std())

    models_score[name].append(cv_results_auprc.mean())

    models_score[name].append(cv_results_auprc.std())

    

    results_recall.append(cv_results_recall)

    results_aupcr.append(cv_results_auprc)

    names.append(name)



    print('----------',name,'----------')

    print('Recall:',models_score[name][0],'(',models_score[name][1],')')

    print('AUPRC:',models_score[name][2],'(',models_score[name][3],')\n')
# 10-Fold cross validation on our models

for name,model in models:

    cross_validation(name,model,models_score,results_recall,results_aupcr)
# Compare Classifiers regarding Recall

fig = plt.figure()

fig.suptitle('Classifiers Recall Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results_recall)

ax.set_xticklabels(names)

plt.show()
from sklearn.model_selection import GridSearchCV



# Function executing the Grid Search and printing the result

def search_param(model,X_train,y_train,param_grid,scoring,kfold):

    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

    grid_result = grid.fit(X_train, y_train)

    

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']

    stds = grid_result.cv_results_['std_test_score']

    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):

        print("%f (%f) with: %r" % (mean, stdev, param))
# /!\ May take some time

# SVC Parameters values that will be tested:

C = [0.001,0.01,0.1,1]

kernel_values = ['rbf', 'sigmoid', 'linear']



param_grid = dict(C=C,kernel=kernel_values)



model = SVC(random_state=seed,gamma='scale')



search_param(model,X_train,y_train,param_grid,scoring,kfold)
# kernel = 'linear'

# C contained in [0.001;0.01]

C = np.linspace(0.001,0.01,10)

param_grid = dict(C=C)



model = SVC(kernel='linear',random_state=seed)



search_param(model,X_train,y_train,param_grid,scoring,kfold)
# Check the Recall for the tuned SVM

model = SVC(kernel='linear',C=0.002,random_state=seed)

cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring='recall')



print('Recall:',cv_results.mean(),'(',cv_results.std(),')')
# /!\ Take some time

# XGBoost Parameters values that will be tested:

learning_rate = [0.01,0.1,1]

n_estimators = [10,100,1000]

max_depth = np.linspace(2,5,4).astype('int')



param_grid = dict(learning_rate=learning_rate,n_estimators=n_estimators,max_depth=max_depth)



model = XGBClassifier(random_state=seed)



search_param(model,X_train,y_train,param_grid,scoring,kfold)
# /!\ Take some time

# max_depth = 2



learning_rate = np.linspace(0.1,1,10)

n_estimators = np.linspace(10,100,10).astype('int')



param_grid = dict(learning_rate=learning_rate,n_estimators=n_estimators)



model = XGBClassifier(max_depth=2,random_state=seed)



search_param(model,X_train,y_train,param_grid,scoring,kfold)
# Check the Recall for the tuned XGB

model = XGBClassifier(max_depth=2, learning_rate=0.2,n_estimators=60,random_state=seed)

cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring='recall')



print('Recall:',cv_results.mean(),'(',cv_results.std(),')')