import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from collections import defaultdict, Counter

from matplotlib.pyplot import plot

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#names of 21 columns

cols = ['existingchecking', 'duration', 'credithistory', 'purpose', 'creditamount', 

         'savings', 'employmentsince', 'installmentrate', 'statussex', 'otherdebtors', 

         'residencesince', 'property', 'age', 'otherinstallmentplans', 'housing', 

         'existingcredits', 'job', 'peopleliable', 'telephone', 'foreignworker', 'classification']



#load the dataset

data = pd.read_csv('/kaggle/input/germancreditdata/german.data', names = cols, delimiter=' ')



# preprocess numerical features

num_features = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age', 

           'existingcredits', 'peopleliable']



# standardization

data[num_features] = StandardScaler().fit_transform(data[num_features])



#preprocess categorical features

cat_features = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',

           'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job', 

           'telephone', 'foreignworker']



# one-hot encoding each of every categorical features

data = pd.get_dummies(data, columns = cat_features)



# features and target set

x = data.drop('classification', axis = 1)

# replace targets with 1=good, 0=bad

data.classification.replace([1,2], [1,0], inplace=True)

y = data.classification
# split features and target

x = data.drop('classification', axis = 1)

y = data.classification

print('x.shape:', x.shape, '\ny.shape:', y.shape)
# Create correlation matrix

f = plt.figure(figsize=(20, 20))

plt.matshow(x.corr(), fignum=f.number)

plt.xticks(range(x.shape[1]), fontsize=14, rotation=45)

plt.yticks(range(x.shape[1]), fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
# Principal component analysis

cov_mat = np.cov(x.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

tot = sum(eig_vals)

var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

print("1. Variance Explained\n", var_exp)

cum_var_exp = np.cumsum(var_exp)

print("\n\n2. Cumulative Variance Explained by the first 50 PC\n", cum_var_exp[0:51])

print("\n\n3. Percentage of Variance Explained by the first 46 PC together sums up to:", cum_var_exp[46])
# Dimensional reduction from 61 to 46

pca = PCA(n_components=46)

principalComponents = pca.fit_transform(x)

x_pca = pd.DataFrame(data = principalComponents)

x_pca.shape
# split train, test set

xtrain, xtest, ytrain, ytest = train_test_split(x_pca, y,test_size = 0.2)
#Since its a classification problem, its important to know if data is balanced or not

print(ytrain.value_counts())

ytrain.value_counts().plot.bar()

plt.show()
# Apply resampling

sm = SMOTE()

xtrain_res, ytrain_res = sm.fit_sample(xtrain, ytrain)

# Print number of 'good' credits and 'bad credits, should be fairly balanced now

print("Before SMOTE")

unique, counts = np.unique(ytrain, return_counts=True)

print(dict(zip(unique, counts)))

print("After SMOTE")

unique, counts = np.unique(ytrain_res, return_counts=True)

print(dict(zip(unique, counts)))
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from sklearn.pipeline import make_pipeline

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
Classifiers=[LogisticRegression(),

             DecisionTreeClassifier(),

             RandomForestClassifier(),

             GradientBoostingClassifier(),

             AdaBoostClassifier(),

             ExtraTreesClassifier(),

             KNeighborsClassifier(),

             SVC(),

             GaussianNB()]



pipelines = []

for classifier in Classifiers:

    pipeline = make_pipeline(classifier)

    pipelines.append(pipeline)



cv_acc = []

model_names = ['LogisticRegression','DecisionTreeClassifier','RandomForestClassifier','GradientBoostingClassifier','AdaBoostClassifier','ExtraTreesClassifier','KNeighborsClassifier','SVC','GaussianNB']

for name, pipeline in zip(model_names,pipelines):

    pipeline.fit(xtrain_res, ytrain_res) 

    pred = pipeline.predict(xtest)

    cv_accuracies = cross_val_score(estimator=pipeline, X=xtrain_res, y=ytrain_res, cv=5)    

    cv_acc.append(cv_accuracies.mean())

    print(name)

    print('Train acc: ', pipeline.score(xtrain_res, ytrain_res))

    print('Test acc: ', pipeline.score(xtest, ytest))

    print(f'CV acc: {cv_accuracies.mean()}')

    print(classification_report(ytest, pred))

    print('Confusion_matrix:')

    print(f'{confusion_matrix(ytest, pred)}')

    print('*'*100)
# Parameter values at which cross-validation performance is maximum

best_acc = np.argmax(cv_acc)

best_classifier = Classifiers[best_acc]

print("Best classifier: {},\n Train accuracy: {}, Cv accuracy: {}, Test accuracy: {}".format(best_classifier, best_classifier.score(xtrain_res, ytrain_res), cv_acc[best_acc], best_classifier.score(xtest, ytest)))
params={

    'bootstrap': [False, True],

    'class_weight': ['balanced', 'balanced_subsample', None],

    'max_depth': [None, 3, 5, 7],

    'max_features': ['auto', 'sqrt', 'log2', None],

    'n_estimators': [50, 100, 200, 300, 400, 500],

    'criterion': ['gini', 'entropy'],

    'max_depth': [3, 5, 7]}



clf = ExtraTreesClassifier()



rs = GridSearchCV(clf, params, cv=5, scoring= 'accuracy')

rs.fit(xtrain_res, ytrain_res)
best_model = rs.best_estimator_

preds = best_model.predict(xtest)

cv_accuracies = cross_val_score(estimator=best_model, X=xtrain_res, y=ytrain_res, cv=5)    

cv_acc = cv_accuracies.mean()

test_acc = accuracy_score(ytest, preds)

print(best_model)

print('CV accuracy:', cv_acc, 'Test accuracy:', test_acc)
a = pd.DataFrame(ytest)

a.insert(1,'predictions', preds)

a

pd.merge(data, a.predictions, left_index=True, right_index=True)