# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, normalize
from sklearn import datasets
from sklearn.cross_validation import KFold, cross_val_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import time

bcdf = pd.read_csv("../input/breastCancer.csv")
bcdf.shape
bcdf.head(5)
col = bcdf.columns
print (col)
y = bcdf.diagnosis
list = ['Unnamed: 32', 'id', 'diagnosis']
x = bcdf.drop(list, axis=1 )
x.head()
# Diagnosis

ax = sns.countplot(y, label='count')
B, M = y.value_counts()

print ('Number of Benign:', B)
print ('Number of Malignant:', M)
# first ten features
data_dia = y
data = x
data_n_2 = (data - data.mean()) / (data.std())
data = pd.concat([y, data_n_2.iloc[:,0:10]], axis=1)
data = pd.melt (data, id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data, split=True, inner="quart")
plt.xticks(rotation=90)
# dari plot diatas dapat disimpulkan bahwa texture_mean feature, median dari Malignant dan Benign terlihat seperti
# terpisah dan itu bagus untuk mengklasifikasi
# second ten features
data = pd.concat([y, data_n_2.iloc[:,10:20]], axis=1)
data = pd.melt (data, id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data, split=True, inner="quart")
# third ten features
data = pd.concat ([y, data_n_2.iloc[:,20:31]], axis=1)
data = pd.melt (data, id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x="features", y="value", hue="diagnosis", data=data, split=True, inner="quart") 
# fourth ten features
# alternative dari violin plot adalah box plot
# box plot sangatn berguna untuk melihat outliers
plt.figure(figsize=(10,10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)
# concavity_worst dan cocave_points_worst terlihat similiar
# untuk mengetahui lebih dalam, lihat joint plot dibawah
sns.jointplot(x.loc[:,"concavity_worst"], x.loc[:,"concave points_worst"], kind="regg", color="#ce1414")
sns.set(style="white")
df = x.loc[:,["radius_worst", "perimeter_worst", "area_worst"]]
g = sns.PairGrid(df, diag_sharey=False)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_upper(plt.scatter)
g.map_diag(sns.kdeplot, lw=3)
sns.set(style="whitegrid", palette="muted")
data_dia = y
data = x
data_n_2 = (data - data.std()) / (data.std())
data = pd.concat ([y, data_n_2.iloc[:,0:10]], axis=1)
data = pd.melt (data, id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)
data = pd.concat ([y, data_n_2.iloc[:,10:20]], axis=1)
data = pd.melt (data, id_vars="diagnosis",
               var_name="features",
               value_name='value')
plt.figure(figsize=(10,10))
tic = time.time()
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
plt.xticks(rotation=90)
data = pd.concat([y,data_n_2.iloc[:,20:31]],axis=1)
data = pd.melt(data,id_vars="diagnosis",
                    var_name="features",
                    value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
toc = time.time()
plt.xticks(rotation=90)
print("swarm plot time: ", toc-tic ," s")
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
sns.heatmap(x.corr())
sns.set_style("whitegrid")
plt.show()
# split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=.8)

# Normalization

norm = Normalizer()

# Fit

norm.fit(x_train)

# Transform both training and testing sets

x_train_norm = norm.transform(x_train)
x_test_norm = norm.transform(x_test)
# Define parameters for optimization using dictionaries {parameter name: parameter list}

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
SVM_params = {'C': [0.001, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']} 
LR_params = {'C': [0.001, 0.1, 1, 10, 100]}
LDA_params = {'n_components':[None, 1,2,3], 'solver':['svd'], 'shrinkage' :[None]}
KNN_params = {'n_neighbors': [1,5,10,20,50], 'p' :[2], 'metric': ['minkowski']}
RF_params = {'n_estimators': [10,50,100]}
DTC_params = {'criterion':['entropy', 'gini'], 'max_depth':[10,50,100]}
# Append list of models with parameter dictionaries

models_opt = []
models_opt.append(('SVM', SVC(), SVM_params))
models_opt.append(('LR', LogisticRegression(), LR_params))
models_opt.append(('LDA', LinearDiscriminantAnalysis(), LDA_params))
models_opt.append(('KNN', KNeighborsClassifier(), KNN_params))
models_opt.append(('RF', RandomForestClassifier(), RF_params))
models_opt.append(('DTC', DecisionTreeClassifier(), DTC_params))
results = []
names = []


def estimator_function(parameter_dictionary, scoring = 'accuracy'):
    
    
    for name, model, params in models_opt:
    
        kfold = KFold(len(x_train_norm), n_folds=5, random_state=2, shuffle=True)

        model_grid = GridSearchCV(model, params)

        cv_results = cross_val_score(model_grid, x_train_norm, y_train, cv = kfold, scoring=scoring)

        results.append(cv_results)

        names.append(name)

        msg = "Cross Validation Accuracy %s: Accarcy: %f SD: %f" % (name, cv_results.mean(), cv_results.std())

        print(msg)

estimator_function(models_opt, scoring = 'accuracy')
# Ensemble Voting

# Create list for estimators
estimators = []

# Create estimator object
model1 = LogisticRegression()

# Append list with estimator name and object
estimators.append(("logistic", model1))
model2 = DecisionTreeClassifier()
estimators.append(("cart", model2))
model3 = SVC()
estimators.append(("svm", model3))
model4 = KNeighborsClassifier()
estimators.append(("KNN", model4))
model5 = RandomForestClassifier()
estimators.append(("RFC", model5))
model6 = LinearDiscriminantAnalysis()
estimators.append(("LDA", model6))

voting = VotingClassifier(estimators)


kfold = KFold(len(x_train_norm), n_folds=5, random_state=2, shuffle=True)


results_voting = cross_val_score(voting, x_train_norm, y_train, cv=kfold)
results.append(results_voting)
names.append('Voting')

print('Accuracy: {} SD: {}'.format(results_voting.mean(), results_voting.std()))
# Instantiate a new LDA model
lda_2 = LinearDiscriminantAnalysis()

# Fit LDA model to the entire training data
lda_2.fit(x_train_norm, y_train)

# Test LDA model on test data
lda_2_predicted = lda_2.predict(x_test_norm)
accuracy_score(y_test, lda_2_predicted)
print(classification_report(y_test, lda_2_predicted))
f1_score(y_test, lda_2_predicted, average='micro')
# Parameters
RC_params = {'n_estimators':[10,50,100,200]}

# Instantiate RFC
RFC_2 = RandomForestClassifier(random_state=42)

# Fit model to traing Data
RFC_2.fit(x_train_norm, y_train)

# Test
RFC_2_predicted = RFC_2.predict(x_test_norm)
print('Accuracy score: {}'.format(accuracy_score(y_test, RFC_2_predicted)))
print(classification_report(y_test, RFC_2_predicted))
f1_score(y_test, RFC_2_predicted, average='micro')
LR_2 = LogisticRegression()
LR_2.fit(x_train_norm, y_train)
LR_2_pred = LR_2.predict(x_test_norm)
print (accuracy_score(y_test, LR_2_pred))
print (classification_report(y_test, LR_2_pred))
f1_score(y_test, LR_2_pred, average='micro')
