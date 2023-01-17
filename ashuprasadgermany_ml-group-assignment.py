# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load the dataframe

dataframe = pd.read_csv("../input/santander-customer-satisfaction/train.csv")



# Print the first 20 rows

dataframe.head(20)
# We print the sum of NaNs in each coloumn

np.isnan(dataframe).sum()
# Assigning data to the train dataframe

train_data = dataframe
# Dropping Target as we are supposed to predict that

# Dropping ID as it is unique to all rows

train_data = train_data.drop(['TARGET'], axis = 1)

train_data = train_data.drop(['ID'], axis = 1)
# We append the coloumn names that have 0 standard deviation as we can't gather much info from these due to low variance

remove_col_std = []

for i in train_data.columns:

    if(train_data[i].std() == 0):

        remove_col_std.append(i)
# Redefining train dataframe by removing the 0 standard deviation coloumns

train_data = train_data.drop(remove_col_std, axis = 1)
# Removing columns that are identical to one another

remove_col_redund = []

count = 0

for i in range(len(train_data.columns)):

    i_values = train_data[train_data.columns[i]].values

    for j in range(i+1, len(train_data.columns)):

        if(np.array_equal(i_values, train_data[train_data.columns[j]].values)):

            remove_col_redund.append(train_data.columns[j])
# Redefining the train dataframe once more by dropping redundant coloumns

train_data = train_data.drop(remove_col_redund, axis = 1)
# We select the first 20 features and the target coloumn

first_df = pd.concat([train_data.iloc[:, :20], train_data.iloc[:, 305]], axis = 1)
# We print the correlation heatmap for these 20 features with the target variable

plt.figure(figsize = (20 ,20))

corrmat = first_df.corr()

top_corr_features = corrmat.index

sns.heatmap(first_df[top_corr_features].corr(), annot = True, cmap = 'RdYlGn')
first_df = pd.concat([train_data.iloc[:, 296:306], train_data.iloc[:, 305]], axis = 1)
plt.figure(figsize = (20 ,20))

corrmat = first_df.corr()

top_corr_features = corrmat.index

sns.heatmap(first_df[top_corr_features].corr(), annot = True, cmap = 'RdYlGn')
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif
X = train_data.iloc[:, :306]

y = dataframe.iloc[:, 370]
bestfeatures = SelectKBest(score_func = f_classif, k = 40)

fit = bestfeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns, dfscores], axis = 1)

featureScores.columns = ['Features', 'Score']
featureScores
print(featureScores.nlargest(40, 'Score'))
plt.figure(figsize = (25, 6))

sns.barplot(x = featureScores.nlargest(30, 'Score')['Features'], y = featureScores.nlargest(30, 'Score')['Score'])

plt.xticks(rotation = 45)

ax = plt.gca()

plt.show()
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()

selector = model.fit(X, y)
plt.figure(figsize = (60, 40))

feat_importances = pd.Series(model.feature_importances_, index = X.columns)

feat_importances.nlargest(40).plot(kind = 'barh')

plt.show()
plt.figure(figsize = (10, 6))

sns.countplot(dataframe['TARGET'].values)
dataframe[dataframe['TARGET'] == 0].shape[0]
dataframe[dataframe['TARGET'] == 1].shape[0]
# List of the top 40 features selected by the ExtraTreesClassifier

list(feat_importances.nlargest(40).index)
X_feat_1 = X[list(feat_importances.nlargest(40).index)]
print("X shape: " +  str(X_feat_1.shape) + " || y shape: " + str(y.shape))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_feat_1, y, test_size = 0.25, random_state = 4)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]   

}
# Instantiating the XGBClassfier

classifier_etc = XGBClassifier()
random_search = RandomizedSearchCV(classifier_etc, param_distributions = params, n_iter = 5, scoring = 'roc_auc', n_jobs = -1, cv = 5, verbose = 3)
random_search.fit(X_train,y_train)
random_search.best_estimator_
random_search.best_params_
classifier_etc = random_search.best_estimator_
from sklearn.model_selection import cross_val_score

score=cross_val_score(classifier_etc,X,y,cv=10)
score
score.mean()
classifier_etc.fit(X_train, y_train)
y_pred = classifier_etc.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import roc_auc_score

print("Roc AUC: ", roc_auc_score(y_test, classifier_etc.predict_proba(X_test)[:,1], average='macro'))
list(featureScores.nlargest(40, 'Score')['Features'])
X_feat_2 = X[list(featureScores.nlargest(40, 'Score')['Features'])]
print("X shape: " +  str(X_feat_2.shape) + " || y shape: " + str(y.shape))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_feat_2, y, test_size = 0.20, random_state = 4)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV
params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]   

}
classifier_k_best = XGBClassifier()
random_search = RandomizedSearchCV(classifier_k_best, param_distributions = params, n_iter = 5, scoring = 'roc_auc', n_jobs = -1, cv = 5, verbose = 3)
random_search.fit(X_train,y_train)
random_search.best_estimator_
random_search.best_params_
classifier_k_best = random_search.best_estimator_
from sklearn.model_selection import cross_val_score

score=cross_val_score(classifier_k_best,X,y,cv=10)
score
score.mean()
classifier_k_best.fit(X_train, y_train)
y_pred = classifier_k_best.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.metrics import roc_auc_score

print("Roc AUC: ", roc_auc_score(y_test, classifier_k_best.predict_proba(X_test)[:,1], average='macro'))
# Creating the test dataframe

test_df = pd.read_csv("../input/santander-customer-satisfaction/test.csv")
test_df_X = test_df[list(feat_importances.nlargest(40).index)]
test_df_X.shape
test_df_X = sc.fit_transform(test_df_X)
test_ID = test_df.ID
probs = classifier_etc.predict_proba(test_df_X)
submission = pd.DataFrame({"ID":test_ID, "TARGET": probs[:,1]})

submission.to_csv("submission.csv", index=False)