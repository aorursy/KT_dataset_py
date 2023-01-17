# Import libraries necessary for this project

import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import Imputer

from sklearn.cross_validation import train_test_split

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.grid_search import GridSearchCV

from sklearn.metrics import fbeta_score, make_scorer
# EXPLORING THE DATA

# Load the wholesale customers dataset

data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")

full_set = data.append(test_data)

full_set.set_index('PassengerId', inplace = True)

full_set = full_set.drop('Ticket', axis=1)

print("The titanic dataset has {} samples with {} features each.".format(*data.shape))



# simple data exploration

n_survivors = (data['Survived']==1).sum()

n_fatalities = (data['Survived']==0).sum()



survived_percent = float(n_survivors) / len(data) * 100

print("Percentage of passengers who survived the titanic disaster: {:.2f}%".format(survived_percent))
# DATA PRE-PROCESSING

# define a simple function for feature engineering with the names

def get_title(string):

    string = string.split(", ")[1].strip()

    string = string.split(" ")[0].strip()

    return string



full_set['Name'] = full_set.apply(lambda row: get_title(row['Name']), axis=1)
# DATA PRE-PROCESSING

# check if any continuous features (numerical) are skewed

data['Age'].hist()

data['SibSp'].hist()

data['Fare'].hist()

data['Parch'].hist()
# DATA PRE-PROCESSING

# split the data into features and target variable

survive_raw = full_set['Survived']

features_raw = full_set.drop('Survived', axis=1)
# DATA PRE-PROCESSING

# imputation (get rid of the nan)

num_features_raw = ['Age', 'SibSp', 'Fare', 'Parch']

num_imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

features_raw[num_features_raw] = num_imp.fit_transform(features_raw[num_features_raw])



features_raw['Cabin'].fillna('U', inplace=True)

features_raw['Embarked'].fillna('U', inplace=True)



def get_cabin(cabin):

    cabin = cabin[0]

    return cabin



features_raw['Cabin'] = features_raw.apply(lambda row: get_cabin(row['Cabin']), axis=1)

features_raw.isnull().sum()



features_raw_drop_na = features_raw.dropna(axis=0, how='any')
# DATA PRE-PROCESSING

# log transform the skewed data

skewed = ['Age', 'SibSp', 'Fare', 'Parch']

features_log_transformed = pd.DataFrame(data = features_raw_drop_na)

features_log_transformed[skewed] = features_raw_drop_na[skewed].apply(lambda x: np.log(x + 1))
# DATA PRE-PROCESSING

# normalizing numerical features

scaler = MinMaxScaler()

numerical = ['Age', 'SibSp', 'Fare', 'Parch']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)

features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
# DATA PRE-PROCESSING

# creating the final feature set

cat_features_raw = ['Pclass', 'Name', 'Sex', 'Cabin', 'Embarked']



final_cat_features = pd.get_dummies(features_log_minmax_transform[cat_features_raw])

final_num_features = features_log_minmax_transform[skewed]

final_features = pd.concat([final_cat_features, final_num_features], axis = 1)
# DATA PRE-PROCESSING

# now that we have done feature engineering with both train and test data, we need to split them apart again

train_features = final_features[:len(data.index)]

train_target = survive_raw[:len(data.index)]



test_features = final_features[len(data.index):]
# DATA PRE-PROCESSING

# shuffle and split the data into training and cross-validation sets

X_train, X_test, y_train, y_test = train_test_split(train_features, train_target, test_size = 0.2, random_state = 0)



# Show the results of the split

print("Training set has {} samples.".format(X_train.shape[0]))

print("Testing set has {} samples.".format(X_test.shape[0]))
# NAIVE PREDICTOR PERFORMANCE

# if we chose a model that always predict '1' (i.e. passenger survives) then what is the model's accuracy and F-score?

accuracy = float(np.sum(survive_raw == 1))/len(survive_raw)

recall = 1

precision = accuracy



beta = 0.5

fscore = (1 + beta**2) * (precision*recall) / ((beta**2 * precision) + recall)



# print the results 

print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))
# ML IMPLEMENTATION

# first we call our DT classifier function

clf1 = DecisionTreeClassifier(random_state=0)

# train our DT classifier with the training data

clf1.fit(X_train, y_train)

# make predictions with CV

clf1_predictions = clf1.predict(X_test)

# find accuracy and f1 score of our DT model

DT_accuracy = accuracy_score(y_test, clf1_predictions)

DT_F1 = f1_score(y_test, clf1_predictions)

# print the results 

print("Decision Tree Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(DT_accuracy, DT_F1))



# we repeat the above process with another learner

# first we call our SVM classifier function

clf2 = svm.SVC(random_state=0)

# train our SVM classifier with the training data

clf2.fit(X_train, y_train)

# make predictions with CV

clf2_predictions = clf2.predict(X_test)

# find accuracy and f1 score of our SVM model

SVM_accuracy = accuracy_score(y_test, clf2_predictions)

SVM_F1 = f1_score(y_test, clf2_predictions)

# print the results 

print("SVM Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(SVM_accuracy, SVM_F1))



# we repeat the above process with a third learner

# first we call our SVM classifier function

clf3 = KNeighborsClassifier()

# train our DT classifier with the training data

clf3.fit(X_train, y_train)

# make predictions with CV

clf3_predictions = clf3.predict(X_test)

# find accuracy and f1 score of our DT model

kNN_accuracy = accuracy_score(y_test, clf3_predictions)

kNN_F1 = f1_score(y_test, clf3_predictions)

# print the results 

print("kNN Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(kNN_accuracy, kNN_F1))
# GRID SEARCH

# after we have found the best performing model, we can run grid-search to make the model perform even better

# set our classifier as the best performing classifier

clf = clf3

# choose some parameters for gridsearch

parameters = {'n_neighbors':[3, 5, 7], 'weights':['uniform', 'distance']}

# make an fbeta score scoring object

scorer = make_scorer(fbeta_score, beta=0.5)

# perform grid search on the classifier using 'scorer' as the scoring method using gridsearch

grid_obj = GridSearchCV(estimator=clf, param_grid=parameters, scoring = scorer)

# fit the grid search object to the training data and find the optimal parameters

grid_fit = grid_obj.fit(X_train, y_train)

# get the estimator

best_clf = grid_fit.best_estimator_

# make predictions using the new optimized model

best_predictions = best_clf.predict(X_test)



# find accuracy and f1 score of our optimized kNN model

kNN_optimized_accuracy = accuracy_score(y_test, best_predictions)

kNN_optimized_F1 = f1_score(y_test, best_predictions)
# MODEL EVALUATION

# print results of the benchmark, un-optimized and the optimized models

print("Naive predictor:")

print("[Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

print("Un-optimized kNN:")

print("[Accuracy score: {:.4f}, F-score: {:.4f}]".format(kNN_accuracy, kNN_F1))

print("Optimized kNN:")

print("[Accuracy score: {:.4f}, F-score: {:.4f}]".format(kNN_optimized_accuracy, kNN_optimized_F1))
# MAKE TEST PREDICTION

# finally we have our 'perfect' model and it is time to make predictions for our test set

output = best_clf.predict(test_features).astype(int)

df_output = pd.DataFrame()

aux = pd.read_csv("../input/test.csv")

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)