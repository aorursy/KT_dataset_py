# Imports

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

import pandas as pd

import numpy as np

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

import statistics

import datetime

import string

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
train_data = pd.read_csv("train.csv") 

test_data = pd.read_csv("test_2.csv")



Y_train = train_data["class"]



X_train = train_data.drop("class", axis=1)

X_train = X_train.drop("ID", axis=1)

X_test = test_data.drop("ID", axis=1)



X_train.head()
#Combined the datasets to create new features

X_combined = X_train.append(X_test, ignore_index=True)

print(X_train.shape)

print(X_test.shape)

print(X_combined.shape)
#Create any new features

#Title = length of title

X_combined["titleLength"] = X_combined['title'].str.len()

X_combined["titleCaps"] = X_combined['title'].str.findall(r'[A-Z]').str.len()

X_combined["titlePercCaps"] = X_combined['titleCaps']/X_combined['titleLength']



#Detect '!' points

X_combined["titleExclamation"] = X_combined['title'].str.findall(r'[!]').str.len()

X_combined["titlePercExclamation"] = X_combined['titleExclamation']/X_combined['titleLength']



#Detect '?' marks

X_combined["titleQuestion"] = X_combined['title'].str.findall(r'[?]').str.len()

X_combined["titlePercQuestion"] = X_combined['titleQuestion']/X_combined['titleLength']



#Convert title to lowercase

X_combined['title'] = X_combined['title'].str.lower()



#Convert comments to lowercase

for i in range(1,11):

    X_combined['user_comment_{}'.format(i)] = X_combined['user_comment_{}'.format(i)].str.lower()



#Like percentage column

X_combined["dislikePerc"] = X_combined["dislikeCount"]/(X_combined["likeCount"]+X_combined["dislikeCount"])

X_combined['commentPerc'] = X_combined["commentCount"]/X_combined["viewCount"]

X_combined['dislikeView'] = X_combined["dislikeCount"]/X_combined["viewCount"]



#Numbers?

X_combined["titleNums"] = X_combined['title'].str.findall(r'[0-9]').str.len()

X_combined["titleNums"] = X_combined["titleNums"].apply(lambda x: x == 0)



X_combined["combinedPercentage"] = X_combined["titlePercQuestion"]+X_combined["titlePercExclamation"]+X_combined["titlePercCaps"]

#Select columns to use for model and separate test/train

X_combined_model = X_combined[['titlePercCaps', 'titlePercExclamation','titlePercQuestion','viewCount','commentPerc', 'dislikePerc', 'dislikeView', "titleNums"]]

X_train = X_combined_model[0:7105]

X_test = X_combined_model[7105:]

print(X_train.shape)

print(X_test.shape)

print(X_train.head())

#print(Y_train.head())

print(X_test.head())
'''

#Scale data for neural network - not used in final

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)

scaled_X_test = scaler.transform(X_test)

print(X_train)

X_train = pd.DataFrame(scaled_X_train, index=X_train.index, columns=X_train.columns)

X_test = pd.DataFrame(scaled_X_test, index=X_test.index, columns=X_test.columns)

'''
'''

#Hyperpameter tuning code



#mlp = MLPClassifier(max_iter=100)



rf = RandomForestClassifier()

parameter_space = {

    'hidden_layer_sizes': [(6,6,6), (3,6,3), (10,10)],

    'activation': ['tanh', 'relu'],

    'solver': ['sgd', 'adam'],

    'alpha': [0.0001, 0.05],

    'learning_rate': ['constant','adaptive'],

}



parameter_space = {

    'n_estimators': [200,500],

    'max_depth': [10,15,20],

    'min_samples_leaf': [2,5,10]

}





clf = GridSearchCV(rf, parameter_space, n_jobs=-1, cv=2)

clf.fit(X_train, Y_train)



# Best paramete set

print('Best parameters found:\n', clf.best_params_)



# All results

means = clf.cv_results_['mean_test_score']

stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):

    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

'''
#Train model with k-fold validation

skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)



f1 = []



fold = 1



for train_index, test_index in skf.split(X_train, Y_train):

    #print("Fold Number:", fold)

    #print("Training Data Index", train_index)

    #print("Testing Data Index:", test_index)

    

    x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]

    y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]

    

    #print("Testing Data Target Output:", y_test)

    

    clf = RandomForestClassifier(n_estimators = 500, max_depth = 20, min_samples_leaf = 2)

    #clf = MLPClassifier(activation = 'relu', alpha = 0.05, hidden_layer_sizes = (10,10), learning_rate = 'constant', solver = 'adam')

    clf.fit(x_train, y_train)

    

    predicted_y_test = clf.predict(x_test)

    #print(predicted_y_test)

    #print("Testing Data Prediction Output:", predicted_y_test,)

    

    f1.append(f1_score(y_test, predicted_y_test))

    

    print("f1 Score:", f1_score(y_test, predicted_y_test))

    

    fold += 1



print("F1: ", statistics.mean(f1))

#Train full real model

clf = RandomForestClassifier(n_estimators = 500, max_depth = 20, min_samples_leaf = 2)

clf.fit(X_train,Y_train)
feature_importances = pd.DataFrame(clf.feature_importances_, index = X_train.columns,

                                    columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)
#Make new predictions

Y_pred = clf.predict(X_test)



test_data["class"] = Y_pred

test_data["class"] = test_data["class"].map(lambda x: "True" if x==1 else "False")

result = test_data[["ID","class"]]

result.to_csv("submission.csv", index=False)

result.head()