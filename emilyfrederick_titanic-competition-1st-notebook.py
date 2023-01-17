#load packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os as os
%matplotlib inline 
import sys

#basic methods from scikit learn package
from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(all_data):
    # Returns divided dfs of training and test set
    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

#Read in data
df_train = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')
df_test = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')
df_all = concat_df(df_train, df_test)

df_train.name = 'Training Set'
df_test.name = 'Test Set'
df_all.name = 'All Set' 
#check head to verify data read in properly
df_all.head()
#view shape of dataframes
print(df_train.shape)
print(df_test.shape)
print(df_all.shape)
#view mean, stdev, etc. for numeric features
df_all.describe()
#view passenger class distribution
groupby = df_all.groupby('Pclass')
targetEDA=groupby['Pclass'].aggregate(len)
plt.figure()
targetEDA.plot(kind='bar', title = 'Passenger Class Distribution', grid=False)
plt.axhline(0, color='k')
plt.ylabel('Number of Passengers')
#show average ticket price per passenger class
df = df_all.groupby('Pclass')['Fare'].mean()
df.plot.bar(title = 'Average Fare by Passenger Class')
plt.ylabel('Fare (dollars)')
#show average age by passenger class
df = df_all.groupby('Pclass')['Age'].mean()
df.plot.bar(title = 'Average Age by Passenger Class')
plt.ylabel('Age (years)')
#show average surival rate by passenger class
df = df_all.groupby('Pclass')['Survived'].mean()
df.plot.bar(title = 'Passenger survival rate by Pclass')
plt.ylabel('Average Survival Rate')
#view survival rate by gender
df = df_all.groupby('Sex')['Survived'].mean()
df.plot.bar(title = 'Passenger survival rate by gender')
plt.ylabel('Average Survival Rate')
#View make-up of 'Embarked'
groupby = df_all.groupby('Embarked')
targetEDA=groupby['Embarked'].aggregate(len)
plt.figure()
targetEDA.plot(kind='bar', title = 'Embarked locaction' ,grid=False)
plt.axhline(0, color='k')
plt.ylabel('Number of Passengers')
#View make up of target variable, survived
targetName = 'Survived'
groupby = df_train.groupby(targetName)
targetEDA=groupby[targetName].aggregate(len)
print(targetEDA)
plt.figure()
targetEDA.plot(kind='bar', title = 'Passenger Survival',grid=False)
plt.axhline(0, color='k')
plt.ylabel('Number of Passengers')
#check for missing values by variable
df_all.isnull().sum()
#impute age based on mean value of passenger class
df_all["Age"] = df_all.groupby("Pclass").transform(lambda x: x.fillna(x.median()))['Age'] 
#impute fare based on median value of passenger class
df_all["Fare"] = df_all.groupby("Pclass").transform(lambda x: x.fillna(x.median()))['Fare'] 
#Create new column with first letter of Cabin 
df_all['CabinLevel'] = df_all['Cabin'].astype(str).str[0]
df_all.head(3)
#check number of passengers per Cabin Level
df_all.groupby('CabinLevel')['Name'].nunique()
#replace CabinLevel "T" with "n"
df_all['CabinLevel'] = df_all['CabinLevel'].str.replace('T','A')
#replace CabinLevel "n" with "missing" so that it makes more sense
df_all['CabinLevel'] = df_all['CabinLevel'].str.replace('n','missing')
df_all.head(3)
df_all['Embarked'] = df_all['Embarked'].fillna('S')
df_all['Family'] = df_all['SibSp'] + df_all['Parch'] + 1 
df_all['Title'] = df_all['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
df_all['Title'].unique()
#Apply dictionary to group the various titles
Title_Dictionary = {"Capt":"Officer","Col":"Officer","Major":"Officer","Dr":"Doctor","Rev":"Clergy","Jonkheer":"Royalty",
        "Don":"Royalty","Sir":"Royalty","Countess":"Royalty","Dona":"Royalty","Lady":"Royalty","Mme":"Mrs","Ms":"Mrs",
        "Mrs":"Mrs","Mlle":"Miss","Miss":"Miss","Mr":"Mr","Master":"Master"}
    
# map each title to correct category
df_all['Title'] = df_all['Title'].map(Title_Dictionary)
df_all.head(3)
#Group fare into 13 bins
df_all['Fare'] = pd.qcut(df_all['Fare'], 13)
#show average surival rate by Fare bin
df = df_all.groupby('Fare')['Survived'].mean()
df.plot.bar(title = 'Passenger survival rate by Fare bin')
plt.ylabel('Average Survival Rate')
#Group age into 10 bins
df_all['Age'] = pd.qcut(df_all['Age'], 10, duplicates = 'drop')
#show average surival rate by Age bin
df = df_all.groupby('Age')['Survived'].mean()
df.plot.bar(title = 'Passenger survival rate by Age bin')
plt.ylabel('Average Survival Rate')
#drop Name, Ticket, Cabin, SibSp, and Parch 
df_all.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace = True)
df_all.head()
#Move 'Survived' to first column
col_name = 'Survived'
df_all.insert(0, col_name, df_all.pop(col_name))
df_all.head()
#Perform label encoding for Fare and Age bins
from sklearn.preprocessing import LabelEncoder

non_numeric_features = ['Age', 'Fare']

for feature in non_numeric_features:        
     df_all[feature] = LabelEncoder().fit_transform(df_all[feature])
#create dummy variable for categorical variables
for col in df_all.columns[1:]:
    attName = col
    dType = df_all[col].dtype
    if dType == object:
        df_all = pd.concat([df_all, pd.get_dummies(df_all[col], prefix=col)], axis=1)
        del df_all[attName]
df_all.head()
#split data back into train and test sets
df_train, df_test = divide_df(df_all)
# split df_train dataset into 60/40 testing and training sets
# column location 1 to end of dataframe are the features.
# column location 0 is the target
features_train, features_test, target_train, target_test = train_test_split(
    df_train.iloc[:,1:].values, df_train.iloc[:,0].values, test_size=0.40, random_state=0)
#print shape train/test dataframes
print(features_test.shape)
print(features_train.shape)
print(target_test.shape)
print(target_train.shape)
#Build KNN classifier with default parameters for baseline, neighbors = 5
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
clf_knn = make_pipeline(StandardScaler(), KNeighborsClassifier())
print(clf_knn)
#Train
clf_knn = clf_knn.fit(features_train, target_train)
#Validate
target_predicted_knn = clf_knn.predict(features_test)
print("KNN Accuracy Score", accuracy_score(target_test, target_predicted_knn))
print(classification_report(target_test, target_predicted_knn))
print(confusion_matrix(target_test, target_predicted_knn))

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(target_test, target_predicted_knn).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
#Determine best value for n_neighbors
k_range = [1,3,5,7,9]

# list of scores from k_range
k_scores = []

for k in k_range:
    clf_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = k))
    scores = cross_val_score(clf_knn, features_train, target_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
#Re-run with best n_neighbors
clf_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors = 7))

#Train
clf_knn = clf_knn.fit(features_train, target_train)

#Validate
target_predicted_knn = clf_knn.predict(features_test)

#Results
print("KNN Accuracy Score", accuracy_score(target_test, target_predicted_knn))
print(classification_report(target_test, target_predicted_knn))
print(confusion_matrix(target_test, target_predicted_knn))

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(target_test, target_predicted_knn).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
#verify with 5-fold cross validation
scores = cross_val_score(clf_knn, features_train, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean()  
#Build default decision tree.
from sklearn import tree 
clf_dt = tree.DecisionTreeClassifier(random_state = 123)
#Train
clf_dt = clf_dt.fit(features_train, target_train)
#Validate
target_predicted_dt = clf_dt.predict(features_test)
#DT Results
print("DT Accuracy Score", accuracy_score(target_test, target_predicted_dt))
print(classification_report(target_test, target_predicted_dt))
print(confusion_matrix(target_test, target_predicted_dt))

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(target_test, target_predicted_dt).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
#determine optimal parameters via grid search
param_grid={"max_depth": range(2,10,2),
           "min_samples_split": range(2,10,2),
           "max_features": range(2,10,2),
           "min_samples_leaf": [1,3,5,7],
           "criterion": ["gini","entropy"]}
clf_dt = tree.DecisionTreeClassifier(random_state = 123)
grid_search = GridSearchCV(clf_dt, param_grid,n_jobs=-1, cv=5)
grid_search.fit(features_train, target_train)

print("Best", grid_search.best_params_)  
#Build decision tree with best params
clf_dt = tree.DecisionTreeClassifier(criterion = 'gini',max_depth = 6, max_features = 6, min_samples_leaf = 7, min_samples_split = 2, random_state = 123)
print(clf_dt)

#Train
clf_dt = clf_dt.fit(features_train, target_train)

#Validate
target_predicted_dt = clf_dt.predict(features_test)

#DT Results
print("DT Accuracy Score", accuracy_score(target_test, target_predicted_dt))
print(classification_report(target_test, target_predicted_dt))
print(confusion_matrix(target_test, target_predicted_dt))

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(target_test, target_predicted_dt).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
#verify with 5-fold cross validation
scores = cross_val_score(clf_dt, features_train, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean() 
#Train Random Forest model with default criteria
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(random_state = 123)
clf_rf = forest.fit(features_train, target_train)
print(clf_rf)
#make predictions
target_predicted_rf = clf_rf.predict(features_test)
#Validate
print("RF Accuracy Score", accuracy_score(target_test, target_predicted_rf))
print(classification_report(target_test, target_predicted_rf))
print(confusion_matrix(target_test, target_predicted_rf))

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(target_test, target_predicted_rf).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
#determine optimal params via grid search
param_grid={"n_estimators": [10,25,50],
           "max_depth": range(2,10,2),
           "min_samples_split":range(2,10,2),
           "min_samples_leaf": range(2,10,2),
           "max_features": range(1,10,1)}
clf_rf = RandomForestClassifier(random_state = 123)
grid_search = GridSearchCV(clf_rf, param_grid,n_jobs=-1, cv=5)
grid_search.fit(features_train, target_train)

print("Best", grid_search.best_params_)  
#re-run with best params
forest = RandomForestClassifier(n_estimators = 25, max_depth = 8, max_features = 8, min_samples_leaf = 2, min_samples_split = 6, random_state = 123)
clf_rf = forest.fit(features_train, target_train)
print(clf_rf)

#make predictions
target_predicted_rf = clf_rf.predict(features_test)

#Validate
print("RF Accuracy Score", accuracy_score(target_test, target_predicted_rf))
print(classification_report(target_test, target_predicted_rf))
print(confusion_matrix(target_test, target_predicted_rf))

#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(target_test, target_predicted_rf).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
#verify with 5-fold cross validation
scores = cross_val_score(clf_rf, features_train, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean() 
from sklearn.svm import SVC
#Build
clf_rbfSVC = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced', random_state = 123))
print(clf_rbfSVC)
#Train
clf_rbfSVC.fit(features_train, target_train)
#Validate
target_predicted = clf_rbfSVC.predict(features_test)
print("RBF SVM Accuracy Score", accuracy_score(target_test, target_predicted))
print(classification_report(target_test, target_predicted)) #no=0; yes=1
print(confusion_matrix(target_test, target_predicted))
#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(target_test, target_predicted).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=123))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1500.0,2000.0]

param_grid = [{'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

gs = GridSearchCV(estimator=pipe_svc, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  cv=10,
                  n_jobs=-1)
gs = gs.fit(features_train, target_train)
print(gs.best_score_)
print(gs.best_params_)
#Re-run with best C and gamma
#Build
clf_rbfSVC = make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced', C = 1000, gamma = 0.001, random_state = 123))
print(clf_rbfSVC)

#Train
clf_rbfSVC.fit(features_train, target_train)

#Validate
target_predicted = clf_rbfSVC.predict(features_test)
print("RBF SVM Accuracy Score", accuracy_score(target_test, target_predicted))
print(classification_report(target_test, target_predicted)) #no=0; yes=1
print(confusion_matrix(target_test, target_predicted))
#extracting true_positives, false_positives, true_negatives, false_negatives
tn, fp, fn, tp = confusion_matrix(target_test, target_predicted).ravel()
print("True Negatives: ",tn)
print("False Positives: ",fp)
print("False Negatives: ",fn)
print("True Positives: ",tp)
#verify with 5-fold cross validation
scores = cross_val_score(clf_rbfSVC, features_train, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean()  
#Scale data before SGD
scaler = StandardScaler()  
scaler.fit(features_train)  
features_train_norm = scaler.transform(features_train)  
# apply same transformation to test data
features_test_norm = scaler.transform(features_test)
#Build default SGD classifier
from sklearn.linear_model import SGDClassifier
clf_sgd =SGDClassifier(random_state = 123)
print(clf_sgd)
#Train
clf_sgd.fit(features_train_norm, target_train)
#Validate
target_predicted = clf_sgd.predict(features_test_norm)
print("Accuracy", accuracy_score(target_test, target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
param_grid={"alpha": [0.0001,0.0005,0.001,0.005,0.01,0.05],
           "loss": ["hinge","log","modified_huber","squared_hinge"],
           "penalty": ['l2', 'l1', 'elasticnet']} 
clf_sgd = SGDClassifier(random_state = 123)
grid_sgd = GridSearchCV(clf_sgd, param_grid,n_jobs=-1, cv=5)
grid_sgd.fit(features_train_norm, target_train)

print("BEST SCORE", grid_sgd.best_score_)
print("BEST PARAM", grid_sgd.best_params_)
#re-run SGD classifier with best parameters
clf_sgd =SGDClassifier(loss='log', alpha = 0.01, penalty = 'elasticnet', random_state = 123)
print(clf_sgd)
#Train
clf_sgd.fit(features_train_norm, target_train)
#Validate
target_predicted = clf_sgd.predict(features_test_norm)
print("Accuracy", accuracy_score(target_test, target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
#verify with 5-fold cross validation
scores = cross_val_score(clf_sgd, features_train_norm, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean()  
#build default adaboost model
from sklearn.ensemble import AdaBoostClassifier
tree = tree.DecisionTreeClassifier(max_depth = 1) #Decision Tree stump
clf_dt_ab = AdaBoostClassifier(base_estimator = tree,
                               n_estimators = 100,
                               algorithm="SAMME.R",
                              random_state = 123)
print(clf_dt_ab)
clf_dt_ab.fit(features_train, target_train)
target_predicted=clf_dt_ab.predict(features_test)
print("Adaboost Accuracy", accuracy_score(target_test,target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
param_grid={"learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0, 1.5],
           "n_estimators": [100,250,500]} 
clf_dt_ab = AdaBoostClassifier(base_estimator = tree,
                               algorithm="SAMME.R",
                              random_state = 123)
grid_svm = GridSearchCV(clf_dt_ab, param_grid,n_jobs=-1, cv=5)
grid_svm.fit(features_train, target_train)

print("BEST SCORE", grid_svm.best_score_)
print("BEST PARAM", grid_svm.best_params_)
#Re-run Adaboost with best params
clf_dt_ab = AdaBoostClassifier(base_estimator = tree,
                               n_estimators = 250,
                               learning_rate = 0.05,
                               algorithm="SAMME.R",
                              random_state = 123)
print(clf_dt_ab)
clf_dt_ab.fit(features_train, target_train)
target_predicted=clf_dt_ab.predict(features_test)
print("Adaboost Accuracy", accuracy_score(target_test,target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
#verify with 5-fold cross validation
scores = cross_val_score(clf_dt_ab, features_train, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean()  
from sklearn.ensemble import BaggingClassifier
#Build 100 trained decision tree models on 100 bootstrapped training sets
clf_bag = BaggingClassifier(n_estimators=100, random_state = 123)
print(clf_bag)
#Train
clf_bag.fit(features_train, target_train)
#Validate 
target_predicted=clf_bag.predict(features_test)
print("Bagging Accuracy", accuracy_score(target_test, target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
#grid search for best max_features
param_grid={"max_features": range(2,10,1),
            "n_estimators": [100,200,300]} 
clf_bag = BaggingClassifier(random_state = 123)
grid_bag = GridSearchCV(clf_bag, param_grid,n_jobs=-1, cv=5)
grid_bag.fit(features_train, target_train)

print("BEST SCORE", grid_bag.best_score_)
print("BEST PARAM", grid_bag.best_params_)
#re-run with best max_features and n_estimators
clf_bag = BaggingClassifier(n_estimators=100, max_features = 8, random_state = 123)
print(clf_bag)
#Train
clf_bag.fit(features_train, target_train)
#Validate 
target_predicted=clf_bag.predict(features_test)
print("Bagging Accuracy", accuracy_score(target_test, target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
#verify with 5-fold cross validation
scores = cross_val_score(clf_bag, features_train, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean()  
#Build Gradient Boosting model
from sklearn.ensemble import GradientBoostingClassifier
clf_GBC = GradientBoostingClassifier(n_estimators=100, random_state=123) #default learning rate is 0.1
print(clf_GBC)
clf_GBC.fit(features_train, target_train)
target_predicted=clf_GBC.predict(features_test)
print("Gradient Boost Accuracy", accuracy_score(target_test,target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
#grid search for best min_samples_split
param_grid={"min_samples_split": range(2,20,2),
           "n_estimators": [100,200,300],
           "max_depth": range(2,10,2),
           "subsample": [0.6,0.8,1],
           "learning_rate": [0.01,0.05,0.1]} 
clf_GBC = GradientBoostingClassifier(random_state=123) 
grid_GBC = GridSearchCV(clf_GBC, param_grid,n_jobs=-1, cv=5)
grid_GBC.fit(features_train, target_train)

print("BEST SCORE", grid_GBC.best_score_)
print("BEST PARAM", grid_GBC.best_params_)
#Re-run with best params
clf_GBC = GradientBoostingClassifier(n_estimators=100, 
                                     min_samples_split = 16,
                                     learning_rate = 0.01,
                                     max_depth = 4,
                                     subsample = 1,
                                     random_state=123) 
print(clf_GBC)
clf_GBC.fit(features_train, target_train)
target_predicted=clf_GBC.predict(features_test)
print("Gradient Boost Accuracy", accuracy_score(target_test,target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
#verify with 5-fold cross validation
scores = cross_val_score(clf_GBC, features_train, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean()  
from sklearn.ensemble import ExtraTreesClassifier

#Build Extra Trees Classifier 
clf_xdt = ExtraTreesClassifier(n_estimators= 100, n_jobs=-1,class_weight="balanced", random_state=123)
#features used to split are picked randomly
print(clf_xdt)
clf_xdt.fit(features_train, target_train)
target_predicted=clf_xdt.predict(features_test)
print("Extra Trees Accuracy", accuracy_score(target_test,target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
#grid search for best params
param_grid={"max_depth": range(2,10,2),
           "n_estimators": [100,200,300],
           "min_samples_split": range(2,10,2),
           "criterion": ["gini","entropy"]} 
clf_xdt = ExtraTreesClassifier(class_weight="balanced", random_state=123)
grid_xdt = GridSearchCV(clf_xdt, param_grid,n_jobs=-1, cv=5)
grid_xdt.fit(features_train, target_train)

print("BEST SCORE", grid_xdt.best_score_)
print("BEST PARAM", grid_xdt.best_params_)
#Re-run with best params
clf_xdt = ExtraTreesClassifier(n_estimators= 200, max_depth = 8, min_samples_split = 2, criterion = 'entropy',class_weight="balanced", random_state=123)
#features used to split are picked randomly
print(clf_xdt)
clf_xdt.fit(features_train, target_train)
target_predicted=clf_xdt.predict(features_test)
print("Extra Trees Accuracy", accuracy_score(target_test,target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
#verify with 5-fold cross validation
scores = cross_val_score(clf_xdt, features_train, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean() 
#Build
from sklearn.neural_network import MLPClassifier
clf_nn = MLPClassifier(hidden_layer_sizes=(3), #one hidden layer with three units/nodes
                       solver="lbfgs", 
                       learning_rate = 'adaptive',
                       max_iter = 1000,
                       random_state=123) 
print(clf_nn)
#Train
clf_nn.fit(features_train_norm, target_train)
#Validate
target_predicted = clf_nn.predict(features_test_norm)
print("Accuracy", accuracy_score(target_test, target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
#node values to test
MLP_range = [1,3,5,7,9]

# list of scores from MLP_range
MLP_scores = []

for i in MLP_range:
    clf_nn = MLPClassifier(hidden_layer_sizes=(i), solver="lbfgs", learning_rate = 'adaptive', max_iter = 10000, random_state=123)
    scores = cross_val_score(clf_nn, features_train_norm, target_train, cv=5, scoring='accuracy')
    MLP_scores.append(scores.mean())
print(MLP_scores)
#node values to test
MLP_range = [1,3,5,7,9]

# list of scores from MLP_range
MLP_scores = []

for i in MLP_range:
    for j in MLP_range: 
        clf_nn = MLPClassifier(hidden_layer_sizes=(i,j), solver="lbfgs", learning_rate = 'adaptive', max_iter = 10000, random_state=123)
        scores = cross_val_score(clf_nn, features_train_norm, target_train, cv=5, scoring='accuracy')
        MLP_scores.append(scores.mean())
print(MLP_scores)
#one hidden layers with five units/nodes
clf_nn = MLPClassifier(hidden_layer_sizes=(5), solver="lbfgs", learning_rate = 'adaptive', max_iter = 1000, random_state=123) 
print(clf_nn)
#Train
clf_nn.fit(features_train_norm, target_train)
#Validate
target_predicted = clf_nn.predict(features_test_norm)
print("Accuracy", accuracy_score(target_test, target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
#verify with 5-fold cross validation
scores = cross_val_score(clf_nn, features_train_norm, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean() 
from sklearn.ensemble import VotingClassifier

#Stacking three tuned learners: RandomForestClassifier, BaggingClassifier, and GradientBoost

learner_1 = BaggingClassifier(n_estimators=100, max_features = 8, random_state = 123)
learner_2 = RandomForestClassifier(n_estimators = 25, max_depth = 8, max_features = 8, min_samples_leaf = 2, min_samples_split = 6, random_state = 123)
learner_3 = GradientBoostingClassifier(n_estimators=100, 
                                     min_samples_split = 16,
                                     learning_rate = 0.01,
                                     max_depth = 4,
                                     subsample = 1,
                                     random_state=123)

stacked_learner = VotingClassifier(estimators=[('bag', learner_1), ('rf', learner_2),
                                              ('gb', learner_3)], voting='hard') #count yes and no

for MV, label in zip([learner_1, learner_2, learner_3, stacked_learner], 
                     ['Bagging', 'Random Forest', 'Gradient Boosting', 'Second Stage Learner']):
    scores2 = cross_val_score(MV, features_train, target_train, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores2.mean(), scores2.std(), label))
#Validate performance
stacked_learner.fit(features_train, target_train)
target_predicted=stacked_learner.predict(features_test)
print("Stacked Learner Accuracy", accuracy_score(target_test,target_predicted))
print(classification_report(target_test, target_predicted))
print(confusion_matrix(target_test, target_predicted))
#verify with 5-fold cross validation
scores = cross_val_score(stacked_learner, features_train, target_train, cv=5)
print("Cross Validation Score for each K",scores)
scores.mean() 
from sklearn.metrics import roc_curve, roc_auc_score

# Instantiate the classfiers and make a list
classifiers = [BaggingClassifier(n_estimators=100, max_features = 8, random_state = 123),
               RandomForestClassifier(n_estimators = 25, max_depth = 8, max_features = 8, min_samples_leaf = 2, min_samples_split = 6, random_state = 123),
               GradientBoostingClassifier(n_estimators=100, 
                                     min_samples_split = 16,
                                     learning_rate = 0.01,
                                     max_depth = 4,
                                     subsample = 1,
                                     random_state=123),
               VotingClassifier(estimators=[('bag', learner_1), ('rf', learner_2),
                                              ('gb', learner_3)], voting='soft')]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(features_train, target_train)
    target_predicted = model.predict_proba(features_test)[::,1]
    
    fpr, tpr, _ = roc_curve(target_test,  target_predicted)
    auc = roc_auc_score(target_test, target_predicted)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)


fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()
predictions = clf_GBC.predict(df_test).astype(int)

output = pd.DataFrame({'PassengerId': df_test.index + 1, 'Survived': predictions})
output.to_csv('final_submission.csv', index=False)