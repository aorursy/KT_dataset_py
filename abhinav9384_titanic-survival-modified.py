# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine_df = [train_df, test_df]
# TO DO activities
# 1. check usefulness of ticket column
# 2. do missing value imputation in embarked and check if it is useful
# Till then , drop both of them to get model going
# 3. instead of pd.get dummies, do 1 hot encoding on title and gender
# 4. data visualization
# 5. do age imputation through code, not manually
# 6. Do Grid Search / Paramter Tuning


# count the number of NaN values in each column
print(train_df.isnull().sum())
print(test_df.isnull().sum())

#Imputation for Fare since its just 1 record missing in test.csv/test.df
#Lets see what data this contains
test_df[test_df.Fare.isnull()]
'''
test_df[(test_df["Pclass"] == 3) & (test_df["Sex"] == "male") & (test_df["Embarked"] == "S") 
& (test_df["Age"] > 49.9) & (test_df["Age"] < 75)]
#since test df has only 1 row which satisfies above conditions, checking in train_df
train_df[(train_df["Pclass"] == 3) & (train_df["Sex"] == "male") & (train_df["Embarked"] == "S") 
& (train_df["Age"] > 49.9) & (train_df["Age"] < 75)]
'''
#By seeing above data, filling only NaN fare in test_df, with passenger id of 327 who has similar 
#characteristics : male, pclass=3, age=61
test_df["Fare"] = test_df["Fare"].fillna(6.2375)
#confirm imputation happened
test_df[test_df.Fare.isnull()]  #gives no results after imputation
test_df[test_df["PassengerId"] == 1044]

#Before doing age imputation, create new features: familySize, isAlone, Title
for df in combine_df:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
train_df["IsAlone"] = [1 if i<2 else 0 for i in train_df.FamilySize]
test_df["IsAlone"] = [1 if i<2 else 0 for i in test_df.FamilySize]

for df in combine_df:
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#Check if any relation between Age and Title
'''
train_df[['Title', 'Sex', 'Age']].groupby(['Title', 'Sex'], as_index=False).mean()
train_df[['Title', 'Sex', 'Age']].groupby(['Title', 'Sex'], as_index=False).max()
train_df[['Title', 'Sex', 'Age']].groupby(['Title', 'Sex'], as_index=False).min()

test_df[['Title', 'Sex', 'Age']].groupby(['Title', 'Sex'], as_index=False).mean()
test_df[['Title', 'Sex', 'Age']].groupby(['Title', 'Sex'], as_index=False).max()
test_df[['Title', 'Sex', 'Age']].groupby(['Title', 'Sex'], as_index=False).min()
'''
#Surprise in data
#Some girls with age less than 18 are also marked as Mrs in data


# reduce titles so that it is easy to categorize people in groups
#Not changing Title Capt as its 60+ yrs, just want to keep it separate
for df in combine_df:
       df['Title'] = np.where((df['Sex'] == 'male') & 
            (df['Title'].isin (['Col', 'Don', 'Dr', 'Rev', 'Sir', 'Jonkheer', 'Major'])), 
                              'Mr', df['Title'])
# marking all ladies in just master and Ms
#This is because some girls with age less than 18 are also marked as Mrs in data
for df in combine_df:        
       df['Title'] = np.where((df['Sex'] == 'female') & 
            (df['Title'].isin (['Countess', 'Lady', 'Dr', 'Dona', 'Mrs', 'Miss', 'Mme', 'Mlle'])), 
                              'Ms', df['Title'])


#Do age imputation based on Title & Sex
# We have 2 values in Sex and 4 values in Title (Capt, Master, Mr, Ms)
# create temporary df and store all results of sex, title, age
# then take their mean and impute
tmp_df1 = train_df.iloc[:, [4, 14, 5]]
tmp_df2 = test_df.iloc[:, [3, 13, 4]]
tmp_df = pd.concat([tmp_df1, tmp_df2], axis = 0)
tmp_df.dropna(inplace=True)

'''
tmp_df[['Title', 'Sex', 'Age']].groupby(['Title', 'Sex'], as_index=False).mean()
above o/p looks is. Using it manually to impute values as not able to write python code

    Title     Sex        Age
0    Capt    male  70.000000
1  Master    male   5.482642
2      Mr    male  32.722682
3      Ms  female  28.687088
'''
#if sex= female & title is Ms, then impute age as 29.68
train_df['Age'] = np.where(((train_df.Age.isnull()) & 
        (train_df['Sex'] == 'female') & (train_df['Title'] == 'Ms')), 29.68, train_df['Age'])
test_df['Age'] = np.where(((test_df.Age.isnull()) & 
        (test_df['Sex'] == 'female') & (test_df['Title'] == 'Ms')), 29.68, test_df['Age'])

#if sex= male & title is Capt, then impute age as 70
train_df['Age'] = np.where(((train_df.Age.isnull()) & 
        (train_df['Sex'] == 'male') & (train_df['Title'] == 'Capt')), 70, train_df['Age'])
test_df['Age'] = np.where(((test_df.Age.isnull()) & 
        (test_df['Sex'] == 'male') & (test_df['Title'] == 'Capt')), 70, test_df['Age'])


#if sex= male & title is Mr, then impute age as 32.72
train_df['Age'] = np.where(((train_df.Age.isnull()) & 
        (train_df['Sex'] == 'male') & (train_df['Title'] == 'Mr')), 32.72, train_df['Age'])
test_df['Age'] = np.where(((test_df.Age.isnull()) & 
        (test_df['Sex'] == 'male') & (test_df['Title'] == 'Mr')), 32.72, test_df['Age'])


#if sex= male & title is Master, then impute age as 5.48
train_df['Age'] = np.where(((train_df.Age.isnull()) & 
        (train_df['Sex'] == 'male') & (train_df['Title'] == 'Master')), 5.48, train_df['Age'])
test_df['Age'] = np.where(((test_df.Age.isnull()) & 
        (test_df['Sex'] == 'male') & (test_df['Title'] == 'Master')), 5.48, test_df['Age'])

train_df = pd.get_dummies(train_df, prefix="G", columns=["Sex"])
test_df = pd.get_dummies(test_df, prefix="G", columns=["Sex"])

train_df = pd.get_dummies(train_df, columns=["Title"])
test_df = pd.get_dummies(test_df, columns=["Title"])

X = train_df.loc[:, ['Pclass', 'Age', 'FamilySize', 'IsAlone', 'G_female', 'G_male', 'Title_Master', 'Title_Mr', 'Title_Ms']].values
y = train_df.loc[:,['Survived']].values

#to be used in submission
X_test_df = test_df.loc[:, ['Pclass', 'Age', 'FamilySize', 'IsAlone', 'G_female', 'G_male', 'Title_Master', 'Title_Mr', 'Title_Ms']].values

# Splitting the dataset into the Training set and Test set from train_df itself
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#test.csv data
X_test_df = sc.transform(X_test_df)

#-------------------- Logistic Regression ------------------------------
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression()
classifier_lr.fit(X_train, y_train)


# Predicting the Test set results
y_pred_lr = classifier_lr.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm_lr = confusion_matrix(y_test, y_pred_lr)
ac_lr = accuracy_score(y_test, y_pred_lr)
print(classification_report(y_test, y_pred_lr))


#-------------------- KNN ------------------------------
# Fitting KNN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, p = 2)
classifier_knn.fit(X_train, y_train)

# Predicting the Test set results
y_pred_knn = classifier_knn.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm_knn = confusion_matrix(y_test, y_pred_knn)
ac_knn = accuracy_score(y_test, y_pred_knn)
print(classification_report(y_test, y_pred_knn))

#-------------------- Decision Tree ------------------------------
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier()
classifier_dt.fit(X_train, y_train)


# Predicting the Test set results
y_pred_dt = classifier_dt.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm_dt = confusion_matrix(y_test, y_pred_dt)
ac_dt = accuracy_score(y_test, y_pred_dt)
print(classification_report(y_test, y_pred_dt))

#-------------------- Random Forest ------------------------------
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators=100)
classifier_rf.fit(X_train, y_train)


# Predicting the Test set results
y_pred_rf = classifier_rf.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm_rf = confusion_matrix(y_test, y_pred_rf)
ac_rf = accuracy_score(y_test, y_pred_rf)
print(classification_report(y_test, y_pred_rf))

#-------------------- SVM ------------------------------
# Fitting SVM to the Training set
from sklearn.svm import SVC, LinearSVC
classifier_svc = SVC()
classifier_svc.fit(X_train, y_train)

# Predicting the Test set results
y_pred_svc = classifier_knn.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
cm_svc = confusion_matrix(y_test, y_pred_svc)
ac_svc = accuracy_score(y_test, y_pred_svc)
print(classification_report(y_test, y_pred_svc))


score_dict = {"Random Forest Score": round(ac_rf*100, 2),
              "Decision Tree Score": round(ac_dt*100, 2),
              "KNN Score": round(ac_knn*100, 2),
              "Logistic Regression Score": round(ac_lr*100, 2),
              "SVC": round(ac_svc*100, 2)
        }

#Improving model performance
#-------------------- Applying k-fold cross validation ------------------------
from sklearn.model_selection import cross_val_score
# -- for logistic regression
accuracies_lr = cross_val_score(classifier_lr, X = X_train, y = y_train, cv=10)
accuracies_lr.mean()
accuracies_lr.std()
# -- for SVC
accuracies_svc = cross_val_score(classifier_svc, X = X_train, y = y_train, cv=10)
accuracies_svc.mean()
accuracies_svc.std()
# -- for random forest regression
accuracies_rf = cross_val_score(classifier_rf, X = X_train, y = y_train, cv=10)
accuracies_rf.mean()
accuracies_rf.std()

#Applying grid search to find best model and best parameters
#SVC
'''
from sklearn.model_selection import GridSearchCV
parameters_svc = [{'C':[1, 10, 100, 1000], 'kernel': ['linear']},
                  {'C':[1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
                 ]

grid_search_svc = GridSearchCV(estimator = classifier_svc,
                                param_grid = parameters_svc,
                                scoring = 'accuracy',
                                cv = 10,
                                n_jobs = -1
                                )
grid_search_svc = grid_search_svc.fit(X_train, y_train) 
best_accuracy_svc = grid_search_svc.best_score_
best_params_svc = grid_search_svc.best_params_

# Fitting Final Model on training set
from sklearn.svm import SVC, LinearSVC
classifier_svc = SVC(C = 1, kernel = 'rbf', gamma = 0.4, random_state = 0)
classifier_svc.fit(X_train, y_train)
cm_svc_tuned = confusion_matrix(y_test, y_pred_svc)
ac_svc_tuned = accuracy_score(y_test, y_pred_svc)
print(classification_report(y_test, y_pred_svc))'''
#this gives no improvement, so not using it

#Random Forest
parameters_rf = {"n_estimators": [10, 20, 30, 50, 100, 300, 500, 1000],
              "max_depth": [1, 3, 5],
              "min_samples_split": [5, 10, 15],
              "min_samples_leaf": [5, 10, 15],
              "min_weight_fraction_leaf": [0.1, 0.05, 0.005]}

grid_search_rf = GridSearchCV(estimator = classifier_rf,
                                param_grid = parameters_rf,
                                scoring = 'accuracy',
                                cv = 10,
                                n_jobs = -1
                                )
grid_search_rf = grid_search_rf.fit(X_train, y_train) 
best_accuracy_rf = grid_search_rf.best_score_
best_params_rf = grid_search_rf.best_params_

# Fitting Final Model on training set
classifier_rf_tuned = RandomForestClassifier(random_state = 0)
classifier_rf_tuned.fit(X_train, y_train)
cm_rf_tuned = confusion_matrix(y_test, y_pred_rf)
ac_rf_tuned = accuracy_score(y_test, y_pred_rf)
print(classification_report(y_test, y_pred_rf))



#Since lr score is best, using it to derive y_pred for test_df
y_pred_final = classifier_lr.predict(X_test_df)

submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": y_pred_final})
submission.to_csv("Titanic_Baseline_Model_Submission.csv", index=False)

