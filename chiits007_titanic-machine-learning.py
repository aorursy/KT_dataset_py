# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import seaborn as sns # Good visualization library

import matplotlib.pyplot as plt # Basic visualisation library

%matplotlib inline



from sklearn.preprocessing import LabelEncoder # For converting the objects into numbers

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix

import xgboost

# Any results you write to the current directory are saved as output.
raw_train=pd.read_csv('../input/train.csv', index_col = 'PassengerId')

raw_test=pd.read_csv('../input/test.csv', index_col = 'PassengerId')
raw_train.head()
raw_train.shape
raw_train.dtypes  # Finding the datatypes of the dataset
raw_train.isnull().sum() # Finding the number of null values in the dataset
# Cleaning of dataframe given to us



# Filling the null values of embarked columns with the mode

def fill_embarked_na(df):

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace= True)



# Filling the null values of cabin column with the N    

def fill_cabin_na(df):

    df['Cabin'].fillna('No_Cabin'[0], inplace= True)

        

# Filling the age column with the median grouped by sex

def fill_age_na(df):

    df.loc[df.Age.isnull(), 'Age'] = df.groupby('Sex').Age.transform('median')

        

# Making a new column family members

def family_members(df):

    df['Family_members'] = df['Parch'] + df['SibSp'] + 1

        

def modify_sex_column(df):

    df["Sex"] = df["Sex"].apply(lambda sex: 1 if sex == 'male' else 0)

    

def is_alone(df):

    df['Is_alone'] = 0

    df["Is_alone"].loc[df['Family_members'] < 2] = 1

    

def has_cabin(df):

    df['Has_cabin'] = 1

    df["Has_cabin"].loc[df["Cabin"] != "N"] = 0
# Calling all the functions made to clean the data



fill_embarked_na(raw_train)

fill_cabin_na(raw_train)

fill_age_na(raw_train)

family_members(raw_train)

is_alone(raw_train)

modify_sex_column(raw_train)

has_cabin(raw_train)
raw_train.isnull().sum()   # Checking the dataframe once more
raw_train.head()
name = raw_train.Name

def include_title(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'

titles = sorted(set([x for x in raw_train.Name.map(lambda x: include_title(x))]))



def replace_titles(x):

    title = x['Title']

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:

        return 'Mr'

    elif title in ['the Countess', 'Mme', 'Lady']:

        return 'Mrs'

    elif title in ['Mlle', 'Ms']:

        return 'Miss'

    elif title =='Dr':

        if x['Sex']=='male':

            return 'Mr'

        else:

            return 'Mrs'

    else:

        return title

raw_train['Title'] = raw_train['Name'].map(lambda x: include_title(x))
raw_train['Title'] = raw_train.apply(replace_titles, axis=1)
raw_train.head()
raw_train.Title.unique()
raw_train.dtypes
raw_train.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis = 1, inplace = True)

x = raw_train.columns

x
# Label Encoding

le = LabelEncoder()



def label_encoding(df):

    df["Embarked"] = le.fit_transform(df["Embarked"])

    df["Title"] = le.fit_transform(df["Title"])
label_train = raw_train
raw_test.head()
raw_test.isnull().sum()
# Calling all the functions made to clean the test data



fill_embarked_na(raw_test)

fill_cabin_na(raw_test)

fill_age_na(raw_test)

family_members(raw_test)

is_alone(raw_test)

modify_sex_column(raw_test)

has_cabin(raw_test)

#raw_test['Title'] = raw_test.apply(replace_titles, axis=1)
name = raw_test.Name

titles = sorted(set([x for x in raw_test.Name.map(lambda x: include_title(x))]))

raw_test['Title'] = raw_test['Name'].map(lambda x: include_title(x))

raw_test['Title'] = raw_test.apply(replace_titles, axis=1)
label_encoding(label_train)

label_train.head()
raw_test.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis = 1, inplace = True)

raw_test["Fare"].fillna(raw_test["Fare"].mean(), inplace=True)

print(raw_test.isnull().sum())

label_test = raw_test

label_encoding(label_test)

label_test.head()
plt.figure(figsize = (15,15))

sns.heatmap(label_train.corr(), linewidths = .5, annot= True, cmap="YlGnBu")
x = label_train.drop('Survived', axis =1)

y = label_train['Survived']

x_label_train, x_label_test, y_label_train, y_label_test = train_test_split(x, y, test_size=0.20)
# Logistic Regressiom Model

lr = LogisticRegression(C = 1)

print(lr.fit(x_label_train, y_label_train))
print("Accuracy of Logit Model on train:",accuracy_score(y_label_train, lr.predict(x_label_train)))

print("Accuracy of Logit Model on test:",accuracy_score(y_label_test, lr.predict(x_label_test)))
score_lr = cross_val_score(lr, x.values, y.values, cv=5)

print("Cross Validation score : " + str(score_lr))

print("Cross Validation Mean score : " + str(score_lr.mean()))
# Decision Tree Classifier Model

tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11)

print(tree.fit(x_label_train, y_label_train))
print("Accuracy of Decision Tree Model on train:",accuracy_score(y_label_train, tree.predict(x_label_train)))

print("Accuracy of Decision Tree Model on test:",accuracy_score(y_label_test, tree.predict(x_label_test)))
score_tree = cross_val_score(tree, x.values, y.values, cv=5)

print("Cross Validation score : " + str(score_tree))

print("Cross Validation Mean score : " + str(score_tree.mean()))
# Random Forest Classifier Model

forest = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', max_depth = 5, 

                                min_samples_split = 3)

print(forest.fit(x_label_train, y_label_train))
print("Accuracy of Random Forest Model on train:",accuracy_score(y_label_train, forest.predict(x_label_train)))

print("Accuracy of Random Forest Model on test:",accuracy_score(y_label_test, forest.predict(x_label_test)))
# Using confusion matrix to validate the model

conf = confusion_matrix(y_label_test, forest.predict(x_label_test))

sns.heatmap(conf, linewidths = 0.5, annot=True)

plt.xlabel('Predicted')

plt.ylabel('Actual')
print(forest.predict(label_test))

forest_pred= pd.DataFrame({'PassengerId' : np.arange(892,1310), 'Survived': forest.predict(label_test)})
#forest_pred.to_csv('Titanic_Submission_Random_forrest.csv', index=False)
score_forest = cross_val_score(forest, x.values, y.values, cv=5)

print("Cross Validation score : " + str(score_forest))

print("Cross Validation Mean score : " + str(score_forest.mean()))
# Extra Trees Classifier Model

extra_tree = ExtraTreesClassifier(n_estimators = 1000, criterion = 'entropy', max_depth = 5, 

                                  min_samples_split = 3)

print(extra_tree.fit(x_label_train, y_label_train))
print("Accuracy of Extra Tree Model on train:",accuracy_score(y_label_train, extra_tree.predict(x_label_train)))

print("Accuracy of Extra Tree Model on test:",accuracy_score(y_label_test, extra_tree.predict(x_label_test)))
score_extra_tree = cross_val_score(extra_tree, x.values, y.values, cv=5)

print("Cross Validation score : " + str(score_extra_tree))

print("Cross Validation Mean score : " + str(score_extra_tree.mean()))
# KNN Model

knn = KNeighborsClassifier(n_neighbors = 29, weights ='distance')

print(knn.fit(x_label_train, y_label_train))
print("Accuracy of KNN Model on train:",accuracy_score(y_label_train, knn.predict(x_label_train)))

print("Accuracy of KNN Model on test:",accuracy_score(y_label_test, knn.predict(x_label_test)))
score_knn = cross_val_score(knn, x.values, y.values, cv=5)

print("Cross Validation score : " + str(score_knn))

print("Cross Validation Mean score : " + str(score_knn.mean()))
knn2 = KNeighborsClassifier(n_neighbors = 15, weights ='distance')

print(knn2.fit(x_label_train, y_label_train))

print("Accuracy of KNN Model on train:",accuracy_score(y_label_train, knn2.predict(x_label_train)))

print("Accuracy of KNN Model on test:",accuracy_score(y_label_test, knn2.predict(x_label_test)))
score_knn2 = cross_val_score(knn2, x.values, y.values, cv=5)

print("Cross Validation score : " + str(score_knn2))

print("Cross Validation Mean score : " + str(score_knn2.mean()))
# SVM Model

svc = SVC(C=1, kernel = 'sigmoid')

print(svc.fit(x_label_train, y_label_train))
print("Accuracy of SVM Model on train:",accuracy_score(y_label_train, svc.predict(x_label_train)))

print("Accuracy of SVM Model on test:",accuracy_score(y_label_test, svc.predict(x_label_test)))
score_svc = cross_val_score(svc, x.values, y.values, cv=5)

print("Cross Validation score : " + str(score_svc))

print("Cross Validation Mean score : " + str(score_svc.mean()))
#Adaboost Model

ada = AdaBoostClassifier(learning_rate = 0.1)

print(ada.fit(x_label_train, y_label_train))
print("Accuracy of SVM Model on train:",accuracy_score(y_label_train, ada.predict(x_label_train)))

print("Accuracy of SVM Model on test:",accuracy_score(y_label_test, ada.predict(x_label_test)))
score_ada = cross_val_score(ada, x.values, y.values, cv=5)

print("Cross Validation score : " + str(score_ada))

print("Cross Validation Mean score : " + str(score_ada.mean()))
# Gradient Boosting Algorithm

gbc = GradientBoostingClassifier()

print(gbc.fit(x_label_train, y_label_train))
print("Accuracy of GBC Model on train:",accuracy_score(y_label_train, gbc.predict(x_label_train)))

print("Accuracy of GBC Model on test:",accuracy_score(y_label_test, gbc.predict(x_label_test)))
score_gbc = cross_val_score(gbc, x.values, y.values, cv=5)

print("Cross Validation score : " + str(score_gbc))

print("Cross Validation Mean score : " + str(score_gbc.mean()))
conf = confusion_matrix(y_label_test, gbc.predict(x_label_test))

sns.heatmap(conf, linewidths = 0.5, annot=True)

plt.xlabel('Predicted')

plt.ylabel('Actual')
#Predicting the test_label model on Gradient Boosting Model

gbc.predict(label_test)

gbc_pred= pd.DataFrame({'PassengerId' : np.arange(892,1310), 'Survived': gbc.predict(label_test)})

gbc_pred.to_csv('Titanic_Submission_GBC.csv', index=False)
# Making another set of models based on dummy encoding of the dataset

dummy_train = raw_train

dummy_test = raw_test
dummy_train.head()
pclass_dummy = pd.get_dummies(dummy_train.Pclass, prefix = 'Pclass')

pclass_dummy.drop(pclass_dummy.columns[0], axis=1, inplace=True)



embarked_dummy = pd.get_dummies(dummy_train.Embarked, prefix = 'Embarked')

embarked_dummy.drop(embarked_dummy.columns[0], axis=1, inplace=True)



family_dummy = pd.get_dummies(dummy_train.Family_members, prefix = 'Family_members')

family_dummy.drop(family_dummy.columns[0], axis=1, inplace=True)



title_dummy = pd.get_dummies(dummy_train.Pclass, prefix = 'Title')

title_dummy.drop(title_dummy.columns[0], axis=1, inplace=True)



dummy_train = pd.concat([dummy_train, pclass_dummy, embarked_dummy, title_dummy], axis=1)

dummy_train.drop(['Pclass', 'Embarked', 'Family_members', 'Title'], axis = 1, inplace= True)
dummy_train.head()
# Train_test_split on dummy model

x_dummy = dummy_train.drop('Survived', axis = 1)

y_dummy = dummy_train.Survived

x_dummy_train, x_dummy_test, y_dummy_train, y_dummy_test = train_test_split(x_dummy, y_dummy, test_size = 0.3)
# Logistic Regression Model on dummy model

lr_dummy = LogisticRegression(C=1)

lr_dummy.fit(x_dummy_train, y_dummy_train)

print("Accuracy of Logit Model on dummy_train:",accuracy_score(y_dummy_train, lr_dummy.predict(x_dummy_train)))

print("Accuracy of Logit Model on dummy_test:",accuracy_score(y_dummy_test, lr_dummy.predict(x_dummy_test)))



score_lr_dummy = cross_val_score(lr_dummy, x_dummy.values, y_dummy.values, cv=5)

print("Cross Validation score : " + str(score_lr_dummy))

print("Cross Validation Mean score : " + str(score_lr_dummy.mean()))
# Decision Tree Model for Dummy dataset

tree_dummy = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11)

tree_dummy.fit(x_dummy_train, y_dummy_train)

print("Accuracy of Logit Model on dummy_train:",accuracy_score(y_dummy_train, tree_dummy.predict(x_dummy_train)))

print("Accuracy of Logit Model on dummy_test:",accuracy_score(y_dummy_test, tree_dummy.predict(x_dummy_test)))



score_tree_dummy = cross_val_score(tree_dummy, x_dummy.values, y_dummy.values, cv=5)

print("Cross Validation score : " + str(score_tree_dummy))

print("Cross Validation Mean score : " + str(score_tree_dummy.mean()))
# Random Forest Model for dummy dataset

forest_dummy = RandomForestClassifier(criterion = 'entropy', n_estimators = 1000, max_depth = 5,

                                     min_samples_split = 3)

forest_dummy.fit(x_dummy_train, y_dummy_train)

print("Accuracy of Logit Model on dummy_train:",accuracy_score(y_dummy_train, forest_dummy.predict(x_dummy_train)))

print("Accuracy of Logit Model on dummy_test:",accuracy_score(y_dummy_test, forest_dummy.predict(x_dummy_test)))



score_forest_dummy = cross_val_score(forest_dummy, x_dummy.values, y_dummy.values, cv=5)

print("Cross Validation score : " + str(score_forest_dummy))

print("Cross Validation Mean score : " + str(score_forest_dummy.mean()))
# Extra Trees Model for the dummy dataset

extra_dummy = ExtraTreesClassifier(criterion = 'entropy', n_estimators = 1000, max_depth = 5,

                                     min_samples_split = 3)

extra_dummy.fit(x_dummy_train, y_dummy_train)

print("Accuracy of Logit Model on dummy_train:",accuracy_score(y_dummy_train, extra_dummy.predict(x_dummy_train)))

print("Accuracy of Logit Model on dummy_test:",accuracy_score(y_dummy_test, extra_dummy.predict(x_dummy_test)))



score_extra_dummy = cross_val_score(extra_dummy, x_dummy.values, y_dummy.values, cv=5)

print("Cross Validation score : " + str(score_extra_dummy))

print("Cross Validation Mean score : " + str(score_extra_dummy.mean()))

#KNN Model for the dummy model

knn_dummy= KNeighborsClassifier(n_neighbors = 29, weights = 'distance')

knn_dummy.fit(x_dummy_train, y_dummy_train)

print("Accuracy of Logit Model on dummy_train:",accuracy_score(y_dummy_train, knn_dummy.predict(x_dummy_train)))

print("Accuracy of Logit Model on dummy_test:",accuracy_score(y_dummy_test, knn_dummy.predict(x_dummy_test)))



score_knn_dummy = cross_val_score(knn_dummy, x_dummy.values, y_dummy.values, cv=5)

print("Cross Validation score : " + str(score_knn_dummy))

print("Cross Validation Mean score : " + str(score_knn_dummy.mean()))
#SVC Model for the dummy model

svc_dummy = SVC(C = 1, kernel = 'sigmoid')

svc_dummy.fit(x_dummy_train, y_dummy_train)

print("Accuracy of Logit Model on dummy_train:",accuracy_score(y_dummy_train, svc_dummy.predict(x_dummy_train)))

print("Accuracy of Logit Model on dummy_test:",accuracy_score(y_dummy_test, svc_dummy.predict(x_dummy_test)))



score_svc_dummy = cross_val_score(svc_dummy, x_dummy.values, y_dummy.values, cv=5)

print("Cross Validation score : " + str(score_svc_dummy))

print("Cross Validation Mean score : " + str(score_svc_dummy.mean()))