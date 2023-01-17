#Machine Learning Packages

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from xgboost import XGBClassifier

from sklearn import model_selection



#Data Processing Packages

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection



#Data Visualization Packages

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Metrics

from sklearn import metrics



#Data Analysis Packages

import pandas as pd

import numpy as np



#Ignore Warnings

import warnings

warnings.filterwarnings('ignore')
#Loading the data sets

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

target = train["Survived"]

data = pd.concat([train.drop("Survived", axis=1),test], axis=0).reset_index(drop=True)
data.head()
data.info()
data.isnull().sum()
#Plotting the relations between Age and other features (Mean).

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,12))



#Age vs Pclass

sns.barplot(x="Pclass", y="Age", data=data, ax=ax[0,0])

#Age vs Sex

sns.barplot(x="Sex", y="Age", data=data, ax=ax[0,1])

#Age vs SibSp

sns.barplot(x="SibSp", y="Age", data=data, ax=ax[0,2])

#Age vs Parch

sns.barplot(x="Parch", y="Age", data=data, ax=ax[1,0])

#Age vs Family_size

sns.barplot(x=(data["Parch"] + data["SibSp"]), y="Age", data=data, ax=ax[1,1])

ax[1,1].set(xlabel='Family Size')

#Age vs Embarked

sns.barplot(x="Embarked", y="Age", data=data, ax=ax[1,2])
#Plotting relations between Age and other features (Median).



fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,12))

#Age vs Pclass

sns.boxplot(x="Pclass", y="Age", data=data, ax=ax[0,0])

#Age vs Sex

sns.boxplot(x="Sex", y="Age", data=data, ax=ax[0,1])

#Age vs SibSp

sns.boxplot(x="SibSp", y="Age", data=data, ax=ax[0,2])

#Age vs Parch

sns.boxplot(x="Parch", y="Age", data=data, ax=ax[1,0])

#Age vs Family_size

sns.boxplot(x=(data["Parch"] + data["SibSp"]), y="Age", data=data, ax=ax[1,1])

ax[1,1].set(xlabel='Family Size')

#Age vs Embarked

sns.boxplot(x="Embarked", y="Age", data=data, ax=ax[1,2])
fig, ax = plt.subplots(figsize=(10,7))



#Relation between Pclass and Embarked

sns.countplot(x="Pclass", data=data, hue="Embarked")
# We will use Pclass, Family Size and Embarked features to fill in the missing age values



# First Lets create the feature Family_Size

data["Family_Size"] = data["SibSp"] + data["Parch"]



#Filling in the missing Age values

missing_age_value = data[data["Age"].isnull()]

for index, row in missing_age_value.iterrows():

    median = data["Age"][(data["Pclass"] == row["Pclass"]) & (data["Embarked"] == row["Embarked"]) & (data["Family_Size"] == row["Family_Size"])].median()

    if not np.isnan(median):

        data["Age"][index] = median

    else:

        data["Age"][index] = np.median(data["Age"].dropna())
#Relation Between Fare and Pclass

fig, ax = plt.subplots(figsize=(7,5))

sns.boxplot(x="Pclass", y="Fare", data=data)



#Since we have only 1 missing value for Fare we can just fill it according to Pclass feature

print("Pclass of the data point with missing Fare value:", int(data[data["Fare"].isnull()]["Pclass"]))

median = data[data["Pclass"] == 3]["Fare"].median()

data["Fare"].fillna(median, inplace=True)
for index, rows in data.iterrows():

    if pd.isnull(rows["Cabin"]):

        data["Cabin"][index] = 'X'

    else:

        data["Cabin"][index] = str(rows["Cabin"])[0]
#Since we only have 2 missing Embarked values we will just fill the missing values with mode.

data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,5))



#Survived vs Pclass

sns.barplot(x="Pclass", y=target, data=data[:891], ax=ax[0])



#Survived vs Sex

sns.barplot(x="Sex", y=target, data=data[:891], ax=ax[1])
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,5))



#Survived vs Family_Size

sns.barplot(x="Family_Size", y=target, data=data[:891], ax=ax[0])



#Sex vs Single Passengers

sns.countplot(x="Sex", data=data[data["Family_Size"] == 0], ax=ax[1])



#Dividing Family_Size into 3 groups

data["Family_Size"] = data["Family_Size"].map({0:0, 1:1, 2:1, 3:1, 4:2, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2})
data.head()
plt.figure(figsize=(12,7))



#Survived vs Cabin

plt.subplot(121)

sns.barplot(x="Cabin", y=target, data=data[:891])



#Survived vs Embarked

plt.subplot(122)

sns.barplot(x="Embarked", y=target, data=data[:891])
plt.figure(figsize=(12,7))



#Plotting Kde for Fare

plt.subplot(121)

sns.kdeplot(data["Fare"])



#Plotting Kde for Fare with Survived as hue

plt.subplot(122)

sns.kdeplot(np.log(data[:891][target == 1]["Fare"]), color='blue', shade=True)

sns.kdeplot(np.log(data[:891][target == 0]["Fare"]), color='red', shade=True)

plt.legend(["Survived", "Not Survived"])



#Since skewness can result in false conclusions we reduce skew for fare by taking log.

data["Fare"] = np.log(data["Fare"])



#Dividing Fare into different categories

data["Fare"] = pd.qcut(data["Fare"], 5)
label = LabelEncoder()

data["Age"] = label.fit_transform(pd.cut(data["Age"].astype(int), 5))

sns.barplot(x="Age", y=target, data=data[0:891])
data["Name"] = data["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())

data["Name"] = data["Name"].map({'Mr':1, 'Miss':2, 'Mrs':3, 'Ms':3, 'Mlle':3, 'Mme':3, 'Master':4, 'Dr':5, 'Rev':5, 'Col':5, "Major":5, "Dona":5, "Sir":5, "Lady":5, "Jonkheer":5, "Don":5, "the Countess":5, "Capt":5})

sns.barplot(x="Name", y=target, data=data[0:891])
data["Ticket"] = data["Ticket"].apply(lambda x: x.replace(".","").replace('/',"").strip().split(' ')[0] if not x.isdigit() else 'X')

data["Ticket"] = label.fit_transform(data["Ticket"])

sns.barplot(x="Ticket", y=target, data=data[0:891])
#OneHot encoding with pd.get_dummies

data.drop(["SibSp", "Parch"], inplace=True, axis=1)

data = pd.get_dummies(data=data, columns=["Pclass", "Name", "Sex", "Age", "Cabin", "Embarked", "Family_Size", "Ticket", "Fare"], drop_first=True)
#Splitting into train and test again

train = data[:891]

test = data[891:]
# Modeling step Test differents algorithms 

cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.7, random_state=42)



classifiers = [

    SVC(random_state=42),

    DecisionTreeClassifier(random_state=42),

    AdaBoostClassifier(DecisionTreeClassifier(random_state=42),random_state=42,learning_rate=0.1),

    RandomForestClassifier(random_state=42),

    ExtraTreesClassifier(random_state=42),

    GradientBoostingClassifier(random_state=42),

    MLPClassifier(random_state=42),

    KNeighborsClassifier(),

    LogisticRegression(random_state=42),

    LinearDiscriminantAnalysis()

]



cv_train_mean = []

cv_test_mean = []

cv_score_time = []

cv_fit_time = []

cv_name = []

predictions = []



for classifier in classifiers :

    cv_results = model_selection.cross_validate(classifier, train.drop(['PassengerId'], axis=1), target, cv=cv_split, return_train_score=True)

    cv_train_mean.append(cv_results['train_score'].mean())

    cv_test_mean.append(cv_results['test_score'].mean())

    cv_score_time.append(cv_results['score_time'].mean())

    cv_fit_time.append(cv_results['fit_time'].mean())

    cv_name.append(str(classifier.__class__.__name__))

    classifier.fit(train.drop(['PassengerId'], axis=1), target)

    predictions.append(classifier.predict(test.drop(['PassengerId'], axis=1)))

    



performance_df = pd.DataFrame({"Algorithm":cv_name, "Train Score":cv_train_mean, "Test Score":cv_test_mean, 'Score Time':cv_score_time, 'Fit Time':cv_fit_time})

performance_df
#Plotting the performance on test set

sns.barplot('Test Score', 'Algorithm', data=performance_df)
#Plotting prediction correlation of the algorithms

sns.heatmap(pd.DataFrame(predictions, index=cv_name).T.corr(), annot=True)
tuned_clf = {

    'DecisionTreeClassifier':DecisionTreeClassifier(random_state=42),

    'AdaBoostClassifier':AdaBoostClassifier(DecisionTreeClassifier(random_state=42),random_state=42,learning_rate=0.1),

    'RandomForestClassifier':RandomForestClassifier(random_state=42),

    'ExtraTreesClassifier':ExtraTreesClassifier(random_state=42),

    

    'GradientBoostingClassifier':GradientBoostingClassifier(random_state=42),

    'MLPClassifier':MLPClassifier(random_state=42),

    'LogisticRegression':LogisticRegression(random_state=42),

    'LinearDiscriminantAnalysis':LinearDiscriminantAnalysis()

}
#DecisionTreeClassifier

grid = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'], 'max_depth': [2,4,6,8,10,None], 

        'min_samples_split': [2,5,10,.03,.05], 'min_samples_leaf': [1,5,10,.03,.05], 'max_features': [None, 'auto']}



tune_model = model_selection.GridSearchCV(tuned_clf['DecisionTreeClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

tune_model.fit(train.drop(['PassengerId'], axis=1), target)

print("Best Parameters:")

print(tune_model.best_params_)

tuned_clf['DecisionTreeClassifier'].set_params(**tune_model.best_params_)
#AdaBoostClassifier

grid = {'n_estimators': [10, 50, 100, 300], 'learning_rate': [.01, .03, .05, .1, .25], 'algorithm': ['SAMME', 'SAMME.R'] }



tune_model = model_selection.GridSearchCV(tuned_clf['AdaBoostClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

tune_model.fit(train.drop(['PassengerId'], axis=1), target)

print("Best Parameters:")

print(tune_model.best_params_)

tuned_clf['AdaBoostClassifier'].set_params(**tune_model.best_params_)
#RandomForestClassifier

grid = {'n_estimators': [10, 50, 100, 300], 'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10, None], 

        'oob_score': [True] }

 

tune_model = model_selection.GridSearchCV(tuned_clf['RandomForestClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

tune_model.fit(train.drop(['PassengerId'], axis=1), target)

print("Best Parameters:")

print(tune_model.best_params_)

tuned_clf['RandomForestClassifier'].set_params(**tune_model.best_params_)
#ExtraTreesClassifier

grid = {'n_estimators': [10, 50, 100, 300], 'criterion': ['gini', 'entropy'], 'max_depth': [2, 4, 6, 8, 10, None]}

 

tune_model = model_selection.GridSearchCV(tuned_clf['ExtraTreesClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

tune_model.fit(train.drop(['PassengerId'], axis=1), target)

print("Best Parameters:")

print(tune_model.best_params_)

tuned_clf['ExtraTreesClassifier'].set_params(**tune_model.best_params_)
#GradientBoostingClassifier

grid = {#'loss': ['deviance', 'exponential'], 'learning_rate': [.01, .03, .05, .1, .25], 

        'n_estimators': [300],

        #'criterion': ['friedman_mse', 'mse', 'mae'], 

        'max_depth': [4] }



tune_model = model_selection.GridSearchCV(tuned_clf['GradientBoostingClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

tune_model.fit(train.drop(['PassengerId'], axis=1), target)

print("Best Parameters:")

print(tune_model.best_params_)

tuned_clf['GradientBoostingClassifier'].set_params(**tune_model.best_params_)
#MLPClassifier

grid = {'learning_rate': ["constant", "invscaling", "adaptive"], 'alpha': 10.0 ** -np.arange(1, 7), 'activation': ["logistic", "relu", "tanh"]}



tune_model = model_selection.GridSearchCV(tuned_clf['MLPClassifier'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

tune_model.fit(train.drop(['PassengerId'], axis=1), target)

print("Best Parameters:")

print(tune_model.best_params_)

tuned_clf['MLPClassifier'].set_params(**tune_model.best_params_)
#LogisticRegression

grid = {'fit_intercept': [True, False], 'penalty': ['l2'], 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

 

tune_model = model_selection.GridSearchCV(tuned_clf['LogisticRegression'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

tune_model.fit(train.drop(['PassengerId'], axis=1), target)

print("Best Parameters:")

print(tune_model.best_params_)

tuned_clf['LogisticRegression'].set_params(**tune_model.best_params_)
#LinearDiscriminantAnalysis

grid = {"solver" : ["svd"], "tol" : [0.0001,0.0002,0.0003]}

 

tune_model = model_selection.GridSearchCV(tuned_clf['LinearDiscriminantAnalysis'], param_grid=grid, scoring = 'roc_auc', cv = cv_split, return_train_score=True)

tune_model.fit(train.drop(['PassengerId'], axis=1), target)

print("Best Parameters:")

print(tune_model.best_params_)

tuned_clf['LinearDiscriminantAnalysis'].set_params(**tune_model.best_params_)
#Evaluating the performance of our tuned models



cv_train_mean = []

cv_test_mean = []

cv_score_time = []

cv_fit_time = []

cv_name = []

predictions = []



for _, classifier in tuned_clf.items():

    cv_results = model_selection.cross_validate(classifier, train.drop(['PassengerId'], axis=1), target, cv=cv_split, return_train_score=True)

    cv_train_mean.append(cv_results['train_score'].mean())

    cv_test_mean.append(cv_results['test_score'].mean())

    cv_score_time.append(cv_results['score_time'].mean())

    cv_fit_time.append(cv_results['fit_time'].mean())

    cv_name.append(str(classifier.__class__.__name__))

    classifier.fit(train.drop(['PassengerId'], axis=1), target)

    predictions.append(classifier.predict(test.drop(['PassengerId'], axis=1)))

    



performance_df = pd.DataFrame({"Algorithm":cv_name, "Train Score":cv_train_mean, "Test Score":cv_test_mean, 'Score Time':cv_score_time, 'Fit Time':cv_fit_time})

performance_df
sns.barplot('Test Score', 'Algorithm', data=performance_df)
#voting_1 = VotingClassifier(estimators=[

#    ('DecisionTreeClassifier', tuned_clf['DecisionTreeClassifier']), 

#    ('AdaBoostClassifier', tuned_clf['AdaBoostClassifier']),

#    ('RandomForestClassifier', tuned_clf['RandomForestClassifier']), 

#    ('ExtraTreesClassifier',tuned_clf['ExtraTreesClassifier'])], voting='soft', n_jobs=4)



#voting_1 = voting_1.fit(train.drop(['PassengerId'], axis=1), target)

#pred_1 = voting_1.predict(test.drop(['PassengerId'], axis=1))
#voting_2 = VotingClassifier(estimators=[

#    ('GradientBoostingClassifier',tuned_clf['GradientBoostingClassifier']), 

#    ('MLPClassifier',tuned_clf['MLPClassifier']), 

#    ('LogisticRegression', tuned_clf['LogisticRegression']), 

#    ('LinearDiscriminantAnalysis', tuned_clf['LinearDiscriminantAnalysis'])], voting='soft', n_jobs=4)



#voting_2 = voting_2.fit(train.drop(['PassengerId'], axis=1), target)

#pred_2 = voting_2.predict(test.drop(['PassengerId'], axis=1))
voting_3 = VotingClassifier(estimators=[

    ('DecisionTreeClassifier', tuned_clf['DecisionTreeClassifier']), 

    ('AdaBoostClassifier', tuned_clf['AdaBoostClassifier']),

    ('RandomForestClassifier', tuned_clf['RandomForestClassifier']), 

    ('ExtraTreesClassifier',tuned_clf['ExtraTreesClassifier']),

    ('GradientBoostingClassifier',tuned_clf['GradientBoostingClassifier']), 

    ('MLPClassifier',tuned_clf['MLPClassifier']), 

    ('LogisticRegression', tuned_clf['LogisticRegression']), 

    ('LinearDiscriminantAnalysis', tuned_clf['LinearDiscriminantAnalysis'])], voting='soft', n_jobs=4)



voting_3 = voting_3.fit(train.drop(['PassengerId'], axis=1), target)

pred_3 = voting_3.predict(test.drop(['PassengerId'], axis=1))
#sol1 = pd.DataFrame(data=pred_1, columns=['Survived'], index=test['PassengerId'])

#sol2 = pd.DataFrame(data=pred_2, columns=['Survived'], index=test['PassengerId'])

sol3 = pd.DataFrame(data=pred_3, columns=['Survived'], index=test['PassengerId'])
#sol1.to_csv("sol1.csv")

#sol2.to_csv("sol2.csv")

sol3.to_csv("sol3.csv")