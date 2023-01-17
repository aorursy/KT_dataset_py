import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.linear_model import LogisticRegressionCV, PassiveAggressiveClassifier, RidgeClassifierCV, SGDClassifier, Perceptron

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, NuSVC, LinearSVC



from bayes_opt import BayesianOptimization



from xgboost import XGBClassifier



from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.feature_selection import RFECV

from sklearn.metrics import accuracy_score



import warnings

warnings.filterwarnings('ignore') 



%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

train.head()
test = pd.read_csv('../input/titanic/test.csv')

test.head()
print("Train Dataset")

print("Lines: {}".format(train.shape[0]))

print("Columns: {}".format(", ".join(train.columns)))



print("\n\nTest Dataset")

print("Lines: {}".format(test.shape[0]))

print("Columns: {}".format(", ".join(test.columns)))
train.dtypes
test.dtypes
train.describe()
test.describe()
(train.isna().sum() / train.shape[0]) * 100
(test.isna().sum() / test.shape[0]) * 100
f, ax = plt.subplots(figsize = (14, 14))

colormap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(

    train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr('pearson'), 

    cmap=colormap,

    square=True, 

    cbar_kws={'shrink': .9}, 

    ax=ax,

    annot=True, 

    linewidths=0.1, vmax=1.0, linecolor='white',

    annot_kws={'fontsize': 10}

)



plt.title('Pearson Correlation of Train Dataset', y=1.05, size=15)
f, ax = plt.subplots(figsize = (14, 14))

colormap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(

    test[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr('pearson'), 

    cmap=colormap,

    square=True, 

    cbar_kws={'shrink': .9}, 

    ax=ax,

    annot=True, 

    linewidths=0.1, vmax=1.0, linecolor='white',

    annot_kws={'fontsize': 10}

)



plt.title('Pearson Correlation of Test Dataset', y=1.05, size=15)
data = [train, test]
train.drop('PassengerId', axis=1, inplace=True)

train.head()
fig, ax = plt.subplots()



labels = ['1', '2', '3']



x = np.arange(len(labels))

width = 0.35



ax.bar(x - width/2, train.Pclass.value_counts().sort_index() / train.Pclass.count(), width, label='Train', color="green")

ax.bar(x + width/2, test.Pclass.value_counts().sort_index() / test.Pclass.count(), width, label='Test', color="green", alpha=0.5)



ax.set_ylabel('% Ticket Class')

ax.set_title('Ticket Class Normalized')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()



plt.show()
fig, ax = plt.subplots()



labels = ['1', '2', '3']



x = np.arange(len(labels))

width = 0.35



ax.bar(x - width/2, train[train.Survived == 1].Pclass.value_counts().sort_index(), width, label='Survived')

ax.bar(x + width/2, train[train.Survived == 0].Pclass.value_counts().sort_index(), width, label='Died')



ax.set_ylabel('Number of Ticket Class')

ax.set_title('Ticket Class vs Survived')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()



plt.show()
le = LabelEncoder()

le.fit(train.Pclass)



for dataset in data:

    dataset.Pclass = le.transform(dataset.Pclass)
train.Pclass.value_counts()
train.head()
train.Name.head(10)
for dataset in data:

    dataset['Title'] = dataset.Name.str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
pd.crosstab(train.Title, train.Sex)
pd.crosstab(test.Title, test.Sex)
titles = ["Master", "Miss", "Mr", "Royals", "Professionals"]



for dataset in data:

    dataset.Title = dataset.Title.replace(['Lady', 'the Countess', 'Countess', 'Don', 'Jonkheer', 'Dona', 'Sir'], 'Royals')

    dataset.Title = dataset.Title.replace(['Col', 'Dr', 'Major', 'Capt'], 'Professionals')

    dataset.Title = dataset.Title.replace(["Ms", "Mme", "Mlle", "Mrs"], 'Miss')

    dataset.Title = dataset.Title.replace(['Master', 'Rev'], 'Mas/Rev')

    dataset.Title = dataset.Title.map({"Mas/Rev": 0, "Miss": 1, "Mr": 2, "Royals": 3, "Professionals": 4})
train.Title.value_counts()
sns.factorplot(x="Title", y="Survived", data=train, kind="bar").set_xticklabels(titles).set_ylabels("Survival Probability")
for dataset in data:

    dataset.drop('Name', axis=1, inplace=True)
h = sns.FacetGrid(train, row="Title", hue="Survived")

h.map(plt.hist, 'Age', alpha=.75)

h.add_legend()
train.head()
fig, ax = plt.subplots()



labels = ['female', 'male']



x = np.arange(len(labels))

width = 0.35



ax.bar(x - width/2, train.Sex.value_counts().sort_index() / train.Sex.count(), width, label='Train', color="green")

ax.bar(x + width/2, test.Sex.value_counts().sort_index() / test.Sex.count(), width, label='Test', color="green", alpha=0.5)



ax.set_ylabel('% Sex')

ax.set_title('Sex Normalized')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()



plt.show()
sns.factorplot(x="Sex", y="Survived", data=train, kind="bar").set_ylabels("Survival Probability")
h = sns.FacetGrid(train, row='Sex', col='Pclass', hue='Survived')

h.map(plt.hist, 'Age', alpha=.75)

h.add_legend()
for dataset in data:

    dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1})
train.head()
train.Age.isna().sum() / train.shape[0]
test.Age.isna().sum() / test.shape[0]
min(train.Age), max(train.Age)
min(test.Age), max(test.Age)
a = sns.FacetGrid(train, hue='Survived', aspect=4)

a.map(sns.kdeplot, 'Age', shade=True)

a.set(xlim=(0, train['Age'].max()))

a.add_legend()
g = sns.factorplot(y="Age",x="Sex",data=train, kind="box")

g = sns.factorplot(y="Age",x="Pclass", data=train, kind="box")

g = sns.factorplot(y="Age",x="Parch", data=train, kind="box")

g = sns.factorplot(y="Age",x="SibSp", data=train, kind="box")

g = sns.factorplot(y="Age",x="Title", data=train, kind="box")
train[train.Age.isna()].Title.value_counts()
def fill_age_missing_values(df):

    age_null = list(df[df["Age"].isnull()].index)



    for index in age_null:

        temp_Pclass = df.iloc[index]["Pclass"]

        temp_SibSp = df.iloc[index]["SibSp"]

        temp_Parch = df.iloc[index]["Parch"]

        temp_Title = df.iloc[index]["Title"]

        

        age_median = df["Age"][(

            (df["Pclass"] == temp_Pclass) & 

            (df["Pclass"] == temp_Pclass) & 

            (df["SibSp"] == temp_SibSp) & 

            (df["Parch"] == temp_Parch) & 

            (df["Title"] == temp_Title)

        )].median()

        

        df["Age"].iloc[index] = age_median if (df.iloc[index]["Age"] == True) and (np.isnan(age_median) == False) else df["Age"].median()

    return df
for dataset in data:

    dataset = fill_age_missing_values(dataset)
train.Age.isna().sum(), test.Age.isna().sum()
train[train.Age < 1].Survived.mean()
age_groups = ['Baby',  'Child', 'Young Adult', 'Adult', 'Senior']



for dataset in data:

    dataset.loc[(dataset['Age'] <= 2), 'Age Group'] = 0

    dataset.loc[((dataset["Age"] > 2) & (dataset['Age'] <= 10)), 'Age Group'] = 1 

    dataset.loc[((dataset["Age"] > 10) & (dataset['Age'] <= 19)), 'Age Group'] = 2

    dataset.loc[((dataset["Age"] > 19) & (dataset['Age'] <= 60)), 'Age Group'] = 3

    dataset.loc[(dataset["Age"] > 60), 'Age Group'] = 4

    

    dataset["Age"] = dataset["Age"].astype(int)

    dataset["Age Group"] = dataset["Age Group"].astype(int)
sns.factorplot(x="Age Group", y="Survived", data=train, kind="bar").set_xticklabels(age_groups).set_ylabels("Survival Probability")
train.head()
sns.factorplot(x="SibSp", y="Survived", data=train, kind="bar").set_ylabels("Survival Probability")
train[train.SibSp > 4].Survived.mean()
train[train.SibSp > 4]
sns.factorplot(x="Parch", y="Survived", data=train, kind="bar").set_ylabels("Survival Probability")
train[train.Parch > 5].Survived.mean()
train[train.Parch > 5]
for dataset in data:

    dataset['Family'] = dataset.SibSp + dataset.Parch
sns.factorplot(x="Family", y="Survived", data=train, kind="bar").set_ylabels("Survival Probability")
train.head()
test.head()
test.Fare.isna().sum()
test[test.Fare.isna()]
test.loc[test.Fare.isna(), 'Fare'] = test[test.Pclass == 3].Fare.mean()
sns.distplot(train["Fare"], color="m", label="Skewness : %.2f"%(train["Fare"].skew())).legend(loc="best")
sns.distplot(test["Fare"], color="m", label="Skewness : %.2f"%(test["Fare"].skew())).legend(loc="best")
min(train.Fare), min(test.Fare)
for dataset in data:

    dataset["Fare"] = dataset["Fare"].map(lambda x: np.log(x) if x > 0 else 0)
sns.distplot(train["Fare"], color="g", label="Skewness : %.2f"%(train["Fare"].skew())).legend(loc="best")
sns.distplot(test["Fare"], color="g", label="Skewness : %.2f"%(test["Fare"].skew())).legend(loc="best")
train.head()
train.Cabin.isna().sum() / train.shape[0]
for dataset in data:

    dataset.Cabin.fillna("U", inplace=True)

    dataset.Cabin = dataset.Cabin.apply(lambda x: x.split(" ")[-1][0] if len(x) > 0 else "U")
train.Cabin.value_counts()
test.Cabin.value_counts()
print("Survival rate (%)")

print("With Cabin: {:.2f}".format(train[train.Cabin != "U"].Survived.sum() / train[train.Cabin != "U"].shape[0]))

print("Without Cabin: {:.2f}".format(train[train.Cabin == "U"].Survived.sum() / train[train.Cabin == "U"].shape[0]))
h = sns.FacetGrid(train[train.Cabin != "U"])

h.map(plt.hist, "Pclass", alpha=0.75)
sns.factorplot(x="Cabin", y="Survived", data=train, kind="bar").set_ylabels("Survival Probability")
for dataset in data:

    dataset.Cabin = dataset.Cabin.map({'U': 0, 'T': 0, 'G': 1, 'A': 2, 'C': 3, 'B': 4, 'D': 5, 'E': 6, 'F': 7})
sns.factorplot(x="Cabin", y="Survived", data=train, kind="bar").set_ylabels("Survival Probability")
train.head()
train.Embarked.isna().sum()
train[train.Embarked.isna()]
train.loc[train.Embarked.isna(), 'Embarked'] = train.Embarked.mode()[0]
sns.factorplot(x="Embarked", y="Survived", data=train, kind="bar").set_ylabels("Survival Probability")
sns.factorplot("Pclass", col="Embarked",  data=train, size=6, kind="count")
for dataset in data:

    dataset.Embarked = dataset.Embarked.map({"C": 2, "Q": 1, "S": 0})
train.head()
for dataset in data:

    dataset.drop(columns=['Parch', 'SibSp', 'Ticket', 'Age'], inplace=True)
train.head()
train.Survived.value_counts()
train.Survived.mean()
f, ax = plt.subplots(figsize = (15, 15))

colormap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(

    train.corr('pearson'), 

    cmap=colormap,

    square=True, 

    cbar_kws={'shrink': .8}, 

    ax=ax,

    annot=True, 

    linewidths=0.1, vmax=1.0, linecolor='white',

    annot_kws={'fontsize': 9}

)



plt.title('Pearson Correlation of Features', y=1.05, size=15)
train = pd.get_dummies(train, columns = ["Title"])

train = pd.get_dummies(train, columns = ["Embarked"])

train = pd.get_dummies(train, columns = ["Pclass"])

train = pd.get_dummies(train, columns = ["Age Group"])
train.head()
test = pd.get_dummies(test, columns = ["Title"])

test = pd.get_dummies(test, columns = ["Embarked"])

test = pd.get_dummies(test, columns = ["Pclass"])

test = pd.get_dummies(test, columns = ["Age Group"])
test.head()
preds = list(train.columns)

preds.remove('Survived')

preds.remove('Cabin')

preds
X_train = train[preds]

y_train = train.Survived



X_test = test[preds]
scaler = StandardScaler()

scaler.fit(X_train)



X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)

X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)



# X_train_scaled = X_train

# X_test_scaled = X_test
importances = []

for i in range(10):

    rf = RandomForestClassifier()

    rf.fit(X_train_scaled, y_train)

    if len(importances) > 0:

        importances = [x + y for x, y in zip(importances, rf.feature_importances_)]

    else:

        importances = rf.feature_importances_



importances = [x / 10 for x in importances]
importances = pd.DataFrame({'feature': preds, 'importance':importances})

importances
importances.sort_values('importance', ascending=False, inplace=True)
acc = []

for i in importances.importance.values:

    acc.append(i + acc[-1] if len(acc) > 0 else i)

importances['acc'] = acc

importances
importances.set_index('feature', drop=True, inplace=True)
fig, ax = plt.subplots()



ax.bar(importances.index, importances.importance)

ax.plot(importances.index, importances.acc, '--', color="red")

ax.set_ylabel('Importance')

ax.set_title('Feature Importances')

plt.xticks(rotation=90)



plt.show()
MODELS = [

    #Ensemble Methods

    AdaBoostClassifier(),

    BaggingClassifier(),

    ExtraTreesClassifier(),

    GradientBoostingClassifier(),

    RandomForestClassifier(),



    #Gaussian Processes

    GaussianProcessClassifier(),

    

    #GLM

    LogisticRegressionCV(),

    PassiveAggressiveClassifier(),

    RidgeClassifierCV(),

    SGDClassifier(),

    Perceptron(),

    

    #Navies Bayes

    BernoulliNB(),

    GaussianNB(),

    

    #Nearest Neighbor

    KNeighborsClassifier(),

    

    #SVM

    SVC(probability=True),

    NuSVC(probability=True),

    LinearSVC(),

    

    #Trees    

    DecisionTreeClassifier(),

    ExtraTreeClassifier(),

    

    #Discriminant Analysis

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),



    #xgboost

    XGBClassifier()    

]



k_fold = StratifiedKFold(n_splits=5)



columns = ['Model Name', 'Parameters','Train Accuracy Mean', 'Test Accuracy Mean', 'Test Accuracy STD * 3', 'Model', 'Time']

models = pd.DataFrame(columns=columns)



row_index = 0

for ml in MODELS:

    model_name = ml.__class__.__name__

    models.loc[row_index, 'Model Name'] = model_name

    models.loc[row_index, 'Parameters'] = str(ml.get_params())

    

    cv_results = cross_validate(ml, X_train_scaled, y_train, n_jobs=4, cv=k_fold, return_train_score=True, return_estimator=True)



    models.loc[row_index, 'Time'] = cv_results['fit_time'].mean()

    models.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_score'].mean()

    models.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_score'].mean()

    models.loc[row_index, 'Test Accuracy STD * 3'] = cv_results['test_score'].std() * 3

    models.loc[row_index, 'Model'] = cv_results['estimator']

    

    row_index+=1



models.sort_values(by=['Test Accuracy Mean'], ascending=False, inplace=True)

models.reset_index(drop=True, inplace=True)

models
sns.barplot(x='Test Accuracy Mean', y='Model Name', data=models, color='m')



plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
# 77% accuracy

top_k_models = 5



y_test = []

for i in range(top_k_models):

    preds = []

    for ml in models.loc[i, 'Model']:

        preds = [x + y for x, y in zip(ml.predict(X_test_scaled), preds)] if len(preds) > 0 else ml.predict(X_test_scaled)

    

    preds = [x/len(models.loc[i, 'Model']) for x in preds]

    y_test = [x + y for x, y in zip(preds, y_test)] if len(y_test) > 0 else preds



y_test = [x/top_k_models for x in y_test]

y_test = [1 if x >= .5 else 0 for x in y_test]



test['Survived'] = y_test

print('Survival rate: {}'.format(test['Survived'].mean()))



# test[['PassengerId', 'Survived']].to_csv('submission.csv', index=None)
gbc = GradientBoostingClassifier()



param_grid = {

    "max_depth": [1, 3, 5, 7, None],

    "min_samples_split": [2, 3, 10],

    "min_samples_leaf": [1, 3, 10],

    "n_estimators" :[100, 200, 300]

}



gridcv = GridSearchCV(gbc, param_grid = param_grid, cv=k_fold, scoring="accuracy", n_jobs=4, verbose=1)



gridcv.fit(X_train, y_train)



gbc_best = gridcv.best_estimator_



# Best Accuracy Score

gridcv.best_score_
xgb = XGBClassifier()



param_grid = {

    'gamma': [i/10.0 for i in range(0,5)],

    'subsample': [i/10.0 for i in range(6,10)],

    'colsample_bytree': [i/10.0 for i in range(6,10)],

    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05, 0.1, 1]

}



gridcv = GridSearchCV(xgb, param_grid = param_grid, cv=k_fold, scoring="accuracy", n_jobs=4, verbose=1)



gridcv.fit(X_train, y_train)



xgb_best = gridcv.best_estimator_



# Best Accuracy Score

gridcv.best_score_
rfc = RandomForestClassifier()



param_grid = {

    "max_depth": [None],

    "max_features": [1, 3, 10],

    "min_samples_split": [2, 3, 10],

    "min_samples_leaf": [1, 3, 10],

    "bootstrap": [False],

    "n_estimators" :[100, 200, 300],

    "criterion": ["entropy"]

}



gridcv = GridSearchCV(rfc, param_grid = param_grid, cv=k_fold, scoring="accuracy", n_jobs=4, verbose=1)



gridcv.fit(X_train, y_train)



rfc_best = gridcv.best_estimator_



# Best Accuracy Score

gridcv.best_score_
best_estimator = rfc_best

best_estimator.fit(X_train, y_train)



test['Survived'] = best_estimator.predict(X_test)

test[['PassengerId', 'Survived']].to_csv('submission.csv', index=False)
test.Survived.mean()