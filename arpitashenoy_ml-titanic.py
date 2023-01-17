# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



sns.set(style='white', context='notebook', palette='deep')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load the data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

IDtest = test['PassengerId']
#outlier detection using Tukey Method

def detect_outlier(df,n,features):

    

    outlier_indices = []

    

    #Iterate over features

    for col in features:

        #1st quartile(25%)

        Q1 = np.percentile(df[col].dropna(), 25)

        #3rd quartile(75%)

        Q3 = np.percentile(df[col].dropna(), 75)

        #Interquartile range

        IQR = Q3-Q1

        

        #outlier step

        outlier_step = 1.5*IQR

        

        #determine a list of indices of outlier for feature col

        outliers_list_col = df[(df[col]<Q1-outlier_step)|(df[col]>Q3+outlier_step)].index

        

        #append the found outlier indices for col to the list of outlier indices

        outlier_indices.extend(outliers_list_col)

        

    #select observations containing more than 2 outliers.

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(k for k,v in outlier_indices.items() if v>n)

    

    return multiple_outliers

    
#detect outliers from Age, SibSp, Parch and Fare

outliers_to_drop = detect_outlier(train, 2, ["Age", "SibSp", "Parch", "Fare"])
#show the outlier rows

train.loc[outliers_to_drop]
#drop outliers



train = train.drop(outliers_to_drop, axis = 0).reset_index(drop = True)

#joining training and test set

train_len = len(train)



dataset = pd.concat(objs = [train,test], axis = 0, sort = False).reset_index(drop = True)
#fill empty and nan values with nan

dataset = dataset.fillna(np.nan)
#check for null values

dataset.isnull().sum()
#info

train.info()

train.isnull().sum()
train.head()
train.dtypes
#sumarize data

train.describe()
#correlation matrix between numerical values and Survived feature

g = sns.heatmap(train[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot = True, fmt = ".2f", cmap = "coolwarm")
#Explore SibSp feature vs Survived

g = sns.catplot(x = 'SibSp', y = 'Survived', data = train, kind = 'bar', height = 6, palette = 'muted')

#g.despine(left = True)

g.set_ylabels("Survival Probability")
#explore parch feature vs survived

g = sns.catplot(x = 'Parch', y = 'Survived', data = train, height = 6, kind = 'bar', palette = 'muted')
#explore age vs survived

g = sns.FacetGrid(train, col = 'Survived')

g = g.map(sns.distplot, 'Age')
#explore age distribution

g = sns.kdeplot(train['Age'][(train['Survived']==0)&(train['Age'].notnull())], color = 'r', shade = True)

g = sns.kdeplot(train['Age'][(train['Survived']==1)&(train['Age'].notnull())], color = 'b', shade = True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g.legend(["Not Survived", "Survived"])
#Fare

dataset['Fare'].isnull().sum()
#fill fare missing values with median values

dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
#explore fare distribution

g = sns.distplot(dataset['Fare'], color = 'm', label = 'skewness: %2f' %(dataset['Fare'].skew()))

g.legend(loc = 'best')
#apply log function to reduce skewness

dataset['Fare'] = dataset['Fare'].map(lambda i : np.log(i) if i>0 else 0)
g = sns.distplot(dataset['Fare'], color = 'b', label = "skewness: %2f" %(dataset['Fare'].skew()))

g.legend(loc = "best")
#sex

g = sns.barplot(x = 'Sex', y = 'Survived', data = train)

g.set_ylabel("Survival Probability")
train[['Sex', 'Survived']].groupby('Sex').mean()
#explore Pclass vs Survived

g = sns.catplot(x = 'Pclass', y = 'Survived', data= train, kind = 'bar', height = 6, palette = 'muted')

g.set_ylabels('Survival Probability')
#Embarked

dataset['Embarked'].isnull().sum()
#fill missing values with most frequent values

dataset['Embarked'] = dataset['Embarked'].fillna('S')
#explore Embarked vs survived

g = sns.catplot(x = 'Embarked', y = 'Survived', data = train, height = 6, kind = 'bar', palette = 'muted')

g.set_ylabels('Survival probability')
#explore pclass vs embarked

g = sns.catplot('Pclass', col = 'Embarked', data = train, height = 6, kind = 'count', palette = 'muted')

g.set_ylabels('count')
#explore age vs Sex, Parch, Pclass and SibSp

g = sns.catplot(y = 'Age', x = 'Sex', data = dataset, kind = 'box')

g = sns.catplot(y = 'Age', x = 'Sex', hue = 'Pclass', data = dataset, kind = 'box')

g = sns.catplot(y = 'Age', x = 'Parch', data = dataset, kind = 'box')

g = sns.catplot(y = 'Age', x = 'SibSp', data = dataset, kind = 'box')
#convert sex into categorical value 0 for male and 1 for female

dataset['Sex'] = dataset['Sex'].map({'male' : 0 , 'female' : 1})
g = sns.heatmap(dataset[['Age', 'Sex', 'SibSp', 'Parch', 'Pclass']].corr(), cmap = "BrBG", annot = True)
X_age = dataset[['Age', 'Sex', 'SibSp', 'Parch', 'Pclass']]
index_age = list(dataset['Age'][dataset['Age'].isnull()].index)
X_age_test = X_age.loc[index_age]
X_age_test = X_age_test.drop(columns = 'Age')
X_age_test.head()
X_age_train1 = X_age.dropna(axis = 0)
X_age_train = X_age_train1.drop(columns = 'Age')

X_age_train.head()
y_age_train = X_age_train1['Age']
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
regr.fit(X_age_train, y_age_train)
predictions = pd.Series(regr.predict(X_age_test))
predictions.head()
j=0

for i in index_age:

    dataset['Age'].iloc[i] = predictions[j]

    j+=1
dataset['Age'].isnull().sum()
g = sns.catplot(x = 'Survived', y = 'Age', data = train, kind = 'box')

g = sns.catplot(x = 'Survived', y = 'Age', data = train, kind = 'violin')
#Name/Title

dataset['Name'].head()

#get the title from name

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset['Name']]

dataset['Title'] = pd.Series(dataset_title)

dataset['Title'].head()
g = sns.countplot(x = 'Title', data = dataset)

g = plt.setp(g.get_xticklabels(), rotation = 45)
#convert title to categorical values

dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona' ], 'Rare')

dataset['Title'] = dataset['Title'].map({"Master":0, "Miss": 1, "Ms":1, "Mme": 1, "Mlle": 1, "Mrs":1, "Mr" :2, "Rare":3})

dataset['Title'] = dataset['Title'].astype(int)
g = sns.countplot(dataset['Title'])

g.set_xticklabels(['Master', 'Miss', 'Mr', 'Rare'])
g = sns.catplot(x = 'Title', y = 'Survived', data = dataset, kind = 'bar')

g.set_xticklabels(['Master', 'Miss/Mrs', 'Mr', 'Rare'])

g.set_ylabels('Survival probability')
#drop name column

dataset.drop(labels = ['Name'], axis = 1, inplace = True)
# Family Size

# create a family size descriptor from SibSp and Parch

dataset['Fsize'] = dataset['SibSp']+dataset['Parch']+1
g = sns.catplot(x = 'Fsize', y = 'Survived', data = dataset, kind = 'bar')

g.set_ylabels('Survival probability')
#create new features of family size

dataset['Single'] = dataset['Fsize'].map(lambda s : 1 if s == 1 else 0)

dataset['SmallF'] = dataset['Fsize'].map(lambda s : 1 if s == 2 else 0)

dataset['MedF'] = dataset['Fsize'].map(lambda s : 1 if 3<=s<=4 else 0)

dataset['LargeF'] = dataset['Fsize'].map(lambda s : 1 if s >=5 else 0)
g = sns.catplot(x = 'Single', y = 'Survived', data = dataset, kind = 'bar')

g.set_ylabels('Survival probability')

g = sns.catplot(x = 'SmallF', y = 'Survived', data = dataset, kind = 'bar')

g.set_ylabels('Survival Probability')

g = sns.catplot(x = 'MedF', y = 'Survived', data = dataset, kind = 'bar')

g.set_ylabels('Survival Probability')

g = sns.catplot(x = 'LargeF', y = 'Survived', data = dataset, kind = 'bar')

g.set_ylabels('Survival Probability')
#Convert to indicator values Title and Embarked

dataset = pd.get_dummies(dataset, columns = ['Title'])

dataset = pd.get_dummies(dataset, columns = ['Embarked'], prefix = 'Em')

dataset.head()
#Cabin

dataset['Cabin'].head()
dataset['Cabin'].describe()
dataset['Cabin'].isnull().sum()
dataset['Cabin'][dataset['Cabin'].notnull()].head()
#Replace the cabin no. by the type of cabin 'x' if not

dataset['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'x' for i in dataset['Cabin']])
g = sns.countplot(dataset['Cabin'], order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'x'])

g = sns.catplot(y = 'Survived', x = 'Cabin', data = dataset, kind = 'bar', order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T', 'x'])

g.set_ylabels('Survival Probability')
dataset = pd.get_dummies(dataset, columns = ['Cabin'], prefix = 'Cabin')
#Ticket

dataset['Ticket'].head()
#treat the ticket by extracting the ticket prefix when there is no prefix it returns x

Ticket = []

for i in list(dataset.Ticket):

    if not i.isdigit():

        Ticket.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        Ticket.append("x")
dataset["Ticket"] = Ticket
dataset['Ticket'].head()
dataset = pd.get_dummies(dataset, columns = ['Ticket'], prefix = 'T')
#create categorical values for Pclass

dataset['Pclass'] = dataset['Pclass'].astype("category")

dataset = pd.get_dummies(dataset, columns = ['Pclass'], prefix = 'PC')
#drop useless variables

dataset.drop(labels = ['PassengerId'], axis = 1, inplace = True)
dataset.head()
##Separate train dataset and test dataset

train = dataset[:train_len]

test = dataset[train_len:]

test.drop(labels = ['Survived'], axis = 1, inplace = True)
#Separate train features and labels

train['Survived'] = train['Survived'].astype(int)

y_train = train['Survived']

X_train = train.drop(labels = ['Survived'], axis =1)
#cross validate model with kfold stratified cross val 

kfold = StratifiedKFold(n_splits = 10)
#modeling step - test different algorithms

classifiers = []

classifiers.append(SVC(random_state = 2))

classifiers.append(DecisionTreeClassifier(random_state = 2))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state = 2),random_state = 2, learning_rate = 0.1))

classifiers.append(RandomForestClassifier(random_state = 2))

classifiers.append(ExtraTreesClassifier(random_state = 2))

classifiers.append(GradientBoostingClassifier(random_state = 2))

classifiers.append(MLPClassifier(random_state = 2))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = 2))

classifiers.append(LinearDiscriminantAnalysis())
cv_results = []

for classifier in classifiers:

    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = 'accuracy', cv = kfold, n_jobs = 4))
cv_mean = []

cv_std = []

for cv_result in cv_results:

    cv_mean.append(cv_result.mean())

    cv_std.append(cv_result.std())
cv_res = pd.DataFrame({"CrossValMeans": cv_mean, "CrossValError": cv_std, "Algorithm":["SVC", "DecisionTree", "AdaBoost", "RandomForest", "ExtraTrees", "GradientBoosting", "MultiLayerPerceptron", "KNeighbors", "LogisticRegression", "LinearDescriminantAnalysis"]})
g = sns.barplot("CrossValMeans", "Algorithm", data = cv_res, palette = "Set3", orient = 'h', **{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scored")
#Meta modeling



#SVC classifier

SVMC = SVC(probability = True)

SVC_param_grid = {'kernel': ['rbf'],

                 'gamma': [0.001,0.01,0.1,1],

                 'C': [1,10,50,100,200,300,1000]}



gsSVMC = GridSearchCV(SVMC, param_grid = SVC_param_grid, cv = kfold, scoring = "accuracy", n_jobs=4, verbose=1)



gsSVMC.fit(X_train, y_train)



SVMC_best = gsSVMC.best_estimator_
#best score

gsSVMC.best_score_
#AdaBoost 

DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"],

                  "base_estimator__splitter": ["best", "random"], 

                  "algorithm": [ "SAMME", "SAMME.R"], 

                  "n_estimators": [1, 2],

                  "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}



gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)



gsadaDTC.fit(X_train, y_train)



ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_
#RandomForest

RFC = RandomForestClassifier()



rf_param_grid = {"max_depth": [None],

                "max_features": [1,3,10],

                "min_samples_split": [2,3,10],

                "min_samples_leaf": [1,3,10],

                "n_estimators": [100,300],

                "bootstrap": [False],

                "criterion": ["gini"]}



gsrfc = GridSearchCV(RFC, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)



gsrfc.fit(X_train, y_train)



rfc_best = gsrfc.best_estimator_
gsrfc.best_score_
#gradient boosting

GBC = GradientBoostingClassifier()



gb_param_grid = {"loss": ["deviance"],

                "n_estimators": [100,200,300],

                "learning_rate": [0.1,0.05,0.01],

                "max_depth": [4,8],

                "min_samples_leaf": [100,150],

                "max_features": [0.3,0.1]}



gsGBC = GridSearchCV(GBC, param_grid=gb_param_grid, cv=kfold, scoring='accuracy', n_jobs=4, verbose=1)



gsGBC.fit(X_train,y_train)



GBC_best = gsGBC.best_estimator_
gsGBC.best_score_
#MLP



mlp = MLPClassifier()



mlp_param_grid = {"alpha": [0.0001,0.001,0.01,0.1,0.3,0.03,0.003,0.0003],

                 "hidden_layer_sizes": [100,200],

                 "learning_rate_init": [0.001,0.003,0.01,0.03,0.1,0.3,1.0],

                 "max_iter": [200,300]}



gsmlp = GridSearchCV(mlp, param_grid=mlp_param_grid, cv = kfold, scoring ="accuracy", n_jobs=4, verbose=1)



gsmlp.fit(X_train, y_train)



gsmlp_best = gsmlp.best_estimator_
gsmlp.best_score_
#Logistic Regression

lr = LogisticRegression()



lr_param_grid = {"C": [1, 10, 50,100,200,500,1000],

                "max_iter": [100,200,300]}



gslr = GridSearchCV(lr, param_grid=lr_param_grid, cv = kfold, scoring = "accuracy", n_jobs = 4, verbose =1)



gslr.fit(X_train, y_train)



lr_best = gslr.best_estimator_
gslr.best_score_
def plot_learning_curve(estimator, title, X, y, ylim = None, cv=None, n_jobs =1,train_sizes = np.linspace(0.1,1.0,5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training example")

    plt.ylabel("score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv = cv, n_jobs = n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis = 1)

    train_scores_std = np.std(train_scores, axis =1)

    test_scores_mean = np.mean(test_scores, axis = 1)

    test_scores_std = np.std(test_scores, axis = 1)

    plt.grid()

    

    plt.fill_between(train_sizes, train_scores_mean-train_scores_std, train_scores_mean+train_scores_std, alpha=0.1, color = 'r')

    plt.fill_between(train_sizes, test_scores_mean-test_scores_std, test_scores_mean+test_scores_std, alpha=0.1, color = 'g')

    

    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r', label = "Training Score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g', label = "Crossvalidation Score")

    

    plt.legend(loc="best")

    return plt

    

    
g = plot_learning_curve(gsrfc.best_estimator_, "RFC curve", X_train, y_train, cv=kfold)

g = plot_learning_curve(gsadaDTC.best_estimator_, "ADA curve", X_train, y_train, cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_, "GBC curve", X_train, y_train, cv=kfold)

g = plot_learning_curve(gslr.best_estimator_, "LR curve", X_train, y_train, cv=kfold)

g = plot_learning_curve(gsmlp.best_estimator_, "MLP curve", X_train, y_train, cv=kfold)

g = plot_learning_curve(gsSVMC.best_estimator_, "SVM curve", X_train, y_train, cv=kfold)
test_rfc = pd.Series(rfc_best.predict(test), name = "RFC")

test_ada = pd.Series(ada_best.predict(test), name = "Ada")

test_svc = pd.Series(SVMC_best.predict(test), name = "SVC")

test_gbc = pd.Series(GBC_best.predict(test), name = "GBC")

test_mlp = pd.Series(gsmlp_best.predict(test), name = "MLP")

test_lr = pd.Series(lr_best.predict(test), name = "LR")
#concatenate all classifiers results

ensemble_results = pd.concat([test_rfc,test_ada,test_svc,test_gbc, test_mlp, test_lr], axis=1)
g= sns.heatmap(ensemble_results.corr(),annot=True)
# Ensemble modeling



votingC = VotingClassifier(estimators = [('rfc', rfc_best), ('ada', ada_best), ('svc', SVMC_best), ('gbc', GBC_best), ('mlp', gsmlp_best),('lr', lr_best)], voting = "soft", n_jobs=4)



votingC.fit(X_train, y_train)
#Predictions

test_survived = pd.Series(votingC.predict(test), name = "Survived")

results = pd.concat([IDtest, test_survived], axis = 1)

results.to_csv("submission.csv", index = False)