# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test =  pd.read_csv('/kaggle/input/titanic/test.csv')
train.head(5)
train.info()
train.describe().T
test.head(5)
test.info()
test.describe().T
# Missing Value in test data set



train.isnull().sum()
## As we can see Age and Cabin have missing values.
test.isnull().sum()
# Combining the train and test data



combinedata = [train,test]
combinedata
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# Checking the corelation



correlation = train.corr()

plt.figure(figsize=(10, 6))

sns.heatmap(correlation, annot=True, linewidths=0, vmin=-1, cmap="RdBu_r")
correlation['Survived'].sort_values(ascending=False)
# Checking Outliers value

from collections import Counter

def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
train.loc[Outliers_to_drop] # Show the outliers rows
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
train_len = len(train)

combineddataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
combineddataset.isnull().sum()
# Explore SibSp feature vs Survived

g = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Parch feature vs Survived

g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Age vs Survived

g = sns.FacetGrid(train, col='Survived')

g = g.map(sns.distplot, "Age")


# Explore Age distibution 

g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
# Explore skewness of Fare distribution 

combineddataset["Fare"] = combineddataset["Fare"].fillna(combineddataset["Fare"].median())

g = sns.distplot(combineddataset["Fare"], color="m", label="Skewness : %.2f"%(combineddataset["Fare"].skew()))

g = g.legend(loc="best")
combineddataset["Fare"] = combineddataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(combineddataset["Fare"], color="m", label="Skewness : %.2f"%(combineddataset["Fare"].skew()))

g = g.legend(loc="best")
g = sns.barplot(x="Sex",y="Survived",data=train)

g = g.set_ylabel("Survival Probability")
# Explore Pclass vs Survived

g = sns.factorplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Survived by Sex

g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
combineddataset["Embarked"].isnull().sum()
combineddataset["Embarked"] = combineddataset["Embarked"].fillna("S")
g = sns.factorplot(x="Embarked", y="Survived",  data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Embarked 

g = sns.factorplot("Pclass", col="Embarked",  data=train,

                   size=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")
# Explore Age vs Sex, Parch , Pclass and SibSP

g = sns.factorplot(y="Age",x="Sex",data=combineddataset,kind="box")

g = sns.factorplot(y="Age",x="Sex",hue="Pclass", data=combineddataset,kind="box")

g = sns.factorplot(y="Age",x="Parch", data=combineddataset,kind="box")

g = sns.factorplot(y="Age",x="SibSp", data=combineddataset,kind="box")
# convert Sex into categorical value 0 for male and 1 for female

combineddataset["Sex"] = combineddataset["Sex"].map({"male": 0, "female":1})
g = sns.heatmap(combineddataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
# Filling missing value of Age 



## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(combineddataset["Age"][combineddataset["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = combineddataset["Age"].median()

    age_pred = combineddataset["Age"][((combineddataset['SibSp'] == combineddataset.iloc[i]["SibSp"]) & (combineddataset['Parch'] == combineddataset.iloc[i]["Parch"]) & (combineddataset['Pclass'] == combineddataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        combineddataset['Age'].iloc[i] = age_pred

    else :

        combineddataset['Age'].iloc[i] = age_med
combineddataset.isnull().sum()
# Feature Enginnering

combineddataset["Name"].head()


# Get Title from Name

combineddataset_title = [i.split(",")[1].split(".")[0].strip() for i in combineddataset["Name"]]

combineddataset["Title"] = pd.Series(combineddataset_title)

combineddataset["Title"].head()


g = sns.countplot(x="Title",data=combineddataset)

g = plt.setp(g.get_xticklabels(), rotation=45)
# Convert to categorical values Title 

combineddataset["Title"] = combineddataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

combineddataset["Title"] = combineddataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

combineddataset["Title"] = combineddataset["Title"].astype(int)
g = sns.countplot(combineddataset["Title"])

g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
g = sns.factorplot(x="Title",y="Survived",data=combineddataset,kind="bar")

g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])

g = g.set_ylabels("survival probability")
# Drop Name variable

combineddataset.drop(labels = ["Name"], axis = 1, inplace = True)
# Create a family size descriptor from SibSp and Parch

combineddataset["Fsize"] = combineddataset["SibSp"] + combineddataset["Parch"] + 1
g = sns.factorplot(x="Fsize",y="Survived",data = combineddataset)

g = g.set_ylabels("Survival Probability")
# Create new feature of family size

combineddataset['Single'] = combineddataset['Fsize'].map(lambda s: 1 if s == 1 else 0)

combineddataset['SmallF'] = combineddataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)

combineddataset['MedF'] = combineddataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

combineddataset['LargeF'] = combineddataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
g = sns.factorplot(x="Single",y="Survived",data=combineddataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="SmallF",y="Survived",data=combineddataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="MedF",y="Survived",data=combineddataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="LargeF",y="Survived",data=combineddataset,kind="bar")

g = g.set_ylabels("Survival Probability")
# convert to indicator values Title and Embarked 

combineddataset = pd.get_dummies(combineddataset, columns = ["Title"])

combineddataset = pd.get_dummies(combineddataset, columns = ["Embarked"], prefix="Em")
combineddataset.head()
combineddataset["Cabin"].head()
# Create categorical values for Pclass

combineddataset["Pclass"] = combineddataset["Pclass"].astype("category")

combineddataset = pd.get_dummies(combineddataset, columns = ["Pclass"],prefix="Pc")
# Drop useless variables 

combineddataset.drop(labels = ["PassengerId","Cabin"], axis = 1, inplace = True)
combineddataset.head()
combineddataset.dtypes
train.dtypes
train = pd.get_dummies(train, columns = ["Embarked"], prefix="Em")


train = train.drop("Ticket", axis=1)
train.head()
test = pd.get_dummies(test, columns = ["Embarked"], prefix="Em")
test.head()
test = test.drop("Ticket", axis=1)
combineddataset = combineddataset.drop("Ticket", axis=1)
combineddataset.head()
## Separate train dataset and test dataset



train = combineddataset[:train_len]

test1 = combineddataset[train_len:]

test1.drop(labels=["Survived"],axis = 1,inplace=True)
## Separate train features and label 



train["Survived"] = train["Survived"].astype(int)



Y_train = train["Survived"]



X_train = train.drop(labels = ["Survived"],axis = 1)
X_train.head()
test1.head()
X_test1 = test1
X_test1.head()
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred_logreg = logreg.predict(X_test1)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred_svc = svc.predict(X_test1)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred_knn = knn.predict(X_test1)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred_gaussian = gaussian.predict(X_test1)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred_dt = decision_tree.predict(X_test1)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

random_forest.score(X_train, Y_train)

Y_pred_rfc = random_forest.predict(X_test1)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
# Gradient boosting tunning



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

kfold = StratifiedKFold(n_splits=10)

GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train,Y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 

              'Gradient Boosting', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, 

              gsGBC.best_score_, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred_rfc

    })

submission.to_csv('../working/submission.csv', index=False)
submission1 = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred_knn

    })

submission.to_csv('../working/submission1.csv', index=False)
submission.head()
submission1.head()
submission2 = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred_svc

    })

submission.to_csv('../working/submission2.csv', index=False)
submission3 = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred_dt

    })

submission.to_csv('../working/submission3.csv', index=False)
from sklearn import tree, ensemble, linear_model

ensemble = Y_pred_rfc*0.70 + Y_pred_svc*0.15 + Y_pred_logreg*0.15
submission5 = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": ensemble

    })

submission.to_csv('../working/submission5.csv', index=False)
# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
### META MODELING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING



# Adaboost

DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsadaDTC.fit(X_train,Y_train)



ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_
#ExtraTrees 

ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsExtC.fit(X_train,Y_train)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
# RFC Parameters tunning 

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(X_train,Y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train,Y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
### SVC classifier

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(X_train,Y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt



g = plot_learning_curve(gsRFC.best_estimator_,"RF mearning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)
# Feature Importance



nrows = ncols = 2

fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))



names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),("GradientBoosting",GBC_best)]



nclassifier = 0

for row in range(nrows):

    for col in range(ncols):

        name = names_classifiers[nclassifier][0]

        classifier = names_classifiers[nclassifier][1]

        indices = np.argsort(classifier.feature_importances_)[::-1][:40]

        g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])

        g.set_xlabel("Relative importance",fontsize=12)

        g.set_ylabel("Features",fontsize=12)

        g.tick_params(labelsize=9)

        g.set_title(name + " feature importance")

        nclassifier += 1
test_Survived_RFC = pd.Series(RFC_best.predict(test1), name="RFC")

test_Survived_ExtC = pd.Series(ExtC_best.predict(test1), name="ExtC")

test_Survived_SVMC = pd.Series(SVMC_best.predict(test1), name="SVC")

test_Survived_AdaC = pd.Series(ada_best.predict(test1), name="Ada")

test_Survived_GBC = pd.Series(GBC_best.predict(test1), name="GBC")





# Concatenate all classifier results

ensemble_results = pd.concat([test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)





g= sns.heatmap(ensemble_results.corr(),annot=True)
# Voting Classifier



votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),

('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)



votingC = votingC.fit(X_train, Y_train)

submissionNew = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": votingC.predict(test1)

    })

submission.to_csv('../working/submissionNew.csv', index=False)
submissionNew.head()