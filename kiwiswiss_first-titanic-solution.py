# data analysis and wrangling

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
# acquire data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
# check feature's name and other info

print(train_df.columns.values)

print(train_df.info())

print ('*' *40)

print(test_df.info())
# preview the data

train_df.head()
train_df.tail()
# check the distriburion of numeric and catergorical features

train_df.describe()
train_df.describe(include=['O'])
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df['Family'] = train_df.SibSp + train_df.Parch + 1

train_df[["Family", "Survived"]].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df['IsAlone'] = 0

train_df['IsAlone'][train_df['Family'] == 1] = 1

train_df[["IsAlone", "Survived"]].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False)
test_df['Family'] = test_df.SibSp + test_df.Parch + 1

test_df['IsAlone'] = 0

test_df['IsAlone'][test_df['Family'] == 1] = 1

combine = [train_df, test_df]
#delete uncorrelated features

train_df = train_df.drop(['SibSp', 'Parch', 'Family'], axis = 1)

test_df = test_df.drop(['SibSp', 'Parch', 'Family'], axis = 1)

combine = [train_df, test_df]
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]



train_df.shape, test_df.shape
#analysis of feature embarked, age and fare by vasulization

#grid = sns.FacetGrid(train_df, col='Embarked', size=3.2, aspect=1.2)

grid = sns.FacetGrid(train_df, row='Pclass', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Embarked','Survived','Sex', palette='deep')

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex','Fare', alpha=0.5, ci=None)

grid.add_legend()
ax1=sns.kdeplot(train_df['Age'][train_df['Survived']==1],color='r')

ax2=sns.kdeplot(train_df['Age'][train_df['Survived']==0],color='b')

ax2=sns.kdeplot(train_df['Age'],color='g')



grid = sns.FacetGrid(train_df, size=2.2, aspect=1.6)

grid.map(sns.stripplot,x="Survived", y="Age", data=train_df)

grid.add_legend()
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=80)

grid.add_legend()
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)



train_df.head()
for dataset in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 10)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 8, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 8) & (dataset['Age'] <= 16), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 24), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 24) & (dataset['Age'] <= 32), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 40), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 48), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 56), 'Age'] = 6

    dataset.loc[(dataset['Age'] > 56) & (dataset['Age'] <= 64), 'Age'] = 7

    dataset.loc[(dataset['Age'] > 64) & (dataset['Age'] <= 72), 'Age'] = 8

    dataset.loc[ dataset['Age'] > 72, 'Age']

train_df.head()
train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

test_df.head()
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)



train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]

    

train_df.head(10)
test_df.head(10)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, test_df.shape, X_test.shape
# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 

random_state = 2

classifiers = []

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(SVC(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(GaussianNB())

classifiers.append(Perceptron(random_state=random_state))

classifiers.append(LinearSVC(random_state=random_state))

classifiers.append(SGDClassifier(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())

    

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["LogisticRegression","SVC","KNeighboors","GaussianNaiveBayes","Perceptron","LinearSVC","StochasticGradientDescent","DecisionTree",

"RandomForest","GradientBoosting", "AdaBoostClassifier","ExtraTreesClassifier","MLPClassifier", "LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
cv_res.sort_values(by='CrossValMeans', ascending=False)
# SVC classifier

SVMC = SVC(probability=True)



## Search grid for optimal parameters

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 2, verbose = 1)



gsSVMC.fit(X_train,Y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
# DTC Parameters tunning 

DTC = DecisionTreeClassifier()



## Search grid for optimal parameters

rf_param_grid = {"max_depth": [3, 6],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 8]}



gsDTC = GridSearchCV(DTC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 2, verbose = 1)



gsDTC.fit(X_train,Y_train)



DTC_best = gsDTC.best_estimator_



# Best score

gsDTC.best_score_
# RFC Parameters tunning 

RFC = RandomForestClassifier()



## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 6],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 2, verbose = 1)



gsRFC.fit(X_train,Y_train)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
# Gradient boosting tunning

GBC = GradientBoostingClassifier()



## Search grid for optimal parameters

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 2, verbose = 1)



gsGBC.fit(X_train,Y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
# Adaboost

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

              "max_features": [1, 3, 6],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}



gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 2, verbose = 1)



gsExtC.fit(X_train,Y_train)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
models = pd.DataFrame({

    'Model': ['SupportVectorMachines', 'GradientBoosting', 'RandomForest',

              'AdaBoostClassifier', 'ExtraTreesClassifier', 'DecisionTree'],

    'Score': [gsSVMC.best_score_, gsGBC.best_score_, gsRFC.best_score_, 

              gsadaDTC.best_score_, gsExtC.best_score_, gsDTC.best_score_,]})

models.sort_values(by='Score', ascending=False)
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



g = plot_learning_curve(gsDTC.best_estimator_,"DecisionTree curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsRFC.best_estimator_,"RF curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsSVMC.best_estimator_,"SVC learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost learning curves",X_train,Y_train,cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting learning curves",X_train,Y_train,cv=kfold)
nrows = 2

ncols = 2

fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))



names_classifiers = [("AdaBoosting", ada_best),("ExtraTrees",ExtC_best),("RandomForest",RFC_best),

                     ("GradientBoosting",GBC_best)]



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
test_Survived_DTC = pd.Series(DTC_best.predict(X_test), name="DTC")

test_Survived_RFC = pd.Series(RFC_best.predict(X_test), name="RFC")

test_Survived_ExtC = pd.Series(ExtC_best.predict(X_test), name="ExtC")

test_Survived_SVMC = pd.Series(SVMC_best.predict(X_test), name="SVC")

test_Survived_AdaC = pd.Series(ada_best.predict(X_test), name="Ada")

test_Survived_GBC = pd.Series(GBC_best.predict(X_test), name="GBC")





# Concatenate all classifier results

ensemble_results = pd.concat([test_Survived_DTC,test_Survived_RFC,test_Survived_ExtC,test_Survived_AdaC,test_Survived_GBC, test_Survived_SVMC],axis=1)





g= sns.heatmap(ensemble_results.corr(),annot=True)
votingC = VotingClassifier(estimators=[('dtc', DTC_best), ('rfc', RFC_best), ('extc', ExtC_best),

('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)



votingC = votingC.fit(X_train, Y_train)
test_Survived = pd.Series(votingC.predict(X_test), name="Survived")



IDtest = test_df["PassengerId"]

results = pd.concat([IDtest,test_Survived],axis=1)



results.to_csv("../output/submission.csv",index=False)