import pandas as pd

import numpy as np



import re

#ignore warnings

import warnings

warnings.filterwarnings('ignore')



#Modle Algorithms

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifierCV, SGDClassifier, Perceptron,RidgeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import BernoulliNB, GaussianNB

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC, NuSVC,LinearSVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

PassengerId = test['PassengerId']
# data information and describe

train.info()
test.info()

train.head()
train.isnull().sum()
test.isnull().sum()
train["Pclass"][train["Survived"] == 1].value_counts()/train["Pclass"].value_counts()
g = sns.factorplot(x = "Pclass",y = "Survived",data = train,kind = "bar", palette = "Greens_d")
train["Sex"][train["Survived"] == 1].value_counts()/train["Sex"].value_counts()
g = sns.factorplot(x="Sex", y="Survived", data=train, kind="bar", palette="Greens_d")
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", kind="bar", data=train, palette="Greens_d")
g = sns.kdeplot(train["Age"][train["Age"].notnull()], color="green", shade=True)
g = sns.kdeplot(train["Age"][train["Survived"] == 1][train["Age"].notnull()], color="blue", shade=True)

g = sns.kdeplot(train["Age"][train["Survived"] == 0][train["Age"].notnull()], color="red", shade=True)

g = g.legend(["Survived","Not Survived"])
g = sns.kdeplot(train["Age"][train["Pclass"]== 1][train["Age"].notnull()], color="green",shade=True)

g = sns.kdeplot(train["Age"][train["Pclass"]== 2][train["Age"].notnull()], color="red",shade=True)

g = sns.kdeplot(train["Age"][train["Pclass"]== 3][train["Age"].notnull()], color="yellow", shade=True)

g = g.legend(["Pclass_1","Pclass_2","Pclass_3"])
g = sns.kdeplot(train["Age"][train["Sex"]== 'male'][train["Age"].notnull()], color="blue",shade=True)

g = sns.kdeplot(train["Age"][train["Sex"]== 'female'][train["Age"].notnull()], color="red",shade=True)

g =g.legend(["male","female"])
fig, (saxis1,saxis2) = plt.subplots(1,2,figsize=(8,4))



sns.barplot(x='SibSp', y='Survived', data=train, palette="Greens_d", ax =saxis1)

sns.barplot(x='Parch', y="Survived", data=train, palette="Greens_d", ax=saxis2 )
g = sns.distplot(train["Fare"],color='g')
train["Fare"][train["Pclass"] == 1].sum()/len(train[train["Pclass"] == 1])
train["Fare"][train["Pclass"] == 2].sum()/len(train[train["Pclass"] == 2])
train["Fare"][train["Pclass"] == 3].sum()/len(train[train["Pclass"] == 3])
fig, (saxis1,saxis2) = plt.subplots(1,2,figsize=(12,4))

sns.countplot(x="Embarked", data=train, palette="Greens_d", ax=saxis1)

saxis1.set_ylabel("PassengerNum")

sns.barplot(x="Embarked", y="Survived", data=train, palette="Greens_d", ax=saxis2)
g = sns.heatmap(train[["Survived","Age","Fare","Parch","SibSp"]].corr(), cmap=sns.diverging_palette(220, 10, as_cmap = True), annot=True,square=True,linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 })

plt.title('Correlation of Numerical Features', y=1.05, size=15)
def outlier_detecte(data,features):

    outlier_indices = []

    for col in features:

        # 1st quartile(25%)

        q1 = np.percentile(data[col],25)

        # 3rd quartile(75%)

        q3 = np.percentile(data[col],75)

        # IQR

        IQR = q3 - q1

        

        # get outlier indices 

        outlier_col = data[(data[col] < (q1 - 1.5 * IQR)) | (data[col] > (q3 + 1.5 * IQR))].index

        outlier_indices.extend(outlier_col)

    outlier_indices = pd.DataFrame({"outlier":outlier_indices})["outlier"].value_counts()

    return outlier_indices[outlier_indices > 2].index
outlier_to_drop = outlier_detecte(train,["Age","Parch","SibSp","Fare"])
train.loc[outlier_to_drop]
# Drop outliers

train = train.drop(outlier_to_drop, axis = 0).reset_index(drop=True)
train.isnull().sum()
test.isnull().sum()
def fill_missing_value(data):

    # Pclass1 Average age

    pl1_ave_age = data.Age[data.Pclass == 1].mean()



    # Pclass2 Average age

    pl2_ave_age = data.Age[data.Pclass == 2].mean()



    # Pclass3 Average age

    pl3_ave_age = data.Age[data.Pclass == 3].mean()



    # Pclass1 age std

    pl1_std_age = data.Age[data.Pclass == 1].std()



    # Pclass2 age std

    pl2_std_age = data.Age[data.Pclass == 2].std()



    # Pclass3 age std

    pl3_std_age = data.Age[data.Pclass == 3].std()



    # 3 Pclass null value number

    pl1_num_null = data.Age[data.Pclass == 1].isnull().sum()

    pl2_num_null = data.Age[data.Pclass == 2].isnull().sum()

    pl3_num_null = data.Age[data.Pclass == 3].isnull().sum()



    # Fill the null value with random value between (mean - std) and (mean + std]

    data.loc[(data.Age.isnull()) & (data.Pclass == 1),'Age'] = np.random.randint(pl1_ave_age - pl1_std_age,pl1_ave_age + pl1_std_age, size = pl1_num_null)

    data.loc[(data.Age.isnull()) & (data.Pclass == 2),'Age'] = np.random.randint(pl2_ave_age - pl2_std_age,pl2_ave_age + pl2_std_age, size = pl2_num_null)

    data.loc[(data.Age.isnull()) & (data.Pclass == 3),'Age'] = np.random.randint(pl3_ave_age - pl3_std_age,pl3_ave_age + pl3_std_age, size = pl3_num_null)

    

    # Fill the fare missing value

    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    

    # deal with Cabin 

    data['Cabin'] = data['Cabin'].apply(lambda x: 1 if type(x) == float else 0 )

    

    # Fill Embarked missing value with Embarked mode.

    data['Embarked'] = data.Embarked.fillna('S')

    return data
train = fill_missing_value(train)

test = fill_missing_value(test)
def new_feature(data):

    def get_title(name):

        title_search = re.search('([A-Za-z]+)\.',name)

        # if the title exists, extract and return it.

        if title_search:

            return title_search.group(1)

        return ""

    

    # Create a new feature stands for family size.

    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    

    # Passenger is alone when he dont have family.

    data['Alone'] = data.FamilySize.apply(lambda x:1 if x == 1 else 0)

    

    # Get passenger's name title

    data['Title'] = data.Name.apply(get_title)

    

    # Replace the rare title with 'rate'

    data["Title"] = data["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    return data
train = new_feature(train)

test = new_feature(test)
def transform(data):

    data['Fare'] = data['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

    return data
train = transform(train)

test = transform(test)
def discretize(data):

    data['Age'] = pd.cut(data.Age,5,labels = [0,1,2,3,4])

    return data
train = discretize(train)

test = discretize(test)
def decompose(data):

    data['Sex'] = data.Sex.map({'female':0,'male':1}).astype(int)

    data['Embarked'] = data.Embarked.map({'Q':0,'S':1,'C':2}).astype(int)

    data["Title"] = data["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3}).astype(int)

    return data
train = decompose(train)

test = decompose(test)
def select(data):

    data.drop(['PassengerId','Name','SibSp','Parch','Ticket'],axis = 1,inplace = True)

    return data
train = select(train)

test  = select(test)
train.head()

test.head()

# Get model data

y_train = train["Survived"]

x_train = train.drop(labels = ["Survived"], axis=1)
classifiers = [

    #Ensemble Methods

    AdaBoostClassifier(),

    BaggingClassifier(),

    ExtraTreesClassifier(),

    GradientBoostingClassifier(),

    RandomForestClassifier(),



    #Gaussian Processes

    GaussianProcessClassifier(),

    

    #GLM

    LogisticRegression(),

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

    ]

#split dataset in cross-validation

kfold = StratifiedKFold(n_splits=10)

clf_name = []

cv_results = []

cv_means = []

cv_std = []

for clf in classifiers:

    cv_results.append(cross_val_score(clf, x_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

    clf_name.append(clf.__class__.__name__)

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":clf_name})

cv_res.sort_values(by="CrossValMeans",ascending=False,inplace=True)

cv_res
g = sns.barplot(y="Algorithm", x="CrossValMeans", data=cv_res, color="g")

g.set_title("machine learning algorithm accuracy score")

g.set_xlabel("score")

g.set_ylabel("algorithm")
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
# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance","exponential"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1,"auto","sqrt","log2",None]

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)



gsGBC.fit(x_train,y_train)



GBC_best = gsGBC.best_estimator_



# Best score

print(gsGBC.best_score_)

print(GBC_best)
g = plot_learning_curve(GBC_best,'Gradient boosting leanring curve',x_train,y_train,cv=kfold)
### SVC classifier

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'],

              'gamma': [ 0.001, 0.01, 0.1, 1],

              'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(x_train,y_train)



SVMC_best = gsSVMC.best_estimator_



# Best score

print(gsSVMC.best_score_)

print(SVMC_best)
g = plot_learning_curve(SVMC_best,'SVC leanring curve',x_train,y_train,cv=kfold)
DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[3,4,5],

              "learning_rate":  [0.1,0.2,0.3]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsadaDTC.fit(x_train,y_train)



ada_best = gsadaDTC.best_estimator_



print(ada_best)

print(gsadaDTC.best_score_)
g = plot_learning_curve(ada_best,'AdaBoostClassifier leanring curve',x_train,y_train,cv=kfold)
# BaggingClassifier

bagclf = BaggingClassifier()

bag_param_grid = {"n_estimators": [20,30,50,60,70],

                 "max_samples": [1,2,3,4,5,6,7,8,9],

                 "max_features":[2,3,4],

                  "n_jobs":[-1]

                 }

gsbag = GridSearchCV(bagclf,param_grid = bag_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsbag.fit(x_train,y_train)

bag_best  = gsbag.best_estimator_

print(bag_best)

print(gsbag.best_score_)
g = plot_learning_curve(bag_best,'BaggingClassifier leanring curve',x_train,y_train,cv=kfold)
# RFC Parameters tunning

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [2,3,4],

              "min_samples_split": [9,10,11],

              "min_samples_leaf": [9,10,11],

              "bootstrap": [False],

              "n_estimators" :[300,400,500,450],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(x_train,y_train)



RFC_best = gsRFC.best_estimator_



print(gsRFC.best_score_)

print(RFC_best)
g = plot_learning_curve(RFC_best,'RandomForestClassifier leanring curve',x_train,y_train,cv=kfold)
#ExtraTrees

ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 8],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsExtC.fit(x_train,y_train)



ExtC_best = gsExtC.best_estimator_



print(gsExtC.best_score_)

print(ExtC_best)
g = plot_learning_curve(ExtC_best,'ExtraTreesClassifier leanring curve',x_train,y_train,cv=kfold)
votingclf = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),

('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)



votingclf.fit(x_train, y_train)



test_Survived = pd.Series(votingclf.predict(test), name="Survived")



results = pd.concat([PassengerId,test_Survived],axis=1)



results.to_csv("my_prediction.csv",index=False)

print('done')