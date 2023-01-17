import pandas as pd

import numpy as np

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

df_train=pd.read_csv('/kaggle/input/titanic/train.csv')

df_test=pd.read_csv('/kaggle/input/titanic/test.csv')



IDtest = df_test["PassengerId"]
df_train.shape
df_test.shape
df_train.head()
df_test.head()
df_train["Survived"].value_counts()
df_train.info()
dataset =  pd.concat(objs=[df_train, df_test], axis=0).reset_index(drop=True)
# Fill empty and NaNs values with NaN

dataset = dataset.fillna(np.nan)



# Check for Null values

dataset.isnull().sum()
df_train.isnull().sum()
df_train.dtypes
#Summarize and statistics

df_train.describe()
g = sns.heatmap(df_train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
g = sns.factorplot(x="SibSp",y="Survived",data=df_train,kind="bar", size = 6 ,palette = "muted")

g.despine(left=True)

g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="Parch",y="Survived",data=df_train,kind="bar", size = 6 ,palette = "muted")

g.despine(left=True)

g = g.set_ylabels("Survival Probability")
g = sns.FacetGrid(df_train, col='Survived')

g = g.map(sns.distplot, "Age", color="c")
# Explore Age Distibution

g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 0) & (df_train["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(df_train["Age"][(df_train["Survived"] == 1) & (df_train["Age"].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
df_train['Fare'].isnull().sum()
dataset['Fare'].isnull().sum()
#Let's fill the Fare missing values with the median value

dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
# Explore Fare distribution 

g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))

g = g.legend(loc="best")
# Apply log to Fare to reduce skewness distribution

dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
#After log transformation to reduce the skewness

g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))

g = g.legend(loc="best")
g = sns.barplot(x="Sex",y="Survived",data=df_train)

g = g.set_ylabel("Survival Probability")
df_train[["Sex","Survived"]].groupby('Sex').mean()
g = sns.factorplot(x="Pclass",y="Survived", data=df_train, kind="bar", size = 6 ,palette = "muted")

g.despine(left=True)

g = g.set_ylabels("Survival Probability")
# Explore Pclass vs Survived by Sex

g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=df_train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Survival Probability")
df_train["Embarked"].isnull().sum()
dataset["Embarked"].isnull().sum()
dataset['Embarked'].value_counts()
#Let's fill Embarked missing values of dataset set with 'S' most frequent value

dataset["Embarked"] = dataset["Embarked"].fillna("S")
# Explore Embarked vs Survived 

g = sns.factorplot(x="Embarked", y="Survived",  data=df_train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
dataset.isnull().sum()
# Explore Age vs Sex, Parch, Pclass and SibSP.

g = sns.catplot(y="Age",x="Sex",data=dataset,kind="box")

g = sns.catplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")

g = sns.catplot(y="Age",x="Parch", data=dataset,kind="box")

g = sns.catplot(y="Age",x="SibSp", data=dataset,kind="box")
# Let's convert Sex into 0 for male and 1 for female

dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(), cmap="BrBG", annot=True)
# Filling missing value of Age 



Index_Missing_Age = list(dataset["Age"][dataset["Age"].isnull()].index)



for i in Index_Missing_Age:

    age_med = dataset["Age"].median()

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        dataset['Age'].iloc[i] = age_pred

    else :

        dataset['Age'].iloc[i] = age_med
g = sns.catplot(x="Survived", y = "Age",data = df_train, kind="box")

g = sns.catplot(x="Survived", y = "Age",data = df_train, kind="violin")
dataset["Name"].head()
# Get Title from Name

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset["Title"] = pd.Series(dataset_title)



dataset["Title"].head()
g = sns.countplot(x="Title",data=dataset)

g = plt.setp(g.get_xticklabels(), rotation=90) 
# Convert to categorical values Title 

dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)
g = sns.countplot(dataset["Title"])

g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
#Surviving probabilities of this groups

g = sns.catplot(x="Title",y="Survived",data=dataset,kind="bar")

g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])

g = g.set_ylabels("Survival Probability")
# Drop Name variable

dataset.drop(labels = ["Name"], axis = 1, inplace = True)
# Create a family size descriptor from SibSp and Parch

dataset["Family_Size"] = dataset["SibSp"] + dataset["Parch"] + 1
g = sns.catplot(x="Family_Size",y="Survived",data = dataset, kind = "point")

g = g.set_ylabels("Survival Probability")
#Create categories of the family size

dataset['Single'] = dataset['Family_Size'].map(lambda s: 1 if s == 1 else 0)

dataset['Small_Family'] = dataset['Family_Size'].map(lambda s: 1 if  s == 2  else 0)

dataset['Medium_Family'] = dataset['Family_Size'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['Large_Family'] = dataset['Family_Size'].map(lambda s: 1 if s >= 5 else 0)
g = sns.catplot(x="Single", y="Survived", data=dataset, kind="bar")

g = g.set_ylabels("Survival Probability")



g = sns.catplot(x="Small_Family", y="Survived", data=dataset, kind="bar")

g = g.set_ylabels("Survival Probability")



g = sns.catplot(x="Medium_Family", y="Survived", data=dataset, kind="bar")

g = g.set_ylabels("Survival Probability")



g = sns.catplot(x="Large_Family", y="Survived", data=dataset, kind="bar")

g = g.set_ylabels("Survival Probability")
#Convert to itle and Embarked (from long format to wide format)

dataset = pd.get_dummies(dataset, columns = ["Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset.head()
dataset["Cabin"].head()
dataset["Cabin"].describe()
dataset["Cabin"].isnull().sum()
# Replace the Cabin number by the type of cabin 'X' if it is null

dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])
g = sns.catplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])

g = g.set_ylabels("Survival Probability")
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")
dataset["Ticket"].head()
# Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 



Ticket = []

for i in list(dataset.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix

    else:

        Ticket.append("X")

        

dataset["Ticket"] = Ticket

dataset["Ticket"].head()
dataset = pd.get_dummies(dataset, columns = ["Ticket"], prefix="T")
# Create categorical values for Pclass

dataset["Pclass"] = dataset["Pclass"].astype("category")

dataset = pd.get_dummies(dataset, columns = ["Pclass"],prefix="Pc")
# Drop unnecessary variables 

dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
dataset.columns
dataset.head()
dataset['Survived'].value_counts()
# Separate the dataset into train and test.

train_len = len(df_train)



df_train = dataset[:train_len]

df_test = dataset[train_len:]

df_test.drop(labels=["Survived"],axis = 1,inplace=True)
df_train.shape
df_test.shape
# Separate train features and label 



df_train["Survived"] = df_train["Survived"].astype(int)



y_train = df_train["Survived"]



X_train = df_train.drop(labels = ["Survived"],axis = 1)
# Cross validate model with Kfold stratified cross validation

kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 

random_state = 1992

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

    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))



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
# AdaBoost



DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2],

              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsadaDTC.fit(X_train,y_train)



ada_best = gsadaDTC.best_estimator_





# Best score

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



gsExtC.fit(X_train,y_train)



ExtC_best = gsExtC.best_estimator_





# Best score

gsExtC.best_score_
# Random Forest 

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [4,5,7],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[80,100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(X_train,y_train)



RFC_best = gsRFC.best_estimator_





# Best score

gsRFC.best_score_
# Gradient Boosting



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [80,100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4,5,6,8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train,y_train)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
# SVC

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(X_train,y_train)



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



g = plot_learning_curve(gsRFC.best_estimator_,"RF Learning Curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsExtC.best_estimator_,"ExtraTrees Learning Curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsSVMC.best_estimator_,"SVC Learning Curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsadaDTC.best_estimator_,"AdaBoost Learning Curves",X_train,y_train,cv=kfold)

g = plot_learning_curve(gsGBC.best_estimator_,"GradientBoosting Learning Curves",X_train,y_train,cv=kfold)
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

        g.set_title(name + " Feature Importance")

        nclassifier += 1
voting_clas = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),

('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)



voting_clas = voting_clas.fit(X_train, y_train)
test_Survived = pd.Series(voting_clas.predict(df_test), name="Survived")



results = pd.concat([IDtest,test_Survived],axis=1)



results.to_csv("Ensemble_Python_Voting_Results.csv",index=False)