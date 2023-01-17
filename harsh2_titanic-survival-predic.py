import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re

%matplotlib inline
rcParams['figure.figsize'] = 10,8

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


print(df_train.info())

print(df_test.info())

df_train.describe()
#Take a look at the data
print(df_train.head())

df_train["Name"].head(10)

df_train['Title'] = df_train.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
sns.countplot(x='Title', data=df_train, palette="hls")
plt.xticks(rotation=45)
plt.show()

df_test['Title'] = df_test.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
#Now i identify the sicial status of employee
Title_Dictionary = {
        "Capt":       "Officer",
        "Col":        "Officer",
        "Major":      "Officer",
        "Dr":         "Officer",
        "Rev":        "Officer",
        "Jonkheer":   "Royalty",
        "Don":        "Royalty",
        "Sir" :       "Royalty",
        "the Countess":"Royalty",
        "Dona":       "Royalty",
        "Lady" :      "Royalty",
        "Mme":        "Mrs",
        "Ms":         "Mrs",
        "Mrs" :       "Mrs",
        "Mlle":       "Miss",
        "Miss" :      "Miss",
        "Mr" :        "Mr",
        "Master" :    "Master"}


# we map each title to correct category
df_train['Title'] = df_train.Title.map(Title_Dictionary)
df_test['Title'] = df_test.Title.map(Title_Dictionary)

print("Chances to survive based on titles: ")
print(df_train.groupby("Title")["Survived"].mean())

#Plotting the results
sns.countplot(x='Title', data=df_train, palette="hls",hue="Survived")
plt.xticks(rotation=45)
plt.show()

age_high_zero_died = df_train[(df_train["Age"] > 0) & 
                              (df_train["Survived"] == 0)]
age_high_zero_surv = df_train[(df_train["Age"] > 0) & 
                              (df_train["Survived"] == 1)]


sns.distplot(age_high_zero_surv["Age"], bins=24, color='green')
sns.distplot(age_high_zero_died["Age"], bins=24, color='red')
plt.title("Distribuition and density by Age",fontsize=15)
plt.xlabel("Age",fontsize=12)
plt.ylabel("Density Died and Survived",fontsize=12)
plt.show()

age_group = df_train.groupby(["Sex","Pclass","Title"])["Age"]

print(age_group.median())

df_train.loc[df_train.Age.isnull(), 'Age'] = df_train.groupby(['Sex','Pclass','Title']).Age.transform('median')

print(df_train["Age"].isnull().sum())


#Let's see the result of the inputation
sns.distplot(df_train["Age"], bins=24)
plt.title("Distribuition and density by Age")
plt.xlabel("Age")
plt.show()

#separate by survivors or not
g = sns.FacetGrid(df_train, col='Survived',size=5)
g = g.map(sns.distplot, "Age")
plt.show()

df_train.Age = df_train.Age.fillna(-0.5)


interval = (0, 5, 12, 18, 25, 35, 60, 120)
cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']

df_train["Age_cat"] = pd.cut(df_train.Age, interval, labels=cats)



df_train["Age_cat"].head()

interval = (0, 5, 12, 18, 25, 35, 60, 120)
cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']

df_test["Age_cat"] = pd.cut(df_test.Age, interval, labels=cats)
print(pd.crosstab(df_train.Age_cat, df_train.Survived))
#Plotting the result
sns.countplot("Age_cat",data=df_train,hue="Survived", palette="hls")
plt.xlabel("Categories names")
plt.title("Age Distribution ")

sns.distplot(df_train[df_train.Survived == 1]["Fare"], 
             bins=50, color='g')
sns.distplot(df_train[df_train.Survived == 0]["Fare"], 
             bins=50, color='r')
plt.title("Fare Distribuition by Survived", fontsize=15)
plt.xlabel("Fare", fontsize=12)
plt.ylabel("Density",fontsize=12)
plt.show()

df_train.Fare = df_train.Fare.fillna(-0.5)
quant = (-1, 0, 8, 15, 31, 600)
#Labels without input values
label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']

#doing the cut in fare and puting in a new column
df_train["Fare_cat"] = pd.cut(df_train.Fare, quant, labels=label_quants)

#Description of transformation
print(pd.crosstab(df_train.Fare_cat, df_train.Survived))

#Plotting the new feature
sns.countplot(x="Fare_cat", hue="Survived", data=df_train, palette="hls")
plt.title("Count of survived x Fare expending")

# Replicate the same to df_test
df_test.Fare = df_test.Fare.fillna(-0.5)

quant = (-1, 0, 8, 15, 31, 1000)
label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']

df_test["Fare_cat"] = pd.cut(df_test.Fare, quant, labels=label_quants)

#Now lets drop the variable Fare, Age and ticket that is irrelevant now
del df_train["Fare"]
del df_train["Ticket"]
del df_train["Age"]
del df_train["Cabin"]
del df_train["Name"]

#same in df_test
del df_test["Fare"]
del df_test["Ticket"]
del df_test["Age"]
del df_test["Cabin"]
del df_test["Name"]

#Looking the result of transformations
df_train.head()

# Let see how many people die or survived
print("Total of Survived or not: ")
print(df_train.groupby("Survived")["PassengerId"].count())
sns.countplot(x="Survived", data=df_train,palette="hls")
plt.title('Total Distribuition by survived or not')


print(pd.crosstab(df_train.Survived, df_train.Sex))
sns.countplot(x="Sex", data=df_train, hue="Survived",palette="hls")
plt.title('Sex Distribuition by survived or not')

# Distribuition by class
print(pd.crosstab(df_train.Pclass, df_train.Embarked))
sns.countplot(x="Embarked",data=df_train, hue="Pclass",palette="hls")
plt.title('Embarked x Pclass')


#lets input the NA's with the highest frequency
df_train["Embarked"] = df_train["Embarked"].fillna('S')
# Exploring Survivors vs Embarked
print(pd.crosstab(df_train.Survived, df_train.Embarked))
sns.countplot(x="Embarked", data=df_train, hue="Survived",palette="hls")
plt.title('Class Distribuition by survived or not')


# Exploring Survivors vs Pclass
print(pd.crosstab(df_train.Survived, df_train.Pclass))
sns.countplot(x="Pclass", data=df_train, hue="Survived",palette="hls")
plt.title('Class Distribuition by survived or not')

g = sns.factorplot(x="SibSp",y="Survived",data=df_train,kind="bar", size = 6, palette = "hls")
g = g.set_ylabels("Probability(Survive)")

g  = sns.factorplot(x="Parch",y="Survived",data=df_train, kind="bar", size = 6,palette = "hls")
g = g.set_ylabels("survival probability")

#Create a new column and sum the Parch + SibSp + 1 that refers the people self
df_train["FSize"] = df_train["Parch"] + df_train["SibSp"] + 1

df_test["FSize"] = df_test["Parch"] + df_test["SibSp"] + 1
print(pd.crosstab(df_train.FSize, df_train.Survived))


sns.factorplot(x="FSize",y="Survived", data=df_train, kind="bar",size=6)
plt.show()
del df_train["SibSp"]
del df_train["Parch"]

del df_test["SibSp"]
del df_test["Parch"]

df_train.head()
df_train = pd.get_dummies(df_train, columns=["Sex","Embarked","Age_cat","Fare_cat","Title"],\
                          prefix=["Sex","Emb","Age","Fare","Prefix"], drop_first=True)

df_test = pd.get_dummies(df_test, columns=["Sex","Embarked","Age_cat","Fare_cat","Title"],\
                         prefix=["Sex","Emb","Age","Fare","Prefix"], drop_first=True)

#Finallt, lets look the correlation of df_train
plt.figure(figsize=(15,12))
plt.title('Correlation of Features for Train Set')
sns.heatmap(df_train.astype(float).corr(),vmax=1.0,  annot=True)
plt.show()

train = df_train.drop(["Survived","PassengerId"],axis=1)
train_ = df_train["Survived"]

test_ = df_test.drop(["PassengerId"],axis=1)

X = train.values
y = train_.values

X_test = test_.values
X_test = X_test.astype(np.float64, copy=False)

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve, KFold


#I have used the gridSearchCV to find the best parameters
parameters = {"max_depth": [4, None],
              "max_features": [6, 10,15,25],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [10,15,30],
              "bootstrap": [False],
              "n_estimators" :[15,30,50,100,300],
              "criterion": ["gini","entropy"]
             }

#param to use in kaggle
parameters_test = {"max_depth": [None,5], "n_estimators":[5,10,25]}

#Creating a RandomForestClassifier
rfc = RandomForestClassifier()

#I'm using the "parameters_test", just because the time to execute the code
rfc_cv = GridSearchCV(rfc, parameters_test, cv=5,verbose=1)


#fitting he model to training data
model_rf = rfc_cv.fit(X, y)


print("Best score: ", rfc_cv.best_score_)
print("Best params: ", rfc_cv.best_params_)

# XGBClassifier model
param_xgb = {
 'max_depth':range(3,15,2),
 'min_child_weight':range(1,12,2),
 'colsample_bytree':[i/10.0 for i in range(6,10)],
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100, 0.001, 0.005, 0.01, 0.05],
 'n_estimators':range(10,500,50)
}

#param to use in kaggle
param_xgb_test = {'max_depth':range(3,15,2)}

#Calling XGBClassier algorithm 
XGB_class = XGBClassifier()

#Crossing all params searching for a best params 
XGB = GridSearchCV(XGB_class, param_grid=param_xgb_test, cv=5, verbose=1)

#fiting it to X y
model_xgb = XGB.fit(X, y)

#printing best params and socre
print("Best params: ", model_xgb.best_params_)
print("Best score: ", model_xgb.best_score_)

#LogisticRegression Model

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {"C": range(1,50,3), 
              'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5, verbose=1)

# Fit it to the training data
model_logreg = logreg_cv.fit(X, y)

# Print the optimal parameters and best score
print("Best params: ", model_logreg.best_params_)
print("Best score: ", model_logreg.best_score_)

#Getting the best estimators to fit the learning curve
model_rf_best = model_rf.best_estimator_
model_logreg_best = model_logreg.best_estimator_
model_xgb_best = XGB.best_estimator_

#Creating a fold to validation in graph
kfold = KFold(n_splits=5)



#Lets look the learning curve off each of our models
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
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

g = plot_learning_curve(model_rf_best,"RF mearning curves",X,y,cv=kfold)
g = plot_learning_curve(model_logreg_best,"LogReg learning curves",X,y,cv=kfold)
g = plot_learning_curve(model_xgb_best,"EXBoosting learning curves",X,y,cv=kfold)





#Use the Voting Classifier using the estimators to cross the results
votingC = VotingClassifier(estimators=[('RFC', model_rf_best), ('LogReg', model_logreg_best),('XGBC', model_xgb_best)], voting='soft', n_jobs=1)

votingC = votingC.fit(X, y)

#Use the Voting Classifier using the estimators to cross the results
votingC = VotingClassifier(estimators=[('RFC', model_rf_best), ('LogReg', model_logreg_best),('XGBC', model_xgb_best)], voting='soft', n_jobs=1)

votingC = votingC.fit(X, y)

test_Survived = pd.Series(votingC.predict(X_test), name="Survived")

df_test["Survived"] = test_Survived


