# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# preview the data

titanic_df.head()
train = titanic_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

test    = test_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)

#We also add a 1/0 variable for Sex

df_sex=pd.get_dummies(train['Sex'],drop_first=True)

train=train.join(df_sex)



df_sex_2=pd.get_dummies(test['Sex'],drop_first=True)

test=test.join(df_sex_2)

#Also df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)





#Dummies for Pclass too



df_pclass=pd.get_dummies(train['Pclass'],prefix='Class').astype(int)

train=train.join(df_pclass)



df_pclass_2=pd.get_dummies(test['Pclass'],prefix='Class').astype(int)

test=test.join(df_pclass_2)

train.head()
#Deal with age



avg_age_train=train['Age'].mean()

std_age_train=train['Age'].std()

nans_age_train=train['Age'].isnull().sum()



avg_age_test=test['Age'].mean()

std_age_test=test['Age'].std()

nans_age_test=test['Age'].isnull().sum()



#Generate random ages

rand_1 = np.random.randint(avg_age_train-std_age_train,avg_age_train+std_age_train,size=nans_age_train)

rand_2 = np.random.randint(avg_age_test-std_age_test,avg_age_test+std_age_test,size=nans_age_test)



#Fill NaNs

#train["Age"][np.isnan(train["Age"])] = rand_1

#test["Age"][np.isnan(test["Age"])] = rand_2

#Median better than mean to avoid outliers

train['Age'].fillna(train['Age'].median(), inplace=True)

test['Age'].fillna(test['Age'].median(), inplace=True)

#Check

np.all(~np.isnan(train["Age"]))
corrmat=train[['Survived','Class_1','Class_2','Class_3','SibSp','Parch','Fare','male','Age']].corr()

print(corrmat)

sns.heatmap(corrmat,vmax=.8,annot=True)
sns.countplot(x="Embarked", data=train);

plt.figure();

sns.barplot(x='Embarked',y="Survived", data=train);
train['Embarked']=train['Embarked'].fillna('S')

test['Embarked']=test['Embarked'].fillna('S')



df_em=pd.get_dummies(train['Embarked'],prefix='Embarked')
# First, I need to extract the fares and  group them by Pclass, then plot.

#pclass_fare=train[['Fare','Pclass']]

#print(pclass_fare.groupby(['Pclass']).describe())

#print(pclass_fare.groupby(['Pclass']).mean())

#sns.boxplot(x="Pclass", y="Fare", data=pclass_fare);
sns.distplot(train[train['Class_1']==1]['Fare'])

plt.figure()

sns.distplot(train[train['Class_2']==1]['Fare'])

sns.distplot(train[train['Class_3']==1]['Fare'])
# result = sm.ols(formula="Survived ~Embarked_Q+Embarked_C+Embarked_S ", data=train).fit()

# print(result.params)

# print (result.summary())

#R2 too small, disregard.
# define training and testing sets

#There happens to be one missing element in Fare. So let's fix that

test["Fare"].fillna(test["Fare"].median(), inplace=True)

X_train = train[['Pclass','Fare','male','Age']]

Y_train = train[["Survived"]]

X_test  = test[['Pclass','Fare','male','Age']]
# Logistic Regression

# machine learning

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()



logreg.fit(X_train, Y_train.values.ravel())



Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
train['FamilySize'] = train['SibSp'] + train['Parch']

test['FamilySize'] = test['SibSp'] + test['Parch']



# If greater than zero, it'll be one. If not, zero

train['HaveFamily'] = 0

test['HaveFamily'] = 0

test['FamilySize'] = test['SibSp'] + test['Parch']

train['FamilySize'] = train['SibSp'] + train['Parch']



test.loc[test['FamilySize']>0,'HaveFamily']=1

train.loc[train['FamilySize']>0,'HaveFamily']=1



#Childs

train['child'] = 0

test['child'] = 0

test.loc[test['Age']<15,'child']=1

train.loc[train['Age']<15,'child']=1

predictors=['Parch','SibSp','Fare','male','Age','FamilySize','HaveFamily','Class_1','Class_2','Class_3','child']

X_train = train[predictors]



Y_train = train[["Survived"]]

X_test  = test[predictors]
corrmat=train[predictors].corr()

print(corrmat)



a=sns.heatmap(corrmat,annot=True)
logreg = LogisticRegression()



logreg.fit(X_train, Y_train.values.ravel())



Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)
# Import the random forest package

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import learning_curve





# Create the random forest object which will include all the parameters

# for the fit

forest=RandomForestClassifier(max_features='sqrt',max_depth=8,n_estimators=240)



# Fit the training data to the Survived labels and create the decision trees

forest.fit(X_train,Y_train.values.ravel())



# Take the same decision trees and run it on the test data

Y_pred = forest.predict(X_test)

forest.score(X_train, Y_train)

# What features are important?

features = pd.DataFrame()

features['feature'] = X_train.columns

features['importance'] = forest.feature_importances_

features.sort(['importance'],ascending=False)
from sklearn.model_selection import validation_curve

X, y = X_train,Y_train['Survived']

#X, y = train_new,Y_train['Survived']

def cross_val(model,X,y,pname):

    param_range = np.array([100,110,140,150])

    train_scores, test_scores = validation_curve(

        model, X, y, param_name=pname, param_range=param_range, scoring="accuracy", n_jobs=4)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    plt.title("Validation Curve with Random Forest")

    plt.xlabel("Number of estimators")

    plt.ylabel("Score")

    lw = 2

    plt.plot(param_range, train_scores_mean, label="Training score",

                 color="darkorange", lw=lw)

    plt.fill_between(param_range, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.2,

                     color="darkorange", lw=lw)

    plt.plot(param_range, test_scores_mean, label="Cross-validation score",

                 color="navy", lw=lw)

    plt.fill_between(param_range, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.2,

                     color="navy", lw=lw)

    plt.legend(loc="best")

    plt.show()

cross_val(RandomForestClassifier(),X,y,'n_estimators')
from sklearn.model_selection import learning_curve

from sklearn.model_selection import ShuffleSplit





def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    """

    Generate a simple plot of the test and training learning curve.



    Parameters

    ----------

    estimator : object type that implements the "fit" and "predict" methods

        An object of that type which is cloned for each validation.



    title : string

        Title for the chart.



    X : array-like, shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples and

        n_features is the number of features.



    y : array-like, shape (n_samples) or (n_samples, n_features), optional

        Target relative to X for classification or regression;

        None for unsupervised learning.



    ylim : tuple, shape (ymin, ymax), optional

        Defines minimum and maximum yvalues plotted.



    cv : int, cross-validation generator or an iterable, optional

        Determines the cross-validation splitting strategy.

        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,

          - integer, to specify the number of folds.

          - An object to be used as a cross-validation generator.

          - An iterable yielding train/test splits.



        For integer/None inputs, if ``y`` is binary or multiclass,

        :class:`StratifiedKFold` used. If the estimator is not a classifier

        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.



        Refer :ref:`User Guide <cross_validation>` for the various

        cross-validators that can be used here.



    n_jobs : integer, optional

        Number of jobs to run in parallel (default 1).

    """

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







title = "Learning Curves (RandomForest)"

# Cross validation with 100 iterations to get smoother mean test and train

# score curves, each time with 20% data randomly selected as a validation set.

#cv = ShuffleSplit(n_splits=20, test_size=0.33, random_state=0)



#estimator=RandomForestClassifier(n_estimators=100)

#plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

# fit model no training data

model = XGBClassifier()

model_params = {

    'learning_rate': [0.05, 0.1],

    'n_estimators': [100,150,200],

    'max_depth': [2, 3,7, 10],

}

cv = StratifiedKFold()

cv.get_n_splits(X_train,Y_train)

grid= GridSearchCV(model,model_params,scoring='roc_auc',cv=cv,verbose=2)

best_params={'n_estimators': 100, 'max_depth': 2, 'learning_rate': 0.05}

#grid.fit(X_train, Y_train.values.ravel())

model = XGBClassifier(n_estimators= 100, max_depth= 2, learning_rate= 0.05)

model.fit(X_train,Y_train.values.ravel())



# Take the same decision trees and run it on the test data

Y_pred = model.predict(X_test)

#Y_pred = [round(value) for value in Y_pred]
#cross_val(XGBClassifier(),X,y,'n_estimators')

#About 200 seems okay
title = "Learning Curves (RandomForest)"

# Cross validation with 100 iterations to get smoother mean test and train

# score curves, each time with 20% data randomly selected as a validation set.

cv = ShuffleSplit(n_splits=10, test_size=0.33, random_state=0)





plot_learning_curve(model, title, X, y, ylim=(0.7, 1.01), cv=cv)
test.head()
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)