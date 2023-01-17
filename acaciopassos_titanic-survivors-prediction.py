%matplotlib inline

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
df_titanic_train = pd.read_csv('../input/train.csv')
df_titanic_test = pd.read_csv('../input/test.csv')
PassengerId = df_titanic_test["PassengerId"]
# Outlier detection 

def detect_outliers(df, n, features):
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

# detect outliers from Age, SibSp, Parch and Fare
outliers_to_drop = detect_outliers(df_titanic_train, 2, ["Age", "SibSp", "Parch", "Fare"])

# Outliers

df_titanic_train.loc[outliers_to_drop]
# Drop outliers

df_titanic_train = df_titanic_train.drop(outliers_to_drop, axis = 0).reset_index(drop=True)
# Concatenate the two dataframes to minimize the bias and to have the same columns after feature engineering

train_size = len(df_titanic_train)
df_titanic =  pd.concat(objs=[df_titanic_train, df_titanic_test], axis=0).reset_index(drop=True)
df_titanic.head()
# Filling empty values with NaN and checking the null values

df_titanic = df_titanic.fillna(np.nan)

# Survived will not be considered because the empty values are from test dataset

df_titanic.isnull().sum()
def absolute_relative_freq(variable):
    absolute_frequency = variable.value_counts()
    relative_frequency = round(variable.value_counts(normalize = True)*100, 2) 
    df = pd.DataFrame({'Absolute Frequency':absolute_frequency, 'Relative Frequency(%)':relative_frequency})
    print('Absolute and Relative Frequency of [',variable.name,']')
    display(df)
# Get some conclusion about the correlation among 'Survived' and SibSp, Parch, Age and Fare.

fig, ax = plt.subplots(figsize=(12,8))
g = sns.heatmap(
    df_titanic[["Survived", "SibSp", "Age", "Parch", "Fare"]].corr(),
    annot=True, 
    fmt = ".3f", 
    cmap = "Greens",
    ax=ax)
# View the proportion between SibSp and Survived

g = sns.factorplot(x="SibSp", 
                   y="Survived", 
                   data=df_titanic, 
                   kind="bar", 
                   size=5, 
                   palette = "Greens")

g = g.set_ylabels("Survived")
# View the distribution of Age

g = sns.FacetGrid(df_titanic, 
                  col='Survived',
                  aspect=2)

g = g.map(sns.distplot, "Age", 
          bins=20, 
          color='g', 
          hist_kws=dict(edgecolor="w", linewidth=1))
# View the proportion between Parch and Survived

g = sns.factorplot(x="Parch", 
                   y="Survived", 
                   data=df_titanic, 
                   kind="bar", 
                   size=5, 
                   palette = "Greens")

g = g.set_ylabels("Survived")
# Filling with the median the only one Fare equals to NaN

df_titanic['Fare'] = df_titanic['Fare'].fillna(df_titanic['Fare'].median())
# Viewing the Fare distribution
 
fig, ax = plt.subplots(figsize=(7,5))
g = sns.distplot(df_titanic["Fare"], 
                 color="g", 
                 label="Skewness : %.3f"%(df_titanic["Fare"].skew()), 
                 hist_kws=dict(edgecolor="w", linewidth=1),
                 ax=ax)
                 
g = g.legend(loc="best")
df_titanic["Fare"] = df_titanic["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
# Viewing the Fare distribution after applying log function
 
fig, ax = plt.subplots(figsize=(7,5))
g = sns.distplot(df_titanic["Fare"], 
                 color="g", 
                 label="Skewness : %.3f"%(df_titanic["Fare"].skew()), 
                 hist_kws=dict(edgecolor="w", linewidth=1),
                 ax=ax)
                 
g = g.legend(loc="best")
# View the proportion between Sex and Survived

g = sns.barplot(x="Sex", y="Survived",data=df_titanic, palette='cool')
g = g.set_ylabel("Survived")
absolute_relative_freq(df_titanic['Sex'])
# View the relationship between Sex and Age

g = sns.factorplot(x="Sex", 
                   y="Age", 
                   data=df_titanic, 
                   kind="box", 
                   size=5, 
                   palette = "cool")

g = g.set_ylabels("Survived")
# View the proportion between Pclass, Sex and Survived

g = sns.factorplot(x="Pclass", 
                   y="Survived", 
                   data=df_titanic,
                   hue='Sex',
                   kind="bar", 
                   size=5, 
                   palette = "cool")

g = g.set_ylabels("Survived")
absolute_relative_freq(df_titanic['Pclass'])
df_titanic['Embarked'].value_counts()
# View the proportion between Embarked and Survived

# Filling the NaN value with 'S' (more frequent city)

df_titanic['Embarked'].fillna('S', inplace=True)

g = sns.barplot(x="Embarked", y="Survived",data=df_titanic, palette='Greens')
g = g.set_ylabel("Survived")
absolute_relative_freq(df_titanic['Embarked'])
# View the proportion between Embarked, Sex and Survived

g = sns.factorplot(x="Embarked", 
                   y="Survived", 
                   data=df_titanic,
                   hue='Sex',
                   kind="bar", 
                   size=5, 
                   palette = "cool")

g = g.set_ylabels("Survived")
# View the proportion between Embarked, Age less than 10 and Survived

g = sns.factorplot(x="Embarked", 
                   y="Survived", 
                   data=df_titanic[df_titanic['Age'] < 10] ,
                   kind="bar", 
                   size=5, 
                   palette = "Greens")

g = g.set_ylabels("Survived")
# View the proportion between Pclass and Embarked

g = sns.factorplot("Pclass",
                   col="Embarked",
                   data=df_titanic,
                   size=5, 
                   kind="count", 
                   palette="Greens")

g = g.set_ylabels("Count")
labelEncoder = LabelEncoder()
df_titanic['Embarked'] = labelEncoder.fit_transform(df_titanic['Embarked'])
df_titanic['Age'].isnull().sum()
labelEncoder = LabelEncoder()
df_titanic['Sex'] = labelEncoder.fit_transform(df_titanic['Sex'])
fig, ax = plt.subplots(figsize=(12,8))
g = sns.heatmap(
    df_titanic[["Age", "SibSp", "Parch", "Pclass", "Sex"]].corr(),
    annot=True, 
    fmt = ".3f", 
    cmap = "Greens",
    ax=ax)
df_titanic['Age'].isnull().sum()
# Rows with Age equals to NaN

condition = df_titanic['Age'].isnull()
age_NaN = df_titanic['Age'][condition].index

for age in age_NaN :
    
    # Conditions
    
    condition1 = df_titanic['SibSp'] == df_titanic.iloc[age]["SibSp"]
    condition2 = df_titanic['Pclass'] == df_titanic.iloc[age]["Pclass"]
    condition3 = df_titanic['Parch'] == df_titanic.iloc[age]["Parch"]
    condition = condition1 & condition2 & condition3
    
    new_age = df_titanic['Age'][condition].median()
    df_titanic['Age'].iloc[age] = new_age if not np.isnan(new_age) else df_titanic['Age'].median()
df_titanic['Age'].isnull().sum()
df_titanic['Age'] = (df_titanic['Age'] - df_titanic['Age'].mean()) / df_titanic['Age'].std()
# Viewing the Age distribution
 
fig, ax = plt.subplots(figsize=(7,5))
g = sns.distplot(df_titanic["Age"], 
                 color="g", 
                 label="Skewness : %.3f"%(df_titanic["Age"].skew()), 
                 hist_kws=dict(edgecolor="w", linewidth=1),
                 ax=ax)
                 
g = g.legend(loc="best")
df_titanic['Name'].head()
# Getting the titles from Name feature

df_titanic['Title'] = df_titanic['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df_titanic['Title'].unique()
# Replace the values to new categories and converting the new feature to numeric

df_titanic['Title'] = df_titanic['Title'].replace(['Don', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 
                                                   'Col', 'Capt', 'Countess', 'Jonkheer', 'Dona'], 'Rare')
# View the proportion between Title and Survived

g = sns.factorplot(x="Title", 
                   y="Survived", 
                   data=df_titanic,
                   kind="bar", 
                   size=5)

g = g.set_ylabels("Survived")
absolute_relative_freq(df_titanic['Title'])
df_titanic["Title"] = df_titanic['Title'].map({"Master":0, "Miss":1, "Ms" : 1, "Mme":1, 
                                               "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

df_titanic['Title'] = df_titanic["Title"].astype(int)
# Creating a column with the Surname

df_titanic['Surname'] = df_titanic['Name'].map(lambda i: i.split(',')[0])
# Deleting Name

del df_titanic['Name']
df_titanic['Cabin'].isnull().sum()
df_titanic['Cabin'].unique()
df_titanic['Cabin'] = df_titanic['Cabin'].map(lambda i: i[0] if not pd.isnull(i) else 'Z')
df_titanic['Cabin'].unique()
# View the proportion between Cabin and Survived

g = sns.factorplot(x="Cabin", 
                   y="Survived", 
                   data=df_titanic,
                   kind="bar", 
                   size=5, 
                   order=['A','B','C','D','E','F','G','T','Z'])

g = g.set_ylabels("Survived")
absolute_relative_freq(df_titanic['Cabin'])
df_titanic['Ticket'].unique()
# Getting the first information of the ticket

df_titanic['Ticket'] = df_titanic['Ticket'].map(
    lambda i: i.replace(".","").replace("/","").strip().split(' ')[0] if not i.isdigit() else "TKT")
df_titanic['Ticket'].unique()
df_titanic['Family'] = df_titanic['SibSp'] + df_titanic['Parch'] + 1
df_titanic['Family'].unique()
# Creating new features: 
#   1: Alone
#   2: Small family
#   3 to 4: Medium family
#   larger than 5: Large family

df_titanic['Alone'] = df_titanic['Family'].map(lambda i: 1 if i == 1 else 0)
df_titanic['Small'] = df_titanic['Family'].map(lambda i: 1 if i == 2 else 0)
df_titanic['Medium'] = df_titanic['Family'].map(lambda i: 1 if 3 <= i <= 4 else 0)
df_titanic['Large'] = df_titanic['Family'].map(lambda i: 1 if i >= 5 else 0)
df_titanic.head()
# df_titanic['Title'] = df_titanic['Title'].astype("category")
df_titanic['Pclass'] = df_titanic['Pclass'].astype("category")

# Creating dummies...

columns = ['Title', 'Surname', 'Cabin', 'Ticket', 'Pclass']
for col in columns:
    df_titanic = pd.get_dummies(df_titanic, columns=[col], prefix=col)
df_titanic.drop(labels = ["PassengerId"], axis = 1, inplace = True)
df_titanic.head()
df_titanic_train = df_titanic[:train_size]
df_titanic_test = df_titanic[train_size:]

df_titanic_train['Survived'] = df_titanic_train['Survived'].astype(int)
del df_titanic_test['Survived']
X_train = df_titanic_train.drop(['Survived'], axis=1)
y_train = df_titanic_train['Survived']
X_test = df_titanic_test

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)
print(X_test.shape)
# Validation of the model with Kfold stratified splitting the data into 10 parts

kfold = StratifiedKFold(n_splits=10)

seed = 20

# List of classifiers to test

clfs = []
clfs.append(SVC(random_state=seed))
clfs.append(DecisionTreeClassifier(random_state=seed))
clfs.append(RandomForestClassifier(random_state=seed))
clfs.append(ExtraTreesClassifier(random_state=seed))
clfs.append(GradientBoostingClassifier(random_state=seed))
clfs.append(MLPClassifier(random_state=seed))
clfs.append(KNeighborsClassifier())
clfs.append(LogisticRegression(random_state=seed))
clfs.append(XGBClassifier(random_state = seed))
# Getting all results from 10 validations for each classifier

clf_results = []
for clf in clfs :
    clf_results.append(cross_val_score(clf, X_train, y=y_train, scoring = "accuracy", cv=kfold, n_jobs=1))
# Getting the mean and standard deviation from each classifier's result after 10 validations

clf_means = []
clf_std = []
for clf_result in clf_results:
    clf_means.append(clf_result.mean())
    clf_std.append(clf_result.std())
# Let's see which are the best scores

df_result = pd.DataFrame({"Means":clf_means, 
                          "Stds": clf_std, 
                          "Algorithm":["SVC", 
                                       "DecisionTree", 
                                       "RandomForest",
                                       "ExtraTrees",
                                       "GradientBoosting",
                                       "MLPClassifier",
                                       "KNeighboors",
                                       "LogisticRegression", 
                                       "XGBoost"]})

df_result.sort_values(by=['Means'], ascending=False)
# Plotting learning curves of the algorithms
#------------------------------------------------------------------------------------------------
# Code from http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
#------------------------------------------------------------------------------------------------

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 20)):
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
# Logistic Regression, XGBoost, Gradient Boosting, Random Forest, Extra Trees

extraTrees = ExtraTreesClassifier(random_state=seed)
gBoosting = GradientBoostingClassifier(random_state=seed)
randomForest = RandomForestClassifier(random_state=seed)
logReg = LogisticRegression(random_state=seed)
xgbc = XGBClassifier(random_state=seed)
clfs = []
clfs.append(extraTrees)
clfs.append(gBoosting)
clfs.append(randomForest)
clfs.append(logReg)
clfs.append(xgbc)

titles = ['Learning Curves (Extra Tree)', 'Learning Curves (Gradient Boosting)',
          'Learning Curves (Random Forest)', 'Learning Curves (Logistic Regression)',
          'Learning Curves (XGBoost)']

for clf, title in zip(clfs, titles):
    plot_learning_curve(clf, title, X_train, y_train, ylim=(0.7, 1.01), cv=kfold, n_jobs=1);
## Search grid for optimal parameters (Extra Trees)

param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


grid_result = GridSearchCV(extraTrees,
                      param_grid = param_grid, 
                      cv=kfold, 
                      scoring="accuracy", 
                      n_jobs= -1, 
                      verbose = 1)

grid_result.fit(X_train,y_train)

extraTrees_best_result = grid_result.best_estimator_

# Best score
print('Best score:', np.round(grid_result.best_score_*100, 2))

# Best estimator
print('Best estimator:', extraTrees_best_result)
## Search grid for optimal parameters (Gradient Boosting)

param_grid = {'learning_rate': [0.01, 0.02],
              'max_depth': [4, 5, 6],
              'max_features': [0.2, 0.3, 0.4], 
              'min_samples_split': [2, 3, 4],
              'random_state':[seed]}

grid_result = GridSearchCV(gBoosting, 
                           param_grid=param_grid, 
                           cv=kfold, 
                           scoring="accuracy", 
                           n_jobs=-1,
                           verbose=1)

grid_result.fit(X_train, y_train)

gBoosting_best_result = grid_result.best_estimator_

# Best score
print('Best score:', np.round(grid_result.best_score_*100, 2))

# Best estimator
print('Best estimator:', gBoosting_best_result)
## Search grid for optimal parameters (Random Forest)

param_grid = {"max_depth": [None],
              "max_features": [1, 2],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

grid_result = GridSearchCV(randomForest, 
                           param_grid=param_grid, 
                           cv=kfold, 
                           scoring="accuracy", 
                           n_jobs= -1,
                           verbose = 1)

grid_result.fit(X_train, y_train)

randomForest_best_result = grid_result.best_estimator_

# Best score
print('Best score:', np.round(grid_result.best_score_*100, 2))

# Best estimator
print('Best estimator:', randomForest_best_result)
## Search grid for optimal parameters (Logistic Regression)

param_grid = {'penalty' : ['l1', 'l2'],
              'C': np.logspace(0, 4, 10),
              'solver' : ['liblinear', 'saga']
              }

grid_result = GridSearchCV(logReg, 
                           param_grid=param_grid, 
                           cv=kfold, 
                           scoring="accuracy", 
                           n_jobs= -1,
                           verbose = 1)

grid_result.fit(X_train, y_train)

logReg_best_result = grid_result.best_estimator_

# Best score
print('Best score:', np.round(grid_result.best_score_*100, 2))

# Best estimator
print('Best estimator:', logReg_best_result)
## Search grid for optimal parameters (XGBoost)

param_grid = {'n_estimators': [275, 280],
              'learning_rate': [0.01, 0.03],
              'subsample': [0.9, 1],
              'max_depth': [3, 4],
              'colsample_bytree': [0.8, 0.9],
              'min_child_weight': [2, 3],
              'random_state':[seed]}

grid_result = GridSearchCV(xgbc, 
                           param_grid=param_grid, 
                           cv=kfold, 
                           scoring="accuracy", 
                           n_jobs= -1,
                           verbose = 1)

grid_result.fit(X_train, y_train)

xgbc_best_result = grid_result.best_estimator_

# Best score
print('Best score:', np.round(grid_result.best_score_*100, 2))

# Best estimator
print('Best estimator:', xgbc_best_result)
# List of classifiers to retrain

clfs = []
clfs.append(extraTrees_best_result)
clfs.append(gBoosting_best_result)
clfs.append(randomForest_best_result)
clfs.append(logReg_best_result)
clfs.append(xgbc_best_result)

# Getting all results from 10 validations for each classifier

clf_results = []
for clf in clfs :
    clf_results.append(cross_val_score(clf, X_train, y=y_train, scoring = "accuracy", cv=kfold, n_jobs=1))

# Getting the mean and standard deviation from each classifier's result after 10 validations

clf_means = []
clf_std = []
for clf_result in clf_results:
    clf_means.append(clf_result.mean())
    clf_std.append(clf_result.std())

# Let's see which are the best scores

df_result = pd.DataFrame({"Means":clf_means, 
                          "Stds": clf_std, 
                          "Algorithm":["Extra Trees",
                                       "GradientBoosting",
                                       "Random Forest",
                                       "LogisticRegression", 
                                       "XGBoost"]})

df_result.sort_values(by=['Means'], ascending=False)
titles = ['Learning Curves (Extra Tree)', 'Learning Curves (Gradient Boosting)',
          'Learning Curves (Random Forest)', 'Learning Curves (Logistic Regression)',
          'Learning Curves (XGBoost)']

for clf, title in zip(clfs, titles):
    plot_learning_curve(clf, title, X_train, y_train, ylim=(0.7, 1.01), cv=kfold, n_jobs=1);
survived_ET = pd.Series(extraTrees_best_result.predict(X_test), name="ET")
survived_GB = pd.Series(gBoosting_best_result.predict(X_test), name="GB")
survived_RF = pd.Series(randomForest_best_result.predict(X_test), name="RF")
survived_LR = pd.Series(logReg_best_result.predict(X_test), name="LR")
survived_XB = pd.Series(xgbc_best_result.predict(X_test), name="XB")

# Concatenate all classifiers results
ensemble_results = pd.concat([survived_ET,
                              survived_GB,
                              survived_RF,
                              survived_LR,
                              survived_XB],
                             axis=1)

fig, ax = plt.subplots(figsize=(12,8))
g= sns.heatmap(ensemble_results.corr(),
               annot=True, 
               fmt = ".3f", 
               cmap = "Greens",
               ax=ax)
# Using voting soft (XB, GB, RF, LR, and ET)

voting = VotingClassifier(estimators=[('XB', xgbc_best_result), 
                                      ('GB', gBoosting_best_result),
                                      ('RF', randomForest_best_result),
                                      ('LR', logReg_best_result),
                                      ('ET', extraTrees_best_result)],
                           voting='soft', n_jobs=-1)
voting.fit(X_train, y_train)
print("Score (Voting): " + str(voting.score(X_train, y_train)))
# Predicting survivors

y_predict = voting.predict(X_test)
solution = pd.DataFrame({
                        "PassengerId": PassengerId,
                        "Survived": y_predict.astype(int)
                        })

solution.to_csv('solution_final_v1.csv', index=False)
df_solution = pd.read_csv('solution_final_v1.csv')
absolute_relative_freq(df_solution['Survived'])
