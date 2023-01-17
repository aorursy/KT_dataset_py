import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale=2.5)



import missingno as msno



#ignore warnings

import warnings

warnings.filterwarnings('ignore') 



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
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



# separate dataset
# Outlier detection by IQR

def detect_outliers(df, n, features):

    outlier_indices = []

    for col in features:

        Q1 = np.percentile(df[col], 25)

        Q3 = np.percentile(df[col], 75)

        IQR = Q3 - Q1

        

        outlier_step = 1.5 * IQR

        

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

        

    return multiple_outliers

        

Outliers_to_drop = detect_outliers(df_train, 2, ["Age", "SibSp", "Parch", "Fare"])
# check the row where the outlier was found

df_train.loc[Outliers_to_drop]
# remove outlier

df_train = df_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
df_train.head(10)
df_train.describe()

# statistical figures representation of data of train data
df_test.describe()

# statistical figures representation of data of test data
df_train.columns

# attribute of each column
for col in df_train.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)

    

    # The process of identifying the missing value of each column
for col in df_test.columns:

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_test[col].isnull().sum() / df_test[col].shape[0]))

    print(msg)
msno.matrix(df=df_train.iloc[:, :], figsize=(8,8), color=(0.1, 0.6, 0.8))



# msno.matrix creates the same matrix as shown below 

# empty space NULL data 
msno.bar(df=df_train.iloc[:, :], figsize=(8,8), color=(0.1, 0.6, 0.8))



# It makes with the graph of the bar type.
f, ax = plt.subplots(1,2, figsize = (18,8)) # 도화지를 준비(행,열,사이즈)



df_train['Survived'].value_counts().plot.pie(explode = [0, 0.1], autopct = '%1.1f%%', ax=ax[0], shadow = True)

# It draws a series-type pieplot. 

ax[0].set_title('Pie plot - Survived')

# Set the title for the first plot

ax[0].set_ylabel('')

# Set the ylabel for the first plot

sns.countplot('Survived', data = df_train, ax=ax[1])

#  Draw a count plot.

ax[1].set_title('Count plot - Survived')

# Set the title for the count plot

plt.show()



# The ratio of survival 0 to 1 of the train set is shown graphically.
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = True).count()



# Tie with groupby and count how many counts.
pd.crosstab(df_train['Pclass'], df_train['Survived'], margins = True).style.background_gradient(cmap='Pastel1')

# margin show total
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index = True).mean().sort_values(by='Survived', ascending = False).plot.bar()



# It means Survival rate
y_position = 1.02

f, ax = plt.subplots(1, 2, figsize= (18,8))

df_train["Pclass"].value_counts().plot.bar(color = ["#CD7F32", "#FFDF00", "#D3D3D3"], ax = ax[0])

ax[0].set_title("Number of passengers By Pclass")

ax[0].set_ylabel("Count")

sns.countplot("Pclass", hue = "Survived", data = df_train, ax = ax[1])

ax[1].set_title("Pclass: Survived vs Dead", y = y_position)

plt.show()

     

# The number of passengers and the survival rate according to the Passenger Class can be known.3 class (I think it is economy) was the most on board, and FirstClass passengers had the highest survival rate.
print("the oldest passenger : {:.1f} years".format(df_train["Age"].max()))

print("the youngest passenger : {:.1f} years".format(df_train["Age"].min()))

print("average of passengers age : {:.1f} years".format(df_train["Age"].mean()))

      
fix, ax = plt.subplots(1, 1, figsize = (9, 5))

sns.kdeplot(df_train[df_train["Survived"] == 1]["Age"], ax=ax)

sns.kdeplot(df_train[df_train["Survived"] == 0]["Age"], ax=ax)    

plt.legend(["Survived == 1", "Survived == 0"])

plt.show()



# kdeplot is used to estimate the distribution of data.
fix, ax = plt.subplots(1, 1, figsize = (9, 7))

sns.kdeplot(df_train[df_train["Pclass"] == 1]["Age"], ax=ax)

sns.kdeplot(df_train[df_train["Pclass"] == 2]["Age"], ax=ax)

sns.kdeplot(df_train[df_train["Pclass"] == 3]["Age"], ax=ax)

plt.xlabel("Age")

plt.title("Age Distribution within classes")

plt.legend(["1st Class", "2nd Class", "3rd Class"])

plt.show()                       



# In this situation, if you use histogram, you can use kde because you can not overlap

# The Age distribution according to the Class can be known.
fig, ax  = plt.subplots(1, 1, figsize = (9, 5))

sns.kdeplot(df_train[(df_train["Survived"] == 0) & (df_train["Pclass"] == 1)]["Age"], ax=ax)

sns.kdeplot(df_train[(df_train["Survived"] == 1) & (df_train["Pclass"] == 1)]["Age"], ax=ax)

plt.legend(["Survived == 0", "Survived == 1"])

plt.title("1st Class")

plt.show()



# Age distribution of non-survival people with first class

# Age distribution of survival people with first class
fig, ax  = plt.subplots(1, 1, figsize = (9, 5))

sns.kdeplot(df_train[(df_train["Survived"] == 0) & (df_train["Pclass"] == 2)]["Age"], ax=ax)

sns.kdeplot(df_train[(df_train["Survived"] == 1) & (df_train["Pclass"] == 2)]["Age"], ax=ax)

plt.legend(["Survived == 0", "Survived == 1"])

plt.title("2nd Class")

plt.show()



# Age distribution of non-survival people with second class

# Age distribution of survival people with second class
fig, ax  = plt.subplots(1, 1, figsize = (9, 5))

sns.kdeplot(df_train[(df_train["Survived"] == 0) & (df_train["Pclass"] == 3)]["Age"], ax=ax)

sns.kdeplot(df_train[(df_train["Survived"] == 1) & (df_train["Pclass"] == 3)]["Age"], ax=ax)

plt.legend(["Survived == 0", "Survived == 1"])

plt.title("3rd Class")

plt.show()



# Age distribution of non-survival people with third class

# Age distribution of survival people with third class
chage_age_range_survival_ratio = []

i = 80

for i in range(1,81):

    chage_age_range_survival_ratio.append(df_train[df_train["Age"] < i]["Survived"].sum()/len(df_train[df_train["Age"] < i]["Survived"])) # i보다 작은 나이의 사람들이 생존률



plt.figure(figsize = (7, 7))

plt.plot(chage_age_range_survival_ratio)

plt.title("Survival rate change depending on range of Age", y = 1.02)

plt.ylabel("Survival rate")

plt.xlabel("Range of Age(0-x)")

plt.show()

    

# The younger the age, the higher the probability of survival, The older the age, the less the probability of survival.
f, ax = plt.subplots(1, 2, figsize=(18, 8))

sns.violinplot("Pclass","Age", hue = "Survived", data = df_train, scale = "count", split = True, ax=ax[0])

ax[0].set_title("Pclass and Age vs Survived")

ax[0].set_yticks(range(0, 110, 10))



sns.violinplot("Sex", "Age", hue = "Survived", data = df_train, scale = "count", split = True, ax=ax[1])

ax[1].set_title("Sex and Age vs Survived")

ax[1].set_yticks(range(0, 110, 10))



plt.show()



# Based on age, the survival rate according to Pclass and the survival rate according to gender can be seen at a glance.

# As a result, the better the Pclass, the higher the survival rate and the higher the survival rate of women than men.
f, ax = plt.subplots(1, 1, figsize=(7,7))

df_train[["Embarked","Survived"]].groupby(["Embarked"], as_index=True).mean().sort_values(by="Survived",

                                                                                         ascending = False).plot.bar(ax=ax)
f, ax = plt.subplots(2, 2, figsize=(20,15))

sns.countplot("Embarked", data = df_train, ax=ax[0,0])

ax[0,0].set_title("(1) No. of Passengers Boared")



sns.countplot("Embarked", hue = "Sex", data = df_train, ax=ax[0,1])

ax[0,1].set_title("(2) Male-Female split for Embarked")



sns.countplot("Embarked", hue = "Survived", data = df_train, ax=ax[1,0])

ax[1,0].set_title("(3) Embarked vs Survived")



sns.countplot("Embarked", hue = "Pclass", data = df_train, ax=ax[1,1])

ax[1,1].set_title("(4) Embarked vs Pclass")



plt.subplots_adjust(wspace = 0.4, hspace = 0.5) 

plt.show()



# As a result, the survival rate is high because the people on board C have a lot of first class and many women.
df_train["FamilySize"] = df_train["SibSp"] + df_train["Parch"]+1

df_test["FamilySize"] = df_test["SibSp"] + df_test["Parch"]+1



# Create a new feature, "FamilySize".
df_train["FamilySize"].head(5)
df_test["FamilySize"].head()
print("Maximum size of Family: ", df_train["FamilySize"].max())

print("Minimum size of Family: ", df_train["FamilySize"].min())
f, ax = plt.subplots(1, 3, figsize = (40, 10))

sns.countplot("FamilySize", data = df_train, ax = ax[0])

ax[0].set_title("(1) No. of Passenger Boarded", y = 1.02)



sns.countplot("FamilySize", hue = "Survived", data = df_train, ax = ax[1])

ax[1].set_title("(2) Survived countplot depending of FamilySize")



df_train[["FamilySize", "Survived"]].groupby(["FamilySize"], as_index = True).mean().sort_values(by = "Survived",

                                                                                                      ascending = False).plot.bar(ax = ax[2])

ax[2].set_title("(3) Survived rate depending on FamilySize", y = 1.02)



plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

plt.show()



# The first plot is the number of passengers according to the number of family members (1 to 11), the second plot is the number of survivors according to the number of family members, and the third plot is the survival rate according to the number of family members.

# The family with four families has the highest survival rate.
f, ax = plt.subplots(1, 1, figsize = (8,8))

g = sns.distplot(df_train["Fare"], color = "b", label="Skewness: {:2f}".format(df_train["Fare"].skew()), ax=ax)

g = g.legend(loc = "best")



# The skewness tells us how asymmetric the distribution is.
# Log to get rid of the skewness.



df_train["Fare"] = df_train["Fare"].map(lambda i:np.log(i) if i>0 else 0)
df_train["Fare"].head(5)
f, ax = plt.subplots(1, 1, figsize = (8,8))

g = sns.distplot(df_train["Fare"], color = "b", label="Skewness: {:2f}".format(df_train["Fare"].skew()), ax=ax)

g = g.legend(loc = "best")



# normal approximation
# First, fill NULL data.



df_train["Age"].isnull().sum()
# It groups by using the title (mr, ms, mrs) etc. entering into name.

# It extracts by using the normal expression.



df_train["Initial"] = df_train["Name"].str.extract("([A-Za-z]+)\.") # Initial로 호칭을 저장해준다.

df_test["Initial"] = df_test["Name"].str.extract("([A-Za-z]+)\.")
df_train.head()
df_test.head()
pd.crosstab(df_train["Initial"], df_train["Sex"]).T.style.background_gradient(cmap = "Pastel2")



# Identify the title by gender and use crossstab to create a frequency table.
# The several titles is substituted simply. 



df_train["Initial"].replace(["Mlle","Mme", "Ms", "Dr","Major","Lady","Countess", "Jonkheer", "Col", "Rev", "Capt", "Sir", "Don", "Dona"],

                           ["Miss", "Miss","Miss", "Mr", "Mr", "Mrs", "Mrs", "Other", "Other", "Other", "Mr", "Mr", "Mr", "Mr"], inplace = True)



df_test["Initial"].replace(["Mlle","Mme", "Ms", "Dr","Major","Lady","Countess", "Jonkheer", "Col", "Rev", "Capt", "Sir", "Don", "Dona"],

                           ["Miss", "Miss","Miss", "Mr", "Mr", "Mrs", "Mrs", "Other", "Other", "Other", "Mr", "Mr", "Mr", "Mr"], inplace = True)
df_train.groupby("Initial").mean()
df_train.groupby("Initial")["Survived"].mean().plot.bar()



# Women such as Miss and Mrs have a high survival rate, and Master has a low average age, but a high survival rate.

# In addition, Mr has a low survival rate
# train, test Two data sets are combined and statistics is confirmed.

# (Use concat: A function that builds a dataset on a dataset)

df_all = pd.concat([df_train,df_test])

df_all.shape
df_all.groupby("Initial").mean()



# NULL values are filled by using the average of Age.
df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Mr"), "Age"] = 33

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Master"),"Age"] = 5

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Miss"), "Age"] = 22

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Mrs"), "Age"] = 37

df_train.loc[(df_train["Age"].isnull()) & (df_train["Initial"] == "Other"), "Age"] = 45



df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Mr"), "Age"] = 33

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Master"),"Age"] = 5

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Miss"), "Age"] = 22

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Mrs"), "Age"] = 37

df_test.loc[(df_test["Age"].isnull()) & (df_test["Initial"] == "Other"), "Age"] = 45



# The Age returns NULL, Initial is Mr, draws only the Age column and fills it all with 33 (average of the Age seen above).

# Fill all the NULL data in the same way.
df_train["Age"].isnull().sum()
df_test["Age"].isnull().sum()
df_train["Embarked"].isnull().sum()
df_train.shape



# Of the 891 rows, only two have no missing value, so replace it with the most frequent value.
df_train["Embarked"].fillna("S", inplace = True)



# fillna fills the missing value value with the designated value.

# In the EDA process, S is the most common, so it replaces.
df_test["Embarked"].isnull().sum()
df_train["Age_Categ"] = 0

df_test["Age_Categ"] = 0



# create a new feature
def category_age(x):

    if x < 10:

        return 0

    elif x < 20:

        return 1

    elif x < 30:

        return 2

    elif x < 40:

        return 3 

    elif x < 50:

        return 4

    elif x < 60: 

        return 5

    elif x < 70: 

        return 6

    else:

        return 7

    

# Make a function for apply use.  
df_train["Age_Categ"] = df_train["Age"].apply(category_age)

df_test["Age_Categ"] = df_test["Age"].apply(category_age)



# By using the apply function, the information of the age column categorized is added to train and test set.
df_train["Age_Categ"].head(10)
df_test["Age_Categ"].head(10)
df_train.head()
df_test.head()
# Since categorizing Age, the unnecessary Age column is deleted.



df_train.drop(["Age"], axis = 1 ,inplace = True)

df_test.drop(["Age"], axis = 1, inplace = True)
df_train.head()
df_train["Initial"].unique()
df_train["Initial"] = df_train["Initial"].map({"Master" : 0, "Miss" : 1, "Mr" : 2, "Mrs" : 3, "Other" : 4})

df_test["Initial"] = df_test["Initial"].map({"Master" : 0, "Miss" : 1, "Mr" : 2, "Mrs" : 3, "Other" : 4})
df_train["Embarked"].value_counts()
df_train["Embarked"] = df_train["Embarked"].map({"C" : 0, "Q" : 1, "S" : 2})

df_test["Embarked"] = df_test["Embarked"].map({"C" : 0, "Q" : 1, "S" : 2})
df_train.head()
df_test.head()
df_train["Sex"].unique()
df_train["Sex"] = df_train["Sex"].map({"female" : 0, "male" : 1})

df_test["Sex"] = df_test["Sex"].map({"female" : 0, "male" : 1})
heatmap_data = df_train[["Survived", "Pclass", "Sex", "Fare", "Embarked", "FamilySize", "Initial", "Age_Categ"]]
colormap = plt.cm.PuBu

plt.figure(figsize=(10, 8))

plt.title("Person Correlation of Features", y = 1.05, size = 15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths = 0.1, vmax = 1.0,

           square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 16})





# Correlation coefficient analysis shows whether there are overlapping features and which features show correlation.
# o improve the performance of the model, it is a task to change the form to use the information of the data that was categorized.

# One Hot Encoding is to make these vector (Dummy)



df_train = pd.get_dummies(df_train, columns = ["Initial"], prefix = "Initial")

df_test = pd.get_dummies(df_test, columns = ["Initial"], prefix = "Initial")
df_train = pd.get_dummies(df_train, columns = ["Embarked"], prefix = "Embarked")

df_test = pd.get_dummies(df_test, columns = ["Embarked"], prefix = "Embarked")
# Features which are not used are deleted.

df_train.head(1)
df_train.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"], axis = 1, inplace = True)

df_test.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"], axis = 1, inplace = True)
df_train.head() 
df_test.head()
kfold = StratifiedKFold(n_splits=10)
df_train["Survived"] = df_train["Survived"].astype(int)



Y_train = df_train["Survived"]



X_train = df_train.drop(labels = ["Survived"],axis = 1)
# modeling test with various algorithms

random_state = 2

classifiers = []

classifiers.append(SVC(random_state = random_state))

classifiers.append(DecisionTreeClassifier(random_state = random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state = random_state), random_state = random_state, learning_rate = 0.1))

classifiers.append(RandomForestClassifier(random_state = random_state))

classifiers.append(ExtraTreesClassifier(random_state = random_state))

classifiers.append(GradientBoostingClassifier(random_state = random_state))

classifiers.append(MLPClassifier(random_state = random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results = []

for classifier in classifiers:

    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs = 4))

    

cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())

    

cv_res = pd.DataFrame({"CrossValMeans": cv_means, "CrossValerrors": cv_std,

                       "Algorithm": ["SVC", "DecisionTree", "AdaBoost", "RandomForest",

                                     "ExtraTrees", "GradientBoosting", "MultipleLayerPerceptron", "KNeighboors",

                                    "LogisticRegression", "LinearDiscriminantAnalysis"]})



g = sns.barplot("CrossValMeans", "Algorithm", data = cv_res, palette = "Set3",

               orient = "h", **{'xerr': cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
# Grid Search Optimization for Five Models

    

# Adaboost

DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state = 7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

                 "base_estimator__splitter": ["best", "random"],

                 "algorithm": ["SAMME", "SAMME.R"],

                 "n_estimators": [1,2],

                 "learning_rate": [0.0001,0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}



gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv = kfold, scoring = "accuracy",

                       n_jobs = 4, verbose = 1)

gsadaDTC.fit(X_train, Y_train)

ada_best = gsadaDTC.best_estimator_



gsadaDTC.best_score_
# ExtraTrees



ExtC = ExtraTreesClassifier()



# Search Grid for Optimal Parameters



ex_param_grid = {"max_depth": [None],

                "max_features": [1,2,10],

                "min_samples_split": [2, 3, 10],

                "min_samples_leaf": [1,3,10],

                "bootstrap": [False],

                "n_estimators": [100, 300],

                "criterion": ["gini"]}



gsExtC = GridSearchCV(ExtC, param_grid = ex_param_grid, cv = kfold, scoring = "accuracy",

                     n_jobs = 4, verbose = 1)



gsExtC.fit(X_train, Y_train)

ExtC_best = gsExtC.best_estimator_



gsExtC.best_score_
# RandomForestClassifier

RFC = RandomForestClassifier()



# Search Grid for Optimal Parameters

rf_param_grid = {"max_depth": [None],

                "max_features": [1,3,10],

                "min_samples_split": [2,3,10],

                "min_samples_leaf": [1,2,10],

                "bootstrap": [False],

                "n_estimators": [100,300],

                "criterion": ["gini"]}



gsRFC = GridSearchCV(RFC, param_grid = rf_param_grid, cv=kfold, scoring = "accuracy", n_jobs = 4,

                    verbose = 1)



gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_



gsRFC.best_score_

# Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {"loss": ["deviance"],

                "n_estimators": [100,200,300],

                "learning_rate": [0.1, 0.05, 0.01],

                "max_depth": [4, 8],

                "min_samples_leaf": [100,150],

                "max_features": [0.3, 0.1]}



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv = kfold, scoring = "accuracy",

                    n_jobs = 4, verbose = 1)



gsGBC.fit(X_train, Y_train)

GBC_best = gsGBC.best_estimator_



gsGBC.best_score_
# SVC classifier

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
votingC = VotingClassifier(estimators = [("rfc", RFC_best), ("extc", ExtC_best),

                                        ("svc", SVMC_best), ("adac", ada_best),

                                        ("gbc", GBC_best)], voting = "soft", n_jobs = 4)



votingC = votingC.fit(X_train, Y_train)
submission = pd.read_csv("../input/gender_submission.csv")
submission.head()
df_test["Fare"].fillna("35.6271", inplace = True)

X_test = df_test.values
prediction = votingC.predict(X_test)
submission["Survived"] = prediction
submission.to_csv("./The_first_submission.csv", index = False)