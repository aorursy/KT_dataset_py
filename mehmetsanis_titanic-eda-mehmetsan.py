# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import matplotlib.pyplot as plt



from collections import Counter



import seaborn as sns



import warnings

warnings.filterwarnings("ignore")


train = pd.read_csv('../input/titanic/train.csv')

test  = pd.read_csv('../input/titanic/test.csv')



trainSize = len( train )

test_PassengerId = test["PassengerId"]



data = pd.concat( [train , test], axis = 0 ).reset_index(drop = True)



train.columns
train.head()
train.describe()
train.describe(include=['O'])
train.info()
def barPlot( variable ):

    

    var = train[variable]

    varCounts = var.value_counts()

    

    # visualize

    

    plt.figure( figsize = (9,3))

    plt.bar( varCounts.index, varCounts )

    plt.xticks( varCounts.index, varCounts.index.values)

    plt.ylabel(' Frequency ')

    

    plt.title( variable )

    plt.show()

    

    print( '{}: \n {}'.format(variable, varCounts))
data.drop( labels = ['Cabin'], axis=1, inplace=True)
barPlot( 'Sex' )
data.drop(['PassengerId'], axis = 1, inplace=True)
# Sex vs Survived

train[ ['Sex','Survived'] ].groupby(['Sex'], as_index = False).mean().sort_values( by= 'Survived', ascending = False)
# Pclass vs Survived

train[ ['Pclass','Survived'] ].groupby(['Pclass'], as_index = False).mean().sort_values( by= 'Survived', ascending = False)
# Embarked vs Survived

train[ ['Embarked','Survived'] ].groupby(['Embarked'], as_index = False).mean().sort_values( by= 'Survived', ascending = False)
# SibSp vs Survived

train[ ['SibSp','Survived'] ].groupby(['SibSp']).mean().sort_values( by= 'Survived', ascending = False)
# Parch vs Survived

train[ ['Parch','Survived'] ].groupby(['Parch']).mean().sort_values( by= 'Survived', ascending = False)
def detectOutlier(df, features):

    

    outlier_indeces = []

    

    for c in features:

        

        # 1st quartile

        Q1 = np.percentile( df[c], 25)

        

        #3rd quartile

        Q3 = np.percentile( df[c], 75)

        

        # INTERQUARTILE RANGE

        IQR = Q3 - Q1

        

        #OUTLIER STEP

        OUTLIER_STEP = IQR * 1.5

        

        #DETECT OUTLIER INDECES

        outlier_list_column = df[ (df[c] < Q1 - OUTLIER_STEP)  | (df[c] > Q3 + OUTLIER_STEP)].index

        

        #STORE THEM FOR TEACH OF THE FEATURES

        outlier_indeces.extend( outlier_list_column )

        

    outlier_indeces = Counter( outlier_indeces )

    

    multiple_outliers = list(i for i, v in outlier_indeces.items() if v > 2)

    

    return multiple_outliers
train.loc[ detectOutlier(train, [ 'Age', 'SibSp', 'Parch', 'Fare']) ]
print(train.shape)

train = train.drop(detectOutlier(train,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)

print(train.shape)
data.columns[ data.isnull().any() ]
data.isnull().sum()
data[ data['Embarked'].isnull()]
data.boxplot( column = 'Fare', by = 'Embarked')

plt.show()
data['Embarked'] = data['Embarked'].fillna('C')

data[ data['Embarked'].isnull() ]
data[ data['Fare'].isnull() ]
data['Fare'] = data['Fare'].fillna( np.mean(data[ data['Pclass'] == 3 ].Fare) )

data[ data['Fare'].isnull() ]
features = ['SibSp', 'Parch', 'Age', 'Fare', 'Survived']



sns.heatmap( data[features].corr(), annot = True, fmt = '0.2f' )



plt.show()
g = sns.catplot( x = 'SibSp', y = 'Survived', data = data, kind = 'bar', height = 7)

g.set_ylabels('Survival Probability')

plt.show()
g = sns.catplot( x = 'Parch', y = 'Survived', data = data, kind = 'bar', height = 5)

g.set_ylabels('Survival Probability')

plt.show()
g = sns.catplot( x = 'Pclass', y = 'Survived', data = data, kind = 'bar', height = 5)

g.set_ylabels('Survival Probability')

plt.show()
g = sns.FacetGrid( data, col = 'Survived')



g.map(sns.distplot, 'Age' , bins = 20)



plt.show()
g = sns.FacetGrid( data, col = 'Survived', row = 'Pclass', height = 3)



g.map( plt.hist , 'Age', bins = 25)



plt.show()
g = sns.FacetGrid(data, row = "Embarked", height = 2)

g.map(sns.pointplot, "Pclass","Survived", "Sex")

g.add_legend()

plt.show()
g = sns.FacetGrid(data , row = "Embarked", col = "Survived", size = 2)

g.map(sns.barplot, "Sex", "Fare")

g.add_legend()

plt.show()
data[ data['Age'].isnull() ]
g = sns.catplot( x = 'Sex', y = 'Age', data = data, kind = 'box')

plt.show()
g = sns.catplot( x = 'Sex', y = 'Age', hue = 'Pclass', data = data, kind = 'box')

plt.show()
g = sns.catplot( x = 'SibSp', y = 'Age', data = data, kind = 'box', height = 4)

g = sns.catplot( x = 'Parch', y = 'Age', data = data, kind = 'box', height = 4)

plt.show()
data['Sex'] = [0 if each == 'male' else 1 for each in data['Sex']]
sns.heatmap( data[["Age","Sex","SibSp","Parch","Pclass"]].corr(), annot = True )

plt.show()
index_nan_age = list(data["Age"][data["Age"].isnull()].index)
index_nan_age = list(data["Age"][data["Age"].isnull()].index)



age_med = data["Age"].median()



for i in index_nan_age:

    age_pred = data["Age"][((data["SibSp"] == data.iloc[i]["SibSp"]) &(data["Parch"] == data.iloc[i]["Parch"])& (data["Pclass"] == data.iloc[i]["Pclass"]))].median()

    

    if not np.isnan(age_pred):

        data["Age"].iloc[i] = age_pred

    else:

        data["Age"].iloc[i] = age_med

        

data[data["Age"].isnull()]
data['Name'].head(10)
names = data['Name']



titles = [each.split('.')[0].split(' ')[-1].strip()  for each in names]



data['Title'] = titles
sns.countplot(x="Title", data = data)

plt.xticks(rotation = 60)

plt.show()
data['Title'] = [0 if each == 'Mr' else 1 if each == 'Mrs' or each == 'Miss' or each == 'Ms' or each == 'Mlle' else 2 if each == 'Master' else 3 for each in data['Title']]



data['Title'].head(10)
sns.countplot(x="Title", data = data)

plt.xticks(rotation = 60)

plt.show()
g = sns.factorplot(x = "Title", y = "Survived", data = data, kind = "bar")

g.set_xticklabels(["Master","Mrs","Mr","Other"])

g.set_ylabels("Survival Probability")

plt.show()
data.drop(labels = ["Name"], axis = 1, inplace = True)
data = pd.get_dummies(data,columns=["Title"])

data.head()
data.head()
data["Fsize"] = data["SibSp"] + data["Parch"] + 1
data.head()
g = sns.factorplot(x = "Fsize", y = "Survived", data = data, kind = "bar")

g.set_ylabels("Survival")

plt.show()
data["family_size"] = [1 if i < 5 else 0 for i in data["Fsize"]]
data.head(10)
sns.countplot(x = "family_size", data = data)

plt.show()
g = sns.factorplot(x = "family_size", y = "Survived", data = data, kind = "bar")

g.set_ylabels("Survival")

plt.show()
data = pd.get_dummies(data, columns= ["family_size"])

data.head()
data.head()
g = sns.countplot(x = "Embarked", data = data)

plt.show()
g = sns.factorplot(x = "Embarked", y = "Survived", data = data, kind = "bar")

g.set_ylabels("Survival")

plt.show()
data = pd.get_dummies(data, columns=["Embarked"])

data.head()
tickets = []

for i in list(data.Ticket):

    if not i.isdigit():

        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])

    else:

        tickets.append("x")

data["Ticket"] = tickets
data["Ticket"].head()
data = pd.get_dummies(data, columns= ["Ticket"], prefix = "T")

data.head()
data.head()
g = sns.countplot( x = 'Pclass', data=data)

plt.show()
g = sns.catplot(x = "Pclass", y = "Survived", data = data, kind = "bar")

g.set_ylabels("Survival")

plt.show()
data = pd.get_dummies(data = data , columns = ['Pclass'])
data.head()
data.head()
data = pd.get_dummies(data, columns = ['Sex'])

data.head()
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
train = data[ : trainSize]

test = data[trainSize : ]



test.drop(labels = ["Survived"],axis = 1, inplace = True)
test.head()
train.head()
X_train = train.drop(labels = "Survived", axis = 1)

y_train = train["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

acc_log_train = round(logreg.score(X_train, y_train)*100,2) 

acc_log_test = round(logreg.score(X_test,y_test)*100,2)

print("Training Accuracy: % {}".format(acc_log_train))

print("Testing Accuracy: % {}".format(acc_log_test))
random_state = 42

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier()]



dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}



svc_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
votingC = VotingClassifier(estimators = [("dt",best_estimators[0]),

                                        ("rfc",best_estimators[2]),

                                        ("lr",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(X_train, y_train)

print(accuracy_score(votingC.predict(X_test),y_test))
test_survived = pd.Series(votingC.predict(test), name = "Survived").astype(int)

results = pd.concat([test_PassengerId, test_survived],axis = 1)

results.to_csv("titanic.csv", index = False)