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

import seaborn as sns

%matplotlib inline

from collections import Counter
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

IDtest = test["PassengerId"]
train.head()
print("trainig data shape",train.shape)

print("testing data shape",test.shape)
numerical_features=[features for features in train.columns if train[features].dtype!='O' ]
for feature in numerical_features:

    print(feature)
# detect outliers from Age, SibSp , Parch and Fare





def detect_outliers(train,n,features):



    outlier_indices = []

    

    

    for col in features:

        

        # gonna use IQR method for detecting and deleting outliers

        

        # 1st quartile (25%)

        Q1 = np.percentile(train[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(train[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = train[(train[col] < Q1 - outlier_step) | (train[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])

train.loc[Outliers_to_drop] # Show the outliers rows
# Drop outliers

train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
# before that remember the length of train data

print("shape of train data:",train.shape)
dataset =  pd.concat([train, test], axis=0).reset_index(drop=True)
dataset.head()

print(dataset.shape)
# Fill empty and NaNs values with NaN

dataset = dataset.fillna(np.nan)



# Check for Null values

dataset.isnull().sum()
# Infos

train.info()

train.isnull().sum()
test.isnull().sum()
#Age

g = sns.FacetGrid(train,col="Survived")

g = g.map(sns.distplot,"Age")
# Explore Age distibution 

g = sns.kdeplot(train["Age"][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)

g = sns.kdeplot(train["Age"][(train["Survived"] == 1) & (train["Age"].notnull())], ax =g, color="Blue", shade= True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
#  Parch feature vs Survived

g  = sns.catplot(x="Parch",y="Survived",data=train,kind="bar", size = 6 , palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
#SibSp vs survival



# Explore SibSp feature vs Survived

g = sns.catplot(x="SibSp",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
dataset["Fare"].isnull().sum()
#Fill Fare missing values with the median value

dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
# Explore Fare distribution 

g = sns.distplot(dataset["Fare"],color='R')
dataset['Fare']=dataset['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(dataset["Fare"], color="b")

g = sns.barplot(x="Sex",y="Survived",data=train)

g = g.set_ylabel("Survival Probability")

pd.DataFrame(train.groupby('Sex')['Survived'].mean())
# Explore Pclass vs Survived

g = sns.catplot(x="Pclass",y="Survived",data=train,kind="bar", size = 6 , 

palette = "muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Survived by Sex

g = sns.catplot(x="Pclass", y="Survived", hue="Sex", data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
print("No of null Vales in Embarked Feature are",dataset["Embarked"].isnull().sum())

print("the most repeating category in Embarked Feature is:",dataset["Embarked"].mode())
#Fill Embarked nan values of dataset set with 'S' most frequent value

dataset["Embarked"] = dataset["Embarked"].fillna("S")
# Explore Embarked vs Survived 

g = sns.catplot(x="Embarked", y="Survived",  data=train,

                   size=6, kind="bar", palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Pclass vs Embarked 

g = sns.factorplot("Pclass", col="Embarked",  data=train,

                   size=6, kind="count", palette="muted")

g.despine(left=True)

g = g.set_ylabels("Count")
# Explore Age vs Sex, Parch , Pclass and SibSP

g = sns.catplot(y="Age",x="Sex",data=dataset,kind="box")

g = sns.catplot(y="Age",x="Sex",hue="Pclass", data=dataset,kind="box")

g = sns.catplot(y="Age",x="Parch", data=dataset,kind="box")

g = sns.catplot(y="Age",x="SibSp", data=dataset,kind="box")
# convert Sex into categorical value 0 for male and 1 for female

dataset["Sex"] = dataset["Sex"].map({"male": 0, "female":1})
g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
# Filling missing value of Age 



## Fill Age with the median age of similar rows according to Pclass, Parch and SibSp

# Index of NaN age rows

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)



for i in index_NaN_age :

    age_med = dataset["Age"].median()

    age_pred = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"]))].median()

    if not np.isnan(age_pred) :

        dataset['Age'].iloc[i] = age_pred

    else :

        dataset['Age'].iloc[i] = age_med

g = sns.catplot(x="Survived", y = "Age",data = train, kind="box")

g = sns.catplot(x="Survived", y = "Age",data = train, kind="violin")
dataset["Name"].head()
# Get Title from Name

dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

dataset['Title']=dataset_title





dataset.head()
g = sns.countplot(x="Title",data=dataset)

g = plt.setp(g.get_xticklabels(), rotation=45) 
# Convert to categorical values Title 

dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)
g = sns.countplot(dataset["Title"])

g = g.set_xticklabels(["Master","Miss/Ms/Mme/Mlle/Mrs","Mr","Rare"])
g = sns.catplot(x="Title",y="Survived",data=dataset,kind="bar")

g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])

g = g.set_ylabels("survival probability")
# Drop Name variable

dataset.drop(labels = ["Name"], axis = 1, inplace = True)
# Create a family size descriptor from SibSp and Parch

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
g = sns.catplot(x="Fsize",y="Survived",data = dataset,kind='bar')

g = g.set_ylabels("Survival Probability")
# Create new feature of family size

dataset['Single'] = dataset['Fsize'].map(lambda s: 1 if s == 1 else 0)

dataset['SmallF'] = dataset['Fsize'].map(lambda s: 1 if  s == 2  else 0)

dataset['MedF'] = dataset['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

dataset['LargeF'] = dataset['Fsize'].map(lambda s: 1 if s >= 5 else 0)
g = sns.catplot(x="Single",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.catplot(x="SmallF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.catplot(x="MedF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.catplot(x="LargeF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")
# convert to indicator values Title and Embarked 

dataset = pd.get_dummies(dataset, columns = ["Title"])

dataset = pd.get_dummies(dataset, columns = ["Embarked"], prefix="Em")
dataset.head()
dataset.shape
dataset["Cabin"].isnull().sum()
dataset["Cabin"].head()
dataset["Cabin"].describe()
dataset["Cabin"][dataset["Cabin"].notnull()].head()
# Replace the Cabin number by the type of cabin 'X' if not

dataset["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dataset['Cabin'] ])
g = sns.countplot(dataset["Cabin"],order=['A','B','C','D','E','F','G','T','X'])
g = sns.catplot(y="Survived",x="Cabin",data=dataset,kind="bar",order=['A','B','C','D','E','F','G','T','X'])

g = g.set_ylabels("Survival Probability")
dataset = pd.get_dummies(dataset, columns = ["Cabin"],prefix="Cabin")
dataset["Ticket"].head()
## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 



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

dataset.head()
# Drop useless variables 

dataset.drop(labels = ["PassengerId"], axis = 1, inplace = True)
dataset.head()
## Separate train dataset and test dataset



train = dataset[:881]

test = dataset[881:]

test.drop(labels=["Survived"],axis = 1,inplace=True)
## Separate train features and label 



train["Survived"] = train["Survived"].astype(int)



Y_train = train["Survived"]



X_train = train.drop(labels = ["Survived"],axis = 1)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve



sns.set(style='white', context='notebook', palette='deep')
# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)
# Modeling step Test differents algorithms 

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

    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=-1))
cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())
cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,

                       "Algorithm":["SVC","DecisionTree","AdaBoost",

                       "RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron",

                        "KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})
cv_res
g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
cv_res.nlargest(5,"CrossValMeans")
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
test_Survived = pd.Series(gsGBC.predict(test), name="Survived")



results = pd.concat([IDtest,test_Survived],axis=1)



results.to_csv("sub_titanic.csv",index=False)