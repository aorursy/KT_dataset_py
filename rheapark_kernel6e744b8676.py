# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats



from sklearn.linear_model import (LogisticRegression,SGDClassifier)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier)

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score, learning_curve)





from collections import Counter

%matplotlib inline

plt.style.use('ggplot')

mpl.rcParams['axes.unicode_minus'] = False
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

IDtest = test['PassengerId']
# Outlier detection #### IMPORTANT



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

train.loc[Outliers_to_drop]
train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)
train_len= len(train)

dataset= pd.concat(objs=[train, test], axis=0).reset_index(drop=True)
#fill empty and NaNs values with NaN

dataset = dataset.fillna(np.nan)

dataset.isnull().sum()
train.info()

train.isnull().sum()
train.head()
#correlation matrix between numerical values

g = sns.heatmap(train[["Survived","SibSp","Parch", "Age", "Fare"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
#Explore SibSp Feature vs Survived



g = sns.factorplot(x="SibSp", y="Survived", data=train, kind="bar", size=6,

                  palette="muted")

g.despine(left=True)

g = g.set_ylabels("survival probability")
# Explore Parch feature vs Survived

g = sns.factorplot(x="Parch", y="Survived", data=train,

                  kind="bar", size=6, palette="muted")

g.despine(left=True)

g= g.set_ylabels("survival probability")

#Explore Age vs Survived

g = sns.FacetGrid(train, col = "Survived")

g = g.map(sns.distplot, "Age")
# explore age distribution and split them

g = sns.kdeplot(train["Age"][(train["Survived"]==0) &

               (train["Age"].notnull())], color="Red", shade=True)

g = sns.kdeplot(train["Age"][(train["Survived"]==1) &

                            (train["Age"].notnull())],

               ax=g, color="Blue", shade=True)

g.set_xlabel("Age")

g.set_ylabel("Frequency")

g = g.legend(["Not Survived","Survived"])
dataset["Fare"].isnull().sum()
#Fill Fare missing values with the median value

dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
#Explore Fare distribution

g = sns.distplot(dataset["Fare"], color="m", label="Skewness : %.2f"%(dataset["Fare"].skew()))

g = g.legend(loc="best")
# Apply log to Fare to reduce skewness distribution

dataset["Fare"] = dataset["Fare"].map(lambda i

                                      :np.log(i) if i>0 else 0)
g = sns.distplot(dataset["Fare"], color="b", label="Skewness : %.2f"%(dataset["Fare"].skew()))

g = g.legend(loc="best")
g = sns.barplot(x="Sex", y="Survived", data=train)

g = g.set_ylabel("Survival Probability")
train[["Sex","Survived"]].groupby("Sex").mean()
#Explore PClass Vs Survived

g = sns.factorplot(x="Pclass", y="Survived", data=train,

                  kind="bar", size=6, palette="muted")

g.despine(left=True)

g= g.set_ylabels("survival probability")
#Explore Pclass vs Survived by Sex

g = sns.factorplot(x="Pclass", y="Survived", hue="Sex",

                  data=train, size=6, kind="bar", palette="muted")

g.despine(left=True)

g= g.set_ylabels("survival probability")
dataset["Embarked"].isnull().sum()
#Fill Embarked nan values of dataset with 'S' most frequent value

dataset["Embarked"] = dataset["Embarked"].fillna("S")
#Explore Embarked vs Survived

g= sns.factorplot(x="Embarked", y="Survived", 

                 data=train, size=6, kind="bar",

                 palette="muted")

g.despine(left=True)

g= g.set_ylabels("survival probability")
#Explore Pclass vs Embarked

g = sns.factorplot("Pclass", col="Embarked", data=train, size=6, kind="count", palette="muted")

g.despine(left=True)

g= g.set_ylabels("Count")
# Age NAN=256 counts, leverage correlated features

# Explore Age vs Sex, Parch, Pclass and SibSp

g= sns.factorplot(y="Age", x="Sex", data=dataset, kind="box")

g= sns.factorplot(y="Age", x="Sex", hue="Pclass",

                 data=dataset, kind="box")

g= sns.factorplot(y="Age", x="Parch", data=dataset, kind="box")

g= sns.factorplot(y="Age", x="SibSp", data=dataset, kind="box")

#convert Sex into categorical value 0 for male and 1 for female

dataset["Sex"]= dataset["Sex"].map({"male":0, "female":1})
g = sns.heatmap(dataset[["Age","Sex","SibSp","Parch","Pclass"]].corr(), cmap="BrBG", annot=True)
#Filling missing value of Age  ###IMPORTANT

#Fill Age with the median age of similar rows according to

#Pclassm Parch and SibSp

index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)



for i in index_NaN_age:

    age_med = dataset["Age"].median()

    age_pred = dataset["Age"][((dataset["SibSp"]==dataset.iloc[i]["SibSp"])

                              & (dataset["Parch"]==dataset.iloc[i]["Parch"])

                              &(dataset["Pclass"]==dataset.iloc[i]["Pclass"]))].median()

    

    if not np.isnan(age_pred):

        dataset["Age"].iloc[i] = age_pred

    else:

        dataset["Age"].iloc[i] = age_med

    
### What if, 단순 median이 아닌, corr가 큰 PClass에 w 가중치 줘서, median 나오는 것??

g = sns.factorplot(x="Survived", y="Age", data=train, kind="box")

g = sns.factorplot(x="Survived", y="Age", data=train, kind="violin")
dataset['Name'].head()


#Get Title from Name



dataset_title=[i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"]=pd.Series(dataset_title)

dataset["Title"].head()
g = sns.countplot(x="Title", data=dataset)

g = plt.setp(g.get_xticklabels(), rotation=45)
# Convert to categorical values Title 

dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)

g = sns.countplot(dataset["Title"])

g =g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr","Rare"])
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar")

g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])

g = g.set_ylabels("survival probability")
#drop Name variable

dataset.drop(labels=["Name"], axis=1, inplace=True)
#Create Family size descriptor from SibSp and Parch

dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] +1
g = sns.factorplot(x="Fsize", y="Survived", data=dataset)

g = g.set_ylabels("Survival Probability")
#1-2 Convert FSize into categorical value 0 for alone 1 for family

#dataset["Alone"] = dataset["Fsize"].map(lambda s: 1 if s==1 else 0)

#dataset.drop(labels=["Fsize"], axis=1, inplace=True)
#1-1Create new feature of family size

dataset["Single"] = dataset["Fsize"].map(lambda s: 1 if s==1 else 0)

dataset["SmallF"] = dataset["Fsize"].map(lambda s: 1 if s==2 else 0)

dataset["MedF"] = dataset["Fsize"].map(lambda s: 1 if 3<=s <=4 else 0)

dataset["LargeF"] = dataset["Fsize"].map(lambda s: 1 if s>=5 else 0)
#1-1

g = sns.factorplot(x="Single",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="SmallF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="MedF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")

g = sns.factorplot(x="LargeF",y="Survived",data=dataset,kind="bar")

g = g.set_ylabels("Survival Probability")
#convert to indicator valuees Title an Embarked

dataset = pd.get_dummies(dataset, columns=["Title"])

dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
dataset.head()
g = sns.heatmap(dataset[["Survived","SibSp","Parch","Fsize","Single"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
dataset.drop(labels=["SibSp"], axis=1, inplace=True)

dataset.drop(labels=["Parch"], axis=1, inplace=True)
dataset["Cabin"].head()
dataset["Cabin"].describe()
dataset["Cabin"].isnull().sum()
dataset["Cabin"][dataset["Cabin"].notnull()].head()
#Replace the Cabin Number by the type of cabin'X' if not

dataset["Cabin"]= pd.Series([i[0] if not pd.isnull(i)

                            else 'X' for i in dataset["Cabin"]])
g = sns.countplot(dataset["Cabin"], order=["A","B","C","D","E","F","G","T","X"])
g = sns.factorplot(y="Survived", x="Cabin", data=dataset, kind="bar",

                  order=["A","B","C","D","E","F","G","T","X"])

g= g.set_ylabels("Survival Probability")
dataset = pd.get_dummies(dataset, columns=["Cabin"],prefix="Cabin")
dataset["Ticket"].head()
#Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X.

#1. Ticket Dummies

Ticket = []

for i in list(dataset.Ticket):

    if not i.isdigit() :

        Ticket.append(i.replace(".","").replace("/","").

                     strip().split(' ')[0])

    else:

        Ticket.append("X")
#dataset["Ticket"] = Ticket

dataset["Ticket"].head()

dataset.drop(labels=["Ticket"], axis=1, inplace=True)
#dataset= pd.get_dummies(dataset, columns=["Ticket"], prefix="T")
#Create categorical values for Pclass

dataset["Pclass"] = dataset["Pclass"].astype("category")

dataset= pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")
#Drop useless variables

dataset.drop(labels = ["PassengerId"], axis=1, inplace=True)
dataset.head()
dataset.isnull().sum()
train = dataset[:train_len]

test = dataset[train_len:]

test.drop(labels=["Survived"],axis = 1,inplace=True)



train["Survived"] = train["Survived"].astype(int)



Y_train = train["Survived"]



X_train = train.drop(labels = ["Survived"],axis = 1)
from sklearn.linear_model import (LogisticRegression,SGDClassifier)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, VotingClassifier)

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score, learning_curve)

from collections import Counter
from sklearn.model_selection import KFold

Model = []

RMSE = []

ACC = []

cv = KFold(10, random_state = 42)



def input_scores_defaultparam(name, model, X, Y):

    Model.append(name)

    RMSE.append(np.sqrt((-1) * cross_val_score(model, X, Y, cv=cv, 

                                               scoring='neg_mean_squared_error')))

    ACC.append(cross_val_score(model, X, Y, cv=cv, scoring='accuracy'))
Y_train_raveled = np.ravel(Y_train)
names = ["Logistic","K-NN","SVC","RandomForest","SGD","DecisionTree","GradientBoosting","GaussianNB"]

models = [LogisticRegression(), KNeighborsClassifier(3), SVC(), RandomForestClassifier(),

          SGDClassifier(),DecisionTreeClassifier(), GradientBoostingClassifier(),GaussianNB()]



#Running all algorithms

for name, model in zip(names, models): 

    input_scores_defaultparam(name, model,X_train, Y_train_raveled)# x_train_scaled, y_train)
evaluation_def = pd.DataFrame({'Model': Model,

                           'RMSE': RMSE,

                           'ACC': ACC})

print("FOLLOWING ARE THE TRAINING SCORES: ")

evaluation_def 
#RMSE mean_계산 

RMSE_mean=[]

for i in range(len(RMSE)):

    RMSE_mean.append(np.mean(RMSE[i]))



#R_squared mean_계산 

ACC_mean=[]

for i in range(len(ACC )):

    ACC_mean.append(np.mean(ACC[i]))    
evaluation_mean =pd.DataFrame({'Model': Model,

                           'RMSE': RMSE_mean,

                           'ACC': ACC_mean})

print("FOLLOWING ARE THE TRAINING SCORES: ")

evaluation_mean
a=[evaluation_def.loc[i,'RMSE'] for i in range(0,8)]

aa=[evaluation_def.loc[i,'ACC'] for i in range(0,8)]
b = pd.DataFrame(a, columns=['Fold_'+str(i) for i in range(1,11)]) #RMSE

bb = pd.DataFrame(aa, columns=['Fold_'+str(i) for i in range(1,11)]) #'R-Squared'
bb['Model']=["1.LogisticRegression","2.K-NNClassifier","3.SVC","4.RandomForestClassifier",

            "5.SGDClassifier","6.DecisionTreeClassifier","7.GradientBoostingClassifier","8.GaussianNB"]
index_best = evaluation_def['Model'].ravel()
mpl.rcdefaults()

for i in range(1,11):

    plt.plot(b['Fold_'+str(i)],index_best)

plt.xlabel('RMSE')

plt.grid(linestyle=":")

plt.title('RMSE using Default Parameter')

plt.show()
mpl.rcdefaults()

for i in range(1,11):

    plt.plot(bb['Fold_'+str(i)],index_best)

plt.xlabel('ACC')

plt.grid(linestyle=":")

plt.title('ACC using Default Parameter')

plt.show()
KNN = KNeighborsClassifier(3)

kfold=10

kn_param_grid = {'leaf_size':[10, 100, 10],

             'n_neighbors':range(1, 11, 1), 'p':[1,2]}



gsKNN= GridSearchCV(KNN, param_grid = kn_param_grid, scoring='accuracy', 

                   cv=kfold, n_jobs=4).fit(X_train, Y_train_raveled)

KNN_best = gsKNN.best_estimator_



# Best score

gsKNN.best_score_
RFC = RandomForestClassifier()

kfold=10



## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[100,300],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(X_train,Y_train_raveled)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
#Gradient boosting tunning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200,300],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train,Y_train_raveled)



GBC_best = gsGBC.best_estimator_



# Best score

gsGBC.best_score_
### SVC classifier

SVMC = SVC(probability=True)

svc_param_grid = {'kernel': ['rbf'], 

                  'gamma': [ 0.001, 0.01, 0.1, 1],

                  'C': [1, 10, 50, 100,200,300, 1000]}



gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsSVMC.fit(X_train,Y_train_raveled)



SVMC_best = gsSVMC.best_estimator_



# Best score

gsSVMC.best_score_
test_Survived_KNN = pd.Series(KNN_best.predict(test), name="KNN")

test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")

test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")

test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")
# Concatenate all classifier results

ensemble_results = pd.concat([test_Survived_KNN,test_Survived_RFC,test_Survived_GBC, test_Survived_SVMC],axis=1)





g= sns.heatmap(ensemble_results.corr(),annot=True)
votingC = VotingClassifier(estimators=[('knn',KNN_best),('rfc', RFC_best), 

('svc', SVMC_best),('gbc',GBC_best)], voting='soft', n_jobs=4)



votingC = votingC.fit(X_train, Y_train)
test_Survived = pd.Series(votingC.predict(test), name="Survived")



results = pd.concat([IDtest,test_Survived],axis=1)



results.to_csv("ensemble_rp_python_voting.csv",index=False)