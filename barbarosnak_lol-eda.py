# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import warnings

warnings.filterwarnings("ignore")



from collections import Counter



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
data.head()
data.columns
id_list=["Wins","WardsPlaced","WardsDestroyed","FirstBlood","Kills","Deaths","Assists","EliteMonsters","Dragons","Heralds","TowersDestroyed","TotalGold","AvgLevel","TotalExperience","TotalMinionsKilled","TotalJungleMinionsKilled","GoldDiff","ExperienceDiff","CSPerMin","GoldPerMin"]

blue_data=[]

blue_data=pd.DataFrame(blue_data)

for i in id_list:

    blue_data[i]=data["blue"+i]
blue_data.head()
id_list=id_list[1:]
print(id_list)
red_data=[]

red_data=pd.DataFrame(red_data)

wins=[0 if each == 1 else 1 for each in blue_data["Wins"]]

red_data["Wins"]=wins

for i in id_list:

    red_data[i]=data["red"+i]
red_data.head()
corr_list=["Wins","WardsPlaced","WardsDestroyed","TotalGold","TotalExperience","TotalMinionsKilled","GoldDiff","ExperienceDiff","CSPerMin","GoldPerMin","Kills","AvgLevel","Deaths","FirstBlood","EliteMonsters","Dragons","Heralds","Assists","TowersDestroyed"]



f,ax=plt.subplots(figsize=(12,12))

sns.heatmap(blue_data[corr_list].corr(),annot=True,linewidths=.5,fmt=".1f",ax=ax)

plt.show()
g = sns.factorplot(x="Kills",y="Wins", data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
g = sns.factorplot(x="AvgLevel",y="Wins", data=blue_data, kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
g = sns.factorplot(x="Deaths",y="Wins", data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
g = sns.factorplot(x="FirstBlood",y="Wins", data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
g = sns.factorplot(x="EliteMonsters",y="Wins",data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
g = sns.factorplot(x="Dragons",y="Wins",data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
g = sns.factorplot(x="Heralds",y="Wins",data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
g = sns.factorplot(x="Assists",y="Wins",data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
g = sns.factorplot(x="TowersDestroyed",y="Wins",data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
blue_data["RoundedCSPerMin"]=np.round(blue_data["CSPerMin"])
g = sns.factorplot(x="RoundedCSPerMin",y="Wins",data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
print("max:",np.max(blue_data.WardsPlaced),"min:",np.min(blue_data.WardsPlaced))
i=0

while i<5:

    data1=blue_data[blue_data.WardsPlaced<=(i+1)*50]

    data2=data1[data1.WardsPlaced>i*50]

    g=sns.factorplot(x="WardsPlaced",y="Wins",data=data2,kind="bar",height=12)

    g.set_ylabels("Win Probability")

    plt.show()

    i=i+1



print("max:",np.max(blue_data.WardsDestroyed),"min:",np.min(blue_data.WardsDestroyed))
g=sns.factorplot(x="WardsDestroyed",y="Wins",data=blue_data,kind="bar",height=12)

g.set_ylabels("Win Probability")

plt.show()
print(np.mean(blue_data.TotalGold))

liste=[1 if each>=16503 else 0 for each in blue_data.TotalGold]

blue_data["totalgold"]=liste

g=sns.factorplot(x="totalgold",y="Wins",data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
print("mean:",np.mean(blue_data.TotalExperience),"max:",np.max(blue_data.TotalExperience),"min:",np.min(blue_data.TotalExperience))

liste=[1 if each>=20000 else 0 for each in blue_data.TotalExperience]

blue_data["totalexp"]=liste

g=sns.factorplot(x="totalexp",y="Wins",data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
print("mean:",np.mean(blue_data.GoldPerMin),"max:",np.max(blue_data.GoldPerMin),"min:",np.min(blue_data.GoldPerMin))
liste=[1 if each>=2000 else 0 for each in blue_data.GoldPerMin]

blue_data["goldpermin"]=liste

g=sns.factorplot(x="goldpermin",y="Wins",data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
print("mean:",np.mean(blue_data.ExperienceDiff),"max:",np.max(blue_data.ExperienceDiff),"min:",np.min(blue_data.ExperienceDiff))
liste=[1 if each>=0 else 0 for each in blue_data.ExperienceDiff]

blue_data["expdiff"]=liste

g=sns.factorplot(x="expdiff",y="Wins",data=blue_data,kind="bar",height=7)

g.set_ylabels("Win Probability")

plt.show()
def detect_outliers(df,features):

    outlier_indices=[]

    

    for c in features:

        #1st quartile

        Q1 = np.percentile(df[c],25)

        #3rd quartile

        Q3= np.percentile(df[c],75)

        # IQR

        IQR= Q3-Q1

        # Outlier step

        outlier_step=IQR*1.5

        #detect outliers and their indices

        outlier_list_col=df[(df[c]<Q1-outlier_step) |( df[c]>Q3+outlier_step)].index

        # Store indices

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices=Counter(outlier_indices)

    multiple_outliers=list(i for i, v in outlier_indices.items() if v>2)

    

    return multiple_outliers
blue_data.loc[detect_outliers(blue_data,["Kills","AvgLevel","Deaths","FirstBlood","EliteMonsters","Dragons","Heralds","Assists","TowersDestroyed","RoundedCSPerMin","totalgold","totalexp","goldpermin","expdiff"])]
blue_data=blue_data.drop(detect_outliers(blue_data,["Kills","AvgLevel","Deaths","FirstBlood","EliteMonsters","Dragons","Heralds","Assists","TowersDestroyed","RoundedCSPerMin","totalgold","totalexp","goldpermin","expdiff"]),axis=0).reset_index(drop = True)
blue_data.Kills=[2 if i>=12 else 1 if i>=6 and i<12 else 0 for i in blue_data.Kills]
blue_data.Kills.unique()
sns.countplot(x="Kills",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["Kills"])

blue_data.head()
blue_data.AvgLevel=[1 if i>=7.2 else 0 for i in blue_data.AvgLevel]
blue_data.AvgLevel.unique()
sns.countplot(x="AvgLevel",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["AvgLevel"])

blue_data.head()
blue_data.Deaths=[1 if i<5 else 0 for i in blue_data.Deaths]
blue_data.Deaths.unique()
sns.countplot(x="Deaths",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["Deaths"])

blue_data.head()
sns.countplot(x="FirstBlood",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["FirstBlood"])

blue_data.head()
sns.countplot(x="EliteMonsters",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["EliteMonsters"])

blue_data.head()
sns.countplot(x="Dragons",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["Dragons"])

blue_data.head()
sns.countplot(x="Heralds",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["Heralds"])

blue_data.head()
blue_data.Assists=[1 if i>=6 and i<15 else 2 if i>=15 else 0 for i in blue_data.Assists]
blue_data.Assists.unique()
sns.countplot(x="Assists",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["Assists"])

blue_data.head()
sns.countplot(x="TowersDestroyed",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["TowersDestroyed"])

blue_data.head()
blue_data.RoundedCSPerMin=[1 if i>=24 else 0 for i in blue_data.RoundedCSPerMin]
blue_data.RoundedCSPerMin.unique()
sns.countplot(x="RoundedCSPerMin",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["RoundedCSPerMin"])

blue_data.head()
sns.countplot(x="totalgold",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["totalgold"])

blue_data.head()
sns.countplot(x="totalexp",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["totalexp"])

blue_data.head()
sns.countplot(x="goldpermin",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["goldpermin"])

blue_data.head()
sns.countplot(x="expdiff",data=blue_data)

plt.xticks(rotation=60)

plt.show()
blue_data=pd.get_dummies(blue_data,columns=["expdiff"])

blue_data.head()
blue_data.columns
blue_data=blue_data.drop(["TotalGold","TotalExperience","TotalMinionsKilled","TotalJungleMinionsKilled","GoldDiff","ExperienceDiff","CSPerMin","GoldPerMin"],axis=1)

blue_data.head()
blue_data.columns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
length=6000

test=blue_data[length:]

test.drop(["Wins"],axis=1,inplace=True)
test.head()
train=blue_data[:length]

X_train=train.drop(["Wins"],axis=1)

Y_train=train.Wins



x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.3,random_state=42)

print("x_train:",len(x_train))

print("x_test:",len(x_test))

print("y_train:",len(y_train))

print("y_test:",len(y_test))

print("test:",len(test))
logreg=LogisticRegression()

logreg.fit(x_train,y_train)

acc_log_train=round(logreg.score(x_train,y_train)*100,2)

acc_log_test=round(logreg.score(x_test,y_test)*100,2)

print("Training Accuracy: %{}".format(acc_log_train))

print("Test Accuracy: %{}".format(acc_log_test))
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

    clf.fit(x_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results=pd.DataFrame({"Cross Validation Means":cv_result,"ML Models":["DecisionTreeClassifier","SVC","RandomForestClassifier","LogisticRegression","KNeighborsClassifier"]})



g= sns.barplot("Cross Validation Means","ML Models",data=cv_results)

g.set_xlabel("Mean Acc.")

g.set_title("Cross Validation Scores")

plt.show()
votingC=VotingClassifier(estimators=[("dt",best_estimators[0]),

                                    ("rf",best_estimators[2]),

                                    ("lr",best_estimators[3])],

                                    voting="soft",n_jobs=-1)

votingC=votingC.fit(x_train,y_train)

print(accuracy_score(votingC.predict(x_test),y_test))