# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from collections import Counter

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
lol_df = pd.read_csv("/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
lol_df.head()

lol_df.describe
lol_df.info()
blue_df = lol_df.iloc[:,1:20]

red_df = lol_df.iloc[:,21:]

red_df["redWins"] = lol_df.blueWins

red_df["redWins"] = [1 if i==0 else 0 for i in red_df["redWins"]]
fig, ax = plt.subplots(figsize=(20,5))

sns.heatmap(blue_df.corr(), annot = True, fmt = ".2f",ax=ax)

plt.show()
fig, ax = plt.subplots(figsize=(20,5))

sns.heatmap(red_df.corr(), annot = True, fmt = ".2f",ax=ax)

plt.show()
sns.countplot(lol_df.blueWins)

#sns.countplot(kill.manner_of_death)

plt.title("Blue Wins",color = 'blue',fontsize=15)
g = sns.factorplot(x = "redWins", y = "redTotalExperience", kind = "bar", data = red_df, size = 6)

g.set_ylabels("Red Total Experince")

g.set_xlabels("Red Wins")

plt.show()
g = sns.factorplot(x = "blueWins", y = "blueTotalExperience", kind = "bar", data = blue_df, size = 6)

g.set_ylabels("Blue Total Experince")

g.set_xlabels("Blue Wins")

plt.show()
g = sns.factorplot(x = "blueWins", y = "blueAvgLevel", kind = "bar", data = blue_df, size = 6)

g.set_ylabels("Blue Average Level")

g.set_xlabels("Blue Wins")

plt.show()
g = sns.factorplot(x = "redWins", y = "redAvgLevel", kind = "bar", data = red_df, size = 6)

g.set_ylabels("Blue Average Level")

g.set_xlabels("Blue Wins")

plt.show()
g = sns.factorplot(x = "redWins", y = "redTotalGold", kind = "bar", data = red_df, size = 6)

g.set_ylabels("Red Total Gold")

g.set_xlabels("Red Wins")

plt.show()
g = sns.factorplot(x = "blueWins", y = "blueCSPerMin",data=lol_df, kind = "bar", size = 6)

g.set_ylabels("Blue CS Per min")

g.set_xlabels("Blue Wins")

a = sns.factorplot(x = "blueWins", y = "redCSPerMin",data=lol_df ,kind = "bar", size = 6)

a.set_ylabels("Red CS Per min")

a.set_xlabels("Blue Wins")

plt.show()

list1=lol_df[lol_df.columns[1:-1]].apply(lambda x: x.corr(lol_df['blueWins']))

list2=[]

for i in list1.index:

    if (-0.2<list1[i]<0.2):

        list2.append(i)

        

lol_df.drop(list2,axis=1,inplace=True)

lol_df.drop("gameId",axis=1,inplace=True)
y=lol_df["blueWins"]

x=lol_df.iloc[:,1:]


x = (x-np.min(x))/(np.max(x)-np.min(x))
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



grid = {"n_neighbors":np.arange(1,50)}



knn = KNeighborsClassifier()



knn_cv=GridSearchCV(knn,grid,cv=10)



knn_cv.fit(x_train,y_train)



print("tuned hypermeter = ",knn_cv.best_params_)



print("tuned hypermeter = ",knn_cv.best_score_)
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier 

grid = {"n_estimators":[100,200,300,400,500]}

rf = RandomForestClassifier()

rf_cv=GridSearchCV(rf,grid,cv=10,n_jobs = -1,verbose = 1)

rf_cv.fit(x_train,y_train)

print("tuned hypermeter = ",rf_cv.best_params_)



print("tuned hypermeter = ",rf_cv.best_score_)
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

dt_param_grid={"min_samples_split":range(10,500,20),

              "max_depth":range(1,20,2)}

dt_cv=GridSearchCV(dt,dt_param_grid,cv=10)

dt_cv.fit(x_train,y_train)

print("tuned hypermeter = ",dt_cv.best_params_)



print("tuned hypermeter = ",dt_cv.best_score_)
from sklearn.svm import SVC



svm =SVC()

from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(estimator=svm,X = x_train,y = y_train,cv=10)



print("mean =", np.mean(accuracies))



print("std =", np.std(accuracies))
from sklearn.naive_bayes import GaussianNB



nb=GaussianNB()



accuracies2 = cross_val_score(estimator=nb,X = x_train,y = y_train,cv=10)



print("mean =", np.mean(accuracies2))



print("std =", np.std(accuracies2))
from sklearn.linear_model import LogisticRegression



param_girid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]} #l1=lassa l2 =ridge



logreg= LogisticRegression()



logreg_cv=GridSearchCV(logreg,param_girid,cv=10)



logreg_cv.fit(x_train,y_train)





print("tuned hyperparameter = ",logreg_cv.best_params_)



print("accuracy = ", logreg_cv.best_score_)
from sklearn.metrics import confusion_matrix

log_reg = LogisticRegression(C=1, penalty="l2")

log_reg.fit(x_train,y_train)

y_predict=log_reg.predict(x_train)

cm = confusion_matrix(y_train,y_predict)



#%%

import seaborn as sns

import matplotlib.pyplot as plt

f , ax = plt.subplots(figsize=(5,5))



sns.heatmap(cm,annot = True,linewidths =0.5,linecolor ="Red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()

print("Score= ",log_reg.score(x_test,y_test))
from sklearn.metrics import confusion_matrix

nb=GaussianNB()

nb.fit(x_train,y_train)

y_predict2=nb.predict(x_train)

cm = confusion_matrix(y_train,y_predict2)



#%%

import seaborn as sns

import matplotlib.pyplot as plt

f , ax = plt.subplots(figsize=(5,5))



sns.heatmap(cm,annot = True,linewidths =0.5,linecolor ="Red",fmt=".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()

print("Score= ",nb.score(x_test,y_test))
from sklearn.ensemble import VotingClassifier

votingC = VotingClassifier(estimators = [("lr",LogisticRegression(C=1, penalty="l2")),

                                        ("gnb",GaussianNB())],                        

                                        voting = "soft")

votingC = votingC.fit(x_train, y_train)

print("Score= ", votingC.score(x_test,y_test))