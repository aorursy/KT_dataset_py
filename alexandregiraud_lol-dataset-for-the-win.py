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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import seaborn as sns
Df=pd.read_csv("../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
Df.head()
print(Df.shape)

Df.columns
Df.dtypes.value_counts()
Df.info()
Df.describe()
plt.figure(figsize=(20,10))

sns.heatmap(Df.isna(),cbar=False)
Df = Df.copy()

Df.drop(columns=["gameId"],inplace=True)
Df_Discret = Df.select_dtypes('int64')

Df_Continious = Df.select_dtypes('float64')
for col in Df_Continious.columns:

    plt.figure()

    sns.distplot(Df[col])
sns.pairplot(Df_Continious)
import warnings

warnings.filterwarnings('ignore')



for col in Df_Discret.columns:

    plt.figure()

    plt.title("{}".format(col))

    Df[col].plot.hist()
for col in Df.columns:

    print(f'{col :-<50} {"Average :" ,Df[col].mean()} {"Std :",Df[col].std()}')

    
Df_win = Df[Df["blueWins"]==1]

print(Df_win.shape)

Avg = Df_win.groupby(["blueWins"]).mean()

Avg
Values = [Avg["blueFirstBlood"].values,Avg["redFirstBlood"].values]

indexes = ['blue','red']

Série = pd.DataFrame(Values,index = indexes,columns = ['FirstBlood'])

print(Série)

Série.plot.pie(y='FirstBlood',fontsize=18,figsize=(9,9),title='First Blood',autopct='%1.1f%%',cmap="coolwarm")
Values_drake = [Avg["blueDragons"].values,1-Avg["redDragons"].values-Avg["blueDragons"].values,Avg["redDragons"].values]

indexes_drake = ['blue','No drake','red']

Serie_drake = pd.DataFrame(Values_drake,index = indexes_drake,columns = ['Nb'])

print(Serie_drake)



Values_her = [Avg["blueHeralds"].values,1-Avg["redHeralds"].values-Avg["blueHeralds"].values,Avg["redHeralds"].values]

indexes_her = ['blue','No Herald','red']

Serie_her = pd.DataFrame(Values_her,index = indexes_her,columns = ['Nb'])

print(Serie_her)



plt.figure()

Serie_drake.plot.pie(y='Nb',fontsize=18,figsize=(9,9),title='Drake',autopct='%1.1f%%',cmap="coolwarm")

Serie_her.plot.pie(y='Nb',fontsize=18,figsize=(9,9),title='Herald',autopct='%1.1f%%',cmap="coolwarm")

Avg_EliteMonster = Df.groupby(["blueEliteMonsters"]).mean()

Avg_EliteMonster

sns.barplot(x= Avg_EliteMonster.index , y=Avg_EliteMonster["blueWins"],palette="rocket")

print(Avg_EliteMonster["blueWins"])
Values = [Avg["blueWardsPlaced"].values,Avg["redWardsPlaced"].values]

indexes = ['blue','red']

Série = pd.DataFrame(Values,index = indexes,columns = ['Wards'])

print(Série)

Série.plot.pie(y='Wards',fontsize=18,figsize=(9,9),title='Wards Placed',autopct='%1.1f%%',cmap="coolwarm")

Df_lose = Df[Df["blueWins"]==0]
for col in Df_Continious.columns:

    plt.figure()

    sns.distplot(Df_win[col], label='Blue team win',color='blue')

    sns.distplot(Df_lose[col], label='Red team win',color='red')

    plt.legend()
#Some features in int64 has also interesting distributions

columns = ["blueTotalGold","blueTotalExperience","blueTotalMinionsKilled","blueGoldDiff","blueExperienceDiff"]

for col in columns:

    plt.figure()

    sns.distplot(Df_win[col], label='Blue team win',color='blue')

    sns.distplot(Df_lose[col], label='Red team win',color='red')

    plt.legend()
Binary_col = []

for col in Df.select_dtypes("int64").drop(columns =["blueWins"]):

    if len(Df[col].unique()) <5:

        Binary_col.append(col)

Binary_col
for col in Binary_col:

    plt.figure()

    sns.heatmap(pd.crosstab(Df["blueWins"],Df[col]),annot=True,fmt='d')
plt.figure(figsize=(20,12))

corr = Df.corr()

mask = np.triu(np.ones(corr.shape)).astype(np.bool)

sns.heatmap(corr,annot=False,mask = mask,cmap = "coolwarm")


sns.clustermap(Df.corr(),annot=False,cmap="coolwarm")
Df.corr()['blueWins'].sort_values()
plt.figure(figsize=(8,8))

X1 = Df["blueGoldDiff"][Df["blueWins"]==1]

Y1 = Df["blueWins"][Df["blueWins"]==1]

X2 = Df["blueGoldDiff"][Df["blueWins"]==0]

Y2 = Df["blueWins"][Df["blueWins"]==0]

plt.scatter(X1,Y1,c="blue",label="Blue Team Win")

plt.scatter(X2,Y2,c="red",label="Red Team Win")

plt.grid()

plt.title("Gold difference")

plt.legend()
from sklearn.model_selection import train_test_split
X = Df.drop(columns=["blueWins"])

y = Df["blueWins"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)



print("X_train shape :",X_train.shape)

print("X_test shape :",X_test.shape)

print("y_train shape :",y_train.shape)

print("y_test shape :",y_test.shape)
print(y_train.value_counts())

print(y_test.value_counts())
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC #Good on small datasets

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import SelectKBest,f_classif

from sklearn.preprocessing import PolynomialFeatures
model0 = DecisionTreeClassifier(random_state=0)

model1 = RandomForestClassifier(random_state=0)
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from sklearn.model_selection import learning_curve





def evaluation(model):

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    print(confusion_matrix(y_test,y_pred))

    print(classification_report(y_test,y_pred))

    

    N, train_score, val_score = learning_curve(model, X_train,y_train,cv=4,scoring='f1',

                                              train_sizes=np.linspace(0.1,1,10))

    

    plt.figure(figsize=(12,8))

    plt.plot(N,train_score.mean(axis=1),label='train_score')

    plt.plot(N,val_score.mean(axis=1),label='val_score')

    plt.legend()
evaluation(model0)

evaluation(model1)
preprocessor = make_pipeline(PolynomialFeatures(2,include_bias=False),SelectKBest(f_classif,k=10))
RandomForest = make_pipeline(preprocessor,RandomForestClassifier(random_state=0))

AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))

SVM = make_pipeline(preprocessor,StandardScaler(),SVC(random_state=0))

KNN = make_pipeline(preprocessor,StandardScaler(),KNeighborsClassifier())
dict_of_models = {"RandomForest":RandomForest,"AdaBoost":AdaBoost,"SVM":SVM,"KNN":KNN}
import warnings

warnings.filterwarnings('ignore')



for key,model in dict_of_models.items():

    print(key)

    evaluation(model)
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV
SVM
params = {"svc__gamma":[1e-3,1e-4],

              "svc__C":[1,10,100,1000],"svc__degree":[2,3,4,5]}
grid = RandomizedSearchCV(SVM,params,scoring="accuracy",cv=4, n_iter=10)



grid.fit(X_train, y_train)

#Very Long , use RandomizedSearchCV



print(grid.best_params_)



y_pred = grid.predict(X_test)



print(classification_report(y_test,y_pred))
from sklearn.metrics import precision_recall_curve
precision,recall,threshold = precision_recall_curve(y_test,grid.best_estimator_.decision_function(X_test))
plt.plot(threshold,precision[:-1],label ='precision')

plt.plot(threshold,recall[:-1],label ='recall')

plt.legend()
def model_final(model,X,threshold=0):

    return model.decision_function(X) > threshold
y_pred = model_final(grid.best_estimator_,X_test)
model_final = grid.best_estimator_

print("f1_score = " ,f1_score(y_test,y_pred))

print(model_final.score(X_test,y_test))
from sklearn.manifold import TSNE



tsne = TSNE(n_components=2, random_state=42)

X_reduced = tsne.fit_transform(X)
import matplotlib.pyplot as plt

plt.figure(figsize=(13,10))

plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="coolwarm")

plt.axis('off')

plt.colorbar()

plt.show()