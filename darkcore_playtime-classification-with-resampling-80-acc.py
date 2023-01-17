import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from warnings import filterwarnings

filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv("/kaggle/input/performance-prediction/summary.csv")

data.head()
data.info()
data.describe()
data["Target"].value_counts()
names = data["Name"]

data.drop(["Name"],inplace=True,axis=1)
import seaborn as sns

plt.subplots(figsize=(18,14))

sns.heatmap(data.corr(),annot=True,linewidths=0.4,linecolor="black",fmt="1.2f",cbar=False)

plt.title("Correlation",fontsize=50)

plt.xticks(rotation=35)

plt.show()
targetLoves = ["GamesPlayed","MinutesPlayed","PointsPerGame","FieldGoalsMade","FieldGoalsAttempt","FreeThrowMade","FreeThrowAttempt"]
import plotly.graph_objects as go

from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

import plotly.figure_factory as ff

import plotly.io as pio



init_notebook_mode(True)

#I will round floats for better visualize 

rounded_data = data.apply(lambda x : round(x))

rounded_data.head()
fig1 = px.scatter_matrix(rounded_data,color="Target",dimensions=list(rounded_data.columns[:9])+["Target"])

fig1.update_traces(marker=dict(showscale=False),diagonal_visible=False)

fig1.update_layout(title={"text":"Scatter Matrix-1","x":0.5,"font":{"size":35}},

                   height=1500,showlegend=False)

fig1.layout.coloraxis.showscale = False

fig1.show()
fig2 = px.scatter_matrix(rounded_data,color="Target",dimensions=list(rounded_data.columns[10:]))

fig2.update_traces(marker=dict(showscale=False),diagonal_visible=False)

fig2.update_layout(title={"text":"Scatter Matrix-2","x":0.5,"font":{"size":35}},

                   height=1500,showlegend=False)

fig2.layout.coloraxis.showscale = False

fig2.show()
px.box(rounded_data)
distlabels = ["FieldGoalsMade","FreeThrowMade","Steals","Blocks","Turnovers"]

hist_data = rounded_data[distlabels]

fig3 = ff.create_distplot([hist_data[i] for i in list(hist_data)],group_labels=distlabels,curve_type="normal")

fig3.show()
nameList = []

def splitMyName(x):

    global nameList

    for name in x.split():

        nameList.append(name)

names.apply(splitMyName)

nameList[:10]
from collections import Counter



nameCount = Counter(nameList)

countedNameDict = dict(nameCount)

sortedNameDict = sorted(countedNameDict.items(),key = lambda x : x[1],reverse=True)

print("Most Used 20 Names")

for name,counted in sortedNameDict[0:20]:

    print("{} : {}".format(name,counted))
from wordcloud import WordCloud



namecloud = WordCloud(max_words=500,background_color="white",min_font_size=4).generate_from_frequencies(countedNameDict)

plt.figure(figsize=[13,10])

plt.axis("off")

plt.title("Name Cloud",fontsize=20)

plt.imshow(namecloud)

plt.show()
list(names[9:13])
dictForNameCorr = dict()

nameSurnameCounter = 1

def nameSurnameEqualMyNumber(x):

    global dictForNameCorr

    global nameSurnameCounter

    x = x.split()

    if x[0] in dictForNameCorr:

        return dictForNameCorr[x[0]]

    elif x[1] in dictForNameCorr:

        return dictForNameCorr[x[1]]

    else:

        dictForNameCorr[x[0]] = nameSurnameCounter

        nameSurnameCounter += 1

        dictForNameCorr[x[1]] = nameSurnameCounter

        nameSurnameCounter += 1

        return dictForNameCorr[x[0]]

for i in names[9:13]:

    print(f"{i} : {nameSurnameEqualMyNumber(i)}")
nameForCorr = names.apply(nameSurnameEqualMyNumber)
nameForCorr
sns.heatmap(np.corrcoef(nameForCorr,data["Target"]),annot=True)

plt.title("Correlation between Name and Target")

plt.show()
forSample = pd.DataFrame({"Name":nameForCorr,"Target":data["Target"]})

print(forSample.Target.value_counts())

forSample.head()
forSample = forSample.sort_values(by="Target")[:509*2]

forSample.Target.value_counts()
from sklearn.model_selection import train_test_split



name_train,name_test,target_train,target_test = train_test_split(forSample["Name"].values.reshape(-1,1),forSample["Target"].values,random_state = 40,test_size = 0.2)
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, confusion_matrix



xgb = XGBClassifier()

xgb.fit(name_train,target_train)

ypred = xgb.predict(name_test)

plt.subplots(figsize=(18,14))

sns.heatmap(confusion_matrix(ypred,target_test),annot=True,fmt="1.2f",cbar=False,annot_kws={"size": 20})

plt.title(f"Name->Target Accuracy: {accuracy_score(ypred,target_test)}",fontsize=40)

plt.xlabel("Target",fontsize=30)

plt.show()
targetLoves
X = data[targetLoves]

y = data["Target"]
from imblearn.combine import SMOTETomek





smothy = SMOTETomek(random_state = 42)

smothy.fit(X,y)

X_smothy,y_smothy = smothy.fit_resample(X,y)
print("New Counts After Combining Under and Over Sampling")

print(y_smothy[y_smothy==0].value_counts())

print(y_smothy[y_smothy==1].value_counts())
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler(feature_range=(0,1))

x = mms.fit_transform(X_smothy)
x_train,x_test,y_train,y_test = train_test_split(x,y_smothy,test_size=0.2,random_state=67)
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier



classifier_list = [["Logistic Regression",LogisticRegression()],

                  ["RandomForest Classifier",RandomForestClassifier()],

                  ["AdaBoost Classifier",AdaBoostClassifier()],

                  ["DecisionTree Classifier",DecisionTreeClassifier()],

                  ["KNeighbors Classifier",KNeighborsClassifier()],

                  ["SVC",SVC()],

                  ["GaussianNB",GaussianNB()],

                  ["LGBM Classifier",LGBMClassifier()],

                  ["XGB Classifier",XGBClassifier()]]

for modelName, classifier in classifier_list:

    classifier.fit(x_train,y_train)

    print(f"{modelName} Accuracy: {accuracy_score(classifier.predict(x_test),y_test)}")

lgbm = LGBMClassifier(boosting_type='gbdt', num_leaves=30, max_depth=56, learning_rate=0.1, 

                      n_estimators=42, subsample_for_bin=30000, objective=None, 

                      class_weight=None, min_split_gain=0.0, min_child_weight=0.1, 

                      min_child_samples=20, subsample=0.4, subsample_freq=0, colsample_bytree=0.3, 

                      reg_alpha=0.0, reg_lambda=0.0, random_state=42, n_jobs=- 1, silent=True, 

                      importance_type='split')

lgbm.fit(x_train,y_train)

ypred = lgbm.predict(x_test)

plt.subplots(figsize=(18,14))

sns.heatmap(confusion_matrix(ypred,y_test),annot=True,fmt="1.0f",cbar=False,annot_kws={"size": 20})

plt.title(f"LGBM Accuracy: {accuracy_score(ypred,y_test)}",fontsize=40)

plt.xlabel("Target",fontsize=30)

plt.show()
xgb = XGBClassifier(base_score=0.4, booster='gbtree', colsample_bylevel=1,

       colsample_bynode=1, colsample_bytree=0.4, gamma=0.00025,

       importance_type='gain', learning_rate=0.0099, max_delta_step=0,

       max_depth=8, min_child_weight=0, missing=None, n_estimators=512,

       n_jobs=1, nthread=None, random_state=0,

       reg_alpha=0.00004, reg_lambda=1, scale_pos_weight=1, seed=42,

       silent=None, subsample=0.3, verbosity=1)

xgb.fit(x_train,y_train)

ypred = xgb.predict(x_test)

plt.subplots(figsize=(18,14))

sns.heatmap(confusion_matrix(ypred,y_test),annot=True,fmt="1.0f",cbar=False,annot_kws={"size": 20})

plt.title(f"XGB Accuracy: {accuracy_score(ypred,y_test)}",fontsize=40)

plt.xlabel("Target",fontsize=30)

plt.show()