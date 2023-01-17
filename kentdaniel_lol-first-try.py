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
df=pd.read_csv("/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
df.head()
df.info()
def show_unique(df,col):
    df_info=pd.DataFrame(index=df.columns,data=[df[i].nunique() for i in col])
    return df_info.T.style.background_gradient(axis=1)
show_unique(df,df.columns)
df_blue=pd.DataFrame(columns=[i for i in df.columns if i.startswith("blue")])
for i in df_blue.columns:
    df_blue[i]=df[i]
df_blue.head()
show_unique(df_blue,df_blue.columns)
df_red=pd.DataFrame(columns=[i for i in df.columns if i.startswith("red")])
for i in df_red.columns:
    df_red[i]=df[i]
df_red.head()
show_unique(df_red,df_red.columns)
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
sns.set_color_codes(palette='colorblind')
target="blueWins"
sns.countplot("blueWins",data=df_blue)
def plot_cont(df,con_ft,size):
    fig, ax = plt.subplots(ncols=2, nrows=0, figsize=size)
    plt.subplots_adjust(right=2)
    plt.subplots_adjust(top=2)
    for i, feature in enumerate(list(df[con_ft]),1):
        plt.subplot(len(list(con_ft)), 3, i)
        if feature.startswith("red"):
            sns.distplot(df[feature],color="red",kde=False)
        else:
            sns.distplot(df[feature],kde=False)

        plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
        plt.ylabel('skewness : %2f'%df[feature].skew(), size=15, labelpad=12.5)

        for j in range(2):
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})
    plt.show()
cont_ft=[i for i in df.columns if df[i].nunique()>100 and i!="gameId"]
plot_cont(df,cont_ft,(15,40))
cont_ft
def plot_disc(df,disc_ft,size):
    fig, ax = plt.subplots(ncols=2, nrows=0, figsize=size)
    plt.subplots_adjust(right=2)
    plt.subplots_adjust(top=2)
    for i, feature in enumerate(list(df[disc_ft]),1):
        plt.subplot(len(list(disc_ft)), 3, i)
        sns.countplot(df[feature])

        plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)

        for j in range(2):
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})

    plt.show()
disc_ft=[i for i in df.columns if df[i].nunique()<20 and i!="gameId"]
plot_disc(df,disc_ft,(12,30))
cat_ft=["blueFirstBlood","redFirstBlood","blueDragons","redDragons","blueHeralds","redHeralds"]
def plot_cont_target(con_ft,df,target,size):
    fig, ax = plt.subplots(ncols=2, nrows=0, figsize=size)
    plt.subplots_adjust(right=2)
    plt.subplots_adjust(top=2)
    for i, feature in enumerate(list(df[con_ft]),1):
        plt.subplot(len(list(con_ft)), 3, i)
        
        sns.boxplot(x=feature, y=target, data=df)

        plt.xlabel('{}'.format(feature), size=15,labelpad=12.5)
    

        for j in range(2):
            plt.tick_params(axis='x', labelsize=12)
            plt.tick_params(axis='y', labelsize=12)

        plt.legend(loc='best', prop={'size': 10})
    plt.show()
plot_cont_target(cont_ft,df,target,(15,40))
def catplot(ft,df,target):
    x,y = ft, target
    return (df
    .groupby(x)[y]
    .value_counts(normalize=True)
    .rename('percent')
    .reset_index()
    .pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
def plot_cat_target(df,cat_ft,target):
    for i in cat_ft:
        catplot(i,df,target)
plot_cat_target(df,cat_ft,target)
df_blue.head()
df_blue.describe().style.background_gradient()
# by taking just the difference the value represents both blue team and red team property for example if blue diffrence is negative then it will mean red team has a higher value in the feature, vice versa
df_blue["blueWardsPlaced"]=df["blueWardsPlaced"]-df["redWardsPlaced"]
df_blue["blueWardsDestroyed"]=df["blueWardsDestroyed"]-df["redWardsDestroyed"]
df_blue["blueKills"]=df["blueKills"]-df["redKills"]
df_blue["blueDeaths"]=df["blueDeaths"]-df["redDeaths"]
df_blue["blueAssists"]=df["blueAssists"]-df["redAssists"]
df_blue["blueEliteMonsters"]=df["blueEliteMonsters"]-df["redEliteMonsters"]
df_blue["blueDragons"]=df["blueDragons"]-df["redDragons"]
df_blue["blueHeralds"]=df["blueHeralds"]-df["redHeralds"]
df_blue["blueTowersDestroyed"]=df["blueTowersDestroyed"]-df["redTowersDestroyed"]
df_blue["blueAvgLevel"]=df["blueAvgLevel"]-df["redAvgLevel"]
df_blue["blueTotalMinionsKilled"]=df["blueTotalMinionsKilled"]-df["redTotalMinionsKilled"]
df_blue["blueTotalJungleMinionsKilled"]=df["blueTotalJungleMinionsKilled"]-df["redTotalJungleMinionsKilled"]
df_blue["blueCSPerMin"]=df["blueCSPerMin"]-df["redCSPerMin"]
df_blue["blueGoldPerMin"]=df["blueGoldPerMin"]-df["redGoldPerMin"]
df_blue.drop(columns=["blueTotalGold","blueTotalExperience"],inplace=True)
df_blue.describe().style.background_gradient()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
def baseline():
    model=LogisticRegression()
    X=df.drop(columns=[target]+["gameId"])
    y=df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15615)
    model.fit(X_train,y_train)
    y_true=y_test
    y_pred=model.predict(X_test)
    print(classification_report(y_true, y_pred))
    plt.figure(figsize = (8,6))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(confusion_matrix(y_true,y_pred), cmap="Blues", annot=True)# font size
baseline()
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB

model=SVC()
X=df_blue.drop(columns=[target])
y=df_blue[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15615)
model.fit(X_train,y_train)
y_true=y_test
y_pred=model.predict(X_test)
print(classification_report(y_true, y_pred))
plt.figure(figsize = (8,6))
sns.set(font_scale=1.4)#for label size
sns.heatmap(confusion_matrix(y_true,y_pred), cmap="Blues", annot=True)# font size
