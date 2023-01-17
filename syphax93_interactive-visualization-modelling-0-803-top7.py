import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import math

from math import *



from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objects as go

from ipywidgets import widgets

from ipywidgets import *

# Ce code fonctionne dans un notebook jupyter.

init_notebook_mode(connected=True)
%matplotlib inline
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')

df_sub = pd.read_csv('../input/titanic/gender_submission.csv')
df_train.dtypes.value_counts()
df_test.dtypes.value_counts()
df_train["Survived"].value_counts()
labels=df_train["Survived"].value_counts().index

values=df_train["Survived"].value_counts().values

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial'

                            )])

fig.show()
w_variable = widgets.Dropdown(

    options=["Age","Fare"],

    value="Age",

    description='Variable:',

    disabled=False,

)



def plot_histo(variable):

    fig, ax = plt.subplots(ncols=1, figsize=(12, 8))

    sns.distplot(df_train[df_train["Survived"]==1][variable].dropna(), color="skyblue",bins=30, label="Survived")

    sns.distplot(df_train[df_train["Survived"]==0][variable].dropna(), color="red",bins=30, label="Not Survived")

    plt.legend()        

widget=interactive(plot_histo,variable=w_variable)

widget
fig, ax = plt.subplots(ncols=2,nrows=5,figsize=(15, 20))



df_train['Sex'].value_counts().plot.pie(ax=ax[0,0])

sns.countplot(x='Sex', hue="Survived", data=df_train,ax=ax[0,1])



df_train['Embarked'].value_counts().plot.pie(ax=ax[1,0])

sns.countplot(x='Embarked', hue="Survived", data=df_train,ax=ax[1,1])



df_train['Pclass'].value_counts().plot.pie(ax=ax[2,0])

sns.countplot(x='Pclass', hue="Survived", data=df_train,ax=ax[2,1])



df_train['SibSp'].value_counts().plot.pie(ax=ax[3,0])

sns.countplot(x='SibSp', hue="Survived", data=df_train,ax=ax[3,1])



df_train['Parch'].value_counts().plot.pie(ax=ax[4,0])

sns.countplot(x='Parch', hue="Survived", data=df_train,ax=ax[4,1])



plt.legend()
fig, ax = plt.subplots(ncols=2, figsize=(20, 8))

sns.heatmap(df_train.isnull(),yticklabels=False,cbar=False,cmap='BuPu',ax=ax[0])

sns.heatmap(df_test.isnull(), yticklabels=False,cbar=False, cmap='BuPu',ax=ax[1])

ax[0].set_title('Train Set Missing Values')

ax[1].set_title('Test Set Missing Values')

plt.xticks(rotation=90)

plt.show()
missing_rate_train = (df_train.isna().sum()/df_train.shape[0]).sort_values()

nb_missing = df_train.isna().sum().sort_values()

print(f'{"Variable" :-<40} {"missing_rate_train":-<30} {"Number of missing values":-<30}')

for n in range(len(missing_rate_train)):

    print(f'{missing_rate_train.index[n] :-<30} {missing_rate_train[n]:-<30} {nb_missing[n]:-<30}')
missing_rate_test = (df_test.isna().sum()/df_test.shape[0]).sort_values()

nb_missing = df_test.isna().sum().sort_values()

print(f'{"Variable" :-<30} {"missing_rate_train":-<30} {"Number of missing values":-<30}')

for n in range(len(missing_rate_test)):

    print(f'{missing_rate_test.index[n] :-<30} {missing_rate_test[n]:-<30} {nb_missing[n]:-<30}')
def TransfromTitle(df_aux):

    

    title = []

    for name in df_aux['Name'] :

        p1 = name.find(',') # position of coma

        p2 = name.find('.') # position of point

        if p1 != -1 or p2!= -1:

            title.append(name[p1+2:p2])

        else :

            title.append(np.nan) 

    df_aux['title'] = pd.DataFrame(title, index=df_aux.index)

    

    title2 = []

    for t in df_aux['title'] :

        if t in ["Mr","Miss","Mrs","Master"]:

            title2.append(t)

        elif t in ['Don', 'Mme', 'Ms', 'Lady', 'Sir', 'Mlle', 'the Countess','Jonkheer', 'Dona']: 

            title2.append("royale")

        elif t in ['Major','Col', 'Capt','Rev','Dr']:

            title2.append("officier/capitaine")

    df_aux['title'] = pd.DataFrame(title2, index=df_aux.index) 

    

    return df_aux.drop(columns=["PassengerId","Name"])
df1 = TransfromTitle(df_train)

df1.head(5)
fig, ax = plt.subplots(ncols=2, figsize=(20, 7))

sns.countplot(x='title', hue="Survived", data=df1,ax=ax[0])

sns.countplot(x='Pclass', hue="Survived", data=df1,ax=ax[1])

ax[0].set_title('Histogram')

ax[1].set_title('Histogram')
fig, ax = plt.subplots(ncols=1, figsize=(15, 5))

sns.distplot(df1[df1["title"]=='Mr']["Age"].dropna(), color="red",bins=30, label='Mr')

sns.distplot(df1[df1["title"]=='Mrs']["Age"].dropna(), color="green",bins=30, label='Mrs')

sns.distplot(df1[df1["title"]=='Miss']["Age"].dropna(), color="blue",bins=30, label='Miss')

sns.distplot(df1[df1["title"]=='Master']["Age"].dropna(), color="skyblue",bins=2, label='Master')

sns.distplot(df1[df1["title"]=='officier/capitaine']["Age"].dropna(), color="yellow",bins=2, label='officier/capitaine')

plt.legend()
fig, ax = plt.subplots(ncols=1, figsize=(15, 5))

sns.distplot(df1[df1["Pclass"]==3]["Age"].dropna(), color="red",bins=30, label="Class 3")

sns.distplot(df1[df1["Pclass"]==2]["Age"].dropna(), color="green",bins=30, label="Class 2")

sns.distplot(df1[df1["Pclass"]==1]["Age"].dropna(), color="blue",bins=30, label="Class 1")

plt.legend()
def TransfromAge(df_aux):

    GroupAge = ['inf-10', '10-18', '18-35', '35-65', 'sup-65']

    cond1 = (df_aux["Age"].isnull())&(df_aux["title"]=="Master")

    df_aux.loc[cond1, 'Age'] = calcul_median(df_aux,"Master")



    cond2 = (df_aux["Age"].isnull())&(df_aux["title"]=="Miss")

    df_aux.loc[cond2, 'Age'] = calcul_median(df_aux,"Miss")



    cond3 = (df_aux["Age"].isnull())&(df_aux["title"]=="Mrs")

    df_aux.loc[cond3, 'Age'] = calcul_median(df_aux,"Mrs")



    cond4 = (df_aux["Age"].isnull())&(df_aux["title"]=="Mr")

    df_aux.loc[cond4, 'Age'] = calcul_median(df_aux,"Mr")



    cond5 = (df_aux["Age"].isnull())&(df_aux["title"]=="officier/capitaine")

    df_aux.loc[cond5, 'Age'] = calcul_median(df_aux,"officier/capitaine")



    cond6 = (df_aux["Age"].isnull())&(df_aux["title"]=="royale")

    df_aux.loc[cond6, 'Age'] = calcul_median(df_aux,"royale")

    # Age group

    bins = [0, 10, 18, 35, 65, np.inf]

    GroupAge = ['inf-10', '10-18', '18-35', '35-65', 'sup-65']

    df_aux['GroupAge'] = pd.cut(df_aux['Age'], bins, labels=GroupAge)

    return df_aux



def calcul_median(df_aux,ch):

    return df_aux[df_aux["title"]==ch].Age.dropna().median()
df_test[df_test["Fare"].isnull()]
fig, ax = plt.subplots(ncols=1, figsize=(15, 5))

sns.distplot(df1[df1["Pclass"]==3]["Fare"].dropna(), color="blue",bins=50, label='Pclass 3')

plt.legend()
def TransfromFare(df_aux):

    df_aux.loc[(df_aux["Fare"].isnull())&(df_aux["Pclass"]==3), 'Fare'] = df_aux[df_aux["Pclass"]==3].Fare.dropna().median()

    # Fare group

    bins = [-1,8,14,20,60,100,600]

    GroupFare = ['0-8£','8-14£','14-20£','20-60£','60-100£','100-515£']

    df_aux['GroupFare'] = pd.cut(df_aux['Fare'], bins, labels=GroupFare)

    return df_aux
def TransfromCabin(df_aux):

    df_aux.loc[(df_aux["Cabin"].isnull()), 'HasOrNotCabinNumber'] = "Has Not Cabin Number"

    df_aux.loc[(df_aux["Cabin"].notnull()), 'HasOrNotCabinNumber'] = "Has Cabin Number"

    return df_aux.drop(columns=["Cabin","Ticket"])
df_train[df_train["Embarked"].isnull()]
fig, ax = plt.subplots(ncols=2, figsize=(20, 5))

sns.countplot(x='Embarked', hue="Survived", data=df_train[(df_train["Pclass"]==1)],ax=ax[0])

df_train['Embarked'].value_counts().plot.pie(ax=ax[1])

ax[0].set_title('Embarked/Survived for first class')

ax[1].set_title('value count of Embarked')

plt.show()
def TransfromEmbarked(df_aux):

    df_aux.loc[(df_aux["Embarked"].isnull())&(df_aux["Pclass"]==1), 'Embarked'] = "S"

    return df_aux
def TransfromFamiliy(df_aux):

    df_aux["familiySize"] = df_aux["SibSp"] + df_aux["Parch"] + 1

    

    AloneTravel = (df_aux['SibSp'] == 0) & (df_aux['Parch'] == 0)

    CoupleTravel = (df_aux['SibSp'] == 0) & (df_aux['Parch'] == 1)

    siblingsTravel = (df_aux['SibSp'] == 1) & (df_aux['Parch'] == 0)

    SmallFamilly = (df_aux['familiySize'] <= 3) & (df_aux['familiySize'] >= 2)

    BigFamilly =(df_aux['familiySize'] >3)

    

    df_aux.loc[AloneTravel, 'familiy'] = "Alone Travel"

    df_aux.loc[CoupleTravel, 'familiy'] = "Couple Travel"

    df_aux.loc[siblingsTravel, 'familiy'] = "siblings Travel"

    df_aux.loc[SmallFamilly, 'familiy'] = "Small Familly"

    df_aux.loc[BigFamilly, 'familiy'] = "Big Familly"

    

    return df_aux
dfTrain = TransfromTitle(df_train)

dfTrain = TransfromAge(dfTrain)

dfTrain = TransfromFare(dfTrain)

dfTrain = TransfromCabin(dfTrain)

dfTrain = TransfromEmbarked(dfTrain)

dfTrain = TransfromFamiliy(dfTrain)
dfTrain = dfTrain[['familiySize','SibSp','Parch','title','GroupAge','GroupFare','HasOrNotCabinNumber',

         'Sex','Embarked','Pclass','familiy','Survived']]
dfTrain.head(10)
dfTest = TransfromTitle(df_test)

dfTest = TransfromAge(dfTest)

dfTest = TransfromFare(dfTest)

dfTest = TransfromCabin(dfTest)

dfTest = TransfromEmbarked(dfTest)

dfTest = TransfromFamiliy(dfTest)
dfTest = dfTest[['familiySize','SibSp','Parch','title','GroupAge','GroupFare','HasOrNotCabinNumber',

         'Sex','Embarked','Pclass','familiy']]
dfTest.head(10)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.impute import KNNImputer

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
X_train = dfTrain.drop(columns = ['Survived']).values

y = dfTrain.Survived.values

X_test = dfTest.values
for i in range(len(X_train)):

    X_train[i,9] = str(X_train[i,9])



X_train[:,0:3] = StandardScaler().fit_transform(X_train[:,0:3])





onehotencoder_1 = OneHotEncoder()

u1 = onehotencoder_1.fit_transform(X_train[:,3:]).toarray()



X_train2 = np.concatenate((X_train[:,0:3], u1), axis=1)

X_train2.shape
StandardScaler().fit_transform(X_test[:,0:3]).shape

X_test[:,0:3].shape
for i in range(len(X_test)):

    X_test[i,9] = str(X_test[i,9])



X_test[:,0:3] = StandardScaler().fit_transform(X_test[:,0:3])



onehotencoder_2 = OneHotEncoder()

u2 = onehotencoder_2.fit_transform(X_test[:,3:]).toarray()



X_test2 = np.concatenate((X_test[:,0:3], u2), axis=1)

X_test2.shape
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import xgboost as xgb

logreg = LogisticRegression()

Gauss = GaussianNB()

rf = RandomForestClassifier()

gboost = GradientBoostingClassifier()

DTC =  DecisionTreeClassifier()

RF = RandomForestClassifier(n_estimators=200)

SVectorMachine = SVC()

xgb = xgb.XGBClassifier(max_depth=3, n_estimators=10, learning_rate=0.01)

models = [logreg,Gauss, gboost,DTC,RF,SVectorMachine,xgb]
def compute_score(clf, X, y, scoring='accuracy'):

    xval = cross_val_score(clf, X, y, cv = 10, scoring=scoring)

    return np.mean(xval)
for model in models:

    print('Cross-validation of : {0}'.format(model.__class__))

    score = compute_score(clf=model, X=X_train2, y=y, scoring='accuracy')

    print('CV score = {0}'.format(score))

    print('----->>>>>>')


X = np.asarray(X_train2).astype(np.float32)

Y = np.asarray(y).astype(np.float32)
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential

from keras.layers import Dense, Dropout



classifier = Sequential() 

classifier.add(Dense(units = 35,activation = "relu",kernel_initializer="uniform",input_dim=33))

classifier.add(Dropout(rate=0.1))

classifier.add(Dense(units = 20,activation = "relu",kernel_initializer="uniform"))

classifier.add(Dense(units = 15,activation = "relu",kernel_initializer="uniform"))

classifier.add(Dense(units = 1,activation = "sigmoid",kernel_initializer="uniform"))

classifier.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=['acc'])

classifier.fit(X,Y,batch_size=50,epochs=110)
result = []

Y_pred = classifier.predict(np.asarray(X_test2).astype(np.float32))

Y_pred = (Y_pred>0.55)



for i in range(len(Y_pred)):

    if Y_pred[i][0] == True :

        result.append(1)

    else :

        result.append(0)

PassengerId = df_test["PassengerId"] # PassengerId,Survived

SurvivedResult = pd.DataFrame({'Survived': result})

results = pd.concat([PassengerId,SurvivedResult],axis=1)

results.to_csv("gender_submission.csv",sep = ',',index=False)
results