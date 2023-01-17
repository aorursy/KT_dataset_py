# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

import re
import os
print(os.listdir("../input"))

#modeling
from subprocess import check_output
from sklearn.svm import SVC
from sklearn import svm, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, AdaBoostClassifier,
GradientBoostingClassifier,ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import xgboost as xbg

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
# Any results you write to the current directory are saved as output.
nulls=df_train.isnull().sum()
print(nulls)

print(df_test.isnull().sum())
#(추후 fare 한개만 따로 채워주기)
length = len(df_train)
percentage = (nulls/length)*100
print (percentage)
#plotting
x = percentage.values
y = np.array(percentage.index)

plt.figure(figsize=(16, 5))
sns.set(font_scale=1.2)
ax = sns.barplot(y, x, palette='hls', log=False)
ax.set(xlabel='Feature', ylabel='(Percentage of Nulls)', title='Number of Nulls')
df_train.head()
for i in np.array(df_train.columns):
    print ("{0} has {1} attributes".format(i, len(df_train[i].unique())))
df = pd.concat([df_train.drop("Survived", axis = 1), df_test], ignore_index = True)
label = df_train["Survived"]
index = df_train.shape[0]
df["Ticket_number"] = df["Ticket"].apply(lambda x: x.split()[-1])
df["Ticket_code"] = df["Ticket"].apply(lambda x: x.split()[0] if len(x.split())!= 1 else "No Code")
df["Ticket_code"].unique()
df['Ticket_code'].value_counts()
df["Ticket_code"]= df["Ticket_code"].apply(lambda a: a[:-1] if a[-1]=="." else a)
df['Ticket_code'].value_counts()
#/와 . 모두 분리
import re
codes= [i for i in df["Ticket_code"].unique() if i!= "No Code"]
def split_codes(code):
    return re.split('[^a-zA-Z0-9]+', code)
    
new_codes = []
for i in codes:
    for j in  split_codes(i):
        new_codes.append(j)
        
pd.Series(new_codes).value_counts()
#/만 분리
def split_codes2(code):
    return re.split('/+', code)
    
new_codes2 = []
for i in codes:
    for j in  split_codes2(i):
        new_codes2.append(j)
        
pd.Series(new_codes2).value_counts()
#.만 분리
def split_codes3(code):
    return re.split('\.+', code)
    
new_codes3 = []
for i in codes:
    for j in  split_codes3(i):
        new_codes3.append(j)
        
pd.Series(new_codes3).value_counts()
df["Has_ticket_codes"] = df["Ticket_code"].apply(lambda x: 0 if x=="No Code" else 1)
df["Has_only_1_codes"] = df["Ticket_code"].apply(lambda x: 0 if len(re.split('[^a-zA-Z0-9]+', x)) !=1 else 1)
df["Number_of_codes"] = df["Ticket_code"].apply(lambda x: 0 if x=="No Code" else len(re.split('[^a-zA-Z0-9]+', x)))

df["Ticket_code_HEAD"] = df["Ticket_code"].apply(lambda x: "No Code" if x == "No Code" else re.split('[^a-zA-Z0-9]+', x)[0])
df["Ticket_code_TAIL"] = df["Ticket_code"].apply(lambda x: "No Code" if x == "No Code" else re.split('[^a-zA-Z0-9]+', x)[-1])

df['Ticket_code_HEAD'].value_counts()
df['Ticket_code_TAIL'].value_counts()
df["Name"]
df["Initial"] = df.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(df["Initial"], df["Sex"]).T.style.background_gradient(cmap='summer_r')
df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
train = df[:length]
train["Survived"] = label
train.groupby('Initial').mean()["Survived"].plot.bar()
train.groupby('Initial').mean()
print ("{} percent of Cabin data is null".format(df["Cabin"].isnull().sum()/len(df)*100))
train_cabin = train[train["Cabin"].notnull()]
train_cabin["Cabin_Initial"] = train_cabin["Cabin"].apply(lambda x: x[0])
train_cabin["Cabin_Number"] = train_cabin["Cabin"].apply(lambda x: (x[1:].split(" ")[0]))
train_cabin["Cabin_Number"].replace("", -1, inplace = True)
train_cabin["Cabin_Number"] = train_cabin["Cabin_Number"].apply(lambda x: int(x))

train_cabin["Cabin_Initial"].value_counts()
train_cabin.groupby("Cabin_Initial").mean()
train_cabin.groupby("Cabin_Initial").mean()["Survived"].plot.bar()
df_cabin = df[df["Cabin"].notnull()]
df_cabin["Cabin_Initial"] = df_cabin["Cabin"].apply(lambda x: x[0])
df_cabin["Cabin_Number"] = df_cabin["Cabin"].apply(lambda x: (x[1:].split(" ")[0]))
df_cabin["Cabin_Number"].replace("", -1, inplace = True)
df_cabin["Cabin_Number"] = df_cabin["Cabin_Number"].apply(lambda x: int(x))

df_cabin.groupby("Cabin_Initial").mean()
#Heatmap 그려보기
df_cabin_heatmap = df_cabin[["Pclass", "Age", "SibSp", "Parch", "Fare", "Cabin_Number", "Cabin_Initial"]]
df_cabin_heatmap['Cabin_Initial'] = df_cabin_heatmap['Cabin_Initial'].map({'A': 0, 'B': 1, "C":2, "D":3, "E":4, "F":5, "G":7, "H":8})

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Correlation, y=1.05, size=15')
sns.heatmap(df_cabin_heatmap.astype(float).corr(), linewidths=0.1, vmax=1.0,
           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})

df["Embarked"].isnull().sum()
df['Embarked'].fillna('S', inplace=True)
from sklearn import preprocessing
#Drop non-using columns
df = df.drop(["Name", "Ticket"], axis = 1)
categorical = ["Sex", "Embarked","Ticket_code", "Ticket_code_HEAD", "Ticket_code_TAIL", "Initial"] 

lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col].fillna('Unknown')
    df[col] = lbl.fit_transform(df[col].astype(str))
df.head()
df.groupby("Initial").mean()
#Initial 을 가지고 Fillna
df["Age"] = df.groupby("Initial").transform(lambda x: x.fillna(x.mean()))["Age"]
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df["Isalone"] = df["FamilySize"].apply(lambda x: 0 if x!=1 else 1)
df= df.drop("Ticket_number", axis = 1)
df['Fare'].fillna((df['Fare'].median()), inplace=True)


#Age / Fare Category 추가해보기

df.head()
#Cabin Initial imputation
df["Has_Cabin"] = df["Cabin"].apply(lambda x: 0 if type(x)==float else 1)
df["Cabin_Initial"] = df["Cabin"].apply(lambda x: x[0] if pd.notnull(x) else x)
df["Cabin_number"] = df["Cabin"].apply(lambda x: x[-1] if pd.notnull(x) else x)
df = df.drop("Cabin", axis = 1)
#Imputation of Cabin initial
train_cabin = df[df["Cabin_Initial"].notnull()]
test_cabin = df[df["Cabin_Initial"].isnull()]

tr = train_cabin.drop(["Cabin_Initial", "Cabin_number"], axis = 1)
la = train_cabin["Cabin_Initial"]
te = test_cabin.drop(["Cabin_Initial", "Cabin_number"], axis = 1)
clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                        ('knn',neighbors.KNeighborsClassifier()),
                        ('rfor',RandomForestClassifier()),
                        ('lr',LogisticRegression()),
                        ('LDA',LinearDiscriminantAnalysis()),
                        ('DC',DecisionTreeClassifier()),
                        ])


clf.fit(tr, la)
predicted_initial = clf.predict(te)
te["predicted_inital"] = predicted_initial

for i in range(0, len(df)):
    if type(df["Cabin_Initial"].iloc[i])==float:
        df.ix[i, "Cabin_Initial"] = te["predicted_inital"].ix[i]

df["Cabin_Initial"] = lbl.fit_transform(df[col].astype(str))
df = df.drop(["Cabin_number","PassengerId"], axis = 1)

colormap = plt.cm.RdBu
plt.figure(figsize = (14, 12))
plt.title("Correlation- Pearson")
sns.heatmap(df.corr(), square = True, cmap = colormap, annot=True)
ntrain = length
ntest = len(df)-length

y_train = label.ravel()
x_train  = df[:length].values
x_test = df[length:].values

SEED = 0
NFOLDS = 5
kf = KFold(n_splits=NFOLDS)
#내가 참고한 커널은 각 모델을 더 용이하게 사용하기 위해 클래스를 정의하였다
class SklearnHelper(object):
    def __init__(self, clf, seed = 0, params=None):
        params["random_state"] = seed
        self.clf = clf(**params)
        
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
    
    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self, x, y):
        return self.clf.fit(x, y)
        
    def feature_importances(self, x, y):
        importance = []
        for i in self.clf.fit(x, y).feature_importances_:
            importance.append(i)
        return importance
        
def get_oof(clf, X, y, X_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print('\nFold {}'.format(i))
        x_tr = X[train_index]
        y_tr = y[train_index]
        x_te = X[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(X_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

#Out of Fold 를 가지고 KFold로 Predict한 뒤 그것들의 평균을 통해 OOF Predicion을 산출함
#Parameter 설정
#Random Forest
rf_params = {
    "n_jobs": -1,
    "n_estimators": 500,
    "warm_start": True,
    #"max_features":0.2,
    "max_depth":6,
    "min_samples_leaf": 2, 
    "max_features": "sqrt",
    "verbose":0
}

#Extra Trees
et_params = {
    "n_jobs": -1,
    "n_estimators": 500,
    #"max_features":0.5,
    "max_depth":8,
    "min_samples_leaf": 2, 
    "verbose":0
}

#AdaBoost
ada_params = {
    "n_estimators" : 500,
    "learning_rate" : 0.75
}

#Gradient Boosting
gb_params = {
    "n_estimators":500,
    #"max_features" : 0.2
    "max_depth" : 5,
    "min_samples_leaf" : 2,
    "verbose" : 0
}

#SVC -Support Vector Classifier

svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }
rf = SklearnHelper(clf = RandomForestClassifier, seed = SEED, params = rf_params)
et = SklearnHelper(clf = ExtraTreesClassifier, seed = SEED, params = et_params)
ada = SklearnHelper(clf = AdaBoostClassifier, seed = SEED, params = ada_params)
gb = SklearnHelper(clf = GradientBoostingClassifier, seed = SEED, params = gb_params)

svc = SklearnHelper(clf = SVC, seed = SEED, params = svc_params)
#First Level Prediction - OOF train and test
print ("Generating OOFs")

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")
rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)
cols = df[:length].columns.values
feature_df = pd.DataFrame({
    "features":cols,
    'Random Forest feature importances': rf_feature,
     'Extra Trees  feature importances': et_feature,
      'AdaBoost feature importances': ada_feature,
    'Gradient Boost feature importances': gb_feature
})
# Scatter plot 
trace = go.Scatter(
    y = feature_df['Random Forest feature importances'].values,
    x = feature_df['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_df['Random Forest feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_df['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Random Forest Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_df['Extra Trees  feature importances'].values,
    x = feature_df['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_df['Extra Trees  feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_df['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Extra Trees Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_df['AdaBoost feature importances'].values,
    x = feature_df['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_df['AdaBoost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_df['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'AdaBoost Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')

# Scatter plot 
trace = go.Scatter(
    y = feature_df['Gradient Boost feature importances'].values,
    x = feature_df['features'].values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1,
        size = 25,
#       size= feature_dataframe['AdaBoost feature importances'].values,
        #color = np.random.randn(500), #set color equal to a variable
        color = feature_df['Gradient Boost feature importances'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = feature_df['features'].values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Gradient Boosting Feature Importance',
    hovermode= 'closest',
#     xaxis= dict(
#         title= 'Pop',
#         ticklen= 5,
#         zeroline= False,
#         gridwidth= 2,
#     ),
    yaxis=dict(
        title= 'Feature Importance',
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')
feature_df["mean"] = feature_df.mean(axis = 1)
feature_df.head()

data =[
    go.Bar(
    x =feature_df["features"].values,
    y = feature_df["mean"].values
    )
]

layout = go.Layout(
    title = "Feature Importance-Mean",
    yaxis =dict(
        title = "Importance", 
    )
)

fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = "BAR")
base_predictions_train = pd.DataFrame( {'RandomForest': rf_oof_train[:,0],
     'ExtraTrees': et_oof_train[:,0],
     'AdaBoost': ada_oof_train[:,0],
      'GradientBoost': gb_oof_train[:,0]
    })
base_predictions_train.head(10)

df.head()
#Train에 ADD하기
x_train = np.concatenate((x_train, et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis = 1)
x_test = np.concatenate((x_test, et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
from sklearn.model_selection import train_test_split
xtrain, xvalid, ytrain, yvalid = train_test_split(
      x_train, y_train, test_size=0.30, random_state=5)
#VotingClasifier
from sklearn.linear_model import LogisticRegression
from subprocess import check_output
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                        ('knn',neighbors.KNeighborsClassifier()),
                        ('rfor',RandomForestClassifier()),
                        ('lr',LogisticRegression()),
                        ('LDA',LinearDiscriminantAnalysis()),
                        ('DC',DecisionTreeClassifier()),
                        ('GB',GradientBoostingClassifier()),
                        ('XGB',XGBClassifier()),
                        ('Ada', AdaBoostClassifier()),
                        ('GNB', GaussianNB())
                        ])
clf.fit(xtrain, ytrain)
confidence = clf.score(xvalid, yvalid)
print('Confidence: ',confidence)

predictions = clf.predict(x_test)
submission = pd.DataFrame({'PassengerId': df_test['PassengerId'],
                    'Survived': predictions})
submission.to_csv('Ensemble_with_OOF_190131_ver7.csv', index=False)