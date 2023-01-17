# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as ml

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import plotly.figure_factory as ff

import matplotlib.pyplot as plt

%matplotlib inline

ml.style.use('ggplot')



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix

from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
td = pd.read_csv('/kaggle/input/titanic/train.csv')

print(td.shape)

td.head(10)
td.describe()
td.columns
td.info()
td.hist(figsize=(20,10), color='maroon', bins=25)

plt.show()
plt.figure(figsize=(20,10))

sns.distplot(td[td.Survived==1]['Age'])

sns.distplot(td[td.Survived==0]['Age'])

plt.legend(['SURVIVED','DID NOT SURVIVE'])

plt.show()
fig = go.Figure(data=[

    go.Bar(name='SURVIVED', x=list(td.Sex.value_counts().index), y=td[td.Survived==1]['Sex'].value_counts().values),

    go.Bar(name='DID NOT SURVIVE', x=list(td.Sex.value_counts().index), y=td[td.Survived==0]['Sex'].value_counts().values)

])

fig.update_layout(barmode='group',title="SEX")

fig.show()



fig = go.Figure(data=[go.Pie(labels=['MALES','FEMALES'],

                             values=[td[(td.Sex=='male') & (td.Survived==1)].shape[0],td[(td.Sex=='female') & (td.Survived==1)].shape[0]])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=['yellow','purple'],line=dict(color='#000000', width=2)))

fig.show()
fig = go.Figure(data=[

    go.Bar(name='SURVIVED', x=list(td.Pclass.value_counts().index), y=td[td.Survived==1]['Pclass'].value_counts().values),

    go.Bar(name='DID NOT SURVIVE', x=list(td.Pclass.value_counts().index), y=td[td.Survived==0]['Pclass'].value_counts().values)

])

fig.update_layout(barmode='group',title="PCLASS")

fig.show()



fig = go.Figure(data=[go.Pie(labels=['1st CLASS','2nd CLASS','3rd CLASS'],

                             values=[td[(td.Pclass==1) & (td.Survived==1)].shape[0],td[(td.Pclass==2) & (td.Survived==1)].shape[0],td[(td.Pclass==3) & (td.Survived==1)].shape[0]])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=['yellow','lightgreen','darkorange'],line=dict(color='#000000', width=2)))

fig.show()
fig = go.Figure(data=[

    go.Bar(name='SURVIVED', x=list(td.Embarked.value_counts().index), y=td[td.Survived==1]['Embarked'].value_counts().values),

    go.Bar(name='DID NOT SURVIVE', x=list(td.Embarked.value_counts().index), y=td[td.Survived==0]['Embarked'].value_counts().values)

])

fig.update_layout(barmode='group',title="EMBARKED")

fig.show()



fig = go.Figure(data=[go.Pie(labels=['S','C','Q'],

                             values=[td[(td.Embarked=='S') & (td.Survived==1)].shape[0],

                                     td[(td.Embarked=='C') & (td.Survived==1)].shape[0],

                                     td[(td.Embarked=='Q') & (td.Survived==1)].shape[0]])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=['maroon','pink','darkturquoise'],line=dict(color='#000000', width=2)))

fig.show()
plt.figure(figsize=(20,10))

sns.heatmap(td.corr(),annot=True,linewidth=1,linecolor='white')

plt.show()
td.drop(columns=['Fare'],inplace=True)

td.head()
print(td.isnull().sum())

print("\nFraction of values that are missing in the 'Cabin' feature : ", (td.Cabin.isnull().sum()/td.shape[0]))
td.drop(columns=['Cabin'],inplace=True)

td.head()
td1 = td.copy()

td2 = td.copy()
td1.Age.fillna(np.mean(td.Age),inplace=True)

td1.Age.isnull().sum()
print(td2.groupby('Pclass')['Age'].mean())

mean_list = list(td2.groupby('Pclass')['Age'].mean().values)

print("\nList of means of Ages grouped according to Pclass",mean_list)
# Replacing by looping through Pclass values. Total number of unique values of Pclass = 3. So the loop runs 3 times.

for i in range(3):

    td2.loc[td2['Pclass']==i+1,'Age'] = td2.loc[td2['Pclass']==i+1,'Age'].fillna(mean_list[i])

print(td2.Age.isnull().sum())

td = td2
td.isnull().sum()
plt.figure(figsize=(20,10))

sns.distplot(td['Age'])

plt.show()
print("Most popular type : ", td.Embarked.value_counts().sort_values(ascending=False).index[0])

to_replace = td.Embarked.value_counts().sort_values(ascending=False).index[0]

sns.countplot(x='Embarked',data=td)

plt.show()
td.Embarked.fillna(to_replace,inplace=True)

td.isnull().sum()
td['Fam'] = td['SibSp'] + td['Parch']

td.drop(columns=['SibSp','Parch'],inplace=True)

td.head(20)
plt.figure(figsize=(20,10))

sns.distplot(td['Fam'])

plt.title("DISTRIBUTION OF FAMILY")

plt.show()



plt.figure(figsize=(20,10))

sns.countplot(x='Fam',data=td,hue='Pclass')

plt.title("NO. OF FAMILY MEMBERS VS PCLASS")

plt.show()
td.drop(columns=['PassengerId','Ticket'],inplace=True)

td.head(10)
td['Title'] = td['Name']



# Apply regex per name

# Use function : Series.str.extract()

for name in td['Name']:

    td['Title'] = td['Name'].str.extract('([A-Za-z]+)\.',expand=True)    # Regex to get title : ([A-Za-z]+)\.



# Drop Name

td.drop(columns=['Name'],inplace=True)

td.head()
# Check extracted data for quality

td.Title.unique()
title_mapping = {'Don':'Rare','Rev':'Rare','Mme':'Miss','Ms':'Miss','Major':'Rare','Lady':'Royal','Sir':'Royal','Mlle':'Miss','Col':'Rare','Capt':'Rare','Countess':'Royal','Jonkheer':'Royal'}



td.replace({'Title':title_mapping},inplace=True)

td.Title.unique()
td['Pclass_new']=np.nan

rep_list = ['first','second','third']



# Decode manually for all 3 columns

for i in range(3):

    td.loc[td['Pclass']==i+1,'Pclass_new'] = rep_list[i]

    

# Drop Pclass

td.drop(columns=['Pclass'],inplace=True)

td.head()
td.isnull().sum()
# Use pd.get_dummies(data,drop_first)

encd_col = ['Pclass_new','Sex','Embarked','Title']



ohe_features = pd.get_dummies(data=td.loc[:,encd_col],drop_first=True)   # In OHE we usually create k-1 encoded features for k classes.

# Drop original columns

td.drop(columns=encd_col,inplace=True)

td = td.join(ohe_features)

td.head(10)
plt.figure(figsize=(20,10))

sns.heatmap(td.corr(),annot=True)

plt.show()
tstd = pd.read_csv('/kaggle/input/titanic/test.csv')

tstd.head()
tstd.drop(columns=['Ticket','Fare','Cabin'],inplace=True)              # We don't drop PassengerID because we need it for creating o/p file

tstd.isnull().sum()
age_lst = list(tstd.groupby('Pclass')['Age'].mean().values)

for i in range(3):

    tstd.loc[tstd['Pclass']==i+1,'Age'] = tstd.loc[tstd['Pclass']==i+1,'Age'].fillna(age_lst[i])

tstd.isnull().sum()
tstd['Title'] = tstd['Name']

for i in tstd['Name']:

    tstd['Title'] = tstd['Name'].str.extract('([A-Za-z]+)\.',expand=True)

# Dropping Name

tstd.drop(columns=['Name'],inplace=True)

# Replacing by mapping

title_mapping = {'Don':'Rare','Rev':'Rare','Mme':'Miss','Ms':'Miss','Major':'Rare','Dona':'Royal','Mlle':'Miss','Col':'Rare','Capt':'Rare'}



tstd.replace({'Title':title_mapping},inplace=True)

print(tstd.Title.unique())

tstd.head()
tstd['Pclass_new'] = np.nan

new_pc = ['first','second','third']

for i in range(3):

    tstd.loc[tstd.Pclass==i+1,'Pclass_new'] = new_pc[i]

tstd.drop(columns=['Pclass'],inplace=True)

tstd.head()
tstd['Fam'] = tstd['SibSp'] + tstd['Parch']

tstd.drop(columns=['SibSp','Parch'],inplace=True)

tstd.head()
# Use pd.get_dummies(data,drop_first)

encd_col1 = ['Pclass_new','Sex','Embarked','Title']



ohe_features2 = pd.get_dummies(data=tstd.loc[:,encd_col1],drop_first=True)   # In OHE we usually create k-1 encoded features for k classes.

# Drop original columns

tstd.drop(columns=encd_col1,inplace=True)

tstd = tstd.join(ohe_features2)

tstd.head(10)
td.head()
tstd.head()
# Split the training data by the conventional 80-20 split

X = td.drop(columns=['Survived'])

Y = td['Survived'].values

trainx, testx, trainy, testy = train_test_split(X,Y,test_size=0.2)

x,y = np.array(td.iloc[:,1:].values),np.array(td.iloc[:,0].values)

test = np.array(tstd.iloc[:,:].values)

print("Train : ",trainx.shape,trainy.shape)

print("Test : ",testx.shape,testy.shape)



# Creating the model

logr = LogisticRegression(penalty='l2',C=1.0,solver='lbfgs')

logr.fit(trainx,trainy)



# Preds and accuracy

y_pred1 = logr.predict_proba(testx)

# We are interested in the True and False Positives only.

fptp = y_pred1[:,1]  # As 2nd value tells the probability of getting a 1



# Getting the ROC-AUC score and plotting the ROC curve

logr_score = roc_auc_score(testy,fptp)

print("ROC AUC score = ",logr_score)

lr_fp,lr_tp,_ = roc_curve(testy,fptp)   # Returns FPR, TPR and thresholds.

plt.figure(figsize=(20,10))

plt.plot(lr_fp,lr_tp,marker='.',label="Logistic Regression ROC Curve")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
y_pred2 = logr.predict(testx)

tn,fp,fn,tp = confusion_matrix(testy,y_pred2).ravel()

acc1 = (tp+tn)/(tp+tn+fp+fn)

print(acc1)
op1 = logr.predict(tstd.drop(columns=['PassengerId'],axis=1))

opf_df1 = pd.DataFrame({'PassengerId': tstd.PassengerId, 'Survived': op1})

opf_df1.to_csv('Balaka_LGR.csv', index=False)
knn = KNeighborsClassifier()

scaler = StandardScaler()

trainx_scaled = scaler.fit_transform(trainx)



# Hyperparameter tuning

param_grid = {

    'n_neighbors' : [3,5,7,9],

    'weights' : ['uniform','distance'],

    'metric' : ['euclidean','manhattan','minkowski'],

    'algorithm' : ['auto','ball_tree','kd_tree','brute']

}

knn_gs = GridSearchCV(estimator=knn,param_grid=param_grid,cv=10)

knn_gs.fit(trainx_scaled,trainy)

print(knn_gs.best_score_)

print(knn_gs.best_params_)
# Defining the knn classifier

knn_best = KNeighborsClassifier(n_neighbors=knn_gs.best_params_.get('n_neighbors'),weights=knn_gs.best_params_.get('weights'),algorithm=knn_gs.best_params_.get('algorithm'),metric=knn_gs.best_params_.get('metric'))

knn_best.fit(trainx,trainy)



# ROC-AUC score

y_pred3 = knn_best.predict_proba(testx)

# We are interested in the True and False Positives only.

fptp2 = y_pred3[:,1]  # As 2nd value tells the probability of getting a 1



# Getting the ROC-AUC score and plotting the ROC curve

knn_score = roc_auc_score(testy,fptp2)

print("ROC AUC score = ",knn_score)

lr_fp2,lr_tp2,_ = roc_curve(testy,fptp2)   # Returns FPR, TPR and thresholds.

plt.figure(figsize=(20,10))

plt.plot(lr_fp2,lr_tp2,marker='.',label="KNN ROC Curve")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
y_pred4 = knn_best.predict(testx)

tn,fp,fn,tp = confusion_matrix(testy,y_pred4).ravel()

acc2 = (tp+tn)/(tp+tn+fp+fn)

print(acc2)
op2 = knn_best.predict(tstd.drop(columns=['PassengerId'],axis=1))

opf_df2 = pd.DataFrame({'PassengerId': tstd.PassengerId, 'Survived': op2})

opf_df2.to_csv('Balaka_KNN.csv', index=False)
rf = RandomForestClassifier()



# Hyperparameter tuning

param_grid = {

    'n_estimators' : [80,90,100],

    'criterion' : ['gini','entropy'],

    'max_depth' : [5,6,7,9],

    'max_features' : ['auto','sqrt','log2']

}

rf_gs = GridSearchCV(estimator=rf,param_grid=param_grid,cv=10)

rf_gs.fit(trainx,trainy)

print(rf_gs.best_score_)

print(rf_gs.best_params_)
# Defining the rf classifier

rf_best = RandomForestClassifier(n_estimators=rf_gs.best_params_.get('n_estimators'),criterion=rf_gs.best_params_.get('criterion'),max_depth=rf_gs.best_params_.get('max_depth'),max_features=rf_gs.best_params_.get('max_features'))

rf_best.fit(trainx,trainy)



# ROC-AUC score

y_pred5 = rf_best.predict_proba(testx)

# We are interested in the True and False Positives only.

fptp3 = y_pred5[:,1]  # As 2nd value tells the probability of getting a 1



# Getting the ROC-AUC score and plotting the ROC curve

rf_score = roc_auc_score(testy,fptp3)

print("ROC AUC score = ",rf_score)

lr_fp3,lr_tp3,_ = roc_curve(testy,fptp3)   # Returns FPR, TPR and thresholds.

plt.figure(figsize=(20,10))

plt.plot(lr_fp3,lr_tp3,marker='.',label="RF ROC Curve")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
op3 = rf_best.predict(tstd.drop(columns=['PassengerId'],axis=1))

opf_df3 = pd.DataFrame({'PassengerId': tstd.PassengerId, 'Survived': op3})

opf_df3.to_csv('Balaka_RF2.csv', index=False)
adb = AdaBoostClassifier()



# Hyperparameter tuning

param_grid = {

    'n_estimators' : [20,30,40,50],

    'algorithm' : ['SAMME', 'SAMME.R']

}

adb_gs = GridSearchCV(estimator=adb,param_grid=param_grid,cv=10)

adb_gs.fit(trainx,trainy)

print(adb_gs.best_score_)

print(adb_gs.best_params_)
# Defining the adb classifier

adb_best = AdaBoostClassifier(n_estimators=adb_gs.best_params_.get('n_estimators'),algorithm=adb_gs.best_params_.get('algorithm'))

adb_best.fit(trainx,trainy)



# ROC-AUC score

y_pred7 = adb_best.predict_proba(testx)

# We are interested in the True and False Positives only.

fptp4 = y_pred7[:,1]  # As 2nd value tells the probability of getting a 1



# Getting the ROC-AUC score and plotting the ROC curve

adb_score = roc_auc_score(testy,fptp4)

print("ROC AUC score = ",adb_score)

lr_fp4,lr_tp4,_ = roc_curve(testy,fptp4)   # Returns FPR, TPR and thresholds.

plt.figure(figsize=(20,10))

plt.plot(lr_fp4,lr_tp4,marker='.',label="AdaBoost ROC Curve")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
y_pred8 = adb_best.predict(testx)

tn,fp,fn,tp = confusion_matrix(testy,y_pred8).ravel()

acc4 = (tp+tn)/(tp+tn+fp+fn)

print(acc4)
op4 = adb_best.predict(tstd.drop(columns=['PassengerId'],axis=1))

opf_df4 = pd.DataFrame({'PassengerId': tstd.PassengerId, 'Survived': op4})

opf_df4.to_csv('Balaka_ADBoost.csv', index=False)
names = ['Logistic Regression','KNN','Random Forest','Adaboost']

vals = [acc1,knn_gs.best_score_,rf_gs.best_score_,adb_gs.best_score_]

res_df = pd.DataFrame({'Algorithm': names,'Accuracy': vals})

res_df
from xgboost import XGBClassifier

xgb = XGBClassifier()

param_grid = { 

    'learning_rate' : [0.1, 0.2],

    'max_depth': [3, 5, 7],   

}



xgb_gs = GridSearchCV(estimator = xgb,param_grid=param_grid,cv=3)

xgb_gs.fit(trainx,trainy)

print(xgb_gs.best_score_)

print(xgb_gs.best_params_)
xgb_best = XGBClassifier(learning_rate=xgb_gs.best_params_.get('learning_rate'),max_depth=xgb_gs.best_params_.get('max_depth'))

xgb_best.fit(trainx,trainy)



# ROC-AUC score

y_pred11 = xgb_best.predict_proba(testx)

# We are interested in the True and False Positives only.

fptp6 = y_pred11[:,1]  # As 2nd value tells the probability of getting a 1



# Getting the ROC-AUC score and plotting the ROC curve

xgb_score = roc_auc_score(testy,fptp6)

print("ROC AUC score = ",xgb_score)

lr_fp6,lr_tp6,_ = roc_curve(testy,fptp6)   # Returns FPR, TPR and thresholds.

plt.figure(figsize=(20,10))

plt.plot(lr_fp6,lr_tp6,marker='.',label="XGBoost ROC Curve")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
y_pred9 = xgb_best.predict(testx)

tn,fp,fn,tp = confusion_matrix(testy,y_pred9).ravel()

acc5 = (tp+tn)/(tp+tn+fp+fn)

print(acc5)
op5 = xgb_best.predict(tstd.drop(columns=['PassengerId'],axis=1))

opf_df5 = pd.DataFrame({'PassengerId': tstd.PassengerId, 'Survived': op5})

opf_df5.to_csv('Balaka_XGBoost2.csv', index=False)