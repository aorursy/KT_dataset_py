

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder

import warnings

warnings.filterwarnings("ignore")

df_train=pd.read_csv('../input/airplane-accident-dataset/train.csv')

df_test=pd.read_csv('../input/airplane-accident-dataset/test.csv')

df_train.head()
obj=LabelEncoder()

df_train['target']=obj.fit_transform(df_train['Severity'])

df=pd.concat([df_train.drop(['Severity','target'],axis=1),df_test],axis=0,sort=False)

df.head()
df['Total_Safety_Complaints']=pd.qcut(df['Total_Safety_Complaints'],3)

df['Cabin_Temperature']=pd.qcut(df['Cabin_Temperature'],3)

df['Violations']=df['Violations'].map({2:0,1:1,3:2,0:4,4:4,5:5})



df['Adverse_Weather_Metric']=pd.qcut(df['Adverse_Weather_Metric'],3)



df['Max_Elevation']=pd.qcut(df['Max_Elevation'],3)



df['Turbulence_In_gforces']=pd.qcut(df['Turbulence_In_gforces'],3)
from sklearn.preprocessing import LabelEncoder

lbl=LabelEncoder()



df['Total_Safety_Complaints']=lbl.fit_transform(df['Total_Safety_Complaints'])

df['Cabin_Temperature']=lbl.fit_transform(df['Cabin_Temperature'])

df['Max_Elevation']=lbl.fit_transform(df['Max_Elevation'])

df['Turbulence_In_gforces']=lbl.fit_transform(df['Turbulence_In_gforces'])



df['Adverse_Weather_Metric']=lbl.fit_transform(df['Adverse_Weather_Metric'])
col1=['Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints',

       'Accident_Type_Code','Control_Metric' ]

col3=['Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints',

       'Accident_Type_Code','Control_Metric','Turbulence_In_gforces', 'Cabin_Temperature',

        'Violations',

       'Adverse_Weather_Metric','Max_Elevation']

obj=StandardScaler()

df[col1]=obj.fit_transform(df[col1])

obj1=MinMaxScaler()

df[col3]=obj1.fit_transform(df[col3])
column=['Safety_Score', 'Days_Since_Inspection', 'Total_Safety_Complaints',

       'Accident_Type_Code','Control_Metric','Turbulence_In_gforces', 'Cabin_Temperature','Violations','Adverse_Weather_Metric',

       'Max_Elevation']
X=df.iloc[0:10000,:][column]

y=df_train['target']

x=df.iloc[10000:12500,:][column]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2020,test_size=0.25)

import lightgbm as lgb

from sklearn.model_selection import cross_val_score

model = lgb.LGBMClassifier( learning_rate=0.2, n_estimators= 1000)

result=cross_val_score(estimator=model,X=X_train,y=y_train,cv=10)

print(result)

print(result.mean())
X=df.iloc[0:10000,:][column]

y=df_train['target']

x=df.iloc[10000:12500,:][column]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2020,test_size=0.25)

import xgboost as xgb

from sklearn.model_selection import cross_val_score

model1=xgb.XGBClassifier(colsample_bylevel= 1, learning_rate= 0.1,max_depth=10, n_estimators= 1000)

result=cross_val_score(estimator=model1,X=X_train,y=y_train,cv=10)

print(result)

print(result.mean())
model.fit(X,y)

id=df_test['Accident_ID']

y_pred=model.predict(x)

submission=pd.DataFrame({'Accident_ID':id,'Severity':y_pred})

submission.head()

submission['Severity']=submission['Severity'].map({1:'Minor_Damage_And_Injuries',2:'Significant_Damage_And_Fatalities',3:'Significant_Damage_And_Serious_Injuries',0:'Highly_Fatal_And_Damaging'})

#submission.to_csv('submission.csv',index=False)
model1.fit(X,y)

id=df_test['Accident_ID']

y_pred1=model1.predict(x)

submission1=pd.DataFrame({'Accident_ID':id,'Severity':y_pred1})

submission1.head()

submission1['Severity']=submission1['Severity'].map({1:'Minor_Damage_And_Injuries',2:'Significant_Damage_And_Fatalities',3:'Significant_Damage_And_Serious_Injuries',0:'Highly_Fatal_And_Damaging'})

#submission1.to_csv('F:\\PYTHON PROGRAM\\JAISHREERAMhacker75.csv',index=False)
indices=np.argsort(model1.feature_importances_)

plt.figure(figsize=(10,10))

g = sns.barplot(y=X_train.columns[indices][:40],x = model1.feature_importances_[indices][:40] , orient='h')
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)


GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [400,500],

              'learning_rate': [0.1, 0.2],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(X_train,y_train)



model2 = gsGBC.best_estimator_



# Best score

print(gsGBC.best_score_)

print(gsGBC.best_params_)
model2 = gsGBC.best_estimator_
model2.fit(X,y)

id=df_test['Accident_ID']

y_pred2=model2.predict(x)

submission2=pd.DataFrame({'Accident_ID':id,'Severity':y_pred2})

submission2.head()

submission2['Severity']=submission2['Severity'].map({1:'Minor_Damage_And_Injuries',2:'Significant_Damage_And_Fatalities',3:'Significant_Damage_And_Serious_Injuries',0:'Highly_Fatal_And_Damaging'})

#submission2.to_csv('F:\\PYTHON PROGRAM\\JAISHREERAMhacker36.csv',index=False)
# RFC Parameters tunning 

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

            

              "n_estimators" :[400,500,1000],

              "criterion": ["gini"]}





gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(X_train,y_train)



model3 = gsRFC.best_estimator_



# Best score

print(gsRFC.best_score_)

print(gsRFC.best_params_)
model3.fit(X,y)

id=df_test['Accident_ID']

y_pred3=model3.predict(x)

submission3=pd.DataFrame({'Accident_ID':id,'Severity':y_pred3})

submission3.head()

submission3['Severity']=submission3['Severity'].map({1:'Minor_Damage_And_Injuries',2:'Significant_Damage_And_Fatalities',3:'Significant_Damage_And_Serious_Injuries',0:'Highly_Fatal_And_Damaging'})

#submission3.to_csv('F:\\PYTHON PROGRAM\\JAISHREERAMhacker57.csv',index=False)
#ExtraTrees 

ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 10],

              "min_samples_split": [2, 3, 10],

              "min_samples_leaf": [1, 3, 10],

              "bootstrap": [False],

              "n_estimators" :[400,500],

              "criterion": ["gini"]}





gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsExtC.fit(X_train,y_train)



model5 = gsExtC.best_estimator_



# Best score

print(gsExtC.best_score_)

print(gsExtC.best_params_)
model5.fit(X,y)

id=df_test['Accident_ID']

y_pred5=model5.predict(x)

submission5=pd.DataFrame({'Accident_ID':id,'Severity':y_pred5})

submission5.head()

submission5['Severity']=submission5['Severity'].map({1:'Minor_Damage_And_Injuries',2:'Significant_Damage_And_Fatalities',3:'Significant_Damage_And_Serious_Injuries',0:'Highly_Fatal_And_Damaging'})

#submission5.to_csv('F:\\PYTHON PROGRAM\\JAISHREERAMhacker36.csv',index=False)
model = lgb.LGBMClassifier( learning_rate=0.2, n_estimators= 500,max_depth=10)

model1=xgb.XGBClassifier(colsample_bylevel= 1, learning_rate= 0.1, max_depth= 10, n_estimators= 400)

model2 = GradientBoostingClassifier(learning_rate= 0.2, loss= 'deviance', max_depth= 8, max_features =0.3, min_samples_leaf= 100, n_estimators= 500)

model3 = RandomForestClassifier(criterion= 'gini', min_samples_leaf= 1, min_samples_split= 3, n_estimators= 500)
from sklearn.ensemble import VotingClassifier

votingC = VotingClassifier(estimators=[('gbc',model2),('rfc',model3),('xgb',model1)], voting='soft', n_jobs=4)

votingC.fit(X,y)

id=df_test['Accident_ID']

y_pred2=votingC.predict(x)

submission=pd.DataFrame({'Accident_ID':id,'Severity':y_pred2})

submission.head()

submission['Severity']=submission['Severity'].map({1:'Minor_Damage_And_Injuries',2:'Significant_Damage_And_Fatalities',3:'Significant_Damage_And_Serious_Injuries',0:'Highly_Fatal_And_Damaging'})

#submission.to_csv('F:\\PYTHON PROGRAM\\JAISHREERAMhacker72.csv',index=False)