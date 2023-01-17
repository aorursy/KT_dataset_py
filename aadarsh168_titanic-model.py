import numpy as np

import pandas as pd

import seaborn as sns

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')

target='Survived'
data = pd.concat([train,test],axis=0)
data.head()
data[['Name','SibSp','Parch']].isna().sum()
data['Name_title']=data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split('.')[0])

data['Family_size']=data['SibSp']+data['Parch']+1

data['Solo']=0

data.loc[data['Family_size']==1,'Solo']=1
data.describe(include='all')
data.nunique()
# data= pd.get_dummies(data,columns=['Pclass','Sex','SibSp','Parch'])
data.isna().sum()
plt.figure(figsize=(12,12))

sns.heatmap(data.corr())
data['Embarked'].isna().sum()
emb_train = data.loc[data['Embarked'].notna(),['Pclass','Fare','Parch','SibSp','Embarked']]

emb_missed= data.loc[data['Embarked'].isna(),['Pclass','Fare','Parch','SibSp']]

emb_train=emb_train.dropna()

emb_train=pd.get_dummies(emb_train,columns=['Pclass','Parch','SibSp'])

emb_missed=pd.get_dummies(emb_missed,columns=['Pclass','Parch','SibSp'])
emb_x=emb_train.columns[emb_train.columns != 'Embarked'].tolist()

emb_y='Embarked'

emb_train1, emb_val, train1_embarked,val_embarked = train_test_split(emb_train[emb_x],emb_train[emb_y],stratify=emb_train[emb_y], test_size=0.15, random_state=122)
from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 3).fit(emb_train1, train1_embarked)

print('Accuracy: {:.0f}%'.format(knn.score(emb_val,val_embarked)*100))

del(emb_train1, emb_val, train1_embarked,val_embarked)
emb_model = KNeighborsClassifier(n_neighbors=3)

emb_model.fit(emb_train[emb_x],emb_train[emb_y])

data_emb=data.copy()

data_emb=pd.get_dummies(data_emb,columns=['Pclass','Parch','SibSp'])

data.loc[data['Embarked'].isna(),'Embarked']=emb_model.predict(data_emb.loc[data_emb['Embarked'].isna(),emb_x])

del(data_emb,emb_x,emb_y)
data['Embarked'].isna().sum()
data['Fare'].isna().sum()
#Function to get the closest value from the dataset for the passed value

def get_closest(data,value):

    a=data.unique().tolist()

    min_idx=min(range(len(a)),key=lambda i: abs(a[i]-value))

    return a[min_idx]
data[data[target].isna()]['Fare']=data[data[target].isna()]['Fare'].apply(lambda x: get_closest(data[data[target].notna()]['Fare'],x))
data.groupby(['Embarked','Pclass'])['Fare'].mean()
#Getting the Pclass and Embarked of the missing value of Fare

data.loc[data['Fare'].isna(),['Pclass','Embarked']]
#Since the mean of Pclass=3 and Embarked=S is 14.44

Fare=data.loc[(data[target].isna())&(data['Pclass']==3)&(data['Embarked']=='S'),'Fare'].unique().tolist()

data.loc[data['Fare'].isna(),'Fare']=Fare[min(range(len(Fare)),key=lambda i: abs(Fare[i]-14.435422))]
data['Fare'].isna().sum()
data['Age'].isna().sum()
data.loc[data['Age'].notna(),'Name_title'].value_counts()
data.loc[data['Age'].isna(),'Name_title'].value_counts()
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt

age_df=data[data['Age'].notna()][['Name_title','Solo','Age']]

age_df=pd.get_dummies(age_df,columns=['Name_title'],drop_first=True)

age_feat=age_df.columns[age_df.columns != 'Age'].tolist()

age_train,age_val,age_train_lbl,age_val_lbl=train_test_split(age_df[age_feat],age_df['Age'], test_size = 0.3,random_state=10)

    

from sklearn.model_selection import GridSearchCV

params = {'n_neighbors':range(1,20)}



knn = KNeighborsRegressor()



model = GridSearchCV(knn, params, cv=25)

model.fit(age_train,age_train_lbl)

print(model.best_params_)



del(age_train,age_val,age_train_lbl,age_val_lbl,model)
age_model = KNeighborsRegressor(n_neighbors=18)

age_model.fit(age_df[age_feat],age_df['Age'])

data_age=data.copy()

data_age=pd.get_dummies(data_age,columns=['Name_title'])

data.loc[data['Age'].isna(),'Age']=age_model.predict(data_age.loc[data_age['Age'].isna(),age_feat])

del(age_df,data_age,age_feat,age_model)
data['Age'].isna().sum()
data['Age_bins']=pd.qcut(data['Age'],5)

data['Fare_bins']=pd.qcut(data['Fare'],4)
data=data.drop(['PassengerId','Cabin','Name','SibSp','Parch','Ticket','Age','Fare'],axis=1)
data['Sex'] = data['Sex'].map({'male':0,'female':1})

data.head()
data=pd.get_dummies(data,columns=['Pclass','Name_title','Embarked','Age_bins','Fare_bins'],drop_first=True)

features=data.columns.tolist()

features.remove(target)

data_train=data.loc[data[target].notna()]

data_test=data.loc[data[target].isna()]
x_train=data_train[features]

y_train=data_train[target].astype(int)

x_test=data_test[features]
y_train = np.array(y_train)
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(x_train)

x_scaled_train=scaler.transform(x_train)

x_scaled_train
x_scaled_test=scaler.transform(x_test)

x_scaled_test
param_test1 = {'max_depth':range(1,10),

               'min_child_weight':range(1,6)

              }

gsearch1 = GridSearchCV(

    estimator = XGBClassifier(learning_rate =0.05,

                              n_estimators=500,

                              max_depth=5,

                              min_child_weight=1,

                              gamma=0,

                              subsample=0.8,

                              colsample_bytree=0.8,

                              objective= 'binary:logistic',

                              nthread=4,

                              scale_pos_weight=1,

                              seed=27),

    param_grid = param_test1,

    scoring='roc_auc',

    n_jobs=4,

    cv=5)

gsearch1.fit(x_scaled_train,y_train)

gsearch1.best_params_, gsearch1.best_score_
param_test2 = {'gamma':[i/10.0 for i in range(0,5)]

              }

gsearch2 = GridSearchCV(

    estimator = XGBClassifier(learning_rate =0.05,

                              n_estimators=500,

                              max_depth=2,

                              min_child_weight=2,

                              gamma=0,

                              subsample=0.8,

                              colsample_bytree=0.8,

                              objective= 'binary:logistic',

                              nthread=4,

                              scale_pos_weight=1,

                              seed=27),

    param_grid = param_test2,

    scoring='roc_auc',

    n_jobs=4,

    cv=5)

gsearch2.fit(x_scaled_train,y_train)

gsearch2.best_params_, gsearch2.best_score_
param_test3 = {'subsample':[i/100.0 for i in range(75,90,5)],

               'colsample_bytree':[i/100.0 for i in range(75,90,5)]

              }



gsearch3 = GridSearchCV(

    estimator = XGBClassifier(learning_rate =0.05,

                              n_estimators=500,

                              max_depth=2,

                              min_child_weight=2,

                              gamma=0,

                              subsample=0.8,

                              colsample_bytree=0.8,

                              objective= 'binary:logistic',

                              nthread=4,

                              scale_pos_weight=1,

                              seed=27),

    param_grid = param_test3,

    scoring='roc_auc',

    n_jobs=4,

    cv=5)

gsearch3.fit(x_scaled_train,y_train)

gsearch3.best_params_, gsearch3.best_score_
param_test4 = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}



gsearch4 = GridSearchCV(

    estimator = XGBClassifier(learning_rate =0.05,

                              n_estimators=500,

                              max_depth=2,

                              min_child_weight=2,

                              gamma=0,

                              subsample=0.8,

                              colsample_bytree=0.8,

                              objective= 'binary:logistic',

                              nthread=4,

                              scale_pos_weight=1,

                              seed=27),

    param_grid = param_test4,

    scoring='roc_auc',

    n_jobs=4,

    cv=5)

gsearch4.fit(x_scaled_train,y_train)

gsearch4.best_params_, gsearch4.best_score_
predictor=XGBClassifier(learning_rate =0.05,

                          n_estimators=500,

                          max_depth=2,

                          min_child_weight=2,

                          gamma=0,

                          subsample=0.8,

                          colsample_bytree=0.8,

                          objective= 'binary:logistic',

                          nthread=4,

                          reg_alpha=1e-5,

                          scale_pos_weight=1,

                          seed=27)

predictor.fit(x_scaled_train,y_train)

submission=pd.DataFrame(columns=['PassengerId',target])

submission['PassengerId']=test['PassengerId']

submission[target]=predictor.predict(x_scaled_test)

submission[target]=submission[target].apply(lambda x: int(x))

submission.head()
submission.to_csv('xgb.csv',index=False)