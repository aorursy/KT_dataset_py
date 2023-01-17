import numpy as np 
import pandas as pd 
import tensorflow as tf
import matplotlib.pyplot as plt
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv("../input/titanic/test.csv")
df=[train,test]
test['Survived']=np.nan
train.info()
test.info()
for data in df:
#     print(data.isnull().sum())
#     data['Age'].fillna(data['Age'].mean(),inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0],inplace=True)
    data['Fare'].fillna(data['Fare'].mean(),inplace=True)
    data['Cabin']=data['Cabin'].fillna("Unknown")
    data['Deck']=data['Cabin'].str.get(0)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for data in df:
    data['Sex']=le.fit_transform(data['Sex'])
    
    data['Embarked']=le.fit_transform(data['Embarked'])
    
    data['Name']=data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
    Title_Dict = {}
    Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
    Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
    data['Title'] = data['Name'].map(Title_Dict)
    data['Title']=le.fit_transform(data['Title'])
    
    data['Lastname']=data['Name'].apply(lambda x: str.split(x, ",")[0])

    data['Age'] = data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    
#     data['FareBin'] = pd.qcut(data['Fare'], 5)
#     data['FareBin']=le.fit_transform(data['FareBin'])
    
    data['Fare']=(data['Fare']-data['Fare'].min())/(data['Fare'].max()-data['Fare'].min())

    data['FamilySize']=data['Parch']+data['SibSp']+1
    
    data['TravelAlone']=np.where(data['FamilySize']>1, 0, 1)
    
    data['Deck']=le.fit_transform(data['Deck'])
    
#     data['AgeBin'] = pd.qcut(data['Age'], 4)
#     data['AgeBin']=le.fit_transform(data['AgeBin'])
    
#     Ticket_Count = dict(data['Ticket'].value_counts())
#     data['TicketGroup'] = data['Ticket'].apply(lambda x:Ticket_Count[x])#.apply(label)
    
train.info()
test.info()
X= pd.concat([train, test])
DEFAULT_SURVIVAL_VALUE = 0.5
X['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in X[['Survived','Name', 'Lastname', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Lastname', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 0

for _, grp_df in X.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    X.loc[X['PassengerId'] == passID, 'Family_Survival'] = 0
                        
X.info()
train=X[0:len(train)]
test=X[len(train):]
train=train.drop(['PassengerId','Ticket',"Cabin",'Name','SibSp','Parch','Age','Fare','Lastname'],axis='columns')
test=test.drop(['PassengerId','Survived','Ticket',"Cabin",'Name','SibSp','Parch','Age','Fare','Lastname'],axis='columns')
train.info()
test.info()
train.sample(10)
X=train.drop(['Survived'],axis='columns').values
Y=train['Survived'].values
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
base_model=RandomForestClassifier()
base_model.fit(X,Y)
pred=base_model.predict(X)
# pred=[int(round(x[0])) for x in pred]
# print(pred[0:10])
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y,pred))
from sklearn.metrics import accuracy_score
accuracy_score(Y,pred)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

random_grid = {
                 'bootstrap': [True, False],
 'max_depth': [2,5,7,10,12,15],
 'max_features': [3, 'sqrt','log2'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [1,2,3,4],
 'n_estimators': [50,100,150,200,250,300,350],
 "criterion": ["gini", "entropy"]
             }
clf=RandomForestClassifier()
random_model=RandomizedSearchCV(estimator = clf, param_distributions = random_grid,n_iter=700, cv = sss, verbose=2, n_jobs = -1)
random_model.fit(X,Y)
print(random_model.best_score_)
random_model.best_params_
best_random = random_model.best_estimator_
pred=best_random.predict(X)
# pred=[int(round(x[0])) for x in pred]
# print(pred[0:10])
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y,pred))
from sklearn.metrics import accuracy_score
accuracy_score(Y,pred)
grid_param= {
    'bootstrap': [False],
    'max_depth': [4,5,6],
    'max_features': [3],
    'min_samples_leaf': [3,4,5],
    'min_samples_split': [4],
    'criterion': ['entropy'],
    'n_estimators': [230,240,250]
}
clf1=RandomForestClassifier()
grid_model = GridSearchCV(estimator = clf1, param_grid = grid_param, 
                          cv = sss, n_jobs = -1, verbose = 2)
grid_model.fit(X,Y)
print(grid_model.best_score_)
grid_model.best_params_
best_grid = grid_model.best_estimator_
pred=best_grid.predict(X)
# pred=[int(round(x[0])) for x in pred]
# print(pred[0:10])
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y,pred))
from sklearn.metrics import accuracy_score
accuracy_score(Y,pred)
preds=grid_model.predict(test.values)
preds=[int(x) for x in preds]
samp=pd.read_csv("../input/titanic/gender_submission.csv")
samp["Survived"]=preds
samp["Survived"].value_counts()
samp.to_csv("output.csv",index=False)
