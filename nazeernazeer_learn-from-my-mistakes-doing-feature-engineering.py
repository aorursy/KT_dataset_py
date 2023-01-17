import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn.metrics as metrics

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
# lets import data from the files 

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

submission_data = pd.read_csv('../input/gender_submission.csv')
# lets view the data 

train_data.head()
# Objective of the problem is to classify among the passengers who survived.

# Target variable considered here is Survived

train_target = train_data.Survived

#train_data.drop('Survived',axis=1,inplace=True)

#Since passenger Id doesn't add any significant to Survival lets drop that variable as well

train_data.drop('PassengerId',axis=1,inplace=True)
train_data.head()
train_target.head()
# lets see the basic statistics of data

train_data.describe()
train_data.describe(include=['O'])
# Lets check the correlation matrix

train_data.corr()
# lets Clean the data first

train_data.info() # check the null in data
# from the above its clear Age,Cabin and Embarked are having null values

train_data[train_data.Embarked.isnull()]
# these are the two instance where Embarked in null.

# lets fill them with most frequent value.

train_data[((train_data.Pclass==1)&(train_data.Fare <81)&(train_data.Fare >79)&(train_data.Sex =='female'))].Embarked.mode()
# Lets fill the embarked null values with S

train_data.Embarked.fillna('S',inplace=True)
combine = [train_data,test_data]
# lets convert Sex to binary categorical

for df in combine:

    df['Sex'] = np.where(df.Sex=='male',1,0)
# lets now take care of Age and cabin

for dataset in combine:

    # Filling all Nan with unknown cabin name.

    dataset.Cabin.fillna('Unknown',inplace=True)

    dataset['Cabin_lt'] = dataset.Cabin.apply(lambda x: str(x)[0])
sns.countplot(train_data.Cabin_lt)

plt.show()
sns.countplot(test_data.Cabin_lt)

plt.show()
sns.barplot(x='Cabin_lt',y='Survived',data=train_data)

plt.show()
# lets check how the class and Cabin_lt are related and replace it accordingly

train_data.groupby(['Pclass','Survived','Sex']).Cabin_lt.value_counts()
# lets implement the above in order impute the cabin

train_data['Cabin']= np.where(((train_data.Pclass == 1)&(train_data.Survived == 0)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'C',

                             np.where(((train_data.Pclass == 1)&(train_data.Survived == 1)&(train_data.Sex == 0)&(train_data.Cabin_lt == 'U')),'B',

                             np.where(((train_data.Pclass == 1)&(train_data.Survived == 1)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'C',

                             np.where(((train_data.Pclass == 2)&(train_data.Survived == 0)&(train_data.Sex == 0)&(train_data.Cabin_lt == 'U')),'E',

                             np.where(((train_data.Pclass == 2)&(train_data.Survived == 0)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'D',

                             np.where(((train_data.Pclass == 2)&(train_data.Survived == 1)&(train_data.Sex == 0)&(train_data.Cabin_lt == 'U')),'F',

                             np.where(((train_data.Pclass == 2)&(train_data.Survived == 1)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'F',

                             np.where(((train_data.Pclass == 3)&(train_data.Survived == 0)&(train_data.Sex == 0)&(train_data.Cabin_lt == 'U')),'G',

                             np.where(((train_data.Pclass == 3)&(train_data.Survived == 0)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'F',

                             np.where(((train_data.Pclass == 3)&(train_data.Survived == 1)&(train_data.Sex == 0)&(train_data.Cabin_lt == 'U')),'G',

                             np.where(((train_data.Pclass == 3)&(train_data.Survived == 1)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'U')),'E',

                             np.where(((train_data.Pclass == 1)&(train_data.Survived == 0)&(train_data.Sex == 1)&(train_data.Cabin_lt == 'T')),'C',

                             train_data.Cabin_lt))))))))))))
#Cross checking the update

train_data.groupby(['Pclass','Survived','Sex']).Cabin.value_counts()
# lets check how the class and Cabin_lt are related and replace it accordingly

test_data.groupby(['Pclass','Sex']).Cabin_lt.value_counts()
test_data['Cabin'] = np.where(((test_data.Pclass == 1)&(test_data.Cabin_lt == 'U')),'C',

                             np.where(((test_data.Pclass == 2)&(test_data.Sex == 0)&(test_data.Cabin_lt == 'U')),'F',

                                     np.where(((test_data.Pclass == 2 )&(test_data.Sex == 1)&(test_data.Cabin_lt == 'U')),'D',

                                             np.where(((test_data.Pclass == 3)&(test_data.Sex == 0)&(test_data.Cabin_lt =='U')),'G',

                                                     np.where(((test_data.Pclass == 3)&(test_data.Sex == 1)&(test_data.Cabin_lt =='U')),'F',

                                                             test_data.Cabin_lt)))))
test_data.groupby(['Pclass','Sex']).Cabin.value_counts()
# lets drop the Cabin_lt from train and test

for dataset in combine:

    del dataset['Cabin_lt']
for dataset in combine:

    dataset['title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.',expand=False)

    
train_data.title.value_counts()
train_data[((train_data.Age.isnull())&(train_data.title == 'Dr'))] # since there is only 1 Dr with age NaN so lets put Dr into others
for dataset in combine:# converting the titles into numericals

    dataset['title'] = np.where((dataset.title == 'Mr'),1,

                               np.where(dataset.title == 'Miss',2,

                                       np.where(dataset.title == 'Mrs',3,

                                               np.where(dataset.title == 'Master',4,5))))
train_data.title.value_counts()
# there is one more missing value that tobe imputed that is Age. so lets derive as many as varibles and use them to predict Age

# let us get one more variable Family size which can be derived from SibSp and Parch

for dataset in combine:

    dataset['Family'] = dataset.SibSp+dataset.Parch+1

    dataset['Family'] = np.where(dataset.Family == 1,'Single',

                                np.where(dataset.Family == 2,'Couple',

                                        np.where(dataset.Family <=4 ,'Nuclear','LargeF')))
sns.barplot(x=train_data.Family,y=train_data.Survived,data=train_data)

plt.show()# it shows Nuclear and Couples have survived in larger number
sns.boxplot(y=train_data.Fare)
train_data['Fare'] = train_data.Fare.astype('int32')

test_data[test_data.Fare.isnull()]
test_data.Fare.fillna(test_data[((test_data.Pclass == 3)&(test_data.Sex ==1)&(test_data.Embarked =='S')&(test_data.Family=='Single'))].Fare.mean(),inplace=True)
#lets check if the ticket is having any relation with survival

train_data.groupby(['Pclass']).Fare.min()   #[train_data.Fare >= 500].count()#['Ticket','Fare','Pclass','Family','Parch','SibSp']]
test_data.groupby(['Pclass']).Fare.min()
sns.distplot(train_data.Fare)

plt.show()
sns.distplot(test_data.Fare)

plt.show()
for dataset in combine:

    dataset['Ticket_str'] = dataset.Ticket.apply(lambda x : str(x)[0])
train_data.groupby(['Pclass']).Ticket_str.value_counts()
train_data.head()
for dataset in combine:

    dataset['Ticket_str'] = dataset['Ticket_str'].apply(lambda x: str(x))

    dataset['Ticket_str'] = np.where((dataset['Ticket_str']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), dataset['Ticket_str'],

                                np.where((dataset['Ticket_str']).isin(['W', '4', '7', '6', 'L', '5', '8']),

                                        'Low_ticket', 'Other_ticket'))

    dataset['Ticket_len'] = dataset['Ticket'].apply(lambda x: len(x))

# lets create dummies for all the variables avaiable

def dummy(data):

    dataset = pd.concat([data,pd.get_dummies(data.Pclass,prefix='Pclass'),

                     pd.get_dummies(data.Sex,prefix='Sex'),

                     pd.get_dummies(data.Cabin,prefix='Cabin'),

                    pd.get_dummies(data.Embarked,prefix='Embark'),

                    pd.get_dummies(data.title,prefix='title'),

                    pd.get_dummies(data.Family,prefix='Family'),

                    pd.get_dummies(data.Ticket_str,prefix='ticket_str'),

                    pd.get_dummies(data.Ticket_len,prefix='ticket_len')],axis=1)

    dataset.drop(['Pclass','Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked','title','Family','Ticket_str',

             'Ticket_len'],axis=1,inplace=True)

    return dataset
train = dummy(train_data)

test = dummy(test_data)
train.drop(['Survived'],axis=1,inplace=True)

train.columns
test.drop(['PassengerId'],axis=1,inplace=True)

test.columns
for dataset in [train_data,test_data]:

    for i in list(dataset["Age"][dataset["Age"].isnull()].index):

        age_mean = dataset["Age"][dataset['Family'] == dataset.iloc[i]['Family']].mean()

        Age_mean = dataset["Age"][((dataset['SibSp'] == dataset.iloc[i]["SibSp"]) & (dataset['Parch'] == dataset.iloc[i]["Parch"]) & (dataset['Pclass'] == dataset.iloc[i]["Pclass"])&(dataset['title'] == dataset.iloc[i]["title"]))].mean()

        if not np.isnan(Age_mean) :

            dataset['Age'].iloc[i] = Age_mean

        else :

            dataset['Age'].iloc[i] = age_mean
train['Age'] = train_data['Age']

test['Age'] = test_data['Age']
#trainx,testx,trainy,testy = train_test_split(train,train_target,test_size=0.2)
#param = { "criterion" : ["gini", "entropy"],

#         "max_features": [1,3,8,10,12],

#         "min_samples_leaf" : [1, 5, 10,15],

#         "min_samples_split" : [2, 4, 10, 12, 16,18,20],

#         "n_estimators": [50, 100, 400, 700,800,900, 1000]

#        }

#rf = GridSearchCV(RandomForestClassifier(oob_score=True),param_grid=param,scoring='accuracy',cv=5,n_jobs=-1,verbose=True)
#rf.fit(trainx,trainy)
#rf.best_estimator_
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',

            max_depth=None, max_features=8, max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=4,

            min_weight_fraction_leaf=0.0, n_estimators=900, n_jobs=None,

            oob_score=True, random_state=None, verbose=0, warm_start=False)



rf.fit(train, train_target)

#print("%.4f" % rf.oob_score_)
rf_train_pred = rf.predict(train)
print(metrics.accuracy_score(train_target,rf_train_pred))
test_rf = rf.predict(test)
rf_submission_testdata = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':test_rf})
#submission_testdata.to_csv('submit_learner.csv',index=False)
from xgboost.sklearn import XGBClassifier
#param_xg = {'max_depth':[2,3,4,5,6,8,10],

#        'learning_rate':[0.1,0.01,0.001,0.02],

#        'n_estimators':[100,200,300,400,500,600,800,900,1000],

#        'subsample':[0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],

#        'reg_alpha':[0.0001,0.0002,0.00003,0.00004,0.00005,0.00006,0.00007]}
#xggs =  GridSearchCV(XGBClassifier(),param_grid=param_xg,cv=5,scoring='accuracy',n_jobs=-1,verbose=True)
#xggs.best_estimator_tor_
#xggs.best_score_
xggc = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,

       max_depth=2, min_child_weight=1, missing=None, n_estimators=800,

       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,

       reg_alpha=0.0001, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=0.9)
xggc.fit(train,train_target)
xg_train_pred = xggc.predict(train)
metrics.accuracy_score(train_target,xg_train_pred)
xg_test_pred = xggc.predict(test)
print(metrics.classification_report(train_target,xg_train_pred))
print(metrics.confusion_matrix(train_target,xg_train_pred))
sns.heatmap(metrics.confusion_matrix(train_target,xg_train_pred))
xg_submission_testdata = pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':xg_test_pred})
xg_submission_testdata.to_csv('submit.csv')
xg_submission_testdata