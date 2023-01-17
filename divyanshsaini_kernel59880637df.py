# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print('os')

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#print output

def out(preds):

    output = pd.DataFrame({'PassengerId': test.PassengerId,'Survived': preds})

    output.to_csv('submission.csv', index=False)
import pandas as pd

from sklearn.metrics import roc_auc_score

from sklearn.metrics import mean_absolute_error as mea

from sklearn.metrics import recall_score as recall

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import accuracy_score

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")

rstate=10

t_size=0.3

results=pd.DataFrame(columns=['Method','Roc Score','MEA','Accuracy','Recall','Precision','F1_score'])
#data preprocessor

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

def process_data(data):

    encode = LabelEncoder()

    ss=StandardScaler()

    encode_features=['Sex','Cabin']

    data.SibSp=data.SibSp.fillna(0)+data.Parch.fillna(0)

    sex_sibsp_encode = {'female-5':8,

 'female-8':8,

 'male-3':7,

 'male-5':7,

 'male-8':7,

 'male-4':6,

 'male-0':5,

 'male-2':4,

 'male-1':3,

 'female-3':3,

 'female-4':3,

 'female-1':2,

 'female-2':1,

 'female-0':1}

    data['sex_sibsp']=[sex_sibsp_encode.get(x,5) for x in data.Sex+'-'+list(map(str,data.SibSp))]

    #Sib_data.SibSp='SibPa'

    embarked_encode={'S':1,'C':3,'Q':2,'N':0}

    name_encoded={ 'Don':0, 'Rev':0, 'Sir':0, 'Capt':0, 'Jonkheer':0, 'Major':1, 'Col':1, 'Dr':2, 'Master':3, 'Mr':4, 'Miss':5, 'Mrs':6, 'Mme':7, 'Mlle':7, 'the Countess':7, 'Lady':7 ,'Ms':7}

    data.Embarked=[embarked_encode.get(x,0) for x in data.Embarked.fillna('N')]

    data.Name=[name_encoded.get(y,0) for y in data.Name.apply(lambda x:x.split(',')[1].split('.')[0].strip())]

    data_updated=data[data.columns.drop(['Parch','Ticket','PassengerId'])]

    #data_updated=data_updated[data_updated.columns.drop('Embarked')].join(data_updated.Embarked.fillna('N'))

    data_updated=data_updated[data_updated.columns.drop('Age')].join(data_updated.Age.fillna(0))

    data_updated.Cabin=data_updated.Cabin.fillna('U').apply(lambda x:x[0])

    data_updated= data_updated[data_updated.columns.drop(encode_features)].join(data_updated[encode_features].apply(encode.fit_transform))

    data_updated.Cabin=ss.fit_transform(data_updated.Cabin.values.reshape(-1,1))

    data_updated.Name=ss.fit_transform(data_updated.Name.values.reshape(-1,1))

    data_updated.Embarked=ss.fit_transform(data_updated.Embarked.values.reshape(-1,1))

    data_updated.SibSp=ss.fit_transform(data_updated.SibSp.values.reshape(-1,1))

    data_updated.Fare=ss.fit_transform(data_updated.Fare.fillna(data_updated.Fare.mean()).values.reshape(-1,1))

    data_updated.Age=ss.fit_transform(data_updated.Age.values.reshape(-1,1))

    data_updated.sex_sibsp=ss.fit_transform(data_updated.sex_sibsp.values.reshape(-1,1))

    return data_updated
plot_train=train.copy()

processed_train=process_data(train)

processed_test=process_data(test)

processed_train_y=processed_train.Survived

processed_train_x=processed_train[processed_train.columns.drop('Survived')]
#Plotting

#from sklearn.metrics import confusion_matrix

#confusion_matrix()

processed_train.corr()
from matplotlib import pyplot as plt

train.SibSp=train.SibSp.fillna(train.SibSp.median())+train.Parch.fillna(train.Parch.median())

print(train.SibSp.unique())

fig=plt.figure(figsize=[15,10])

i=1

for j in train.SibSp.unique():

    fig.add_subplot(3,7,i)

    plt.title('For '+str(j))

    train[train.SibSp==j].Survived.value_counts().plot(kind='pie')

    i=i+1
from matplotlib import pyplot as plt

fig=plt.figure(figsize=[15,10])

i=1

plot_train.Cabin=plot_train.Cabin.fillna('U').apply(lambda x:x[0])

for j in plot_train.Cabin.unique():

    fig.add_subplot(3,7,i)

    plt.title('For '+str(j))

    plot_train[plot_train.Cabin==j].Survived.value_counts().plot(kind='pie')

    i=i+1
from matplotlib import pyplot as plt

sb_data=plot_train.loc[(plot_train.Sex=='female') ][['SibSp','Survived']]

fig=plt.figure(figsize=[15,10])

i=1

for j in sb_data.SibSp.unique():

    fig.add_subplot(3,7,i)

    plt.title('For Female '+str(j))

    sb_data[sb_data.SibSp==j].Survived.value_counts().plot(kind='pie')

    i=i+1

sb_data_male=plot_train.loc[(plot_train.Sex=='male') ][['SibSp','Survived']]

fig=plt.figure(figsize=[15,10])

i=1

for j in sb_data_male.SibSp.unique():

    fig.add_subplot(3,7,i)

    plt.title('For Male '+str(j))

    sb_data_male[sb_data_male.SibSp==j].Survived.value_counts().plot(kind='pie')

    i=i+1


# create model iterator

from sklearn.ensemble import RandomForestClassifier as rfc

def model_check(est,train_x,train_y,test_x,test_y):

    model=rfc(n_estimators=est,random_state=rstate)

    model.fit(train_x,train_y)

    preds=model.predict(test_x)

    return roc_auc_score(test_y,preds)
#splitting data

def split(X,Y):

    from sklearn.model_selection import train_test_split

    return train_test_split(X,Y,test_size=t_size,random_state=rstate)
train_x,valid_x,train_y,valid_y=split(processed_train_x,processed_train_y)

mea_dict={}

for i in (100,200,300,400,500,550,600,650,750):

    mea_dict[i]=model_check(i,train_x,train_y,valid_x,valid_y)

print(mea_dict)
#Random Forest Model

model_rf=rfc(n_estimators=100,random_state=rstate)

train_x,valid_x,train_y,valid_y=split(processed_train_x,processed_train_y)

model_rf.fit(train_x,train_y)

preds_rf=model_rf.predict(valid_x)

rf_data=pd.DataFrame({'Method':['Random Forest'],

                      'Roc Score':[roc_auc_score(valid_y,preds_rf)],

                      'MEA':[mea(valid_y,preds_rf)],

                      'Accuracy':[accuracy_score(valid_y,preds_rf)],

                      'Recall':[recall(valid_y,preds_rf)],

                      'Precision' : [precision_score(valid_y,preds_rf)],

                      'F1_score':[f1_score(valid_y,preds_rf)]})

results=results.append(rf_data)
#LightGBM Model

import lightgbm as lgb

train_x,valid_x,train_y,valid_y=split(processed_train_x,processed_train_y)

valid_x,test_x,valid_y,test_y=split(valid_x,valid_y)

Dtrain=lgb.Dataset(train_x,train_y)

Dvalid=lgb.Dataset(valid_x,valid_y)

params={'num_leaves':'70',

        'max_depth':'7',

        'objective':'binary',

        'boosting':'rf',

        'bagging_freq':1,

        'bagging_fraction':'0.9',

        'metric': 'binary_logloss'}

gbm = lgb.train(params,

                Dtrain,

                num_boost_round=9,

                valid_sets=Dvalid  # eval training data

                )

preds_gbm=list(map(lambda x: 1 if x>0.5 else 0,gbm.predict(test_x)))



gb_data=pd.DataFrame({'Method':['LightGBM'],

                      'Roc Score':[roc_auc_score(test_y,preds_gbm)],

                      'MEA':[mea(test_y,preds_gbm)],

                      'Accuracy':[accuracy_score(test_y,preds_gbm)],

                      'Recall':[recall(test_y,preds_gbm)],

                      'Precision' : [precision_score(test_y,preds_gbm)],

                      'F1_score':[f1_score(test_y,preds_gbm)]})

results=results.append(gb_data)
#Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_auc_score

from sklearn.metrics import mean_absolute_error as mea

model_nb=GaussianNB()

train_x,valid_x,train_y,valid_y=split(processed_train_x,processed_train_y)

model_nb.fit(train_x,train_y)

y_pred=model_nb.predict(valid_x)

nb_data=pd.DataFrame({'Method':['Naive Bayes'],

                      'Roc Score':[roc_auc_score(valid_y,y_pred)],

                      'MEA':[mea(valid_y,y_pred)],

                      'Accuracy':[accuracy_score(valid_y,y_pred)],

                      'Recall':[recall(valid_y,y_pred)],

                      'Precision' : [precision_score(valid_y,y_pred)],

                      'F1_score':[f1_score(valid_y,y_pred)]})

results=results.append(nb_data)
#XGB

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

model_xgb = XGBClassifier()#learning_rate=0.001,n_estimators=2500,

train_x,valid_x,train_y,valid_y=split(processed_train_x,processed_train_y)

valid_x,test_x,valid_y,test_y=split(valid_x,valid_y)

model_xgb.fit(train_x, train_y,early_stopping_rounds=10,eval_set=[(test_x,test_y)])

y_preds_x=model_xgb.predict(valid_x)

xg_data=pd.DataFrame({'Method':['XGB'],

                      'Roc Score':[roc_auc_score(valid_y,y_preds_x)],

                      'MEA':[mea(valid_y,y_preds_x)],

                      'Accuracy':[accuracy_score(valid_y,y_preds_x)],

                      'Recall':[recall(valid_y,y_preds_x)],

                      'Precision' : [precision_score(valid_y,y_preds_x)],

                      'F1_score':[f1_score(valid_y,y_preds_x)]})

results=results.append(xg_data)
#MLP

from sklearn.neural_network import MLPClassifier

train_x,valid_x,train_y,valid_y=split(processed_train_x,processed_train_y)

model_MLP=MLPClassifier()#hidden_layer_sizes=(100), max_iter=50,activation = 'relu',solver='lbfgs',random_state=rstate)

model_MLP.fit(train_x,train_y)

preds_mlp_y=model_MLP.predict(valid_x)

ml_data=pd.DataFrame({'Method':['MLP'],

                      'Roc Score':[roc_auc_score(valid_y,preds_mlp_y)],

                      'MEA':[mea(valid_y,preds_mlp_y)],

                      'Accuracy':[accuracy_score(valid_y,preds_mlp_y)],

                      'Recall':[recall(valid_y,preds_mlp_y)],

                      'Precision' : [precision_score(valid_y,preds_mlp_y)],

                      'F1_score':[f1_score(valid_y,preds_mlp_y)]})

results=results.append(ml_data)
#KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

train_x,valid_x,train_y,valid_y=split(processed_train_x,processed_train_y)

model_KN=KNeighborsClassifier()

model_KN.fit(train_x,train_y)

preds_kn_y=model_KN.predict(valid_x)

kn_data=pd.DataFrame({'Method':['Kneigh'],

                      'Roc Score':[roc_auc_score(valid_y,preds_kn_y)],

                      'MEA':[mea(valid_y,preds_kn_y)],

                      'Accuracy':[accuracy_score(valid_y,preds_kn_y)],

                      'Recall':[recall(valid_y,preds_kn_y)],

                      'Precision' : [precision_score(valid_y,preds_kn_y)],

                      'F1_score':[f1_score(valid_y,preds_kn_y)]})

results=results.append(kn_data)
results

#MLP return .77 result.. best... need to do tuning
predictions=model_MLP.predict(processed_test)

out(predictions)