from IPython.display import Image
Image("../input/mypng1/Screenshot-xgb.png")
from IPython.display import Image
Image("../input/mypngs2/Screenshot-nn.png")
import pandas as pd
import numpy as np
import random as rnd
import math
import collections
import pickle


import xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler

import os
print(os.listdir("../input"))
# train_df = pd.read_csv('./kaggle.titanic/train-test.csv' )
train_df = pd.read_csv('../input/mycsvdata/train-test.csv' )
print( train_df.shape )
print( train_df.head(3) )    
print( train_df.tail(3) )    

train_df['Title'] = train_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train_df['Title'].unique()
train_df['Title'] = train_df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train_df['Title'] = train_df['Title'].replace('Mlle', 'Miss')
train_df['Title'] = train_df['Title'].replace('Ms', 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train_df['Title'] = train_df['Title'].map(title_mapping)
train_df['Title'] = train_df['Title'].fillna(0)

print( train_df.shape )
train_df.head()
train_df['Sex'] = train_df['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()
# based on Manav-san's kernel
guess_ages = np.zeros((2,3))
# guess_ages
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = train_df[(train_df['Sex'] == i) & \
                              (train_df['Pclass'] == j+1)]['Age'].dropna()

        # age_mean = guess_df.mean()
        # age_std = guess_df.std()
        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

        age_guess = guess_df.median()

        # Convert random age float to nearest .5 age
        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

for i in range(0, 2):
    for j in range(0, 3):
        train_df.loc[ (train_df.Age.isnull()) & (train_df.Sex == i) & (train_df.Pclass == j+1),\
                'Age'] = guess_ages[i,j]

train_df['Age'] = train_df['Age'].astype(int)

train_df.head()
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df['IsAlone'] = 0
train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df['Age*Class'] = train_df.Age * train_df.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
train_df.columns
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
train_df['Embarked'] = train_df['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df['Embarked'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()
print( train_df['Fare'].describe() )
print( train_df['Fare'].unique().shape )
train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)
train_df.head()
train_df.describe(include="all")
train_df["Fare"].unique()
train_df[["Ticket"]].values[0:10]
train_df.columns
ticket_list = train_df[["Ticket"]].values.tolist()
# 上のtolist()だとリスト化されるがその要素もリストとなる。そのため
# 文字列とするために以下の１行を追加。
ticket_list = [_i[0] for _i in ticket_list]
ticket_count = [0 for _i in range(len(ticket_list))]
c = collections.Counter(ticket_list)
for _i in c.keys():
    # 辞書のキーを順に処理。ticket_listの要素と同じならインデックスを抽出。
    # 抽出されたインデックスに関し出現回数を設定
    for _l in [_j for _j, x in enumerate(ticket_list) if x == _i]: 
#             print(_i,_l,c[_i])
        ticket_count[_l] = c[_i]
train_df = pd.concat([train_df, pd.DataFrame(data=ticket_count, columns=["Ticket C"], dtype='int')], axis=1)
train_df['FareAdj'] = train_df['Fare'] / train_df['Ticket C']

train_df.head(3)
ticket_list = train_df[["Ticket"]].values.tolist()
ticket_list = [_i[0] for _i in ticket_list]
family_survived_count = [0 for _i in range(len(ticket_list))]
c = collections.Counter(ticket_list)

for _i, _t in enumerate(ticket_list):
    same_g = train_df.query('Ticket==@_t')[["PassengerId"]].values.tolist( )
    same_g = [_i[0] for _i in same_g]
    if len(same_g) == 1:
        continue
#         print(_i,same_g," ", end="")
    tmpcounter = 0
    for _l in [_x for  _x in same_g if _x!= (_i+1) ]: 
#             print("_l :",_l, end="")
        if (not math.isnan(train_df.loc[_l-1,"Survived"])) :
            tmpcounter += train_df.loc[_l-1,"Survived"]
#             print( " tmpcounter :",tmpcounter)
#         for _l in [_j for _j, _x in enumerate(ticket_list) if _x == _i]:
#             family_survived_count[_l]=tmpcounter
    family_survived_count[_i]=tmpcounter
train_df = pd.concat([train_df, pd.DataFrame(data=family_survived_count, columns=["Family_S_C"], dtype='int')], axis=1)


train_df
train_df['Simple_S_C']=train_df['Family_S_C'].apply(lambda x: 1 if x >0 else 0)
train_df[['Family_S_C', 'Simple_S_C']]
train_df.columns
train_df = train_df.drop(["Ticket", "Fare", "Age*Class","Ticket C"], axis=1)
train_df.columns
train_df['Cabin'].unique()
train_df['CabinInfo'] = train_df['Cabin'].apply(lambda _x: str(_x)[0:1] if type(_x) == str else 'noinfo' ) 

train_df['CabinInfo'].unique()

title_mapping = {"noinfo":0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F":6, "G":7, "T":8 }
train_df['CabinInfo'] = train_df['CabinInfo'].map(title_mapping)

print( train_df.columns )
train_df['CabinInfo'].unique()
train_df.shape
test_df = train_df.query('PassengerId>=892')
test_df = test_df.drop("Survived", axis=1)
train_df = train_df.query('PassengerId<=891')
print( train_df.shape )
print( test_df.shape )
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
X_train.columns
X_test.columns
X_test
# print( type(X_train) )
# print( X_train.describe() )
X_train.mean()
print("Pclass: ", pd.unique( X_train["Pclass"] ))
print("Sex: ", pd.unique( X_train["Sex"] ))
print("Age: ", pd.unique( X_train["Age"] ))
# print("Fare: ", pd.unique( X_train["Fare"] ))
print("FareAdj: ", pd.unique( X_train["FareAdj"] ))
print("Embarked: ", pd.unique( X_train["Embarked"] ))
print("Title: ", pd.unique( X_train["Title"] ))
# print("IsAlone: ", pd.unique( X_train["IsAlone"] ))
print("FamilySize: ", pd.unique( X_train["FamilySize"] ))
print("Family_S_C: ", pd.unique( X_train["Family_S_C"] ))
print( X_train.columns )
print( X_train["FamilySize"].unique() )
print( X_test["FamilySize"].unique() )
print( X_train["Family_S_C"].unique() )
print( X_test["Family_S_C"].unique() )
scaler_for_family = MinMaxScaler()
# scaler_for_family.fit( X_train[['FamilySize' ]] )
# xxx = X_train[["FamilySize","Family_S_C"]]
scaler_for_family.fit( [[0],[11]] )

print( X_train[["FamilySize","Family_S_C"]].head(12) )
X_train[["FamilySize","Family_S_C"]] =  scaler_for_family.transform(X_train[["FamilySize","Family_S_C" ]])
print( X_train[["FamilySize","Family_S_C"]].head(12) )

print( X_test[["FamilySize","Family_S_C"]].head(12) )
X_test[["FamilySize","Family_S_C"]] =  scaler_for_family.transform(X_test[["FamilySize","Family_S_C" ]])
print( X_test[["FamilySize","Family_S_C"]].head(12) )
columns_to_transform = X_train.columns
scaler = MinMaxScaler()
X_train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Title", "IsAlone", "FareAdj", "Simple_S_C", "CabinInfo" ] ] =  scaler.fit_transform(X_train[ [ "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked", "Title", "IsAlone", "FareAdj", "Simple_S_C", "CabinInfo" ]])

print("Pclass: ", pd.unique( X_train["Pclass"] ))
print("Sex: ", pd.unique( X_train["Sex"] ))
print("Age: ", pd.unique( X_train["Age"] ))
# print("Fare: ", pd.unique( X_train["Fare"] ))
print("FareAdj: ", pd.unique( X_train["FareAdj"] ))
print("Embarked: ", pd.unique( X_train["Embarked"] ))
print("Title: ", pd.unique( X_train["Title"] ))
# print("IsAlone: ", pd.unique( X_train["IsAlone"] ))
print("FamilySize: ", pd.unique( X_train["FamilySize"] ))
print("Family_S_C: ", pd.unique( X_train["Family_S_C"] ))
print( type(X_test) )
print( X_test.describe() )
X_test.mean()
print("Pclass: ", pd.unique( X_test["Pclass"] ))
print("Sex: ", pd.unique( X_test["Sex"] ))
print("Age: ", pd.unique( X_test["Age"] ))
# print("Fare: ", pd.unique( X_test["Fare"] ))
print("FareAdj: ", pd.unique( X_test["FareAdj"] ))
print("Embarked: ", pd.unique( X_test["Embarked"] ))
print("Title: ", pd.unique( X_test["Title"] ))
# print("IsAlone: ", pd.unique( X_test["IsAlone"] ))
print("FamilySize: ", pd.unique( X_test["FamilySize"] ))
print("Family_S_C: ", pd.unique( X_test["Family_S_C"] ))
X_test[["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked",  "Title", "IsAlone", "FareAdj", "Simple_S_C", "CabinInfo" ] ] =  scaler.fit_transform(X_test[ [  "Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked",  "Title", "IsAlone", "FareAdj", "Simple_S_C", "CabinInfo" ]] )


print("Pclass: ", pd.unique( X_test["Pclass"] ))
print("Sex: ", pd.unique( X_test["Sex"] ))
print("Age: ", pd.unique( X_test["Age"] ))
# print("Fare: ", pd.unique( X_test["Fare"] ))
print("FareAdj: ", pd.unique( X_test["FareAdj"] ))
print("Embarked: ", pd.unique( X_test["Embarked"] ))
print("Title: ", pd.unique( X_test["Title"] ))
# print("IsAlone: ", pd.unique( X_test["IsAlone"] ))
print("FamilySize: ", pd.unique( X_test["FamilySize"] ))
print("Family_S_C: ", pd.unique( X_test["Family_S_C"] ))
print("type of Y_train: ",type(Y_train) )
print( Y_train.describe() )
print( Y_train.unique() )
pickle_file = "./titanic.2018-11-24.pickle"

with open(pickle_file, 'wb') as f:
    pickle.dump(X_train, f)
    pickle.dump(X_test, f)
    pickle.dump(Y_train, f)
# pickle_file03 = "/home/hiroshisakuma/ml/machine-learning/titanic.2018-11-03.pickle"
# pickle_file10 = "/home/hiroshisakuma/ml/machine-learning/titanic.2018-11-10.pickle"

# with open(pickle_file03, 'rb') as f:
#     X_train03 = pickle.load(f)
#     X_test03 = pickle.load(f)
#     Y_train03 = pickle.load(f)

# with open(pickle_file10, 'rb') as f:
#     X_train10 = pickle.load(f)
#     X_test10 = pickle.load(f)
#     Y_train10 = pickle.load(f)

# print( X_train03.columns )
# print( X_train10.columns )

# # lst = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'Title', 'FamilySize', 'IsAlone', 'FareAdj', 'Family_S_C', 'Simple_S_C', 'CabinInfo']
# lst = X_train03.columns
# print(lst)
# for _i, _l in enumerate(lst):
#     print( _i, (X_train03[_l] == X_train10[_l]).sum() )
# lst = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin', 'Embarked', 'Title', 'IsAlone', 'FamilySize', 'FareAdj', 'Family_S_C', 'Simple_S_C', 'CabinInfo']
# print(lst)
# for _i,_l in enumerate(lst):
#     print( _i, (X_test03[_l] == X_test10[_l]).sum() )

# print( (Y_train03 == Y_train10).sum() )
import tensorflow as tf
import keras
print(tf.__version__)
print(keras.__version__) # 2.1.5
pickle_file = "./titanic.2018-11-24.pickle"

with open(pickle_file, 'rb') as f:
    X_train = pickle.load(f)
    X_test = pickle.load(f)
    Y_train = pickle.load(f)
data_col= [ "Sex", "Age", "FamilySize", "FareAdj", "Simple_S_C" , "CabinInfo"]
X_train = X_train[ data_col ]
X_test = X_test[ data_col ]

# test = pd.read_csv("./kaggle.titanic/test.csv",header=0)
test = pd.read_csv("../input/titanic/test.csv",header=0)
params={
#         "eta":0.005,
#         "learning_rate" :0.01, 
#         "n_estimators":5000, 
# #         "max_depth":4,
#         "max_depth":5,
#         "min_child_weight":5, 
#         "gamma":0, 
#         "subsample":0.8, 
#         "colsample_bytree":0.95,
#         "reg_alpha":1e-05,
#         "objective": 'binary:logistic', 
#         "nthread":4, 
#         "scale_pos_weight":1, 
#         "seed":29
            "learning_rate":0.01, 
            "n_estimators":5000, 
            "max_depth":5,
            "min_child_weight":5, 
            "gamma":0, 
            "subsample":0.8, 
            "colsample_bytree":0.95,
            "reg_alpha":1e-05,
            "objective": 'binary:logistic', 
            "nthread":4, 
            "scale_pos_weight":1, 
            "seed":29
    }

            
dtrain = xgb.DMatrix(X_train, label=Y_train)
cv=xgb.cv(params,dtrain,num_boost_round=200,nfold=10)
minid=cv[["test-error-mean"]].idxmin()
cv[["test-error-mean"]].min()
cv
minid[["test-error-mean"]].values[0]
# bst=xgb.train(params,dtrain,num_boost_round=minid)
bst=xgb.train(params,dtrain,num_boost_round=minid[["test-error-mean"]].values[0])
dtest = xgb.DMatrix(X_test)
ypred = bst.predict(dtest)
answer = ypred.round().astype(int)
answer

# df_test_origin = pd.read_csv('./kaggle.titanic/test.csv')
df_test_origin = pd.read_csv('../input/titanic/test.csv')
submit_data =  pd.Series(answer, name='Survived', index=df_test_origin['PassengerId'])
# submit_data.to_csv('./kaggle.titanic/submit4.csv', header=True)
submit_data.to_csv('./submit.xgboost.csv', header=True)

import subprocess
import sys
import datetime
import os
import random as rn

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import regularizers
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout
from keras.models import Model, load_model
# pickle_file = "titanic.2018-11-03.pickle"
pickle_file = "./titanic.2018-11-24.pickle"
# mygdrive = "/gdrive/My Drive/"

# res = subprocess.run(["uname",  "-a"], stdout=subprocess.PIPE)
# myenv = res.stdout.split()
    
# if myenv[1] != b'hiroshisakuma-ThinkPad-E480':
#     from google.colab import drive
#     drive.mount('/gdrive')


# if myenv[1] != b'hiroshisakuma-ThinkPad-E480':
#     print("Data Processing in Colab env.")
#     from google.colab import drive
#     drive.mount('/gdrive')
#     pickle_file = mygdrive + pickle_file
    

# else:
    
#     pickle_file = './' + pickle_file

with open(pickle_file, 'rb') as f:
    X_train = pickle.load(f)
    X_test = pickle.load(f)
    Y_train = pickle.load(f)
    
X_train.info()
print('_'*40)
X_test.info()
X_train = pd.concat([X_train, Y_train], axis=1)

print( X_train.columns )
print( X_test.columns )
data_col= [ "Sex", "Age", "FamilySize", "FareAdj", "Simple_S_C" ]

train_targets = Y_train.values
train_data = X_train[ data_col ].values
test_data   = X_test[ data_col ].values
train_data
fn = 0           # number of units of fhe first  dense
dim_of_hlayer=[] # number of units of for hidden layers

# _id : paramter for ft.set_random_seed
# _dr : dropout ratio
# _ac : activation function
# _kreg : value for regularizers
def build_model(_id, _dr, _ac, _kreg): 
    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(_id)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
    
    # first layer
    inputs2 = Input(shape=(train_data.shape[1],))
    x2 = Dense(fn, activation=_ac, kernel_regularizer=regularizers.l1(_kreg), bias_regularizer=regularizers.l1(_kreg) )(inputs2) 
#     x2 = Dense(fn, activation=_ac, kernel_regularizer=regularizers.l1_l2(_kreg,_kreg) )(inputs2) 
    x2 = Dropout(_dr)(x2)

    # hidden layers if dim_of_hlayer contains values
    if len(dim_of_hlayer):
        for _l in dim_of_hlayer:
            x2 = Dense(_l, activation=_ac, kernel_regularizer=regularizers.l1(_kreg), bias_regularizer=regularizers.l1(_kreg) )(x2)
#             x2 = Dense(_l, activation=_ac, kernel_regularizer=regularizers.l1_l2(_kreg,_kreg))(x2)
            x2 = Dropout(_dr)(x2)
        
    main_output = Dense(1, activation='sigmoid', name='main_output')(x2)
    
    model = Model(inputs=inputs2, outputs=main_output)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
print( X_train.columns.values )
print( X_test.columns.values )
print( train_data.shape, test_data.shape )
def check_result(_model,_id):
  y_test_nn = _model.predict( X_test[ data_col ].values )
  y_pred = np.round(y_test_nn)
  result_df = X_test
  result_df = result_df.reset_index()############################################################
  result_df = result_df.drop(['FareAdj','Embarked'],  axis=1) 
  result_df = pd.concat([result_df, pd.DataFrame(data=y_pred, columns=["Survived"], dtype='int')], axis=1)
  print( "result at ",_id," ",result_df.query('Sex==0 & Survived==1').shape, result_df.query('Sex==1 & Survived==1').shape    ,#) #######################
    ((result_df['Sex']==0) & (result_df['Pclass']==0) & (result_df['Survived']==1)).sum(),
    ((result_df['Sex']==0) & (result_df['Pclass']==0.5) & (result_df['Survived']==1)).sum(),
    ((result_df['Sex']==0) & (result_df['Pclass']==1) & (result_df['Survived']==1)).sum(),
    ((result_df['Sex']==1) & (result_df['Pclass']==0) & (result_df['Survived']==1)).sum(),
    ((result_df['Sex']==1) & (result_df['Pclass']==0.5) & (result_df['Survived']==1)).sum(),
    ((result_df['Sex']==1) & (result_df['Pclass']==1) & (result_df['Survived']==1)).sum()
    )    

  return y_pred, result_df
k = 5
num_epochs = 75
batch_size=32
models = []

# _id : paramter for ft.set_random_seed
# _dr : dropout ratio
# _ac : activation function
# _kreg : value for regularizers
#  they are paraters for build_model()
def run_model(_id, _dr, _ac, _kreg): 
    
    num_val_samples = len(train_data) // k
    all_scores = []
    for i in range(k):
        val_data    = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
        
        partial_train_data = np.concatenate( [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
        partial_train_targets = np.concatenate( [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)
        
        model = build_model(_id, _dr, _ac, _kreg)
#         if not models: 
#             model = build_model(_id, _dr, _ac, _kreg)
        
        if i == 0:
            dt_now = datetime.datetime.now()
            print(dt_now)
            print( "part train data & target   val data & target")
            print( len(partial_train_data), len(partial_train_targets), len(val_data), len(val_targets) )
            print( "# of folds: ", k, "num_epochs: ", num_epochs, "batch_size: ", batch_size )
            model.summary()
            
        def step_decay(epoch):
          x = 0.0001 
          if epoch >= 25: x = 0.00001
          if epoch >= 50: x = 0.000001
          return x
        
        lr_decay = LearningRateScheduler(step_decay,verbose=0)
        history = model.fit(partial_train_data, 
              partial_train_targets,
              epochs=num_epochs, batch_size=batch_size, verbose=0,
              validation_data=(val_data, val_targets) ,
              callbacks=[lr_decay])
        
#         if myenv[1] != b'hiroshisakuma-ThinkPad-E480':
#             model.save('/gdrive/My Drive/tatanic.models/my_model'+str(_id)+"-"+str(i)+'.h5')
#         else:
#             model.save('./titanic.models/my_model'+str(_id)+"-"+str(i)+'.h5')
        model.save('./my_model'+str(_id)+"-"+str(i)+'.h5')
        models.append(model)
        ypred, r_df = check_result(model,_id)
        all_scores.append(history)
        
    return all_scores, history
# data_col= [ "Sex", "Age", "FamilySize", "FareAdj", "IsAlone", "Simple_S_C" , "CabinInfo"]
# data_col= [ "Sex", "Age", "FamilySize", "FareAdj", "Simple_S_C" , "CabinInfo"]
data_col= [ "Sex", "Age", "FamilySize", "FareAdj", "Simple_S_C" ]


models=[];  all_scores=[]; num_epochs=100; fn=50; dim_of_hlayer=[50,50,50,50,50,50]; print("I set fn=",fn) 
for _i in  [ 0 ]:
  all_scores, history = run_model(_i, 0, 'relu', 0 ) # random_seed, dropr, activation, weight
  for _d in all_scores:
      print( "acc {:.2}  val_acc {:.2}".format( np.mean(_d.history['acc']), np.mean(_d.history['val_acc'])  ))
  models=[]

# # data_col= [ "Sex", "Age", "FamilySize", "FareAdj", "IsAlone", "Simple_S_C" , "CabinInfo"]
# # data_col= [ "Sex", "Age", "FamilySize", "FareAdj", "Simple_S_C" , "CabinInfo"]
# data_col= [ "Sex", "Age", "FamilySize", "FareAdj", "Simple_S_C" ]

# models=[];  all_scores=[]; num_epochs=100; fn=50; dim_of_hlayer=[50,50,50,50,50,50,50]; print("I set fn=",fn) 
# for _i in  [ 0 ]:
#   all_scores, history = run_model(_i, 0, 'relu', 0 ) # random_seed, dropr, activation, weight
#   for _d in all_scores:
#       print( "acc {:.2}  val_acc {:.2}".format( np.mean(_d.history['acc']), np.mean(_d.history['val_acc'])  ))
#   models=[]

_w = "0"            # paramter for ft.set_random_seed
model_file_id="4"   # last model

history = all_scores[int(model_file_id)]
history_dict = history.history
history_dict.keys()

for _d in all_scores:
    print( "acc {:.2}  val_acc {:.2}".format(
            np.mean(_d.history['acc']),
            np.mean(_d.history['val_acc']) 
        ))
    
plt.style.use('dark_background')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
# if myenv[1] != b'hiroshisakuma-ThinkPad-E480':
#     model = load_model('/gdrive/My Drive/tatanic.models/my_model'+_w+'-'+model_file_id+'.h5')
#     print('/gdrive/My Drive/tatanic.models/my_model'+_w+'-'+model_file_id+'.h5')
    
# else:
#     model = load_model('./titanic.models/my_model'+_w+'-'+model_file_id+'.h5')
#     print('./titanic.models//my_model'+_w+'-'+model_file_id+'.h5')

model = load_model('./my_model'+_w+'-'+model_file_id+'.h5')
print('./my_model'+_w+'-'+model_file_id+'.h5')

y_test_nn = model.predict(  X_test[ data_col ].values, verbose=1 )
y_pred = np.round(y_test_nn)

# if myenv[1] != b'hiroshisakuma-ThinkPad-E480':
#     result_df = pd.read_csv("/gdrive/My Drive/kaggle.titanic/test.csv")
# else:
#     result_df = pd.read_csv("./kaggle.titanic/test.csv")

result_df = pd.read_csv("../input/titanic/test.csv")

result_df = pd.concat([result_df, pd.DataFrame(data=y_pred, columns=["Survived"], dtype='int')], axis=1)
result_df = result_df.drop(['Pclass', 'Name', 'Sex', 'Age','SibSp', 'Parch','Ticket','Fare','Cabin','Embarked'],  axis=1) 
result_df.columns
# _t = '2018-11-24v1-0-4'

# if myenv[1] != b'hiroshisakuma-ThinkPad-E480':
#     result_df.to_csv('/gdrive/My Drive/kaggle.titanic/submission.'+_t+'_'+model_file_id+'.csv', index=False)
# else:
#     result_df.to_csv('./kaggle.titanic/submission.'+_t+'_'+model_file_id+'.csv', index=False)

submit_data.to_csv('./submit.keras-nn.csv', header=True)