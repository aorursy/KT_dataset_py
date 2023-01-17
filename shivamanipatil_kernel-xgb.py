import os

print((os.listdir('../input/')))
import pandas as pd

import numpy as np

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import matplotlib

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,precision_recall_curve,roc_curve,f1_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier

import xgboost as xgb

%matplotlib inline

matplotlib.rcParams.update({'font.size': 20})

df_train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

df_test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
train_X = df_train.loc[:, 'V1':'V16']    

train_y = df_train.loc[:, 'Class'] 

test_X = df_test.loc[:, 'V1':'V16']

count = 0

for i in range(30000):

    if(train_y[i] == 1):

        count += 1

print("no outputs as one: " + str(count))                #classes are highly imbalanced 



test_index=df_test['Unnamed: 0'] #copying test index for later

def getDummy(V_name,df):

    dfDummies = pd.get_dummies(df[V_name], prefix = V_name)

    return dfDummies

#For training set

dummy_v2 = getDummy('V2',df_train)

df_train1 = pd.concat([df_train.loc[:, 'V1':'V1'], dummy_v2], axis=1)

dummy_v3 = getDummy('V3',df_train)

df_train1 = pd.concat([df_train1, dummy_v3], axis=1)

dummy_v4 = getDummy('V4',df_train)

df_train1 = pd.concat([df_train1, dummy_v4], axis=1)

dummy_v5 = getDummy('V5',df_train)

df_train1 = pd.concat([df_train1, dummy_v5], axis=1)

df_train1 = pd.concat([df_train1, df_train.loc[:, 'V6':'V6']], axis=1)

dummy_v7 = getDummy('V7',df_train)

df_train1 = pd.concat([df_train1, dummy_v7], axis=1)

dummy_v8 = getDummy('V8',df_train)

df_train1 = pd.concat([df_train1, dummy_v8], axis=1)

dummy_v9 = getDummy('V9',df_train)

df_train1 = pd.concat([df_train1, dummy_v9], axis=1)

df_train1 = pd.concat([df_train1, df_train.loc[:, 'V10':'V10']], axis=1)

dummy_v11 = getDummy('V11',df_train)

df_train1 = pd.concat([df_train1, dummy_v11], axis=1)

df_train1 = pd.concat([df_train1, df_train.loc[:, 'V12':'V15']], axis=1)

dummy_v16 = getDummy('V16',df_train)

df_train1 = pd.concat([df_train1, dummy_v16], axis=1)

df_train1 = pd.concat([df_train1, df_train.loc[:, 'Class':'Class']], axis=1)



#for test set



dummy_v2 = getDummy('V2',df_test)

df_test1 = pd.concat([df_test.loc[:, 'V1':'V1'], dummy_v2], axis=1)

dummy_v3 = getDummy('V3',df_test)

df_test1 = pd.concat([df_test1, dummy_v3], axis=1)

dummy_v4 = getDummy('V4',df_test)

df_test1 = pd.concat([df_test1, dummy_v4], axis=1)

dummy_v5 = getDummy('V5',df_test)

df_test1 = pd.concat([df_test1, dummy_v5], axis=1)

df_test1 = pd.concat([df_test1, df_test.loc[:, 'V6':'V6']], axis=1)

dummy_v7 = getDummy('V7',df_test)

df_test1 = pd.concat([df_test1, dummy_v7], axis=1)

dummy_v8 = getDummy('V8',df_test)

df_test1 = pd.concat([df_test1, dummy_v8], axis=1)

dummy_v9 = getDummy('V9',df_test)

df_test1 = pd.concat([df_test1, dummy_v9], axis=1)

df_test1 = pd.concat([df_test1, df_test.loc[:, 'V10':'V10']], axis=1)

dummy_v11 = getDummy('V11',df_test)

df_test1 = pd.concat([df_test1, dummy_v11], axis=1)

#df_test1 = pd.concat([df_test1, df_test.loc[:, 'V11':'V11']], axis=1)

df_test1 = pd.concat([df_test1, df_test.loc[:, 'V12':'V15']], axis=1)

dummy_v16 = getDummy('V16',df_test)

df_test1 = pd.concat([df_test1, dummy_v16], axis=1)

train_X1 = df_train1.loc[:, 'V1':'V16_3']

train_X1.head()
df_test1 = df_test1.drop(['V11_0', 'V11_7', 'V11_11'], axis=1)   #dropping columns in test set which are not there in train set

test_X1 = df_test1

test_X1.head()
xTrain1, xTest1, yTrain1, yTest1 = train_test_split(train_X1, train_y, test_size = 0.2, random_state = 0)  # creaeting test and train sets from original training set

"""tried these but auc score are always high and couldnt differentiate"""

"""params = {

        'min_child_weight': [1, 5, 10],                                                                            

        'colsample_bytree': [0.2, 0.3, 0.4],

        'max_depth': [5,6, 7 , 8 ],

        'learning_rate':[0.03,0.05,0.07,0.1], 

         'n_estimators' :[50, 80, 100 ,130, 150 , 200, 230]

            

        }"""

"""params = {

    'boosting_type': ['gbdt', 'goss', 'dart'],

    'num_leaves': list(range(20, 150)),

    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),

    'subsample_for_bin': list(range(20000, 300000, 20000)),

    'min_child_samples': list(range(20, 500, 5)),

    'reg_alpha': list(np.linspace(0, 1)),

    'reg_lambda': list(np.linspace(0, 1)),

    'colsample_bytree': list(np.linspace(0.6, 1, 10)),

    'subsample': list(np.linspace(0.5, 1, 100)),

    'is_unbalance': [True, False]

}"""



#xgb = XGBClassifier(learning_rate=0.02,n-estimators = 50, objective='binary:logistic',silent=True, nthread=4)



"""xg = XGBClassifier(objective ='binary:logistic', learning_rate = 0.06,

                         alpha = 10, n_estimators=220, eval_metric='aucpr',scale_pos_weight = 19.74)"""

"""from sklearn.model_selection import cross_val_score,KFold

k_fold=KFold(n_splits=5)

scores=cross_val_score(xg,train_X1,train_y,cv=k_fold)

print(scores)

print(scores.mean())"""



#random_search = RandomizedSearchCV(xg, param_distributions=params, n_jobs=4, verbose=3, random_state=123)

#random_search.fit(train_X1, train_y)



""" Trying recall and precision as they give more information when classes are imbalanced using for loops for changing parameters

commented after running search because took time to find those 

"""



"""for est in [50, 100, 150, 200, 230, 250]:                      #scale_pos_weight of 3 is used first list was [3,4,5,6,7] 3 performed almost always better so now it is 3

    for depth in [3,4,5]:

        for g in [0,5,10]:

            for col in [0.4,0.5,0.6]:

                print('--- nof esit of {}'.format(est))

                print("depth: " + str(depth)+ " gamma: " + str(g) + " colsample: "+ str(col))

                xg = XGBClassifier(objective ='binary:logistic', learning_rate = 0.18, colsample_bytree =col,

                                     n_estimators=est, eval_metric='aucpr',scale_pos_weight = 3, gamma=g,max_depth=depth) 

                xg.fit(xTrain1, yTrain1)

                pred = xg.predict(xTest1)

                print("f1 score is: " + str(f1_score(yTest1,pred)))           #f1_score

                print(classification_report(yTest1,pred))

                """

                
""" commented after running this block and getting the learning rate"""

"""learn_rate = list()

f1_scores = list()

#taking estimators as 250 as it gives generally high result above

#now tunning learning rate

for lrate in [0.08,0.1,0.12,0.14,0.16,0.18,0.2]:

    print("learning_rate: " + str(lrate))

    xg = XGBClassifier(objective ='binary:logistic', learning_rate = lrate, colsample_bytree =0.5,

                         n_estimators=250, eval_metric='aucpr',scale_pos_weight = 3, gamma=5,max_depth=3) 

    xg.fit(xTrain1, yTrain1)                    

    pred = xg.predict(xTest1)

    f1 = f1_score(yTest1,pred)        #f1 score

    print("f1 score is: " + str(f1))                       

    print(classification_report(yTest1,pred))

    if f1 > .56:                      #0.56 because above interations could yeild max of 0.56 f1 score only

        learn_rate.append(lrate)

        f1_scores.append(f1)

        

        """
#gives lr =0.18

lr =0.18

print(lr)
xg1 = XGBClassifier(objective ='binary:logistic', learning_rate = lr, colsample_bytree =0.5,

                         n_estimators=250, eval_metric='aucpr',scale_pos_weight = 3, gamma=5,max_depth=3) 

xg1.fit(train_X1, train_y)

test_X1 = test_X1.loc[:, 'V1':'V16_3']

pred = xg1.predict_proba(test_X1)

result=pd.DataFrame()


result['Id'] = test_index=df_test['Unnamed: 0'] #copying test index for later



result['PredictedValue'] = pd.DataFrame(pred[:,1])

print(result.head())

result.to_csv('output.csv', index=False)