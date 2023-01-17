import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
df = pd.concat([train,test],join='outer',keys='PassengerId',sort=False,ignore_index=True)
def fill_na(df):
    df_age = df.groupby('Pclass').median()['Age']
    df_fare = df.groupby('Pclass').median()['Fare']
    df = df.merge(df_age,on='Pclass',suffixes=(None,'_y'))
    df = df.merge(df_fare,on='Pclass',suffixes=(None,'_y'))
    df['Age'].fillna(value=df['Age_y'],inplace=True)
    df['Fare'].fillna(value=df['Fare_y'],inplace=True)
    df['Embarked'].fillna(value=df['Embarked'].mode()[0],inplace=True)
    df = df.drop(['Age_y','Fare_y'],1)
    df = df.sort_values('PassengerId',ascending=True)
    return df

df = fill_na(df)
df.to_csv('df.csv')
def preprocess(df):
    df['Mr.'] = ('Mr.' in df['Name'])
    df['Mrs.'] = ('Mrs.' in df['Name'])
    df['Miss.'] = ('Miss.' in df['Name'])
    df['Master.'] = ('Master.' in df['Name'])
    df['FamSize'] = df['SibSp'] + df['Parch']
    df['NoFam'] = (df['FamSize']==0)
    df['NoSibSp'] = (df['SibSp']==0)
    df['NoParch'] = (df['Parch']==0)
    df['TicketisInt'] = (df['Ticket'].str.isdigit())
    df['CabinAlphabet'] = df['Cabin'].str[0]
    df['toddlerAge'] = (df['Age'] <= 3)
    df['youngAge'] = (df['Age'] > 3) & (df['Age'] < 18)
    df['midAge'] = (df['Age'] >= 18) & (df['Age'] < 60)
    df['oldAge'] = (df['Age'] >= 60)
    df['zeroFare'] = (df['Fare'] == 0)
    categories_to_one_hot = ['Pclass','Sex','CabinAlphabet','Embarked']
    df = pd.get_dummies(df,columns=categories_to_one_hot,drop_first=True,dtype=bool,dummy_na=True)
    df = df.drop(['Ticket','Cabin','Survived','Name'],1)
    return df
df = preprocess(df)
print(df.isna().sum())
print(df.dtypes)
# df[['Age','Fare','SibSp','Parch','FamSize']] = preprocessing.StandardScaler().fit_transform(df[['Age','Fare','SibSp','Parch','FamSize']])
df_trainandval = df.iloc[:len(train)]
df_test = df.iloc[len(train):]
acc_list = []
log_loss_list = []

for i in range(10):

    x_train, x_test, y_train, y_test = train_test_split(df_trainandval, train['Survived'], test_size = 0.2, random_state = i)

    lgb = lightgbm.LGBMClassifier(boosting_type='dart', num_leaves=50, max_depth=-1, learning_rate=0.05, n_estimators=200, 
                                           subsample_for_bin=1000, objective='binary', class_weight=None, min_split_gain=0.0, min_child_weight=0.01, 
                                           min_child_samples=20, subsample=1, subsample_freq=0, colsample_bytree=1, reg_alpha=3.0, reg_lambda=3.0, 
                                           random_state=None, n_jobs=- 1, silent=False, importance_type='split',max_drop=50,skip_drop=0.8,drop_rate=0.15,
                                           extra_trees=True)

    lgb.fit(x_train,y_train,eval_metric=['binary_error','cross_entropy'],eval_set=(x_test,y_test),verbose=200)
    pred = lgb.predict(x_test,num_iteration=-1,raw_score=False)
    acc = metrics.accuracy_score(y_test,pred)
    log_loss = metrics.log_loss(y_test,pred)
    acc_list.append(acc)
    log_loss_list.append(log_loss)

print('final mean acc:',statistics.mean(acc_list))
print('final median acc:',statistics.median(acc_list))
print('final mean log_loss:',statistics.mean(log_loss_list))
print('final median log_loss:',statistics.median(log_loss_list))
df.to_csv('df.csv',index=False)
submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = lgb.predict(df_test)
submission.to_csv('submission_0.8.csv',index=False)


lgb.fit(df_trainandval,train['Survived'])
submission['Survived'] = lgb.predict(df_test)
submission.to_csv('submission_1.csv',index=False)