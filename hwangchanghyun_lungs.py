import multiprocessing

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
files=['../input/osic-pulmonary-fibrosis-progression/train.csv',

       '../input/osic-pulmonary-fibrosis-progression/test.csv',

       '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv']



with multiprocessing.Pool() as pool:

    train,test,submission=pool.map(pd.read_csv,files)
# train.head()

# test.head()

# submission.head()



#####################

# 1. y is FVC 

# 2. confidence is metric=−2–√Δσclipped−ln(2–√σclipped).

# 3. need one-hot encode 

# 4. quantile model

# 4. submission merge test data
train=train.sort_values(['Patient','Weeks'])
train['BaseFVC']=train.groupby('Patient')['FVC'].transform(lambda x:x.iloc[0])

train['BaseWeeks']=train.groupby('Patient')['Weeks'].transform(lambda x:x.iloc[0])

train['BasePercent']=train.groupby('Patient')['Percent'].transform(lambda x:x.iloc[0])
from sklearn.model_selection import StratifiedKFold



X=train.drop(['FVC'],axis=1)

y=train['FVC']

skf=StratifiedKFold(n_splits=4,shuffle=True,random_state=2020)

train['SexMean']=np.nan

train['SmokingMean']=np.nan



mean=train.groupby('Sex')['FVC'].mean()

nrows=train.groupby('Sex')['FVC'].size()

globalMean=train['FVC'].mean()

alpha=5

meanEncode=(mean*nrows+alpha*globalMean)/(nrows+alpha)

test['SexMean']=test['Sex'].map(meanEncode)



mean=train.groupby('SmokingStatus')['FVC'].mean()

nrows=train.groupby('SmokingStatus')['FVC'].size()

globalMean=train['FVC'].mean()

alpha=5

meanEncode=(mean*nrows+alpha*globalMean)/(nrows+alpha)

test['SmokingMean']=test['SmokingStatus'].map(meanEncode)





def featureEngineer(col,call):

    for train_idx,valid_idx in skf.split(X,y):

        X_tr,X_val=train.iloc[train_idx],train.iloc[valid_idx]



        mean=X_tr.groupby(col)['FVC'].mean()

        nrows=X_tr.groupby(col)['FVC'].size()

        globalMean=X_tr['FVC'].mean()

        alpha=5

        meanEncode=(mean*nrows+alpha*globalMean)/(nrows+alpha)

        X_val[call]=X_val[col].map(meanEncode)

        train.iloc[valid_idx]=X_val

    print('done')

    

featureEngineer('Sex','SexMean')

featureEngineer('SmokingStatus','SmokingMean')
# def freqs(df,col):

#     freq=df.groupby(col)[col].size()/df.shape[0]

#     return freq



# freq=freqs(train,'Sex')

# train['freqSex']=train['Sex'].map(freq)

# test['freqSex']=test['Sex'].map(freq)

# freq=freqs(train,'SmokingStatus')

# train['freqSmoking']=train['SmokingStatus'].map(freq)

# test['freqSmoking']=test['SmokingStatus'].map(freq)
def getDummies(df):

    smokingDummies=pd.get_dummies(df['SmokingStatus'])

    sexDummies=pd.get_dummies(df['Sex'])

    df=pd.concat([df,smokingDummies,sexDummies],axis=1)

    return df



train=getDummies(train)

test=getDummies(test)
submission['Patient']=submission['Patient_Week'].apply(lambda x:x.split('_')[0])

submission['Weeks']=submission['Patient_Week'].apply(lambda x:x.split('_')[1]).astype(int)
def ageBand(x):

    if x<=54:

        return 'under54'

    elif x<=64:

        return 'under64'

    elif x<=74:

        return 'under74'

    else:

        return 'other'

    

train['ageBand']=train['Age'].apply(lambda x:ageBand(x))

test['ageBand']=test['Age'].apply(lambda x:ageBand(x))
def getDummies(df):

    dummies=pd.get_dummies(df['ageBand'])

    df=pd.concat([df,dummies],axis=1)

    return df



n=train.shape[0]

data=pd.concat([train,test])

data=getDummies(data)

train=data.iloc[:n]

test=data.iloc[n:].dropna(axis=1)
train=train.drop(['Age','Sex','Currently smokes','Female','other'],axis=1)

test=test.drop(['Age','Sex','other'],axis=1)
train=train.drop(['ageBand','Male','SmokingStatus'],axis=1)

test=test.drop(['ageBand','SmokingStatus'],axis=1)
merge=pd.merge(test,submission,on=['Patient'],how='left').sort_values(['Weeks_y','Patient']).reset_index(drop=True)

merge=merge.drop(['FVC_y'],axis=1)

merge=merge.rename(columns={'FVC_x':'BaseFVC','Weeks_y':'Weeks','Weeks_x':'BaseWeeks','Percent':'BasePercent'})



del test

del submission



test=merge.loc[:,['Patient','Weeks','BaseWeeks','BasePercent','SexMean','SmokingMean',

                  'BaseFVC','Ex-smoker','Never smoked','Male','under54','under64','under74']]

submission=merge.loc[:,['Patient_Week','BaseFVC','Confidence']]

submission=submission.rename(columns={'BaseFVC':'FVC'})
# merge=pd.merge(test,submission,on=['Patient'],how='left').sort_values(['Weeks_y','Patient']).reset_index(drop=True)

# merge=merge.drop(['FVC_y'],axis=1)

# merge=merge.rename(columns={'FVC_x':'BaseFVC','Weeks_y':'Weeks','Weeks_x':'BaseWeeks','Percent':'BasePercent'})



# del test

# del submission



# test=merge.loc[:,['Patient','Weeks','BaseWeeks','BasePercent','freqSex','freqSmoking',

#                   'BaseFVC','Ex-smoker','Never smoked','Male','under54','under64','under74']]

# submission=merge.loc[:,['Patient_Week','BaseFVC','Confidence']]

# submission=submission.rename(columns={'BaseFVC':'FVC'})
feature=['Weeks','SexMean','SmokingMean','BaseFVC','BaseWeeks','BasePercent',

        'Ex-smoker', 'Never smoked','under54', 'under64', 'under74']



X_train=train.loc[:,feature]

y_train=train['FVC']

X_test=test.loc[:,feature]
from sklearn.ensemble import GradientBoostingRegressor



alpha = 0.8

result=pd.DataFrame()



model = GradientBoostingRegressor(loss='quantile', alpha=alpha,

                                n_estimators=500, max_depth=5,

                                learning_rate=.05,random_state=2020)

model.fit(X_train,y_train)

Upper=model.predict(X_test)

result['upper']=Upper



model.set_params(alpha=1.0-alpha)

model.fit(X_train,y_train)

Lower=model.predict(X_test)

result['lower']=Lower



model.set_params(loss='ls')

model.fit(X_train,y_train)

pred=model.predict(X_test)

result['pred']=pred
result=pd.concat([test['Patient'],result],axis=1)
submission['FVC']=result['pred']

submission['Confidence']=result['upper']-result['lower']
submission.to_csv('submission.csv',index=False)