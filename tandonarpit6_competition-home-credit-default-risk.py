import os
print(os.listdir('../input'))
import numpy as np
import pandas as pd

application_test=pd.read_csv('../input/application_test.csv')
application_train=pd.read_csv('../input/application_train.csv')
bureau=pd.read_csv('../input/bureau.csv')
bureau_balance=pd.read_csv('../input/bureau_balance.csv')
credit_card_balance=pd.read_csv('../input/credit_card_balance.csv')
installments=pd.read_csv('../input/installments_payments.csv')
pos=pd.read_csv('../input/POS_CASH_balance.csv')
previous_application=pd.read_csv('../input/previous_application.csv')
application_test.head()
application_test.shape
application_train
bureau.head()
bureau_balance.head()
credit_card_balance.head()
installments.head()
pos.head()
previous_application.head()
X_train=application_train.iloc[:,0:10]
Y_train=X_train['TARGET']

X_test=application_test.iloc[:,0:9]
Y_train.sum()/len(Y_train)
X_train= X_train.drop(['TARGET'],axis=1)
X_train['Total Credit/ Total Income']=X_train['AMT_CREDIT']/X_train['AMT_INCOME_TOTAL']
X_train['Annuity/Income']=X_train['AMT_ANNUITY']/X_train['AMT_INCOME_TOTAL']

X_test['Total Credit/ Total Income']=X_test['AMT_CREDIT']/X_test['AMT_INCOME_TOTAL']
X_test['Annuity/Income']=X_test['AMT_ANNUITY']/X_test['AMT_INCOME_TOTAL']
X_train= X_train.drop(['AMT_CREDIT','AMT_INCOME_TOTAL','AMT_ANNUITY'],axis=1)
X_test= X_test.drop(['AMT_CREDIT','AMT_INCOME_TOTAL','AMT_ANNUITY'],axis=1)
previous_application_temp= previous_application.groupby(['SK_ID_CURR'],as_index=False)['AMT_APPLICATION','AMT_CREDIT'].sum()
previous_application_temp['Previous Credit/ Previous Application']= previous_application_temp['AMT_CREDIT']/previous_application_temp['AMT_APPLICATION']
previous_application_temp= previous_application_temp.drop(['AMT_APPLICATION','AMT_CREDIT'],axis=1)
X_train=pd.merge(X_train,previous_application_temp,on=['SK_ID_CURR'],how='left')
X_test=pd.merge(X_test,previous_application_temp,on=['SK_ID_CURR'],how='left')
credit_card_balance_temp= credit_card_balance.groupby(['SK_ID_CURR'],as_index=False)['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_CURRENT'].sum()
credit_card_balance_temp['Balance/ Credit Limit']= credit_card_balance_temp['AMT_BALANCE']/credit_card_balance_temp['AMT_CREDIT_LIMIT_ACTUAL']
credit_card_balance_temp['Drawing/ Credit Limit']= credit_card_balance_temp['AMT_DRAWINGS_CURRENT']/credit_card_balance_temp['AMT_CREDIT_LIMIT_ACTUAL']
credit_card_balance_temp= credit_card_balance_temp.drop(['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL','AMT_DRAWINGS_CURRENT'],axis=1)
X_train=pd.merge(X_train,credit_card_balance_temp,on=['SK_ID_CURR'],how='left')
X_test=pd.merge(X_test,credit_card_balance_temp,on=['SK_ID_CURR'],how='left')
X_train= X_train.drop(['SK_ID_CURR'],axis=1)
X_test= X_test.drop(['SK_ID_CURR'],axis=1)
X_train= pd.get_dummies(X_train,columns=['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY'])
X_test= pd.get_dummies(X_test,columns=['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY'])

X_train= X_train.drop(columns=['CODE_GENDER_XNA'])
X_train= X_train.replace(np.inf,np.nan)
X_test = X_test.replace(np.inf,np.nan)

X_train= X_train.fillna(0)
X_test= X_test.fillna(0)
from sklearn.linear_model import LogisticRegression

classifier_LR=LogisticRegression()
classifier_LR.fit(X_train,Y_train)

Y_pred_LR=classifier_LR.predict(X_test)
Y_pred_LR
from sklearn.ensemble import RandomForestClassifier

classifier_RF=RandomForestClassifier()
classifier_RF.fit(X_train,Y_train)

Y_pred_RF=classifier_RF.predict(X_test)
import xgboost as xgb

classifier_xgb=xgb.XGBClassifier()
classifier_xgb.fit(X_train,Y_train)

Y_pred_xgb=classifier_xgb.predict(X_test)
from keras import models
from keras import layers
from keras.layers import Dense, Dropout

model=models.Sequential()
model.add(layers.Dense(14, activation='relu',input_dim=14))
model.add(Dropout(0.6))

model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train,Y_train,batch_size=256,epochs=1)

Y_pred_DL=model.predict(X_test)
Y_pred_DL
count=0
for x in Y_pred_DL:
    if x >0.1338:
        count+=1
print(count)
print(count/len(X_test))
submission_DL=pd.DataFrame(application_test['SK_ID_CURR'],columns=['SK_ID_CURR'])
submission_LR=pd.DataFrame(application_test['SK_ID_CURR'],columns=['SK_ID_CURR'])
submission_RF=pd.DataFrame(application_test['SK_ID_CURR'],columns=['SK_ID_CURR'])
submission_DL['TARGET']=pd.DataFrame({'TARGET':Y_pred_DL[:,0]})
submission_LR['TARGET']=pd.DataFrame({'TARGET':Y_pred_LR[:,0]})
submission_RF['TARGET']=pd.DataFrame({'TARGET':Y_pred_RF[:,0]})
submission_DL.to_csv('Submission File_DL.csv',index=False)
submission_LR.to_csv('Submission File_LR.csv',index=False)
submission_RF.to_csv('Submission File_RF.csv',index=False)