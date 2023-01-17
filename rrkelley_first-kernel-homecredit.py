import os
import pandas as pd
import numpy as np

def prep_data(dataframe):
    df = dataframe
    #Drop the FLAG_DOCUMENT_ variables
    documents = list()
    for i in range(2, 22):
        documents.append('FLAG_DOCUMENT_'+str(i))
    df.drop(documents, axis=1, inplace=True)
    
    #Recode Contract Types
    df.NAME_CONTRACT_TYPE.unique()
    df.NAME_CONTRACT_TYPE.replace('Cash loans', 0, inplace=True)
    df.NAME_CONTRACT_TYPE.replace('Revolving loans', 1, inplace=True)
    
    #Recode Gender
    df.CODE_GENDER.replace('M', 0, inplace=True)
    df.CODE_GENDER.replace('F', 1, inplace=True)
    df.CODE_GENDER.replace('XNA', 2, inplace=True)
    
    #Recode Own Car
    df.FLAG_OWN_CAR.replace('Y', 1, inplace=True)
    df.FLAG_OWN_CAR.replace('N', 0, inplace=True)
    
    #Record Own Realty
    df.FLAG_OWN_REALTY.replace('Y', 1, inplace=True)
    df.FLAG_OWN_REALTY.replace('N', 0, inplace=True)
    
    #Replace missing data with zeros (0) or 'Unknown' for Categorical variables.
    df.AMT_ANNUITY.fillna(0.0, inplace=True)
    df.AMT_GOODS_PRICE.fillna(0.0, inplace=True)
    #df.NAME_TYPE_SUITE.fillna('Unknown', inplace=True)
    df.OWN_CAR_AGE.fillna(0, inplace=True)
    #df.OCCUPATION_TYPE.fillna('Unknown', inplace=True)
    df.CNT_FAM_MEMBERS.fillna(0, inplace=True)
    df.fillna(0, inplace=True)
    
    #Hot encode the columns with string/categorical data.
    string_columns = df.select_dtypes(include='O').columns
    onehot_df = df[string_columns]
    df.drop(string_columns, axis=1, inplace=True)
    onehots = list()
    onehots.append(df)
    for c in string_columns:
        onehots.append(pd.get_dummies(onehot_df[c]))
    master = pd.concat(onehots,axis=1)
    return master
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
test['TARGET'] = 2
frames =[train, test]
data = pd.concat(frames)
master = prep_data(data)
master_train = master[master['TARGET']!=2]
master_test = master[master['TARGET']==2]
master_test = master_test.drop(['TARGET'], axis=1)
#master_train.to_csv('train.csv', index=False)
#master_test.to_csv('test.csv', index=False)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
#master_train = pd.read_csv('train.csv')
#master_test = pd.read_csv('test.csv')
y = master_train['TARGET']
X = master_train.drop(['TARGET'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
lr = LogisticRegression()
X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(X_train, y_train)
lr.fit(X_resampled, y_resampled)
predicted = lr.predict(X_test)
probs = lr.predict_proba(X_test)
print('Accuracy: ', accuracy_score(y_test, predicted))
print('ROC: ', roc_auc_score(y_test, probs[:,1]))
print(confusion_matrix(y_test,predicted))
print(classification_report(y_test, predicted))
probs = lr.predict_proba(master_test)
predictions = pd.DataFrame(master_test['SK_ID_CURR'])
predictions['TARGET'] = probs[:,1]
predictions.head()
#predictions.to_csv('predictions.csv', index=False)