

import numpy as np 

import pandas as pd 



bank_full='/kaggle/input/bank-direct-marketing/bank-full.csv'

bank='/kaggle/input/bank-direct-marketing/bank.csv'



data_train=pd.read_csv(bank_full,sep=';')

data_test=pd.read_csv(bank,sep=';')



data_train.head()

#are there any empty values

data_train.dropna(axis=1)


from sklearn import preprocessing

import sklearn as sk

columns=['job','marital','education','default','housing','loan','contact','month','poutcome']

for i in range(len(columns)):

    str_t=sk.preprocessing.LabelEncoder()  #encoding strings into numbers

    str_t.fit(data_train[columns[i]])

    data_train[columns[i]]=str_t.transform(data_train[columns[i]])



a=list(range(data_train.shape[1]-1))



x_train=data_train.iloc[1:,a].values



y_train=data_train.iloc[1:,16].values



y_train=np.where(y_train=='yes',1,0)



from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt





classifier_inform=RandomForestClassifier(criterion='entropy',n_estimators=20,n_jobs=-1)



classifier_inform.fit(x_train,y_train)



inform=classifier_inform.feature_importances_  #informational content of signs



plt.bar(range(x_train.shape[1]),inform,color='blue',align='center')



plt.show()
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report





x_train=data_train.iloc[1:,[0,5,9,10,11]].values #the most informative signs



classifier=RandomForestClassifier(criterion='entropy',n_estimators=20,n_jobs=-1)



classifier.fit(x_train,y_train)



pred=classifier.predict(x_train)



print('TRAIN ACCURACY',accuracy_score(pred,y_train))



targets=['no','yes']



print(classification_report(y_train,pred,target_names=targets))
for i in range(len(columns)):

    str_t=sk.preprocessing.LabelEncoder()

    str_t.fit(data_test[columns[i]])

    data_test[columns[i]]=str_t.transform(data_test[columns[i]])

    

x_test=data_test.iloc[1:,[0,5,9,10,11]].values   



y_test=data_test.iloc[1:,16].values



y_test=np.where(y_test=='yes',1,0)
pred=classifier.predict(x_test)



print('TEST ACCURACY',accuracy_score(pred,y_test))



print(classification_report(y_test,pred,target_names=targets))

from sklearn import linear_model
classif_logit=linear_model.LogisticRegression(penalty='l2',tol=0.0001,solver='saga',C=2.0,max_iter=600)



classif_logit.fit(x_train, y_train)



pred_log_train=classif_logit.predict(x_train)



print('TRAIN ACCURACY',accuracy_score(pred_log_train,y_train))



print(classification_report(y_train,pred_log_train,target_names=targets))

pred_log_test=classif_logit.predict(x_test)



print('TEST ACCURACY',accuracy_score(pred_log_test,y_test))



print(classification_report(y_test,pred_log_test,target_names=targets))
