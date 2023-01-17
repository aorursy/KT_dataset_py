import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
tr_data=pd.read_csv('train.csv')
tr_data['Sex']=tr_data['Sex'].replace(['male'],1)
tr_data['Sex']=tr_data['Sex'].replace(['female'],-1)
tr_data=tr_data.drop(columns=['Name','Ticket','Cabin','Embarked'])
tr_data.set_index('PassengerId', inplace=True)

test_data=pd.read_csv('test.csv')
test_data['Sex']=test_data['Sex'].replace(['male'],1)
test_data['Sex']=test_data['Sex'].replace(['female'],-1)
test_data=test_data.drop(columns=['Name','Ticket','Cabin','Embarked'])
test_data.set_index('PassengerId', inplace=True)
tr_data=tr_data.sample(frac=1)
tr_1=tr_data.dropna()
tr_2=tr_data.drop(columns='Age')
tr_3=tr_1.drop(columns='Fare')
te_1=test_data.dropna()
te_2pre=test_data[~test_data.index.isin(te_1.index)]
te_2=te_2pre.drop(columns='Age').dropna()
te_3=te_2pre.drop(columns='Fare').dropna()
model1 =RandomForestClassifier(n_estimators=10000)
model2 =RandomForestClassifier(n_estimators=10000)
model3 =RandomForestClassifier(n_estimators=10000)

X_1=tr_1.drop(columns='Survived')
Y_1=tr_1['Survived']
X_2=tr_2.drop(columns='Survived')
Y_2=tr_2['Survived']
X_3=tr_3.drop(columns='Survived')
Y_3=tr_3['Survived']

model1.fit(X_1,Y_1)
predict1= model1.predict(te_1)
model2.fit(X_2,Y_2)
predict2= model2.predict(te_2)
model3.fit(X_3,Y_3)
predict3= model3.predict(te_3)
te_1['Survived']=predict1
te_2['Survived']=predict2
te_3['Survived']=predict3
te_1=te_1.reset_index()[['PassengerId','Survived']]
te_2=te_2.reset_index()[['PassengerId','Survived']]
te_3=te_3.reset_index()[['PassengerId','Survived']]
test_pred=te_1.merge(te_2,how='outer').merge(te_3,how='outer')
test_pred=test_pred.sort_values(by='PassengerId')
test_pred.set_index('PassengerId', inplace=True)
test_pred.to_csv('predictions.csv')
def K_foldX(data, model, K):
    data_size=data.shape[0]
    ind=np.random.choice(data_size, data_size, replace=False)
    E_new=np.zeros(K)
    
    for i in range(K):
        index=ind[int(i*data_size/K):int((i+1)*data_size/K)]
        
        train=data.iloc[~data.index.isin(ind[index])]
        validation=data.iloc[ind[index]]
        
        Y_train = train['Survived']
        X_train = train.drop(columns='Survived')
    
        X_validation = validation.drop(columns='Survived')
        Y_validation = validation['Survived']
        model.fit(X_train,Y_train)
        predict= model.predict(X_validation)
        E_new[i]=np.mean(predict==Y_validation)
        
    return E_new