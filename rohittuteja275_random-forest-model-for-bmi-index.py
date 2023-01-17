import numpy as np 

import pandas as pd
data=pd.read_csv('../input/500_Person_Gender_Height_Weight_Index.csv')
data.head()
data.dtypes
data.shape
data.isnull().sum()
data.Gender.describe()
X=data.drop(['Index'],axis=1)

X=pd.get_dummies(X)

y=data['Index']
X.head()
y.head()
X.dtypes
import sklearn.model_selection as model_selection

X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.3,random_state=100)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.ensemble import RandomForestClassifier

for i in range(10,100,10):

    reg=RandomForestClassifier(n_estimators=i,max_depth=5,max_features='sqrt',oob_score=True,random_state=100)

    reg.fit(X_train,y_train)

    oob=reg.oob_score_

    print('For n_estimators = '+str(i))

    print('OOB score is '+str(oob))

    print('************************')
reg=RandomForestClassifier(n_estimators=50,max_depth=5,max_features='sqrt',oob_score=True,random_state=100)

reg.fit(X_train,y_train)
reg.score(X_test,y_test)
reg.oob_score_
feature_importance=reg.feature_importances_
feat_imp=pd.Series(feature_importance,index=X.columns.tolist())
feat_imp.sort_values(ascending=False)
pred=reg.predict(X_test)
