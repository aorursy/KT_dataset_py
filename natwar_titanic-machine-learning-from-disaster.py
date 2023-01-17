import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
train_data=pd.read_csv('../input/titanic/train.csv')
train_data.head()
train_data.tail()
male=train_data.loc[train_data.Sex=='male']['Survived']
rate_male=sum(male)/len(male)
rate_male
female=train_data.loc[train_data.Sex=='female']['Survived']
rate_female=sum(female)/len(female)
rate_female
test_data=pd.read_csv('../input/titanic/test.csv')
test_data.head()
from sklearn.ensemble import RandomForestClassifier
y=train_data['Survived']
features=['Pclass','Sex','SibSp','Parch']
X=pd.get_dummies(train_data[features])
X_test=pd.get_dummies(test_data[features])
model=RandomForestClassifier(n_estimators=100,max_depth=5, random_state=1)
model.fit(X,y)
predictions=model.predict(X_test)
output=pd.DataFrame({'PassengerId':test_data.PassengerId,'Survived':predictions})
output.to_csv('my_submission.csv',index=False)
