import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
import seaborn as sns
train_data=pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
imp=SimpleImputer(missing_values=np.nan)
train_data['Age']=imp.fit_transform(train_data[['Age']])
train_data['Sex']=pd.get_dummies(train_data.Sex)

imp=SimpleImputer(missing_values=np.nan)
test_data['Age']=imp.fit_transform(test_data[['Age']])
test_data['Fare']=imp.fit_transform(test_data[['Fare']])
test_data['Sex']=pd.get_dummies(test_data.Sex)
train_data.dropna(subset=['Embarked'], how='any', inplace=True)
train_data=pd.concat([train_data,pd.get_dummies(train_data['Embarked'],prefix='embarked')],axis=1)
train_data.drop(['Embarked'],axis=1,inplace=True)
test_data=pd.concat([test_data,pd.get_dummies(test_data['Embarked'],prefix='embarked')],axis=1)
#test_data.drop(['Embarked'],axis=1,inplace=True)
train_data['FamilySize']=train_data['SibSp'] + train_data['Parch'].shift(0,fill_value=0) +1
test_data['FamilySize']=test_data['SibSp'] + test_data['Parch'].shift(0,fill_value=0) +1
train_data['age_status'] = train_data["Age"].apply(lambda x: "bebek" if x<3 else "cocuk" if (x>=3 and x<13) else "genc" if (x>=13 and x<30) else "olgun" if (x>=30 and x<50) else "yasli") 
#train_data['age_status'] = train_data["age_status"].apply(lambda x: 0 if x=="bebek" else 2 if x=="çocuk" else 2 if x=="genç" else 3 if x=="olgun" else 4) 
test_data['age_status'] = test_data["Age"].apply(lambda x: "bebek" if x<3 else "cocuk" if (x>=3 and x<13) else "genc" if (x>=13 and x<30) else "olgun" if (x>=30 and x<50) else "yasli") 
#test_data['age_status'] = test_data["age_status"].apply(lambda x: 0 if x=="bebek" else 2 if x=="çocuk" else 2 if x=="genç" else 3 if x=="olgun" else 4) 
train_data=pd.concat([train_data,pd.get_dummies(train_data['age_status'],prefix='age_status').groupby(level=0).sum()],axis=1)
train_data.drop(['age_status'],axis=1,inplace=True)
test_data=pd.concat([test_data,pd.get_dummies(test_data['age_status'],prefix='age_status').groupby(level=0).sum()],axis=1)
test_data.drop(['age_status'],axis=1,inplace=True)
train_data.head(2)
test_data.head(2)
features = ['Pclass', 'Sex','Age','Fare','embarked_C','embarked_Q','embarked_S','FamilySize','age_status_bebek','age_status_cocuk','age_status_genc','age_status_olgun','age_status_yasli',]
X=train_data[features]
y=train_data['Survived']
X.shape
X
X.Age=X.Age.round(1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
(X_train.shape,X_test.shape)
parameters = [ {"n_estimators":[5,10,15,20,25,30,35,40], "criterion":["entropy", "gini"], "max_depth":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],"max_features":["sqrt","log2"],"random_state":[0]}]
grid_search_c = GridSearchCV(RandomForestClassifier(), parameters, scoring = "accuracy")
grid_search_c.fit(X_train, y_train)
grid_search_c.best_estimator_
grid_search_c.best_score_
model=RandomForestClassifier(n_estimators=24,
                             criterion="entropy",
                             max_depth=16,
                             max_features='sqrt',
                             random_state=0)
model.fit(X,y)
model.score(X,y)


y_pred = model.predict(X_test)
print(model.__class__.__name__, accuracy_score(y_test, y_pred))
results = confusion_matrix(y_test, y_pred.round())
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(y_test, y_pred.round()) )
print ('Report : ')
print (classification_report(y_test, y_pred.round()) )
pred=model.predict(test_data[features])
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': pred})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")