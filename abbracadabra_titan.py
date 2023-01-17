import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

train=pd.read_csv(r'..\input\train.csv')
test=pd.read_csv(r'..\input\test.csv')
all_data = pd.concat([train,test],axis=0,ignore_index=True)

age_nan=np.random.randint(train['Age'].mean()-train['Age'].std(),train['Age'].mean()+train['Age'].std(),all_data['Age'].isnull().sum())
all_data['Age'][all_data['Age'].isnull()]=age_nan

all_data=all_data.drop(['Cabin','Ticket','PassengerId'],axis=1,errors='ignore')
all_data['Fare'][all_data['Fare'].isnull()]=train['Fare'].mean()
all_data['Embarked'][all_data['Embarked'].isnull()]=train['Embarked'].mode()[0]
all_data['Title']=all_data['Name'].apply(lambda name:re.search(r'(?<=,).*?(?=\.)',name).group(0))
all_data=all_data.drop(['Name'],axis=1)
enc=OneHotEncoder()
all_data=pd.concat([all_data,pd.get_dummies(all_data[['Sex','Embarked','Pclass','Title']])],axis=1)
all_data=all_data.drop(['Sex','Embarked','Pclass','Title'],axis=1)

y_train=all_data.loc[all_data['Survived'].notnull(),:]['Survived']
x_train=all_data.loc[all_data['Survived'].notnull(),:].drop(['Survived'],axis=1)
x_train, x_validate, y_train, y_validate=train_test_split(x_train,y_train,test_size=0.05)
x_test=all_data.loc[all_data['Survived'].isnull(),:].drop(['Survived'],axis=1)

xgbclf=xgb.XGBClassifier()
xgbclf.fit(X=x_train,y=y_train,eval_set=[(x_validate,y_validate)])

submission=pd.DataFrame({'PassengerId':np.arange(start=892,stop=892+len(x_test)),'Survived':xgbclf.predict(x_test)})
submission.to_csv(r'mysubmission.csv',index=False)


