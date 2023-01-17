import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("../input/train.csv")
df.replace('male',0,inplace= True)
df.replace('female',1,inplace= True)
for dataset in range(len(df)):
    df['title']=df['Name'].str.extract('([A-Za-z]+)\.',expand=False)
df['title'].value_counts()
title_mapping={"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Col":3,"Mlle":3,"Major":3,"Don":3,"Lady":3,"Capt":3,"Jonkheer":3,"Sir":3,"Countess":3,"Mme":3,"Ms":3,"Name":3}
df['title']=df['title'].map(title_mapping)
df.isnull().any(axis=0)
df['Age'].fillna(df.groupby('title')['Age'].transform("median"),inplace=True)
df.loc[df['Age']<=16,'Age']=0
df.loc[(df['Age'] >16) & (df['Age']<=26),'Age']=1
df.loc[(df['Age'] >26)&( df['Age']<=36 ),'Age']=2
df.loc[(df['Age']>36)&( df['Age'] <=62),'Age']=3
df.loc[df['Age']>62,'Age']=4
df.loc[df['Fare']<=17,'Fare']=0
df.loc[(df['Fare'] >17) & (df['Fare']<=30),'Fare']=1
df.loc[(df['Fare'] >30)&( df['Fare']<=100 ),'Fare']=2
df.loc[df['Fare']>100,'Fare']=3
df['Embarked'].fillna('S',inplace=True)
del df['Cabin']
df['Familysize']=df['SibSp']+df['Parch']+1
del df['Ticket']
del df['PassengerId']
del df['SibSp']
del df['Parch']
del df['Name']
embarked_mapping={'S':0,'C':1,'Q':2}
df['Embarked']=df['Embarked'].map(embarked_mapping)
family_size={0:0.0,1:0.2,2:0.4,3:0.8,4:1.2,5:1.6,6:2.0,7:2.4,8:2.8,9:3.2,10:3.6,11:4.0}
df['Familysize']=df['Familysize'].map(family_size)
df.isnull().any(axis=0)
test=pd.read_csv("../input/test.csv")
test.replace('male',0,inplace= True)
test.replace('female',1,inplace= True)
for dataset in range(len(test)):
    test['title']=test['Name'].str.extract('([A-Za-z]+)\.',expand=False)
test['title'].value_counts()
title_mapping={"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Col":3,"Mlle":3,"Major":3,"Dona":3,"Lady":3,"Capt":3,"Jonkheer":3,"Sir":3,"Countess":3,"Mme":3,"Ms":3,"Name":3}
test['title']=test['title'].map(title_mapping)
test.isnull().any(axis=0)
test['Age'].fillna(test.groupby('title')['Age'].transform("median"),inplace=True)
test.loc[test['Age']<=16,'Age']=0
test.loc[(test['Age'] >16) & (test['Age']<=26),'Age']=1
test.loc[(test['Age'] >26)&( test['Age']<=36 ),'Age']=2
test.loc[(test['Age']>36)&( test['Age'] <=62),'Age']=3
test.loc[test['Age']>62,'Age']=4
test['Fare'].fillna(test.groupby('title')['Fare'].transform("median"),inplace=True)
test.loc[test['Fare']<=17,'Fare']=0
test.loc[(test['Fare'] >17) & (test['Fare']<=30),'Fare']=1
test.loc[(df['Fare'] >30)&( test['Fare']<=100 ),'Fare']=2
test.loc[test['Fare']>100,'Fare']=3
test['Familysize']=test['SibSp']+test['Parch']+1
del test['Ticket']
del test['SibSp']
del test['Parch']
del test['Name']

del test['Cabin']
test['Embarked'].fillna('S',inplace=True)
embarked_mapping={'S':0,'C':1,'Q':2}
test['Embarked']=test['Embarked'].map(embarked_mapping)
family_size={0:0,1:0.2,2:0.4,3:0.8,4:1.2,5:1.6,6:2.0,7:2.4,8:2.8,9:3.2,10:3.6,11:4.0,12:4.5,13:5.0,14:5.5,15:6.0}
test['Familysize']=test['Familysize'].map(family_size)
test.isnull().any(axis=0)
feature_column_names=['Pclass','Sex','Age','Fare','Embarked','title','Familysize']
predicted_class_names=['Survived']
x=df[feature_column_names].values
y=df[predicted_class_names].values
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
rf_model=RandomForestClassifier(n_estimators=13,random_state=42)  #create random object
#score=cross_val_score(rf_model,x,y,cv=10,n_jobs=1)
#print(score)
rf_model.fit(x,y.ravel())
rf_train_predict=rf_model.predict(x)
print("confusion_matrix")
print(confusion_matrix(y,rf_train_predict))
print("")
print("classification report")
print(classification_report(y,rf_train_predict))
print(accuracy_score(y,rf_train_predict))
x1=test[feature_column_names].values
rf_test_predict=rf_model.predict(x1)
submission=pd.DataFrame({"PassengerId":test['PassengerId'],"Survived":rf_test_predict})
submission.to_csv('submission.csv',index=False)
submission=pd.read_csv('submission.csv')
