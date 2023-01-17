import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
train.head()
train['Embarked']=train['Embarked'].fillna('S')
train['middle']=train['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
train['middle'].value_counts()
test['middle']=test['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
test['middle'].value_counts()
for i in range(len(train)):
        
    if pd.isnull(train['Age'][i]):
        name=train['middle'][i]
        if name=='Mr':
            train['Age'][i]=33
        elif name=='Mrs':
            train['Age'][i]=36
        elif name=='Miss':
            train['Age'][i]=22
        elif name=='Master':
            train['Age'][i]=5
        else:
            train['Age'][i]=30

train['Age'].isnull().sum()
for i in range(len(test)):
    if pd.isnull(test['Age'][i]):
        name=test['middle'][i]
        if name=='Mr':
            test['Age'][i]=33
        elif name=='Mrs':
            test['Age'][i]=36
        elif name=='Miss':
            test['Age'][i]=22
        elif name=='Master':
            test['Age'][i]=5
        else:
            test['Age'][i]=30

test['Age'].isnull().sum()
all_set=pd.concat(objs=[train,test],axis=0).reset_index(drop=True)
train.info()
test.info()
all_set.info()
all_set['Cabin'].head(10)
train['Cabin']=pd.DataFrame([i[0] if not pd.isnull(i) else 'N'] for i in all_set['Cabin'])
train['Cabin'].head()
test['Cabin']=pd.DataFrame([i[0] if not pd.isnull(i) else 'N'] for i in all_set['Cabin'])
test['Cabin'].head()
sns.countplot(train['Cabin'],data=all_set);
sns.countplot(x='Cabin',hue='Survived',data=train);
all_set['Ticket'].head()
Ticket=[]

for i in train.Ticket:
    if i.isdigit():
        Ticket.append('Num')
    else:
        Ticket.append(i[0])
#print(*Ticket)
train['Ticket_Cat']=Ticket
train['Ticket_Cat'].head()
Ticket=[]

for i in test.Ticket:
    if i.isdigit():
        Ticket.append('Num')
    else:
        Ticket.append(i[0])
#print(*Ticket)
test['Ticket_Cat']=Ticket
test['Ticket_Cat'].head()
sns.countplot(x='Ticket_Cat',hue='Survived',data=train);
train.info()
train['Family']=train['SibSp']+train['Parch']+1
test['Family'] = test['SibSp'] + test['Parch'] + 1
train['Family']=pd.cut(train.Family, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])
test['Family']= pd.cut(test.Family, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])
train.head()
y=train['Survived']
features=['Pclass','Fare','Sex','Age','Ticket_Cat','Embarked','Family','Cabin']
X=train[features]
X.tail()
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
numerical_cols=['Age','Fare']

categorical_cols=['Pclass','Sex','Ticket_Cat','Embarked','Family','Cabin']

numerical_transformer = SimpleImputer(strategy='median')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestClassifier(random_state=0, 
                                                               n_estimators=600, max_depth=5))
                             ])
model.fit(X,y);
X_test=test[features]
X.head()
preds=model.predict(X_test)
result = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': preds}) 
result.to_csv('submission.csv', index=False)
