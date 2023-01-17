from pandas import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from matplotlib.pyplot import *
%matplotlib inline
from sklearn.model_selection import train_test_split
from seaborn import *
from sklearn.metrics import accuracy_score 
data_train = read_csv('../input/train.csv')
data_test = read_csv('../input/test.csv')
test_data = data_test

x = DataFrame(data_train.drop(['Survived'],axis=1))
y = DataFrame(data_train['Survived'])
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

le = LabelEncoder()
i = 1
for dataset in [x_train , x_test,data_test]:
    dataset.loc[:,'Age'] = dataset.loc[:,'Age'].fillna(dataset.loc[:,'Age'].mean()) 
    dataset.loc[:,'Fare'] = dataset.loc[:,'Fare'].fillna(dataset.loc[:,'Fare'].mean())
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.',expand = False)
    dataset['Sex'] = le.fit_transform(dataset['Sex'])
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset.loc[dataset['Age'] <= 16,'Age'] = 1
    dataset.loc[(dataset['Age']>16) & (dataset['Age']<=40),'Age'] = 2
    dataset.loc[(dataset['Age']>40) & (dataset['Age']<=64),'Age'] = 3
    dataset.loc[(dataset['Age']>64),'Age'] = 4
    dataset.loc[dataset['Fare']<=70.0,'Fare'] = 1
    dataset.loc[(dataset['Fare']>70.0) & (dataset['Fare'] <= 250),'Fare'] = 2
    dataset.loc[dataset['Fare']>250.0,'Fare'] = 3
    dataset = dataset.drop(['SibSp','Parch','Embarked','Cabin','Name','Ticket','PassengerId'],axis=1)
    dataset['Title'] = le.fit_transform(dataset['Title'].astype(str))
    dataset['Age'] = dataset['Age'].astype(int)
    dataset['Fare'] = dataset['Fare'].astype(int)
    if i == 1:
            x_train = dataset
            i = 2
    elif i==2:
            x_test = dataset
            i = 3
    elif i==3:
            data_test = dataset
rfc = RandomForestClassifier(random_state=20)

rfc.fit(x_train,y_train)
pred_val = rfc.predict(x_test)
pred_val1 = rfc.predict(data_test)
accuracy_score(pred_val,y_test)
submission = DataFrame({"PassengerId":test_data['PassengerId'],'Survived':pred_val1})

submission.to_csv('submission1.csv',index = False)