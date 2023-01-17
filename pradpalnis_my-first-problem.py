import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

data = pd.read_csv('../input/train.csv')
data.head()
data.shape
data.PassengerId.unique().shape
data.Sex.unique()
fig = data['Sex'].value_counts().plot.bar()
fig.set_title('Sex')
fig.set_ylabel('No of passengars')
data.Pclass.unique()
fig = data.Pclass.value_counts().plot.bar()
fig.set_title('Pclass')
fig.set_ylabel('No of passengers')
data.Fare.unique()
fig = data.Fare.hist(bins=50)
fig.set_title('Fare')
fig.set_xlabel('Fare Amt')
fig.set_ylabel('No of Passengers')
data.Age.unique()
fig = data.Age.hist(bins=50)
fig.set_title('Age')
fig.set_xlabel('Age of Passengers')
fig.set_ylabel('No of Passengers')
data.SibSp.unique() #Discrete Feature
fig = data.SibSp.hist(bins=50)
fig.set_title('SibSp')
fig.set_xlabel('SibSp count')
fig.set_ylabel('No of Passengers')
data.Parch.unique() # Discrete Feature
fig = data.Parch.hist(bins=50)
fig.set_title('Parch')
fig.set_xlabel('Parch Count')
fig.set_ylabel('No Of Passengers')
data.Embarked.unique() # Discrete Feature
fig = data.Embarked.value_counts().plot.bar()
fig.set_title('Embarked')
fig.set_ylabel('No of Passengers')
data.Cabin.unique()
fig = data.Cabin.value_counts().plot.bar()
fig.set_title('Cabin')
fig.set_ylabel('No of Passengers')
data.Survived.unique()
fig = data.Survived.value_counts().plot.bar()
fig.set_title('Survived')
fig.set_ylabel('No of Passengers')
survived_sex = data[data.Survived==1]['Sex'].value_counts()
not_survived_sex = data[data.Survived==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,not_survived_sex])
df.index =['survived_sex','not_survived_sex']
df.plot(kind='bar')
survived_pclass = data[data.Survived==1]['Pclass'].value_counts()
not_survived_pclass = data[data.Survived==0]['Pclass'].value_counts()
df = pd.DataFrame([survived_pclass,not_survived_pclass])
df.index =['survived_pclass','not_survived_pclass']
df.plot(kind='bar')
survived_sibsp = data[data.Survived==1]['SibSp'].value_counts()
not_survived_sibsp = data[data.Survived==0]['SibSp'].value_counts()
df = pd.DataFrame([survived_sibsp,not_survived_sibsp])
df.index =['survived_sibsp','survived_sibsp']
df.plot(kind='bar')
survived_parch = data[data.Survived==1]['Parch'].value_counts()
not_survived_parch = data[data.Survived==0]['Parch'].value_counts()
df = pd.DataFrame([survived_parch,not_survived_parch])
df.index =['survived_parch','not_survived_parch']
df.plot(kind='bar')
survived_embarked = data[data.Survived==1]['Embarked'].value_counts()
not_survived_embarked = data[data.Survived==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embarked,not_survived_embarked])
df.index =['survived_embarked','not_survived_embarked']
df.plot(kind='bar')
fig = plt.figure()
data1 = data.dropna() #error withou drop NA
plt.hist([data1[data1['Survived']==1]['Age'],data1[data1['Survived']==0]['Age']],bins=30,label=['Survived','Not_Survived'])
plt.xlabel('Age')
plt.ylabel('No of passengers')
plt.legend()
ig = plt.figure()
data1 = data.dropna() #error withou drop NA
plt.hist([data1[data1['Survived']==1]['Fare'],data1[data1['Survived']==0]['Fare']],bins=30,label=['Survived','Not_Survived'])
plt.xlabel('Fare')
plt.ylabel('No of passengers')
plt.legend()
def combine_data():
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    
    y = train.Survived
    train.drop('Survived',axis = 1,inplace = True)
    train.drop('PassengerId',axis = 1,inplace = True)
    test.drop('PassengerId',axis = 1,inplace = True)# High cordinality , remove
    train.drop('Ticket',axis = 1,inplace = True)
    test.drop('Ticket',axis = 1,inplace = True)# High cordinality , remove
    combined = train.append(test)
    combined.reset_index(inplace = True)
    combined.drop('index',axis =1,inplace = True)
    return combined
    
combined_data = combine_data()
print(combined_data.head())    
def get_Name_title():
    global combined_data
    combined_data['Name_title'] = combined_data['Name'].map(lambda name : name.split(',')[1].split('.')[0].strip())
    combined_data.drop('Name',axis =1,inplace = True)
    return combined_data['Name_title']
get_Name_title()

combined_data.head()
fig = combined_data.Name_title.value_counts().plot.bar()
fig.set_title('Name_title')
fig.set_ylabel('No of Passengers')
combined_data.info()
combined_data.isnull().sum()
combined_data.isnull().mean()
combined_data.iloc[:891].Age.isnull().sum()
combined_data.iloc[891:].Age.isnull().sum()
combined_data.iloc[891:]['Age'].describe()
combined_data['Age'].fillna(combined_data.iloc[:891]['Age'].mean(),inplace = True)
#combined_data.Age_fill.isnull().sum()
combined_data.iloc[:891]['Age']
combined_data[combined_data['Fare'].isnull()]
combined_data['Fare'].fillna(combined_data.iloc[:891]['Fare'].mean(),inplace = True)
#combined_data[combined_data['Fare_fill'].isnull()]
combined_data['Embarked'].fillna(combined_data.iloc[:891]['Embarked'].value_counts().index[0],inplace =True)
#combined_data[combined_data['Embarked_fill'].isnull()]
def Cabin_fill():
    global combined_data
    combined_data['Cabin'] = combined_data['Cabin'][combined_data['Cabin'].notnull()].map(lambda c:c[0])
    combined_data['Cabin'].fillna(combined_data['Cabin'].value_counts().index[0],inplace = True)
    return combined_data
combined_data = Cabin_fill()
combined_data.head()
name_dummies = pd.get_dummies(combined_data['Name_title'],prefix='Name')
combined_data = pd.concat([combined_data,name_dummies],axis = 1)
combined_data.drop('Name_title',axis=1,inplace=True)
combined_data.head()
embarked_dummies = pd.get_dummies(combined_data['Embarked'],prefix='Embarked')
combined_data = pd.concat([combined_data,embarked_dummies],axis = 1)
combined_data.drop('Embarked',axis=1,inplace=True)
combined_data.head()
cabin_dummies = pd.get_dummies(combined_data['Cabin'],prefix='Cabin')
combined_data = pd.concat([combined_data,cabin_dummies],axis = 1)
combined_data.drop('Cabin',axis=1,inplace=True)
combined_data.head()
pclass_dummies = pd.get_dummies(combined_data['Pclass'],prefix='Pclass')
combined_data = pd.concat([combined_data,pclass_dummies],axis = 1)
combined_data.drop('Pclass',axis=1,inplace=True)
combined_data.head()
sibSp_dummies = pd.get_dummies(combined_data['SibSp'],prefix='SibSp')
combined_data = pd.concat([combined_data,sibSp_dummies],axis = 1)
combined_data.drop('SibSp',axis=1,inplace=True)
combined_data.head()
parch_dummies = pd.get_dummies(combined_data['Parch'],prefix='Parch')
combined_data = pd.concat([combined_data,parch_dummies],axis = 1)
combined_data.drop('Parch',axis=1,inplace=True)
combined_data.head()
combined_data['Sex'] = combined_data['Sex'].map({'male':1,'female':0})
#combined_data['Sex'] = combined_data['Sex'].map(lambda s :if(s == 'male') 1 else 0)
#combined_data.head()
combined_data.head()
from  sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
def split_data():
    global combined_data
    targets = pd.read_csv('../input/train.csv', usecols=['Survived'])['Survived'].values
    train = combined_data.iloc[:891]
    test = combined_data.iloc[891:]
    return train, test, targets
train, test, targets = split_data()
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)
model = LogisticRegression()
score = compute_score(clf = model, X=train,y= targets,scoring = 'accuracy' )
print( 'CV score = {0}'.format(score))

model.fit(train, targets)
output = model.predict(test).astype(int)
test.head()
df_output = pd.DataFrame()
orginal_test = pd.read_csv('../input/test.csv')
df_output['PassengerId'] = orginal_test['PassengerId']
df_output['Survived'] = output
#predictions_df.columns = ['PassengerId', 'Survived']


df_output.to_csv('submission.csv', index=False)