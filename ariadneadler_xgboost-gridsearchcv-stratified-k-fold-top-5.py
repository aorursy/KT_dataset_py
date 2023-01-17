import numpy as np
import pandas as pd
import re as re

train1 = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
train=train1.drop(columns=['Survived'])
allfeat = pd.concat([train, test],axis=0)
print (train.info())
print (train.shape)
print (test.shape)
allfeat=allfeat.drop(columns=['PassengerId','Cabin','Ticket'])
allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['Pclass'])], axis=1) #getting one hot encoding for PClass and concatenating as new columns
allfeat=allfeat.drop(columns=['Pclass']) #column no longer needed
allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['Sex'])], axis=1) #getting one hot encoding for Sex and concatenating as new columns
allfeat=allfeat.drop(columns=['Sex']) #column no longer needed
allfeat['FamilySize'] = allfeat['SibSp'] + allfeat['Parch'] + 1
allfeat=allfeat.drop(columns=['SibSp','Parch']) #column no longer needed
allfeat['IsAlone'] = 0
allfeat.loc[allfeat['FamilySize'] == 1, 'IsAlone'] = 1
allfeat['Embarked'] = allfeat['Embarked'].fillna('S')
allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['Embarked'])],axis=1) #one-hot encoding the embarked categories
allfeat=allfeat.drop(columns='Embarked') #column no longer needed
allfeat['Fare'] = allfeat['Fare'].fillna(train['Fare'].median())
allfeat['CategoricalFare'] = pd.qcut(allfeat['Fare'], 4)
allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['CategoricalFare'])],axis=1) #one-hot encoding the fare categories
allfeat=allfeat.drop(columns=['Fare','CategoricalFare']) #column no longer needed
avg=allfeat['Age'].mean()
std=allfeat['Age'].std()

allfeat['Age']=allfeat['Age'].fillna(value=np.random.randint(avg-std,avg+std))
allfeat['Age'] = allfeat['Age'].astype(int)
    
allfeat['CategoricalAge'] = pd.cut(allfeat['Age'], 5)
allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['CategoricalAge'])],axis=1) #one-hot encoding the age categories
allfeat=allfeat.drop(columns=['Age','CategoricalAge']) #column no longer needed
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

allfeat['Title'] = allfeat['Name'].apply(get_title)
allfeat=allfeat.drop(columns=['Name']) #column no longer needed
allfeat['Title'] = allfeat['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

allfeat['Title'] = allfeat['Title'].replace('Mlle', 'Miss')
allfeat['Title'] = allfeat['Title'].replace('Ms', 'Miss')
allfeat['Title'] = allfeat['Title'].replace('Mme', 'Mrs')
    
allfeat=pd.concat([allfeat,pd.get_dummies(allfeat['Title'])],axis=1) #one-hot encoding the Title categories
allfeat=allfeat.drop(columns='Title') #column no longer needed
print (list(allfeat))
#for xgboost classifier to work, we must rename the columns, removing the header names containing '()' and '[]'
allfeat.columns=['1', '2', '3', 'female', 'male', 'FamSize', 'IsAlone', 'C', 'Q', 'S', 'fare1', 'fare2', 'fare3', 'fare4', \
                 'age1', 'age2', 'age3', 'age4', 'age5', 'Master', 'Miss', 'Mr', 'Mrs', 'Rare']
print (list(allfeat))

#now divide engineered dataset into train and test dataset
X=allfeat[:][0:891]
testdf=allfeat[:][891:1309]
y=train1['Survived']
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

param_grid = [{'min_child_weight': np.arange(0.1, 10.1, 0.1)}] #set of trial values for min_child_weight
i=1
kf = StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
for train_index,test_index in kf.split(X,y):
     print('\n{} of kfold {}'.format(i,kf.n_splits))
     xtr,xvl = X.loc[train_index],X.loc[test_index]
     ytr,yvl = y[train_index],y[test_index]
     model = GridSearchCV(XGBClassifier(), param_grid, cv=10, scoring= 'f1',iid=True)
     model.fit(xtr, ytr)
     print (model.best_params_)
     pred=model.predict(xvl)
     print('accuracy_score',accuracy_score(yvl,pred))
     i+=1

op=pd.DataFrame(data={'PassengerId':test['PassengerId'],'Survived':model.predict(testdf)})
op.to_csv('KFold_XGB_GridSearchCV_submission.csv',index=False)