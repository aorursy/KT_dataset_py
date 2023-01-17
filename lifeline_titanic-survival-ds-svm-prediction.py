import pandas as pd
import numpy as nm
import matplotlib as plt
import warnings


import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib import pyplot
import seaborn as sns
%matplotlib inline 
warnings.filterwarnings("ignore")
titan_ds = pd.read_csv("/kaggle/input/titanic/train.csv")
titan_ds.head()
titan_ds.shape
titan_ds.dtypes
titan_ds.columns
pd.value_counts(titan_ds['Survived'])
#replace the nans and nulls with median
median = round(titan_ds['Age'].median())
titan_ds['Age'] = titan_ds['Age'].fillna(value=median)
#Fill the cabin data in binary
titan_ds['Cabin'] = titan_ds["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
#Fill the Sex data in binary
titan_ds['Sex'] = titan_ds['Sex'].apply(lambda x : 0 if x=='male' else 1)
titan_ds['FamilySize']=titan_ds['SibSp'] + titan_ds['Parch'] + 1
#create a new column for 'is_Alone' and fill if travelled alone. 
titan_ds['Is_Alone'] =titan_ds['FamilySize']==1
titan_ds['Is_Alone'] = titan_ds['Is_Alone'].apply(lambda x: 1 if x==1 else 0)
#Drop not required variables
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Embarked']
titan_ds = titan_ds.drop(drop_elements, axis = 1)
pd.value_counts(titan_ds['Sex'])
print('missing values ',titan_ds.isna().sum())
print('missing values ',titan_ds.isnull().sum())
titan_ds.columns
len(titan_ds[(titan_ds['Is_Alone'] == 1 ) & (titan_ds['Survived'] == 1)])/ len(titan_ds[(titan_ds['Is_Alone'] == 1 )])
len(titan_ds[(titan_ds['Is_Alone'] != 1 ) & (titan_ds['Survived'] == 1) ])/ len(titan_ds[(titan_ds['Is_Alone'] != 1 )])
len(titan_ds[ (titan_ds['Survived'] == 1) & (titan_ds['Sex'] == 1)])/ len(titan_ds[ (titan_ds['Sex'] == 1)])
len(titan_ds[(titan_ds['Is_Alone'] == 1 ) & (titan_ds['Survived'] == 1) & (titan_ds['Sex'] == 1)])/ len(titan_ds[(titan_ds['Sex'] == 1) & (titan_ds['Is_Alone'] == 1 )])
len(titan_ds[(titan_ds['Is_Alone'] != 1 ) & (titan_ds['Survived'] == 1) & (titan_ds['Sex'] == 1)])/ len(titan_ds[(titan_ds['Sex'] == 1) & (titan_ds['Is_Alone'] != 1 )])
##We could imagine women are 
len(titan_ds[(titan_ds['Is_Alone'] != 1 ) & (titan_ds['Survived'] == 1) & (titan_ds['Sex'] == 0)])/ len(titan_ds[(titan_ds['Sex'] == 0) & (titan_ds['Is_Alone'] != 1 )])
##We could imagine women are 
len(titan_ds[(titan_ds['Is_Alone'] == 1 ) & (titan_ds['Survived'] == 1) & (titan_ds['Sex'] == 0)])/ len(titan_ds[(titan_ds['Sex'] == 0) & (titan_ds['Is_Alone'] == 1 )])
plt.figure(figsize=(10,6))
ax=sns.boxplot('Sex','FamilySize',data=titan_ds,hue='Survived')
ax.set(xticklabels=['Male','Female']);
ax=sns.factorplot('Sex','FamilySize',data=titan_ds,hue='Survived')
ax.set(xticklabels=['Male','Female']);
sns.stripplot(titan_ds['Survived'],titan_ds['Age']);
sns.swarmplot(titan_ds['Survived'],titan_ds['Age']);
plt.figure(figsize=(10,5))
sns.heatmap(titan_ds.corr(), cmap='YlGnBu', annot=True)
plt.title("Titan survival heatmap")
plt.show();
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import numpy as np


#Get x and Y
X_train,y_train = np.array(titan_ds)[ :, 2:9], np.array(titan_ds.Survived)[:]

# Building a Support Vector Machine on train data
svc_model = SVC(kernel='linear')
svc_model.fit(X_train, y_train)

print(svc_model.score(X_train, y_train))
titan_test_ds= pd.read_csv("/kaggle/input/titanic/test.csv")
passenger_ids= titan_test_ds['PassengerId']
passenger_ids = passenger_ids.dropna()

#replace the nans and nulls with median
median = round(titan_test_ds['Age'].median())
titan_test_ds['Age'] = titan_test_ds['Age'].fillna(value=median)

titan_test_ds['Fare'] = titan_test_ds['Fare'].fillna(value=median)
#Fill the cabin data in binary
titan_test_ds['Cabin'] = titan_test_ds["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
#Fill the Sex data in binary
titan_test_ds['Sex'] = titan_test_ds['Sex'].apply(lambda x : 0 if x=='male' else 1)
titan_test_ds['FamilySize']=titan_test_ds['SibSp'] + titan_test_ds['Parch'] + 1
#create a new column for 'is_Alone' and fill if travelled alone. 
titan_test_ds['Is_Alone'] =titan_test_ds['FamilySize']==1
titan_test_ds['Is_Alone'] = titan_test_ds['Is_Alone'].apply(lambda x: 1 if x==1 else 0)
#Drop not required variables
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Embarked']
titan_test_ds = titan_test_ds.drop(drop_elements, axis = 1)
predictions = svc_model.predict(titan_test_ds.iloc[:,1:])
output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
output.to_csv('titanic_prediction.csv', index=False)