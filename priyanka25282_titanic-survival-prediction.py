import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Importing the data and displaying some rows
train= pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
print("Number of data points in train data : ",train.shape[0]) 
print("Number of data points in test data : ",test.shape[0]) 
print("\nNumber of columns in data : ",train.columns) 
print("Sample train datapoint :") 
train.head(5)
print("Sample test datapoint :") 
test.head(5)
class_specific_count = train['Survived'].value_counts()

print("Number of died passengers :", class_specific_count[0],"(",((class_specific_count[0]/len(train))*100),"% )")
print("Number of survived passengers :",class_specific_count[1], "(",((class_specific_count[1]/len(train))*100),"% )\n")

plt.figure(figsize=(5,3))
plt.title("Survival Prediction data")
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Survived',data=train)
plt.show()
#train data
plt.figure(figsize=(5,3))
plt.title("Analysis on Missing data")
sns.heatmap(train.isnull(),yticklabels=False ,cbar=True ,cmap='viridis')
# To find how many null values in some columns
train.info()
#test data
plt.figure(figsize=(5,3))
plt.title("Analysis on Missing data")
sns.heatmap(test.isnull(),yticklabels=False ,cbar=True ,cmap='viridis')
# To find how many null values 
test.info()
# fill NaN values 
train['Age'].fillna(train['Age'].median(),inplace=True)
train['Embarked'].fillna(method='pad',inplace=True)

test['Age'].fillna(test['Age'].median(),inplace=True)
test['Fare'].fillna(test['Fare'].median(),inplace=True)
#drop columns which seems irrelevant 
train = train.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)
test = test.drop(['PassengerId','Name','Cabin','Ticket'],axis=1)
train.head(5)
test.head(5)
# correlation visualisation using heatmap
sns.set(style='darkgrid')
fig=plt.figure(figsize=(5,3))
sns.heatmap(train.corr(),annot=True,linewidths=0.3)
cat_cols = ['Pclass','Sex','Embarked','SibSp','Parch']

num_cols = ['Age','Fare']
for col in cat_cols:
  plt.figure(figsize=(13,8))
  
  plt.subplot(3,2,1)
  sns.set_style('whitegrid')
  plt.title("Analysis on {0} feature".format(col))
  sns.countplot(x=col,hue=col,data=train,palette='rainbow')
  plt.legend(loc=1)

  plt.subplot(3,2,2)
  sns.set_style('whitegrid')
  plt.title("Category wise Analysis".format(col))
  sns.countplot(x=col,hue='Survived',data=train,palette='rainbow')
  plt.legend(loc=1)
plt.figure(figsize=(13,8))

sns.FacetGrid(train,aspect=4).map(sns.kdeplot,'Age',shade= True).add_legend() 
plt.title("Analysis on Age feature")
plt.ylabel('Ratio',fontsize=15) 

sns.FacetGrid(train, hue="Survived",aspect=4).map(sns.kdeplot,'Age',shade= True).add_legend()
plt.title("Classwise Analysis on Age feature") 
plt.ylabel('Ratio',fontsize=15) 
plt.figure(figsize=(13,8))

sns.FacetGrid(train,aspect=4).map(sns.kdeplot,'Fare',shade= True).add_legend() 
plt.title("Analysis on Fare feature")
plt.ylabel('Ratio',fontsize=15) 

sns.FacetGrid(train, hue="Survived",aspect=4).map(sns.kdeplot,'Fare',shade= True).add_legend()
plt.title("Classwise Analysis on Fare feature") 
plt.ylabel('Ratio',fontsize=15) 
sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)
embarked_mapping = {'S':0,'C':1,'Q':2}

train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)
data=[train,test]
for dataset in data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4
for dataset in data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2
    dataset.loc[dataset['Fare'] >= 100, 'Fare'] = 3
train
test
train["FamilySize"] = train["SibSp"] + train["Parch"]
test["FamilySize"] = test["SibSp"] + test["Parch"]
train= train.drop(['SibSp','Parch'],axis=1,inplace=False)
train
test= test.drop(['SibSp','Parch'],axis=1,inplace=False)
test