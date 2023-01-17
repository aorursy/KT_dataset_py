import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# 1. Load train & test Dataset 
train_data = pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

# Profile Report on Dataset 
#pfr = pandas_profiling.ProfileReport(train_data)
#report.to_file("DS_Titanic_AF.html")

print('train data: %s, test data %s' %(str(train_data.shape), str(test_data.shape)) )
train_data.head()
print('_'*40)
train_data.head()
#### 1. Unique Values:
#PasengerID : Yes
#Name :  No
#---------------------------------------------------
# Check If PassengerId is Unique 
if (train_data.PassengerId.nunique() == train_data.shape[0]):
    print('PassengerId is Unique.') 
else:
    print('-[Error]Id is not Unique.')    
    
if (len(np.intersect1d(train_data.PassengerId.values, test_data.PassengerId.values))== 0 ):
    print('train and test datasets are Distinct.')
else:
    print('- [Error] train and test datasets are NOT Distinct.')

# Check If PassengerId is Unique 
if (train_data.Name.nunique() == train_data.shape[0]):
    print('Name is Unique.') 
else:
    print('[Error]Id is not Unique.')    
    
if (len(np.intersect1d(train_data.Name.values, test_data.Name.values))== 0 ):
    print('train and test datasets Names are Distinct.')
else:
    print('- [Error] train and test datasets Names are NOT Distinct.')
#### C. Find Missing Values --> DONE
# Age                 
# Cabin            
# Embarked           
# Fare            
#---------------------------------------------------

# [AF] From describe() we come to know that Age contains Some Missing Data. We need to fix this.
# [AF] We will Check All Variables for Missing values in Columns!

# train_data.apply(lambda x: sum(x.isnull().values), axis = 0) # For columns

# Check for missing data & list them 
nas = pd.concat([train_data.isnull().sum(), test_data.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
print('Nan in the data sets')
print(nas[nas.sum(axis=1) > 0])

# [Extra] A boolean Condition to Check wheather our Dataset have any Missing Value.
#train_data.isnull().values.any()

# Fill Missing Values or Ignore Columns --> DONE
# Age                 
# Cabin            
# Embarked           
# Fare 
# ---------------- Age ----------------

# Fill NaN
train_data['Age'].fillna(train_data['Age'].mean(),inplace=True)
 # convert from float to int
train_data['Age'] = train_data['Age'].astype(int)

# Fill NaN
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
 # convert from float to int
test_data['Age'] = test_data['Age'].astype(int)
g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=10)
# ---------------- Cabin ----------------
# Drop the This not relevent 
train_data.drop(['Cabin','Ticket'], axis=1, inplace=True)
test_data.drop(['Cabin','Ticket'], axis=1, inplace=True)

# ---------------- Embarked ----------------
print(test_data['Embarked'].mode())
print(test_data['Embarked'].mode()[0])
#replacing the missing values in the Embarked feature with S
train_data = train_data.fillna({"Embarked": train_data['Embarked'].mode()})


# TestData doesn't contain Missing Values for Embarked
# ---------------- Fare ----------------
# Fill NaN
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
 # convert from float to int
train_data['Fare'] = train_data['Fare'].astype(int)

# Fill NaN
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
 # convert from float to int
test_data['Fare'] = test_data['Fare'].astype(int)
g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Fare', bins=10)
print('_'*40)

# Check for missing data & list them 
nas = pd.concat([train_data.isnull().sum(), test_data.isnull().sum()], axis=1, keys=['Train Dataset', 'Test Dataset']) 
print('Nan in the data sets')
print(nas[nas.sum(axis=1) > 0])
print('train data: %s, test data %s' %(str(train_data.shape), str(test_data.shape)) )
train_data.info()
train_data.shape
train_data.describe(include=['number'])
train_data.describe(include=['object'])
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
sns.countplot('Survived',data=train_data,ax=axes[0,0])
sns.countplot('Pclass',data=train_data,ax=axes[0,1])
sns.countplot('Sex',data=train_data,ax=axes[0,2])
sns.countplot('SibSp',data=train_data,ax=axes[0,3])
sns.countplot('Parch',data=train_data,ax=axes[1,0])
sns.countplot('Embarked',data=train_data,ax=axes[1,1])
sns.distplot(train_data['Fare'], kde=True,ax=axes[1,2])
sns.distplot(train_data['Age'].dropna(),kde=True,ax=axes[1,3])
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = train_data.corr()
sns.heatmap(corr,
            mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
train_data.head()
# 4 Sex:
def personType(Gender):
    if Gender == "female":
        return 1
    elif Gender =="male":
        return 0
    
train_data['Sex'] = train_data['Sex'].apply(personType)
 
# test_data
test_data['Sex'] = test_data['Sex'].apply(personType)


# 5 Age:
def ageType(passAge):
    if passAge < 16:
        return str('Child')
    else :
        return str('adult')
    
# train_data['ageType'] = train_data['Age'].apply(ageType)

print('train dataset: %s, test dataset %s' %(str(train_data.shape), str(test_data.shape)) )

train_data.head()
test_data.head()

import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(train_data['Age'],bins=15,kde=False)
plt.ylabel('Count')
plt.title('Aget Distribution -AF')
from matplotlib import style
style.use('ggplot')
plt.figure(figsize=(12,4))
sns.boxplot(x='Age', data = train_data)

print('train dataset: %s, test dataset %s' %(str(train_data.shape), str(test_data.shape)) )
train_data.head()
test_data.head()
train_data = train_data.drop(['Name',  'Fare', 'Embarked'],axis=1)
test_data = test_data.drop(['Name', 'Fare', 'Embarked'],axis=1)

#X_train = train_dataset.drop("Survived",axis=1).as_matrix()
#Y_train = train_dataset["Survived"].as_matrix()
#X_test  = test_dataset.drop("PassengerId",axis=1).copy().as_matrix()


X_train = train_data.drop("Survived",axis=1)
Y_train = train_data["Survived"]


print(X_train.shape)
print(Y_train.shape)


X_test = test_data.copy()

X_test.shape
print(X_test.head())
X_train.head()
X_test.head()
Y_train.head()
# machine learning
from sklearn.tree import DecisionTreeClassifier

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
print(Y_pred)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

# Classifier 
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
# Fit Data
model.fit(X_train,Y_train)
print("oob_score : ", model.oob_score_)
y_oob = model.oob_prediction_
print("C-stat: ", roc_auc_score(Y_train,y_oob))
model.feature_importances_
feature_importances = pd.Series(model.feature_importances_,index=X_train.columns)
feature_importances.sort_values()
feature_importances.plot(kind="barh",figsize=(7,6));
my_submission  = pd.DataFrame({
    "PassengerId":X_test["PassengerId"],
    "Survived":Y_pred
})
my_submission.to_csv("afarane_titanic_kaggle.csv",index=False)
my_submission.head()
my_submission.tail()