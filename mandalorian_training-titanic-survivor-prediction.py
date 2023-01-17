
import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head(10)
train.describe()
train.info()
print("\n============ Missing Values ==============\n")
print(train.isnull().sum())
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
def stacked_bar(feature_name):
    survived = train[train['Survived'] == 1][feature_name].value_counts()
    dead = train[train['Survived'] == 0][feature_name].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True,figsize=(10,5))
stacked_bar('Sex')
stacked_bar('Pclass')
stacked_bar('Embarked')
stacked_bar('Parch')
stacked_bar('SibSp')
# combining train and test dataset before extract title from name

train_test_data = [train, test] 

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'].value_counts()
test['Title'].value_counts()
#Convert categorical value of title into numeric

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
# Drop feature name & ticket from train and test dataset

train.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)
test.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
#Convert categorical value of sex into numeric

sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
stacked_bar('Embarked')
train['Embarked'].fillna('S', inplace = True)
test['Embarked'].fillna('S', inplace = True)
#Convert categorical value of Embarked into numeric

embark_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embark_mapping)
#fill age missing value with median value for each title

train['Age'].fillna(train.groupby("Title")["Age"].transform("median"), inplace = True)
test['Age'].fillna(test.groupby("Title")["Age"].transform("median"), inplace = True)
#Convert Age into Categorical feature using bin

for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 17, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 17) & (dataset['Age'] <= 28), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 40), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'Age'] = 3,
    dataset.loc[dataset['Age'] > 62, 'Age'] = 4,
#Combine SibSp with Parch into Familysize

train["Familysize"] = train["SibSp"] + train["Parch"] + 1
test["Familysize"] = test["SibSp"] + test["Parch"] + 1
#Scale the feature values into 0 to 5 range

familysize_mapping = {0: 0, 1: 0.5, 2: 1, 3: 1.5, 4: 2, 5: 2.5, 6: 3, 7: 3.5, 8: 4, 9: 4.5, 10: 5, 11: 5.5, 12: 6}
for dataset in train_test_data:
    dataset['Familysize'] = dataset['Familysize'].map(familysize_mapping)
#Drop SibSp and Parch from Dataset

train.drop(['SibSp','Parch'], axis=1, inplace=True)
test.drop(['SibSp','Parch'], axis=1, inplace=True)
#fill Fare missing value with median value for each Pclass

train['Fare'].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace = True)
test['Fare'].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace = True)
#Binning Fare numerical feature into categorical

for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 20, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 20) & (dataset['Fare'] <= 40), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 40) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[dataset['Fare'] > 100, 'Fare'] = 3
#Check missing values before modelling 

train.info()
print("\n=============================\n")
test.info()
correlation_matrix = train.corr().round(2)
plt.figure(figsize = (16,5))
sns.heatmap(data=correlation_matrix, annot=True)
#Slice Target feature from training data set

target = train['Survived']
train = train.drop('Survived', axis=1)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn import model_selection

import warnings  
warnings.filterwarnings('ignore')

import numpy as np
# Set models array and its parameter

rand_state = 15
models = []
models.append(("Logistic Regression", LogisticRegression(random_state=rand_state)))
models.append(("KNN", KNeighborsClassifier(n_neighbors=rand_state)))
models.append(("Decision Tree", DecisionTreeClassifier(random_state=rand_state)))
models.append(("Random Forest", RandomForestClassifier(random_state=rand_state)))
models.append(("AdaBoost", AdaBoostClassifier(random_state=rand_state)))
models.append(("Gradient Boosting", GradientBoostingClassifier(random_state=rand_state)))
models.append(("XG Boosting", xgb.XGBClassifier()))
models.append(("SVM", SVC(random_state=rand_state)))
# Train the data using k-fold cross validation

kfold = model_selection.KFold(n_splits=10)
model_name = []
model_avgscore = []
for name, model in models:
    cv_results = model_selection.cross_val_score(model,train,target,scoring="accuracy",cv=kfold)
    print("\n"+name)
    print("Avg_score : "+str(cv_results.mean()))
    model_name.append(name)
    model_avgscore.append(cv_results.mean())
# Visualize the result

cv_df = pd.DataFrame({"AverageScore":model_avgscore,"Model":model_name})
sns.barplot("AverageScore","Model",data=cv_df,)
best_model = cv_df.sort_values(by="AverageScore",ascending=False).iloc[0]
print("Best Model = "+best_model.Model)
print("Average Score = "+str(best_model.AverageScore))
svm = SVC()
svm.fit(train,target)
test_id = test['PassengerId']
test_data = test.drop('PassengerId', axis=1)
prediction = svm.predict(test_data)
submission = pd.DataFrame({"PassengerId":test_id, "Survived":prediction}).set_index("PassengerId")
submission.to_csv('submission.csv')