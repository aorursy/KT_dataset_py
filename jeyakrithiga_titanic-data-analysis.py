# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
titanic_train_path = "/kaggle/input/titanic/train.csv"

titanic_test_path = "/kaggle/input/titanic/test.csv"

gender_submission_path = "/kaggle/input/titanic/gender_submission.csv"

titanic_train = pd.read_csv(titanic_train_path)

titanic_test  = pd.read_csv(titanic_test_path)

titanic_gender = pd.read_csv(gender_submission_path)

print(titanic_train.shape)

print(titanic_test.shape)
titanic_train.info()
titanic_train.describe()
titanic_train.head()
titanic_test.head()
categorical_cols = titanic_train.select_dtypes(['object'])

categorical_cols.columns
categorical_cols = categorical_cols.drop(['Cabin'],axis=1)

categorical_cols.head()
titanic_train['Sex'].unique()
titanic_train['Cabin'].unique()
titanic_train['Embarked'].dropna()

titanic_train['Embarked'].unique()
from sklearn.preprocessing import LabelEncoder



labelEncoder = LabelEncoder()



label_train = titanic_train.copy()

label_test = titanic_test.copy()



for col in categorical_cols.columns:

    if(col=='Sex'):

        print(col)

        label_train[col] = pd.DataFrame(labelEncoder.fit_transform(titanic_train[col]))

        label_test[col]  = pd.DataFrame(labelEncoder.transform(titanic_test[col]))

    

label_train.head()

label_test.head()
#label_train = label_train.drop(['Cabin','Embarked'],axis=1)

label_train.head()
target_col = label_train.Survived

train_data = label_train.drop(['Cabin','Embarked'],axis=1)

train_data
train_data['Survived'].value_counts()
import seaborn as sns

sns.lineplot(data=train_data.select_dtypes(['int']))
sns.barplot(data=train_data.select_dtypes(['int']))

sns.barplot(data=train_data.Pclass)

import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))

def displayBarplot(feature):

    survived = titanic_train[titanic_train['Survived']==1][feature].value_counts()

    dead = titanic_train[titanic_train['Survived']==0][feature].value_counts()

    df = pd.DataFrame([survived,dead])

    df.index = ['Survived','Not-Survived']

    df.plot(kind='bar',stacked=True)

    #sns.barplot(data=train_data,x=feature,y="Survived",stacked=True,alpha=0.9)

    plt.title('Number of people Survived in each Class')

    plt.ylabel('Count of People survived', fontsize=12)

    plt.xlabel(feature, fontsize=12)

    plt.show()
displayBarplot("Pclass")
displayBarplot("Sex")
displayBarplot("SibSp")
displayBarplot("Parch")
#displayBarplot("Embarked")
train_test_data = [titanic_train,titanic_test] #Combining train and test data



for dataset in train_test_data:

    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False)
titanic_train['Title'].value_counts()
titanic_test['Title'].value_counts()
title_mapping = {"Mr":0, "Miss":1, "Mrs":2,

                "Master":3, "Dr":3, "Rev":3, "Col":3, "Major":3, "Mlle":3, "Countess":3,

                "Ms":3,"Lady":3, "Jonkheer":3, "Don":3, "Dona":3, "Mme":3, "Capt":3, "Sir":3 }

for dataset in train_test_data:

    dataset['Title'] = dataset['Title'].map(title_mapping)
titanic_train.head()
titanic_test.head()
displayBarplot("Title")
titanic_train.drop('Name',axis=1,inplace=True)

titanic_test.drop('Name',axis=1,inplace=True)
titanic_train.head(10)
titanic_train["Age"].fillna(titanic_train.groupby("Title")["Age"].transform("median"),inplace=True)

titanic_test["Age"].fillna(titanic_test.groupby("Title")["Age"].transform("median"),inplace=True)

titanic_train.head(10)
for dataset in train_test_data:

    dataset.loc[ (dataset['Age'] <= 16, 'Age')] = 0

    dataset.loc[((dataset['Age'] >  16 ) & (dataset['Age'] <= 26), 'Age')] = 1

    dataset.loc[((dataset['Age'] >  26 ) & (dataset['Age'] <= 36), 'Age')] = 2

    dataset.loc[((dataset['Age'] >  36 ) & (dataset['Age'] <= 62), 'Age')] = 3

    dataset.loc[ (dataset['Age'] >  62 ,'Age')] = 4
titanic_train['Age'].head(10)
displayBarplot("Age")
titanic_train.head(10)
sex = {"female":0, "male":1}

for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map(sex)
titanic_train.head(10)
displayBarplot('Sex')
titanic_train.head(10)
titanic_train['Cabin'].value_counts()
Pclass1 = titanic_train[titanic_train['Pclass']==1]['Embarked'].value_counts()

Pclass2 = titanic_train[titanic_train['Pclass']==2]['Embarked'].value_counts()

Pclass3 = titanic_train[titanic_train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1,Pclass2,Pclass3])

df.index=['1st class','2nd class','3rd class']

df.plot(kind='bar',stacked=True,figsize=(10,4))
for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

titanic_train['Embarked'].isnull().sum()

titanic_train.head(10)
listResult = titanic_train['Embarked'].unique()

titanic_test['Embarked'].unique()

listResult
embarked_mapping = {"S":0,"C":1,"Q":2}

for dataset in train_test_data:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
titanic_train["Fare"].fillna(titanic_train.groupby("Pclass")["Fare"].transform("median"),inplace=True)

titanic_test["Fare"].fillna(titanic_test.groupby("Pclass")["Fare"].transform("median"),inplace=True)
titanic_train['Fare'].isnull().sum()

titanic_train.head(10)
for dataset in train_test_data:

    dataset.loc[ dataset['Fare']<=17, "Fare"] =0

    dataset.loc[(dataset['Fare']>17) & (dataset['Fare']<=30),'Fare'] =1

    dataset.loc[(dataset['Fare']>30) & (dataset['Fare']<=100),'Fare'] =2

    dataset.loc[(dataset['Fare']>100),'Fare'] =3
titanic_train.head(10)
displayBarplot('Fare')
displayBarplot('Pclass')
titanic_train.Cabin.isnull().sum()



for dataset in train_test_data:

    dataset["Cabin"] = dataset["Cabin"].str[:1]

    

cabin_mapping = {"A":0,"B":0.4,"C":0.8,"D":1.2,"E":1.6,"F":2.0,"G":2.4,"T":2.8}

for dataset in train_test_data:

    dataset["Cabin"] = dataset["Cabin"].map(cabin_mapping)

displayBarplot("Cabin")

    

titanic_train.head(10)
titanic_train['Cabin'].fillna(titanic_train.groupby('Pclass')['Cabin'].transform("median"),inplace=True)

titanic_test['Cabin'].fillna(titanic_test.groupby('Pclass')['Cabin'].transform("median"),inplace=True)

titanic_train.head(10)
titanic_train['FamilySize'] = titanic_train['SibSp']+titanic_train['Parch']+1

titanic_test['FamilySize'] = titanic_test['SibSp']+titanic_test['Parch']+1
family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2.0, 7:2.4, 8:2.8}

for dataset in train_test_data:

    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)
drop_features = ['SibSp','Parch','Ticket']

titanic_train = titanic_train.drop(drop_features,axis=1)

titanic_test = titanic_test.drop(drop_features,axis=1)



titanic_train.head(10)
from sklearn.tree import DecisionTreeRegressor



train_data_expt_target = pd.DataFrame(titanic_train.drop(['Survived'],axis=1))

#target_col = train_data.Survived

train_model = DecisionTreeRegressor(random_state=1)

train_model.fit(train_data_expt_target.select_dtypes(['int64','float32']),target_col)

#train_data_expt_target.select_dtypes(['int64'])
#target_col = titanic_test.Survived

#titanic_test.select_dtypes(['int64'])

#titanic_test.insert(1, 'Sex', titanic_gender['Sex'],True)

#titanic_gender

#titanic_test

survival = {0.0:0,1.0:1}

predicted_survival = pd.DataFrame(train_model.predict(titanic_test.select_dtypes(['int64','float32'])))

predicted_survival[0] = predicted_survival[0].map(survival)

predicted_survival
from sklearn.metrics import mean_absolute_error



#predicted_survival = train_model.predict(titanic_test.select_dtypes(['int64']))

mean_absolute_error(titanic_gender['Survived'],predicted_survival)
titanic_gender.Survived
from sklearn.ensemble import RandomForestRegressor



#train_data_expt_target = pd.DataFrame(train_data.drop(['Survived'],axis=1))

#target_col = train_data.Survived

train_rf_model = RandomForestRegressor(random_state=1)

train_rf_model.fit(train_data_expt_target.select_dtypes(['int64','float32']),target_col)

#train_data_expt_target.select_dtypes(['int64'])
predicted_rf_survival = pd.DataFrame(train_rf_model.predict(titanic_test.select_dtypes(['int64','float32'])))

predicted_rf_survival
survival = {0.0:0,1.0:1}

#predicted_survival = pd.DataFrame(train_model.predict(titanic_test.select_dtypes(['int64','float32'])))

predicted_rf_survival.loc[predicted_rf_survival[0]<1,0] =0

predicted_rf_survival.loc[predicted_rf_survival[0]>=1,0] =1

predicted_rf_survival[0] = predicted_rf_survival[0].map(survival)

predicted_rf_survival
from sklearn.metrics import mean_absolute_error



#predicted_survival = train_model.predict(titanic_test.select_dtypes(['int64']))

mean_absolute_error(titanic_gender['Survived'],predicted_rf_survival)
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=13,metric='euclidean')

knn_model.fit(train_data_expt_target.select_dtypes(['int64','float32']),target_col)
predicted_knn_survival = pd.DataFrame(knn_model.predict(titanic_test.select_dtypes(['int64','float32'])))

predicted_knn_survival
mean_absolute_error(titanic_gender['Survived'],predicted_knn_survival)
from sklearn.svm import SVC

svm_model = SVC()

svm_model.fit(train_data_expt_target.select_dtypes(['int64','float32']),target_col)

predicted_svm_survival = pd.DataFrame(svm_model.predict(titanic_test.select_dtypes(['int64','float32'])))

predicted_svm_survival
mean_absolute_error(titanic_gender['Survived'],predicted_svm_survival)
#titanic_gender

output = pd.DataFrame({'PassengerId': titanic_test.PassengerId,

                       'Survived': predicted_survival[0]})

output.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission
submission['Survived'].value_counts()