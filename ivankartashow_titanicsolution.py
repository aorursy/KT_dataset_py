import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import sklearn as sk

import seaborn as sns
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

full_data = pd.concat([test_data, train_data]) #Merge data
train_data
labels = ['Exists', 'NaN']

sizes = [100, train_data['Cabin'].isnull().sum()]

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']



fig1, ax1 = plt.subplots()

ax1.pie(sizes, colors = colors, labels = labels)



#Draw a circle

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)



# Equal aspect ratio ensures that pie is drawn as a circle

ax1.axis('equal')  

plt.tight_layout()

plt.show()
train_data['Sex'] = train_data['Sex'].map({'female': 1, 'male': 0}) #Replace str with int

train_data['Embarked'] = train_data['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})



train_data.drop(columns=['Name', 'PassengerId', 'Cabin', 'Ticket'], inplace=True) #Dropping useless columns
test_data['Sex'] = test_data['Sex'].map({'female': 1, 'male': 0}) #Replace str with int

test_data['Embarked'] = test_data['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})



test_data.drop(columns=['Name', 'PassengerId', 'Cabin', 'Ticket'], inplace=True) #Dropping useless columns
NaN_amount = train_data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']].isnull().sum()

sns.barplot(NaN_amount.index, NaN_amount)

NaN_amount
train_data['Age'].fillna(train_data['Age'].median(), inplace=True) # Filling all missing values

train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
NaN_amount = test_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']].isnull().sum()

sns.barplot(NaN_amount.index, NaN_amount)

NaN_amount
test_data['Age'].fillna(test_data['Age'].median(), inplace=True) # Filling all missing values

test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
train_data
#Visualization 1



#sns.jointplot((full_data['Survived'] == 0).sum(), (full_data['Survived'] == 1).sum())



values = full_data['Survived'].value_counts()

sns.barplot(values.index, values)

plt.xlabel('Survived', fontsize=14)

plt.ylabel('')

print('Amount of survived is:', (full_data['Survived'] == 1).sum())

print('Amount of deaths:', (full_data['Survived'] == 0).sum())

#Visualization 2

df1 = train_data.groupby(['Pclass']).mean()['Survived']

sns.barplot(df1.index, df1);

plt.ylabel('Survived', fontsize=14);

plt.xlabel('Class', fontsize=14);
#Visualization 3

df2 = train_data.groupby(['Sex']).mean()['Survived']

sns.barplot(df2.index, df2);
#Visualization 4

train_data['Partners'] = train_data['Parch'] + train_data['SibSp']

df3 = train_data.groupby(['Partners']).mean()['Survived']

sns.barplot(df3.index, df3)

train_data.drop(columns=['SibSp', 'Parch'], inplace=True) #Dropping useless columns - we have 'Partners'
test_data['Partners'] = test_data['Parch'] + test_data['SibSp']

test_data.drop(columns=['SibSp', 'Parch'], inplace=True) #Dropping useless columns - we have 'Partners'
train_data
#Visualization 5

df6 = train_data.groupby(['Age']).mean()['Survived']

df6 = df6.head(15)

sns.barplot(df6.index, df6);

train_data['Child'] = (train_data['Age'] <= 1).astype(int)

test_data['Child'] = (test_data['Age'] <= 1).astype(int)

train_data
#Visualization 6

df7 = train_data.groupby(['Age']).mean()['Survived']

df7 = df7.iloc[-25:]

sns.barplot(df7.index, df7);
#Visualization 7

df8 = train_data.groupby(['Embarked']).mean()['Survived']

sns.barplot(df8.index, df8);
#Visualization 8

df9 = train_data.groupby(['Fare']).mean()['Survived']

df10 = train_data.groupby(['Fare']).mean()['Survived']

df9, df10 = df9.head(5), df10[20:25]



sns.barplot(df9.index, df9);
sns.barplot(df10.index, df10);
#Initially Testing with SVM Model



from sklearn.svm import SVC

#Creation of algorithm

clf = SVC(gamma='auto')
#Target and features selection

y = train_data['Survived']

X = train_data
X.drop(columns=['Survived'], inplace=True)
#Input, fitting

clf.fit(X, y)



#Prediction

x = test_data

y_pred = clf.predict(x)

train_data
test_df_id = pd.read_csv('/kaggle/input/titanic/test.csv')
submission = pd.DataFrame({

    "PassengerId": test_df_id["PassengerId"],

    "Survived": y_pred

})
#submission.to_csv('submission.csv', index=False) Accuracy ~ 0.6
#Then let's try RandomTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import GridSearchCV #Using Grid Search
# 'min_samples_leaf': [1, 2, 4],

 #'min_samples_split': [2, 5, 10],

#'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

# 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}]



# Set the parameters by cross-validation

tuned_parameters = [{'bootstrap': [True, False],

 'max_depth': [10, 40, 50, 60, 70, 80, 90],

 'max_features': ['auto', 'sqrt'],

 'n_estimators': [200, 400, 1200, 1800]}]
#clf = RandomForestClassifier(max_depth=2, random_state=0)



#Creating a model

clf = GridSearchCV(

        RandomForestClassifier(), tuned_parameters)
#Input, fitting

clf.fit(X, y)

#Showing the best parameters

print(clf.best_params_)
#Prediction

x = test_data

y_pred = clf.predict(x)

#Submission



submission = pd.DataFrame({

    "PassengerId": test_df_id["PassengerId"],

    "Survived": y_pred

})



submission.to_csv('submission.csv', index=False)
#XGBoost Usage



import xgboost as xgb
#Setting some parameters

param={'objective':'multi:softprob','num_class':6,'n_jobs':4,'seed':42}
model=xgb.XGBClassifier(**param,n_estimators=300)

model.fit(X,y)
#Prediction

x = test_data

y_pred = model.predict(x)

#Accuracy is around 0.67