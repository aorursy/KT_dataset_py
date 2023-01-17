# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
titanic_train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

titanic_train_data.head()
titanic_test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

titanic_test_data.head()
women = pd.DataFrame(columns= ['Survived'])

for i in range(len(titanic_train_data.Sex)):

    if (titanic_train_data.Sex[i] == 'female'):

        women.Survived.loc[i] = titanic_train_data.Survived[i]

        
test = titanic_train_data.iloc[(titanic_train_data.Sex == 'female').values]['Survived']



test
sum(women.Survived)/len(women.Survived)
# women = titanic_train_data.loc[titanic_train_data.Sex == 'female']["Survived"]

# #rate_women = sum(women)/len(women)

# sum(women)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report



feature_set = ['Pclass','Parch','SibSp','Sex']



X = pd.get_dummies(titanic_train_data[feature_set])

test_X =  pd.get_dummies(titanic_test_data[feature_set])

y = titanic_train_data['Survived']



model = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=1)

model.fit(X,y)

yhat = model.predict(test_X)



output = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId, 'Survived': yhat}).to_csv("my_submission.csv", index=False)

print("Your submission was successfully saved!")
from sklearn.linear_model import LogisticRegression
titanic_train_data.isna().sum()
dummy = pd.get_dummies(titanic_train_data.Sex)

X = titanic_train_data

X = pd.concat([X, dummy], axis=1)

X
X.drop(['female','Sex'], axis =1, inplace=True)

#Removing categorical Sex & Female column and changing Male column name to Sex
X.rename(columns = {'male':'Sex'},inplace = True)

X
# Countinous variables Decribe:

X.describe(include='O')
#Let's do correlation: - First check with Pclass:



X[['Pclass', 'Survived']].groupby('Pclass').mean()
X[['Sex', 'Survived']].groupby('Sex').mean()
X[['SibSp', 'Survived']].groupby('SibSp').mean()
X[['Parch', 'Survived']].groupby('Parch').mean().sort_values(by = 'Survived', ascending = False)
X.drop(['PassengerId','Ticket' ], axis =1, inplace=True)

X
X.isna().sum()
#sb.countplot(X.Survived)
X.Embarked.value_counts()
X.Embarked.mode()[0]
X.Embarked.fillna(X.Embarked.mode()[0], inplace=True)

X.Embarked.value_counts()
X.Name[30]
temp = X.Name.str.split('.',expand = True)[0]

temp[0:50]
X['Salutation'] = temp.str.split(',',expand = True)[1]
X.Salutation = X.Salutation.str.lstrip()
X = X[['Salutation','Name', 'Age','Sex','Pclass','SibSp','Parch','Fare','Cabin','Embarked','Survived']]

X.head(30)
X.Salutation.unique()
#sb.boxplot(x=X.Salutation, y= X.Age)

#sb.countplot(X.Salutation)

X.Salutation.groupby(X.Salutation).count().sort_values(ascending=False)
Mr = 0

Miss = 0

Mrs=0

Master  = 0

Dr = 0

Rev= 0

Col= 0        

Mlle = 0    

Major = 0





for i in range(len(X.Age)):

    if X.Age.isnull()[i]:

        if X.Salutation[i] == 'Mr':

            Mr = Mr +1

        elif X.Salutation[i] == 'Miss':

            Miss = Miss +1

        elif X.Salutation[i] == 'Mrs':

            Mrs = Mrs +1

        elif X.Salutation[i] == 'Master':

            Master = Master +1

        elif X.Salutation[i] == 'Dr':

            Dr = Dr +1

        elif X.Salutation[i] == 'Rev':

            Rev = Rev +1

        elif X.Salutation[i] == 'Col':

            Col = Col +1

        elif X.Salutation[i] == 'Mlle':

            Mlle = Mlle +1

        elif X.Salutation[i] == 'Major':

            Major = Major +1



print("Total Mr's = ", Mr)

print("Total Miss = ", Miss)

print("Total Mrs = ", Mrs)

print("Total Master = ", Master)

print("Total Dr = ", Dr)

print("Total Rev = ", Rev)

print("Total Col = ", Col)

print("Total Mlle = ", Mlle)

print("Total Major = ", Major)
np.mean(X.loc[X.Salutation == 'Mr']['Age'])
X.loc[X.Salutation == 'Mr']['Age'].median()
X.Age.groupby(X.Salutation=='Mr').mean()
X.Age.groupby(X.Salutation=='Mr').median()
round(np.mean(X.loc[X.Salutation == 'Mr']['Age']),2)
for i in range(len(X.Age)):

    if X.Age.isnull()[i]:

        if X.Salutation[i] == 'Mr':

            X.Age[i] = round(np.mean(X.loc[X.Salutation == 'Mr']['Age']),2)

        elif X.Salutation[i] == 'Miss':

             X.Age[i] = round(np.mean(X.loc[X.Salutation == 'Miss']['Age']),2)

        elif X.Salutation[i] == 'Mrs':

              X.Age[i] = round(np.mean(X.loc[X.Salutation == 'Mrs']['Age']),2)

        elif X.Salutation[i] == 'Master':

            X.Age[i] = round(np.mean(X.loc[X.Salutation == 'Master']['Age']),2)

        elif X.Salutation[i] == 'Dr':

            X.Age[i] = round(np.mean(X.loc[X.Salutation == 'Dr']['Age']),2)
Mr = 0

Miss = 0

Mrs=0

Master  = 0

Dr = 0

Rev= 0

Col= 0        

Mlle = 0    

Major = 0





for i in range(len(X.Age)):

    if X.Age.isnull()[i]:

        if X.Salutation[i] == 'Mr':

            Mr = Mr +1

        elif X.Salutation[i] == 'Miss':

            Miss = Miss +1

        elif X.Salutation[i] == 'Mrs':

            Mrs = Mrs +1

        elif X.Salutation[i] == 'Master':

            Master = Master +1

        elif X.Salutation[i] == 'Dr':

            Dr = Dr +1

        elif X.Salutation[i] == 'Rev':

            Rev = Rev +1

        elif X.Salutation[i] == 'Col':

            Col = Col +1

        elif X.Salutation[i] == 'Mlle':

            Mlle = Mlle +1

        elif X.Salutation[i] == 'Major':

            Major = Major +1



print("Total Mr's = ", Mr)

print("Total Miss = ", Miss)

print("Total Mrs = ", Mrs)

print("Total Master = ", Master)

print("Total Dr = ", Dr)

print("Total Rev = ", Rev)

print("Total Col = ", Col)

print("Total Mlle = ", Mlle)

print("Total Major = ", Major)
X.head(30)
X.isna().sum()
X.drop('Cabin', axis=1, inplace=True)
X.isna().sum()
print(X.to_string())
X.Embarked.unique()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

X.Embarked = label_encoder.fit_transform(X.Embarked)

#Embarked is multi-nomial variable and has to be changed in BINARY - Using PD.Dummy
dummy_embarked = pd.get_dummies(X.Embarked)

dummy_embarked.rename(columns={0:'C',1:'Q',2:'S'}, inplace=True)

dummy_embarked
X = pd.concat([X,dummy_embarked],axis=1)

X
X.drop('Embarked', inplace=True, axis = 1)
X.info()
X_test = titanic_test_data

 
X_test.isna().sum()

#titanic_test_data.isna().sum()
temp_test = X_test.Name.str.split('.',expand = True)[0]

temp_test.head(2)
X_test['Salutation'] = temp_test.str.split(',', expand = True,)[1]
X_test.Salutation = X_test.Salutation.str.lstrip()

X_test.Salutation.values
for i in range(len(X_test.Age)):

    if X_test.Age.isnull()[i]:

        if X_test.Salutation[i] == 'Mr':

            X_test.Age[i] = round(np.mean(X_test.loc[X_test.Salutation == 'Mr']['Age']),2)

        elif X_test.Salutation[i] == 'Miss':

             X_test.Age[i] = round(np.mean(X_test.loc[X_test.Salutation == 'Miss']['Age']),2)

        elif X_test.Salutation[i] == 'Mrs':

              X_test.Age[i] = round(np.mean(X_test.loc[X_test.Salutation == 'Mrs']['Age']),2)

        elif X_test.Salutation[i] == 'Master':

            X_test.Age[i] = round(np.mean(X_test.loc[X_test.Salutation == 'Master']['Age']),2)

        elif X_test.Salutation[i] == 'Dr':

            X_test.Age[i] = round(np.mean(X_test.loc[X_test.Salutation == 'Dr']['Age']),2)

        elif X_test.Salutation[i] == 'Ms':

            X_test.Age[i] = round(np.mean(X.loc[X.Salutation == 'Ms']['Age']),2)

            
X_test.isna().sum()
Mr = 0

Miss = 0

Mrs=0

Master  = 0

Dr = 0

Rev= 0

Col= 0        

Mlle = 0    

Major = 0

Ms = 0





for i in range(len(X_test.Age)):

    if X_test.Age.isnull()[i]:

        if X_test.Salutation[i] == 'Mr':

            Mr = Mr +1

        elif X_test.Salutation[i] == 'Miss':

            Miss = Miss +1

        elif X_test.Salutation[i] == 'Mrs':

            Mrs = Mrs +1

        elif X_test.Salutation[i] == 'Master':

            Master = Master +1

        elif X_test.Salutation[i] == 'Dr':

            Dr = Dr +1

        elif X_test.Salutation[i] == 'Rev':

            Rev = Rev +1

        elif X_test.Salutation[i] == 'Col':

            Col = Col +1

        elif X_test.Salutation[i] == 'Mlle':

            Mlle = Mlle +1

        elif X_test.Salutation[i] == 'Major':

            Major = Major +1

        elif X_test.Salutation[i] == 'Ms':

            Ms = Ms +1



print("Total Mr's = ", Mr)

print("Total Miss = ", Miss)

print("Total Mrs = ", Mrs)

print("Total Master = ", Master)

print("Total Dr = ", Dr)

print("Total Rev = ", Rev)

print("Total Col = ", Col)

print("Total Mlle = ", Mlle)

print("Total Major = ", Major)

print("Total Ms = ", Ms)
#X_test.Embarked = label_encoder.fit_transform(X_test.Embarked)

X_test
dummy_embarked_test = pd.get_dummies(X_test.Embarked)
dummy_embarked_test.rename(columns={0:'C',1:'Q',2:'S'}, inplace=True)

dummy_embarked_test
X_test = pd.concat([X_test,dummy_embarked_test], axis=1)

# X_test
X_test.drop(['Embarked','Name', 'Ticket', 'Fare', 'Cabin'], axis =1, inplace=True)
input_feature_set = ['Age','Sex','Pclass','SibSp','Parch','C','Q','S']

output_feature_set = ['Survived']
X_test
gender = pd.get_dummies(X_test.Sex)
X_test = pd.concat([X_test, gender], axis=1)
X_test.drop(['Sex', 'female'],axis =1, inplace=True)
X_test.rename(columns={'male':'Sex'}, inplace=True)
X_test
from sklearn.linear_model import LogisticRegression

LR_Model = LogisticRegression(solver='liblinear')

LR_Model.fit(X[input_feature_set], X[output_feature_set].values.ravel())
LR_yhat = LR_Model.predict(X_test[input_feature_set])
output = pd.DataFrame({'PassengerId': titanic_test_data.PassengerId, 'Survived': LR_yhat}).to_csv("my_Logistics_regression_submission_1.csv", index=False)

print("Your submission was successfully saved!")