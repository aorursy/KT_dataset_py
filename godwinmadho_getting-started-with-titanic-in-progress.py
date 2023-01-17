# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("../input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv")

test_data.head()
complete_data=pd.concat(objs=[train_data, test_data], axis=0).reset_index(drop=True)

sns.pairplot(complete_data)

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot(complete_data['Pclass'],complete_data['Fare'],hue=complete_data['Sex'],split=True,ax=ax[0])



sns.violinplot(complete_data['Pclass'],complete_data['Age'],hue=complete_data['Sex'],split=True,ax=ax[1])

plt.show()
# Some plots using the train_data

f,ax=plt.subplots(1,2,figsize=(18,8))

sns.countplot('Pclass',hue='Sex',data=complete_data,ax=ax[0])

ax[0].set_xlabel('Passenger Class')

ax[0].set_title('Passenger Distribution')

ax[0].legend(loc=2)



sns.countplot('Embarked',hue='Sex',data=complete_data,ax=ax[1])

ax[1].set_xlabel('Embarked Port')

ax[1].set_title('Passenger Distribution')



plt.show()
plt.figure(figsize=(8,8))

#sns.lmplot(x='Pclass',y ='Fare',data=complete_data)

sns.scatterplot(complete_data['Pclass'],complete_data['Fare'],hue=complete_data['Sex'])

plt.show()
plt.figure(figsize=(8,8))

sns.heatmap(complete_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

plt.show()
complete_data[complete_data['Cabin'].isnull()]
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print('During the sinking of the Titanic '+str(round(rate_women*100,2))+'% of women survived while '+str(round(rate_men*100,2))+'% of men survived')
# Survival Plot

grp_name=['Died','Survived']

f,ax=plt.subplots(1,2,figsize=(18,8)) # 1 row , 2 columns subplots



women.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[0],shadow=True)

ax[0].set_title('Women Survival rate') 

ax[0].set_ylabel('')

ax[0].legend(labels=grp_name,loc="best")



men.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],labels=None,shadow=True)

ax[1].set_title('Men Survival rate') 

ax[1].set_ylabel('')

ax[1].legend(labels=grp_name,loc="best")

plt.show()
grp_name=['Died','Survived']



f,ax=plt.subplots(1,3,figsize=(25,10))

sns.countplot('Embarked',hue='Survived',data=train_data,ax=ax[0])

ax[0].set_xlabel('Embarked Port')

ax[0].set_title('Passenger Distribution')

ax[0].legend(labels=grp_name,loc="best")



sns.countplot('Sex',hue='Survived',data=train_data,ax=ax[1])

ax[1].set_title('Passenger Distribution')

ax[1].legend(labels=grp_name,loc="best")



sns.countplot('Pclass',hue='Survived',data=train_data,ax=ax[2])

ax[2].set_title('Passenger Distribution')

ax[2].legend(labels=grp_name,loc="best")



plt.show()
from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame(test_data)

output['Survived']= predictions



output_csv = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})



output.head()



# Comand to submit the results

output_csv.to_csv('submission-tutorial.csv', index=False)

print("Your submission was successfully saved!")
grp_name=['Died','Survived']

f,ax=plt.subplots(1,2,figsize=(18,8))



sns.countplot('Embarked',hue='Survived',data=output,ax=ax[0])

ax[0].set_xlabel('Embarked Port')

ax[0].set_title('Model Survival Prediction')

ax[0].legend(labels=grp_name,loc="best")



sns.countplot('Sex',hue='Survived',data=output,ax=ax[1])

ax[1].set_title('Model Survival Prediction')

ax[1].legend(labels=grp_name,loc="best")



plt.show()
women = output.loc[output.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



men = output.loc[output.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



# Survival Plot

grp_name=['Died','Survived']

f,ax=plt.subplots(1,2,figsize=(18,8)) # 1 row , 2 columns subplots



women.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[0],shadow=True)

ax[0].set_title('Women Survival rate') 

ax[0].set_ylabel('')

ax[0].legend(labels=grp_name,loc="best")



men.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],labels=None,shadow=True)

ax[1].set_title('Men Survival rate') 

ax[1].set_ylabel('')

ax[1].legend(labels=grp_name,loc="best")

plt.show()
#Using the mean age of different passenger class to fill in the age gaps

def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age,axis=1)

test_data['Age'] = test_data[['Age','Pclass']].apply(impute_age,axis=1)
y = train_data["Survived"]



features = ["Pclass","Age","Sex","SibSp","Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)

output = pd.DataFrame(test_data)

output['Survived']= predictions



output_csv = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})



output.head()



# Comand to submit the results

output_csv.to_csv('submission-age.csv', index=False)

print("Your submission was successfully saved!")
women = output.loc[output.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



men = output.loc[output.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



# Survival Plot

grp_name=['Died','Survived']

f,ax=plt.subplots(1,2,figsize=(18,8)) # 1 row , 2 columns subplots



women.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[0],shadow=True)

ax[0].set_title('Women Survival rate') 

ax[0].set_ylabel('')

ax[0].legend(labels=grp_name,loc="best")



men.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],labels=None,shadow=True)

ax[1].set_title('Men Survival rate') 

ax[1].set_ylabel('')

ax[1].legend(labels=grp_name,loc="best")

plt.show()
from sklearn.linear_model import LogisticRegression



y = train_data["Survived"]



features = ["Pclass","Age","Sex","SibSp","Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



logmodel = LogisticRegression()

logmodel.fit(X,y)

predictions = logmodel.predict(X_test)

output = pd.DataFrame(test_data)

output['Survived']= predictions



output_csv = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})



output.head()



# Comand to submit the results

output_csv.to_csv('submission-LogR.csv', index=False)

print("Your submission was successfully saved!")
women = output.loc[output.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



men = output.loc[output.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



# Survival Plot

grp_name=['Died','Survived']

f,ax=plt.subplots(1,2,figsize=(18,8)) # 1 row , 2 columns subplots



women.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[0],shadow=True)

ax[0].set_title('Women Survival rate') 

ax[0].set_ylabel('')

ax[0].legend(labels=grp_name,loc="best")



men.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],labels=None,shadow=True)

ax[1].set_title('Men Survival rate') 

ax[1].set_ylabel('')

ax[1].legend(labels=grp_name,loc="best")

plt.show()
# Changing for the train data

train_1=train_data.drop(['Cabin'],axis=1)

train_1.dropna(inplace=True)

features=['PassengerId','Pclass','Age','SibSp','Parch','Fare','Survived']

sex = pd.get_dummies(train_1['Sex'],drop_first=True)

embark = pd.get_dummies(train_1['Embarked'].dropna(),drop_first=True)

train = pd.concat([train_1[features],sex,embark],axis=1)

train.head()
# Changing for the test data

test_1=test_data.drop(['Cabin'],axis=1)

test_1.dropna(inplace=True)

features=['PassengerId','Pclass','Age','SibSp','Parch','Fare']

sex = pd.get_dummies(test_1['Sex'],drop_first=True)

embark = pd.get_dummies(test_1['Embarked'].dropna(),drop_first=True)

test = pd.concat([test_1[features],sex,embark],axis=1)

test.head()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[0])

ax[0].set_title('Null values in train data') 



sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis',ax=ax[1])

ax[1].set_title('Null values in test data')



plt.show()
features=['Pclass','Age','SibSp','Parch','Fare','male','Q','S']

y = train['Survived']

X = train[features]



X_test = test[features]



logmodel = LogisticRegression(max_iter=1000)

logmodel.fit(X,y)

predictions = logmodel.predict(X_test)

output = pd.DataFrame(test)

output['Survived']= predictions



output_csv = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})



# Comand to submit the results

output_csv.to_csv('submission-LogR2.csv', index=False)

print("Your submission was successfully saved!")
#output.head()



women = output.loc[output.male == 0]["Survived"]

rate_women = sum(women)/len(women)



men = output.loc[output.male == 1]["Survived"]

rate_men = sum(men)/len(men)



# Survival Plot

grp_name=['Died','Survived']

f,ax=plt.subplots(1,2,figsize=(18,8)) # 1 row , 2 columns subplots



women.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[0],shadow=True)

ax[0].set_title('Women Survival rate') 

ax[0].set_ylabel('')

ax[0].legend(labels=grp_name,loc="best")



men.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[1],labels=None,shadow=True)

ax[1].set_title('Men Survival rate') 

ax[1].set_ylabel('')

ax[1].legend(labels=grp_name,loc="best")

plt.show()