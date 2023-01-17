# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

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
train_data=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

train_data.info()
train_data.head()
train_data.describe()
test_data=pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')

train_data.columns
plt.figure(figsize=(10,10))

sns.heatmap(train_data.corr(),annot = True,cbar=False,cmap='viridis')

plt.show()
test_data.head()
complete_data=pd.concat(objs=[train_data, test_data], axis=0).reset_index(drop=True)
f,ax=plt.subplots(1,3,figsize=(18,5))

sns.distplot(complete_data['Age'],ax=ax[0])

sns.distplot(complete_data['Annual_Premium'],ax=ax[1])

sns.distplot(complete_data['Vintage'],ax=ax[2])

plt.show()
plt.figure(figsize=(18,8))

sns.countplot('Age',hue='Gender',data=complete_data)

plt.title('Client Age and Gender distribution')

plt.show()
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot(complete_data['Vehicle_Age'],complete_data['Annual_Premium'],hue=complete_data['Gender'],split=True,ax=ax[0])

ax[0].set_ylim(-10000,100000)



sns.violinplot(complete_data['Vehicle_Age'],complete_data['Vintage'],hue=complete_data['Gender'],split=True,ax=ax[1])

plt.show()

plt.figure(figsize=(18,8))

sns.scatterplot(complete_data['Age'],complete_data['Annual_Premium'],hue=complete_data['Vehicle_Age'])

plt.show()
plt.figure(figsize=(18,8))

sns.scatterplot(complete_data['Age'],complete_data['Annual_Premium'],hue=complete_data['Response'])

plt.show()
Females=complete_data.loc[complete_data.Gender == 'Female']["Driving_License"]

Males=complete_data.loc[complete_data.Gender == 'Male']["Driving_License"]



grp_name=['No DL','DL']



f,ax=plt.subplots(1,2,figsize=(18,8)) # 1 row , 2 columns subplots

Females.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[0],shadow=True)

ax[0].set_title('Women with Drivers Licence') 

ax[0].set_ylabel('')

ax[0].legend(labels=grp_name,loc="best")



Males.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[1],shadow=True)

ax[1].set_title('Men with Drivers Licence') 

ax[1].set_ylabel('')

ax[1].legend(labels=grp_name,loc="best")



plt.show()
Females=complete_data.loc[complete_data.Gender == 'Female']["Previously_Insured"]

Males=complete_data.loc[complete_data.Gender == 'Male']["Previously_Insured"]



grp_name=['No Insurance','Insured']



f,ax=plt.subplots(1,2,figsize=(18,8)) # 1 row , 2 columns subplots

Females.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[0],shadow=True)

ax[0].set_title('Women with previous Car Insurance') 

ax[0].set_ylabel('')

ax[0].legend(labels=grp_name,loc="best")



Males.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[1],shadow=True)

ax[1].set_title('Men with previous Car Insurance')

ax[1].set_ylabel('')

ax[1].legend(labels=grp_name,loc="best")



plt.show()
plt.figure(figsize=(18,8))

sns.countplot('Age',hue='Previously_Insured',data=complete_data)#.loc[complete_data.Gender == 'Female'])

plt.title('Client Age and Insurance distribution')

plt.show()
Females=complete_data.loc[complete_data.Gender == 'Female']["Vehicle_Damage"]

Males=complete_data.loc[complete_data.Gender == 'Male']["Vehicle_Damage"]



grp_name=['No Vehicle Damage','Vehicle Damaged']



f,ax=plt.subplots(1,2,figsize=(18,8)) # 1 row , 2 columns subplots

Females.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[0],shadow=True)

ax[0].set_title('Women Vehicle Damage') 

ax[0].set_ylabel('')

ax[0].legend(labels=grp_name,loc="best")



Males.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[1],shadow=True)

ax[1].set_title('Men Vehicle Damage')

ax[1].set_ylabel('')

ax[1].legend(labels=grp_name,loc="best")



plt.show()
plt.figure(figsize=(18,8))

sns.countplot('Age',hue='Vehicle_Damage',data=complete_data)

plt.title('Client Age and Vehicle Damage distribution')

plt.show()
Females=train_data.loc[train_data.Gender == 'Female']["Response"]

Males=train_data.loc[train_data.Gender == 'Male']["Response"]



grp_name=['Response No','Response Yes']



f,ax=plt.subplots(1,2,figsize=(18,8)) # 1 row , 2 columns subplots

Females.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[0],shadow=True)

ax[0].set_title('Women response to Vehicle Insurance') 

ax[0].set_ylabel('')

ax[0].legend(labels=grp_name,loc="best")



Males.value_counts(sort=False).plot.pie(explode=[0,0.1],autopct='%1.1f%%',labels=None,ax=ax[1],shadow=True)

ax[1].set_title('Men response to Vehicle Insurance')

ax[1].set_ylabel('')

ax[1].legend(labels=grp_name,loc="best")

plt.show()
plt.figure(figsize=(18,8))

sns.countplot('Age',hue='Response',data=complete_data)

plt.title('Client Age and Response distribution')

plt.show()
Male=pd.get_dummies(train_data['Gender'],drop_first=True)

Damage=pd.get_dummies(train_data['Vehicle_Damage'],drop_first=True)

Veh_age=pd.get_dummies(train_data['Vehicle_Age'],drop_first=True)



finaltrain=train_data.drop(['Gender','Vehicle_Age','Vehicle_Damage','Response'],axis=1)

finaltrain=pd.concat([finaltrain,Male,Damage,Veh_age],axis=1)



finaltrain.head()
Male=pd.get_dummies(test_data['Gender'],drop_first=True)

Damage=pd.get_dummies(test_data['Vehicle_Damage'],drop_first=True)

Veh_age=pd.get_dummies(test_data['Vehicle_Age'],drop_first=True)



finaltest=test_data.drop(['Gender','Vehicle_Age','Vehicle_Damage'],axis=1)

finaltest=pd.concat([finaltest,Male,Damage,Veh_age],axis=1)



finaltest.head()
from sklearn.model_selection import train_test_split

X=finaltrain

y=train_data['Response']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
from sklearn.ensemble import RandomForestClassifier



RFC = RandomForestClassifier(n_estimators=100)

RFC.fit(X_train,y_train)



predictions = RFC.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
RFC = RandomForestClassifier(n_estimators=100)

RFC.fit(finaltrain,y)



predictions = RFC.predict(finaltest)
output_csv = pd.DataFrame({'id': test_data.id, 'Response': predictions})

output_csv.to_csv('RFC_submission.csv', index=False)

print("Your submission was successfully saved!")