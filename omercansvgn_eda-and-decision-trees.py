import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
passanger = pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')

data = passanger.copy()

data.head()
# We check missing values

data.isnull().any()
# How many men and women were on board

import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))

sns.countplot(data['Sex'],palette='coolwarm');

print(data['Sex'].value_counts())
plt.figure(figsize=(10,8))

sns.countplot(data['Sex'],hue=data['Survived'],palette='BuGn_r')

plt.xlabel('Sex')

plt.ylabel('Number Of People')

plt.title('Survived Status by Gender');
plt.figure(figsize=(10,8))

sns.countplot(data['Survived'],hue=data['Category'],palette='vlag')

plt.xlabel('Survived')

plt.ylabel('Number of people')

plt.title('Crew Or Passenger Survived Status');
plt.figure(figsize=(15,10))

sns.countplot(data['Country'],hue=data['Category'],palette='rocket')

plt.xlabel('Country')

plt.ylabel('Number Of People')

plt.title('Where Do The Team Or Passengers Join From')

plt.legend(loc='upper right');
plt.figure(figsize=(10,8))

labels = ['No Survive','Survive']

explode = (0, 0.1) 

plt.pie(data['Survived'].value_counts(),labels=labels,explode=explode,autopct='%1.1f%%',shadow=True, startangle=180,colors=['#F3CFB3','#F3C4BF'])

plt.title('Survival Rate')

plt.legend();
# Datasets Seperation

data['Category'] = [1 if i.strip() == 'P'else 0 for i in data.Category]

data['Sex'] = [1 if i.strip() == 'M'else 0 for i in data.Sex]

y = data['Survived']

x = data.loc[:,['Sex','Age','Category']]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

from sklearn.tree import DecisionTreeClassifier

DC = DecisionTreeClassifier()

DC.fit(x_train,y_train)
from sklearn.metrics import accuracy_score

y_pred = DC.predict(x_test)

print('Basit Accuracy Score:',accuracy_score(y_test,y_pred))
from sklearn.model_selection import GridSearchCV

dc_params = {'criterion':['gini','entropy'],

            'splitter':['best','random'],

            'max_depth':np.arange(1,10),

            'min_samples_split':np.arange(1,10),

            'min_samples_leaf':np.arange(1,10),

            'max_features':['auto','sqrt','log2']}

dc = DecisionTreeClassifier()

dc_cv = GridSearchCV(dc,dc_params,cv=10,n_jobs=-1,verbose=2)

dc_cv.fit(x_train,y_train)
dc_tuned = DecisionTreeClassifier(criterion=dc_cv.best_params_['criterion'],

                                 max_depth=dc_cv.best_params_['max_depth'],

                                 max_features=dc_cv.best_params_['max_features'],

                                 min_samples_leaf=dc_cv.best_params_['min_samples_leaf'],

                                 min_samples_split=dc_cv.best_params_['min_samples_split'],

                                 splitter=dc_cv.best_params_['splitter']).fit(x_train,y_train)
y_pred = dc_tuned.predict(x_test)

print('Verified Accuracy Score:',accuracy_score(y_test,y_pred))
Importance = pd.DataFrame({'Importance':dc_tuned.feature_importances_*100},

                         index=x_train.columns)

x = Importance.sort_values(by='Importance',axis=0,ascending=True).plot(kind='barh',color='#9B5F2B')