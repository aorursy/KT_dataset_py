# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import necessary modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
#reading data

training_data = pd.read_csv('/kaggle/input/titanic/train.csv')

testing_data = pd.read_csv('/kaggle/input/titanic/test.csv')

combined_data1 = [training_data, testing_data]



for data in combined_data1:

    print('\n',data.info())
#checking data

def show_me_data(combined_data):

    dt = ['Train Data','Test Data']

    for name,data in zip(dt,combined_data):

        print(name)

        display(data.head())



show_me_data(combined_data1)
training_data.corr()['Survived'].sort_values(ascending = False)
#feature extracting from Name column

for data in combined_data1:

    data['title'] = data['Name'].apply(lambda x: x[x.find(',')+2:x.find('.')])

    print(data['title'].value_counts())

    print('\n','='*50)
#just converting rarely occuring titles to other

def converttitle(x):

    if x not in ['Mr','Miss','Mrs','Master']:

        return 'other'

    else:

        return x



for data in combined_data1:

    data['title'] = data['title'].apply(converttitle)

    print(data['title'].value_counts())

    print('\n','='*50)

    
#checking whether passenger is travelling alone or not

for data in combined_data1:

    data['family_members'] = data['SibSp'] + data['Parch']

    

    data['aboard_alone'] = data['family_members'].apply(lambda x: 'yes' if x == 0 else 'no')

    

    print(data.aboard_alone.value_counts())
def pclass(x):

        if x==1:

            return 'Upper'

        elif x==2:

            return 'Middle'

        else:

            return 'Lower'



for data in combined_data1:

    data['Pclass'] = data['Pclass'].apply(pclass)

    print(data.Pclass.value_counts(),'\n')
show_me_data(combined_data1)
train_data = training_data[['Survived','Pclass','Sex','Age','Fare','title','Embarked','aboard_alone']]

test_data = testing_data[['Pclass','Sex','Age','Fare','title','Embarked','aboard_alone']]

combined_data = [train_data, test_data]



for data in combined_data:

    print(data.isnull().sum(),'\n')
#for age

train_data.Age.fillna(train_data['Age'].median(), inplace =True)

test_data.Age.fillna(test_data['Age'].median(), inplace =True)



#for embarked

train_data.Embarked.fillna('S', inplace =True)



#for Fare

test_data.Fare.fillna(test_data['Fare'].mean(), inplace =True)



for data in combined_data:

    print(data.isnull().sum(),'\n')
sns.set(style="darkgrid")

fig, ax = plt.subplots(2,2, figsize= (10,10))



sns.boxplot(y= 'Age', data= train_data, ax =ax[0,0])

ax[0,0].set_title('Training Age')



sns.distplot(train_data['Age'], bins= 10, ax =ax[0,1])

ax[0,1].set_xticks(range(0,100,10))



sns.boxplot(y= 'Age', data= test_data, ax =ax[1,0])

ax[1,1].set_title('Testing Age')



sns.distplot(train_data['Age'],bins= 10, ax =ax[1,1])

ax[1,1].set_xticks(range(0,100,10))



plt.show()
for data in combined_data:

    data['Age'] = pd.cut(data['Age'], bins = [0,20,40,60,100], labels = ['child','young','adult','old'])

    print(data.Age.value_counts(),'\n')
fig, ax = plt.subplots(2,2, figsize= (10,10))



sns.boxplot(y= 'Fare', data= train_data, ax =ax[0,0])

ax[0,0].set_title('Training Fare')



sns.distplot(train_data['Fare'], ax =ax[0,1])



sns.boxplot(y= 'Fare', data= test_data, ax =ax[1,0])

ax[1,0].set_title('Testing Fare')



sns.distplot(test_data['Fare'], ax =ax[1,1])



plt.show()
def fare(x):

        if x>300:

            return mean

        else:

            return x



for data in combined_data:

    

    mean = data.drop(data[data['Fare']>300].index)['Fare'].mean()

    data['Fare'] = data['Fare'].apply(fare)
fig, ax = plt.subplots(2,2, figsize= (10,10))



sns.boxplot(y= 'Fare', data= train_data, ax =ax[0,0])

ax[0,0].set_title('Training Fare')

ax[0,0].set_yticks(range(0,275,25))



sns.distplot(train_data['Fare'], ax =ax[0,1])

ax[0,1].set_xticks(range(0,275,25))



sns.boxplot(y= 'Fare', data= test_data, ax =ax[1,0])

ax[1,0].set_title('Testing Fare')

ax[1,0].set_yticks(range(0,275,25))



sns.distplot(test_data['Fare'], ax =ax[1,1])

ax[1,1].set_xticks(range(0,275,25))



plt.show()
for data in combined_data:

    

    data['Fare'] = pd.cut(data['Fare'], bins = [0,20,50,100,300], labels = ['Eco','business','prime','Deluxe'])

    print(data.Fare.value_counts())
show_me_data(combined_data)
final_train = pd.get_dummies(train_data, drop_first= True)

display('Train',final_train)

final_test = pd.get_dummies(test_data, drop_first= True)

display('Test',final_test)
final_train.corr()['Survived'].sort_values(ascending = False)
#import Machine Learning libraries

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV



import sklearn.metrics as sm
x = final_train.drop('Survived', axis = 1)

y = final_train['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify = y)
lr_model = LogisticRegression()

lr_model.fit(x_train,y_train)

lr_predict = lr_model.predict(x_test)



print(lr_predict[:5])

print(y_test.head())

sm.r2_score(y_test,lr_predict)
print(sm.classification_report(y_test,lr_predict))
sns.heatmap(sm.confusion_matrix(y_test,lr_predict), annot=True, cmap="YlGnBu")

plt.show()
dt_model = DecisionTreeClassifier(random_state = 0)

clf = GridSearchCV(dt_model, param_grid = {'criterion':('gini', 'entropy'),

                                           'max_depth':[2,3,4,5,6]},

                   cv=5)

clf.fit(x_train,y_train)

print(clf.best_params_)

print(clf.best_score_)



#dt_model.fit(x_train,y_train)

dt_predict = clf.predict(x_test)



#print(dt_predict[:5])

#print(y_test.head())
sm.r2_score(y_test,dt_predict)
print(sm.classification_report(y_test,dt_predict))
sns.heatmap(sm.confusion_matrix(y_test,dt_predict), annot=True, cmap="YlGnBu")

plt.show()
rf_model = RandomForestClassifier(random_state=0)

rf_clf = GridSearchCV(rf_model, param_grid = {'n_estimators': [200, 300,400,500],

                                              'max_depth' : [3,4,5,6,7],

                                              'criterion' :['gini', 'entropy']},

                     cv=3)

rf_clf.fit(x_train,y_train)

print(rf_clf.best_params_)

print(rf_clf.best_score_)



rf_predict = rf_clf.predict(x_test)

#rf_model.fit(x_train,y_train)

#rf_predict = rf_model.predict(x_test)



#print(rf_predict[:5])

#print(y_test.head())
print(sm.classification_report(y_test,rf_predict))
sm.r2_score(y_test,rf_predict)
sns.heatmap(sm.confusion_matrix(y_test,rf_predict), annot=True, cmap="YlGnBu")

plt.show()
model = RandomForestClassifier(random_state=0,

                               max_depth = 6,

                               n_estimators = 200)

model.fit(x,y)

pred = (model.predict(final_test))
submission = pd.DataFrame({

                            'PassengerId': testing_data['PassengerId'],

                            'Survived': pred

})
submission
submission.to_csv('Submission.csv', index=False)