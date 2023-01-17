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
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

training=pd.read_csv('/kaggle/input/titanic/train.csv')

test=pd.read_csv('/kaggle/input/titanic/test.csv')

training['train_test']=1

test['train_test']=0

all_data=pd.concat([training,test])

all_data.columns



training.info()
training.describe()
training.describe().columns
df_num=training[['Age','SibSp','Parch','Fare']]

df_cat = training[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
for i in df_num.columns:

    plt.hist(df_num[i])

    plt.title(i)

    plt.show()
print(df_num.corr())

sns.heatmap(df_num.corr())

sns.pairplot(df_num)
pd.pivot_table(training,index='Survived',values=['Age','SibSp','Parch','Fare'],aggfunc='count')
for i in df_cat.columns:

    sns.barplot(df_cat[i].value_counts().index,df_cat[i].value_counts())

    plt.title(i)

    plt.show()

    
print(pd.pivot_table(training, index = 'Survived', columns = 'Pclass',

                     values = 'Ticket' ,aggfunc ='count'))

print()

print(pd.pivot_table(training, index = 'Survived', columns = 'Sex',

                     values = 'Ticket' ,aggfunc ='count'))

print()

print(pd.pivot_table(training, index = 'Survived', columns = 'Embarked',

                     values = 'Ticket' ,aggfunc ='count'))
training['cabin_multiple']=training.Cabin.apply(lambda x:0 if pd.isna(x) else len(x.split(' ')))

training['cabin_multiple'].value_counts()
pd.pivot_table(training,index='Survived',columns='cabin_multiple',

               values='Ticket',aggfunc='count')
training['cabin_adv'] = training.Cabin.apply(lambda x: str(x)[0])

print(training['cabin_adv'].value_counts())

pd.pivot_table(training,index='Survived',columns='cabin_adv',

               values='Name',aggfunc='count')

#understand ticket values better 

#numeric vs non numeric

training['numeric_ticket']=training.Ticket.apply(lambda x:1 if x.isnumeric() else 0)

training['ticket_letters'] = training.Ticket.apply(lambda x: ''.join

(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
training['numeric_ticket'].value_counts()
training['ticket_letters'].value_counts()
#difference in numeric vs non-numeric tickets in survival rate 

pd.pivot_table(training,index='Survived',columns='numeric_ticket',

               values = 'Ticket', aggfunc='count')
pd.pivot_table(training,index='Survived',columns='ticket_letters',

               values = 'Ticket', aggfunc='count')
#feature engineering on person's name

training['Name'].head(20)

training['name_title']=training.Name.apply(lambda x:x.split(',')[1].split('.')[0])
training['name_title'].value_counts()
all_data['cabin_multiple'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))



all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])



all_data['numeric_ticket'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)



all_data['ticket_letters'] = all_data.Ticket.apply(lambda x: ''.join

(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)



all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())







#impute nulls for continuous data 

all_data.Age = all_data.Age.fillna(training.Age.median())

all_data.Fare = all_data.Fare.fillna(training.Fare.median())



#drop null 'embarked' rows. Only 2 instances of this in training and 0 in test 

all_data.dropna(subset=['Embarked'],inplace = True)



sns.heatmap(all_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#we are not going to include cabin in our data,we have already our cabin_multiple etc.









all_dummies = pd.get_dummies(all_data[['Pclass','Sex','SibSp','Parch','Embarked','cabin_adv',

                                       'cabin_multiple','numeric_ticket','name_title','train_test']])





age=all_data['Age']

fare=all_data['Fare']

all_dummies=pd.concat([all_dummies,age,fare],axis=1)



#Split to train test again

X_train = all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis =1)

X_test = all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)





y_train = all_data[all_data.train_test==1].Survived

y_train.shape
#first  we need some feature scaling in our data

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

all_dummies_scaled = all_dummies.copy()

all_dummies_scaled[['Age','Fare']]= scale.fit_transform(all_dummies_scaled[['Age','Fare']])

all_dummies_scaled



X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)

X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)



y_train = all_data[all_data.train_test==1].Survived
#let's import class of different algorithm

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
#let's see naive bayes algorithim.

gnb = GaussianNB()

cv = cross_val_score(gnb,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
lr = LogisticRegression(max_iter = 2000)

cv = cross_val_score(lr,X_train,y_train,cv=5)

print(cv)

print(cv.mean())
#this is for scaled data

lr = LogisticRegression(max_iter = 2000)

cv = cross_val_score(lr,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
dt = DecisionTreeClassifier(criterion='gini',random_state = 1)

cv = cross_val_score(dt,X_train,y_train,cv=5)

print(cv)

print(cv.mean())
dt = DecisionTreeClassifier(criterion='gini',random_state = 1)

cv = cross_val_score(dt,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
knn = KNeighborsClassifier(n_neighbors=3,metric='minkowski',p=2)

cv = cross_val_score(knn,X_train,y_train,cv=5)

print(cv)

print(cv.mean())
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

cv = cross_val_score(knn,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
rf = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state = 1)

cv = cross_val_score(rf,X_train,y_train,cv=5)

print(cv)

print(cv.mean())
rf = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state = 1)

cv = cross_val_score(rf,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
svc = SVC(kernel='rbf',probability = True)

cv = cross_val_score(svc,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state =1)

cv = cross_val_score(xgb,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),('rf',rf),('gnb',gnb),

                                            ('svc',svc),('xgb',xgb)], voting = 'soft') 
cv = cross_val_score(voting_clf,X_train_scaled,y_train,cv=5)

print(cv)

print(cv.mean())
voting_clf.fit(X_train_scaled,y_train)

y_pred = voting_clf.predict(X_test_scaled).astype(int)

basic_output = {'PassengerId': test.PassengerId, 'Survived': y_pred}

basic_output=pd.DataFrame(data=basic_output)

basic_output.to_csv('basic_output.csv', index=False)

print('successfully saved,thanks Ken Jee')