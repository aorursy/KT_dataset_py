# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #for any plotting that might be necessary 

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
tit_train=pd.read_csv('/kaggle/input/titanic/train.csv')

tit_test=pd.read_csv('/kaggle/input/titanic/test.csv')



#change the catoegorical variable types to category

tit_train['Sex']=tit_train['Sex'].astype('category')

tit_train['Pclass']=tit_train['Pclass'].astype('category')

tit_train['Embarked']=tit_train['Embarked'].astype('category')

tit_train['SibSp']=tit_train['SibSp'].astype('category')

tit_train['Parch']=tit_train['Parch'].astype('category')



tit_test['Sex']=tit_train['Sex'].astype('category')

tit_test['Pclass']=tit_train['Pclass'].astype('category')

tit_test['Embarked']=tit_train['Embarked'].astype('category')

tit_test['SibSp']=tit_test['SibSp'].astype('category')

tit_test['Parch']=tit_test['Parch'].astype('category')



print(tit_train.head())

print(tit_train.shape)
#2a. Data Cleaning for NA values 

print(tit_train.isnull().sum(axis = 0))

print(tit_test.isnull().sum(axis = 0))

tit_train['Name']
import re

title_of_pas=re.compile('\w[A-z]+\.')

tit_train['title']=(tit_train.apply(lambda x: title_of_pas.findall(x.Name)[0],axis=1))

tit_train.head()
ages=pd.DataFrame(tit_train[['Age','title']].groupby(['title']).mean()).to_dict()

ages_final=ages['Age']

ages_final
tit_train['Age']=(tit_train.apply(lambda x: ages_final[x.title] if np.isnan(x.Age) else x.Age,axis=1))

print(tit_train.isnull().sum(axis = 0))

tit_train
#re pattern for just gettign the first letter

#print(str(tit_train['Cabin'][2])[0])

tit_train['Cabin']=[str(tit_train['Cabin'][x])[0] for x in range(tit_train.shape[0])]

tit_train.head()
fare_cab=tit_train[['Embarked','Fare','Cabin']].groupby(['Embarked','Cabin']).describe()

fare_cab
tit_train.loc[tit_train['Embarked'].isnull()]
tit_train['Embarked']=tit_train["Embarked"].fillna('S')

print(tit_train.isnull().sum(axis = 0))
#tit_train['title'].value_counts()

#groupby(['title']).unique()

pd.pivot_table(tit_train[['title','Cabin']],index=['Cabin'],columns=['title'],aggfunc=len).fillna(0)
pd.pivot_table(tit_train[['Parch','Cabin']],index=['Cabin'],columns=['Parch'],aggfunc=len).fillna(0)
pd.pivot_table(tit_train[['Pclass','Cabin']],index=['Cabin'],columns=['Pclass'],aggfunc=len).fillna(0)
tit_train[['Pclass','Fare','Cabin']].groupby(['Pclass','Cabin']).describe()
for x in range(tit_train.shape[0]):

    if tit_train['Cabin'][x]=='n':

        #take note of the class as well as the fare

        curr_class=tit_train['Pclass'][x]

        curr_fare=tit_train['Fare'][x]

        #now we filter the dataframe with values of the class and the fare

        temp=tit_train.loc[tit_train['Pclass']==curr_class]

        temp=temp.loc[temp['Fare']==curr_fare]

        temp=temp.loc[temp['Cabin']!='n']

        #if there are values in the dataframe then replace n with the corresponding cabin 

        temp=temp.reset_index()

       

        

        if temp.shape[0]!=0:

            

            cabin=temp['Cabin'][0]

            tit_train['Cabin'][x]=cabin

    

    
(tit_train['Cabin']=='n').describe()
tit_train=tit_train.drop(columns=['Cabin'])

tit_train.head()


#settle age first 

title_of_pas=re.compile('\w[A-z]+\.')

tit_test['title']=(tit_test.apply(lambda x: title_of_pas.findall(x.Name)[0],axis=1))

tit_test.head()





ages=pd.DataFrame(tit_test[['Age','title']].groupby(['title']).mean()).to_dict()

ages_final=ages['Age']

print(ages_final)



tit_test['Age']=(tit_test.apply(lambda x: ages_final[x.title] if np.isnan(x.Age) else x.Age,axis=1))

print(tit_test.isnull().sum(axis = 0))



#impute the missing one age with 'miss age'



tit_test['Age']=tit_test["Age"].fillna(21.8)

print(tit_test.isnull().sum(axis = 0))
tit_test=tit_test.drop(columns=['Cabin'])
tit_test.loc[tit_test['Fare'].isnull()]
temp=tit_test.loc[tit_test['Ticket']=='3701']

temp=temp.loc[temp['Pclass']==3]

temp
pd.DataFrame(tit_test[['Pclass','Embarked','Fare']].groupby(['Pclass','Embarked']).mean())
tit_test['Fare']=tit_train["Fare"].fillna(36.5)
print(tit_test.isnull().sum(axis = 0))
tit_test.loc[tit_test['Embarked'].isnull()]
tit_test.groupby('Embarked').nunique()
tit_test['Embarked']=tit_train["Embarked"].fillna('S')
print(tit_test.isnull().sum(axis = 0))
tit_train.head()
tit_test.head()
#1 age to categorical

def age(x):

    if 0<=x<=14:

        return 'Child'

    elif 14<=x<=24:

        return "Youth"

    elif 25<=x<=64:

        return 'Adult'

    else:

        return 'Senior'



    

tit_train['Age']=tit_train["Age"].apply(lambda x:age(int(x)))



tit_test['Age']=tit_test["Age"].apply(lambda x:age(int(x)))

    

  

    
def string_test(x):

    try: 

        target=int(x)

    except:

        return 'Special'

    else:

        return 'Normal'

            

tit_train['Ticket']=tit_train["Ticket"].apply(lambda x:string_test(x))

tit_test['Ticket']=tit_test["Ticket"].apply(lambda x:string_test(x))







answer_sheet=tit_test["PassengerId"]

tit_train=tit_train.drop(columns=['Name','PassengerId','title'])

tit_train=pd.get_dummies(tit_train)

print(tit_train.shape)



tit_test=tit_test.drop(columns=['Name','PassengerId','title'])

tit_test=pd.get_dummies(tit_test)

print(tit_test.shape)

tit_train.columns
tit_test.columns
tit_train['Parch_9']=0

tit_train.head()


x_train=tit_train.iloc[:,1:]

y_train=tit_train.iloc[:,0]

x_test=tit_test





# Feature Scaling

from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.ensemble import RandomForestClassifier



regressor = RandomForestClassifier(n_estimators=20, random_state=0)

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_train)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print(confusion_matrix(y_train,y_pred))

print(classification_report(y_train,y_pred))

print(accuracy_score(y_train, y_pred))
temp=tit_train.iloc[:,1:]

feature_importances = pd.DataFrame(regressor.feature_importances_,index = temp.columns,columns=['importance']).sort_values('importance',ascending=False)

feature_importances.iloc[:6,:]
answer=regressor.predict(x_test)

answers=pd.DataFrame({'PassengerId':answer_sheet,'Survived':answer})

answers.head()
answers.to_csv('rf.csv',index=False)
from sklearn.model_selection import KFold,RandomizedSearchCV,GridSearchCV

from sklearn.linear_model import LogisticRegression
#log-r model 

logistic=LogisticRegression(solver='liblinear')

#create a list of types of hyperparameters 

penalty=['l1','l2']

#values to test 

C=np.logspace(0,4,10)

hyperparam=dict(C=C,penalty=penalty)

#fit the model using gridsearch 

clf=GridSearchCV(logistic,hyperparam,cv=10,verbose=0)

best_model1=clf.fit(x_train,y_train)

print(best_model1.best_estimator_.get_params()['penalty'])

print(best_model1.best_estimator_.get_params()['C'])
from sklearn.model_selection import KFold,RandomizedSearchCV,GridSearchCV,cross_val_score

kfold=KFold(n_splits=5,random_state=None)

model=LogisticRegression(C=1,penalty='l2')

results=cross_val_score(model,x_train,y_train,cv=kfold)

print(results.mean()*100)
clf=model.fit(x_train,y_train)

answers2=clf.predict(x_test)

answers=pd.DataFrame({'PassengerId':answer_sheet,'Survived':answers2})

answers.to_csv('log1.csv',index=False)
#log-r model 

logistic=LogisticRegression(solver='liblinear')

#create a list of types of hyperparameters 

penalty=['l1','l2']

#values to test 

C=np.linspace(1,200)

hyperparam=dict(C=C,penalty=penalty)

#fit the model using gridsearch 

clf2=RandomizedSearchCV(logistic,hyperparam,n_iter=100,random_state=41)

clf2.fit(x_train,y_train)

print(clf2.best_params_)

kfold=KFold(n_splits=5,random_state=None)

model=LogisticRegression(C=5.0612245,penalty='l2')

results=cross_val_score(model,x_train,y_train,cv=kfold)

print(results.mean()*100)
clf=model.fit(x_train,y_train)

answers2=clf.predict(x_test)

answers=pd.DataFrame({'PassengerId':answer_sheet,'Survived':answers2})

answers.to_csv('log2.csv',index=False)