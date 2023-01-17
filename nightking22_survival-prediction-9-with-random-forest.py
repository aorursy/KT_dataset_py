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
#Importing necessary packages and loading the train and test dataset



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.rc('font',size=14)

import seaborn as sns

from sklearn import preprocessing

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFECV

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score

import scipy.stats as stats

train= pd.read_csv("../input/titanic/train.csv")

test=  pd.read_csv("../input/titanic/test.csv")

import re



from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
train.info()
test.info()
train.head(10)
train['Title']=train.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.',x).group(1))

test['Title']=test.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))
#Countplot with Survived features

plt.figure(figsize=(12,5))

sns.countplot(x='Title',data=train,hue='Survived',palette='hls')

plt.show()
Title_Dictionary = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "the Countess":"Royalty",

        "Countess":   "Royalty",

        "Dona":       "Royalty",

        "Lady" :      "Royalty",

        "Mme":        "Miss",

        "Ms":         "Miss",

        "Mrs" :       "Miss",

        "Mlle":       "Miss",

        "Miss" :      "Miss",

        "Mr" :        "Mr",

        "Master" :    "Master"

                   }

# we map each title to correct category

train['Title'] = train.Title.map(Title_Dictionary)

test['Title'] = test.Title.map(Title_Dictionary)

plt.figure(figsize=(12,5))

sns.countplot(x='Title',data=train,hue='Survived',palette='hls')

plt.show()
train['Surname']=train.Name.str.split(',').str[0]

test['Surname']=test.Name.str.split(',').str[0]

print(train[['Name','Surname']].head())

print(test[['Name','Surname']].head())
train['Cabin_notnull']=np.where(train['Cabin'].isnull(),0,1)

test['Cabin_notnull']=np.where(test['Cabin'].isnull(),0,1)

pd.crosstab(train['Cabin_notnull'],train['Survived'])
plt.figure()

sns.countplot(x="Embarked",data=train,hue="Survived",palette="hls")

plt.show()
train['Embarked'].fillna(train['Embarked'].value_counts().idxmax(),inplace=True)

test['Embarked'].fillna(test['Embarked'].value_counts().idxmax(),inplace=True)
train.groupby(['Embarked','Sex','Pclass','Title'])['Age'].median()
#Flling the null age values with respective group's median values

train.Age.loc[train.Age.isnull()] = train.groupby(['Embarked','Sex','Pclass','Title']).Age.transform('median')

print(train["Age"].isnull().sum())

test.Age.loc[test.Age.isnull()] = test.groupby(['Embarked','Sex','Pclass','Title']).Age.transform('median')
train.Age.describe()
#creating the intervals that we need to cut each range of ages

interval = (0, 5, 12, 18, 23, 36, 60, 120) 



#Seting the names that we want use to the categorys

cats = ['babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']



# Applying the pd.cut and using the parameters that we created 

train["Age_cat"] = pd.cut(train.Age, interval, labels=cats)



# Printing the new Category

train["Age_cat"].head()

test["Age_cat"] = pd.cut(test.Age, interval, labels=cats)
#train.groupby(['Pclass','Title','Sex'])['Fair_Fare'].median()
train_ticket_freq=train.groupby('Ticket').Ticket.count()

test_ticket_freq=test.groupby('Ticket').Ticket.count()
train_ticket_count=pd.DataFrame({'Ticket':train_ticket_freq.index,

                          'Count':train_ticket_freq.values})

test_ticket_count=pd.DataFrame({'Ticket':test_ticket_freq.index,

                          'Count':test_ticket_freq.values})
train_ticket_count.head(5)
train=pd.merge(train,

                train_ticket_count,

                on='Ticket',how='left')

test=pd.merge(test,

                test_ticket_count,

                on='Ticket',how='left')

train.head()

test.head()
train['Fair_Fare']=train['Fare']/train['Count']

train.head()

test['Fair_Fare']=test['Fare']/train['Count']
test.loc[test.Fare.isnull()]
train.Fair_Fare.loc[train.Fair_Fare.isnull()] = train.groupby(['Pclass','Title','Sex']).Fair_Fare.transform('median')

test.Fair_Fare.loc[test.Fair_Fare.isnull()] = test.groupby(['Pclass','Title','Sex']).Fair_Fare.transform('median')
train['Family']=train['SibSp']+train['Parch']+1

test['Family']=test['SibSp']+test['Parch']+1
# Explore Parch feature vs Survived

g  = sns.factorplot(x="Family",y="Survived",data=train, kind="bar", size = 6,palette = "hls")

g = g.set_ylabels("survival probability")
train['IsAlone'] = 1 #initialize to yes/1 is alone

train['IsAlone'].loc[train['Family'] > 1] = 0 # now update to no/0 if family size is greater than 1

test['IsAlone'] = 1 #initialize to yes/1 is alone

test['IsAlone'].loc[test['Family'] > 1] = 0 # now update to no/0 if family size is greater than 1
train.head()
plt.figure()

sns.countplot(x='Family',data=train,hue='Survived',palette='hls')

plt.show()
a=[]

ts=test['Surname']

for x in ts:

    if x not in a:

        a.append(x)

print(len(a))
import category_encoders as ce

target_enc=ce.TargetEncoder(cols=['Embarked','Sex'])

# Fit the encoder using the categorical features and target

target_enc.fit(train[['Embarked','Sex']],train['Survived'])

train=train.join(target_enc.transform(train[['Embarked','Sex']]).

                add_suffix('_target'))

train.head()
test['Sex_target']=0.742038

test.loc[test['Sex']=='female']['Sex_target']=0.188908

test.head()
#Embarked Target Values to be imputed in Test Data

print(train[['Embarked','Embarked_target']].head(10))
test['Embarked_target']=0.339009

test.loc[test['Embarked']=='C']['Embarked_target']=0.553571

test.loc[test['Embarked']=='Q']['Embarked_target']=0.389610

test.head()
train = pd.get_dummies(train, columns=["Title","Age_cat"],\

                          prefix=["Prefix","Age"], drop_first=True)



test = pd.get_dummies(test, columns=["Title","Age_cat"],\

                          prefix=["Prefix","Age"], drop_first=True)
#Checking data shapes

print(train.shape)

print(test.shape)
train.head()
#Label Encoding Surnames

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

train_Surname=train.Surname

test_Surname=test.Surname

se=train_Surname.append(test_Surname)

total=pd.DataFrame({'Surname_Encoded':se.values})





total=total[['Surname_Encoded']].apply(encoder.fit_transform)

test_se=total[891:]

train_se=total[:891]

test['Surname_Encoded']=test_se

for i,x in enumerate(test_se.Surname_Encoded):

    test.Surname_Encoded.iloc[i]=int(x)

train['Surname_Encoded']=train_se.Surname_Encoded

test['Surname_Encoded']=pd.to_numeric(test['Surname_Encoded'], downcast='signed')
del train['Name']

del train['Ticket']

del train['PassengerId']

del train['Age']

del train['Cabin']

del train['Parch']

del train['SibSp']

del train['Fare']

del train['Surname']

del test['Surname']

del test['Fare']

del test['PassengerId']

del test['Age']

del test['Ticket']

del test['Name']

del test['Cabin']

del test['Parch']

del test['SibSp']

del train['Embarked']

del test['Embarked']

del train['Sex']

del test['Sex']

train.info()

plt.figure(figsize=(15,12))

plt.title('Correlation of Features for Train Set')

sns.heatmap(train.astype(float).corr(),vmax=1.0,annot=True)

plt.show()

train_=train['Survived']

print(train_.head())

y=train_

print(y.head())

y.shape

trained=train.drop(['Survived'],axis=1)

X=trained

print(X.shape)

print(y.shape)

print(test.shape)

print(X.columns)

print(test.columns)
imp_features=['Prefix_Mr','Sex_target','Prefix_Miss','Fair_Fare','Surname_Encoded','Pclass','Family','Count','Cabin_notnull','Embarked_target']

X=X[imp_features]

test=test[imp_features]
'''from sklearn.ensemble import RandomForestClassifier

#Decision Tree Algorithm

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from scipy.stats import randint 

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

param_dist = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

tree = RandomForestClassifier(random_state=42) 

  

# Instantiating RandomizedSearchCV object 

tree_cv = RandomizedSearchCV(tree, param_dist, cv = 10) 

  

tree_cv.fit(X, y)



# Print the tuned parameters and score 

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_)) 

print("Best score is {}".format(tree_cv.best_score_)) '''
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

i=1

k=10

scores=[0]*k

#Random Forest Best

model=RandomForestClassifier(random_state=42,n_estimators= 200, min_samples_split= 2, min_samples_leaf= 4, 

                             max_features='auto' ,max_depth=50, bootstrap= False)

#model=RandomForestClassifier()

#Logistic Regression Best

#model = LogisticRegression(random_state=1,C = 10)   

kf = StratifiedKFold(n_splits=k,random_state=1,shuffle=True)

for train_index,test_index in kf.split(X,y):    

    #print('\n{} of kfold {}'.format(i,kf.n_splits))    

    xtr,xvl = X.loc[train_index],X.loc[test_index]    

    ytr,yvl = y[train_index],y[test_index]        

    #print(xtr.head())

    model.fit(xtr, ytr)    

    pred_test = model.predict(xvl) 

    score = accuracy_score(yvl,pred_test)    

    print('accuracy_score',score)    

    scores[i-1]=score

    i+=1

    #pred_test = model.predict(test)

    #pred=model.predict_proba(xvl)[:,1]

print("AVERAGE CROSS VALIDATION SCORE: ",sum(scores)/len(scores))

feature_importances = pd.DataFrame(model.feature_importances_,

                                   index = X.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)
pred_test = model.predict(test)

submission = pd.read_csv("../input/titanic/gender_submission.csv",index_col='PassengerId')

submission['Survived'] = pred_test.astype(int)

submission.to_csv('Titanic_RF_new.csv')