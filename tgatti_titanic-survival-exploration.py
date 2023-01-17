#import Statements:

import numpy as np

import random

import pandas as pd

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

from sklearn import preprocessing

import seaborn as sns



from sklearn import svm

from sklearn.ensemble import RandomForestClassifier



# loading the train & test csv files as a DataFrame

train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
# Looking at info of the data sets

train_df.info()

print("----------------------------")

test_df.info()
#Embarked

sns.countplot(x="Embarked", data=train_df)
train_df['Embarked'] = train_df['Embarked'].fillna(value='S')

train_df.info()

print("----------------------------")

test_df.info()
#Cabin

train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis=1)
#Fare

fare_Plot = test_df['Fare'].dropna(axis = 0)

fare_Plot=fare_Plot.astype(int)

sns.distplot(fare_Plot , bins= 20)

median_Fare = test_df['Fare'].median()

print (median_Fare)
#Fare

test_df['Fare'] = test_df['Fare'].fillna(value=median_Fare)



train_df.info()

print("----------------------------")

test_df.info()
# Age

#Looking at the data

age_plot = train_df.loc[train_df['Survived'] == 0, 'Age'].dropna(axis = 0)

age_plot_survived = train_df.loc[train_df['Survived'] == 1, 'Age'].dropna(axis = 0)

age_STD = age_plot.std()

age_mean = age_plot.mean()

age_median = age_plot.median()

print (age_STD)

print (age_mean)

print ('-------------------')

age_STD_survived = age_plot_survived.std()

age_mean_survived = age_plot_survived.mean()

age_median_survived = age_plot_survived.median()

print (age_STD_survived)

print (age_mean_survived)
#Age

#Filling in the data.

train_null= train_df.loc[train_df['Age'].isnull() == True]

test_null= test_df.loc[test_df['Age'].isnull() == True]

train_index = train_null['Age'].index.tolist()

test_index = test_null['Age'].index.tolist()

min_age_range = age_mean - age_STD

min_age_range=int(min_age_range)

max_age_range = age_mean + age_STD

max_age_range = int(max_age_range)



train_filler =np.random.randint(min_age_range, high=max_age_range, size=len(train_null))

test_filler = np.random.randint(min_age_range, high=max_age_range, size=len(test_null))



train_Replace = pd.Series(train_filler, index=train_index)

train_df['Age']= train_df['Age'].fillna(train_Replace)



test_Replace = pd.Series(test_filler, index=test_index)

test_df['Age']= test_df['Age'].fillna(test_Replace)



train_df.info()

print("----------------------------")

test_df.info()




from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(train_df['Sex'])

print(list(le.classes_))

train_df['Sex']=le.transform(train_df['Sex'])



le.fit(test_df['Sex'])

test_df['Sex']=le.transform(test_df['Sex'])
#Passenger

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

sns.distplot(train_df.loc[train_df['Survived'] == 0, 'PassengerId'] , bins= 20, ax=ax1)

sns.distplot(train_df.loc[train_df['Survived'] == 1, 'PassengerId'] , bins= 20, ax=ax2)
#Embarked

fig, (ax1, ax2) = plt.subplots(2, sharex=True)

sns.countplot(x="Embarked", hue='Survived', data=train_df, ax = ax1)

sns.countplot(x="Embarked", hue='Sex', data=train_df, ax = ax2)
#Ticket

plot_df = train_df

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(plot_df['Ticket'])

plot_df['Ticket'] = le.transform(plot_df['Ticket'])

    

count_plot = train_df['Ticket'].value_counts().head(20)

count_plot_index=count_plot.index.tolist()



ticket_count=[]

survived_count=[]

WC_count=[]

for x in range(0, len(count_plot_index)):

    new_ticket_count = 0

    new_survived_count = 0

    new_WC_count = 0

    for y in range(0,len(train_df['Ticket'])):

        if train_df['Ticket'][y]== count_plot_index[x]:

            new_ticket_count =new_ticket_count+1

            if (train_df['Age'][y]<16) or (train_df['Sex'][y]==0):

                new_WC_count = new_WC_count+ 1

            if train_df['Survived'][y]== 1:

                new_survived_count = new_survived_count+1

                

    ticket_count.append(new_ticket_count)

    survived_count.append(new_survived_count)

    WC_count.append(new_WC_count)

ag_count_plot = pd.DataFrame({'Ticket_Number': count_plot_index, 

                              'Ticket_Count': ticket_count,

                              'Survived_Count':survived_count,

                              'WC_Count': WC_count})

#To Do add Class



g =sns.barplot(x="Ticket_Number", y='Ticket_Count', data=ag_count_plot, color = "red")

topbar =sns.barplot(x="Ticket_Number", y='WC_Count', data=ag_count_plot, color = 'yellow', )

bottombar =sns.barplot(x="Ticket_Number", y='Survived_Count', data=ag_count_plot, linewidth=2.5, facecolor=(1, 1, 1, 0))



print(ag_count_plot)


#storing PassengerId for Submission:

Test_PId= test_df['PassengerId']



#Droping PassengerId

train_df=train_df.drop(['PassengerId'], axis=1)

test_df=test_df.drop(['PassengerId'], axis=1)

#Droping Embarked

train_df=train_df.drop(['Embarked'], axis=1)

test_df=test_df.drop(['Embarked'], axis=1)





train_df.info()

print("----------------------------")

test_df.info()
TitleTrain=[]

TitleTest=[]

trainTitle_index =  train_df['Name'].index.tolist()

testTitle_index =  test_df['Name'].index.tolist()



for X in train_df['Name']:

    NameTitle = X.partition(', ')[-1].rpartition('.')[0] 

    TitleTrain.append(NameTitle)

    

for X in test_df['Name']:

    NameTitle = X.partition(', ')[-1].rpartition('.')[0] 

    TitleTest.append(NameTitle)



trainTitle_Replace = pd.Series(TitleTrain, index=trainTitle_index)

train_df['Name']= trainTitle_Replace



testTitle_Replace = pd.Series(TitleTest, index=testTitle_index)

test_df['Name']= testTitle_Replace



#Changing MRS and MISS to one category:

train_df.loc[train_df['Name'] == 'Mrs', 'Name'] = 'Miss'

test_df.loc[test_df['Name'] == 'Mrs', 'Name'] = 'Miss'



NameListIndex = train_df['Name'].value_counts().index.tolist()



NameList = train_df['Name'].value_counts().tolist()

for x in range(0,len(NameListIndex)):

    if NameList[x] <10:

        train_df.loc[train_df['Name'] == NameListIndex[x], 'Name'] = 'Misc'

    else:

        train_df.loc[train_df['Name'] == NameListIndex[x], 'Name'] = NameListIndex[x]



NameTestListIndex = test_df['Name'].value_counts().index.tolist()

NameTestList = test_df['Name'].value_counts().tolist()

for x in range(0,len(NameTestListIndex)):

    if NameTestList[x] <10:

        test_df.loc[test_df['Name'] == NameTestListIndex[x], 'Name'] = 'Misc'

    else:

        test_df.loc[test_df['Name'] == NameTestListIndex[x], 'Name'] = NameTestListIndex[x]



sns.countplot(x="Name", hue="Survived", data=train_df)

print(train_df['Name'].value_counts())


le.fit(train_df['Name'])

train_df['Name'] = le.transform(train_df['Name'])

le.fit(test_df['Name'])

test_df['Name'] = le.transform(test_df['Name'])



le.fit(test_df['Ticket'])

test_df['Ticket'] = le.transform(test_df['Ticket'])


#Now to split the data into a form that can be run in a model 

X_train = train_df.drop(['Survived'], axis=1)

Y_train = train_df['Survived']

X_Pred = test_df

print(X_train.columns.values)
from sklearn import  grid_search 

parameters = {'n_estimators':[100, 150, 200]}

random_forest = RandomForestClassifier()

RF_clf = grid_search.GridSearchCV(random_forest, parameters)

RF_clf.fit(X_train, Y_train)



#Cross Validation Output

from sklearn.cross_validation import KFold, cross_val_score

k_fold = KFold(len(Y_train), n_folds=10, shuffle=True, random_state=0)

CV_AVG = cross_val_score(random_forest, X_train, Y_train, cv=k_fold, n_jobs=1)

print (sum(CV_AVG) / float(len(CV_AVG)))



# Submission Output

#Y_Pred = random_forest.predict(X_Pred)

#print(Y_Pred)
from sklearn.ensemble import AdaBoostClassifier



parameters = {'n_estimators':[100, 150, 200]}

Ada = AdaBoostClassifier()

Ada_clf = grid_search.GridSearchCV(Ada, parameters)

Ada_clf.fit(X_train, Y_train)



#Cross Validation Output

k_fold = KFold(len(Y_train), n_folds=10, shuffle=True, random_state=0)

CV_AVG = cross_val_score(Ada_clf, X_train, Y_train, cv=k_fold, n_jobs=1)

print (sum(CV_AVG) / float(len(CV_AVG)))



# Submission Output

Y_Pred = Ada_clf.predict(X_Pred)

print(Y_Pred)
submission = pd.DataFrame({

        "PassengerId": Test_PId,

        "Survived": Y_Pred

    })

submission.to_csv('titanic.csv', index=False)