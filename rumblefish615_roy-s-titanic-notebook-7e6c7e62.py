# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.info())

#print(train.describe())
import re

pattern = re.compile(r'.*?,(.*?)\.')

def getTitle(x):

    result = pattern.search(x)

    if result:

        return result.group(1).strip()

    else:

        return ''



train['Title'] = train['Name'].map(getTitle)

test['Title'] = test['Name'].map(getTitle)

#train['Title'].value_counts()



#Check how many rows missing the Age by Title

print(train['Title'][train['Age'].isnull()].value_counts())

print(test['Title'][test['Age'].isnull()].value_counts())



#Set the missing Age of Title 'Master' 

master_age_mean = train['Age'][(train['Title']=='Master')&(train['Age']>0)].mean()

train.loc[train[(train['Title']=='Master')&(train['Age'].isnull())].index, 'Age'] = master_age_mean

test.loc[test[(test['Title']=='Master')&(test['Age'].isnull())].index, 'Age'] = master_age_mean



#Set the missing Age of Title 'Mr' 

mr_age_mean = train['Age'][(train['Title']=='Mr')&(train['Age']>0)].mean()

train.loc[train[(train['Title']=='Mr')&(train['Age'].isnull())].index, 'Age'] = mr_age_mean

test.loc[test[(test['Title']=='Mr')&(test['Age'].isnull())].index, 'Age'] = mr_age_mean



#Set the missing Age of Title 'Miss' or 'Ms'

miss_age_mean = train['Age'][(train['Title']=='Miss')&(train['Age']>0)].mean()

train.loc[train[(train['Title']=='Miss')&(train['Age'].isnull())].index, 'Age'] = miss_age_mean

test.loc[test[((test['Title']=='Miss')|(test['Title']=='Ms'))&(test['Age'].isnull())].index, 'Age'] = miss_age_mean



#Set the missing Age of Title 'Mrs' 

mrs_age_mean = train['Age'][(train['Title']=='Mrs')&(train['Age']>0)].mean()

train.loc[train[(train['Title']=='Mrs')&(train['Age'].isnull())].index, 'Age'] = mrs_age_mean

test.loc[test[(test['Title']=='Mrs')&(test['Age'].isnull())].index, 'Age'] = mrs_age_mean



#Set the missing Age of Title 'Dr' 

dr_age_mean = train['Age'][(train['Title']=='Dr')&(train['Age']>0)].mean()

train.loc[train[(train['Title']=='Dr')&(train['Age'].isnull())].index, 'Age'] = dr_age_mean

test.loc[test[(test['Title']=='Mrs')&(test['Age'].isnull())].index, 'Age'] = dr_age_mean



print(train['Age'].describe())

print(test['Age'].describe())
import matplotlib.pyplot as plt

alpha = 0.6

fig = plt.figure(figsize=(8, 12))

grouped = train.groupby(['Survived'])

group0 = grouped.get_group(0)

group1 = grouped.get_group(1)



plot_rows = 5

plot_cols = 2

#ax1 = fig.add_subplot(2,2,1)

ax1 = plt.subplot2grid((plot_rows,plot_cols), (0,0), rowspan=1, colspan=1)

plt.hist([group0.Age, group1.Age], bins=16, range=(0,80), stacked=True, 

        label=['Not Survived', 'Survived'], alpha=alpha)

plt.legend(loc='best', fontsize='x-small')

ax1.set_title('Survival distribution by Age')



#ax2 = fig.add_subplot(2,2,2)

ax2 = plt.subplot2grid((plot_rows,plot_cols), (0,1), rowspan=1, colspan=1)

n, bins, patches = plt.hist([group0.Pclass, group1.Pclass], bins=5, range=(0,5), 

        stacked=True, label=['Not Survived', 'Survived'], alpha=alpha)

plt.legend(loc='best', fontsize='x-small')

ax2.set_xticks([1.5, 2.5, 3.5])

ax2.set_xticklabels(['Class1', 'Class2', 'Class3'], fontsize='small')

ax2.set_yticks([0, 150, 300, 450, 600, 750])

ax2.set_title('Survival distribution by Pclass')



#ax3 = fig.add_subplot(2,2,3)

ax3 = plt.subplot2grid((plot_rows,plot_cols), (1,0), rowspan=1, colspan=2)

ax3.set_title('Survival distribution by Sex')

patches, l_texts, p_texts = plt.pie(train.groupby(['Survived', 'Sex']).size(), 

        labels=['Not Survived Female', 'Not Survived Male', 'Survived Female', 'Survived Male'],

        autopct='%3.1f', labeldistance = 1.1, pctdistance = 0.6)

plt.legend(loc='upper right', fontsize='x-small')

for t in l_texts:

    t.set_size(10)

for p in p_texts:

    p.set_size(10)

#plt.legend(loc='best', fontsize='x-small')

plt.axis('equal')



ax4 = plt.subplot2grid((plot_rows,plot_cols), (2,0), rowspan=1, colspan=1)

ax4.set_title('Survival distribution by SibSp')

plt.hist([group0.SibSp, group1.SibSp], bins=9, range=(0,9), stacked=True, 

        label=['Not Survived', 'Survived'], log=False, alpha=alpha)

plt.legend(loc='best', fontsize='x-small')



ax5 = plt.subplot2grid((plot_rows,plot_cols), (2,1), rowspan=1, colspan=1)

ax5.set_title('Survival distribution by SibSp')

plt.hist([group0[group0.SibSp>1].SibSp, group1[group1.SibSp>1].SibSp], bins=8, range=(1, 9), stacked=True, 

        label=['Not Survived', 'Survived'], log=False, alpha=alpha)

plt.legend(loc='best', fontsize='x-small')



ax6 = plt.subplot2grid((plot_rows,plot_cols), (3,0), rowspan=1, colspan=1)

ax6.set_title('Survival distribution by Parch')

plt.hist([group0.Parch, group1.Parch], bins=7, range=(0,7), stacked=True, 

        label=['Not Survived', 'Survived'], log=False, alpha=alpha)

plt.legend(loc='best', fontsize='x-small')



ax7 = plt.subplot2grid((plot_rows,plot_cols), (3,1), rowspan=1, colspan=1)

ax7.set_title('Survival distribution by Parch')

plt.hist([group0[group0.Parch>1].Parch, group1[group1.Parch>1].Parch], bins=6, range=(1, 7), stacked=True, 

        label=['Not Survived', 'Survived'], log=False, alpha=alpha)

plt.legend(loc='best', fontsize='x-small')



ax8 = plt.subplot2grid((plot_rows,plot_cols), (4,0), rowspan=1, colspan=1)

ax8.set_title('Survival distribution by Fare')

plt.hist([group0.Fare, group1.Fare], bins=11, range=(0, 550), stacked=True, 

        label=['Not Survived', 'Survived'], log=False, alpha=alpha)

plt.legend(loc='best', fontsize='x-small')



ax9 = plt.subplot2grid((plot_rows,plot_cols), (4,1), rowspan=1, colspan=1)

ax9.set_title('Survival distribution by Fare')

plt.hist([group0[group0.Fare>50].Fare, group1[group1.Fare>50].Fare], bins=11, range=(0, 550), stacked=True, 

        label=['Not Survived', 'Survived'], log=False, alpha=alpha)

plt.legend(loc='best', fontsize='x-small')

plt.subplots_adjust(wspace=0.3, hspace=0.3)
childgrouped = train[train['Age']<19].groupby(['Survived'])

childgroup0 = childgrouped.get_group(0)

childgroup1 = childgrouped.get_group(1)

parent = train[(train['Age']>18)&(train['Parch']>0)]



merged0 = pd.merge(childgroup0, parent, how='left', on='Ticket')

merged0 = merged0[['Survived_x', 'Sex_x', 'Age_x', 'Survived_y', 'Sex_y', 'Age_y', 'Ticket']]

merged0 = merged0[merged0.Survived_y>=0]

fig = plt.figure(figsize=(8, 4))

plot_rows = 2

plot_cols = 1

ax1 = plt.subplot2grid((plot_rows,plot_cols), (0,0), rowspan=1, colspan=1)

bottom = merged0.Survived_y.value_counts().index

width1 = merged0[merged0['Sex_y']=='female'].Survived_y.value_counts()

plt.barh(bottom, width1, 0.8, 0.0, color='blue', label='mother', alpha=0.6)

width2 = merged0[merged0['Sex_y']=='male'].Survived_y.value_counts()

plt.barh(width2.index, width2, 0.8, width1[width2.index], color='green', label='father', alpha=0.6)

plt.legend(loc='best', fontsize='x-small')

ax1.set_yticks([0.4, 1.4])

ax1.set_yticklabels(['Not Survived Parents', 'Survived Parents'], fontsize='small')

ax1.set_title('Parents survival distribution by not survived child')



merged1 = pd.merge(childgroup1, parent, how='left', on='Ticket')

merged1 = merged1[['Survived_x', 'Sex_x', 'Age_x', 'Survived_y', 'Sex_y', 'Age_y', 'Ticket']]

merged1 = merged1[merged1.Survived_y>=0]

ax2 = plt.subplot2grid((plot_rows,plot_cols), (1,0), rowspan=1, colspan=1)

bottom = merged1.Survived_y.value_counts().index

width1 = merged1[merged1['Sex_y']=='female'].Survived_y.value_counts()

plt.barh(bottom, width1, 0.8, 0.0, color='blue', label='mother', alpha=0.6)

width2 = merged1[merged1['Sex_y']=='male'].Survived_y.value_counts()

plt.barh(width2.index, width2, 0.8, width1[width2.index], color='green', label='father', alpha=0.6)

plt.legend(loc='best', fontsize='x-small')

ax2.set_yticks([0.4, 1.4])

ax2.set_yticklabels(['Not Survived Parents', 'Survived Parents'], fontsize='small')

ax2.set_title('Parents survival distribution by survived child')



plt.subplots_adjust(hspace=1.0)
grouped = train[train['Parch']>0].groupby('Ticket')

grouped.get_group('349909')
ticket = train['Ticket'][train['Parch']==0]

ticket_dup = ticket.duplicated(False)

index = ticket_dup[ticket_dup==True].index

new_train = train.loc[index]

new_train['FriendsSurvived'] = -1

for i in range(0, len(index)):

    ticketID = new_train.loc[index[i]]['Ticket']

    passengerID = new_train.loc[index[i]]['PassengerId']

    survived = new_train['Survived'][(new_train['Ticket']==ticketID)&(new_train['PassengerId']!=passengerID)]

    new_train.loc[index[i], 'FriendsSurvived'] = round(float(survived.sum())/len(survived))

#print(new_train[['Ticket', 'FriendsSurvived', 'Survived', 'Sex']])

print(new_train[(new_train['Sex']=='female')&(new_train['Survived']==0)].FriendsSurvived.value_counts())

print(new_train[(new_train['Sex']=='male')&(new_train['Survived']==0)].FriendsSurvived.value_counts())

print(new_train[(new_train['Sex']=='female')&(new_train['Survived']==1)].FriendsSurvived.value_counts())

print(new_train[(new_train['Sex']=='male')&(new_train['Survived']==1)].FriendsSurvived.value_counts())
fig = plt.figure(figsize=(8, 4))

plot_rows = 2

plot_cols = 1

ax1 = plt.subplot2grid((plot_rows,plot_cols), (0,0), rowspan=1, colspan=1)

width1 = new_train[(new_train['Sex']=='female')&(new_train['Survived']==0)].FriendsSurvived.value_counts()

plt.barh(width1.index, width1, 0.8, 0.0, color='blue', label='Not survived female', alpha=0.6)

width2 = new_train[(new_train['Sex']=='male')&(new_train['Survived']==0)].FriendsSurvived.value_counts()

plt.barh(width2.index, width2, 0.8, [width1, 0.0], color='green', label='Not survived male', alpha=0.6)

plt.legend(loc='best', fontsize='x-small')

ax1.set_yticks([0.4, 1.4])

ax1.set_yticklabels(['Friends not survived', 'Friends survived'], fontsize='small')

ax1.set_title('Not survived sex distribution by friends survival')



ax2 = plt.subplot2grid((plot_rows,plot_cols), (1,0), rowspan=1, colspan=1)

width1 = new_train[(new_train['Sex']=='female')&(new_train['Survived']==1)].FriendsSurvived.value_counts()

plt.barh(width1.index, width1, 0.8, 0.0, color='blue', label='Survived female', alpha=0.6)

width2 = new_train[(new_train['Sex']=='male')&(new_train['Survived']==1)].FriendsSurvived.value_counts()

plt.barh(width2.index, width2, 0.8, width1[width2.index], color='green', label='Survived male', alpha=0.6)

plt.legend(loc='best', fontsize='x-small')

ax2.set_yticks([0.4, 1.4])

ax2.set_yticklabels(['Friends not survived', 'Friends survived'], fontsize='small')

ax2.set_title('Survived sex distribution by friends survival')



plt.subplots_adjust(hspace=1.0)
ticket = train['Ticket'][train['Parch']==0]

ticket_dup = ticket.duplicated(False)

print(ticket_dup)
sex_to_int = {'male':1, 'female':0}

train['SexInt'] = train['Sex'].map(sex_to_int)

embark_to_int = {'S': 0, 'C':1, 'Q':2}

train['EmbarkedInt'] = train['Embarked'].map(embark_to_int)

train['EmbarkedInt'] = train['EmbarkedInt'].fillna(0)

#train['AgeInt'] = train['Age'].fillna(train['Age'].mean())

print(train.describe())

test['SexInt'] = test['Sex'].map(sex_to_int)

test['EmbarkedInt'] = test['Embarked'].map(embark_to_int)

test['EmbarkedInt'] = test['EmbarkedInt'].fillna(0)

#test['AgeInt'] = test['Age'].fillna(train['Age'].mean())

test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

train['FamilySize'] = train['SibSp'] + train['Parch']

test['FamilySize'] = test['SibSp'] + test['Parch']

#test.describe()
ticket = train[train['Parch']==0]

ticket = ticket.loc[ticket.Ticket.duplicated(False)]

grouped = ticket.groupby(['Ticket'])

train['Friends'] = 0

#train['FriendsNumber'] = 0

#train['FriendsSex'] = -1

#train['FriendsSurvived'] = -1

train['Male_Friends_Survived'] = 0

train['Male_Friends_NotSurvived'] = 0

train['Female_Friends_Survived'] = 0

train['Female_Friends_NotSurvived'] = 0

print(type(grouped.groups))

for (k, v) in grouped.groups.items():

    for i in range(0, len(v)):

        train.loc[v[i], 'Friends'] = 1

        #train.loc[v[i], 'FriendsSex'] = train[(train.Ticket==k)&(train.index!=v[i])].SexInt.sum()/(len(v)-1)

        #train.loc[v[i], 'FriendsSurvived'] = train[(train.Ticket==k)&(train.index!=v[i])].Survived.sum()/(len(v)-1)

        train.loc[v[i], 'Male_Friends_Survived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='male')&(train.Survived==1)].Survived.count()

        train.loc[v[i], 'Male_Friends_NotSurvived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='male')&(train.Survived==0)].Survived.count()

        train.loc[v[i], 'Female_Friends_Survived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='female')&(train.Survived==1)].Survived.count()

        train.loc[v[i], 'Female_Friends_NotSurvived'] = train[(train.Ticket==k)&(train.index!=v[i])&(train.Sex=='female')&(train.Survived==0)].Survived.count()

#print(train[train.Friends>0].head(100))

#ticket = ticket.loc[ticket.duplicated(False)]

#grouped = ticket.groupby(['Ticket'])

#print(ticket)

#for i in range(0, ticket.shape[0]):
test_ticket = test[test['Parch']==0]

test['Friends'] = 0

#test['FriendsNumber'] = 0

#test['FriendsSex'] = -1

#test['FriendsSurvived'] = -1

test['Male_Friends_Survived'] = 0

test['Male_Friends_NotSurvived'] = 0

test['Female_Friends_Survived'] = 0

test['Female_Friends_NotSurvived'] = 0



grouped = test_ticket.groupby(['Ticket'])

for (k, v) in grouped.groups.items():

    temp_df = train[train.Ticket==k]

    length = temp_df.shape[0]

    if temp_df.shape[0]>0:

        for i in range(0, len(v)):

            test.loc[v[i], 'Friends'] = 1

            #test.loc[v[i], 'FriendsSex'] =temp_df.SexInt.sum()/length

            #test.loc[v[i], 'FriendsSurvived'] = temp_df.Survived.sum()/length

            test.loc[v[i], 'Male_Friends_Survived'] = temp_df[(temp_df.Sex=='male')&(temp_df.Survived==1)].shape[0]

            test.loc[v[i], 'Male_Friends_NotSurvived'] = temp_df[(temp_df.Sex=='male')&(temp_df.Survived==0)].shape[0]

            test.loc[v[i], 'Female_Friends_Survived'] = temp_df[(temp_df.Sex=='female')&(temp_df.Survived==1)].shape[0]

            test.loc[v[i], 'Female_Friends_NotSurvived'] = temp_df[(temp_df.Sex=='female')&(temp_df.Survived==0)].shape[0]
train['FatherOnBoard'] = 0

train['FatherSurvived'] = 0

train['MotherOnBoard'] = 0

train['MotherSurvived'] = 0

train['ChildOnBoard'] = 0

train['ChildSurvived'] = 0

train['ChildNotSurvived'] = 0

grouped = train[train.Parch>0].groupby('Ticket')

for (k, v) in grouped.groups.items():

    for i in range(0, len(v)):

        if train.loc[v[i], 'Age']<19:

            temp = train[(train.Ticket==k)&(train.Age>18)]

            if temp[temp.SexInt==1].shape[0] == 1:

                train.loc[v[i], 'FatherOnBoard'] = 1

                train.loc[v[i], 'FatherSurvived'] = temp[temp.SexInt==1].Survived.sum()

            if temp[temp.SexInt==0].shape[0] == 1:

                train.loc[v[i], 'MotherOnBoard'] = 1

                train.loc[v[i], 'MotherSurvived'] = temp[temp.SexInt==0].Survived.sum()

        else:

            temp = train[(train.Ticket==k)&(train.Age<19)]

            length = temp.shape[0]

            if length>0:

                train.loc[v[i], 'ChildOnBoard'] = 1

                #train.loc[v[i], 'ChildSurvived'] = temp.Survived.sum()/length  

                train.loc[v[i], 'ChildSurvived'] = temp[temp.Survived==1].shape[0]

                train.loc[v[i], 'ChildNotSurvived'] = temp[temp.Survived==0].shape[0]

                
test['FatherOnBoard'] = 0

test['FatherSurvived'] = 0

test['MotherOnBoard'] = 0

test['MotherSurvived'] = 0

test['ChildOnBoard'] = 0

test['ChildSurvived'] = 0

test['ChildNotSurvived'] = 0

grouped = test[test.Parch>0].groupby('Ticket')

for (k, v) in grouped.groups.items():

    temp = train[train.Ticket==k]

    length = temp.shape[0]

    if length>0:

        for i in range(0, len(v)):

            if test.loc[v[i], 'Age']<19:

                if temp[(temp.SexInt==1)&(temp.Age>18)].shape[0] == 1:

                    test.loc[v[i], 'FatherOnBoard'] = 1

                    test.loc[v[i], 'FatherSurvived'] = temp[(temp.SexInt==1)&(temp.Age>18)].Survived.sum()

                if temp[(temp.SexInt==0)&(temp.Age>18)].shape[0] == 1:

                    test.loc[v[i], 'MotherOnBoard'] = 1

                    test.loc[v[i], 'MotherSurvived'] = temp[(temp.SexInt==0)&(temp.Age>18)].Survived.sum()

            else:

                length = temp[temp.Age<19].shape[0]

                if length>0:

                    test.loc[v[i], 'ChildOnBoard'] = 1

                    #test.loc[v[i], 'ChildSurvived'] = temp[temp.Age<19].Survived.sum()/length

                    test.loc[v[i], 'ChildSurvived'] = temp[(temp.Age<19)&(temp.Survived==1)].shape[0]

                    test.loc[v[i], 'ChildNotSurvived'] = temp[(temp.Age<19)&(temp.Survived==0)].shape[0]
#print(test[test['FatherSurvived']!=-1].head(10))

print(test.info())
#print(train[train['FatherSurvived']!=-1].head(10))
test_ticket = test[test['Parch']==0]

grouped = ticket.groupby(['Ticket'])
print(test.head(10))
fig = plt.figure(figsize=(8, 1))

grouped = train.groupby(['Survived'])

group0 = grouped.get_group(0)

group1 = grouped.get_group(1)



ax1 = plt.subplot2grid((1,2), (0,0), rowspan=1, colspan=2)

bottom = group0.EmbarkedInt.value_counts().index

width1 = group0.EmbarkedInt.value_counts()

plt.barh(bottom, width1, 0.8, 0.0, color='blue', label='Not Survived', alpha=0.6)

width2 = group1.EmbarkedInt.value_counts()

plt.barh(bottom, width2, 0.8, width1, color='green', label='Survived', alpha=0.6)

plt.legend(loc='best', fontsize='x-small')

ax1.set_yticks([0.4, 1.4, 2.4])

ax1.set_yticklabels(['Southampton', 'Cherbourg', 'Queenstown'], fontsize='small')

ax1.set_title('Survival distribution by Embarked')
#print(train['Title'].value_counts())

#print(test['Title'].value_counts())



title_to_int = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':1, 'Dr':4, 'Rev':4, 'Mlle':2, 'Major':4, 'Col':4,

        'Ms':3, 'Lady':3, 'the Countess':4, 'Sir':4, 'Mme':3, 'Capt':4, 'Jonkheer':4, 'Don':1, 'Dona':3}

train['TitleInt'] = train['Title'].map(title_to_int)

test['TitleInt'] = test['Title'].map(title_to_int)

train.loc[train[train['Age']<13].index, 'TitleInt'] = 5

test.loc[test[test['Age']<13].index, 'TitleInt'] = 5

#train['TitleInt'][train['Age']<13] = 5

#test['TitleInt'][test['Age']<13] = 5



train['FareCat'] = pd.cut(train['Fare'], [-0.1, 50, 100, 150, 200, 300, 1000], right=True, 

        labels=[0, 1, 2, 3, 4, 5])

test['FareCat'] = pd.cut(test['Fare'], [-0.1, 50, 100, 150, 200, 300, 1000], right=True, 

        labels=[0, 1, 2, 3, 4, 5])

train['AgeCat'] = pd.cut(train['Age'], [-0.1, 12.1, 20, 30, 35, 40, 45, 50, 55, 65, 100], right=True, 

        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

test['AgeCat'] = pd.cut(test['Age'], [-0.1, 12.1, 20, 30, 35, 40, 45, 50, 55, 65, 100], right=True, 

        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

train.head(10)

train.info()

#train.describe()

#test.describe()
fig = plt.figure(figsize=(8, 1))

grouped = train.groupby(['Survived'])

group0 = grouped.get_group(0)

group1 = grouped.get_group(1)



ax1 = plt.subplot2grid((1,2), (0,0), rowspan=1, colspan=2)

bottom = group0.TitleInt.value_counts().index

width1 = group0.TitleInt.value_counts()

plt.barh(bottom, width1, 0.8, 0.0, color='blue', label='Not Survived', alpha=0.6)

width2 = group1.TitleInt.value_counts()

plt.barh(bottom, width2, 0.8, width1, color='green', label='Survived', alpha=0.6)

plt.legend(loc='best', fontsize='x-small')

ax1.set_yticks([1.4, 2.4, 3.4, 4.4, 5.4])

ax1.set_yticklabels(['Mr', 'Miss', 'Mrs', 'Profession', 'Child'], fontsize='small')

ax1.set_title('Survival distribution by Title')
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import train_test_split

#selected = SelectKBest(f_classif, 8)

#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat']

#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeInt', 'TitleInt', 'Fare']

#columns = ['Pclass', 'FamilySize', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat']

#columns = ['Pclass', 'FamilySize', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat',

#        'Friends', 'FriendsSex', 'FriendsSurvived']

#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat',

#        'Friends', 'FriendsSex', 'FriendsSurvived', 'FatherSurvived', 'MotherSurvived', 'ChildSurvived']

#columns = ['Pclass', 'SibSp', 'Parch', 'SexInt', 'EmbarkedInt', 'AgeCat', 'TitleInt', 'FareCat']

#        'Friends', 'FriendsSex', 'FriendsSurvived']

#columns = ['Pclass', 'SexInt', 'EmbarkedInt', 'Age', 'TitleInt','Fare', 'Friends', 'FriendsSex', 'FriendsSurvived', 'FatherSurvived', 'MotherSurvived', 'ChildSurvived']

columns = ['Pclass', 'SexInt', 'EmbarkedInt', 'Age', 'TitleInt','Fare', 

        'Friends', 'Male_Friends_Survived', 'Male_Friends_NotSurvived', 'Female_Friends_Survived', 'Female_Friends_NotSurvived',

        'FatherOnBoard', 'FatherSurvived', 'MotherOnBoard', 'MotherSurvived', 'ChildOnBoard', 'ChildSurvived', 'ChildNotSurvived']

X_train, X_test, y_train, y_test = train_test_split(train[columns], train['Survived'], test_size=0.2, random_state=123)

#selected.fit(X_train, y_train)

#X_train_selected = selected.transform(X_train)

#X_test_selected = selected.transform(X_test)

#print(selected.scores_)

#print(selected.pvalues_)

print(X_train.info())

print(X_train.Friends.head(10))

X_train.Pclass = X_train.Pclass.astype('float')

#X_train.EmbarkedInt = X_train.EmbarkedInt.astype('float')

X_train.Friends = X_train.Friends.astype('bool')

X_train.SexInt = X_train.SexInt.astype('bool')

X_train.FatherOnBoard = X_train.FatherOnBoard.astype('bool')

X_train.MotherOnBoard = X_train.MotherOnBoard.astype('bool')

X_train.ChildOnBoard = X_train.ChildOnBoard.astype('bool')

print(X_train.info())

print(X_train.Friends.head(10))
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

param_grid = {'n_estimators': [10, 50, 100, 150], 'min_samples_leaf': [1, 2, 4, 8], 

        'max_depth': [None, 5, 10, 50], 'max_features': [None, 'auto'], 'min_samples_split': [2, 4, 8]}

rfc = RandomForestClassifier(criterion='gini', min_weight_fraction_leaf=0.0, 

        max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, 

        n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None)

classifer = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1)

#classifer.fit(X_train, y_train)

#print(classifer.grid_scores_)

#print(classifer.best_params_)

#print(X_train.info())
rfc = RandomForestClassifier(n_estimators=150, criterion='gini', max_depth=10, 

        min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 

        max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, 

        oob_score=False, n_jobs=1, random_state=232, verbose=0, warm_start=False, class_weight=None)



rfc.fit(X_train, y_train)

result = rfc.predict(X_test)

rightnum = 0



for i in range(0, result.shape[0]):

    if result[i] == y_test.iloc[i]:

        rightnum += 1

print(rightnum/result.shape[0])



predict = rfc.predict(test[columns])



submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": predict

    })

submission.to_csv("titanic_predict_13.csv", index=False)
from deap import algorithms

from deap import base

from deap import creator

from deap import tools

from deap import gp



import operator

import random

import numpy



print(X_train.shape)

print(type(X_train))



# defined a new primitive set for strongly typed GP

inputTypes = [int, bool, int, float, int, float, bool, int, int, int, int, bool, int, bool, int, bool, int, int]

pset = gp.PrimitiveSetTyped("MAIN", inputTypes, bool, "IN")



# boolean operators

pset.addPrimitive(operator.and_, [bool, bool], bool)

pset.addPrimitive(operator.or_, [bool, bool], bool)

pset.addPrimitive(operator.not_, [bool], bool)



# floating point operators

# Define a protected division function

def protectedDiv(left, right):

    try: return left / right

    except ZeroDivisionError: return 1



pset.addPrimitive(operator.add, [float,float], float)

pset.addPrimitive(operator.sub, [float,float], float)

pset.addPrimitive(operator.mul, [float,float], float)

pset.addPrimitive(protectedDiv, [float,float], float)



# logic operators

# Define a new if-then-else function

def if_then_else(input, output1, output2):

    if input: return output1

    else: return output2



pset.addPrimitive(operator.lt, [float, float], bool)

pset.addPrimitive(operator.eq, [float, float], bool)

pset.addPrimitive(if_then_else, [bool, float, float], float)



# terminals

pset.addEphemeralConstant("abcde", lambda: random.random()*100, float)

pset.addTerminal(False, bool)

pset.addTerminal(True, bool)



creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)



toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)



def evalSurvival(individual):

    # Transform the tree expression in a callable function

    func = toolbox.compile(expr=individual)

    # Randomly sample 400 mails in the spam database

    #spam_samp = random.sample(spam, 400)

    # Evaluate the sum of correctly identified mail as spam

    #result = sum(bool(func(*mail[:57])) is bool(mail[57]) for mail in spam_samp)

    result = 0

    for i in range(0, X_train.shape[0]):

        if bool(func(*X_train.iloc[i, :])) is bool(y_train.iloc[i]):

            result += 1

    return result,

    

toolbox.register("evaluate", evalSurvival)

toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("mate", gp.cxOnePoint)

toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)

toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)



random.seed(10)

pop = toolbox.population(n=100)

hof = tools.HallOfFame(1)

stats = tools.Statistics(lambda ind: ind.fitness.values)

stats.register("avg", numpy.mean)

stats.register("std", numpy.std)

stats.register("min", numpy.min)

stats.register("max", numpy.max)

  

return_pop = algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 2, stats, halloffame=hof)

print(return_pop)
print(return_pop[0][40])

print(hof[0][0])
import xgboost as xgb

xgbclassifer = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=50, silent=True, objective='binary:logistic', nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)

xgbclassifer.fit(X_train, y_train)

result = xgbclassifer.predict(X_test)

#print(result[:10])

rightnum = 0

for i in range(0, result.shape[0]):

    if result[i] == y_test.iloc[i]:

        rightnum += 1

print(rightnum/result.shape[0])
rfc.fit(X_train[:, [0,1,2]], y_train)

result = rfc.predict(X_test[:, [0,1,2]])

rightnum = 0

for i in range(0, result.shape[0]):

    if result[i] == y_test.iloc[i]:

        rightnum += 1

print(rightnum/result.shape[0])
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.regularizers import l2, l1

from sklearn.preprocessing import StandardScaler



stdScaler = StandardScaler()

X_train = stdScaler.fit_transform(X_train)

X_test = stdScaler.transform(X_test)

model = Sequential()

#model.add(Dense(700, input_dim=7, init='normal', activation='relu'))

#model.add(Dropout(0.5))

model.add(Dense(1800, input_dim=18, init='normal', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, init='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop')

model.fit(X_train, y_train, nb_epoch=20, batch_size=32)

result = model.predict(X_test)

rightnum = 0

for i in range(0, result.shape[0]):

    if result[i] >= 0.5:

        result[i] = 1

    else:

        result[i] = 0

    if result[i] == y_test.iloc[i]:

        rightnum += 1

print(rightnum/result.shape[0])



#train_scaler = stdScaler.fit_transform(train[columns])

#model.fit(train_scaler, train['Survived'], nb_epoch=20, batch_size=32)
from sklearn import svm

clf = svm.SVC()

clf.fit(X_train, y_train)

result = clf.predict(X_test)

rightnum = 0

for i in range(0, result.shape[0]):

    if result[i] == y_test.iloc[i]:

        rightnum += 1

print(rightnum/result.shape[0])
rfc_predict = rfc.predict(test[columns])

NN_test = stdScaler.transform(test[columns])

NN_predict = np.ravel(model.predict(NN_test))

svc_predict = clf.predict(test[columns])



print(rfc_predict.shape)

#print(NN_predict.shape)

print(svc_predict.shape)



#print(rfc_predict[:10])

#print(NN_predict[:10,0])

#print(svc_predict[:10])

new_predict = np.zeros(rfc_predict.shape[0])

new_predict.dtype='int'

print(new_predict.shape)

for i in range(0, NN_predict.shape[0]):

    if NN_predict[i] >= 0.5:

        new_predict[i] = 1

    else:

        new_predict[i] = 0

test['predict']=new_predict

#combine_predict = np.zeros(rfc_predict.shape[0])

#combine_predict.dtype='int'

#for i in range(0, combine_predict.shape[0]):

#    if (rfc_predict[i]+svc_predict[i]+new_predict[i]) >=2:

#        combine_predict[i] = 1

#print(combine_predict[:10])     

submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test['predict']})

submission.to_csv("titanic_NN_predict_7.csv", index=False)
print(check_output(["ls", "/kaggle/working"]).decode("utf8"))