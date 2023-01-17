# Imports



# pandas

import pandas as pd

from pandas import Series,DataFrame



# numpy, matplotlib, seaborn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df    = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )



# preview the data

titanic_df.head()
titanic_df.info()

print("----------------------------")

test_df.info()
# Uses relevent info in names:

#Names_test = list(test_df['Name'])

Prefix_train = []

FamilyName_train = []

for i in list(titanic_df['Name']): 

    ind1 = i.index(',')

    ind2 = i.index('.')

    FamilyName_train.append(i[:ind1])

    Prefix_train.append(i[ind1+2:ind2])    



Prefix_test = []

FamilyName_test = []

for i in list(test_df['Name']): 

    ind1 = i.index(',')

    ind2 = i.index('.')

    FamilyName_test.append(i[:ind1])

    Prefix_test.append(i[ind1+2:ind2])    



for i in range(len(titanic_df['Name'])):

    if Prefix_train[i] == 'Dr':

        if titanic_df.iloc[i]['Sex'] == 'male':

            Prefix_train[i]='Mr'

        if titanic_df.iloc[i]['Sex'] == 'female':

            Prefix_train[i]='Mrs'    

            

for i in range(len(test_df['Name'])):

    if Prefix_train[i] == 'Dr':

        if test_df.iloc[i]['Sex'] == 'male':

            Prefix_train[i]='Mr'

        if test_df.iloc[i]['Sex'] == 'female':

            Prefix_train[i]='Mrs'                



#print(set(Prefix_train))  

#print(set(Prefix_test))



# Inspecting low occurencies prefix...

# 'Dr', 4 died, 3 survived: rename 'Mr'/'Mrs'

# 'Major', 1 died, 1survived: rename 'Mr'

# 'Col', 1 died, 1survived: rename 'Mr'

# 'Dona', 'Ms', its Mrs in another language/country: rename 'Mrs'

# 'Don', 'Capt', 'Jonkheer', 'Sir': singular and/or not in the test set: rename 'Mr'

# 'The Countess' : singular and/or not in the test set: rename 'Mrs'

# 'Rev', 7 died, 0 survived: keep this variable for evaluate survivability...



titanic_df['FamilyName'] = titanic_df['Name']

test_df['FamilyName'] = test_df['Name']

titanic_df['Prefix'] = titanic_df['Name']

test_df['Prefix'] = test_df['Name']



for i in range(len(test_df['FamilyName'])):

    test_df.set_value(i,'FamilyName',FamilyName_test[i])

    test_df.set_value(i,'Prefix',Prefix_test[i])

for i in range(len(titanic_df['Prefix'])):

    titanic_df.set_value(i,'FamilyName',FamilyName_train[i])

    titanic_df.set_value(i,'Prefix',Prefix_train[i])   

    

#titanic_df['Prefix'].loc[titanic_df['Prefix'] == 'Don'] = 'Mr'

#titanic_df['Prefix'].loc[titanic_df['Prefix'] == 'Dr'] = 'Mr'

#titanic_df['Prefix'].loc[titanic_df['Prefix'] == 'Major'] = 'Mr'

#titanic_df['Prefix'].loc[titanic_df['Prefix'] == 'Col'] = 'Mr'

#titanic_df['Prefix'].loc[titanic_df['Prefix'] == 'Capt'] = 'Mr'

#titanic_df['Prefix'].loc[titanic_df['Prefix'] == 'Sir'] = 'Mr'

#titanic_df['Prefix'].loc[titanic_df['Prefix'] == 'Jonkheer'] = 'Mr'
         

titanic_df['Prefix'] = titanic_df['Prefix'].replace('Major', 'Mr')

titanic_df['Prefix'] = titanic_df['Prefix'].replace('Col', 'Mr')

titanic_df['Prefix'] = titanic_df['Prefix'].replace('Capt', 'Mr')

titanic_df['Prefix'] = titanic_df['Prefix'].replace('Sir', 'Mr')

titanic_df['Prefix'] = titanic_df['Prefix'].replace('Jonkheer', 'Mr')

titanic_df['Prefix'] = titanic_df['Prefix'].replace('Don', 'Mr')

titanic_df['Prefix'] = titanic_df['Prefix'].replace('Ms', 'Mrs')

titanic_df['Prefix'] = titanic_df['Prefix'].replace('Lady', 'Mrs')

titanic_df['Prefix'] = titanic_df['Prefix'].replace('Mme', 'Mrs')

titanic_df['Prefix'] = titanic_df['Prefix'].replace('Mlle', 'Miss')

titanic_df['Prefix'] = titanic_df['Prefix'].replace('the Countess', 'Mrs')



test_df['Prefix'] = test_df['Prefix'].replace('Dona', 'Mrs')

test_df['Prefix'] = test_df['Prefix'].replace('Ms', 'Mrs')

test_df['Prefix'] = test_df['Prefix'].replace('Col', 'Mr')

test_df['Prefix'] = test_df['Prefix'].replace('Dr', 'Mr')





print(set(list(titanic_df['Prefix'])))

print(set(list(test_df['Prefix'])))
AgeDiff = []

AgeList_train = list(titanic_df['Age'])

AgeList_test = list(test_df['Age'])

for i in range(len(titanic_df['Name'])):

    if (('(' in titanic_df.iloc[i]['Name'])and('Mrs.' in titanic_df.iloc[i]['Name'])):

        #print(titanic_df.iloc[i]['Name'])

        Name1 = titanic_df.iloc[i]['Name'][titanic_df.iloc[i]['Name'].index('.')+2:titanic_df.iloc[i]['Name'].index('(')-1]

        #print("Name1: ", Name1)

        for j in range(len(titanic_df['Name'])):

            if titanic_df.iloc[i]['FamilyName'] == titanic_df.iloc[j]['FamilyName']:

                if titanic_df.iloc[j]['Prefix']=='Mr':

                    TEST = False

                    if Name1 in titanic_df.iloc[j]['Name']:

                        #print(titanic_df.iloc[i]['Name'],titanic_df.iloc[i]['Age'] )

                        #print(titanic_df.iloc[j]['Name'], titanic_df.iloc[j]['Age'])

                        print(titanic_df.iloc[i]['Name'],titanic_df.iloc[i]['Cabin'] )

                        print(titanic_df.iloc[j]['Name'], titanic_df.iloc[j]['Cabin'])

                        if not np.isnan(titanic_df.iloc[i]['Age']):

                            if not np.isnan(titanic_df.iloc[j]['Age']):

                                ageDiff = titanic_df.iloc[j]['Age']-titanic_df.iloc[i]['Age']

                                #To avoid sons named as there father (like Walter and Walter Jr.)

                                if ageDiff >= -10:

                                    if ageDiff <= 25:

                                        AgeDiff.append(ageDiff)

                                        #print(ageDiff)

                        if np.isnan(titanic_df.iloc[i]['Age']):

                            if not np.isnan(titanic_df.iloc[j]['Age']):

                                AgeList_train[i]=titanic_df.iloc[j]['Age']-4

                                print(AgeList_train[i])

                        if np.isnan(titanic_df.iloc[j]['Age']):      

                            if not np.isnan(titanic_df.iloc[i]['Age']):

                                AgeList_train[j]=titanic_df.iloc[i]['Age']+4

                                print(AgeList_train[j])

                        print()

        for j in range(len(test_df['Name'])):

            if titanic_df.iloc[i]['FamilyName'] == test_df.iloc[j]['FamilyName']:

                if test_df.iloc[j]['Prefix']=='Mr':

                    TEST = False

                    if Name1 in test_df.iloc[j]['Name']:

                        #print(titanic_df.iloc[i]['Name'], titanic_df.iloc[i]['Age'])

                        #print(test_df.iloc[j]['Name'], test_df.iloc[j]['Age'])

                        print(titanic_df.iloc[i]['Name'], titanic_df.iloc[i]['Cabin'])

                        print(test_df.iloc[j]['Name'], test_df.iloc[j]['Cabin'])

                        if not np.isnan(titanic_df.iloc[i]['Age']):

                             if not np.isnan(test_df.iloc[j]['Age']):

                                ageDiff = test_df.iloc[j]['Age']-titanic_df.iloc[i]['Age']

                                if ageDiff >= -10:

                                    if ageDiff <= 25:

                                        #print(ageDiff)

                                        AgeDiff.append(ageDiff)

                        if np.isnan(titanic_df.iloc[i]['Age']):

                            if not np.isnan(test_df.iloc[j]['Age']):

                                AgeList_train[i]=test_df.iloc[j]['Age']-4

                                print(AgeList_train[i])

                        if np.isnan(test_df.iloc[j]['Age']):      

                            if not np.isnan(titanic_df.iloc[i]['Age']):

                                AgeList_test[j]=titanic_df.iloc[i]['Age']+4

                                print(AgeList_test[j])

                        print()

                        

for i in range(len(test_df['Name'])):

    if (('(' in test_df.iloc[i]['Name'])and('Mrs.' in test_df.iloc[i]['Name'])):

        #print(titanic_df.iloc[i]['Name'])

        Name1 = test_df.iloc[i]['Name'][test_df.iloc[i]['Name'].index('.')+2:test_df.iloc[i]['Name'].index('(')-1]

        #print("Name1: ", Name1)

        for j in range(len(titanic_df['Name'])):

            if test_df.iloc[i]['FamilyName'] == titanic_df.iloc[j]['FamilyName']:

                if titanic_df.iloc[j]['Prefix']=='Mr':

                    TEST = False

                    if Name1 in titanic_df.iloc[j]['Name']:

                        #print(test_df.iloc[i]['Name'],test_df.iloc[i]['Age'] )

                        #print(titanic_df.iloc[j]['Name'], titanic_df.iloc[j]['Age'])

                        print(test_df.iloc[i]['Name'],test_df.iloc[i]['Cabin'] )

                        print(titanic_df.iloc[j]['Name'], titanic_df.iloc[j]['Cabin'])

                        if not np.isnan(test_df.iloc[i]['Age']):

                            if not np.isnan(titanic_df.iloc[j]['Age']):

                                ageDiff = titanic_df.iloc[j]['Age']-test_df.iloc[i]['Age']

                                #To avoid sons named as there father (like Walter and Walter Jr.)

                                if ageDiff >= -10:

                                    if ageDiff <= 25:

                                        AgeDiff.append(ageDiff)

                                        #print(ageDiff)

                        if np.isnan(test_df.iloc[i]['Age']):

                            if not np.isnan(titanic_df.iloc[j]['Age']):

                                AgeList_test[i]=titanic_df.iloc[j]['Age']-4

                                #print(AgeList_test[i])

                        if np.isnan(titanic_df.iloc[j]['Age']):      

                            if not np.isnan(test_df.iloc[i]['Age']):

                                AgeList_train[j]=test_df.iloc[i]['Age']+4

                                #print(AgeList_train[i])

                        #print()

        for j in range(len(test_df['Name'])):

            if test_df.iloc[i]['FamilyName'] == test_df.iloc[j]['FamilyName']:

                if test_df.iloc[j]['Prefix']=='Mr':

                    TEST = False

                    if Name1 in test_df.iloc[j]['Name']:

                        #print(test_df.iloc[i]['Name'], test_df.iloc[i]['Age'])

                        #print(test_df.iloc[j]['Name'], test_df.iloc[j]['Age'])

                        print(test_df.iloc[i]['Name'], test_df.iloc[i]['Cabin'])

                        print(test_df.iloc[j]['Name'], test_df.iloc[j]['Cabin'])

                        if not np.isnan(test_df.iloc[i]['Age']):

                             if not np.isnan(test_df.iloc[j]['Age']):

                                ageDiff = test_df.iloc[j]['Age']-test_df.iloc[i]['Age']

                                if ageDiff >= -10:

                                    if ageDiff <= 25:

                                        #print(ageDiff)

                                        AgeDiff.append(ageDiff)

                        if np.isnan(test_df.iloc[i]['Age']):

                            if not np.isnan(test_df.iloc[j]['Age']):

                                AgeList_test[i]=test_df.iloc[j]['Age']-4

                                #print(AgeList_test[i])

                        if np.isnan(test_df.iloc[j]['Age']):      

                            if not np.isnan(test_df.iloc[i]['Age']):

                                AgeList_test[j]=test_df.iloc[i]['Age']+4

                                #print(AgeList_test[j])

                        #print()

print(np.mean(np.array(AgeDiff)),np.std(np.array(AgeDiff))) # 4.44186046512
MrAges = []

MrAges_00 = []

MrAges_01 = []

MrAges_10 = []

MrAges_11 = []

MasterAges = []

MissAges = []

MissAges_00 = []

MissAges_01 = []

MissAges_10 = []

MissAges_11 = []

MrsAges = []

MrsAges_00 = []

MrsAges_01 = []

MrsAges_10 = []

MrsAges_11 = []



for i in range(len(titanic_df['Age'])):

    if not np.isnan(titanic_df.iloc[i]['Age']):

        if titanic_df.iloc[i]['Prefix']=='Mr':

            if (titanic_df.iloc[i]['SibSp']==0 and titanic_df.iloc[i]['Parch']==0):

                MrAges_00.append(titanic_df.iloc[i]['Age'])

            if (titanic_df.iloc[i]['SibSp']==0 and titanic_df.iloc[i]['Parch']!=0):

                MrAges_01.append(titanic_df.iloc[i]['Age'])

            if (titanic_df.iloc[i]['SibSp']!=0 and titanic_df.iloc[i]['Parch']==0):

                MrAges_10.append(titanic_df.iloc[i]['Age'])

            if (titanic_df.iloc[i]['SibSp']!=0 and titanic_df.iloc[i]['Parch']!=0):

                MrAges_11.append(titanic_df.iloc[i]['Age'])

            MrAges.append(titanic_df.iloc[i]['Age'])

        if titanic_df.iloc[i]['Prefix']=='Mrs':

            if (titanic_df.iloc[i]['SibSp']==0 and titanic_df.iloc[i]['Parch']==0):

                MrsAges_00.append(titanic_df.iloc[i]['Age'])

            if (titanic_df.iloc[i]['SibSp']==0 and titanic_df.iloc[i]['Parch']!=0):

                MrsAges_01.append(titanic_df.iloc[i]['Age'])

            if (titanic_df.iloc[i]['SibSp']!=0 and titanic_df.iloc[i]['Parch']==0):

                MrsAges_10.append(titanic_df.iloc[i]['Age'])

            if (titanic_df.iloc[i]['SibSp']!=0 and titanic_df.iloc[i]['Parch']!=0):

                MrsAges_11.append(titanic_df.iloc[i]['Age'])

            MrsAges.append(titanic_df.iloc[i]['Age'])  

        if titanic_df.iloc[i]['Prefix']=='Master':

            MasterAges.append(titanic_df.iloc[i]['Age']) 

        if titanic_df.iloc[i]['Prefix']=='Miss':

            if (titanic_df.iloc[i]['SibSp']==0 and titanic_df.iloc[i]['Parch']==0):

                #print("MissAges_00",titanic_df.iloc[i]['SibSp'],titanic_df.iloc[i]['Parch'])

                MissAges_00.append(titanic_df.iloc[i]['Age'])

            if (titanic_df.iloc[i]['SibSp']==0 and titanic_df.iloc[i]['Parch']!=0):

                #print("MissAges_01",titanic_df.iloc[i]['SibSp'],titanic_df.iloc[i]['Parch'])

                MissAges_01.append(titanic_df.iloc[i]['Age'])

            if (titanic_df.iloc[i]['SibSp']!=0 and titanic_df.iloc[i]['Parch']==0):

                #print("MissAges_10",titanic_df.iloc[i]['SibSp'],titanic_df.iloc[i]['Parch'])

                MissAges_10.append(titanic_df.iloc[i]['Age'])

            if (titanic_df.iloc[i]['SibSp']!=0 and titanic_df.iloc[i]['Parch']!=0):

                #print("MissAges_11",titanic_df.iloc[i]['SibSp'],titanic_df.iloc[i]['Parch'])

                MissAges_11.append(titanic_df.iloc[i]['Age'])

            MissAges.append(titanic_df.iloc[i]['Age'])   

        if titanic_df.iloc[i]['Prefix']=='Rev':

            MrAges.append(titanic_df.iloc[i]['Age']) 

            

for i in range(len(test_df['Age'])):

    if not np.isnan(test_df.iloc[i]['Age']):

        if test_df.iloc[i]['Prefix']=='Mr':

            if (test_df.iloc[i]['SibSp']==0 and test_df.iloc[i]['Parch']==0):

                MrAges_00.append(test_df.iloc[i]['Age'])

            if (test_df.iloc[i]['SibSp']==0 and test_df.iloc[i]['Parch']!=0):

                MrAges_01.append(test_df.iloc[i]['Age'])

            if (test_df.iloc[i]['SibSp']!=0 and test_df.iloc[i]['Parch']==0):

                MrAges_10.append(test_df.iloc[i]['Age'])

            if (test_df.iloc[i]['SibSp']!=0 and test_df.iloc[i]['Parch']!=0):

                MrAges_11.append(test_df.iloc[i]['Age'])

            MrAges.append(test_df.iloc[i]['Age'])

        if test_df.iloc[i]['Prefix']=='Mrs':

            if (test_df.iloc[i]['SibSp']==0 and test_df.iloc[i]['Parch']==0):

                MrsAges_00.append(test_df.iloc[i]['Age'])

            if (test_df.iloc[i]['SibSp']==0 and test_df.iloc[i]['Parch']!=0):

                MrsAges_01.append(test_df.iloc[i]['Age'])

            if (test_df.iloc[i]['SibSp']!=0 and test_df.iloc[i]['Parch']==0):

                MrsAges_10.append(test_df.iloc[i]['Age'])

            if (test_df.iloc[i]['SibSp']!=0 and test_df.iloc[i]['Parch']!=0):

                MrsAges_11.append(test_df.iloc[i]['Age'])

            MrsAges.append(test_df.iloc[i]['Age'])  

        if test_df.iloc[i]['Prefix']=='Master':

            MasterAges.append(test_df.iloc[i]['Age']) 

        if test_df.iloc[i]['Prefix']=='Miss':

            if (test_df.iloc[i]['SibSp']==0 and test_df.iloc[i]['Parch']==0):

                #print("MissAges_00",titanic_df.iloc[i]['SibSp'],titanic_df.iloc[i]['Parch'])

                MissAges_00.append(test_df.iloc[i]['Age'])

            if (test_df.iloc[i]['SibSp']==0 and test_df.iloc[i]['Parch']!=0):

                #print("MissAges_01",titanic_df.iloc[i]['SibSp'],titanic_df.iloc[i]['Parch'])

                MissAges_01.append(test_df.iloc[i]['Age'])

            if (test_df.iloc[i]['SibSp']!=0 and test_df.iloc[i]['Parch']==0):

                #print("MissAges_10",titanic_df.iloc[i]['SibSp'],titanic_df.iloc[i]['Parch'])

                MissAges_10.append(test_df.iloc[i]['Age'])

            if (test_df.iloc[i]['SibSp']!=0 and test_df.iloc[i]['Parch']!=0):

                #print("MissAges_11",titanic_df.iloc[i]['SibSp'],titanic_df.iloc[i]['Parch'])

                MissAges_11.append(test_df.iloc[i]['Age'])

            MissAges.append(test_df.iloc[i]['Age'])   

        if test_df.iloc[i]['Prefix']=='Rev':

            MrAges.append(test_df.iloc[i]['Age'])  

            

#print("Mr: ", len(MrAges) , np.mean(np.array(MrAges)), np.std(np.array(MrAges)))

#print("Mrs: ", len(MrsAges), np.mean(np.array(MrsAges)), np.std(np.array(MrsAges)))

#print("Master: ", len(MasterAges), np.mean(np.array(MasterAges)), np.std(np.array(MasterAges)))

#print("Miss: ", len(MissAges), np.mean(np.array(MissAges)), np.std(np.array(MissAges)))

#print("MissAges_00: ", len(MissAges_00), np.mean(np.array(MissAges_00)), np.std(np.array(MissAges_00)))

#print("MissAges_01: ", len(MissAges_01), np.mean(np.array(MissAges_01)), np.std(np.array(MissAges_01)))

#print("MissAges_10: ", len(MissAges_10), np.mean(np.array(MissAges_10)), np.std(np.array(MissAges_10)))

#print("MissAges_11: ", len(MissAges_11), np.mean(np.array(MissAges_11)), np.std(np.array(MissAges_11)))

#print("MrsAges_00: ", len(MrsAges_00), np.mean(np.array(MrsAges_00)), np.std(np.array(MrsAges_00)))

#print("MrsAges_01: ", len(MrsAges_01), np.mean(np.array(MrsAges_01)), np.std(np.array(MrsAges_01)))

#print("MrsAges_10: ", len(MrsAges_10), np.mean(np.array(MrsAges_10)), np.std(np.array(MrsAges_10)))

#print("MrsAges_11: ", len(MrsAges_11), np.mean(np.array(MrsAges_11)), np.std(np.array(MrsAges_11)))

#print("MrAges_00: ", len(MrAges_00), np.mean(np.array(MrAges_00)), np.std(np.array(MrAges_00)))

#print("MrAges_01: ", len(MrAges_01), np.mean(np.array(MrAges_01)), np.std(np.array(MrAges_01)))

#print("MrAges_10: ", len(MrAges_10), np.mean(np.array(MrAges_10)), np.std(np.array(MrAges_10)))

#print("MrAges_11: ", len(MrAges_11), np.mean(np.array(MrAges_11)), np.std(np.array(MrAges_11)))

#print()

#we can combine Mr together, as well for Mrs...

#For Miss: combine 00 with 10 and 01 with 11...



MrNans_train = []

MasterNans_train = []

MrsNans_train = []

MissNans_00_10_train = []

MissNans_11_01_train = []

for i in range(len(titanic_df['Age'])):

    if np.isnan(titanic_df.iloc[i]['Age']):

        if titanic_df.iloc[i]['Prefix']=='Mr':

            MrNans_train.append(1)

        if titanic_df.iloc[i]['Prefix']=='Mrs':

            MrsNans_train.append(1)

        if titanic_df.iloc[i]['Prefix']=='Master':

            MasterNans_train.append(1)

        if titanic_df.iloc[i]['Prefix']=='Miss':

            if (titanic_df.iloc[i]['SibSp']==0 and titanic_df.iloc[i]['Parch']==0):

                MissNans_00_10_train.append(1)

            if (titanic_df.iloc[i]['SibSp']==0 and titanic_df.iloc[i]['Parch']!=0):

                MissNans_11_01_train.append(1)

            if (titanic_df.iloc[i]['SibSp']!=0 and titanic_df.iloc[i]['Parch']==0):

                MissNans_00_10_train.append(1)

            if (titanic_df.iloc[i]['SibSp']!=0 and titanic_df.iloc[i]['Parch']!=0):

                MissNans_11_01_train.append(1)

                

MrNans_test = []

MasterNans_test = []

MrsNans_test = []

MissNans_00_10_test = []

MissNans_11_01_test = []

for i in range(len(test_df['Age'])):

    if np.isnan(test_df.iloc[i]['Age']):

        if test_df.iloc[i]['Prefix']=='Mr':

            MrNans_test.append(1)

        if test_df.iloc[i]['Prefix']=='Mrs':

            MrsNans_test.append(1)

        if test_df.iloc[i]['Prefix']=='Master':

            MasterNans_test.append(1)

        if test_df.iloc[i]['Prefix']=='Miss':

            if (test_df.iloc[i]['SibSp']==0 and test_df.iloc[i]['Parch']==0):

                MissNans_00_10_test.append(1)

            if (test_df.iloc[i]['SibSp']==0 and test_df.iloc[i]['Parch']!=0):

                MissNans_11_01_test.append(1)

            if (test_df.iloc[i]['SibSp']!=0 and test_df.iloc[i]['Parch']==0):

                MissNans_00_10_test.append(1)

            if (test_df.iloc[i]['SibSp']!=0 and test_df.iloc[i]['Parch']!=0):

                MissNans_11_01_test.append(1)                

                

print("Mr: ", len(MrNans_train), len(MrNans_test), np.mean(np.array(MrAges)), np.std(np.array(MrAges)))

print("Mrs: ", len(MrsNans_train), len(MrsNans_test),np.mean(np.array(MrsAges)), np.std(np.array(MrsAges)))

print("Master: ", len(MasterNans_train), len(MasterNans_test), np.mean(np.array(MasterAges)), np.std(np.array(MasterAges)))

print("MissAges_00_10: ", len(MissNans_00_10_train), len(MissNans_00_10_test), np.mean(np.array(MissAges_00+MissAges_10)), np.std(np.array(MissAges_00+MissAges_10)))

print("MissAges_01_11: ", len(MissNans_11_01_train), len(MissNans_11_01_test), np.mean(np.array(MissAges_01+MissAges_11)), np.std(np.array(MissAges_01+MissAges_11)))           
#Age

#randMr = np.random.randint(np.mean(np.array(MrAges)) - np.std(np.array(MrAges)), np.mean(np.array(MrAges)) + np.std(np.array(MrAges)))

#randMrs = np.random.randint(np.mean(np.array(MrsAges)) - np.std(np.array(MrsAges)), np.mean(np.array(MrsAges)) + np.std(np.array(MrsAges)))

#randMaster = np.random.randint(np.mean(np.array(MasterAges)) - np.std(np.array(MasterAges)), np.mean(np.array(MasterAges)) + np.std(np.array(MasterAges)))

#randMiss00 = np.random.randint(np.mean(np.array(MissAges_00+MissAges_10)) - np.std(np.array(MissAges_00+MissAges_10)), np.mean(np.array(MissAges_00+MissAges_10)) + np.std(np.array(MissAges_00+MissAges_10)))

#randMiss11 = np.random.randint(np.mean(np.array(MissAges_01+MissAges_11)) - np.std(np.array(MissAges_01+MissAges_11)), np.mean(np.array(MissAges_01+MissAges_11)) + np.std(np.array(MissAges_01+MissAges_11)))

#print(randMr, randMrs, randMaster, randMiss00, randMiss11)



for i in range(len(titanic_df['Age'])):

    if np.isnan(titanic_df.iloc[i]['Age']):

        prefix = str(titanic_df.iloc[i]['Prefix'])

        if prefix=='Mr':

            rand = np.random.randint(np.mean(np.array(MrAges)) - np.std(np.array(MrAges)), np.mean(np.array(MrAges)) + np.std(np.array(MrAges)))

        elif prefix=='Mrs':

            rand = np.random.randint(np.mean(np.array(MrsAges)) - np.std(np.array(MrsAges)), np.mean(np.array(MrsAges)) + np.std(np.array(MrsAges)))

        elif prefix=='Master':

            rand = np.random.randint(np.mean(np.array(MasterAges)) - np.std(np.array(MasterAges)), np.mean(np.array(MasterAges)) + np.std(np.array(MasterAges)))

        elif prefix=='Miss':

            if titanic_df.iloc[i]['Parch']==0:

                rand = np.random.randint(np.mean(np.array(MissAges_00+MissAges_10)) - np.std(np.array(MissAges_00+MissAges_10)), np.mean(np.array(MissAges_00+MissAges_10)) + np.std(np.array(MissAges_00+MissAges_10)))

            elif titanic_df.iloc[i]['Parch']!=0:

                rand = np.random.randint(np.mean(np.array(MissAges_01+MissAges_11)) - np.std(np.array(MissAges_01+MissAges_11)), np.mean(np.array(MissAges_01+MissAges_11)) + np.std(np.array(MissAges_01+MissAges_11)))

        titanic_df.set_value(i,'Age',rand)           



for i in range(len(test_df['Age'])):

    if np.isnan(test_df.iloc[i]['Age']):

        prefix = str(test_df.iloc[i]['Prefix'])

        if prefix=='Mr':

            rand = np.random.randint(np.mean(np.array(MrAges)) - np.std(np.array(MrAges)), np.mean(np.array(MrAges)) + np.std(np.array(MrAges)))

        elif prefix=='Mrs':

            rand = np.random.randint(np.mean(np.array(MrsAges)) - np.std(np.array(MrsAges)), np.mean(np.array(MrsAges)) + np.std(np.array(MrsAges)))

        elif prefix=='Master':

            rand = np.random.randint(np.mean(np.array(MasterAges)) - np.std(np.array(MasterAges)), np.mean(np.array(MasterAges)) + np.std(np.array(MasterAges)))

        elif prefix=='Miss':

            if test_df.iloc[i]['Parch']==0:

                rand = np.random.randint(np.mean(np.array(MissAges_00+MissAges_10)) - np.std(np.array(MissAges_00+MissAges_10)), np.mean(np.array(MissAges_00+MissAges_10)) + np.std(np.array(MissAges_00+MissAges_10)))

            elif test_df.iloc[i]['Parch']!=0:

                rand = np.random.randint(np.mean(np.array(MissAges_01+MissAges_11)) - np.std(np.array(MissAges_01+MissAges_11)), np.mean(np.array(MissAges_01+MissAges_11)) + np.std(np.array(MissAges_01+MissAges_11)))

        test_df.set_value(i,'Age',rand)    

        

titanic_df['Age'] = titanic_df['Age'].astype(int)

test_df['Age']    = test_df['Age'].astype(int)       

titanic_df.info()

print("----------------------------")

test_df.info()                
# Embarked

# Either to consider Embarked column in predictions,

# and remove "S" dummy variable, 

# and leave "C" & "Q", since they seem to have a good rate for Survival.

# OR, don't create dummy variables for Embarked column, just drop it, 

# because logically, Embarked doesn't seem to be useful in prediction.

#La survabilité de 'C' et 'Q' devrait être expliqué par une différence de proportion des autres variables...

#On peut vérifier s'il y avait un effet "premier arrivé premier servi" dans les lits 3eme classe...

sf3 = []

sm3 = []

cf3 = []

cm3 = []

qf3 = []

qm3 = []



for i in range(len(titanic_df['Embarked'])):

    if str(titanic_df.iloc[i]['Embarked'])=='S':

        if str(titanic_df.iloc[i]['Sex'])=='male':

            if str(titanic_df.iloc[i]['Pclass'])=='3':

                sm3.append(titanic_df.iloc[i]['Survived'])

    if str(titanic_df.iloc[i]['Embarked'])=='S':

        if str(titanic_df.iloc[i]['Sex'])=='female':

            if str(titanic_df.iloc[i]['Pclass'])=='3':

                sf3.append(titanic_df.iloc[i]['Survived'])

    if str(titanic_df.iloc[i]['Embarked'])=='C':

        if str(titanic_df.iloc[i]['Sex'])=='male':

            if str(titanic_df.iloc[i]['Pclass'])=='3':

                cm3.append(titanic_df.iloc[i]['Survived'])          

    if str(titanic_df.iloc[i]['Embarked'])=='C':

        if str(titanic_df.iloc[i]['Sex'])=='female':

            if str(titanic_df.iloc[i]['Pclass'])=='3':

                cf3.append(titanic_df.iloc[i]['Survived'])

    if str(titanic_df.iloc[i]['Embarked'])=='Q':

        if str(titanic_df.iloc[i]['Sex'])=='male':

            if str(titanic_df.iloc[i]['Pclass'])=='3':

                qm3.append(titanic_df.iloc[i]['Survived'])

    if str(titanic_df.iloc[i]['Embarked'])=='Q':

        if str(titanic_df.iloc[i]['Sex'])=='female':

            if str(titanic_df.iloc[i]['Pclass'])=='3':

                qf3.append(titanic_df.iloc[i]['Survived'])                      

        

print("sf3: ", len(sf3), np.mean(np.array(sf3))) #sf3: 88 0.375

print("sm3: ", len(sm3), np.mean(np.array(sm3))) #sm3: 265 0.128301886792

print("cf3: ", len(cf3), np.mean(np.array(cf3))) #cf3: 23 0.652173913043

print("cm3: ", len(cm3), np.mean(np.array(cm3))) #cm3: 43 0.232558139535

print("qf3: ", len(qf3), np.mean(np.array(qf3))) #qf3: 33 0.727272727273 

print("qm3: ", len(qm3), np.mean(np.array(qm3))) #qm3: 39 0.0769230769231



#2eme Classe

sf2 = []

sm2 = []

cf2 = []

cm2 = []

qf2 = []

qm2 = []



for i in range(len(titanic_df['Embarked'])):

    if str(titanic_df.iloc[i]['Embarked'])=='S':

        if str(titanic_df.iloc[i]['Sex'])=='male':

            if str(titanic_df.iloc[i]['Pclass'])=='2':

                sm2.append(titanic_df.iloc[i]['Survived'])

    if str(titanic_df.iloc[i]['Embarked'])=='S':

        if str(titanic_df.iloc[i]['Sex'])=='female':

            if str(titanic_df.iloc[i]['Pclass'])=='2':

                sf2.append(titanic_df.iloc[i]['Survived'])

    if str(titanic_df.iloc[i]['Embarked'])=='C':

        if str(titanic_df.iloc[i]['Sex'])=='male':

            if str(titanic_df.iloc[i]['Pclass'])=='2':

                cm2.append(titanic_df.iloc[i]['Survived'])          

    if str(titanic_df.iloc[i]['Embarked'])=='C':

        if str(titanic_df.iloc[i]['Sex'])=='female':

            if str(titanic_df.iloc[i]['Pclass'])=='2':

                cf2.append(titanic_df.iloc[i]['Survived'])

    if str(titanic_df.iloc[i]['Embarked'])=='Q':

        if str(titanic_df.iloc[i]['Sex'])=='male':

            if str(titanic_df.iloc[i]['Pclass'])=='2':

                qm2.append(titanic_df.iloc[i]['Survived'])

    if str(titanic_df.iloc[i]['Embarked'])=='Q':

        if str(titanic_df.iloc[i]['Sex'])=='female':

            if str(titanic_df.iloc[i]['Pclass'])=='2':

                qf2.append(titanic_df.iloc[i]['Survived'])                      

print()        

print("sf2: ", len(sf2), np.mean(np.array(sf2)))

print("sm2: ", len(sm2), np.mean(np.array(sm2)))

print("cf2: ", len(cf2), np.mean(np.array(cf2)))

print("cm2: ", len(cm2), np.mean(np.array(cm2)))

print("qf2: ", len(qf2), np.mean(np.array(qf2)))

print("qm2: ", len(qm2), np.mean(np.array(qm2)))
# only in titanic_df, fill the two missing values with the most occurred value, which is "S".

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")

#Let the embarked variable for third class only...

Embarked_3_train = []

for i in range(len(titanic_df['Name'])):

    if str(titanic_df.iloc[i]['Pclass'])=='3':

        Embarked_3_train.append(titanic_df.iloc[i]['Embarked'])

    elif str(titanic_df.iloc[i]['Pclass'])!='3':    

        Embarked_3_train.append('N')

Embarked_3_test = []

for i in range(len(test_df['Name'])):

    if str(test_df.iloc[i]['Pclass'])=='3':

        Embarked_3_test.append(test_df.iloc[i]['Embarked'])

    elif str(test_df.iloc[i]['Pclass'])!='3':    

        Embarked_3_test.append('N')        

        

        

        

titanic_df['Embarked3'] = titanic_df['Embarked']

test_df['Embarked3'] = test_df['Embarked']



for i in range(len(test_df['Embarked3'])):

    test_df.set_value(i,'Embarked3',Embarked_3_test[i])

for i in range(len(titanic_df['Embarked3'])):

    titanic_df.set_value(i,'Embarked3',Embarked_3_train[i])    

    



embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked3'])

embark_dummies_titanic.drop(['N'], axis=1, inplace=True)



embark_dummies_test  = pd.get_dummies(test_df['Embarked3'])

embark_dummies_test.drop(['N'], axis=1, inplace=True)



titanic_df = titanic_df.join(embark_dummies_titanic)

test_df    = test_df.join(embark_dummies_test)



titanic_df.drop(['Embarked'], axis=1,inplace=True)

test_df.drop(['Embarked'], axis=1,inplace=True) 



titanic_df.drop(['Embarked3'], axis=1,inplace=True)

test_df.drop(['Embarked3'], axis=1,inplace=True) 



# preview the data

titanic_df.head()



        
# Fare

for i in range(len(test_df['Fare'])):

     if np.isnan(test_df.iloc[i]['Fare']):

            print(test_df.iloc[i]) # third class, no cabin, no family        
# Fare



# only for test_df, since there is a missing "Fare" values

test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)



# convert from float to int

titanic_df['Fare'] = titanic_df['Fare'].astype(int)

test_df['Fare']    = test_df['Fare'].astype(int)



# get fare for survived & didn't survive passengers 

fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]

fare_survived     = titanic_df["Fare"][titanic_df["Survived"] == 1]



# get average and std for fare of survived/not survived passengers

avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])

std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])



# plot

titanic_df['Fare'].plot(kind='hist', figsize=(15,3),bins=100, xlim=(0,50))



avgerage_fare.index.names = std_fare.index.names = ["Survived"]

avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
# Avoir une cabine ne devrait pas augmenter les chance de survie. Cependant, certaines cabines ont puent augmenter les chance de mourir, si elle sont situées près de la zône d'impact par exemple...

# Après observation, aucune ne semble plus mortelle qu'une autre et les statistiques de survit semble explicable avec l'age, le sex et la classe...

# Cabin

# It has a lot of NaN values, so it won't cause a remarkable impact on prediction

titanic_df.drop("Cabin",axis=1,inplace=True)

test_df.drop("Cabin",axis=1,inplace=True)
# Family



# Instead of having two columns Parch & SibSp, 

# we can have only one column represent if the passenger had any family member aboard or not,

# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]

titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1

titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0



test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]

test_df['Family'].loc[test_df['Family'] > 0] = 1

test_df['Family'].loc[test_df['Family'] == 0] = 0



# drop Parch & SibSp

titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)

test_df    = test_df.drop(['SibSp','Parch'], axis=1)



# plot

#fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)

#sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)



# average of survived for those who had/didn't have any family member

#family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()

#sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)



#axis1.set_xticklabels(["With Family","Alone"], rotation=0)
# Person defined by prefix (Depends on age sex, family, etc.)



# No need to use Sex column since we created Person column

#titanic_df.drop(['Sex'],axis=1,inplace=True)

#test_df.drop(['Sex'],axis=1,inplace=True)



# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers

person_dummies_titanic  = pd.get_dummies(titanic_df['Prefix'])

#person_dummies_titanic.columns = ['Miss', 'Mrs', 'Master', 'Rev', 'Mr']

#person_dummies_titanic.drop(['Male'], axis=1, inplace=True)



person_dummies_test  = pd.get_dummies(test_df['Prefix'])

#person_dummies_test.columns = ['Miss', 'Mrs', 'Master', 'Rev', 'Mr']

#person_dummies_test.drop(['Male'], axis=1, inplace=True)



titanic_df = titanic_df.join(person_dummies_titanic)

test_df    = test_df.join(person_dummies_test)



#fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))



# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)

#sns.countplot(x='Person', data=titanic_df, ax=axis1)



# average of survived for each Person(male, female, or child)

#person_perc = titanic_df[["Prefix", "Survived"]].groupby(['Prefix'],as_index=False).mean()

#sns.barplot(x='Prefix', y='Survived', data=person_perc, ax=axis2, order=['Miss', 'Mrs', 'Master', 'Rev', 'Mr'])



#titanic_df.drop(['Person'],axis=1,inplace=True)

#test_df.drop(['Person'],axis=1,inplace=True)
# Pclass



# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])

sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)



# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers

pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])

pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)



pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])

pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)



titanic_df.drop(['Pclass'],axis=1,inplace=True)

test_df.drop(['Pclass'],axis=1,inplace=True)



titanic_df = titanic_df.join(pclass_dummies_titanic)

test_df    = test_df.join(pclass_dummies_test)
## drop unnecessary columns, these columns won't be useful in analysis and prediction

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket','Sex','FamilyName','Prefix'], axis=1)

test_df    = test_df.drop(['Name','Ticket','Sex','FamilyName','Prefix'], axis=1)
titanic_df.head()
# define training and testing sets



X_train = titanic_df.drop("Survived",axis=1)

Y_train = titanic_df["Survived"]

X_test  = test_df.drop("PassengerId",axis=1).copy()
# Logistic Regression



#logreg = LogisticRegression()



#logreg.fit(X_train, Y_train)



#Y_pred = logreg.predict(X_test)



#logreg.score(X_train, Y_train)
# Support Vector Machines



#svc = SVC()



#svc.fit(X_train, Y_train)



#Y_pred = svc.predict(X_test)



#svc.score(X_train, Y_train)
# Random Forests



random_forest = RandomForestClassifier(n_estimators=100)



random_forest.fit(X_train, Y_train)



Y_pred = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
#knn = KNeighborsClassifier(n_neighbors = 3)



#knn.fit(X_train, Y_train)



#Y_pred = knn.predict(X_test)



#knn.score(X_train, Y_train)
# Gaussian Naive Bayes



#gaussian = GaussianNB()



#gaussian.fit(X_train, Y_train)



#Y_pred = gaussian.predict(X_test)



#gaussian.score(X_train, Y_train)
# get Correlation Coefficient for each feature using Logistic Regression

coeff_df = DataFrame(titanic_df.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])



# preview

coeff_df
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)