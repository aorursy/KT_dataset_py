import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 80)





#  Kaggle directories

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))





#  Load the Datasets

df_TRN = pd.read_csv('../input/train.csv')

df_TST = pd.read_csv('../input/test.csv')

print("training set:\t", df_TRN.shape, "\ntest set:\t",df_TST.shape, "\t- no \"Survived\" column")

print("\nColumns:\n",df_TRN.columns.values)
df_TRN.describe()  # NUMERIC DATA
df_TRN.describe(include='O')  # CATEGORICAL DATA
#  heatmap of null values

fig = plt.figure(figsize=(12,5))

fig.add_subplot(121)

plt.title('df_TRN - "NULLs\"')

sns.heatmap(df_TRN.isnull(), cmap='gray')

fig.add_subplot(122)

plt.title('df_TST - "NULLs\"')

sns.heatmap(df_TST.isnull(), cmap='gray')

plt.show()



for i in [df_TRN,df_TST]:

    nulls = i.isnull().sum().sort_values(ascending = False)

    prcet = round(nulls/len(i)*100,2)

    i.null = pd.concat([nulls, prcet], axis = 1,keys= ['Total', 'Percent'])

print(pd.concat([df_TRN.null,df_TST.null], axis=1,keys=['TRAIN Data - NULL', 'TEST Data - NULL']))



#  check for duplicate values

print('\n\nDuplicated - TRAIN:  {}'.format(df_TRN.duplicated().sum()))

print('Duplicated - TEST:   {}'.format(df_TST.duplicated().sum()))
#  check for null

print(df_TRN[['PassengerId','Embarked']][df_TRN['Embarked'].isnull()])

print(df_TST[['PassengerId','Fare']][df_TST['Fare'].isnull()])



df_TRN['Embarked'].fillna(df_TRN.Embarked.mode()[0], inplace=True) # fill with mode

df_TST['Fare'].fillna(df_TST['Fare'].mean(), inplace=True)      # fill with mean



#  verify nulls were filled

print(df_TRN[['PassengerId','Embarked']][df_TRN['PassengerId'].isin([62,830])])

print(df_TST[['PassengerId','Fare']][df_TST['PassengerId'] == 1044])
fig = plt.figure(figsize=(12,5))

fig.add_subplot(121)

plt.title('TRAIN - Age/Sex per Passenger Class')

sns.barplot(data=df_TRN, x='Pclass',y='Age',hue='Sex')

fig.add_subplot(122)

plt.title('TEST - Age/Sex per Passenger Class')

sns.barplot(data=df_TST, x='Pclass',y='Age',hue='Sex')

plt.show()
#  calculate age per pclass and sex

#  training - mean Age per Pclass and Sex

meanAgeTrnMale = round(df_TRN[(df_TRN['Sex'] == "male")]['Age'].groupby(df_TRN['Pclass']).mean(),2)

meanAgeTrnFeMale = round(df_TRN[(df_TRN['Sex'] == "female")]['Age'].groupby(df_TRN['Pclass']).mean(),2)



#  test - - mean Age per Pclass and Sex

meanAgeTstMale = round(df_TST[(df_TST['Sex'] == "male")]['Age'].groupby(df_TST['Pclass']).mean(),2)

meanAgeTstFeMale = round(df_TST[(df_TST['Sex'] == "female")]['Age'].groupby(df_TST['Pclass']).mean(),2)



print('\n\t\tMEAN AGE PER SEX PER PCLASS')

print(pd.concat([meanAgeTrnMale, meanAgeTrnFeMale,meanAgeTstMale, meanAgeTstFeMale], axis = 1,keys= ['TRN-Male','TRN-Female','TST-Male','TST-Female']))
#  define function APS to fill Age NaN for training data

def age_fillna_TRN(APStrn):

    Age     = APStrn[0]

    Pclass  = APStrn[1]

    Sex     = APStrn[2]

    

    if pd.isnull(Age):

        if Sex == 'male':

            if Pclass == 1:

                return 41.28

            if Pclass == 2:

                return 30.74

            if Pclass == 3:

                return 26.51



        if Sex == 'female':

            if Pclass == 1:

                return 34.61

            if Pclass == 2:

                return 28.72

            if Pclass == 3:

                return 21.75

    else:

        return Age



#  define function APS to fill Age NaN for test data

def age_fillna_TST(APStst):

    Age     = APStst[0]

    Pclass  = APStst[1]

    Sex     = APStst[2]

    

    if pd.isnull(Age):

        if Sex == 'male':

            if Pclass == 1:

                return 40.52

            if Pclass == 2:

                return 30.94

            if Pclass == 3:

                return 24.53



        if Sex == 'female':

            if Pclass == 1:

                return 41.33

            if Pclass == 2:

                return 24.38

            if Pclass == 3:

                return 23.07

    else:

        return Age
#  execute Age functions

df_TRN['Age'] = df_TRN[['Age','Pclass','Sex']].apply(age_fillna_TRN,axis=1)

df_TST['Age'] = df_TST[['Age','Pclass','Sex']].apply(age_fillna_TST,axis=1)



#  Check missing Age values

print('Missing values for Age: \ntraining\t', df_TRN.Age.isnull().sum(), "\ntest\t\t",df_TST.Age.isnull().sum())
#  Steps 1, 2 and 3

for i in [df_TRN, df_TST]:

    i.fillna("N", inplace = True)      #  step 1

    i.Cabin = [j[0] for j in i.Cabin]  #  step 2

    i.uniq = i.Cabin.value_counts()

    i.cost = i.groupby('Cabin')['Fare'].mean()  # step 3



print('Cabin Data and mean Fare\n',pd.concat([df_TRN.uniq,df_TST.uniq,df_TRN.cost,df_TST.cost], axis=1,keys=['TRAIN Cabin', 'TEST Cabin', 'Train Fare', 'Test Fare']))
#  create dataframes with Cabin != N

df_TRN_noN = df_TRN[df_TRN['Cabin'] != 'N']

df_TST_noN = df_TST[df_TST['Cabin'] != 'N']



#  plot TRAIN Cabin and Fare

fig = plt.figure(figsize=(10,4))

fig.add_subplot(121)

plt.title('TRAIN - Cabin')

sns.countplot(data=df_TRN_noN, x=df_TRN_noN['Cabin'].sort_values())

fig.add_subplot(122)

plt.title('TRAIN - Fare vs. Cabin')

sns.lineplot(data=df_TRN_noN, x='Cabin', y='Fare')

plt.show()



#  plot TEST Cabin and Fare

fig = plt.figure(figsize=(10,4))

fig.add_subplot(121)

plt.title('TEST - Cabin')

sns.countplot(data=df_TST_noN, x=df_TST_noN['Cabin'].sort_values())

fig.add_subplot(122)

plt.title('TEST - Fare vs. Cabin')

sns.lineplot(data=df_TST_noN, x='Cabin', y='Fare')

plt.show()
#  Step 4 - Assign Cabin letter to all "N" based on Fare

df_TRN.groupby('Cabin')['Fare'].mean().sort_values()

def cabin_fillN_TRN(i):

    j = 0

    if i < 16:

        j = "G"

    elif i >= 16 and i <27:

        j = "F"

    elif i >= 27 and i <37:

        j = "T"

    elif i >= 37 and i <43:

        j = "A"

    elif i >= 43 and i <51:

        j = "E"

    elif i >= 51 and i <79:

        j = "D"

    elif i >= 79 and i <107:

        j = "C"

    else:

        j = "B"

    return j



df_TST.groupby('Cabin')['Fare'].mean().sort_values()

def cabin_fillN_TST(i):

    j = 0

    if i < 17:

        j = "G"

    elif i >= 17 and i <30:

        j = "F"

    elif i >= 30 and i <43:

        j = "D"

    elif i >= 43 and i <64:

        j = "A"

    elif i >= 64 and i <103:

        j = "E"

    elif i >= 103 and i <133:

        j = "C"

    else:

        j = "B"

    return j



#  Run function - fill out all of Cabin per mean

print("BEFORE\ntraining Cabin values:\t",df_TRN.Cabin.sort_values().unique())

print("test Cabin values:\t",df_TST.Cabin.sort_values().unique())



df_TRN.Cabin =  df_TRN.Fare.apply(cabin_fillN_TRN)

df_TST.Cabin =  df_TST.Fare.apply(cabin_fillN_TST)



print("\nAFTER\ntraining Cabin values:\t",df_TRN.Cabin.sort_values().unique())

print("test Cabin values:\t",df_TST.Cabin.sort_values().unique())
#  heatmap of null values

fig = plt.figure(figsize=(12,5))

fig.add_subplot(121)

plt.title('df_TRN - "NULLs\"')

sns.heatmap(df_TRN.isnull(), cmap='gray')

fig.add_subplot(122)

plt.title('df_TST - "NULLs\"')

sns.heatmap(df_TST.isnull(), cmap='gray')

plt.show()



for i in [df_TRN,df_TST]:

    nulls = i.isnull().sum().sort_values(ascending = False)

    prcet = round(nulls/len(i)*100,2)

    i.null = pd.concat([nulls, prcet], axis = 1,keys= ['Total', 'Percent'])

print(pd.concat([df_TRN.null,df_TST.null], axis=1,keys=['TRAIN Data - NULL', 'TEST Data - NULL']))



#  check for duplicate values

print('\n\nDuplicated - TRAIN:  {}'.format(df_TRN.duplicated().sum()))

print('Duplicated - TEST:   {}'.format(df_TST.duplicated().sum()))
df_TRN['Title'] = df_TRN['Name'].str.split(',', expand = True)[1].str.split(' ', expand = True)[1]

df_TST['Title'] = df_TST['Name'].str.split(',', expand = True)[1].str.split(' ', expand = True)[1]



#  create a list of all the titles

titles = sorted(pd.concat([df_TRN['Title'], df_TST['Title']]).unique())

print(titles)
#  Check Titles that may be male or female

for i in [df_TRN,df_TST]:

    print(i[['PassengerId','Title','Sex']][i['Title'].isin(['Capt.', 'Col.', 'Dr.', 'Major.', 'Rev.', 'the'])],"\n")

    

#  

print(df_TRN[['PassengerId','Title','Sex']][df_TRN['PassengerId'].isin([760,797])])
def replace_titles(x):

    title=x['Title']

    if title in ['Don.', 'Sir.']:

        return 'Mr.'       # adult male

    elif title in ['Jonkheer.', 'Master.']:

        return 'Master.'    # young male

    elif title in ['Dona.', 'Lady.', 'Mme.']:

        return 'Mrs.'      # adult female

    elif title in ['Mlle.', 'Ms.']:

        return 'Miss.'     #  young female

    elif title in ['Capt.', 'Col.', 'Dr.', 'Major.', 'Rev.', 'the']:

        if x['Sex']=='Male':

            return 'Mr.'

        else:

            return 'Mrs.'

    else:

        return title



#  Run the function

df_TRN['Title']=df_TRN.apply(replace_titles, axis=1)

df_TST['Title']=df_TST.apply(replace_titles, axis=1)



print('df_TRN Titles:\t',df_TRN.Title.unique())

print('df_TST Titles:\t',df_TST.Title.unique())
#  Re-Check Titles that may be male or female

print(df_TRN[['PassengerId','Title','Sex']][df_TRN['PassengerId'].isin([760,797])])
fig = plt.figure(figsize=(12,5))

fig.add_subplot(121)

plt.title('df_TRN - "Title\"')

ax = sns.countplot(data = df_TRN, x = 'Title')

for p in ax.patches:

    ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

fig.add_subplot(122)

plt.title('df_TST - "Title\"')

ax = sns.countplot(data = df_TST, x = 'Title')

for p in ax.patches:

    ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()
for i in [df_TRN,df_TST]:

    i['FamilySize'] = i['SibSp'] + i['Parch'] + 1

    #  IsAlone - create attribute

    i['IsAlone']  = 0    # set default to '0'

    i.loc[i['FamilySize'] == 1, 'IsAlone'] = 1



#  check

print(sorted(df_TRN.FamilySize.unique()))

print(sorted(df_TST.FamilySize.unique()))
fig = plt.figure(figsize=(12,4))

fig.add_subplot(121)

plt.title('df_TRN - "FamilySize\"')

ax = sns.countplot(data = df_TRN, x = 'FamilySize')

for p in ax.patches:

    ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

fig.add_subplot(122)

plt.title('df_TST - "FamilySize\"')

ax = sns.countplot(data = df_TST, x = 'FamilySize')

for p in ax.patches:

    ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()
df_TRN['FarePerPerson'] = df_TRN['Fare']/(df_TRN['FamilySize'])

df_TST['FarePerPerson'] = df_TST['Fare']/(df_TST['FamilySize'])



print(df_TRN[['Fare','FamilySize','FarePerPerson']].groupby(['FamilySize']).mean())
fig = plt.figure(figsize=(12,4))

fig.add_subplot(121)

plt.title('df_TRN - "FarePerPerson\"')

sns.barplot(x='FamilySize',y='FarePerPerson', data=df_TRN)

fig.add_subplot(122)

plt.title('df_TST - "FarePerPerson\"')

sns.barplot(x='FamilySize',y='FarePerPerson', data=df_TST)

plt.show()
def ageGroup(i):

    j = 0

    if i < 1:

        j = "Infant"

    elif i >= 1 and i <13:

        j = "Child"

    elif i >= 13 and i <19:

        j = "Teenager"

    elif i >= 19 and i <35:

        j = "Young Adult"

    elif i >= 35 and i <65:

        j = "Adult"

    else:

        j = "Elderly"

    return j



df_TRN['AgeGroup']=df_TRN.Age.apply(ageGroup)

df_TST['AgeGroup']=df_TST.Age.apply(ageGroup)



print(df_TRN.AgeGroup.unique())
fig = plt.figure(figsize=(12,4))

fig.add_subplot(121)

plt.title('df_TRN - "AgeGroup\"')

ax = sns.countplot(data = df_TRN, x = 'AgeGroup',order=['Infant','Child','Teenager','Young Adult','Adult','Elderly'])

for p in ax.patches:

    ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

fig.add_subplot(122)

plt.title('df_TST - "AgeGroup\"')

ax = sns.countplot(data = df_TST, x = 'AgeGroup',order=['Infant','Child','Teenager','Young Adult','Adult','Elderly'])

for p in ax.patches:

    ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()
#  Funtion survival_rate

#  input:  attributes

#  output:  printsurvival rates for all unique values in the attribute

def survival_rate(*args):

    for i in args:

        print("{:12}   ---------------------------------".format(i.upper()))

        x = sorted(df_TRN[i].unique())  # values in attribute

        for j in x:

            y = len(df_TRN[i][(df_TRN[i] == j) & (df_TRN['Survived'] == 1)])  # survived number

            z = len(df_TRN[i][df_TRN[i] == j])   # total number

            print('   {:<12}{:3} out of {:3} survived -  {:3.2%}'.format(j,y,z,y/z))

    return

print("\tfunction \'survival_rate\' created.")
fig = plt.figure(figsize=(6,6))

sns.countplot(x='Survived',data=df_TRN, palette='Set1')

plt.title('Overall Survival (training dataset)',fontsize= 16)

plt.xlabel('Passenger Fate',fontsize = 14)

plt.ylabel('Count/Percentage of Passengers',fontsize = 14)

plt.axis('auto')

plt.xticks(np.arange(2), ['died', 'survived'])

labels = df_TRN['Survived'].value_counts()

for x, y in enumerate(labels):

    z = "{}\n({:.2%})".format(y,y/len(df_TRN))

    plt.text(x, y-60, str(z), ha = 'center', va='center', size = 18)

plt.show()
plotList = ['Sex', 'Title']



fig = plt.figure(figsize=(12,5))

plotNum  = 1     # initialize plot number

for i in plotList:

    fig.add_subplot(1,2,plotNum)

    ax = sns.countplot(x=sorted(df_TRN[i]),hue='Survived',data=df_TRN)

    plt.title('Survival per \"{}\"'.format(i), fontsize=14)

    plt.xlabel(i, fontsize=12)

    plt.ylabel('Survival Rate', fontsize=12)

    plt.axis('auto')

    plt.legend(('died', 'survived'), loc='best')

    for p in ax.patches:

        ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plotNum = plotNum + 1

plt.show()



#  Survival Rate

for i in plotList:

    survival_rate(i)
plotList = ['Pclass','Cabin','Embarked']



fig = plt.figure(figsize=(12,12))

plotNum  = 1     # initialize plot number

for i in plotList:

    fig.add_subplot(2,2,plotNum)

    ax = sns.countplot(x=sorted(df_TRN[i]),hue='Survived',data=df_TRN)

    plt.title('Survival Rate per \"{}\"'.format(i), fontsize=14)

    plt.xlabel(i, fontsize=12)

    plt.ylabel('Survival Rate', fontsize=12)

    plt.axis('auto')

    plt.legend(('died', 'survived'), loc='best')

    for p in ax.patches:

        ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plotNum = plotNum + 1

    if i == 'Pclass':  #  add kde plot for Pclass (numeric)

        fig.add_subplot(2,2,plotNum)

        df_TRN.Pclass[df_TRN.Survived == 0].plot(kind='kde')

        df_TRN.Pclass[df_TRN.Survived == 1].plot(kind='kde')

        plt.title('Survival Rate per \"{}\"'.format(i), fontsize=14)

        plt.xlabel(i, fontsize=12)

        plt.ylabel('Survival Rate', fontsize=12)

        plt.legend(('died', 'survived'), loc='best')

        plt.xlim(0,4)

        plotNum = plotNum + 1

plt.show()



#  Survival Rate

for i in plotList:

    survival_rate(i)
plotList = ['SibSp', 'Parch', 'FamilySize']



fig = plt.figure(figsize=(12,16))

plotNum  = 1     # initialize plot number

for i in plotList:

    fig.add_subplot(3,2,plotNum)

    ax = sns.countplot(x=sorted(df_TRN[i]),hue='Survived',data=df_TRN)

    plt.title('Survival Rate per \"{}\"'.format(i), fontsize=14)

    plt.xlabel(i, fontsize=12)

    plt.ylabel('Survival Rate', fontsize=12)

    plt.axis('auto')

    plt.legend(('died', 'survived'), loc='best')

    for p in ax.patches:

        ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plotNum = plotNum + 1

    fig.add_subplot(3,2,plotNum)

    df_TRN[i][df_TRN.Survived == 0].plot(kind='kde')

    df_TRN[i][df_TRN.Survived == 1].plot(kind='kde')

    plt.title('Survival Rate per \"{}\"'.format(i), fontsize=14)

    plt.xlabel(i)

    plt.legend(('died', 'survived'), loc='best')

    plt.xlim(0,6)

    plotNum = plotNum + 1

plt.show()



#  Survival Rate

for i in plotList:

    survival_rate(i)
fig = plt.figure(figsize=(12,6))

#  Survival per Age

fig.add_subplot(121)

df_TRN.Age[df_TRN.Survived == 0].plot(kind='kde')

df_TRN.Age[df_TRN.Survived == 1].plot(kind='kde')

plt.title('Survival Rate per Age', fontsize=14)

plt.legend(('died', 'survived'), loc='best')

plt.xlim(0,100)



fig.add_subplot(122)

#  Survival per AgeGroup

ax = sns.countplot(x='AgeGroup',data=df_TRN,hue='Survived',order=['Infant','Child','Teenager','Young Adult','Adult','Elderly'], palette='hsv')

plt.title('Survival per Age Group',fontsize= 14)

plt.xlabel('Age Groups',fontsize = 12)

plt.ylabel('Count',fontsize = 12)

plt.legend(['died', 'survived'])

plt.axis('auto')

for p in ax.patches:

   ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()



survival_rate('AgeGroup')

print('\nmean age: {:.2f}.'.format(df_TRN.Age.mean()))

print('mode age: {:.2f}.'.format(df_TRN.Age.mode()[0]))
#  Plot - Survival/Death for Fare

plt.figure(figsize=(12,6))

df_TRN.Fare[df_TRN.Survived == 0].plot(kind='kde')

df_TRN.Fare[df_TRN.Survived == 1].plot(kind='kde')

plt.title('Survival Rate per Fare', fontsize=14)

plt.legend(('died', 'survived'), loc='best')

plt.xlim(0,200)



print('max fare: ${:.2f}.'.format(df_TRN.Fare.max()))
#  Plot - Survival/Death for FarePerPerson

plt.figure(figsize=(12,6))

df_TRN.FarePerPerson[df_TRN.Survived == 0].plot(kind='kde')

df_TRN.FarePerPerson[df_TRN.Survived == 1].plot(kind='kde')

plt.title('Survival Rate per FarePerPerson', fontsize=14)

plt.legend(('died', 'survived'), loc='best')

plt.xlim(0,120)
fig = plt.figure(figsize=(12,5))

fig.add_subplot(121)

df_TRN.Age[df_TRN.Pclass == 1].plot(kind='kde')

df_TRN.Age[df_TRN.Pclass == 2].plot(kind='kde')

df_TRN.Age[df_TRN.Pclass == 3].plot(kind='kde')

plt.title('Age Distribution per Class')

plt.xlabel('Age')

plt.legend(('1st Class','2nd Class','3rd Class'), loc='best')

plt.xlim(0,100)



fig.add_subplot(122)

df_TRN.Fare[df_TRN.Pclass == 1].plot(kind='kde')

df_TRN.Fare[df_TRN.Pclass == 2].plot(kind='kde')

df_TRN.Fare[df_TRN.Pclass == 3].plot(kind='kde')

plt.title('Fare Distribution per Class')

plt.xlabel('Fare')

plt.legend(('1st Class','2nd Class','3rd Class'), loc='best')

plt.xlim(0,120)

plt.show()



#  Statistical Summary 

df_class = pd.DataFrame(columns = {'1st-Age','1st-Fare','2nd-Age','2nd-Fare','3rd-Age','3rd-Fare'})

df_class[['1st-Age','1st-Fare']] = df_TRN[['Age','Fare']][df_TRN.Pclass == 1].describe()

df_class[['2nd-Age','2nd-Fare']] = df_TRN[['Age','Fare']][df_TRN.Pclass == 2].describe()

df_class[['3rd-Age','3rd-Fare']] = df_TRN[['Age','Fare']][df_TRN.Pclass == 3].describe()

df_class = df_class[['1st-Age','2nd-Age','3rd-Age','1st-Fare','2nd-Fare','3rd-Fare']]

df_class
df_TRN['AgeBand'] = pd.cut(df_TRN['Age'],5)

df_TST['AgeBand'] = pd.cut(df_TST['Age'],5)

df_TRN[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for i in [df_TRN,df_TST]:

    #  Age - ordinal values

    i.loc[i['Age'] <= 16, 'Age'] = 0

    i.loc[(i['Age'] > 16) & (i['Age'] <= 32), 'Age'] = 1

    i.loc[(i['Age'] > 32) & (i['Age'] <= 48), 'Age'] = 2

    i.loc[(i['Age'] > 48) & (i['Age'] <= 64), 'Age'] = 3

    i.loc[i['Age'] > 64, 'Age'] = 4



ax = sns.countplot(x=df_TRN.Age)

plt.title('df_TRN.Age')

for p in ax.patches:

   ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()
df_TRN['FareBand'] = pd.qcut(df_TRN['Fare'],4)

df_TST['FareBand'] = pd.qcut(df_TST['Fare'],4)

df_TRN[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for i in [df_TRN,df_TST]:

    #  Fare - ordinal values

    i.loc[i['Fare'] <= 7.91, 'Fare'] = 0

    i.loc[(i['Fare'] > 7.91) & (i['Fare'] <= 14.454), 'Fare'] = 1

    i.loc[(i['Fare'] > 14.454) & (i['Fare'] <= 31), 'Fare'] = 2

    i.loc[i['Fare'] > 31, 'Fare'] = 3



ax = sns.countplot(x=df_TRN.Fare)

plt.title('df_TRN.Fare')

for p in ax.patches:

   ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()
df_TRN['AgeClass'] = df_TRN['Age'] * df_TRN['Pclass']

df_TST['AgeClass'] = df_TST['Age'] * df_TST['Pclass']



ax = sns.countplot(x=df_TRN.AgeClass)

plt.title('df_TRN.AgeClass')

for p in ax.patches:

   ax.annotate("%.0f" % p.get_height(),(p.get_x()+p.get_width()/2.,p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.show()
for i in [df_TRN,df_TST]:

    i['Sex'] = i['Sex'].map({'male':0, 'female':1})

    i['Embarked'] = i['Embarked'].map({'C':0,'Q':1,'S':2})

    i['Title'] = i['Title'].map({'Mr.':0,'Mrs.':1,'Master.':2,'Miss.':3})



df_TRN[['Sex','Embarked','Title']].head(5)
#  copy the PassengerId to another dataframe

#  will need it for submission

passID = df_TST.filter(['PassengerId'], axis=1)



drop_columns = ['AgeBand','AgeGroup','Cabin','FareBand','Name','Parch','PassengerId','SibSp','Ticket']



for i in drop_columns:

    df_TRN = df_TRN.drop(i, axis=1)

    df_TST = df_TST.drop(i, axis=1)



print(df_TRN.columns.values)

print(df_TST.columns.values)
#  change type to int

for i in ['Age','Fare','IsAlone','AgeClass']:

    df_TRN[i] = df_TRN[i].astype('int64')

    df_TST[i] = df_TST[i].astype('int64')
df_TRN.head()
from sklearn.preprocessing import MinMaxScaler



normTRN = MinMaxScaler().fit_transform(df_TRN)

normTST = MinMaxScaler().fit_transform(df_TST)



#  create dataframe with normalized data

df_TRN = pd.DataFrame(normTRN, index=df_TRN.index, columns=df_TRN.columns)

df_TST = pd.DataFrame(normTST, index=df_TST.index, columns=df_TST.columns)



#  move Survived columns to first position

col = df_TRN['Survived']

df_TRN.drop(labels=['Survived'], axis=1, inplace = True)

df_TRN.insert(0, 'Survived', col)

df_TRN.columns.values



df_TRN.head()
from sklearn.ensemble import ExtraTreesClassifier



y = df_TRN['Survived']

#  sort by column names

X = df_TRN.drop(['Survived'], axis=1).sort_index(axis=1)



# Building the model 

extra_tree_forest = ExtraTreesClassifier(n_estimators = 5,criterion ='entropy', max_features = 5) 

# Training the model 

extra_tree_forest.fit(X, y) 

# Computing the importance of each feature 

feature_importance = extra_tree_forest.feature_importances_ 

# Normalizing the individual importances 

feature_importance_normalized = np.std([tree.feature_importances_ for tree in extra_tree_forest.estimators_], axis = 0) 



# Plot - compare feature importance

plt.figure(figsize=(8,6))

sns.barplot(x=feature_importance_normalized,y=X.columns)

plt.xlabel('Feature Importance',fontsize=12)

plt.ylabel('Feature',fontsize=12)

plt.title('Comparison of Feature Importances', fontsize=14)

plt.show()
#  Correlation TABLE

corrALL = df_TRN.corr()['Survived'].sort_values(ascending=False)

corrALL = corrALL.drop(['Survived'])



#  heatmap and barplot

fig = plt.figure(figsize=(16,8))

fig.add_subplot(121)

plt.title('Titanic Survival - Correlation-OVERALL', fontsize=14)

sns.heatmap(df_TRN.corr(), annot=True, fmt='.2f', square=True, cmap = 'Greens')

fig.add_subplot(122)

plt.title('Titanic Survival - Correlation-OVERALL', fontsize=14)

ax = sns.barplot(y=corrALL.index,x=corrALL.values)

for i in ax.patches: 

    plt.text(i.get_width()/1.5, i.get_y()+.5,  

             str(round((i.get_width()), 4)), 

             fontsize = 12, fontweight ='bold', 

             color ='black')

plt.show()
#  Correlation FEMALE - filter dataframe for female

dataFemale = df_TRN[(df_TRN['Sex'] == 1)]

dataFemaleCorr = dataFemale.drop(["Sex","Title"], axis=1).corr()

corrF = dataFemaleCorr['Survived'].sort_values(ascending=False)

corrF = corrF.drop(['Survived'])



#  heatmap and barplot

fig = plt.figure(figsize=(12,6))

fig.add_subplot(121)

plt.title('Titanic Survival - Correlation-FEMALE', fontsize=14)

sns.heatmap(dataFemaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Reds')

fig.add_subplot(122)

plt.title('Titanic Survival - Correlation-FEMALE', fontsize=14)

ax = sns.barplot(y=corrF.index,x=corrF.values)

for i in ax.patches: 

    plt.text(i.get_width()/1.5, i.get_y()+.5,  

             str(round((i.get_width()), 4)), 

             fontsize = 12, fontweight ='bold', 

             color ='black')

plt.show()
dataMale   = df_TRN[(df_TRN['Sex'] == 0)]

dataMaleCorr = dataMale.drop(["Sex","Title"], axis=1).corr()

corrM = dataMaleCorr['Survived'].sort_values(ascending=False)

corrM = corrM.drop(['Survived'])



#  heatmap and barplot

fig = plt.figure(figsize=(12,6))

fig.add_subplot(121)

plt.title('Titanic Survival - Correlation-MALE', fontsize=14)

sns.heatmap(dataMaleCorr, annot=True, fmt='.2f', square=True, cmap = 'Blues')

fig.add_subplot(122)

plt.title('Titanic Survival - Correlation-MALE', fontsize=14)

ax = sns.barplot(y=corrM.index,x=corrM.values)

for i in ax.patches: 

    plt.text(i.get_width()/1.5, i.get_y()+.5,  

             str(round((i.get_width()), 4)), 

             fontsize = 12, fontweight ='bold', 

             color ='black')

plt.show()
corrALL = pd.DataFrame(columns = ['MALE','correlation-m','FEMALE','correlation-f'])

corrALL['MALE']   = corrM.index

corrALL['correlation-m'] = corrM.values

corrALL['FEMALE'] = corrF.index

corrALL['correlation-f'] = corrF.values

corrALL
from sklearn.model_selection import train_test_split



X = df_TRN.drop(['Survived'], axis = 1)

y = df_TRN['Survived']



seed = 7

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = seed)



print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)



X_train.head()
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier



#  Define the Classification Models

models = []

models.append(('DT   ', DecisionTreeClassifier()))

models.append(('KNN  ', KNeighborsClassifier()))

models.append(('LR   ', LogisticRegression(solver='liblinear', multi_class='ovr')))

models.append(('NB   ', GaussianNB()))

models.append(('RF   ', RandomForestClassifier()))

models.append(('SVC  ', SVC(gamma='auto')))

models.append(('lSVC ', LinearSVC()))
from sklearn import model_selection

from sklearn.metrics import accuracy_score



#  Evaluate the Models:

results = []

names = []

modelDF = pd.DataFrame(columns=['model','CV-mean','CV-std','AccuracyScore'])

countDF = 0



for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    results.append(cv_results)

    names.append(name)

    model.fit(X_train,y_train)

    modelPredict = model.predict(X_test)

    accu = accuracy_score(y_test,modelPredict)

    print("{0:s}:  {1:3.5f}  ({2:3.5f})  {3:.2%}".format(name,cv_results.mean(),cv_results.std(),accu))

    modelDF.loc[countDF]=[name,cv_results.mean(),cv_results.std(),accu]

    countDF = countDF + 1
#  pick the best model from ModelDF

max(modelDF['CV-mean'])

maxCV = modelDF[(modelDF['CV-mean'] == max(modelDF['CV-mean']))]

maxCV
best_model = KNeighborsClassifier()

best_model.fit(X_train,y_train)

print(best_model)



#  predict

y_predict = best_model.predict(X_test)

y_predict[0:10]
from sklearn.model_selection import cross_val_score



cross_val = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')



print("Cross Validation Scores:         {}".format(cross_val))

print('Cross Validation Scores - mean:  {:3.4%}'.format(cross_val.mean()))
from sklearn.metrics import accuracy_score



accuracy_score(y_test,y_predict)

print('Accuracy Score:  {:3.4%}'.format(accuracy_score(y_test,y_predict)))
from sklearn.metrics import f1_score



f1score = f1_score(y_test, y_predict)

print('F1 Score:  {:3.4%}'.format(f1score))
from sklearn.metrics import confusion_matrix



conf_matrix = confusion_matrix(y_test, y_predict)



sns.heatmap(conf_matrix, annot=True,cmap='Blues',annot_kws={"size": 36})

plt.title("Confusion Matrix, F1 Score: {:3.4%}".format(f1score))

plt.show()
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



best_model.probability = True   # need for predict_proba to work

best_model.fit(X_train,y_train)

y_predita = best_model.predict_proba(X_test)

y_predita = y_predita[:,1]   # positive values only



ROC_AUC = roc_auc_score(y_test, y_predita)

fpr, tpr, thresholds = roc_curve(y_test, y_predita)



plt.plot([0,1],[0,1], linestyle='--')

plt.plot(fpr, tpr, marker='.')

plt.title("ROC Curve, ROC_AUC Score: {:3.4%}".format(ROC_AUC))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
from sklearn.metrics import classification_report



print(classification_report(y_test,y_predict))
from sklearn.metrics import log_loss



y_predict_prob = best_model.predict_proba(X_test)

print(y_predict_prob[0:5])



print("\nLog Loss:  {:3.4}".format(log_loss(y_test, y_predict_prob)))
# Check shape of TEST data

print('Test data shapes must match order to \"fit\":')

print('shape X_test:\t{}'.format(X_test.shape))

print('shape y_test:\t{}'.format(y_test.shape))

print('shape df_TST:\t{}'.format(df_TST.shape))
print(best_model)

best_model.fit(X_test,y_test)          #  fit

SF = best_model.predict(df_TST)        #  predictions



#  Add PassengerId back into test data

df_TST['PassengerId'] = passID['PassengerId']



SF = pd.DataFrame(SF, columns=['Survived'])

SF_TST = pd.concat([df_TST, SF], axis=1, join='inner')

SF_final = SF_TST[['PassengerId','Survived']].astype("int64")



#  SF_final.to_csv('<path>/predictions.csv', index=False)

print('all done ...')