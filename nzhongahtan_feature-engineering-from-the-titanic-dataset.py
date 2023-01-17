import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import math
train = pd.read_csv('../input/titanic/train.csv')



train.head(5)
cols = train.columns

for i in cols:

    print(i, ": ", train[i].isnull().sum())
plt.bar([1,2,3], height = [(train.Pclass==1).sum(),(train.Pclass==2).sum(),(train.Pclass==3).sum()])

plt.title('Passenger Class')

plt.xticks([1,2,3])

plt.xlabel('Class')

plt.ylabel('Number of People')
pclass = pd.get_dummies(train.Pclass,prefix = 'Pclass')

train = pd.concat([train,pclass],axis = 1)



train.head()
train.Name.unique()[0:20]
titles = []

for i in range (0,len(train)):

    start = train['Name'][i].find(',') + 2

    end = train['Name'][i].find('.')

    titles.append(train['Name'][i][start:end])

train['Titles'] = titles



train.head()
train.Titles.unique()
fig1, ax1 = plt.subplots()

sizes = []

for i in train.Titles.unique():

    sizes.append((train.Titles==i).sum())

ax1.pie(sizes, labels = train.Titles.unique(),autopct = '%1.1f%%')

ax1.axis('equal')

plt.show()
train['SpecialTitle'] = train['Titles'].apply(lambda x: True if x != 'Mr' and x != 'Mrs' and x!= 'Miss' else False)



fig1, ax1 = plt.subplots()

sizes = []

for i in train.SpecialTitle.unique():

    sizes.append((train.SpecialTitle==i).sum())

ax1.pie(sizes, labels = train.SpecialTitle.unique(),autopct = '%1.1f%%')

ax1.axis('equal')

plt.show()

train['Titles'] = train['Titles'].astype('category').cat.codes



train.head()
length = []

for i in range(len(train.Name)):

    end = train['Name'][i].find(',')

    length.append(end)



train['LastNameLength'] = length



counts = []

for i in sorted(list(train.LastNameLength.unique())):

    counts.append((train.LastNameLength == i).sum())

plt.bar(sorted(list(train.LastNameLength.unique())), height = counts)

plt.title('Last Name Lengths')

plt.xlabel('Length')

plt.ylabel('Number of People')
length = []

for i in range(len(train.Name)):

    beg = train['Name'][i].find('.')

    if '(' in train['Name'][i]:

        end = train['Name'][i].find('(')

        if (end-beg-3 < 0):

            close = train['Name'][i].find(')')

            length.append(close-end-1)

        else:

            length.append(end-beg-3)

    else:

        length.append(len(train['Name'][i]) - beg - 1)

        

train['FMNameLength'] = length





counts = []

for i in sorted(list(train.FMNameLength.unique())):

    counts.append((train.FMNameLength == i).sum())

plt.bar(sorted(list(train.FMNameLength.unique())), height = counts)

plt.title('First and Middle Name Lengths')

plt.xlabel('Length')

plt.ylabel('Number of People')
train['NameLength'] = train['Name'].apply(lambda x:len(x))



counts = []

for i in sorted(list(train.NameLength.unique())):

    counts.append((train.NameLength == i).sum())

plt.bar(sorted(list(train.NameLength.unique())), height = counts)

plt.title('Name Lengths')

plt.xlabel('Length')

plt.ylabel('Number of People')
train['SecondName'] = train['Name'].apply(lambda x: True if '(' in x else False)



fig1, ax1 = plt.subplots()

sizes = []

for i in train.SecondName.unique():

    sizes.append((train.SecondName==i).sum())

ax1.pie(sizes, labels = train.SecondName.unique(),autopct = '%1.1f%%')

ax1.axis('equal')

plt.show()

train.head()
fig1, ax1 = plt.subplots()

sizes = []

for i in train.Sex.unique():

    sizes.append((train.Sex==i).sum())

ax1.pie(sizes, labels = train.Sex.unique(),autopct = '%1.1f%%')

ax1.axis('equal')

plt.show()



train['Sex'] = train['Sex'].astype('category').cat.codes
train['AgeNull'] = train['Age'].apply(lambda x: math.isnan(x))

train['Age'] = train['Age'].fillna(train['Age'].mean())



plt.boxplot(train.Age)



print(train.head(5))
train['Child'] = train['Age'].apply(lambda x: True if x < 18 else False)

train['Adult'] = train['Age'].apply(lambda x: True if x >= 18 and x < 65 else False)

train['Senior'] = train['Age'].apply(lambda x: True if x >= 65 else False)



plt.bar(['Child','Adult','Senior'], height = [(train.Child == True).sum(), (train.Adult == True).sum(), (train.Senior == True).sum()])

plt.title('Number of People Based on Age Groups')

plt.xlabel('Age Groups')

plt.ylabel('Number of People')



train.head()
train['EstimatedAge'] = train['Age'].apply(lambda x: True if round(x) != x else False)



fig1, ax1 = plt.subplots()

sizes = []

for i in train.EstimatedAge.unique():

    sizes.append((train.EstimatedAge==i).sum())

ax1.pie(sizes, labels = train.EstimatedAge.unique(),autopct = '%1.1f%%')

ax1.axis('equal')

plt.show()
train['Family'] = train['SibSp'] + train['Parch']



train.head()
counts = []

for i in sorted(list(train.Family.unique())):

    counts.append((train.Family == i).sum())

plt.bar(sorted(list(train.Family.unique())), height = counts)

plt.title('Family Members Excluding Oneself')

plt.xlabel('Count')

plt.ylabel('Number of People')
counts = []

for i in sorted(list(train.SibSp.unique())):

    counts.append((train.SibSp == i).sum())

plt.bar(sorted(list(train.SibSp.unique())), height = counts)

plt.title('Siblings and Spouses')

plt.xlabel('Count')

plt.ylabel('Number of People')
counts = []

for i in sorted(list(train.Parch.unique())):

    counts.append((train.Parch == i).sum())

plt.bar(sorted(list(train.Parch.unique())), height = counts)

plt.title('Parents and Children')

plt.xlabel('Count')

plt.ylabel('Number of People')
nanny = []

for i in range(0,len(train)):

    if (train['Parch'][i] ==0 and train['Age'][i] < 18):

        nanny.append(True)

    else:

        nanny.append(False)

train['Nanny'] = nanny



print(train.head())



fig1, ax1 = plt.subplots()

sizes = []

for i in train.Nanny.unique():

    sizes.append((train.Nanny==i).sum())

ax1.pie(sizes, labels = train.Nanny.unique(),autopct = '%1.1f%%')

ax1.axis('equal')

plt.show()
train.Ticket[:5]
nums = []

prefix = []

for i in range(len(train.Ticket)):

    space = train['Ticket'][i].find(' ')

    if space != -1:

        prefix.append(train['Ticket'][i][:space])

    else:

        prefix.append('')

    if space != 0:

        nums.append(train['Ticket'][i][space+1:])

    else:

        nums.append(train['Ticket'][i])



print(nums[:5])



print(prefix[:5])
punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''

for pre in range (len(prefix)):

    for p in range (len(punc)):

        prefix[pre] = prefix[pre].replace(punc[p],'').upper()

        

prefix[:5]
for i in range (len(nums)):

    if '2. ' in nums[i]:

        nums[i] = nums[i][3:]

train['TicketNum'] = nums

train['TicketNum'] = train['TicketNum'].replace({'2. 3101285': '3101285','Basle 541':'541','LINE':'0','2. 3101294':'3101294'})

train['TicketNum'] = train['TicketNum'].astype('int')



train['TicketPrefix'] = prefix

train['TicketPrefix'] = train['TicketPrefix'].astype('category').cat.codes



print(nums[:5])
counts = []

for i in sorted(list(train.TicketPrefix.unique())):

    counts.append((train.TicketPrefix == i).sum())

plt.bar(sorted(list(train.TicketPrefix.unique())), height = counts)

plt.title('Ticket Prefix Counts')

plt.xlabel('Ticket Prefixes')

plt.ylabel('Number of People')
plt.scatter(train.TicketNum,np.ones(len(train)))
train['Fare'] = train['Fare'].fillna(train['Fare'].mode())



plt.boxplot(train.Fare)
train['Cabin'] = train['Cabin'].fillna('Z')

train['CabinLetter'] = train['Cabin'].apply(lambda x: x[0])



train.head()
nums = []

for i in range (0,len(train['Cabin'])):

    end = 0

    for j in range (1,len(train['Cabin'][i])+1):

        if (train['Cabin'][i][-j].isalpha()):

            end = j-1

            break

    nums.append(train['Cabin'][i][-end:])

train['CabinNum'] = nums

train['CabinNum'] = train['CabinNum'].replace({'':'0','Z':'0','D':'0','T':'0'})

train['CabinNum'] = train['CabinNum'].astype(int)





train.head()
plt.scatter(train.CabinNum,np.ones(len(train)))

plt.show()
counts = []

for i in sorted(list(train.CabinLetter.unique())):

    counts.append((train.CabinLetter == i).sum())

plt.bar(sorted(list(train.CabinLetter.unique())), height = counts)

plt.title('Cabin Letter Counts')

plt.xlabel('Cabin Letter')

plt.ylabel('Number of People')



train['CabinLetter'] = train['CabinLetter'].astype('category').cat.codes
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode())

embarked = pd.get_dummies(train.Embarked, prefix = 'Emb')

train = pd.concat([train,embarked],axis = 1)

train = train.drop(['Embarked'],axis = 1)



train.head()
plt.bar(['C','Q','S'],[(train.Emb_C == 1).sum(),(train.Emb_Q == 1).sum(),(train.Emb_S == 1).sum()])