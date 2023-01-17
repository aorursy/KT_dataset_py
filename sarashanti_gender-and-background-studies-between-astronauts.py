import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/astronaut-yearbook/astronauts.csv',sep=",")
df.shape
df.head()
df.describe()
df.info()
plt.figure(figsize=(8,8))

count_year = df['Year'].value_counts()

sns.countplot(y=df['Year'],order=count_year.nlargest(50).index).set(xlabel='Year',ylabel='Frequency')
gender = df['Gender'].value_counts()



print(gender)



plt.figure(figsize=(5,5))

plt.pie(gender,labels=gender.index,autopct='%1.1f%%')

plt.show()
plt.figure(figsize=(8,8))

count_gender = df['Year'].value_counts()

sns.countplot(y=df["Year"],hue=df["Gender"],order=count_gender.nlargest(50).index)
df['Alma Mater'].value_counts().head(10)
plt.figure(figsize=(11,11))

count_school = df['Alma Mater'].value_counts()

sns.countplot(y=df['Alma Mater'],order=count_school.nlargest(50).index).set(xlabel='Alma Mater',ylabel='Frequency')
df['Undergraduate Major'].value_counts().head(10)
plt.figure(figsize=(8,8))

count_majors1 = df['Undergraduate Major'].value_counts()

sns.countplot(y=df['Undergraduate Major'],order=count_majors1.nlargest(50).index).set(xlabel='Undergraduate Major',ylabel='Frequency')
df['Military Rank'].value_counts().sum()
A = [207, 150]

plt.figure(figsize=(6,6))

plt.pie(A, autopct = '%1.1f%%',labels=("Military","Non-Military"))

plt.show()
gender = df.loc[:,'Gender']

space_hour = df.loc[:,'Space Flight (hr)']

military_service = df.loc[:,'Military Rank']



male = []

male_hr = []



female = []

female_hr = []



for i in range(len(gender)):

    if gender[i] == 'Male':

        male.append(i)        

    else:

        female.append(i)



for j in range(len(male)):

    for k in range(len(space_hour)):

        if male[j] == k:

            male_hr.append(space_hour[k])

        

for n in range(len(female)):

    for o in range(len(space_hour)):

        if female[n] == o:

            female_hr.append(space_hour[o])

            

gen = ('Male','Female')

x = np.arange(len(gen))

hr_av = (np.average(male_hr),np.average(female_hr))



plt.figure(figsize=(5,5))

plt.bar(x,hr_av,align='center',alpha=0.5)

plt.title('Average of Space Flight')

plt.xticks(x,gen)

plt.ylabel('Space Flight (hr)')

plt.show()
male = []

male_military = []



female = []

female_military = []



gender = df.loc[:,'Gender']



#change the missing value in Military Rank column into 0

df['Military Rank'].fillna(0)

military_binary = df['Military Rank'].apply(lambda x:0 if type(x)==float else 1)



for v in range(len(gender)):

    if gender[v] == 'Male':

        male.append(v)

    else:

        female.append(v)



for d in range(len(male)):

    for e in range(len(military_binary)):

        if male[d] == e:

            male_military.append(military_binary[e])

            

for f in range(len(female)):

    for g in range(len(military_binary)):

        if female[f] == g:

            female_military.append(military_binary[g])



sum_male_mil = (male_military.count(0),male_military.count(1))

sum_fem_mil = (female_military.count(0),female_military.count(1))



fig = plt.figure(figsize=(12,5))

ax1 = plt.subplot2grid((1,2),(0,0))

plt.pie(sum_male_mil,colors= ("blue","orange"),labels=("Non-Military Service","Military Service"),autopct='%1.1f%%')

plt.title('Male Astronauts in Military Service')

ax1 = plt.subplot2grid((1,2),(0,1))

plt.pie(sum_fem_mil,colors= ("blue","orange"),labels=("Non-Military Service","Military Service"),autopct='%1.1f%%')

plt.title('Female Astronauts in Military Service')

plt.show()
male_undergrad = []

female_undergrad = []



df['Undergraduate Major'] = df['Undergraduate Major'].replace('',np.nan)

undergrad = df['Undergraduate Major']



for m in range(len(male)):

    for n in range(len(undergrad)):

        if male[m] == n:

            male_undergrad.append(undergrad[n])



for o in range(len(female)):

    for p in range(len(undergrad)):

        if female[o] == p:

            female_undergrad.append(undergrad[o])

            

m_u = pd.Series(np.array(male_undergrad))

f_u = pd.Series(np.array(female_undergrad))



cleaned_m_u = m_u.value_counts().head(11).drop('nan')

cleaned_f_u = f_u.value_counts().head(11).drop('nan')



fig = plt.figure(figsize=(15,5))

ax1 = plt.subplot2grid((1,2),(0,0))

plt.pie(cleaned_m_u,autopct='%1.0f%%',labels=cleaned_m_u.index,startangle=0)

plt.title('Undergrad Major in Male Astronauts')

ax1 = plt.subplot2grid((1,2),(0,1))

plt.pie(cleaned_f_u,autopct='%1.0f%%',labels=cleaned_f_u.index,startangle=0)

plt.title('Undergrad Major in Female Astronauts')

plt.show()
male_mil_undergrad = []

fem_mil_undergrad = []



male_undergrad = []

fem_undergrad = []



for t in range(len(gender)):

    if (gender[t]== 'Male'):

        male_undergrad.append(str(undergrad[t]))

    else:

        fem_undergrad.append(str(undergrad[t]))

    

for q in range(len(male_military)):

    if (male_military[q] == 1):

        male_mil_undergrad.append(str(male_undergrad[q]))



for r in range(len(female_military)):        

    if (female_military[r] == 1):

        fem_mil_undergrad.append(str(fem_undergrad[r]))



mmu2 = pd.Series(np.array(male_mil_undergrad))

fmu2 = pd.Series(np.array(fem_mil_undergrad))



mmu3 = mmu2.value_counts().head(11).drop('nan')

fmu3 = fmu2.value_counts().head(11)



fig = plt.figure(figsize=(15,5))

ax1 = plt.subplot2grid((1,2),(0,0))

plt.pie(mmu3,autopct='%1.0f%%',labels=mmu3.index,startangle=0)

plt.title('Undergrad Major in Male Astronauts with Military Background')

ax1 = plt.subplot2grid((1,2),(0,1))

plt.pie(fmu3,autopct='%1.0f%%',labels=fmu3.index,startangle=0)

plt.title('Undergrad Major in Female Astronauts with Military Background')

plt.show()