import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/who_suicide_statistics.csv')

df.head()

df = df.dropna()
#SUICIDES BASED ON AGE GROUPS:
df['age'].unique() #string '5-14 years' should be at index 0, the corresponding sums need to be coreected aswell
#Suicides number sumed by age group with correct order

A=[]

for i in df['age'].unique():

    byage = df.loc[(df['age']==i) & (df['suicides_no']) & (df['population']), ['age','suicides_no','population']]

    byage = byage['suicides_no'].sum()

    A.append(byage)

    

# correct index order

age = A

a = age[4 - 1]

ageorder = []

for i in age:

    ageorder.append(i)

    if i == age[3]:

        ageorder.remove(age[3])

        ageorder.insert(0, a)

print(ageorder)



# correct index order

age = df['age'].unique()

a = age[4 - 1]

Newlist = []

for i in age:

    Newlist.append(i)

    if i == age[3]:

        Newlist.remove(age[3])

        Newlist.insert(0, a)

print(Newlist)
plt.bar(Newlist,ageorder)

plt.xlabel('years')

plt.ylabel('Death')

plt.xticks(fontsize=11, rotation=30)

plt.title('Suicides Based on Age Group')

plt.show()
#SUICIDES BASED ON GENDER AND AGE
Female=[]

for i in df['age'].unique():

    byage = df.loc[(df['age']==i) & (df['suicides_no']) & (df['population']) & (df['sex']=='female'), ['age','suicides_no','population', 'sex']]

    byage = byage['suicides_no'].sum()

    Female.append(byage)

print(Female)



age = Female

a = age[4 - 1]

ageorderF = []

for i in age:

    ageorderF.append(i)

    if i == age[3]:

        ageorderF.remove(age[3])

        ageorderF.insert(0, a)

print(ageorderF)
Male=[]

for i in df['age'].unique():

    byage = df.loc[(df['age']==i) & (df['suicides_no']) & (df['population']) & (df['sex']=='male'), ['age','suicides_no','population', 'sex']]

    byage = byage['suicides_no'].sum()

    Male.append(byage)

print(Male)



age = Male

a = age[4 - 1]

ageorderM = []

for i in age:

    ageorderM.append(i)

    if i == age[3]:

        ageorderM.remove(age[3])

        ageorderM.insert(0, a)

print(ageorderM)
N = 6

ind = np.arange(N) 

width = 0.35       

plt.bar(ind, ageorderF, width, label='Women')

plt.bar(ind + width, ageorderM, width, label='Men')



plt.ylabel('Suicides')

plt.title('Suicides by age and gender')



plt.xticks(ind + width / 2, Newlist, fontsize=11, rotation=30)

plt.legend(loc='best')

plt.show()
# SUICIDES IN US
US = df.loc[(df['country']=='United States of America') & (df['year']) & (df['suicides_no']), ['country', 'year', 'suicides_no',]]

US.sort_values(by=['year'])



Y=[]

for i in US['year'].unique():

    byyear = US.loc[(df['year']==i) & (df['suicides_no']), ['year','suicides_no']]

    byyear = byyear['suicides_no'].sum()

    Y.append(byyear)

    

print(Y) # Sum of US Suicide throughout years

Years = np.unique(US.year)



plt.plot(Years,Y)

plt.title('Suicide number in US')

plt.xlabel('years')

plt.ylabel('Suicide')

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split
US_data = US.values 

year = US_data[:,1]

x = np.unique(year).reshape(-1,1) 

y = Y



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size =0.3, random_state=42)



reg = LinearRegression()

reg.fit(x_train, y_train)

y_pred = reg.predict(x_test)
# Plot outputs

plt.scatter(x_test, y_test,  color='black')

plt.plot(x_test, y_pred, color='blue', linewidth=3)

plt.xlabel('Years')

plt.ylabel('No of Suicides')



plt.show()