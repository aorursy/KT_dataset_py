# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/indian-candidates-for-general-election-2019/LS_2.0.csv')

print("Total number of rows having NOTA as the candidate:")

print(df.NAME.value_counts().head(1))

df.ASSETS = df.ASSETS.astype(str) 



df.ASSETS = df.ASSETS.apply(lambda num : num.split("\n",1)[0])

df.ASSETS = df.ASSETS.apply(lambda num : num if num == 'nan' else (num[3:]))

df.ASSETS = df.ASSETS.apply(lambda num : np.nan if num == 'nan'or num == '' or num == ' Available' else (int(num.replace(',',''))))

#print(int(df.ASSETS[1]))

print(df.info())
Percentage = []

Percentage.append((len(df[df.GENDER=='MALE'])/float(2018)))

Percentage.append(1-Percentage[0])

my_labels = ['Male','Female']



Percentage_winner = []

winners = len(df[df.WINNER == 1])

Percentage_winner.append((len(df[(df.GENDER=='MALE') & (df.WINNER == 1)])/float(winners)))

Percentage_winner.append(1-Percentage_winner[0])



plt.figure(figsize = (12,6))

plt.subplot(2,1,1)

plt.pie(Percentage,labels=my_labels,autopct='%1.1f%%')

plt.title('2019 Election Candidates % by Gender')

plt.axis('equal')



plt.subplot(2,1,2)

plt.pie(Percentage_winner,labels=my_labels,autopct='%1.1f%%')

plt.title("Percentage of MP's in Lok Sabha by Gender")

plt.axis('equal')

plt.show()
## Seats won by Women

df_female = df[(df.GENDER=='FEMALE') & (df.WINNER == 1)]

print("Total number of seats won by Women: "+ str(len(df_female)))



# ## State Wise where women won

print("Statwise number of the seats won by women:")

table = pd.pivot_table(df_female, index =['STATE'],values=['WINNER'],aggfunc=np.sum) 

table = table.reindex(table.sort_values(by='WINNER', ascending=False).index)

print(table)



#print(df_female[['NAME','ASSETS']].head(10))



df_female_rich = df_female[(df_female.ASSETS > 30000000)]

print("Number of women MPs with more than 3 core assets:" +  str(len(df_female_rich)))

print("\nBelow are top 10 to give you a look:")

print(df_female_rich.sort_values('ASSETS', ascending=False).loc[:,['NAME','ASSETS']].head(10))
df_winner = df[df.WINNER == 1]

df_winner_edu = df_winner.EDUCATION

df_winner_edu = df_winner_edu.apply(lambda x : 'Post Graduate' if x == 'Post Graduate\n' else (x))

df_winner_edu = df_winner_edu.apply(lambda x : 'Graduate' if x == 'Graduate Professional' else (x))

edu_lvls = df_winner_edu.to_list()



from collections import Counter

X = ['Illiterate', 'Literate','5th Pass', '8th Pass', '10th Pass','12th Pass', 'Graduate','Post Graduate', 'Doctorate','Others']

Values = []

d = Counter(edu_lvls) 

for i in range(0,len(X)):

    Values.append(d[X[i]])



fig = plt.figure()

ax = fig.add_axes([0,0,2.2,2])

ax.bar(X,Values)

plt.show()

    
df_winner = df[df.WINNER == 1]



df_c = df_winner[['CRIMINAL\nCASES','PARTY','STATE','WINNER']]

df_c['CRIMINAL\nCASES'] = df_c['CRIMINAL\nCASES'].apply(lambda num : "Clean" if ( num == '0' ) else ("Have a Record"))

df_clean = df_c[df_c['CRIMINAL\nCASES'] == 'Clean']



#Crime

plt.figure(figsize = (15,8))

plt.style.use('fivethirtyeight')

sns.countplot(x = df_clean['PARTY'],order = df_clean['PARTY'].value_counts().index)

plt.legend(loc='upper right', title='Clean MPs')

plt.xticks(rotation= 90)

plt.show()

print("Total number of MP's with a clean record: "+str(len(df_clean)))

df_crime = df_c[df_c['CRIMINAL\nCASES'] == 'Have a Record']

plt.figure(figsize = (15,6))

plt.style.use('fivethirtyeight')

sns.countplot(x = df_crime['PARTY'],order = df_clean['PARTY'].value_counts().index)

plt.legend(loc='upper right', title='MPs with a criminal History')

plt.xticks(rotation= 90)

plt.show()

print("Total number of MP's with a criminal History: "+str(len(df_crime)))
Percentage = []

Percentage.append(306/float(539))

Percentage.append(1-Percentage[0])

my_labels = ['Clean','Criminal record']



plt.pie(Percentage,labels=my_labels,autopct='%1.1f%%')

plt.title("MPs from crime perspective")

plt.axis('equal')

plt.show()
df_winner = df[df.WINNER == 1]

print("MP's distribution in the Lok Sabha based on caste:\n")

print(df_winner['CATEGORY'].value_counts())
df_cat = df_winner[['CATEGORY','PARTY','STATE']] 

df_cat['CATEGORY'] = df_cat['CATEGORY'].apply(lambda num : num if ( num == 'GENERAL' ) else ('RESERVED'))

plt.figure(figsize = (12,6))

plt.style.use('fivethirtyeight')

sns.countplot(x = df_cat['STATE'],hue=df_cat['CATEGORY'],order = df_cat['STATE'].value_counts().index)

plt.legend(loc='upper right', title="Distribution of Seat's among the states")

plt.xticks(rotation= 90)

plt.show()