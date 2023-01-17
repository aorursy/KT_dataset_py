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



#Setup Seaborn



pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
train_path = '../input/titanic/train.csv'

traindf = pd.read_csv(train_path)

traindf.head()
#Plot ages and fares

plt.figure(figsize=(16,6))

plt.title('Age and Fare of each passenger in the Titanic traininig set')

plt.xlabel('Passenger index')

cols = traindf.loc[:,['Age','Fare']]

sns.lineplot(data=cols)
#Average fare by age

afba = traindf.groupby(['Age']).Fare.mean()

#afba.rename_axis('Average Fare')

plt.figure(figsize=(20,5))

plt.title('Average fare by age')

plt.xlabel('Age')

sns.barplot(x=afba.index, y=afba)
#Fares histogram by gender

afba_hist = traindf.hist(column='Fare', by='Sex', bins=10)
#Age bins specification

agebins = [x for x in range(0,130,10)]

#Categorise by age bin

def agecategorise(a):

    abins = [x-a for x in agebins]

    for i,x in enumerate(abins):

        if x>0:

            return  str(agebins[i-1]) + '-' + str(agebins[i])

    return 'Unknown'

def agegrouping(row):

    row.Age = agecategorise(row.Age)

    return row

#Map age to age group

agecat = traindf.apply(agegrouping, axis='columns')

#Average fare by age group

afbag = agecat.groupby('Age').Fare.mean()

#Max fare by age group

Mfbag = agecat.groupby('Age').Fare.max()

#min fare by age group

mfbag = agecat.groupby('Age').Fare.min()

#Total revenue by age group

trbag = agecat.groupby('Age').Fare.sum()



#Bar plot the respective series

plt.figure(figsize=(12,3))

plt.title('Average fare by age group')

plt.xlabel('Age group')

sns.barplot(x=afbag.index, y=afbag)



plt.figure(figsize=(12,3))

plt.title('Minimum fare by age group')

plt.xlabel('Age group')

sns.barplot(x=mfbag.index, y=mfbag)



plt.figure(figsize=(12,3))

plt.title('Maximum fare by age group')

plt.xlabel('Age group')

sns.barplot(x=Mfbag.index, y=Mfbag)



plt.figure(figsize=(12,3))

plt.title('Total revenue by age group')

plt.xlabel('Age group')

sns.barplot(x=trbag.index, y=trbag)



#plt.figure(figsize=(10,10))

#plt.title('Survival probability by Age group and Fare')

#sns.heatmap(x=spbag.index, y=a)
#Survival probability by age group

spbag = agecat.groupby(['Age']).Survived.mean()

#Concatenate with average fare by age group

con = pd.concat([afbag,spbag],axis=1)

con



con_heatmap = con.pivot(columns='Fare', values='Survived' )

con_heatmap

plt.figure(figsize=(10,7))

plt.title('Survival probability by age group and average fare')

sns.heatmap(con_heatmap)