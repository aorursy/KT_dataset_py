# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#WE WILL PREDICT WHAT SORT OF PEOPLE SURVIVED.
#read the csv train file into a dataframe we will call df and display the first 10 entries.

df = pd.read_csv("../input/train.csv")

df.head(10)
df.describe()
# count the nan's

# count the nan's

df.apply(lambda x: sum(x.isnull()))
#remove nan's from Age. Store result into a new DF.

df_1 = df.dropna()

df_1.head()
#let's plot this to get an age distribution.

plt.figure(figsize = (30,20))

ax = sns.countplot(df_1.Age)

# let's try a histogram plot

df_1['Age'].hist(bins=40)
# Let's see the distribution of the Ticket Classes sold.

ax= sns.countplot(df.Pclass)
# Let's see the distribution of the ticket prices. 

df.boxplot(column='Fare')
# IQR = 3rd quartile - 1st quartile

IQR = 31.0 - 7.910400

outlier_fence = (IQR * 1.5) + 31.0

print(outlier_fence)
# create new DF consisting of all fares above 65.

expnsiv_fare = df[df['Fare'] > 65.0]

expnsiv_fare.shape
expnsiv_fare.sort_values(['Fare'],ascending=False).head(30)

#arrange a countplot for fares by gender.

expnsiv_sex = expnsiv_fare.Sex

ax=sns.countplot(expnsiv_sex)
expnsiv_sex.value_counts()

print(((70-46)/(70+46))*100)
# create new dataframe containing only the names.

names = pd.DataFrame(df.Name)

names.tail(100)
# count the nan's

names.apply(lambda x: sum(x.isnull()))
# define a function to get the suffixes. 

def get_suffix(name):

    return name.split(',')[1].split('.')[0].split()

# use the function and feed the result into an array.

name_sffx = [get_suffix(names.Name[i]) for i in range(0,len(names))]

suffix_df = pd.DataFrame(name_sffx,columns=['Suffixes','None'])

suffix_list = suffix_df.Suffixes.unique()

freq=suffix_df.Suffixes.value_counts()

# create a new df:

suffix_count = pd.DataFrame({'Suffixes': suffix_list,'Frequency':freq})

# set the index:

suffix_count.set_index('Suffixes')

del suffix_count['Suffixes']

suffix_count.index



#Lets plot this into a barplot

plt.figure(figsize=(15,10))

ax = sns.barplot(x=suffix_count.index,y=freq)



plt.xticks(rotation=90)

plt.xlabel('Suffixes')

plt.ylabel('Frequency')
#Picking the necessary columns and saving it into a new df.

ages = df[['Survived','Age']]

ages.head(10)
ages.shape
ages = ages.dropna()

ages.head(10)
new_index = (ages['Age'].sort_values(ascending=True)).index.values

sorted_age = ages.reindex(new_index)
sorted_age.head(10)
#Let's plot this now on a bargraph. 

plt.figure(figsize=(35,20))

ax = sns.barplot(x=sorted_age['Age'],y=sorted_age['Survived'])

plt.title('Age versus Survival')

plt.xlabel('Age')

plt.ylabel('Survival')