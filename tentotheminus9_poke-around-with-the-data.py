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
import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv('../input/aegypti_albopictus.csv')
x = df.iloc[:,0].tolist() #Get the list of names

a_alb = x.count('Aedes albopictus') #Count how many 

alb_pc = round(a_alb/len(x)*100,2) #Percentage of Aedes albopictus cases



a_aeg = x.count('Aedes aegypti') #Count how many

aeg_pc = round(a_aeg/len(x)*100,2) #Percentage of Aedes aegypti cases



labels = ['Aedes albopictus', 'Aedes aegypti']

cols = ['green', 'lightblue']

sizes = [alb_pc,aeg_pc]



plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, colors = cols) #Pie chart
country_group = df.groupby(['COUNTRY']).count() #Group by country

country_group.iloc[:,1] = country_group.iloc[:,1].astype(float)



country_group_sorted = country_group.sort('VECTOR', ascending = 0)

countries = country_group_sorted.index.tolist()



country_group_sorted_sub = country_group_sorted.iloc[0:50,:] #Pick top 50

countries_sub = countries[0:50] #Pick top 50



p = sns.barplot(x=countries_sub, y=country_group_sorted_sub.iloc[:,0])

p.set(yscale="log")

p.set_xticklabels(p.get_xticklabels(), rotation=90)

p.set(ylabel='Log No. of Cases')
taiwan = df.loc[df['COUNTRY'] == 'Taiwan']

taiwan['YEAR'] = taiwan['YEAR'].astype(float)



#Are there any missing values in the year data?

taiwan['YEAR'].isnull().value_counts()



#Get rid of them...

taiwan_years = taiwan['YEAR'].dropna()



#Plot the year distribution for Taiwan...

taiwan_years = taiwan_years.astype(int)

p = sns.countplot(taiwan_years)

p.set_xticklabels(p.get_xticklabels(), rotation=90)
kenya = df.loc[df['COUNTRY'] == 'Kenya']

kenya['YEAR'] = kenya['YEAR'].astype(float)



#Are there any missing values in the year data?

kenya['YEAR'].isnull().value_counts()



#Get rid of them...

kenya_years = kenya['YEAR'].dropna()



#Plot the year distribution for Taiwan...

kenya_years = kenya_years.astype(int)

p = sns.countplot(kenya_years)

p.set_xticklabels(p.get_xticklabels(), rotation=90)
p = sns.distplot(taiwan_years)