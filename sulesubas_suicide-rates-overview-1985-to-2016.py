# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read data and write first 5 component

dataset = pd.read_csv("../input/suicideratesoverview1985to2016/master.csv")
dataset.head(10) 
dataset.info()
dataset.describe()
dataset.dropna()
dataset[dataset.suicides_no==min(dataset.suicides_no)]
min(dataset.suicides_no)
dataset[dataset.suicides_no==max(dataset.suicides_no)]

max(dataset.suicides_no)
print(dataset['country'].value_counts(dropna =False)) 
dataset.country.unique()
#here i changed column name because this column name has a space between words

dataset.rename(columns={'country-year':'country_year'})
#dataset.country.value_counts().unique() 

#If this code is equivalent to sql code -> 

#select column_name1,column_name2,sum(column_name3) from table_name 

#WHERE condition group by column_name1,column_name2 ORDER BY column_name1

dataset.groupby(['country']).agg('sum')

#You can also use it like this -> dataset.groupby(['country']).sum()
#temp= dataset[(dataset["country"] == 'Argentina')].sum()

suicideCountryGender=dataset.groupby(['country','sex']).suicides_no.sum()

suicideCountryGender
temp= dataset[(dataset["country"] == 'Antigua and Barbuda')]

country_list = list(temp)

print(temp)
temp= dataset[(dataset["country"] == 'Antigua and Barbuda')].dropna()

temp
#if you want to see not null you can use this code

temp= dataset [(dataset["country"] == 'Antigua and Barbuda') & (dataset['suicides_no']!=0)]

temp
#If you want to get numbers of uniques you can use this code.

##If this code is equivalent to sql code -> 

#select DISTINCT column_name from table_name 

country_list = len(list(dataset['country'].unique()))

country_list
temp= dataset[(dataset["country"] == 'Turkey')]

country_list = list(temp)

print(temp)
dataset['age'].unique()
sns.countplot(dataset[dataset['country']=='Turkey'].age)

plt.title("According to Age Number of Suicide For Turkey ")       

plt.xticks(rotation=90)

plt.ylabel('Number of Suicide', fontsize=15)

plt.xlabel('Turkey', fontsize=15)

plt.show()
sns.countplot(dataset[dataset['country']=='Turkey'].sex)

plt.title("According to Age Number of Suicide For Turkey ")       

plt.xticks(rotation=90)

plt.ylabel('Number of Suicide', fontsize=15)

plt.xlabel('Turkey', fontsize=15)

plt.show()
dataset.head(5)
dataset.info()
dataset.isnull().values.any()
dataset.isnull().any()
dataset.isnull().sum()
#dataset=dataset.drop(['HDIForYear'],axis=1)
#how to do piechart by using dataset #dataset[dataset.groupby(['country']).agg('sum')] 

#suicideCountryGender suicideCountryGender=dataset.groupby(['country','sex']).suicides_no.sum()

country_value = dataset.groupby(['country']).suicides_no.sum() #dataset.country.value_counts()

country_value_new = country_value[:10]

country_list = list(dataset['country'].unique())

country_new = country_list[:10]

vals= country_value_new   

labels= country_new

colors = ['bisque','pink','yellow','blue','green','thistle','olive','white','aqua','m'] #to customize the colors

plt.pie(vals,labels=labels,colors = colors, autopct='%0.2f%%',shadow=True) #explode

plt.title('Top 10 Country')

plt.tight_layout()

#plt.axis('equal')

plt.show
#how to sort according to a column 

dataset.sort_values(by=['suicides_no'],ascending=False)  
country_list = list(dataset['country'].unique())

country_list.sort()

print (country_list)
1

#dataset["country"][10]

2

#country_list = list(dataset['country'].head(10).unique())

#country_list

3

#country_list = list(dataset['country'].head(10).unique())

#dataset.sort_values('country', inplace=True, ascending=False)

4

#matcher = re.compile('^A1.8301$')

#list_of_string = [s for s in stringlist if matcher.match(s)]



#dataset.country[10]
country_list = list(dataset['country'].head(1000).unique())

dataset.sort_values('country', ascending=False)
# Plotting a bar graph of the suicides number in top 10 country

# in the column 'country'

country_count  = dataset.groupby(['country']).suicides_no.sum()

country_count = country_count[:10,]

plt.figure(figsize=(15,5))

sns.barplot(country_count.index, country_count.values, alpha=1)

plt.xticks(rotation= 90,fontsize=13)

plt.yticks(rotation= 0,fontsize=13)



plt.title('Suicide Numbers in Each Countries',fontsize=15)

plt.ylabel('Number of Suicide', fontsize=15)

plt.xlabel('Country', fontsize=15)

plt.show() 
#here is that column values of the albania

temp= dataset [(dataset['country']=='Albania') & (dataset['sex']=='male') ]

temp.head(10000)
dataset.year.plot(kind='hist',bins=50, figsize=(12,12), color="salmon")

plt.show()
#here is that column values of the albania

temp= dataset[(dataset['country']=='Albania')]

temp.head(1000)
#if you want to get number for one value, you can use following code. For example in the code below we use filter and 264 rows is listed. You can use this codes to verify results.

temp= dataset[(dataset['country']=='Albania')]

temp1 = len(temp)

temp1
# Plot the crashes where alcohol was involved

sns.set_color_codes("muted")

sns.barplot(x="year", y="suicides_no", data=dataset, label="Year Suicides", color="r")

plt.xticks(rotation=90)

plt.figure(figsize=(30,10))

plt.show()
plt.figure(figsize=(7,7))

g = sns.FacetGrid(dataset, col="sex", hue="year", subplot_kws=dict(projection='polar'), height=5.5, sharex=False, sharey=False, despine=False)

plt.show()
dataset[dataset['country']=='Albania'].hist()

plt.tight_layout()

plt.show()
sns.boxenplot(x="sex", y="suicides_no",

              color="b",

              scale="linear", data=dataset)

plt.tight_layout()

plt.show()
dataset.head()
sns.distplot(dataset[(dataset['sex']=='female')].age.value_counts().values)

plt.show()
sns.set_color_codes()

sns.distplot(dataset['country'].value_counts().values,color='r')

plt.show()