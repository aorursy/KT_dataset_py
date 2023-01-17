

import numpy as np # linear algebra

import seaborn as sns 

sns.set(color_codes=True)

import matplotlib.pyplot as plt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#read dataset



data=pd.read_csv ('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
#To see column name and dataset



data.head(15) 
#to check for a gap, line, or data type mismatch in the dataset



data.info () 
#While analyzing, columns should be checked individually. UNIQE->like DISTINCT in SQL. 

#What I want to see here is how many kinds of cases are. 

#It may be useful to use unique columns for diversity.



data.case.unique()
#To see how many of each case. 

#value_counts() gives count by group. 

#We do this by using aggregation functions and groupby together in SQL.



data.case.value_counts()









#I wrote to learn the following codes. My aim was to see just for count of one case. 

#Please ignore.





#output_lambda = data.apply (lambda x: [x.value_counts().to_dict()])

#print (output_lambda)

#output_lambda = data.apply (lambda x: [x.value_counts([1])])

#print (output_lambda)

#data.case.value_counts() if data.case==1 

#if data["case"]==1 return data.case.count

data.cc3.unique()
data.cc3.value_counts()

data.country.unique()
data.country.value_counts()
data.banking_crisis.value_counts()
data.year.value_counts()
len(data.year.value_counts())
temp= data [(data["case"] == 1) & (data["cc3"] == 'DZA') & (data['banking_crisis']=='crisis')]

temp.head(5)

#if you want to write in SQL the above code, you shuld write as follows.

#select top 5 * from african_crises ac where ac.case=1 and ac.cc3='DZA' and ac.banking_crisis='crisis'

temp= data [(data["systemic_crisis"] == 0) & (data['banking_crisis']=='crisis')]

temp.head(50)



count_list_=[]

count_list = len(data.country)

#count_list_.append(count_list)

count_list
country_list = list(data['country'].unique())

country_list


from itertools import groupby

x=[9,2,2,2,2,3,4,4,55,55,6,2,2,2,7,0,0]

out = [len([*group]) for i, group in groupby(x)]



out



from itertools import groupby

x=data.country

out = [len([*group]) for i, group in groupby(x)]

out
# Plotting a bar graph of the crisis number in each country

# in the column 'country'

country_count  = data['country'].value_counts()

country_count = country_count[:15,]

plt.figure(figsize=(25,10))

sns.barplot(country_count.index, country_count.values, alpha=1)

plt.xticks(rotation= 30,fontsize=13)

plt.yticks(rotation= 0,fontsize=13)

plt.title('Crisis Numbers in Each Countries',fontsize=20)

plt.ylabel('Number of Crisis', fontsize=20)

plt.xlabel('Country', fontsize=20)

plt.show() 

 

data.head(5)
sns.relplot(x="country",y="year",data=data,height=20)
sns.relplot(x="country", y="year",data=data,height=20, alpha=0.25, edgecolor=None)
# Add "clarity" variable as color

sns.relplot(

            x="country",

            y="year",

            hue="cc3", # added to color axis

            data=data,

            height=20,

            palette="Set1", # change color palette 

            edgeColor=None)
data.head(5)
sns.relplot(

            x="country",

            y="year",

            hue="year",

            size="cc3",   ###

            style="banking_crisis",  ###

            data=data,

            palette="CMRmap_r",

            edgecolor=None,

            height=20)
sns.relplot(

            x="country",

            y="year",

            size="cc3",

            style="systemic_crisis",

            markers=True,

            kind="line",

            data=data,

            hue="banking_crisis",

            height=20

            )



plt.savefig("graph2.png")
sns.relplot(x="country",

            y="cc3",

            col="case", # show region in columns

            data=data,

            height=20)
data.info()
sns.jointplot(x="exch_usd", y="inflation_annual_cpi", data=data);
sns.boxplot(y="case", data=data);
data.head()
data = data.drop(['domestic_debt_in_default'],1)
sns.boxplot(data=data);
data = pd.melt(data, id_vars=["cc3", "country", "year"], var_name="banking_crisis")

data.head()
sns.swarmplot(x="cc3", y="year", data=data, hue="country")