import pandas as pd 

import numpy as np

import matplotlib.pylab as plt

import seaborn as sns

import warnings 

warnings.simplefilter('ignore')
apps =pd.read_csv("../input/goggle-play-data/apps.csv",engine='python')

apps.head()
apps.info()
# Cleaning and Converting column in appropriate data type



#Cleaning



#installs column ["+",","] occuring (i.e 10,000+) 

apps['Installs']=apps['Installs'].str.replace("+","")

apps['Installs']=apps['Installs'].str.replace(",","")



apps.rename({"Unnamed: 0":"index"}, axis="columns", inplace=True)



#Price column in $ occur(i.e $4.00)

apps['Price']=apps['Price'].str.replace("$","")



#Converting data type (object to numeric data)

apps['Installs']=apps['Installs'].astype(int)

apps['Price']=apps['Price'].astype(float)
apps.info()
# In This Analysis in Mostly We focus Below

columns=["App","Category","Rating","Size","Installs","Price","Reviews"]

#1.we need Analyze Category wise apps frequency

Unique_Category = apps['Category'].unique()

Unique_Category
Frequency_Category = apps['Category'].value_counts().to_frame()  # Value_Count

Frequency_Category.rename({"Category": "count"}, axis=1,inplace=True)

Frequency_Category


#plot 



plt.figure(figsize=(12, 4))

plt.bar(Frequency_Category.index,Frequency_Category['count'])

plt.xticks(rotation=90)

plt.title("Category vs Count")

plt.show()





#selecting useful colunmn 

less_performance=apps.loc[0:,['App','Rating','Price','Type','Installs','Reviews']].dropna() #here not count None value 



c1=less_performance['Rating']<3

c2=less_performance['Type']=='Paid'

less_performance=less_performance[c1 & c2 ]





#Total Earning individiual app

less_performance["Total_Earning"]=less_performance['Price']*less_performance['Installs']





# per rating public paid Money (i.e Here Below Table You See "I am extremely Rich" app  earn 131031$ per ratings so here

# public Lost big ,public paid big amount this app get only 1 rate it's worst app)



less_performance['per_rating_paid(by people)']=less_performance["Total_Earning"]/less_performance["Rating"]

less_performance.sort_values(by='per_rating_paid(by people)',ascending=False,inplace=True)

less_performance
## ploting 

plt.figure(figsize=(12,5))

plt.suptitle("App vs Paid(by People per rating)")



#plot 1

plt.subplot(131) #subplot give us better Visualization

less_performance1=less_performance[0:4]

plt.bar(less_performance1['App'],less_performance1['per_rating_paid(by people)'])

plt.xticks(rotation=90)



#plot 2

plt.subplot(132)

less_performance2=less_performance[4:12]

plt.bar(less_performance2['App'],less_performance2['per_rating_paid(by people)'])

plt.xticks(rotation=90)



#plot 3

plt.subplot(133)

less_performance3=less_performance[12:]

plt.bar(less_performance3['App'],less_performance3['per_rating_paid(by people)'])

plt.xticks(rotation=90)



plt.show()
%matplotlib inline
plt.figure(figsize=(12,5))

sns.boxplot('Type','Rating',data=apps) 

# sns.boxplot('Type','Rating',data=apps,showfliers=False) without  outliers

# Below plot you can see paid app Median slightly up.mid 50 % paid app get 4.1 to 4.60  rating while free mid 50% app 

#get 4 to 4.5 ratings here outliers not include 
plt.figure(figsize=(13,5))

Rate_Size=apps[apps['Size']<30].dropna() #None Value not Count

sns.jointplot(data=Rate_Size,x='Rating', y='Size',kind='kde')

# sns.boxplot(apps['Rating'],apps['Size'],showfliers=False)

plt.xticks(rotation=90)

plt.title("Rating Vs Size")

plt.show()

# Here Below plot you can see  4 Mb to 10Mb size give us best Performance 4 to 4.5 Rates 