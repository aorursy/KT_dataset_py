import os

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
paths = [f for f in os.listdir('../input') if f.endswith(".csv")]

df = pd.read_csv('../input/'+paths[0],encoding='ISO-8859-1',low_memory=False)
cols = ['eventid','approxdate','addnotes','motive' ,'guncertain3','property','propextent','propextent_txt','propvalue','propcomment','divert','targsubtype1_txt',

        'crit1','crit2','crit3','doubtterr',

        'targsubtype1','target1','guncertain1','weapsubtype1','weapsubtype1_txt','kidhijcountry','ransomamtus','ransompaidus','addnotes','dbsource','scite1','scite2','scite3','INT_LOG','INT_IDEO','INT_MISC','INT_ANY','related']

df.drop(cols,axis=1,inplace=True)

df.dropna(axis=1,how='all',thresh=160000,inplace=True)

df.weaptype1_txt.replace(['Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)'],['Vehicle'],inplace=True)
pd.set_option('display.max_columns', 50)

df.head()
columns = ['nkill','nwound']

df[columns].describe()
plt.subplots(figsize=(16,9))

sns.countplot('iyear',data=df)

plt.xticks(rotation=90)

plt.title("Number of terrorist attacks by year, worldwide")

plt.xlabel("Year")

plt.ylabel("Number of attacks")

plt.show()
plt.subplots(figsize=(12,6))

sns.countplot(x='region_txt',data=df,order=df['region_txt'].value_counts().index)

plt.xticks(rotation=20,ha="right",fontsize=9)



plt.title("Number of terrorist attacks by region")

plt.xlabel("")

plt.ylabel("Number of attacks")

plt.tight_layout()

plt.show()

by_region = pd.crosstab(df.iyear,df.region_txt)

by_region.plot()

fig = plt.gcf()

fig.set_size_inches(16,8)

plt.title("Attacks by region, yearly")

plt.show()
n = 49

plt.subplots(figsize=(22,15))

sns.countplot(y='country_txt',data=df,order=df['country_txt'].value_counts().iloc[:n].index)

plt.xticks(rotation=0,ha="right",fontsize=9)



plt.title("Top-{0} countries by number of terrorist attacks, desc".format(n))

plt.xlabel("Number of attacks")

plt.ylabel("")

plt.show()

by_region2 = pd.crosstab(df.region_txt,df.attacktype1_txt,margins=True)

by_region2 = by_region2.iloc[:-1,:]

by_region2.sort_values('All',ascending=True,inplace=True)

by_region2.iloc[:,:-1].plot.barh(stacked=True,width=1,color=sns.color_palette('tab20',9))



plt.xlabel("number of attacks")

plt.ylabel("")

plt.title("Terrorist attacks by type and region")

fig=plt.gcf()

fig.set_size_inches(12,8)

plt.show()

plt.figure(figsize=(7,5))

plt.title("Suicidal attacks")

sns.violinplot(x = df.suicide, y = df.iyear,data=df)

plt.show()



plt.figure(figsize=(7,5))

plt.title("Attack outcome")

sns.violinplot(x = df.success, y = df.iyear,data=df)



plt.show()





plt.figure(figsize=(7,5))

plt.title("Individual attacks")

sns.violinplot(x = df.individual, y = df.iyear,data=df)





plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,

                    wspace=0.35)

plt.show()
a = sns.FacetGrid( df, hue = 'success', aspect=3)

a.map(sns.kdeplot, 'iyear', shade= True)

a.set_xlabels("Year")

a.set_ylabels("Success rate")

a.set(xlim=(1969 , df['iyear'].max()))

a.add_legend(title="success of an attack",);

df.head(3)
plt.figure(figsize=(20,12))

sns.boxplot(x = 'weaptype1_txt', y = 'iyear', hue = 'success', data = df)

plt.title('Outcome by weapon type')

plt.xticks(rotation=15)

plt.xlabel("Weapon type")

plt.ylabel("Y")

plt.show()
plt.figure(figsize=(20,12))

sns.boxplot(x = 'attacktype1_txt', y = 'iyear', hue = 'success', data = df)

plt.title('Outcome by attack type')

plt.xticks(rotation=15)

plt.xlabel("Type of attack")

plt.ylabel("Y")

plt.show()
temp = df.attacktype1_txt.value_counts()

s = sum(temp[-i] for i in range(1,4))

temp = temp[:-3]

temp = temp.append(pd.Series(s,index=['Other']))

plt.pie(temp.values, labels = temp.keys(),autopct='%1.1f%%', shadow=True, startangle=40)

plt.axis('equal')

plt.title('Terrorist attacks, by type')

plt.show()



temp = df.weaptype1_txt.value_counts()

s = sum(temp[-i] for i in range(1,9))

temp = temp[:-8]

temp = temp.append(pd.Series(s,index=['Melee and other']))

plt.pie(temp.values,labels=temp.keys(),autopct='%1.1f%%',startangle=40)

plt.axis('equal')

plt.title('Terrorist weapons, by type')

plt.show()


