import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set(color_codes=True)  # visualization tool





from sklearn.linear_model import LinearRegression



import warnings

warnings.filterwarnings("ignore")

df=pd.read_csv("../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv",engine="python")
df.head()
df.drop(["PassengerId"],axis=1,inplace=True)
df.head()
df["name"]=df.Firstname+" "+df.Lastname
df.drop(["Firstname","Lastname"],axis=1,inplace=True)
df.head()
df=df.rename(columns={"Country": "country", "Sex": "sex","Age":"age","Category":"category","Survived":"survived"})
df.head()
df.sex=[1 if each =="F" else 0 for each in df.sex]
df.category.unique()
df.category=[1 if each =="P" else 0 for each in df.category]
df.head()
df.describe()
df.info()
df.isna().sum()
df.dtypes
df.corr()
#visualize the correlation

plt.figure(figsize=(15,10))

sns.heatmap(df.iloc[:,0:15].corr(), annot=True,fmt=".0%")

plt.show()
# histogram subplot with non cumulative and cumulative 

fig,axes=plt.subplots(nrows=2,ncols=1)



df.plot(kind='hist',y='age',bins=50,range=(0,100),density=True,ax=axes[0])

df.plot(kind='hist',y='age',bins=50,range=(0,100),density=True,ax=axes[1],cumulative=True)

plt.show()
print(df['sex'].value_counts(dropna=False))
sns.barplot(x='sex',y='age',data=df)

plt.show()
sns.jointplot(x=df.age, y=df.sex, data=df, kind="kde");
sns.swarmplot(x = 'sex', y = 'age', data = df)

plt.show()
df['age']=df['age']

bins=[0,25,50,75,90]

labels=["Young Adult","Early Adult","Adult","Senior"]

df['age_group']=pd.cut(df['age'],bins,labels=labels)

fig=plt.figure(figsize=(20,5))

sns.barplot(x='age_group',y='sex',data=df)

plt.show()
fig=plt.figure(figsize=(20,5))

sns.violinplot(x ='age_group', y = 'sex', data = df)

plt.show()


fig=plt.figure(figsize=(20,5))

sns.violinplot(x ='age_group', y = 'category', data = df)

plt.show()
fig=plt.figure(figsize=(20,5))

sns.violinplot(x = 'age_group', y = 'survived', data = df)

plt.show()
df.columns
grp =df.groupby("age")

x= grp["sex"].agg(np.mean)

y=grp["survived"].agg(np.mean)

z=grp["category"].agg(np.mean)
plt.figure(figsize=(16,5))

plt.plot(x,'ro',color='r')

plt.xticks(rotation=90)

plt.title("Age wise Sex")

plt.xlabel("Age")

plt.ylabel("Sex")

plt.show()
plt.figure(figsize=(16,5))

plt.plot(y,'r--',color='b')

plt.xticks(rotation=90)

plt.title("Age wise Survived")

plt.xlabel("Age")

plt.ylabel("Survived")

plt.show()
plt.figure(figsize=(16,5))

plt.plot(z,"g^",color='g')

plt.xticks(rotation=90)

plt.xlabel("Age")

plt.ylabel("Category")

plt.show()
fig=plt.figure(figsize=(20,5))

sns.violinplot(x ='age', y = 'sex', data = df)

plt.show()
ax = df.sex.plot.kde()

ax = df.survived.plot.kde()

ax = df.category.plot.kde()

ax.legend()

plt.show()
print(df['category'].value_counts(dropna=False))
fig=plt.figure(figsize=(20,5))

sns.swarmplot(x = 'category', y = 'age', data = df)

plt.show()
fig=plt.figure(figsize=(10,5))

sns.swarmplot(x="sex", y="age",hue="survived", data=df)

plt.show()
fig=plt.figure(figsize=(20,5))

sns.violinplot(x = 'sex', 

               y = 'age', 

               data = df, 

               inner = None, 

               )



sns.swarmplot(x = 'sex', 

              y = 'age', 

              data = df, 

              color = 'k', 

              alpha = 0.7)



plt.title('sex by age')

plt.show()
#barplot

sns.barplot(x='category',y='age',data=df)

plt.show()
#boxplot

df.boxplot(column='age', by='category')

plt.show()
#boxplot

df.boxplot(column='age', by='survived')

plt.show()
#boxplot

df.boxplot(column='age', by='sex')

plt.show()
df.category.dropna(inplace = True)

labels = df.category.value_counts().index

colors = ['green','red']

explode = [0,0]

sizes = df.category.value_counts().values



# visual cp

plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Target People According to category Type',color = 'blue',fontsize = 15)
df.sex.dropna(inplace = True)

labels = df.category.value_counts().index

colors = ['pink','blue']

explode = [0,0]

sizes = df.sex.value_counts().values



# visual cp

plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Target People According to sex Type',color = 'blue',fontsize = 15)
df.survived.dropna(inplace = True)

labels = df.survived.value_counts().index

colors = ['red','green']

explode = [0,0.1]

sizes = df.survived.value_counts().values



# visual

plt.figure(0,figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('survived',color = 'blue',fontsize = 15)
plt.figure(figsize=(20,7))

sns.barplot(x=df["country"].value_counts().index,

y=df["country"].value_counts().values)

plt.title("country other rate")

plt.ylabel("rates")

plt.legend(loc=0)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(22,7))

sns.barplot(x = "country", y = "category", hue = "survived", data = df)

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(15,7))

ax = sns.pointplot(x="age", y="category", hue="sex",data=df)

plt.xticks(rotation=75)

plt.show()
labels=df['age_group'].value_counts().index

colors=['blue','red','yellow','green']

explode=[0,0,0.1,0,]

values=df['age_group'].value_counts().values



#visualization

plt.figure(figsize=(7,7))

plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('age_group According Analysis',color='black',fontsize=10)

plt.show()
df.groupby('age_group')['category'].value_counts()
# Data to plot

labels = 'Young Adult', 'Early Adult', 'Adult', 'Senior'

sizes = df.groupby('age_group')['category'].mean().values

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0.1, 0, 0, 0)  # explode 1st slice

 

# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('category for Every age Mean')

plt.axis('equal')

plt.show()
df.groupby('age_group')['survived'].value_counts()
df.groupby('age_group')['sex'].value_counts()