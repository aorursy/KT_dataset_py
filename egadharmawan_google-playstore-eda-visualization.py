import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
df.head()
df.info()
# Category

cat = df.Category.unique()
cat
plt.figure(figsize=(12,12))

most_cat = df.Category.value_counts()
sns.barplot(x=most_cat, y=most_cat.index, data=df)
# Rating

df.Rating.unique()
df['Rating'].replace(to_replace=[19.0], value=[1.9],inplace=True)
sns.distplot(df.Rating)
g = sns.FacetGrid(df, col='Category', palette="Set1",  col_wrap=5, height=4)
g = (g.map(sns.distplot, "Rating", hist=False, rug=True, color="r"))
# Mean Rating

plt.figure(figsize=(12,12))

mean_rat = df.groupby(['Category'])['Rating'].mean().sort_values(ascending=False)
sns.barplot(x=mean_rat, y=mean_rat.index, data=df)
# Reviews

df.Reviews.unique()
# inside review there is a value with 3.0M with M stand for million, lets change it so it can be measure as float

Reviews = []

for x in df.Reviews:
    x = x.replace('M','00')
    Reviews.append(x)

Reviews = list(map(float, Reviews))
df['reviews'] = Reviews
sns.distplot(Reviews)
g = sns.FacetGrid(df, col='Category', palette="Set1",  col_wrap=5, height=4)
g = (g.map(plt.hist, "Reviews", color="g"))
# Total reviews

plt.figure(figsize=(12,12))
sum_rew = df.groupby(['Category'])['reviews'].sum().sort_values(ascending=False)
sns.barplot(x=sum_rew, y=sum_rew.index, data=df)
# Mean reviews

plt.figure(figsize=(12,12))
mean_rew = df.groupby(['Category'])['reviews'].mean().sort_values(ascending=False)
sns.barplot(x=mean_rew, y=mean_rew.index, data=df)
# Installs

df.Installs.unique()
df['Installs'].replace(to_replace=['0', 'Free'], value=['0+','0+'],inplace=True)
Installs = []

for x in df.Installs:
    x = x.replace(',', '')
    Installs.append(x[:-1])

Installs = list(map(float, Installs))
df['installs'] = Installs
sns.distplot(Installs)
g = sns.FacetGrid(df, col='Category', palette="Set1",  col_wrap=5, height=4)
g = (g.map(plt.hist, "installs", bins=5, color='c'))
# Total Installs

plt.figure(figsize=(12,12))
sum_inst = df.groupby(['Category'])['installs'].sum().sort_values(ascending=False)
sns.barplot(x=sum_inst, y=sum_inst.index, data=df)
# Mean Install

plt.figure(figsize=(12,12))
mean_ints = df.groupby(['Category'])['installs'].mean().sort_values(ascending=False)
sns.barplot(x=mean_ints, y=mean_ints.index, data=df)
df.Size.unique()
df['Size'].replace(to_replace=['Varies with device'], value=['0'],inplace=True)
# i need to diiscard + and , value. amd change M for million. Then check the distibution.

Size = []

for x in df.Size:
    x = x.replace('+', '')
    x = x.replace(',', '')
    if 'M' in x:
        if '.' in x:
            x = x.replace('.', '')
            x = x.replace('M', '00')
        else:
            x = x.replace('M', '000')
    elif 'k' in x:
        x = x.replace('k', '')
    Size.append(x)

Size = list(map(float, Size))
df['size'] = Size
sns.distplot(Size)
g = sns.FacetGrid(df, col='Category',  col_wrap=5, height=4)
g = (g.map(plt.hist, "size", bins=5, color='y'))
# Mean Size

plt.figure(figsize=(12,12))
mean_size = df.groupby(['Category'])['size'].mean().sort_values(ascending=False)
sns.barplot(x=mean_size, y=mean_size.index, data=df)
# Type for category

df.Type.unique()
df['Type'].replace(to_replace=['0'], value=['Free'],inplace=True)
df['Type'].fillna('Free', inplace=True)
print(df.groupby('Category')['Type'].value_counts())
Type_cat = df.groupby('Category')['Type'].value_counts().unstack().plot.barh(figsize=(10,20), width=0.7)
plt.show()
# And Ver

df['Android Ver'].unique()
df['Android Ver'].replace(to_replace=['4.4W and up','Varies with device'], value=['4.4','1.0'],inplace=True)
df['Android Ver'].replace({k: '1.0' for k in ['1.0','1.0 and up','1.5 and up','1.6 and up']},inplace=True)
df['Android Ver'].replace({k: '2.0' for k in ['2.0 and up','2.0.1 and up','2.1 and up','2.2 and up','2.2 - 7.1.1','2.3 and up','2.3.3 and up']},inplace=True)
df['Android Ver'].replace({k: '3.0' for k in ['3.0 and up','3.1 and up','3.2 and up']},inplace=True)
df['Android Ver'].replace({k: '4.0' for k in ['4.0 and up','4.0.3 and up','4.0.3 - 7.1.1','4.1 and up','4.1 - 7.1.1','4.2 and up','4.3 and up','4.4','4.4 and up']},inplace=True)
df['Android Ver'].replace({k: '5.0' for k in ['5.0 - 6.0','5.0 - 7.1.1','5.0 - 8.0','5.0 and up','5.1 and up']},inplace=True)
df['Android Ver'].replace({k: '6.0' for k in ['6.0 and up']},inplace=True)
df['Android Ver'].replace({k: '7.0' for k in ['7.0 - 7.1.1','7.0 and up','7.1 and up']},inplace=True)
df['Android Ver'].replace({k: '8.0' for k in ['8.0 and up']},inplace=True)
df['Android Ver'].fillna('1.0', inplace=True)
print(df.groupby('Category')['Android Ver'].value_counts())
Type_cat = df.groupby('Category')['Android Ver'].value_counts().unstack().plot.barh(figsize=(10,18), width=1)
plt.show()