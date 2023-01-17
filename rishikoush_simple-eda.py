import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')
df.info()
df.isnull().sum()
df.head()
df.Genre.value_counts()
#Let's just visualize the aritists based on the number of records in the data

df['Artist.Name'].value_counts().plot(kind='bar', title='Artist vs No.of Songs', figsize=(14, 8),align='center');
sns.distplot(df['Popularity'],hist=True,rug=True,kde_kws={'shade':True})

plt.show()
#Let's add the mean to the dist plot just to understand better

fig, ax = plt.subplots(figsize=(10,6))

sns.distplot(df['Popularity'],hist=True,kde_kws={'shade':True},ax=ax,color='green')

plt.axvline(df['Popularity'].mean(), color='b',linestyle='dashed', linewidth=1)

_, max_ = plt.ylim()

plt.text(df['Popularity'].mean() + df['Popularity'].mean()/10, 

         max_ - max_/10, 

         'Mean: {:.2f}'.format(df['Popularity'].mean()))

sns.factorplot(data=df,

        x='Energy',

        kind='point',

        col='Genre',

        col_order=['dance pop', 'latin','canadian hip hop','edm','dfw rap'])

plt.show()

plt.clf()

sns.lmplot(data=df,

           x="Popularity",

           y="Liveness",

           col="Artist.Name",

        col_order=['Shawn Mendes', 'Ed Sheeran', 'Marshmello','Billie Eilish'])