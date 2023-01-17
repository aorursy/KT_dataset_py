

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('/kaggle/input/world-university-rankings/cwurData.csv')

df.head()
df.shape
df.info()
count=df['country'].nunique()

unique_countries=[]

for i in df['country']:

    if i not in unique_countries:

        unique_countries.append(i)

print("Total number of unique countries are {}".format(count))       

print("*"*116)

print("Unique_countries are::{}".format(unique_countries))      
df.drop(['broad_impact'],axis=1,inplace=True)
df.describe()
df.nunique()
Top10=df[['year','institution','world_rank']].groupby('year').head(10)



plt.figure(figsize=(20,8))

ax=sns.pointplot(data=Top10, x="year", y="world_rank",hue="institution",marker='o')

ax.grid(True)

plt.title('Changes in Top 10 University Ranking across years',fontsize=20,fontweight='bold')

plt.xlabel('Year',fontsize=20)

plt.ylabel('World Rank',fontsize=14)

plt.legend(loc='upper right', bbox_to_anchor=(0.5, -0.07), ncol=2)
Top10 = df.groupby('institution')['quality_of_education'].mean().nsmallest(10)

#plt.subplots(figsize=(20,5))

g=sns.barplot(Top10.index, Top10.values,orient='v',palette='coolwarm')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

Top10Employment = df.groupby('institution')['alumni_employment'].mean().nsmallest(10)

e=sns.barplot(Top10Employment.index, Top10Employment.values,orient='v',palette='summer')

e.set_xticklabels(g.get_xticklabels(), rotation=90)
correlation=df.corr()

f,ax = plt.subplots(figsize=(16,16))

sns.heatmap(correlation,annot=True,linewidths=5,ax=ax)

plt.show()
correlation["world_rank"].sort_values(ascending=False)
sns.regplot('world_rank','publications',data=df,color='red')
year_2012=df[df['year']==2012]

year_2012
year_2012.shape
scores=year_2012.groupby(['institution','country'])['world_rank'].first().sort_values().head(10)

g=sns.barplot(scores.index,scores.values,orient='v',palette='coolwarm')

g.set_xticklabels(g.get_xticklabels(), rotation=90)

year_2012_USA=year_2012[year_2012['country']=='USA']

year_2012_USA
USA_2012_scores=year_2012_USA.groupby(['institution'])['world_rank'].first().sort_values().head(10)

g=sns.barplot(USA_2012_scores.index,USA_2012_scores.values,orient='v',palette='coolwarm')

g.set_xticklabels(g.get_xticklabels(), rotation=90)
