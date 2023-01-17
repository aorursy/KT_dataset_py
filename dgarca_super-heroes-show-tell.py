import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
print ("Hello eBay")
df1=pd.read_csv('../input/heroes_information.csv')
df1
df1.head()
df1.shape
df1.isnull().sum()
df1.replace(to_replace='-',value='Other',inplace=True)
#Replace the nulls in Publisher column
df1['Publisher'].fillna('Other',inplace=True)
df1.isnull().sum()
#Whaaaat I have two superheroes with unknown weight, let's see who they are. Big boys
df1[df1['Weight'].isnull()]
#We don't need Unnamed column, let's get rid of it
df1.drop('Unnamed: 0',axis=1,inplace=True)
df1.head()
df1.groupby(['Gender'])
#As I see nothing I need to add an aggregation formula, I'll start by the Count
df1.groupby(['Gender']).count()
#Hey I'm curious who are the superherous with undefined Gender
df1[df1['Gender']=='Other']
#Let's use only one column
df1.groupby(['Gender']).count()[['name']]
#This is temporal, I want to generate a plot so it's convenient to store in another DF
df1_gender=df1.groupby(['Gender']).count()[['name']]
df1_gender
df1_gender.rename(columns={'name': 'Total'}, inplace=True)
df1_gender
#We can also sort
df1_gender=df1_gender.sort_values('Total',ascending=True)
df1_gender
df1_gender['Total'].plot.pie(title="Superheroes Gender", figsize=(15, 15))
#This is a workaround, super fast charting and grouping but without storing in another DF
#Ok, bars, nobody likes Pie charts
df1.groupby('Gender')['Gender'].count().plot.bar(title="Superheroes Gender", figsize=(15, 15))
df1.groupby(['Gender', 'Alignment']).count()[['name']]
#And let's visualize... now an horizontal bar
df1.groupby(['Gender', 'Alignment']).count()[['name']].plot.barh(title="Superheroes Gender & Alignment", figsize=(15, 15))
#First lets clean a bit, those values with Height or Weight < 0
df1_pub = df1.drop(df1[(df1.Height < 0) | (df1.Weight < 0)].index)
df1_pub['Publisher'].value_counts()
df1_pub = df1_pub.drop(df1_pub[(df1_pub.Publisher == 'Image Comics') | (df1_pub.Publisher == 'Sony Pictures')].index)
df1_pub['Publisher'].value_counts()
df1_pub_agg=df1_pub.groupby(['Publisher'])
df1_pub_agg=df1_pub_agg.agg({'Height': ['mean','median','std'], 'Weight': ['mean','median','std']}).round(2)
df1_pub_agg
#Let's order a bit
df1_pub_agg=df1_pub_agg.sort_values([('Height','mean')],ascending=False)
df1_pub_agg
df1_pub_agg.plot.bar(title="Superheroes Height and Weight", figsize=(15, 15))
#Split randomly
group_a, group_b = train_test_split(df1_pub, test_size=0.50)
group_a.groupby("Publisher")["Publisher"].count()
group_b.groupby("Publisher")["Publisher"].count()
#Split randomly but keeping ratio in a column
group_a, group_b = train_test_split(df1_pub, test_size=0.50, stratify=df1_pub["Publisher"])
group_a.groupby("Publisher")["Publisher"].count()
group_b.groupby("Publisher")["Publisher"].count()