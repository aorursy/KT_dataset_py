import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))
data = pd.read_csv('../input/athlete_events.csv')
regions = pd.read_csv('../input/noc_regions.csv')
data.head()
data.describe()
data.shape
data.columns
data.info()
regions.head()
combined_df = pd.merge(data, regions, on = 'NOC', how = 'left')
combined_df.head()
gold_medalist = combined_df[combined_df['Medal'] == 'Gold']
gold_medalist.head()
gold_medalist.isnull().any()
gold_medalist.shape
gold_medalist = gold_medalist.dropna(subset = ['Age'])
gold_medalist.shape
plt.figure(figsize = (20,5))
plt.title('Distibution of Gold Medalist')
plt.tight_layout()
sns.countplot(x = 'Age', data = gold_medalist)

plt.show()
gold_medalist['Age'][gold_medalist['Age']>50].count()
disciplines = gold_medalist['Sport'][gold_medalist['Age']>50]
plt.figure(figsize = (10,5))
plt.tight_layout()
plt.title('Disciplines')
sns.countplot(disciplines)
plt.show()
combined_df.head(2)
women_athletes = combined_df[(combined_df['Sex'] == 'F') & (combined_df['Season'] == 'Summer')]
women_athletes.head()
plt.figure(figsize = (15,5))
plt.title('Women Athlets')
plt.tight_layout()
sns.countplot(x = 'Year', data = women_athletes)
plt.show()
women_athletes['Sex'][women_athletes['Year'] == 2016].count()
golds = combined_df[(combined_df['Medal'] == 'Gold')]
total_golds = golds['region'].value_counts().reset_index(name = 'Medal')
total_golds.head(10)
top10_country = total_golds.head(10)
sns.catplot(x = 'index',y = 'Medal', data = top10_country,kind = 'bar', height = 8)
plt.title('Medals per Country')
plt.xlabel('Top10 Countries')
plt.ylabel('Number of Medals')
plt.show()
USA_Goldlist = combined_df[(combined_df['Medal'] == 'Gold') & (combined_df['region'] == 'USA')]

USA_Goldlist.head()
sports = USA_Goldlist['Event'].value_counts().reset_index(name = 'Medal')
sports.head(5)

combined_df.head(2)
BasketBall_USA = combined_df[(combined_df['Sport'] == 'Basketball') & (combined_df['Sex'] == 'M') & 
                         (combined_df['region'] == 'USA')].sort_values(['Year'])
BasketBall_USA.head(5)
gold_medalist.head(5)
gold_medalist.info()
NotNullMedals = gold_medalist[(gold_medalist['Height'].notnull()) & (gold_medalist['Weight'].notnull())]
NotNullMedals.count()
plt.figure(figsize = (10,5))
sns.scatterplot(x = 'Height', y = 'Weight', data = NotNullMedals)
plt.title('Height vs Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()
NotNullMedals[['Name','Sport']][NotNullMedals['Weight']>160]
MaleAthletes = combined_df[['Year', 'Sex']][(combined_df['Sex'] == 'M') & (combined_df['Season'] == 'Summer')]
FemaleAthletes = combined_df[['Year','Sex']][(combined_df['Sex'] == 'F') & (combined_df['Season'] == 'Summer')]
v1 = MaleAthletes['Year'].value_counts().reset_index(name = 'Male_Count')
v2 = FemaleAthletes['Year'].value_counts().reset_index(name = 'Female_Count')
plt.figure(figsize = (10,5))
sns.lineplot(x = 'index', y = 'Male_Count', data  = v1)
sns.lineplot(x = 'index', y = 'Female_Count', data  = v2)
plt.title('Male vs Women Contribution')
plt.xlabel('Year')
plt.ylabel('Male vs Female count')
plt.show()
plt.figure(figsize = (20,10))
plt.tight_layout()
sns.boxplot(x = 'Year', y = 'Age' ,data = combined_df[combined_df['Sex']== 'M'])
plt.show()
plt.figure(figsize = (20,10))
plt.tight_layout()
sns.boxplot(x = 'Year', y = 'Age', data = combined_df[combined_df['Sex'] == 'F'])
plt.show()