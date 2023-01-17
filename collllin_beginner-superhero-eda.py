

import math

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



powers = pd.read_csv('../input/superhero-set/super_hero_powers.csv')

heroes = pd.read_csv('../input/superhero-set/heroes_information.csv')
heroes.head()
powers.head()
heroes= heroes.drop('Unnamed: 0', axis=1)

heroes.info()
heroes['Alignment'].unique()
heroes['Publisher'].unique()

heroes.loc[(heroes['Publisher'] != 'Marvel Comics') & (heroes['Publisher'] != 'DC Comics'),'Publisher'] = 'Other'
heroes['name'].unique()

heroes['name'].isna().value_counts()
heroes.loc[heroes['name'] == '-']
heroes['Gender'].unique()
heroes['Eye color'].unique()
heroes['Race'].unique()
heroes['Skin color'].unique()
heroes['Skin color'].isnull().value_counts()
heroes['Height'].value_counts()
heroes['Weight'].value_counts()
heroes['Hair color'].unique()
heroes.loc[heroes['Gender']=='-']
heroes.loc[heroes['Gender'] == '-','Gender'] = 'Unknown'

heroes.loc[heroes['Eye color'] == '-','Eye color'] = 'Unknown'

heroes.loc[heroes['Hair color'] == '-','Hair color'] = 'Unknown'

heroes.loc[heroes['Hair color'] == 'Brownn','Hair color'] = 'Brown'

heroes.loc[heroes['Hair color'] == 'black','Hair color'] = 'Black'

heroes.loc[heroes['Skin color'] == '-','Skin color'] = 'Unknown'

heroes.loc[heroes['Alignment'] == '-','Alignment'] = 'Unknown'

heroes.loc[heroes['Race'] == '-','Race'] = 'Unknown'

heroes.loc[(heroes['Publisher'] == '-') | (heroes['Publisher'].isna() == True),'Publisher'] = 'Unknown'

heroes.loc[heroes['Height'] < 0,'Height'] = 'Unknown'

heroes.loc[heroes['Weight'] < 0,'Weight'] = 'Unknown'

heroes = heroes.rename(columns={'name': 'hero_names'})
heroes.loc[heroes['Gender'] == 'Unknown']
combined = pd.merge(heroes,powers)
combined.head()
sns.set_palette("pastel")
Publisher_df=combined[['Publisher','Gender']]

plt.figure(figsize=(8,4))

sns.countplot(x='Publisher',data=Publisher_df,hue='Gender')

plt.title('Publishers and the Amount of Superheroes', fontsize=14)



def roundup(x):

    return 50 + int(math.ceil(x / 100.0)) * 100 



total =float(len(Publisher_df))

ax = plt.gca()

y_max = combined['Publisher'].value_counts().max() 

ax.set_ylim([0, roundup(y_max)])



for patch in ax.patches:

    ax.text(patch.get_x() + patch.get_width()/2., patch.get_height(), '{:.0%}'.format(patch.get_height()/total), 

            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
plt.figure(figsize=(8,4))

sns.countplot(x='Publisher',data=combined,hue='Alignment')

total =float(len(combined))

plt.title('Publishers and the Alignment of Their Superheroes', fontsize=14)





ax = plt.gca()

y_max = combined['Publisher'].value_counts().max() 

ax.set_ylim([0, roundup(y_max)])



for patch in ax.patches:

    ax.text(patch.get_x() + patch.get_width()/2., patch.get_height(), '{:.0%}'.format(patch.get_height()/total), 

            fontsize=12, color='black', ha='center', va='bottom')

plt.show()
df = combined.drop(combined[combined.Race == 'Unknown'].index) # Keep





Race_df = df['Race'].value_counts().sort_values(ascending=False).head(10)



label = Race_df.index

value = Race_df.values



plt.figure(figsize=(15,4))

sns.barplot(x=label,y=value)

plt.xlabel('Races')

plt.ylabel('Count')

plt.title("Top 10 Superhero Races")



ax = plt.gca()

y_max = combined['Publisher'].value_counts().max() 

ax.set_ylim([0, roundup(y_max)])



for patch in ax.patches:

    ax.text(patch.get_x() + patch.get_width()/2., patch.get_height(), '{:.0%}'.format(patch.get_height()/total), 

            fontsize=12, color='black', ha='center', va='bottom')



plt.show()
df = combined.drop(combined[combined['Eye color'] == 'Unknown'].index) 

Eyecolor_df = df['Eye color'].value_counts().sort_values(ascending=False).head(10)



label = Eyecolor_df.index

value = Eyecolor_df.values



plt.figure(figsize=(15,4))

sns.barplot(x=label,y=value)

plt.xlabel('Eye colors')

plt.ylabel('Count')

plt.title("Top 10 Superhero Eye Colors")



ax = plt.gca()

y_max = combined['Publisher'].value_counts().max() 

ax.set_ylim([0, roundup(y_max)])



for patch in ax.patches:

    ax.text(patch.get_x() + patch.get_width()/2., patch.get_height(), '{:.0%}'.format(patch.get_height()/total), 

            fontsize=12, color='black', ha='center', va='bottom')



plt.show()
df = combined.drop(combined.columns[10:],axis=1)

df['Height'].unique()
Height_df=df.drop(df[df['Height']=='Unknown'].index)



Height_M=Height_df.drop(Height_df[Height_df['Gender']!= 'Male'].index)

Height_F=Height_df.drop(Height_df[Height_df['Gender']!= 'Female'].index)

Height_U=Height_df.drop(Height_df[Height_df['Gender']!= 'Unknown'].index)
fig=plt.figure(figsize=(14,8))

fig.add_subplot(1,3,1)

plt.ylim(100,300)

sns.boxplot(x='Gender',y='Height',data=Height_M, width =0.75, color='red')

fig.add_subplot(1,3,2)

plt.ylim(100,375)

plt.title('Superheroes and Height Distributions')

sns.boxplot(x='Gender',y='Height',data=Height_F, width =0.75)

fig.add_subplot(1,3,3)

plt.ylim(100,250)

sns.boxplot(x='Gender',y='Height',data=Height_U, width =0.75, color ='green')

plt.show()
Weight_df=df.drop(df[df['Weight']=='Unknown'].index)

Weight_df=Weight_df.dropna()



Weight_M=Weight_df.drop(Weight_df[Weight_df['Gender']!= 'Male'].index)

Weight_F=Weight_df.drop(Weight_df[Weight_df['Gender']!= 'Female'].index)

Weight_U=Weight_df.drop(Weight_df[Weight_df['Gender']!= 'Unknown'].index)
fig=plt.figure(figsize=(14,8))

fig.add_subplot(1,3,1)



sns.boxplot(x='Gender',y='Weight',data=Weight_M, width =0.75,color='red')

fig.add_subplot(1,3,2)



plt.title('Superheroes and Weight Distributions')

sns.boxplot(x='Gender',y='Weight',data=Weight_F, width =0.75)

fig.add_subplot(1,3,3)



sns.boxplot(x='Gender',y='Weight',data=Weight_U, width =0.75, color='green')

plt.show()
hero_powers=combined*1

hero_powers.loc[:, '# of powers'] = hero_powers.iloc[:, 1:].sum(axis=1)

hero_powers

df=hero_powers[['hero_names','# of powers']].sort_values('# of powers',ascending=False)

fig=plt.figure(figsize=(10,5))

plt.title('Superheroes With the Most Amount of Powers')

fig.add_subplot(1,1,1)

sns.barplot(x='hero_names',y='# of powers',data=df.head(15),palette="viridis")

plt.xticks(rotation=60)





plt.show()
df
hp=hero_powers[['hero_names','# of powers','Gender']].sort_values('# of powers',ascending=False)

hp_M=hp.drop(hp[hp['Gender'] != 'Male'].index)

hp_U=hp.drop(hp[hp['Gender'] != 'Unknown'].index)

hp_F=hp.drop(hp[hp['Gender'] != 'Female'].index)
fig=plt.figure(figsize=(20,5))

fig.add_subplot(1,3,1)

sns.barplot(x='hero_names',y='# of powers',data=hp_M.head(15),palette="plasma_r",hue='Gender')

plt.xticks(rotation=80)



fig.add_subplot(1,3,2)

plt.title("Superoheroes With the Most Amount of Powers Related to Genders")

sns.barplot(x='hero_names',y='# of powers',data=hp_F.head(15),palette="Blues",hue='Gender')

plt.xticks(rotation=80)



fig.add_subplot(1,3,3)

sns.barplot(x='hero_names',y='# of powers',data=hp_U.head(15),palette="viridis",hue='Gender')

plt.xticks(rotation=80)

plt.show()



hero_powers=hero_powers.drop(hero_powers.columns[0:10],axis=1)

hero_powers
hero_powers=hero_powers.drop('# of powers', axis =1)

hero_powers_count =pd.DataFrame()



for i in hero_powers.columns:

    hero_powers_count[i] = hero_powers[i].value_counts()
hero_powers_count

hero_powers_count=hero_powers_count.drop([0])

hero_powers_count=hero_powers_count.T

hero_powers_count=hero_powers_count.reset_index()

hero_powers_count.rename(columns={'index': 'Hero Power',1:'Count'}, inplace=True)
hero_powers_count=hero_powers_count.sort_values('Count',ascending=False)



plt.figure(figsize=(20,10))

plt.xticks(rotation=50)

sns.barplot(x='Hero Power',y='Count', data=hero_powers_count.head(15))



total =float(len(combined))

ax = plt.gca()

y_max = combined['Publisher'].value_counts().max() 

ax.set_ylim([0, roundup(y_max)])



for patch in ax.patches:

    ax.text(patch.get_x() + patch.get_width()/2., patch.get_height(), '{:.0%}'.format(patch.get_height()/total), 

            fontsize=12, color='black', ha='center', va='bottom')



plt.title('Most Popular Hero Powers and the Proportion of Heroes That Have Them')

plt.show()