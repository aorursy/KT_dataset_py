import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud,STOPWORDS
df = pd.read_csv("../input/suicides-in-india/Suicides in India 2001-2012.csv")
df.shape
df.info()
df.isnull().sum()
df.duplicated().sum()
df.describe(include='object')
df.head()
df.groupby(by='Gender')['Total'].sum().plot(kind='pie', explode=[0,0.05], autopct='%1.1f%%',startangle=-45, shadow=True)
temp = df[df['Year']==2012]['Total'].sum() / (365*24*24)

print('Suicide per minute in 2012 : '+str(round(temp)))

del temp
temp = df[~((df['State']=='Total (All India)') | (df['State']=='Total (States)'))]

plt.figure(figsize=(15,5))

sns.barplot(data = temp, x='State', y='Total', hue='Gender', estimator=np.sum)

plt.xticks(rotation=90)

plt.title('Death by State')

del temp
g = sns.FacetGrid(data=df, col='State', col_wrap=5, sharey=False)

g.map(sns.lineplot, 'Year', 'Total', 'Gender',  estimator='sum')

del g
temp = df[ ~(df['Age_group'] == '0-100+')]

plt.figure(figsize=(15,5))

sns.barplot(data=temp, x='Age_group', y='Total', hue='Gender', estimator=np.sum)

del temp
df[~(df['Age_group']=='0-100+')].groupby(by='Age_group')['Total'].sum().plot(kind='pie', autopct='%1.1f%%', shadow=True

                                                                            , explode=[0.1,0.1,0.05,0.08,0.08])
cause_df = df[df['Type_code']=='Causes']



eduction_status_df = df[df['Type_code']=='Education_Status']

means_adopted_df = df[df['Type_code']=='Means_adopted']

professional_profile_df = df[df['Type_code']=='Professional_Profile']

social_status_df = df[df['Type_code']=='Social_Status'] 
plt.figure(figsize=(15,5))

temp = cause_df.groupby(by='Type')['Total'].sum().sort_values(ascending=False).head(10)

sns.barplot(temp.index, temp.values)

plt.xticks(rotation=90)

plt.title('Top Causes of Suicides')
temp = cause_df.groupby(by=['Age_group','Type'])['Total'].sum().unstack().T

mode_temp = temp.max()



columns = temp.columns.to_list()



dic = {}

for i in range(len(temp.columns)):

    dic[columns[i]] =  temp.loc[temp.iloc[:,i]== mode_temp.iloc[i]].index[0]



age_cause = pd.DataFrame(pd.Series(dic), columns=['Cause'])



del temp, mode_temp, columns, dic
age_cause
plt.figure(figsize=(10,4))

temp = eduction_status_df.groupby(by='Type')['Total'].sum().sort_values()

sns.barplot(temp.index, temp.values)

plt.xticks(rotation=90)

plt.title('Suicides by Education Status')

del temp
plt.figure(figsize=(10,4))

sns.barplot(data=eduction_status_df, x='Type', y='Total', estimator=np.sum, hue='Gender')

plt.xticks(rotation=90)
plt.figure(figsize=(10,4))

temp = means_adopted_df.groupby(by='Type')['Total'].sum().sort_values()

sns.barplot(temp.index, temp.values)

plt.xticks(rotation=90)

plt.title('Means of suicide')

del temp
plt.figure(figsize=(10,4))

temp = professional_profile_df.groupby(by='Type')['Total'].sum().sort_values()

sns.barplot(temp.index, temp.values)

plt.xticks(rotation=90)

plt.title('Professional Status in suicides')

del temp
plt.figure(figsize=(10,4))

sns.barplot(data=professional_profile_df, x='Type', y='Total', hue='Gender', estimator=np.sum)

plt.xticks(rotation=90)

plt.title('Distribution of male and female suicides amongst Different professions')
plt.figure(figsize=(10,4))

temp = social_status_df.groupby(by='Type')['Total'].sum().sort_values()

sns.barplot(temp.index, temp.values)

plt.xticks(rotation=90)

plt.title('social status in suicide')

del temp
plt.figure(figsize=(8,4))

sns.barplot(data=social_status_df, x='Type', y='Total', hue='Gender', estimator=np.sum)

plt.xticks(rotation=90)

plt.title('Distribution of male and female suicides amongst various social status')