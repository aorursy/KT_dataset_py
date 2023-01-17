import numpy as np

import pandas as pd

import seaborn as sns

%matplotlib inline
suicide_data = pd.read_csv('../input/data.csv')
suicide_data.head()
suicide_data.columns = suicide_data.iloc[0]
suicide_data.head()
suicide_data = suicide_data[1:]
suicide_data.drop([2015.0, 2010.0,2000.0], axis=1,inplace=True)
suicide_data.rename(index=str, columns={2016.0: "Suicide rates per 1Lakh population for 2016"},inplace=True)
suicide_data.head()
suicide_data = suicide_data[(suicide_data['Sex'] == 'Both sexes')]
suicide_data.drop(['Sex'], axis=1,inplace=True)
suicide_data.head()
happiness_data = pd.read_csv('../input/2016.csv')
happiness_data.head()
happiness_data = happiness_data[['Country','Happiness Rank']]
happiness_data.head()
suicide_data.index = suicide_data['Country']

happiness_data.index = happiness_data['Country']
suicide_data.drop(['Country'], axis=1,inplace=True)

happiness_data.drop(['Country'],axis=1,inplace=True)
suicide_data.head()
happiness_data.head()
#inner join

compiled_data = pd.merge(suicide_data, happiness_data, left_index=True, right_index=True) 
compiled_data.head()
compiled_data = compiled_data.sort_values(by=['Happiness Rank'])

compiled_data.head()
sns.scatterplot(x='Happiness Rank',y='Suicide rates per 1Lakh population for 2016',data=compiled_data)
sns.lmplot(x='Happiness Rank',y='Suicide rates per 1Lakh population for 2016',data=compiled_data)
top_40 = compiled_data[0:40]

top_40.head()
sns.lmplot(x='Happiness Rank',y='Suicide rates per 1Lakh population for 2016',data=top_40)

compiled_data.corr()
happiness_index_other_variables = pd.read_csv('../input/2016.csv')
happiness_index_other_variables.head()
happiness_index_other_variables.index = happiness_index_other_variables['Country']

happiness_index_other_variables.drop(['Country'],axis=1,inplace=True)
happiness_index_other_variables.head()
happiness_index_suicide_rankings = pd.merge(suicide_data, happiness_index_other_variables, left_index=True, right_index=True) 
happiness_index_suicide_rankings.head()
happiness_index_suicide_rankings.corr()
sns.lmplot(x='Health (Life Expectancy)',y='Suicide rates per 1Lakh population for 2016',data=happiness_index_suicide_rankings)