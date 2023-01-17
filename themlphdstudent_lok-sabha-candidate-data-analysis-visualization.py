import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



plt.style.use('fivethirtyeight')
%time candidates_2004 = pd.read_csv('../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2004.csv')

%time candidates_2009 = pd.read_csv('../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2009.csv')

%time candidates_2014 = pd.read_csv('../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2014.csv')

%time candidates_2019 = pd.read_csv('../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2019.csv')
candidates_2004['Year'] = 2004

candidates_2009['Year'] = 2009

candidates_2014['Year'] = 2014

candidates_2019['Year'] = 2019
candidates = pd.concat([candidates_2004, candidates_2009, candidates_2014, candidates_2019])
candidates.head(10)
candidates.info()
candidates.describe()
candidates.isnull().sum()
candidates.shape
candidate_age_2004 = candidates_2004[candidates_2004['Age'] > 18]

candidate_age_2009 = candidates_2009[candidates_2009['Age'] > 18]

candidate_age_2014 = candidates_2014[candidates_2014['Age'] > 18]

candidate_age_2019 = candidates_2019[candidates_2019['Age'] > 18]
candidate_age_2004[candidate_age_2004['Age'] == candidate_age_2004['Age'].min()]
candidate_age_2009[candidate_age_2009['Age'] == candidate_age_2009['Age'].min()]
candidate_age_2014[candidate_age_2014['Age'] == candidate_age_2014['Age'].min()]
candidate_age_2019[candidate_age_2019['Age'] == candidate_age_2019['Age'].min()]
plt.figure(figsize=(20,12))

edgecolor=(0,0,0),

sns.countplot(candidate_age_2004['Age'].sort_values(), palette = "Dark2", edgecolor=(0,0,0))

plt.title("2004 Election's Candidate Age Count",fontsize=20)

plt.xlabel('Age')

plt.ylabel('Count')

plt.xticks(fontsize=12,rotation=90)

plt.show()
plt.figure(figsize=(20,12))

edgecolor=(0,0,0),

sns.countplot(candidate_age_2009['Age'].sort_values(), palette = "Dark2", edgecolor=(0,0,0))

plt.title("Candidate Age Count",fontsize=20)

plt.xlabel('Age')

plt.ylabel('Count')

plt.xticks(fontsize=12,rotation=90)

plt.show()
plt.figure(figsize=(20,12))

edgecolor=(0,0,0),

sns.countplot(candidate_age_2014['Age'].sort_values(), palette = "Dark2", edgecolor=(0,0,0))

plt.title("Candidate Age Count",fontsize=20)

plt.xlabel('Age')

plt.ylabel('Count')

plt.xticks(fontsize=12,rotation=90)

plt.show()
plt.figure(figsize=(20,12))

edgecolor=(0,0,0),

sns.countplot(candidate_age_2019['Age'].sort_values(), palette = "Dark2", edgecolor=(0,0,0))

plt.title("Candidate Age Count",fontsize=20)

plt.xlabel('Age')

plt.ylabel('Count')

plt.xticks(fontsize=12,rotation=90)

plt.show()
candidates['Age'].value_counts()
candidates_2004[candidates_2004['Criminal Cases'] == candidates_2004['Criminal Cases'].max()]
candidates_2009[candidates_2009['Criminal Cases'] == candidates_2009['Criminal Cases'].max()]
candidates_2014[candidates_2014['Criminal Cases'] == candidates_2014['Criminal Cases'].max()]
candidates_2019[candidates_2019['Criminal Cases'] == candidates_2019['Criminal Cases'].max()]
candidates.sort_values(['Criminal Cases'], ascending=False).head().style.background_gradient(subset = ['Age', 'Criminal Cases'], cmap = 'YlGn')
criminal_cases_2004 = candidates_2004[['Party', 'Criminal Cases']].groupby('Party').sum('Criminal Cases')

criminal_cases_2009 = candidates_2009[['Party', 'Criminal Cases']].groupby('Party').sum('Criminal Cases')

criminal_cases_2014 = candidates_2014[['Party', 'Criminal Cases']].groupby('Party').sum('Criminal Cases')

criminal_cases_2019 = candidates_2019[['Party', 'Criminal Cases']].groupby('Party').sum('Criminal Cases')
criminal_cases_2004.sort_values(['Criminal Cases'], ascending=False).head(10).style.background_gradient(subset = ['Criminal Cases'], cmap = 'PuBu')
criminal_cases_2009.sort_values(['Criminal Cases'], ascending=False).head(10).style.background_gradient(subset = ['Criminal Cases'], cmap = 'PuBu')
criminal_cases_2014.sort_values(['Criminal Cases'], ascending=False).head(10).style.background_gradient(subset = ['Criminal Cases'], cmap = 'PuBu')
criminal_cases_2019.sort_values(['Criminal Cases'], ascending=False).head(10).style.background_gradient(subset = ['Criminal Cases'], cmap = 'PuBu')
loc = candidates["Party"].value_counts()

sns.set(style="whitegrid")

sns.barplot(y=loc[:10], x=loc[:10].index, palette="Set2")

plt.xticks(rotation=90)

plt.xlabel('Party')

plt.ylabel('Candidate Count')

plt.title("Different Political Party and Candidate Count", fontweight="bold")
plt.figure(figsize=(9,10))

ax = sns.barplot(x=loc[:10], y=loc[:10].index,

                 palette="tab20c",

                 linewidth = 1)

for i,j in enumerate(loc[:10]):

    ax.text(.5, i, j, weight="bold", color = 'black', fontsize = 13)

plt.title("Candidate count of each party since 2004")

ax.set_xlabel(xlabel = 'Candidate Count', fontsize = 10)

ax.set_ylabel(ylabel = 'Party', fontsize = 10)

plt.show()
plt.figure(figsize=(20,12))

edgecolor=(0,0,0),

sns.countplot(candidates['Education'].sort_values(), palette = "Dark2", edgecolor=(0,0,0))

plt.title("Candidate Education Count",fontsize=20)

plt.xlabel('Education')

plt.ylabel('Count')

plt.xticks(fontsize=12,rotation=90)

plt.show()
candidate_education = candidates['Education'].value_counts()
candidate_education
plt.figure(figsize=(20,12))

wedge_dict = {

    'edgecolor': 'black',

    'linewidth': 2        

}



explode = (0, 0.1, 0, 0.1, 0, 0.1, 0, 0.1, 0, 0, 0, 0)



plt.pie(candidate_education, explode=explode, autopct='%1.2f%%', wedgeprops=wedge_dict, labels = candidate_education.index)

plt.show()
city_criminal_cases_2004 = candidates_2004[['Constituency', 'Criminal Cases']].groupby('Constituency').sum('Criminal Cases')

city_criminal_cases_2009 = candidates_2009[['Constituency', 'Criminal Cases']].groupby('Constituency').sum('Criminal Cases')

city_criminal_cases_2014 = candidates_2014[['Constituency', 'Criminal Cases']].groupby('Constituency').sum('Criminal Cases')

city_criminal_cases_2019 = candidates_2019[['Constituency', 'Criminal Cases']].groupby('Constituency').sum('Criminal Cases')
city_criminal_cases_2004.sort_values(['Criminal Cases'], ascending=False).head(10).style.background_gradient(subset = ['Criminal Cases'], cmap = 'PuBu')
cases_2004 = city_criminal_cases_2004.sort_values(['Criminal Cases'], ascending=False)['Criminal Cases'][:10]



sns.barplot(y=cases_2004.values, x=cases_2004.index, palette="Set2")

plt.xticks(rotation=90)

plt.xlabel('Constituency')

plt.ylabel('Criminal Cases')

plt.title("Different Constituency and Criminal cases count in 2004 Elections", fontweight="bold")
city_criminal_cases_2009.sort_values(['Criminal Cases'], ascending=False).head(10).style.background_gradient(subset = ['Criminal Cases'], cmap = 'PuBu')
cases_2009 = city_criminal_cases_2009.sort_values(['Criminal Cases'], ascending=False)['Criminal Cases'][:10]

sns.barplot(y=cases_2009.values, x=cases_2009.index, palette="Set2")

plt.xticks(rotation=90)

plt.xlabel('Constituency')

plt.ylabel('Criminal Cases')

plt.title("Different Constituency and Criminal cases count in 2009 Elections", fontweight="bold")
city_criminal_cases_2014.sort_values(['Criminal Cases'], ascending=False).head(10).style.background_gradient(subset = ['Criminal Cases'], cmap = 'PuBu')
cases_2014 = city_criminal_cases_2014.sort_values(['Criminal Cases'], ascending=False)['Criminal Cases'][:10]

sns.barplot(y=cases_2014.values, x=cases_2014.index, palette="Set2")

plt.xticks(rotation=90)

plt.xlabel('Constituency')

plt.ylabel('Criminal Cases')

plt.title("Different Constituency and Criminal cases count in 2014 Elections", fontweight="bold")
city_criminal_cases_2019.sort_values(['Criminal Cases'], ascending=False).head(10).style.background_gradient(subset = ['Criminal Cases'], cmap = 'PuBu')
cases_2019 = city_criminal_cases_2019.sort_values(['Criminal Cases'], ascending=False)['Criminal Cases'][:10]

sns.barplot(y=cases_2019.values, x=cases_2019.index, palette="Set2")

plt.xticks(rotation=90)

plt.xlabel('Constituency')

plt.ylabel('Criminal Cases')

plt.title("Different Constituency and Criminal cases count in 2019 Elections", fontweight="bold")