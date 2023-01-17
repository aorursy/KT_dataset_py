import pandas as pd
# data.groupby('name').name.count()
# data.groupby('name').age.min()
data = pd.read_csv("../input/stack-overflow-developer-survey-2020/developer_survey_2020/survey_results_public.csv", index_col="Respondent")
data.median()
data.head(15)
data.Country.value_counts()
data.Country.value_counts(normalize=True)
country_groups = data.groupby(['Country'])
filter1 = data['Country'] == 'United States'
data.loc[filter1]
data.loc[filter1]['CompFreq'].value_counts(normalize=True)
country_groups['CompFreq'].value_counts(normalize=True).loc['India']
country_groups['ConvertedComp'].median().loc['United States']
country_groups['ConvertedComp'].agg(['median','mean']).loc['United States']
filter2 = data['Country'] == 'United States'
data.loc[filter2]['LanguageWorkedWith'].str.contains('Python').sum()
country_groups['LanguageWorkedWith'].apply(lambda x: x.str.contains('Python').sum()).loc['United States']
country_respondents = data['Country'].value_counts()

country_respondents
country_uses_python = country_groups['LanguageWorkedWith'].apply(lambda x: x.str.contains('Python').sum())

country_uses_python
df = pd.concat([country_respondents, country_uses_python], axis='columns')

df
df.rename(columns={'Country':'NumRespondents','LanguageWorkedWith':'NumKnowsPython'}, inplace=True)

df
df['PercentKnowsPython'] = (df['NumKnowsPython']/df['NumRespondents'])*100

df
df.sort_values(by='PercentKnowsPython',ascending=False)
df.head(50)
df.loc['Japan']
data.groupby(['Country', 'CompFreq']).Age.agg(['mean', 'median'])
data.sort_values(by=['Country', 'Age'])
data.CompFreq.dtype
data.dtypes
data.Age.astype('object')
data[pd.isnull(data.Age)]
data.Age.fillna("Unknown")
data
data.Country.replace("United States", "US")
data.Age.fillna(data.Age.mean()).sort_values(ascending=False)
data.rename(columns={"Age1stCode":"YoungCode"})
data.rename(index={1:"Person1", 2:"Person2"})
data.rename_axis("ROWS", axis='rows').rename_axis("COLUMNS", axis='columns')
data1 = data.iloc[0:10]

data2 = data.iloc[10:20]
pd.concat([data1,data2])
data1.join(data2, lsuffix='_data1', rsuffix="_data2")