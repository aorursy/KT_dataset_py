import pandas as pd
data= pd.read_csv("../input/data-for-datavis/insurance.csv")
data.region.value_counts()
data.groupby('region').region.count()
data.groupby("region").bmi.min()
data.groupby('age').apply(lambda d: data.age.iloc[0])
data.groupby(['age']).age.agg(['mean','min','max'])
data.sort_values(by='age', ascending=True)
data.sort_index()
data.sort_values(by=['age','bmi'])
data.children.dtype
data.head()
data.bmi.astype("int64")
spotifyData = pd.read_csv("../input/data-for-datavis/spotify.csv")
spotifyData[pd.isnull(spotifyData.Unforgettable)]
spotifyData['HUMBLE.'].fillna(100000)
data.head()
data.region.replace("southwest", "United States")
df = pd.read_csv('../input/stack-overflow-2018-developer-survey/survey_results_public.csv', index_col='Respondent')
df.median()
df.head(15)
df['Country'].value_counts(normalize=True)
country_group = df.groupby(['Country'])
filter1 = df['Country'] == "United States"
filter1
df.loc[filter1]
df.loc[filter1]['Exercise'].value_counts()
df['Exercise'].value_counts()
df.groupby(['Country'])['Exercise'].value_counts()
df.groupby(['Country'])['Exercise'].value_counts().head(50)
country_groups['Exercise'].value_counts().loc['Bangladesh']
country_group['ConvertedSalary'].median().head(50)
country_group['ConvertedSalary'].median().loc['Germany']
country_group['ConvertedSalary'].agg(['median', 'mean'])
country_group['ConvertedSalary'].agg(['median', 'mean']).loc['Canada']
df['Country']
filter2 = df['Country'] == 'United States'
df.loc[filter2]['LanguageWorkedWith'].str.contains('Python').sum()
df.loc[filter2]['LanguageWorkedWith'].str.contains('Python').sum()
country_group['LanguageWorkedWith'].apply(lambda x: x.str.contains('Python').sum())
country_respondents = df["Country"].value_counts()

country_uses_python = country_group['LanguageWorkedWith'].apply(lambda x: x.str.contains('Python').sum())

python_df = pd.concat([country_respondents, country_uses_python], axis='columns')
python_df
python_df = python_df.rename(columns={'Country':'NumRespondents', 'LanguageWorkedWith':'NumKnowsPython'})
python_df['PercentKnowsPython'] = (python_df['NumKnowsPython']/python_df['NumRespondents']) * 100
python_df
python_df.sort_values(by='PercentKnowsPython', ascending=False, inplace=True)
python_df.head(50)
python_df.loc['Japan']
df.head(15)
df.Country.replace("United States","Banana")
df.rename(columns={'Country':'Nation'})
df.rename(index={1:45, 4:1})
df.rename_axis("index", axis='rows').rename_axis("features", axis='columns')
data1 = df.iloc[0:20, :]
data2 = df.iloc[20:50, :]
pd.concat([data1, data2]).head(50)
data1.join(data2, lsuffix='_data1', rsuffix='_data2')