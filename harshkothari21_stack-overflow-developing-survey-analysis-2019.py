import pandas as pd
df = pd.read_csv('../input/survey_results_public.csv', index_col='Respondent')
df
df.shape
df.info
df.info()
pd.set_option('display.max_columns', 85)
schema_df = pd.read_csv('../input/survey_results_schema.csv', index_col='Column')
schema_df
pd.set_option('display.max_rows', 85)
df.head(10)
type(df)
type(df.Hobbyist)
df.columns
df.iloc[2, 2]
df.iloc[2]
df.loc[1:5, 'Hobbyist':'Employment']
df.loc[1:2, 'Hobbyist':'Employment']
df
df.loc[1]
df.iloc[0]
schema_df.loc['Hobbyist']
schema_df.loc['MgrIdiot']
schema_df.loc['MgrIdiot', 'QuestionText']
schema_df.sort_index()
schema_df.sort_index(ascending=False)
schema_df.sort_index(inplace=True)
schema_df
high_salary = (df['ConvertedComp'] > 70000)
df.loc[high_salary]
df.loc[high_salary, ['LanguageWorkedWith', 'ConvertedComp', 'Country']]
df.loc[high_salary, 'LanguageWorkedWith']
countries = ['United States', 'India']
filt1 = df['Country'].isin(countries)
df.loc[filt1, ['LanguageWorkedWith', 'ConvertedComp', 'Country']]
df['LanguageWorkedWith']
filt2 = df['LanguageWorkedWith'].str.contains('Python', na=False)
df.loc[filt2, ['LanguageWorkedWith', 'ConvertedComp']]
df.loc[~filt2, ['LanguageWorkedWith', 'ConvertedComp']]
df['Country']
df['Country'].str.lower()
df['Country'].str.upper()
df['Country']
df.rename(columns={'ConvertedComp' : 'SalaryUSD'}, inplace=True)
df['SalaryUSD']
df['Hobbyist'].map({'Yes' : 'True', 'No' : 'False'})
df['Hobbyist'] = df['Hobbyist'].map({'Yes' : 'True', 'No' : 'False'})
df['Hobbyist']
df['Hobbyist'].apply(len)
df.apply(len)
df.rename(columns = {'SalaryUSD' : 'ConvertedComp'}, inplace=True)
df
df.sort_values(['Country'], inplace=True)
df['Country']
df.sort_values(['Country', 'ConvertedComp'], ascending=[True, False], inplace=True)
df[['Country', 'ConvertedComp']].head(50)
df['ConvertedComp'].nlargest(10)
df.nlargest(5, 'ConvertedComp')
df[['LanguageWorkedWith', 'ConvertedComp', 'DevEnviron']].nlargest(10, 'ConvertedComp') 
df
df.sort_index(inplace=True)
df.describe()
df['Hobbyist'].value_counts()
df['SocialMedia'].value_counts()
df['SocialMedia'].value_counts(normalize=True)
df['Country'].value_counts()
country_grp = df.groupby(['Country'])
country_grp.get_group('India')
country_grp_india = country_grp.get_group('India')
country_grp_india 
country_grp_india['SocialMedia'].value_counts()
df.head(2)
country_grp_india['Age'].value_counts()
country_grp['ConvertedComp'].median().loc['India']
filt = df['Country'] == 'India'
df.loc[filt]['LanguageWorkedWith'].str.contains('Python').sum()
country_grp['LanguageWorkedWith'].apply(lambda x: x.str.contains('Python').sum()).loc['India']
country_res = df['Country'].value_counts()
country_res
uses_python = country_grp['LanguageWorkedWith'].apply(lambda x: x.str.contains('Python').sum())
uses_python
python_df = pd.concat([country_res, uses_python], axis='columns', sort=False)
python_df
python_df.rename(columns={'Country': 'Number of Respond', 'LanguageWorkedWith':'Knows Python'}, inplace=True)
python_df
python_df['pct'] = (python_df['Knows Python']/python_df['Number of Respond'] * 100)
python_df
na_val = ['Na', 'Missing']
df_test = pd.read_csv('../input/survey_results_public.csv', index_col='Respondent', na_values = na_val)
df['YearsCode']
df_test['YearsCode'] =  df['YearsCode'].astype(float)
df_test['YearsCode'].unique()
df_test['YearsCode'].replace('Less than 1 year', 0, inplace=True)
df_test['YearsCode'].replace('More than 50 years', 51, inplace=True)
df_test['YearsCode'].unique()
df_test['YearsCode'] = df_test['YearsCode'].astype(float)
df_test['YearsCode'].describe()





