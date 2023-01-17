import numpy as np

import pandas as pd



complete_survey = pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv")

complete_survey_schema = pd.read_csv("../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv")
 # Let's get a feeling of what the data looks like.

print(f"(rows x columns) = {complete_survey.shape}")

complete_survey
from IPython.core.display import HTML



pd.set_option('display.max_rows', 100) # Ensure that we see all results.

pd.set_option('display.max_colwidth', -1) # Ensure that we display the complete text description.

complete_survey_schema
survey = complete_survey[[

    'MainBranch',

    'Hobbyist',

    'OpenSourcer',

    'Employment',

    'Country',

    'Student',

    'EdLevel',

    'UndergradMajor',

    'DevType',

    'YearsCode',

    'Age1stCode',

    'YearsCodePro',

    'ConvertedComp',

    'LanguageWorkedWith',

    'Age',

    'Gender'

]]

survey
survey['Hobbyist']
# Select rows 0 to 4.

survey[0:5]
# Select Country column

survey['Country']
survey[ ['Country'] ]
# Extract 

survey[0:10][ ['Country', 'Student', 'EdLevel' ] ]
# Select individual rows by index.

survey.loc[ [1, 2, 5] ]
# Select inidivdual columns by name.

survey.loc[ :, ['MainBranch', 'Country'] ]
# Select all columns between MainBranch and Country.

survey.loc[ [1, 3, 10], ['MainBranch', 'Country'] ]
# Select all columns between MainBranch and Country

survey.loc[:, 'MainBranch':'Country']
survey_new_index = survey.head(5)

survey_new_index.index = ['a', 'b', 'c', 'd', 'e']

survey_new_index
survey_new_index.loc['b':'d']
# Select the first row

survey.iloc[0]
# Select the first column

survey.iloc[:,0]
# Selecting the coll in the 4th row and 5th column

survey.iloc[3, 4]
survey.head(3)
survey.tail(3)
survey[survey['Country'] == 'Canada']
survey['Country'] == 'Canada'
survey[ (survey['Country'] == 'Bulgaria') & (survey['Employment'] == 'Employed full-time')]
survey.loc[ (survey['Country'] == 'Bulgaria') & (survey['Employment'] == 'Employed full-time'), ["Country", "Employment"] ]
from IPython.display import HTML, display

import tabulate



def print_series(series):

    """

    A helper function for displaying a series using HTML.

    """

    series_as_table = map(lambda x: [x], series)

    display(HTML(tabulate.tabulate(series_as_table, tablefmt='html')))
print_series(survey['MainBranch'].unique())
print_series(survey['OpenSourcer'].unique())
print_series(survey['Employment'].unique())
survey['MonthlyConvertedComp'] = survey['ConvertedComp']/12
survey_ext = survey.copy()

survey_ext['MonthlyConvertedComp'] = survey_ext['ConvertedComp']/12

survey_ext[ ['ConvertedComp', 'MonthlyConvertedComp'] ]
survey['Country'].value_counts()
survey['Hobbyist'].value_counts(normalize=True) 
survey['EdLevel'].value_counts()
survey_by_country = survey.groupby('Country')

type(survey_by_country)
survey_by_country.indices.keys()
survey_by_country.indices['Afghanistan']
survey.loc[719, 'Country']
survey.groupby('Country')['ConvertedComp'].mean()
survey.groupby('Country')['ConvertedComp'].std()
survey.groupby('Country')['ConvertedComp'].agg(["mean", "std"])
survey.groupby(pd.cut(survey['Age'], np.arange(0, 101, 10)))['Age'].count()
survey['ConvertedComp'].dropna()
survey.dropna(subset=['ConvertedComp'])
# Let's first find rows containing no salary information

survey_index_no_salary = survey['ConvertedComp'].isnull()

survey[survey_index_no_salary]
salary_mean = survey['ConvertedComp'].mean()

survey.fillna({'ConvertedComp': salary_mean})[survey_index_no_salary] # Display the rows which didn't contain salary.
salary_by_age = survey[ ['Age', 'ConvertedComp'] ].dropna().groupby('Age')

salary_by_age.mean()

salary_by_age.mean().dropna().plot()
p5, p95 = survey['ConvertedComp'].quantile(0.05), survey['ConvertedComp'].quantile(0.95)

salary_by_age = survey[ (survey['ConvertedComp'] >= p5) & (survey['ConvertedComp'] <= p95) ] [ ['Age', 'ConvertedComp'] ].dropna().groupby('Age')

salary_by_age.mean().dropna().plot()
survey_uk = survey[survey['Country'] == 'United Kingdom']

p5, p95 = survey_uk['ConvertedComp'].quantile(0.05), survey_uk['ConvertedComp'].quantile(0.95)

salary_by_age = survey_uk[ (survey_uk['ConvertedComp'] >= p5) & (survey_uk['ConvertedComp'] <= p95) ] [ ['Age', 'ConvertedComp'] ].dropna().groupby('Age')

salary_by_age.mean().dropna().plot()
survey.groupby('OpenSourcer')['OpenSourcer'].count().plot(kind='bar')
survey.groupby('OpenSourcer')['OpenSourcer'].count().plot(kind='pie')
series = []

countries = ['United States', 'Japan', 'United Kingdom', 'Canada', 'Germany', 'Italy', 'Russia']

# For each of the countries above, generate an aggregation for the mean of compensation by age.

for country in countries:

    survey_by_country = survey[survey['Country'] == country]

    salary_by_age = survey_by_country[ ['ConvertedComp'] ].groupby(pd.cut(survey_by_country['Age'], np.arange(0, 101, 5)))

    series.append(salary_by_age.mean())

# Concatate the result into a data frame and plot.

c = pd.concat(series, axis=1)

c.columns = countries

c.plot()
