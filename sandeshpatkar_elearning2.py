import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
stack = pd.read_csv('../input/survey.csv')

stack.head()
stack.shape
stack.info()
stack.dtypes
stack.columns.tolist()
keep_columns = ['Respondent',

 'Hobby',

 'OpenSource',

 'Country',

 'Student',

 'Employment',

 'FormalEducation',

 'UndergradMajor',

 'CompanySize',

 'DevType',

 'YearsCoding',

 'YearsCodingProf',

 'HopeFiveYears',

 'Currency',

 'Salary',

 'TimeFullyProductive',

 'EducationTypes',

 'SelfTaughtTypes',

 'TimeAfterBootcamp',

 'HackathonReasons',

 'LanguageWorkedWith',

 'LanguageDesireNextYear',

 'DatabaseWorkedWith',

 'DatabaseDesireNextYear',

 'PlatformWorkedWith',

 'PlatformDesireNextYear',

 'FrameworkWorkedWith',

 'FrameworkDesireNextYear',

 'Gender', 'Age']
stack = stack[keep_columns]

stack.head()
def create_df(series, norm):

    if norm == 1:

        val = series.value_counts(normalize = True)*100

        df = pd.DataFrame(val)

        df['index'] = df.index

    elif norm == 0:

        val = series.value_counts()

        df = pd.DataFrame(val)

        df['index'] = df.index

    return df
country_stack = create_df(stack.Country, 0)

country_stack.head()
def plot_bar(x,y,data, title):

    sns.barplot(x = x, y = y, data = data).set_title(title)

    plt.xticks(rotation = 90)
plot_bar(data = country_stack.head(), y = 'Country', x = 'index', title = 'Country-wise distribution (2018)')
stack.isnull().sum()
stack = stack[stack['SelfTaughtTypes'].isnull()]

stack.head()
del stack['SelfTaughtTypes']
stack.shape
stack.LanguageDesireNextYear.head(10)
stack = stack[stack['LanguageDesireNextYear'].notnull()]

stack
stack.DevType.isna().sum()
stack = stack.dropna(subset = ['DevType']) #removing NA

mobile_n_web = stack.DevType.str.contains('Mobile developer|Back-end|Front-end|Full-stack|Web developer')

mobile_n_web.value_counts()
dev_interested = create_df(mobile_n_web, 1)

dev_interested.head()
sns.barplot(x = 'index', y = 'DevType', data = dev_interested)

plt.title('Job Roles of Interest vs. Other')

plt.xticks([0,1],['Other', 'Web/Mobile Development'])

plt.xlabel('Job Roles')

plt.ylabel('Percentage')
ind_can = stack[(stack['Country'] == 'India') | (stack['Country'] == 'Canada')]

ind_can.shape
ind_can.head(2)
sns.countplot('Country', data = ind_can)
country_stack.head().plot(kind = 'bar', subplots = True, color = 'lightgreen', title = 'Country wise distribution: 2018')
ind_ger = stack[(stack['Country'] == 'India') | (stack['Country'] == 'Germany')]

ind_ger.shape
sns.countplot('Country', data = ind_ger)