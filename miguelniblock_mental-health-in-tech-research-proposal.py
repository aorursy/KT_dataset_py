import pandas as pd

import numpy as np

import statistics as stat

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

# Suppress annoying harmless error.

warnings.filterwarnings(action="ignore")

%matplotlib inline
data = pd.read_csv('../input/survey.csv')
# How many datapoints, how many variables?

data.shape
# What variables do we have?

pd.options.display.max_columns = 30

data.head()
# Check for missing values

data.info()
#data.drop('Timestamp',axis=1,inplace=True)

data.shape
# Replace noise with the mean age.

data.Age[data.Age < 15] = 32

data.Age[data.Age > 100] = 32



# Get a statistical summary, median age and histogram.

print(data.Age.describe())

print('median: ', np.median(data.Age))



sns.set(style="darkgrid")

data.Age.plot(kind='hist')

plt.show()
data.work_interfere.fillna(value='No Issue',inplace=True)

data.work_interfere.value_counts()
sns.set(style="darkgrid")



g = sns.FacetGrid(col='work_interfere', sharey=True,

                col_order=['No Issue','Never','Rarely','Sometimes','Often'],

                data=data,despine=True)



g = g.map(plt.hist, 'Age', bins=np.arange(18,72,5))



g = g.set_ylabels('Counts')



plt.show()
# Let's get a view of all the unique gender raw values.

data.Gender.value_counts()
# If it contains an 'f' or a 'w', then turn into 'F'

data.Gender[data.Gender.apply(lambda x: 'f' in str.lower(x))] = 'F'

data.Gender[data.Gender.apply(lambda x: 'w' in str.lower(x))] = 'F'



# Else, turn into 'M'

data.Gender[data.Gender != 'F'] = 'M'
# How many men and women do we have after cleaning?

data.Gender.value_counts()
sns.set(style="darkgrid")



g = sns.catplot(col='tech_company', x='Gender', data=data, 

                kind='count', sharey=False, height=3)



g.set_axis_labels('Men/Women')



plt.show()
sns.set(style="darkgrid")



g = sns.catplot(hue='work_interfere', x='Gender', col='tech_company', kind='count', 

                data=data, hue_order=['No Issue','Never','Rarely','Sometimes','Often'],

               sharey=False)

g = g.set_axis_labels('Count of "Work Interfere" by "Gender"')

plt.show()
sns.set(style="darkgrid")

g = sns.catplot(x='work_interfere', col='Gender', kind='count', 

                data=data, order=['No Issue','Never','Rarely','Sometimes','Often'],

               sharey=False)

g = g.set_axis_labels('Count of "Work Interfere" by "Gender"')

plt.show()
sns.set(style="darkgrid")



g = sns.catplot(kind='box', y='Age', col='tech_company', x='Gender',

                hue='work_interfere', data=data, 

                hue_order=['No Issue','Never','Rarely','Sometimes','Often'],

               height=7, aspect=0.7)

# Let's first inspect how many respondents are self-employed

g = sns.catplot(kind='count', col='tech_company', x='Gender',

                hue='work_interfere', data=data, row='self_employed',

                hue_order=['No Issue','Never','Rarely','Sometimes','Often'],

               height=3, aspect=1.5, sharey=False, sharex=False)
# Let's isolate the respondents who are in the tech sector for the purposes of this figure.

dfplot = data[data.tech_company == 'Yes']

g = sns.catplot(kind='count', col='self_employed', x='work_interfere', row='Gender',

                data=dfplot, order=['No Issue','Never','Rarely','Sometimes','Often'],

               height=3, aspect=2, sharey=False)
dfplot = data[data.tech_company == 'Yes']

g = sns.catplot(kind='boxen', y='Age',col='self_employed', x='Gender', hue='work_interfere',

                data=dfplot, hue_order=['No Issue','Never','Rarely','Sometimes','Often'],

               height=7, aspect=0.8, sharey=False)
# Create a list of the 10 countries with most respondents

countries = data.Country.value_counts()[:10].index.tolist()



# Slice the data to include only the 10 countries with most respondents, tech, and employees only.

dfplot = data[data.Country.isin(countries)][data.tech_company == 'Yes'][data.self_employed == 'No']



g=sns.catplot(col='Country', x='work_interfere', kind='count', data=dfplot, col_wrap=4, sharey=False,

             sharex=False, height=4, aspect=1.2, col_order=countries, 

              order=['No Issue','Never','Rarely','Sometimes','Often'])



plt.show()
# Slice the data to include only the US, in order to plot by state

states = data.state.value_counts()[:10].index.tolist()



# Slice the data to include only the 10 states with most respondents, tech, and employees only.

dfplot = data[data.state.isin(states)][data.tech_company == 'Yes'][data.self_employed == 'No']



g=sns.catplot(col='state', x='work_interfere', kind='count', data=dfplot, col_wrap=4, sharey=False,

             sharex=False, height=4, aspect=1.2, col_order=states, 

              order=['No Issue','Never','Rarely','Sometimes','Often'])



plt.show()
# Timestamps are currently formatted as strings. Convert them to pandas' time data.

data.Timestamp = pd.to_datetime(data.Timestamp)
# Statistical summary of time series

data.Timestamp.describe()
# Let's put the dates in order

dfplot = data.sort_values(by='Timestamp')



# View the time distribution of raw data

dfplot.Timestamp.hist()

plt.title('Time Data Distribution- All')

plt.xticks(rotation=90)

plt.show()



# Slice a portion of the time series

dfplot = dfplot[dfplot.Timestamp < '2014-09-03']



# View the distribution of the slice

dfplot.Timestamp.hist()

plt.title('Time Data Distribution- First Week Only')

plt.xticks(rotation=90)

plt.show()
# Let's see the categories of each variable related to mental health services

categorical = data.loc[:,'benefits':'leave'].select_dtypes(include=['object'])

for i in categorical:

    column = categorical[i]

    print('\n'+ i.upper())

    print(column.value_counts())

    sns.countplot(data=categorical, x=column)

    plt.xticks(rotation=45)

    plt.show()
# Define a subset of data including mental-care variables, plus 'work_interfere'

# and 'family_history'.



health = pd.concat([data.loc[:,'benefits':'leave'],data[['family_history','work_interfere']]],axis=1)



# Plot a correlation matrix, using dummies.

plt.figure(figsize=(10,8))

sns.heatmap(pd.get_dummies(health).corr(),square=True)

plt.show()
# Let's focus on the people with a family history of mental health.

subset = data[data.family_history=='Yes'][data.tech_company=='Yes'][data.self_employed=='No']



# let's look at the distributions of 'work_interfere', by 'leave' on the subset.

colorder = ["Don't know","Very easy","Somewhat easy","Somewhat difficult","Very difficult"]

xorder = ['No Issue','Never','Rarely','Sometimes','Often']

sns.catplot(data=subset, col='leave', x='work_interfere', kind='count',height=4,

           aspect=1, order=xorder, col_order=colorder, sharey=False)



plt.show()