

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime 

import math



df = pd.read_csv("../input/Serial Killers Data.csv",encoding = "ISO-8859-1")

df.head()
print(df.shape)



row_format = "{:>28}{:6}"

for index, column in enumerate(df.columns):

    print(row_format.format(column, df[column].isnull().sum()), end='')

    if index % 2 == 0:

        print('')
ethnicities = ['White','Black','Hispanic','Asian','NativeAmerican','Aboriginal']

ethnicity_dict = {}

for x,y in zip(range(1, 7, 1), ethnicities):

    ethnicity_dict[x] = y



gender = ['Male','Female']

gender_dict = {}

for a,b in zip(range(1, 3, 1), gender):

    gender_dict[a] = b

    

birth_order = ['Oldest','Middle','Youngest','Only']

birthcat_dict = {}

for c,d in zip(range(1, 5, 1), birth_order):

    birthcat_dict[c] = d
sns.countplot(df['Sex'].map(gender_dict),palette='Blues_d')

plt.title("Gender Makeup")
unknowns = pd.Series(['Unknown'] * df.Race.isnull().sum())

df['Race'].map(ethnicity_dict).append(unknowns).value_counts().plot(kind='bar')

plt.title('Murderers by Ethnicity')
df['BirthCat'].map(birthcat_dict).value_counts().plot(kind='pie', label='')

plt.title('Birth Order')
df.BirthCat.isnull().sum()
df.plot(x="SuspectedDeaths", y="ConvictedDeaths", kind="scatter")

plt.axis([0, 150, 0, 150])
data = df.filter(items=['SuspectedDeaths', 'ConvictedDeaths', 'YearFinal'])

data = data[np.isfinite(df['SuspectedDeaths'])]

data = data[np.isfinite(data['ConvictedDeaths'])]



data['ConvictionsToSuspected'] = data['ConvictedDeaths'] / data['SuspectedDeaths']



data.plot(x="YearFinal", y="ConvictionsToSuspected", kind="scatter")

plt.axis([1850, 2020, 0, 1.6])



def average_per_decade(dframe, column):

    totals = {}

    for index, row in dframe.iterrows():

        year = str(row['YearFinal'])[:3] + '0'

        if year in totals:

            totals[year] = (totals[year][0] + row[column], totals[year][1] + 1)

        else:

            totals[year] = (row[column], 1)



    if 'nan0' in totals: del totals['nan0'] 



    averages_per_decade = {}

    for pair in totals.items():

        year = pair[0]

        average = pair[1][0] / pair[1][1]

        averages_per_decade[year] = average

        

    return pd.Series(averages_per_decade)



plt.plot(average_per_decade(data, 'ConvictionsToSuspected'), 'r-')
common_ethnicity_dict = { 1: 1, 2: 2, 3: 3, 4: 4, 5: math.nan, 6: math.nan }



data = df.filter(items=['SuspectedDeaths', 'ConvictedDeaths', 'YearFinal', 'Race'])

data = data[np.isfinite(df['SuspectedDeaths'])]

data = data[np.isfinite(data['ConvictedDeaths'])]

data['Race'] = data['Race'].map(common_ethnicity_dict)

data = data[np.isfinite(data['Race'])]

data['Race'] = data['Race'].map(ethnicity_dict)



data['ConvictionsToSuspected'] = data['ConvictedDeaths'] / data['SuspectedDeaths']



white_murderers = data[data['Race'] == 'White']

black_murderers = data[data['Race'] == 'Black']

hispanic_murderers= data[data['Race'] == 'Hispanic']

asian_murderers = data[data['Race'] == 'Asian']



avg_white_murderers = average_per_decade(white_murderers, 'ConvictionsToSuspected')

avg_black_murderers = average_per_decade(black_murderers, 'ConvictionsToSuspected')

avg_hispanic_murderers= average_per_decade(hispanic_murderers, 'ConvictionsToSuspected')

avg_asian_murderers = average_per_decade(asian_murderers, 'ConvictionsToSuspected')



colors = { 'white': [1, 0, 0], 'black': [0, 0.5, 1], 'hispanic': [0.5, 0.5, 0], 'asian': [0.5, 0, 0.5] }



white_murderers.plot(

    x="YearFinal", 

    y="ConvictionsToSuspected", 

    kind="scatter",

    color=colors['white'],

    alpha=0.1,

    edgecolors='none'

)

plt.scatter(

    x=black_murderers['YearFinal'], 

    y=black_murderers['ConvictionsToSuspected'],

    color=colors['black'],

    alpha=0.1,

    label=''

)

plt.scatter(

    x=hispanic_murderers['YearFinal'], 

    y=hispanic_murderers['ConvictionsToSuspected'],

    color=colors['hispanic'],

    alpha=0.1,

    label=''

)

plt.scatter(

    x=asian_murderers['YearFinal'], 

    y=asian_murderers['ConvictionsToSuspected'],

    color=colors['asian'],

    alpha=0.1,

    label=''

)

plt.axis([1960, 2020, 0, 1.3])

plt.plot(avg_white_murderers, linestyle='-', color=colors['white'], label='White')

plt.plot(avg_black_murderers, linestyle='-', color=colors['black'], label='Black')

plt.plot(avg_hispanic_murderers, linestyle='-', color=colors['hispanic'], label='Hispanic')

plt.plot(avg_asian_murderers, linestyle='-', color=colors['asian'], label='Asian')

legend = plt.legend(loc='upper left')