import os

import glob

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import pandasql as pdsql # *schnüff*

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as sst

color = sns.color_palette()

%matplotlib inline
from subprocess import check_output

print(check_output(["ls", "../input/kaggle-survey-2017/"]).decode("utf8"))
# Read each of the file

#cvRates = pd.read_csv('../input/conversionRates.csv', encoding="ISO-8859-1")

#freeForm = pd.read_csv('../input/freeformResponses.csv', encoding="ISO-8859-1")

multiChoice = pd.read_csv('../input/kaggle-survey-2017/multipleChoiceResponses.csv', encoding="ISO-8859-1")

countrypop = pd.read_csv('../input/wikipedia-country-population-currencies/countries.csv', encoding="ISO-8859-1")

#schema = pd.read_csv('../input/schema.csv', encoding="ISO-8859-1")
lechien = multiChoice.replace({'Country':{

    'People \'s Republic of China' : 'China',

    'Republic of China' : 'Taiwan'

}}) # Since when are there two?
counts = lechien['Country'].value_counts().to_frame(name='Count')

j = counts.join(countrypop.set_index('Name')).filter(items=['Count','Population2017'])

other = countrypop['Population2017'].sum() - j['Population2017'].sum()

j['Population2017']['Other'] = other
p = j.apply(lambda foo: foo['Count'] / foo['Population2017'] * 1000000, axis=1).sort_values(ascending=False)
# For the sake of a good plot, I will be excluding all those countries from the graph

# where the number of surveys is less 100

plt.figure(figsize=(20,20))

sns.barplot(y=p.head(20).index, x=p.head(20).values, color=color[4], orient='h')

plt.title('N in one million inhabitants took the survey', fontsize=16)

plt.xlabel('Count', fontsize=16)

plt.ylabel('Country', fontsize=16)

plt.show()
# Out of curiosity

p['Other']