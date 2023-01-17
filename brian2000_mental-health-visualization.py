# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Import packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import pie, axis, show

from collections import Counter



# Display plots inline in the notebook

%matplotlib inline 



# Ignore warning related to pandas_profiling (https://github.com/pandas-profiling/pandas-profiling/issues/68)

import warnings

warnings.filterwarnings('ignore') 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



df = pd.read_csv("../input/submission.csv")



# Any results you write to the current directory are saved as output.
# Plot a histogram of the respondents' ages (remove any NaN values)



sns.set(color_codes=True)

sns.set_palette(sns.color_palette("muted"))



sns.distplot(df["age"].dropna());
# Separate by treatment or not



g = sns.FacetGrid(df, col='treatment', size=5)

g = g.map(sns.distplot, "age")
# Create a bar chart comparing gender



df['gender'].value_counts().plot(kind='bar')
country_count = Counter(df['country'].tolist()).most_common(10)

country_idx = [country[0] for country in country_count]

country_val = [country[1] for country in country_count]

fig,ax = plt.subplots(figsize=(8,6))

sns.barplot(x = country_idx,y=country_val ,ax =ax)

plt.title('Top ten country')

plt.xlabel('Country')

plt.ylabel('Count')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
usa = df.loc[df['country'] == 'United States']

top_15_statesUS = usa['state'].value_counts()[:15].to_frame()

plt.figure(figsize=(10,5))

sns.barplot(top_15_statesUS.index,top_15_statesUS['state'])

plt.title('Top 15 US states contributing',fontsize=18,fontweight="bold")

plt.xlabel('')

plt.show()
# Define how to recategorize each state into one of the US Census Bureau regions: West, Midwest, South, Northeast



# Mke a copy of the column to preserve the original data. Work with the new column going forward.

df['region'] = df['state']



# Then, in the new column, assign each type of response to one of the new categories



west = ["WA", "OR", "CA", "NV", "ID", "MT", "WY", "UT", "AZ", "NM", "CO","AK","HI"]

midwest = ["ND", "SD", "NE", "KS", "MN", "IA", "MO", "WI", "IL", "IN", "OH", "MI"]

northeast = ["ME",  "NH", "VT", "MA", "CT", "RI", "NY", "PA", "NJ","DC"]

south = ["MD", "DE", "DC", "WV",  "VA", "NC","SC", "GA", "FL", "KY", "TN", "AL", "MS", "AR", "LA", "OK", "TX"]





df['region'] = df['region'].apply(lambda x:"West" if x in west else x)

df['region'] = df['region'].apply(lambda x:"Midwest" if x in midwest else x)

df['region'] = df['region'].apply(lambda x:"Northeast" if x in northeast else x)

df['region'] = df['region'].apply(lambda x:"South" if x in south else x)



# Make a crosstab to view the count for each of the new categories

region_tab = pd.crosstab(index=df["region"], columns="count")



region_tab.plot(kind="bar", 

                 figsize=(20,7),

                 stacked=True)
X_value=df['no_employees'].value_counts().index



plt.figure(figsize=(10,5))

sns.countplot('no_employees',data = df, order = X_value)

plt.title("Employee Count of Companies",fontsize=20,fontweight="bold")

plt.show()
plt.figure(figsize=(6,5))

sns.countplot(df['treatment'])

plt.title("Treatment Distribution",fontsize=18,fontweight="bold")

plt.show()
plt.figure(figsize=(10,5))

sns.countplot("no_employees", hue="treatment", data=df)

plt.title("Employee count vs Treatment",fontsize=18,fontweight="bold")

plt.ylabel("")

plt.show()
columns =df.columns

not_use_columns=["age","country","state","treatment","disorders"]

for i in range(len(columns)):

    if columns[i] in not_use_columns:

        continue

    else:

        plt.figure(figsize=(10,5))

        sns.countplot(columns[i], hue="treatment", data=df)

        plt.title(columns[i] + " vs Treatment",fontsize=18,fontweight="bold")

        plt.ylabel("")

        plt.show()
disorders = {}



disorderCounts = dict(df['disorders'].value_counts())

for i in disorderCounts:

    # get the disorders separately in case someone answered with more than one disorder

    disorderList = i.split('|')

    for j in disorderList:

        j = j.split(' (')[0]

        disorders[j] = disorders.get(j, 0) + disorderCounts[i]



tmp = pd.DataFrame()

for i in disorders:

    tmp = tmp.append([i] * disorders[i])



tmp[0] = tmp[0].replace([

    'Autism Spectrum Disorder', 'Autism - while not a "mental illness", still greatly affects how I handle anxiety',

    'autism spectrum disorder', 'PDD-NOS'], 'Autism')

tmp[0] = tmp[0].replace(['Aspergers', 'Asperger Syndrome'], "Asperger's Syndrome")

tmp[0] = tmp[0].replace(['posttraumatic stress disourder'], 'Post-traumatic Stress Disorder')

tmp[0] = tmp[0].replace(['ADD', 'Attention Deficit Disorder', 'attention deficit disorder'],

                       'Attention Deficit Hyperactivity Disorder')

tmp[0] = tmp[0].replace(['Schizotypal Personality Disorder'], 'Personality Disorder')

tmp[0] = tmp[0].replace(['Depression'], 'Mood Disorder')

tmp[0] = tmp[0].replace([

    'Autism', "Asperger's Syndrome", 'Intimate Disorder',

    'Seasonal Affective Disorder', 'Burn out', 'Gender Identity Disorder',

    'Suicidal Ideation', 'Gender Dysphoria', 'MCD'], 'Others')



# print(tmp[0].value_counts())

g = sns.countplot(y=tmp[0], order=[

    'Mood Disorder', 'Anxiety Disorder', 'Attention Deficit Hyperactivity Disorder',

    'Post-traumatic Stress Disorder', 'Obsessive-Compulsive Disorder',

    'Stress Response Syndromes', 'Personality Disorder', 'Substance Use Disorder',

    'Eating Disorder', 'Addictive Disorder', 'Dissociative Disorder', 

    'Psychotic Disorder', 'Others'])

g.set_ylabel('Disorders')

g.set_title('Distribution of Mental Health Disorders')

plt.show()