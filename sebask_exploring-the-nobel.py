import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

from subprocess import check_output

import matplotlib.pyplot as plt

import seaborn as sns



#print(check_output(["ls", "../input"]).decode("utf8")) # list files in input directory

# Load data

df = pd.read_csv('../input/archive.csv')

df.head()
dfcit = df.groupby('Birth City', as_index=False).count()

dfcit = dfcit.sort_values('Prize', ascending=False)

sns.barplot(x='Prize', y='Birth City', data=dfcit.head(10),palette='muted')

plt.title('Top 10 Birth Cities of Nobel Laureates')

plt.xlabel('Number of Prizes')

plt.ylabel('')
dfcou = df.groupby('Birth Country', as_index=False).count()

dfcou = dfcou.sort_values('Prize',ascending = False)

sns.barplot(x='Prize', y='Birth Country', data=dfcou.head(10),palette='muted')

plt.title('Top 10 Birth Countries of Nobel Laureates')

plt.xlabel('Number of Prizes')

plt.ylabel('')
dfinst = df.groupby('Organization Name', as_index=False).count()

dfinst = dfinst.sort_values('Prize',ascending = False)

sns.barplot(x='Prize', y='Organization Name', data=dfinst.head(10),palette='muted')

plt.title('Top 10 Institutions of Nobel Laureates')

plt.xlabel('Number of Prizes')

plt.ylabel('')
dfoc = df.groupby('Organization Country', as_index=False).count()

dfoc = dfoc.sort_values('Prize',ascending = False)

sns.barplot(x='Prize', y='Organization Country', data=dfoc.head(10),palette='muted')

plt.title('Top 10 Countries of Laureate Institutions')

plt.xlabel('Number of Prizes')

plt.ylabel('')
usa = df[df['Organization Country'] == 'United States of America']

us_by_year = usa.groupby('Year',as_index=False).count()

#us_by_year.head(15)

sns.lmplot(data=us_by_year, x='Year', y='Prize')
uk = df[df['Organization Country'] == 'United Kingdom']

uk_by_year = uk.groupby('Year',as_index=False).count()

#us_by_year.head(15)

sns.lmplot(data=uk_by_year, x='Year', y='Prize')
deu = df[df['Organization Country'] == 'Germany']

deu_by_year = deu.groupby('Year',as_index=False).count()

#us_by_year.head(15)

sns.lmplot(data=deu_by_year, x='Year', y='Prize')
isr = df[df['Organization Country'] == 'Japan']

isr_by_year = isr.groupby('Year',as_index=False).count()

#us_by_year.head(15)

sns.lmplot(data=isr_by_year, x='Year', y='Prize')