import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')

das = pd.read_csv("../input/rainfall-in-india/rainfall in india 1901-2015.csv")

das = pd.DataFrame(das)

das.head()
das = das.rename(columns = {'SUBDIVISION':'state', 'JAN': 'jan', 'FEB': 'feb', 'MAR': 'mar', 

                       'APR': 'apr' , 'MAY': 'may', 'JUN': 'jun', 'JUL': 'jul', 'AUG': 'aug', 'SEP': 'sep', 'OCT': 'oct'

                      , 'NOV': 'nov', 'DEC': 'dec', 'ANNUAL': 'annual', 'YEAR': 'year'})

das.tail()
das.state.unique()
das.year.unique()
das_kon = das[das.state=='KONKAN & GOA']

das_kel = das[das.state=='KERALA']

#das_kel.head()
plt.figure(figsize = (20,5))

sns.barplot(x='year', y= 'annual', data = das_kon)

plt.xticks(rotation = 90)

plt.title('KONKAN & GOA RAINFALL DATA FROM 1901-2015')

plt.show()
plt.figure(figsize = (20,5))

sns.barplot(x='year', y= 'annual', data = das_kel)

plt.xticks(rotation = 90)

plt.title('KERALA RAINFALL DATA FROM 1901-2015')

plt.show()
plt.figure(figsize=(25,15))

sns.lineplot(x = 'year', y= 'annual', hue = 'state', data = das)
das_kon.groupby(['year'])['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec'].sum().plot.line(figsize=(15,5))

plt.title('KONKAN & GOA')

das_kel.groupby(['year'])['Jan-Feb', 'Mar-May', 'Jun-Sep', 'Oct-Dec'].sum().plot.line(figsize=(15,5))

plt.title('KERALA')

plt.show()
plt.figure(figsize=(15,5))

das.groupby(['state','year'])['annual'].sum().sort_values(ascending=False).plot()

plt.xticks(rotation=90)
das.groupby(['state','year'])['annual'].sum().sort_values(ascending=False)
plt.figure(figsize=(15,5))

das.groupby(['year'])['annual'].sum().sort_values(ascending=False).head(12).plot(kind='bar', color = 'b')

plt.ylabel('Yearly Rainfall')

plt.title('Yearly Rainfall Data')
plt.figure(figsize=(15,5))

das.groupby(['year'])['annual'].sum().sort_values(ascending=True).tail(12).plot(kind='bar', color = 'c')

plt.ylabel('Yearly Rainfall')

plt.title('Yearly Rainfall Data')
plt.figure(figsize=(15,5))

das.groupby(['state'])['annual'].sum().sort_values(ascending=False).head(12).plot(kind='bar', color = 'g')

plt.ylabel('Total Rainfall')

plt.title('Total Rainfall Data')
plt.figure(figsize=(15,5))

das.groupby(['state'])['annual'].sum().sort_values(ascending=True).tail(10).plot(kind='bar', color = 'b')



plt.ylabel('Total Rainfall')

plt.title('Total Rainfall Data')
plt.figure(figsize=(10,5))

das[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug',

       'sep', 'oct', 'nov', 'dec']].mean().plot(kind= 'bar')

plt.xlabel('Months')

plt.ylabel('Avg. Rainfall')

plt.title('Avg. Monthly Rainfall Data')

plt.show()