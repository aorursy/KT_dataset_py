import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("/kaggle/input/world-happiness-report/2020.csv")

df.head(10)
df.columns
df.groupby('Regional indicator').agg({'Country name':'count', 'Ladder score':np.mean})
df=df.sort_values(by='Ladder score', ascending=False)

sns.barplot(df['Country name'][0:10],df['Ladder score'][0:10])

plt.xticks(rotation=90)

plt.title('Top 10 countries with the highest score Ladder')

plt.show()
plot1 = plt.figure(1)

df_temp= df[df['Regional indicator'] == 'Central and Eastern Europe']



sns.barplot(df_temp['Country name'][0:10],df_temp['Ladder score'][0:10])

plt.xticks(rotation=90)

plt.title('Top 10 countries in Central and Eastern Europe with the highest score Ladder')



plot2 = plt.figure(2)

df_temp= df[df['Regional indicator'] == 'Commonwealth of Independent States']



sns.barplot(df_temp['Country name'][0:10],df_temp['Ladder score'][0:10])

plt.xticks(rotation=90)

plt.title('Top 10 countries in Commonwealth of Independent States with the highest score Ladder')



plot3 = plt.figure(3)

df_temp= df[df['Regional indicator'] == 'East Asia']



sns.barplot(df_temp['Country name'][0:10],df_temp['Ladder score'][0:10])

plt.xticks(rotation=90)

plt.title('Top 10 countries in East Asia with the highest score Ladder')



plot4 = plt.figure(4)

df_temp= df[df['Regional indicator'] == 'Latin America and Caribbean']



sns.barplot(df_temp['Country name'][0:10],df_temp['Ladder score'][0:10])

plt.xticks(rotation=90)

plt.title('Top 10 countries in Latin America and Caribbean with the highest score Ladder')



plot5 = plt.figure(5)

df_temp= df[df['Regional indicator'] == 'Middle East and North Africa']



sns.barplot(df_temp['Country name'][0:10],df_temp['Ladder score'][0:10])

plt.xticks(rotation=90)

plt.title('Top 10 countries in Middle East and North Africa with the highest score Ladder')



plot6 = plt.figure(6)

df_temp= df[df['Regional indicator'] == 'North America and ANZ']



sns.barplot(df_temp['Country name'][0:10],df_temp['Ladder score'][0:10])

plt.xticks(rotation=90)

plt.title('Top 10 countries in North America and ANZ with the highest score Ladder')



plot7 = plt.figure(7)

df_temp= df[df['Regional indicator'] == 'South Asia']



sns.barplot(df_temp['Country name'][0:10],df_temp['Ladder score'][0:10])

plt.xticks(rotation=90)

plt.title('Top 10 countries in South Asia with the highest score Ladder')



plot8 = plt.figure(8)

df_temp= df[df['Regional indicator'] == 'Southeast Asia']



sns.barplot(df_temp['Country name'][0:10],df_temp['Ladder score'][0:10])

plt.xticks(rotation=90)

plt.title('Top 10 countries in Southeast Asia with the highest score Ladder')



plot9 = plt.figure(9)

df_temp= df[df['Regional indicator'] == 'Sub-Saharan Africa']



sns.barplot(df_temp['Country name'][0:10],df_temp['Ladder score'][0:10])

plt.xticks(rotation=90)

plt.title('Top 10 countries in Sub-Saharan Africa with the highest score Ladder')



plot10 = plt.figure(10)

df_temp= df[df['Regional indicator'] == 'Western Europe']



sns.barplot(df_temp['Country name'][0:10],df_temp['Ladder score'][0:10])

plt.xticks(rotation=90)

plt.title('Top 10 countries in Western Europe with the highest score Ladder')

plt.show()
plot1 = plt.figure(1)

df_temp= df[df['Regional indicator'] == 'Central and Eastern Europe']



sns.barplot(df_temp['Country name'],df_temp['Ladder score'])

plt.xticks(rotation=90)

plt.title('Countries Happines Rank in Central and Eastern Europe')



plot2 = plt.figure(2)

df_temp= df[df['Regional indicator'] == 'Commonwealth of Independent States']



sns.barplot(df_temp['Country name'],df_temp['Ladder score'])

plt.xticks(rotation=90)

plt.title('Countries Happines Rank in Commonwealth of Independent States')



plot3 = plt.figure(3)

df_temp= df[df['Regional indicator'] == 'East Asia']



sns.barplot(df_temp['Country name'],df_temp['Ladder score'])

plt.xticks(rotation=90)

plt.title('Countries Happines Rank in East Asia ')



plot4 = plt.figure(4)

df_temp= df[df['Regional indicator'] == 'Latin America and Caribbean']



sns.barplot(df_temp['Country name'],df_temp['Ladder score'])

plt.xticks(rotation=90)

plt.title('Countries Happines Rank in Latin America and Caribbean')



plot5 = plt.figure(5)

df_temp= df[df['Regional indicator'] == 'Middle East and North Africa']



sns.barplot(df_temp['Country name'],df_temp['Ladder score'])

plt.xticks(rotation=90)

plt.title('Countries Happines Rank in Middle East and North Africa')



plot6 = plt.figure(6)

df_temp= df[df['Regional indicator'] == 'North America and ANZ']



sns.barplot(df_temp['Country name'],df_temp['Ladder score'])

plt.xticks(rotation=90)

plt.title('Countries Happines Rank in North America and ANZ')



plot7 = plt.figure(7)

df_temp= df[df['Regional indicator'] == 'South Asia']



sns.barplot(df_temp['Country name'],df_temp['Ladder score'])

plt.xticks(rotation=90)

plt.title('Countries Happines Rank in South Asia')



plot8 = plt.figure(8)

df_temp= df[df['Regional indicator'] == 'Southeast Asia']



sns.barplot(df_temp['Country name'],df_temp['Ladder score'])

plt.xticks(rotation=90)

plt.title('Countries Happines Rank in Southeast Asia')



plot9 = plt.figure(9)

df_temp= df[df['Regional indicator'] == 'Sub-Saharan Africa']



sns.barplot(df_temp['Country name'],df_temp['Ladder score'])

plt.xticks(rotation=90)

plt.title('Countries Happines Rank in Sub-Saharan Africa')



plot10 = plt.figure(10)

df_temp= df[df['Regional indicator'] == 'Western Europe']



sns.barplot(df_temp['Country name'],df_temp['Ladder score'])

plt.xticks(rotation=90)

plt.title('Countries Happines Rank in Western Europe')

plt.show()