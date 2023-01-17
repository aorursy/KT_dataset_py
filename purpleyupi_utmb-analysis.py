import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

%matplotlib inline
df_2017 = pd.read_csv("../input/utmb_results_2017.csv")

df_2018 = pd.read_csv("../input/utmb_results_2018.csv")

df_2019 = pd.read_csv("../input/utmb_results_2019.csv")
df_2017.head()
df_2018.head()
df_2019.head()
df_2017['Year'] = 2017

df_2018['Year'] = 2018

df_2019['Year'] = 2019
df = pd.concat([df_2017, df_2018, df_2019]) # combine the three dataframes
df = df.drop(['Unnamed: 0'], axis=1)
df.columns
df.info()
df.head()
def convert_to_minutes(row):

    return sum(i*j for i, j in zip(map(float, row.split(':')), [60, 1, 1/60]))

df['Minutes'] = df['Time'].apply(convert_to_minutes)
df['Minutes'] = df['Minutes'].round(2)

df.head()
plt.figure(num=None, figsize=(8, 6), dpi=80)

plt.hist(df['Minutes'], alpha=0.5)

plt.title('2017 & 2018 & 2019 UTMB Times', fontsize=18, fontweight="bold")

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel('Time (min)', fontsize=18)

plt.ylabel('Frequency', fontsize=18)

plt.show()
df.describe()
plt.figure(figsize=(8, 6), dpi=80)

sns.boxplot(x="Year", y="Minutes", data=df)

plt.title('UTMB Results by Year', fontsize=18, fontweight="bold")

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("Year", fontsize=18)

plt.ylabel("Minutes", fontsize=18)
plt.figure(figsize=(8, 6), dpi=80)

sns.violinplot(x="Year", y="Minutes", data=df, inner='quartile')

plt.title('UTMB Results by Year', fontsize=18, fontweight="bold")

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("Year", fontsize=18)

plt.ylabel("Minutes", fontsize=18)

plt.savefig("UTMB ViolinPlot.png")
df['Cat.'].value_counts()
df[df['Cat.'].str.contains('H', regex=False) == False]
df.loc[df['Cat.'].str.contains('H', regex=False) == False, 'Gender'] = 'Female'

df.loc[df['Cat.'].str.contains('H', regex=False) == True, 'Gender'] = 'Male'
df
plt.figure(figsize=(12, 10), dpi=80)

sns.swarmplot(x="Year", y="Minutes", hue='Gender', data=df)

plt.title('UTMB Results by Year', fontsize=18, fontweight="bold")

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("Year", fontsize=18)

plt.ylabel("Minutes", fontsize=18)
df.loc[df['Cat.'].str.contains('E', regex=False) == True, 'Age group'] = '22-39'

df.loc[df['Cat.'].str.contains('1', regex=False) == True, 'Age group'] = '40-49'

df.loc[df['Cat.'].str.contains('2', regex=False) == True, 'Age group'] = '50-59'

df.loc[df['Cat.'].str.contains('3', regex=False) == True, 'Age group'] = '60-69'

df.loc[df['Cat.'].str.contains('4', regex=False) == True, 'Age group'] = '70'
# subset only men's results

men = df.loc[df['Gender'] == 'Male']



# plot violin and swarm plots by age 

plt.figure(figsize=(8, 6), dpi=80)

sns.violinplot(x="Age group", y="Minutes", data=men, color='lightblue', inner='quartile')

plt.title('Mens UTMB Results by Age', fontsize=18, fontweight="bold")

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("Age Groups", fontsize=18)

plt.ylabel("Minutes", fontsize=18)

plt.savefig("UTMB Mens SwarmPlot.png")
# subset only women's results

women = df.loc[df['Gender'] == 'Female']



# plot violin and swarm plots by age categories

plt.figure(figsize=(8, 6), dpi=80)

sns.violinplot(x="Age group", y="Minutes", data=women, color='lightblue', inner='quartile')

sns.swarmplot(x="Age group", y="Minutes", data=women, color='darkblue')

plt.title('Womens UTMB Results by Age', fontsize=18, fontweight="bold")

plt.xticks(fontsize=16)

plt.yticks(fontsize=16)

plt.xlabel("Age Groups", fontsize=18)

plt.ylabel("Minutes", fontsize=18)

plt.savefig("UTMB Womens SwarmPlot.png")
df[df['Gender'] == 'Female']['Age group'].value_counts()
group_times = women['Minutes'].where(women['Age group'] == '22-39').dropna()

group_times
# subset my age group results

group_times = women['Minutes'].where(women['Age group'] == '22-39').dropna()

# 25, 50 and 75 percentiles for total time

np.round(np.percentile(group_times, [25, 50, 75])/ 60, 1)
# 25, 50 and 75 percentiles for calculated per km pace

np.round(np.percentile(group_times, [25, 50, 75]) / 171, 1)
import plotly.graph_objects as go
fig = go.Figure(data=[go.Table(header=dict(values=['Finish','Total time, hours','Pace required, min/km']),

                 cells=dict(values=[['In the top 25%','In the top 50%','In the top 75%'], 

                                    np.round(np.percentile(group_times, [25, 50, 75])/ 60, 1),

                                    np.round(np.percentile(group_times, [25, 50, 75]) / 171, 1)

                                   ]))

                     ])

fig.show()
df['Nationality'].value_counts()
df['Nationality'].value_counts().to_dict().items()
items = df['Nationality'].value_counts().to_dict().items()

# Filtering only those rows where duplicate entries occur more than n

n = 80

nations = df[df['Nationality'].isin([key for key, val in items if val > n])]['Nationality'].value_counts()

nations
nations['rest'] = df[df['Nationality'].isin([key for key, val in items if val < n])]['Nationality'].value_counts().sum()

nations
nations.tolist()
labels = nations.index.tolist()

counts = nations.tolist()

fig1, ax1 = plt.subplots(figsize=(13,13))

ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=30)

ax1.axis('equal')

plt.show()