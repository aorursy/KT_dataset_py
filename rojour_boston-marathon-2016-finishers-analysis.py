import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("whitegrid")

%matplotlib inline



from datetime import datetime

import time
df = pd.read_csv('../input/marathon_results_2016.csv', index_col='Bib')
df.iloc[:, 8:20] = df.iloc[:, 8:20].apply(pd.to_timedelta)
df.describe()
df.info()
g = sns.countplot('Age', data=df, palette="coolwarm")

g.figure.set_size_inches(13,8)

g.set_title("Participants per age group")
g = sns.countplot('Age', data=df, palette="coolwarm", hue='M/F')

g.figure.set_size_inches(13,8)

g.set_title("Participants per Age & Gender")
g = sns.boxplot(df['M/F'], df['Age'], palette="coolwarm")

g.figure.set_size_inches(13,8)

g.set_title("Distribution of finish times per Age group")
g = sns.jointplot( x=df['Official Time'].apply(lambda x: x.total_seconds()/3600), y=df['Age'], stat_func=None, kind='hex', color="r", size=10)
g = sns.boxplot(df['Age'], df['Official Time'].apply(lambda x: x.total_seconds()/3600), palette="coolwarm")

g.figure.set_size_inches(13,8)

g.set_title("Distribution of finish times per Age group")
g = sns.lmplot(x='Overall', y='Age', data=df, hue='M/F', palette="coolwarm", size=10)
s = df['Name'].apply(lambda x: x.split(', '))

df['First Name'] = s.apply(lambda x: x[1])

df['Last Name'] = s.apply(lambda x: x[0])

df.drop('Name', axis=1, inplace=True)
s = df.groupby('Country').count()['City'].sort_values(ascending=False).head(20)

g = sns.barplot(s.index, s, palette='rainbow')

g.figure.set_size_inches(13,8)

g.set_title("Most popular Country")
s = df.groupby('Country').count()['City'].sort_values(ascending=False).head(21)[1:]

g = sns.barplot(s.index, s, palette='rainbow')

g.figure.set_size_inches(13,8)

g.set_title("Most popular Country (after US)")
s = df.groupby('Country').count()['City'].sort_values(ascending=False).head(22)[2:]

g = sns.barplot(s.index, s, palette='rainbow')

g.figure.set_size_inches(13,8)

g.set_title("Most popular Country (after US and Canada)")
s = df[df['City'].notnull()].groupby('City').count()['Country'].sort_values(ascending=False).head(20)

g = sns.barplot(s.index, s, palette='Reds')

g.figure.set_size_inches(13,8)

g.set_title("Most popular City")
s = df[df['Country'] == 'USA'].groupby('State').count()['Country'].sort_values(ascending=False).head(20)

g = sns.barplot(s.index, s, palette="Oranges")

g.figure.set_size_inches(13,8)

g.set_title("Most popular State")
s = df.groupby('Last Name').count()['First Name'].sort_values(ascending=False).head(20)

g = sns.barplot(s.index, s, palette='Blues')

g.figure.set_size_inches(13,8)

g.set_title("Most popular Last Name")
s = df.groupby('First Name').count()['Last Name'].sort_values(ascending=False).head(20)

g = sns.barplot(s.index, s, palette='Greens')

g.figure.set_size_inches(13,8)

g.set_title("Most popular Name")
df['Half_2'] = df['Official Time'] - df['Half']
df['Half'][df['Half'] == '0']
df['2nd_Split'] = (df['Half_2']-df["Half"])
df['2nd_Split']= df['2nd_Split'].apply(lambda x: x.total_seconds()/60)
sns.lmplot(data=df, y='2nd_Split', x='Overall', size=10, markers='.')
df[df['2nd_Split'] == df['2nd_Split'].max()][['5K', '10K','15K', '20K', '25K', '30K', '35K', '40K']]
df[df['2nd_Split'] == df['2nd_Split'].max()]
df[df['2nd_Split'] == df['2nd_Split'].min()][['5K', '10K', '15K', '20K', '25K', '30K', '35K', '40K', 'Division']]
df[df['2nd_Split'] == df['2nd_Split'].min()]
df[df['2nd_Split'] < 0].sort_values(by='2nd_Split')
(len(df[df['2nd_Split'] < 0].sort_values(by='2nd_Split'))/len(df))*100
df[df['2nd_Split']== 0].sort_values(by='2nd_Split')
def fivek_pace(t):

    minute, second = divmod(t.seconds, 60)

    print('%02d:%02d' % (minute, second))
fivek_pace((df['5K'][df['5K']!='0']/3.1).min())

fivek_pace(((df['10K'] -df['5K'])[(df['10K'] -df['5K'])>'0']/3.1).min())

fivek_pace(((df['15K'] -df['10K'])[(df['15K'] -df['10K'])>'0']/3.1).min())

fivek_pace(((df['20K'] -df['15K'])[(df['20K'] -df['15K'])>'0']/3.1).min())

fivek_pace(((df['25K'] -df['20K'])[(df['25K'] -df['20K'])>'0']/3.1).min())

fivek_pace(((df['30K'] -df['25K'])[(df['30K'] -df['25K'])>'0']/3.1).min())

fivek_pace(((df['35K'] -df['30K'])[(df['35K'] -df['30K'])>'0']/3.1).min())

fivek_pace(((df['40K'] -df['35K'])[(df['40K'] -df['35K'])>'0']/3.1).min())
df[(df['5K']=='00:00:00') & (df['10K'] == '00:00:00') & (df['15K'] == '00:00:00') & (df['20K'] == '00:00:00') & (df['25K'] == '00:00:00')]
win = df[df['Division'] == 1].fillna('')

win
me = df[df['Last Name'] == 'Jourdain']
me
K5 = ((me['5K'])/3.1).iloc[0].total_seconds()/60

K10 = ((me['10K'] - me['5K'])/3.1).iloc[0].total_seconds()/60

K15 = ((me['15K'] - me['10K'])/3.1).iloc[0].total_seconds()/60

K20 = ((me['20K'] - me['15K'])/3.1).iloc[0].total_seconds()/60

K25 = ((me['25K'] - me['20K'])/3.1).iloc[0].total_seconds()/60

K30 = ((me['30K'] - me['25K'])/3.1).iloc[0].total_seconds()/60

K35 = ((me['35K'] - me['30K'])/3.1).iloc[0].total_seconds()/60

K40 = ((me['40K'] - me['35K'])/3.1).iloc[0].total_seconds()/60
markers = {' 5K':K5, '10K':K10, '15K':K15, '20K':K20, '25K':K25, '30K':K30, '35K':K35, '40K':K40}
run = pd.DataFrame(markers, index=markers.keys())
g = sns.barplot(run.columns, run.iloc[0], palette='Reds')

g.figure.set_size_inches(12,8)

g.set_ylabel('Pace per mile')

g.set_title('Pace per mile every 5K')