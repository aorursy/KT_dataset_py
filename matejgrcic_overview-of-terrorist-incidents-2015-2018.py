import json 

import pandas as pd 

from pandas.io.json import json_normalize



with open('../input/dataset.json') as f:

    d = json.load(f)

df = json_normalize(data=d, record_path='dataA', meta=['linkTitle'], errors='ignore')
df= df.replace('', 'Unknown')

df.head()
import re

def extract_first_number(value):

    result = re.search('\d+', value)

    if result == None:

        return 0

    return int(result.group())



def remove_suspected(value):

    return value.replace(' (suspected)', '')
df['dead'] = pd.Series(df['dead']).apply(extract_first_number)

df['injured'] = pd.Series(df['injured']).apply(extract_first_number)

df['perpetrator'] = pd.Series(df['perpetrator']).apply(remove_suspected)

df['date'] = pd.Series(df['date']).apply(extract_first_number)

frames = df['linkTitle'].str.split(" ", n = 1, expand = True)

df['month']= frames[0]

df['year']= frames[1]

df = df.drop('linkTitle', axis=1)

df = df.rename(index=str, columns={'dead': 'min_dead', 'injured': 'min_injured', 'partOf':'conflict', 'date': 'start_date' })

df['min_affected'] = df['min_injured'] + df['min_dead']

df.head()
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn-dark-palette')
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)

deaths_per_year = [sum(df[df['year'] == '2015']['min_dead']),sum(df[df['year'] == '2016']['min_dead']), sum(df[df['year'] == '2017']['min_dead']), sum(df[df['year'] == '2018']['min_dead'])]

labels = [2015, 2016, 2017, 2018]

colors = ['yellowgreen', 'lightblue', 'gold', 'lightgreen']

explode = (0.05, 0.05, 0.05, 0.05)

plt.pie(deaths_per_year, explode=explode, labels=labels, 

        colors=colors, autopct='%4.2f%%',shadow=True, startangle=20)

plt.title('Deaths by year')

plt.axis('equal')

plt.subplot(1, 2, 2)

injured_per_year = [sum(df[df['year'] == '2015']['min_injured']),sum(df[df['year'] == '2016']['min_injured']), sum(df[df['year'] == '2017']['min_injured']), sum(df[df['year'] == '2018']['min_injured'])]

labels = [2015, 2016, 2017, 2018]

colors = ['yellowgreen', 'lightblue', 'gold', 'lightgreen']

explode = (0.05, 0.05, 0.05, 0.05)

plt.pie(injured_per_year, explode=explode, labels=labels, 

        colors=colors, autopct='%4.2f%%',shadow=True, startangle=20)

plt.title('Injuries by year')

plt.axis('equal')

plt.show()
sns.set_style("whitegrid")

plt.figure(figsize=(8, 6))

sns.barplot(x=list(df.groupby('year').groups.keys()), y=df.groupby('year').sum()['min_affected'])

plt.title('Persons affected by terrorist attacks per year')

plt.ylabel('People affected')

plt.xlabel('Year')

plt.show()
ax = df[df.perpetrator != 'Unknown']['perpetrator'].value_counts().head(10).plot.bar(figsize=(8, 6))

ax.set_title('Top 10 perpetrators')

plt.show()
ax = df['type'].value_counts().head(10).plot.bar(figsize=(8, 6))

ax.set_title('Top 10 types of attack')

ax.set_ylabel('Number of attacks')

plt.show()
ax = df[df.location != 'Unknown']['location'].value_counts().head(10).plot.bar(figsize=(8, 6))

ax.set_title('Most critical areas of conflict')

ax.set_ylabel('Number of attacks')

plt.show()
plt.figure(figsize=(8, 6))

sns.lineplot(x=list(df.groupby('start_date').groups.keys()), y=df.groupby('start_date').sum()['min_affected'])

plt.xlabel('Day in month')

plt.ylabel('Persons affected by attack')

plt.show()
deadliest_types = list(df['type'].value_counts().head(10).keys())

data = df[df.type.isin(deadliest_types)]

data = data.drop(['min_affected', 'start_date'], axis=1)

ax = data.groupby('type').sum().plot.bar(figsize=(10, 6))

ax.set_xlabel('Attack type')

plt.show()
from sklearn.cluster import KMeans

inertia = []

X = df[['min_dead', 'min_injured']].values

for n_clusters in range(1, 100, 5): 

    model = KMeans(n_clusters=n_clusters)

    model.fit(X)

    inertia.append(model.inertia_)
plt.figure(figsize=(8, 6))

plt.plot(range(1, 100, 5), inertia, label='Number of clusters')

plt.title('Elbow method')

plt.legend()

plt.show()
model = KMeans(n_clusters=5)

model.fit(X)

y = model.predict(X)

plt.figure(figsize=(15, 15))

for i in range(5):

    plt.scatter(X[y == i, 0], X[y == i, 1])

plt.show()
print(f"There were at least {df['min_dead'].sum()} deaths caused by terrorist attacks between January 2015 - December 2018.")