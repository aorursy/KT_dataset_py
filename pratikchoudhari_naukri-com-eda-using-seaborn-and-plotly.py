import os

import sklearn

import itertools

import collections

import numpy as np

import pandas as pd

import seaborn as sns

import datetime as dt

import plotly.express as px

import matplotlib.pyplot as plt

from wordcloud import WordCloud

from sklearn.preprocessing import LabelEncoder



sns.set(font_scale=1.5)

sns.set(rc={'figure.figsize': (11.7, 8.27)})

sns.set_style("whitegrid", {'axes.grid': False})
df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')

df.info()
df.head(2)
#df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')

def clean(df):

    date_le = LabelEncoder()

    df.loc[:, 'Crawl Timestamp'] = df['Crawl Timestamp'].apply(

        lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S +%f").date())

    #df['Job Experience Required'] = date_le.fit_transform(

    #    list(df['Job Experience Required']))

    df['Job Salary'] = df['Job Salary'].str.replace(',', '', regex=False)

    df['Job Salary'] = df['Job Salary'].str.findall(

        '[0-9]{4,}').apply(lambda x: x if x else ['0']).fillna('0')

    df['Job Salary'] = df['Job Salary'].apply(

        lambda x: '-'.join(x) if ((x[0] != '1') and (len(x)==2)) else '0')

    return df, date_le
df, date_le = clean(df)
temp = df['Job Experience Required'].value_counts()[:10][::-1]

px.bar(x=temp.values, y=temp.index, orientation='h', color_discrete_sequence=['forestgreen'],width=800,labels={'x':'Count','y':'Work experience'})
x = collections.Counter(df['Job Salary']).most_common(20)

a, b = list(map(lambda x: str(x[0]), x)), list(map(lambda x: x[1], x))

plt.xticks(rotation=80)

fig = sns.barplot(x=a[1:], y=b[1:], order=a[1:])
# top job titles for

def salary_top_5(salary, col2):

    temp = df.loc[df['Job Salary'] == salary, col2].value_counts()[:5]

    fig = px.pie(names=temp.index, values=temp.values, hole=0.6)

    fig.show()

    return fig
_ = salary_top_5('200000-400000', 'Job Title')
_ = salary_top_5('200000-400000', 'Role')
_ = salary_top_5('200000-400000', 'Role Category')
THRESH = 10

top_loc = df.groupby('Location').size().nlargest(10).index

loc = []

sal = []

c = []

for group in df[df.Location.isin(top_loc)].groupby('Location'):

    a = group[1]['Job Salary'].value_counts()

    a = a[a>THRESH]

    a = a[a.index!='0']

    for ele in a.index:

        loc.append(group[0])

        sal.append(ele)

        c.append(a[ele])

dr = pd.DataFrame({'Location':loc,'Salary Range':sal,'No. of Postings':c})

px.bar(dr,x='Location',y='No. of Postings',color='Salary Range')
roles = df['Role Category'].value_counts()[:10].index

locs = df['Location'].value_counts()[:10].index

df1 = df[(df['Role Category'].isin(roles)) & (df['Location'].isin(locs))].groupby(

    ['Role Category', 'Location']).size().reset_index().sort_values(by=0, ascending=False)

df1 = df1.pivot(index='Role Category', columns='Location', values=0).fillna(0)

df1 = df1.applymap(lambda x: 1000 if x > 1000 else x)

sns.heatmap(df1)
roles = df['Role'].value_counts()[:10].index

locs = df['Location'].value_counts()[:10].index

df1 = df[(df['Role'].isin(roles)) & (df['Location'].isin(locs))].groupby(

    ['Role', 'Location']).size().reset_index().sort_values(by=0, ascending=False)

df1 = df1.pivot(index='Role', columns='Location', values=0).fillna(0)

df1 = df1.applymap(lambda x: 800 if x > 800 else x)

sns.heatmap(df1)
def nested_wc(col, delim, color='white'):

    a = df[col].apply(lambda x: list(

        map(lambda y: y.strip(), x.strip().split(delim))) if type(x) != float else [])

    freq = {}

    for x in a:

        for y in x:

            if y not in freq.keys():

                freq[y] = 1

            else:

                freq[y] += 1

    wc = WordCloud(width=600, height=400, min_font_size=5, max_font_size=150,

                   background_color=color).generate_from_frequencies(freq)

    plt.axis('off')

    plt.imshow(wc)
# functional Area heatmap

nested_wc('Functional Area', ',', '#ff6347')
# key skills heatmap

nested_wc('Key Skills', '|', 'yellow')
# industry heatmap

nested_wc('Industry', ',', 'green')