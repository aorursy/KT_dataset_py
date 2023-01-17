# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os

import re

import datetime

import json

import plotly.graph_objects as go
with open('../input/arxiv/arxiv-metadata-oai-snapshot-2020-08-14.json', 'r') as f:

    raw_line = f.readline()

    line = json.loads(raw_line)

    

for k, v in line.items():

    print(f'{k}: {v}\n')
line.keys()
COLUMNS = ['id', 'title', 'authors_parsed', 'comments', 'categories', 'abstract']





def get_metadata():

    # Make an iterator to avoid excessive loading of data in the memory. Borrowed the idea from https://www.kaggle.com/artgor/arxiv-metadata-exploration

    with open('../input/arxiv/arxiv-metadata-oai-snapshot-2020-08-14.json', 'r') as f:

        for line in f:

            yield line

            

def preprocess_line(line, cols=COLUMNS):

    parsed_line = json.loads(line)

    data = dict()

    for col in cols:

        data[col] = parsed_line[col]

        

    # We also extract the number of authors

    

    data['number_authors'] = len(parsed_line['authors_parsed'])

    

    # And the upload date

    

    data['upload_date'] = datetime.datetime.strptime(parsed_line['versions'][0]['created'], "%a, %d %b %Y %H:%M:%S %Z").date() 

    

    return data



for k, v in preprocess_line(raw_line).items():

    print(f'{k}: {v}\n')

    
from tqdm import tqdm

metadata = get_metadata()



data = []



for ln in tqdm(metadata):

    data.append(preprocess_line(ln))
df = pd.DataFrame(data)

df.head()

del data
df.groupby('upload_date').count().describe().id
print(df.groupby('upload_date').count().id.idxmax())
daily_grouping = df.groupby('upload_date').count()

daily_grouping.index = pd.to_datetime(daily_grouping.index)

monthly_grouping = daily_grouping.groupby(pd.Grouper(freq='M')).sum()

monthly_grouping = monthly_grouping.loc["1991-09-01":"2020-08-01"] 

# This is just to make sure that we actually have a complete month's worth of data. First publications on the Arxiv were uploaded on the 14th of August (there are some papers 

# by the founder with earlier submission dates though).
fig = go.Figure()



fig.add_trace(go.Scatter(x=monthly_grouping.index,

                         y=monthly_grouping.id,

                         name='Number of publications'

                        )

              )



fig.add_trace(go.Scatter(x=monthly_grouping.index, 

                         y=monthly_grouping.id.rolling(12, min_periods=1).mean(),

                         name='12 month rolling average'

                        )

             )



fig.update_layout(template='plotly', 

                  title='Number of monthly submissions',

#                  plot_bgcolor='rgba(0,0,0,0)', 

#                  paper_bgcolor='rgba(0,0,0,0)', 

                  margin=dict(l=40, r=40, t=40, b=40),

                  legend=dict(

                        yanchor="top",

                        y=0.99,

                        xanchor="left",

                        x=0.01

                    ))



fig.show()
lfy_total = monthly_grouping["2014-01-01":"2020-01-01"].groupby(pd.Grouper(freq='Y')).sum().id
lfy_fig = go.Figure()



for date in lfy_total.index:

    year = date.year

    pubs = monthly_grouping.loc[monthly_grouping.index.year == year].id/lfy_total[date]

     

    lfy_fig.add_trace(go.Scatter(x=pubs.index.month,

                                y=pubs.values,

                                name=year))

    



lfy_fig.update_layout(

    yaxis_tickformat=".3f",

    title='Submissions per month as a proportion of the yearly total',

    xaxis=dict(

        tickvals=pubs.index.month,

        ticktext=pubs.index.strftime("%B")),

    yaxis=dict(tickvals=[0.080,0.085, 0.090, 0.095, 0.01])

    

)



lfy_fig.show()
df.describe()
df.loc[df.number_authors.idxmax()]
author_num=df.groupby(['number_authors']).count().id

OTHER = author_num.loc[author_num.index > 7].sum()

author_num.loc[8] = OTHER

author_num = author_num[:8]

author_num /= author_num.sum()

author_num
author_fig = go.Figure()

x_labels = ["one", "two", "three", "four", "five", "six", "seven", "eight or more"] 



author_fig.add_trace(go.Bar(x=x_labels, 

                            y=author_num, 

                            text=author_num.apply(lambda x: f'{round(100 * x, 2)}%'),

                            textposition='auto'

                           )

                    )



author_fig.update_layout(template='plotly', 

                  title='Percentage of submissions with a given number of authors',

                  margin=dict(l=40, r=40, t=40, b=40), 

                  yaxis_tickformat='%'

)



author_fig.show()
df.number_authors = df.number_authors.apply(lambda x: x if x < 8 else 8)



df["upload_year"]=df.upload_date.apply(lambda x: x.year)

yearly_author_no=df.groupby(['upload_year', 'number_authors']).size()

#print(yearly_author_no.loc[1991].sum())



traces = {i:[] for i in range(1,9)}



for author_no in traces.keys():

    for year in range(1991, 2020):

        try:

            total = yearly_author_no.loc[year].sum()

            traces[author_no].append(yearly_author_no.loc[(year, author_no)] / total )

        except KeyError:

            traces[author_no].append(0)
yearly_fig = go.Figure()

years = [i for i in range(1991, 2020)]



for number in traces.keys():

    

    yearly_fig.add_trace(go.Scatter(x=years,

                                    y=traces[number],

                                    name=f'{number} authors' if number != 8 else '8+ authors'

                                   ))





yearly_fig.update_layout(template='plotly', 

                  title='Proportion of papers with a given number of authors published per year',

                  margin=dict(l=40, r=40, t=40, b=40),  

                  yaxis_tickformat = '%',

                  legend=dict(

                        yanchor="top",

                        y=0.99,

                        xanchor="right",

                        x=0.98)

)



yearly_fig.show()

PATTERN = r'([0-9]+) ?page'

df.comments.str.contains(PATTERN).sum()/df.shape[0]
page_count = df.comments.str.extract(pat = PATTERN)

#page_count.loc[~ page_count[1].isna(),0] = page_count[~ page_count[1].isna()][1]

df['number_pages'] = page_count[0].apply(lambda x: x if pd.isna(x) else int(x)) 

df.head()
df.describe().number_pages
df.loc[df.number_pages.idxmax()]
pages_fig = go.Figure()



pages_fig.add_trace(go.Histogram(x=df.number_pages, xbins=dict(end=100, size=1, start=1)))



pages_fig.update_layout(template='plotly', 

                  title='Page counts of Arxiv submissions',

                  margin=dict(l=40, r=40, t=40, b=40)

)

pages_fig.show()