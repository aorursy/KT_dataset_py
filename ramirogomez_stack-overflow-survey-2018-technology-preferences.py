%matplotlib inline

import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import Counter
from itertools import chain
from matplotlib.ticker import FuncFormatter
from textwrap import shorten

pd.options.display.max_colwidth = 200
mpl.style.use('seaborn-deep')

col_pairs = [
    {'type': 'language', 'cols': ['LanguageWorkedWith', 'LanguageDesireNextYear']},
    {'type': 'database', 'cols': ['DatabaseWorkedWith', 'DatabaseDesireNextYear']},
    {'type': 'platform', 'cols': ['PlatformWorkedWith', 'PlatformDesireNextYear']},
    {'type': 'framework', 'cols': ['FrameworkWorkedWith', 'FrameworkDesireNextYear']}
]
cols = list(chain.from_iterable(pair['cols'] for pair in col_pairs))
df = pd.read_csv(os.path.expanduser('../input/survey_results_public.csv'), usecols=cols, dtype=str)
df.head()
records = []
for row in df.itertuples():
    record = {}
    for pair in col_pairs:
        lists = [getattr(row, col).split(';') for col in pair['cols'] if isinstance(getattr(row, col), str)]
        worked_with = set(lists[0]) if len(lists) > 0 else set()
        desired = set(lists[1]) if len(lists) > 1 else set()
        t = pair['type']
        record.update({
            f'{t}_worked_with': worked_with,
            f'{t}_liked': worked_with & desired,
            f'{t}_new': desired - worked_with
        })
    # Ignore respondents that haven't answered any of the tech questions
    if list(filter(None, record.values())):
        records.append(record)

df_tech = pd.DataFrame(records)
df_tech.head(5)
def plot_axis(ax, series, title):
    labels = [shorten(s, width=30, placeholder='...') for s in series.index]
    y_pos = np.arange(len(series))
    ax.barh(y_pos, series)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, minor=False)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        

def plot_ranking(dataframe, tech, figsize=(18, 12), min_worked_with=1000):
    columns = [f'{tech}_{col}' for col in ['worked_with', 'liked', 'new']]
    
    annotation = '''Graphs 1 & 2 show the percentage of respondents who named the {} relative to how many respondents answered the corresponding question. The 3rd graph shows how many of those who worked 
with the {} want to continue to do so. Data: kaggle.com/stackoverflow/stack-overflow-2018-developer-survey • Author: Ramiro Gómez - ramiro.org'''.format(tech, tech, tech)
    
    # Only count responses with at least one answer to the questions related to this technology
    response_count = len(dataframe[columns].replace(to_replace=set(), value=np.nan).dropna(how='all'))
        
    counters = [Counter(chain.from_iterable(dataframe[col].apply(list))) for col in columns]
    df = pd.DataFrame(counters, index=columns).T.sort_values(f'{tech}_worked_with')
    # Limit to technologies worked with by at least min_worked_with respondents
    df = df[df[f'{tech}_worked_with'] >= min_worked_with]

    worked_with = (df[f'{tech}_worked_with'] / response_count).sort_values()
    liked = (df[f'{tech}_liked'] / df[f'{tech}_worked_with']).sort_values()
    new = (df[f'{tech}_new'] / response_count).sort_values()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    fig.suptitle('Stack Overflow Survey 2018: {} Preferences'.format(tech.title()), y=0.94, size=25)
    fig.subplots_adjust(wspace=0.5)
        
    plot_axis(axes[0], worked_with, 'Worked with')
    plot_axis(axes[1], new, 'Not worked with but desired')
    plot_axis(axes[2], liked, 'Worked with and desired')
    
    plt.annotate(annotation, xy=(5, 30), xycoords='figure pixels', size=12)


plot_ranking(df_tech, 'language')
plot_ranking(df_tech, 'platform')
plot_ranking(df_tech, 'database')
plot_ranking(df_tech, 'framework')