%config InlineBackend.figure_format = 'retina'



from operator import itemgetter

from pathlib import Path

from pdb import set_trace

from textwrap import wrap



import matplotlib.pyplot as plt

import matplotlib.patches as patches

import numpy as np

import pandas as pd





class VisualStyle:

    """Convenience wrapper on top of matplotlib config."""



    def __init__(self, config, default=None):

        if default is None:

            default = plt.rcParams

        self.default = default.copy()

        self.config = config



    def replace(self):

        plt.rcParams = self.config



    def override(self, extra=None):

        plt.rcParams.update(self.config)

        if extra is not None:

            plt.rcParams.update(extra)



    def restore(self):

        plt.rcParams = self.default



    def __enter__(self):

        self.override()

        return self



    def __exit__(self, exc_type, exc_val, exc_tb):

        self.restore()



        

palette1 = '173f5f 20639b 3caea3 f6d55c ed553b'

palette2 = '264653 2a9d8f e9c46a f4a261 e76f51'

tableau = '4e79a7 f28e2b e15759 76b7b2 59a14f edc948 b07aa1 ff9da7 9c755f bab0ac'





def make_cycler(colors):

    from cycler import cycler

    colors_str = ', '.join([f"'{c}'" for c in colors.split()])

    return f"cycler('color', [{colors_str}])"

    

    

class NotebookStyle(VisualStyle):

    def __init__(self):

        super().__init__({

            'figure.figsize': (8, 6),

            'figure.titlesize': 20,

            'font.family': 'monospace',

            'font.monospace': 'Liberation Mono',

            'axes.titlesize': 18,

            'axes.labelsize': 16,

            'axes.spines.right': False,

            'axes.spines.top': False,

            'xtick.labelsize': 14,

            'ytick.labelsize': 14,

            'font.size': 14,

            'axes.prop_cycle': make_cycler(tableau)

        })



        

def show_all(df):

    with pd.option_context('display.max_columns', None, 'display.max_rows', None):

        display(df)

        

        

DATA = Path('/kaggle/input/kaggle-survey-2019/')

SCHEMA = DATA/'survey_schema.csv'

QUESTIONS = DATA/'questions_only.csv'

MULTIPLE = DATA/'multiple_choice_responses.csv'

OTHER = DATA/'other_text_responses.csv'



assert all(p.exists() for p in (SCHEMA, QUESTIONS, MULTIPLE, OTHER))



schema_df = pd.read_csv(SCHEMA)

questions_df = pd.read_csv(QUESTIONS)

multi_df = pd.read_csv(MULTIPLE)

other_df = pd.read_csv(OTHER)



style = NotebookStyle()

style.override()



questions = multi_df.iloc[0]

answers = multi_df.iloc[1:]

print(questions['Q4'])

answers['Q4'].value_counts()

degree_counts = answers['Q4'].fillna('I prefer not to answer').value_counts().reset_index().rename(columns={'index': 'degree', 'Q4': 'count'})

degree_counts['degree'] = degree_counts['degree'].map({

    'Master’s degree': 'MS',

    'Bachelor’s degree': 'BS',

    'Doctoral degree': 'PhD',

    'Some college/university study without earning a bachelor’s degree': 'College/University',

    'Professional degree': 'Professional',

    'I prefer not to answer': 'No answer',

    'No formal education past high school': 'High School'

})

degree_counts['higher_education'] = degree_counts['degree'].isin({'MS', 'BS', 'PhD', 'College/University'}).map({True: 'yes', False: 'no'})

# degree_counts
def plot_yes_no_chart(df, figsize=(12, 4)):

    def create_label(value, dataset):

        return f'{value:2.2f}%'

    f, ax = plt.subplots(1, 2, figsize=figsize)

    ax.flat[0].axis('off')

    ax = ax.flat[-1]

    ax.set_aspect('equal')

    higher_education = degree_counts.groupby('higher_education').sum()

    total = higher_education['count'].sum()

    y_cnt = higher_education.loc['yes']['count']

    n_cnt = higher_education.loc['no']['count']

    _, _, autotexts = ax.pie(

        [y_cnt, n_cnt], 

        labels=['Yes', 'No'], explode=[0, 0.1],

        autopct=lambda x: create_label(x, higher_education), 

        wedgeprops=dict(edgecolor='w'),

        textprops=dict(color="black", size=13, weight='bold'))

    for autotext in autotexts:

        autotext.set_color('white')

    ax.set_title('Higher Education?')

    ax.set_xlabel(f'among {total} respondents in total')

    return ax



ax = plot_yes_no_chart(degree_counts, figsize=(8, 4));
def hex2rgba(hex_value: str) -> tuple:

    n = len(hex_value)

    if n == 6:

        r, g, b = hex_value[:2], hex_value[2:4], hex_value[4:]

        a = 'ff'

    elif n == 8:

        r, g, b, a = [hex_value[i:i+2] for i in (0, 2, 4, 6)]

    else:

        raise ValueError(f'wrong hex string: {hex_value}')

    rgba = tuple(int(value, 16)/255. for value in (r, g, b, a))

    return rgba



def make_colors(base, size):

    from itertools import islice, cycle

    colors = list(islice(cycle([hex2rgba(x) for x in base.split()]), None, size))

    return colors



def plot_bars_with_percentage(dataframe, xcol, ycol, ax, 

                              xlabel='Number of Respondents', 

                              ylabel='',

                              colors=None):

    

    colors = colors if colors is not None else make_colors(tableau, len(dataframe))



    def generate_bars(df, ax=None):

        ax = dataframe.plot.barh(x=xcol, y=ycol, ax=ax, color=colors)

        ax.set_xlabel(xlabel)

        ax.set_ylabel(ylabel)

        ax.set_axisbelow(True)

        ax.grid(True)

        ax.get_legend().remove()

        return ax

    

    def generate_percentage_annotations(df, ax):

        total = df[ycol].sum()

        for i, count in enumerate(df[ycol]):

            ax.text(

                count + 200, i, f'{count/total:2.2%}', fontsize=12, 

                verticalalignment='center', 

                horizontalalignment='left',

                bbox=dict(facecolor='white', edgecolor='black'))

        return ax

    

    def adjust_limits(ax):

        x_min, x_max = ax.get_xlim()

        x_max *= 1.1

        ax.set_xlim(x_min, x_max)

        return ax

    

    ax = generate_bars(dataframe, ax=ax)

    ax = generate_percentage_annotations(dataframe, ax=ax)

    ax = adjust_limits(ax)

    return ax



def plot_pie_chart(df, value_col, label_col, ax, pct=False, colors=None, pctdistance=0.6):

    def autopct(value): return f'{value:2.2f}%'

    rel_counts = df[value_col]/df[value_col].sum()

    explode = [0.01] * len(rel_counts)

    params = dict(

        x=rel_counts, labels=df[label_col], 

        pctdistance=pctdistance,

        colors=colors, wedgeprops=dict(width=0.5, edgecolor='w'),

        textprops=dict(size=16))

    if pct:

        params['autopct'] = autopct

    ax.pie(**params)

    return ax



def plot_education_level(dataframe, figsize=(10, 8)):

    f, axes = plt.subplots(1, 2, figsize=figsize)

    f.suptitle('Survey Participants Education')

    ax1, ax2 = axes.flat

    ax1 = plot_bars_with_percentage(dataframe, ax=ax1, xcol='degree', ycol='count')

    ax2 = plot_pie_chart(dataframe, ax=ax2, value_col='count', label_col='degree')

    return f



plot_education_level(degree_counts, figsize=(12, 6));
# print(questions['Q5'])

# print(questions['Q5_OTHER_TEXT'])



scientific = ['Statistician', 'Research Scientist', 'Data Scientist']

engineering = ['DBA/Database Engineer', 'Data Engineer', 'Software Engineer']

business = ['Business Analyst', 'Data Analyst', 'Product/Project Manager']

other = ['Other', 'Not employed', 'Student']



job_titles = answers['Q5'].fillna('Other').value_counts().reset_index().rename(columns={'Q5': 'count', 'index': 'job_title'})



job_titles['area'] = job_titles['job_title'].map(

    lambda x:

    'Science' if x in scientific else 

    'Engineering' if x in engineering else

    'Business' if x in business else

    'Other')



colors_set = [hex2rgba(x) for x in tableau.split()[:4]]



from collections import OrderedDict

color_map = OrderedDict(zip(sorted(job_titles['area'].unique()), colors_set))



job_titles = job_titles.sort_values(by='area')



areas = job_titles.groupby('area').sum().reset_index()



colors = job_titles['area'].map(color_map)



areas = areas.sort_values(by='area')
f, axes = plt.subplots(1, 2, figsize=(12, 7))

ax1, ax2 = axes.flat

plot_bars_with_percentage(job_titles, ax=ax1, xcol='job_title', ycol='count', colors=colors)

plot_pie_chart(areas, ax=ax2, label_col='area', value_col='count', colors=color_map.values(), pctdistance=0.75, pct=True)

f.suptitle('Job Titles Distribution');
from collections import defaultdict

from spacy.lang.en import STOP_WORDS

from wordcloud import WordCloud



import matplotlib.pyplot as plt

import matplotlib.colors as colors

import numpy as np



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):

    new_cmap = colors.LinearSegmentedColormap.from_list(

        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),

        cmap(np.linspace(minval, maxval, n)))

    return new_cmap



q5_texts = other_df['Q5_OTHER_TEXT'].iloc[1:]

q5_texts = q5_texts[~q5_texts.isna()]

titles = [word.strip().lower() for text in q5_texts.unique() for word in text.split()]



cmap = truncate_colormap(plt.get_cmap('Blues'), 0.4, 0.9)

cloud = WordCloud(stopwords=STOP_WORDS, 

                  width=1400, height=800, 

                  colormap=cmap,

                  background_color='white')

image = cloud.generate(' '.join(titles))

f, ax = plt.subplots(1, 1, figsize=(14, 8))

ax.imshow(image)

ax.axis('off');
import textwrap

import matplotlib.pyplot as plt

import matplotlib.patches as patches

import matplotlib.font_manager as font_manager

from matplotlib.pyplot import get_cmap



def create_axes_if_needed(ax, **fig_kwargs):

    if ax is None:

        f = plt.figure(**fig_kwargs)

        ax = f.add_subplot(111)

    else:

        f = ax.figure

    return f, ax



def plot_heatmap(data, xticks=None, yticks=None, cm='Reds', 

                 annots=None, font_size=20, tick_font_size=14,

                 ticks_wrap=15, annot_threshold=0.5,

                 fmt='{:.2f}', background_colors=None,

                 annot_light='#ffffff', annot_dark='#000000',

                 tight_layout=True, ax=None, **fig_kwargs):

    

    f, ax = create_axes_if_needed(ax, **fig_kwargs)

    n, m = data.shape

    w = h = 1

    ax.set_xlim(left=0, right=m)

    ax.set_ylim(bottom=0, top=n)

    cmap = get_cmap(cm) if isinstance(cm, str) else cm

    values_font = font_manager.FontProperties(size=font_size)

    

    for i in range(m):

        for j in range(n):

            x, y = i*w, j*h

            value = data[n - j - 1, i]

            if background_colors is None:

                color = cmap(value)

            else:

                color = background_colors[n - j - 1, i]

            rect = patches.Rectangle((x, y), w, h, color=color)

            ax.add_patch(rect)

            if annots is not None:

                annot = annots[n - j - 1, i]

            else:

                annot = fmt.format(value)

            annot_color = annot_light if value >= annot_threshold else annot_dark

            ax.annotate(annot, xy=(x + w/2, y + h/2), 

                        va='center', ha='center',

                        color=annot_color, fontproperties=values_font)

            

    xtick_offset, ytick_offset = w/2, w/2

    ax.set_xticks([xtick_offset + i * w for i in range(m)])

    ax.set_yticks([ytick_offset + i * h for i in range(n)])

    ax.xaxis.tick_top()

    

    if xticks is None and yticks is None:

        ax.set_xticks([])

        ax.set_yticks([])

    else:

        fontsize = tick_font_size

        if xticks is not None:            

            if ticks_wrap is not None:

                xticks = [

                    '\n'.join(textwrap.wrap(name, width=ticks_wrap))

                    for name in xticks]

            ax.set_xticklabels(xticks, fontsize=fontsize, rotation=45)

        if yticks is not None:

            if ticks_wrap is not None:

                yticks = [

                    '\n'.join(textwrap.wrap(name, width=ticks_wrap))

                    for name in yticks]

            ax.set_yticklabels(reversed(yticks), fontsize=fontsize)

                    

    if tight_layout:

        ax.figure.tight_layout()

    

    for name in ('left', 'top', 'right', 'bottom'):

        ax.spines[name].set_visible(True)

    return ax



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):

    new_cmap = colors.LinearSegmentedColormap.from_list(

        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),

        cmap(np.linspace(minval, maxval, n)))

    return new_cmap



scientific = ['Statistician', 'Research Scientist', 'Data Scientist']

engineering = ['DBA/Database Engineer', 'Data Engineer', 'Software Engineer']

business = ['Business Analyst', 'Data Analyst', 'Product/Project Manager']

other = ['Other', 'Not employed', 'Student']

labels = ['No answer', 'High School', 'Professional', 'College/University', 'BS', 'MS', 'PhD']

df = answers[['Q4', 'Q5']].copy().rename(columns={'Q4': 'degree', 'Q5': 'occupation'})

df['degree'] = df['degree'].fillna('I prefer not to answer').map({

    'Master’s degree': 'MS',

    'Bachelor’s degree': 'BS',

    'Doctoral degree': 'PhD',

    'Some college/university study without earning a bachelor’s degree': 'College/University',

    'Professional degree': 'Professional',

    'I prefer not to answer': 'No answer',

    'No formal education past high school': 'High School'

})

df['degree'] = pd.Categorical(df['degree'], categories=labels, ordered=True)

df['higher_education'] = df['degree'].isin({'MS', 'BS', 'PhD', 'College/University'}).map({True: 'yes', False: 'no'})

df['occupation'] = df['occupation'].fillna('Other')

df['area'] = df['occupation'].map(

    lambda x:

    'Science' if x in scientific else 

    'Engineering' if x in engineering else

    'Business' if x in business else

    'Other'

)



ct = pd.crosstab(df['degree'], df['occupation'])



color_map = truncate_colormap(get_cmap('Reds'), minval=0.1, maxval=0.9)



values = np.interp(ct.values.ravel(), (ct.min().min(), ct.max().max()), (0, 1)).ravel()

backgrounds = np.array([color_map(v) for v in values])

values = values.reshape(ct.shape)

backgrounds = backgrounds.reshape(ct.shape + (4,))



plot_heatmap(ct.values, cm=color_map, background_colors=backgrounds,

             xticks=ct.columns.tolist(), yticks=ct.index.tolist(), 

             fmt='{:d}', annot_threshold=600, figsize=(12, 8), ticks_wrap=12);
import seaborn as sns

degree_area = df.groupby(['degree', 'area']).count().occupation.reset_index().rename(columns={'occupation': 'count'})

g = sns.catplot(

    x="count", y="degree", col="area",

    data=degree_area, kind="bar", height=5, aspect=0.8)

g.fig.suptitle('Frequency Histograms of Degrees Depending on Area', y=1.1)

for ax in g.axes.flat:

    ax.grid(True, linestyle='--')

    ax.set_axisbelow(True)

    ax.set_xlabel('')

    ax.set_ylabel('')
def parse_salary_range(x):

    if x != 'n/a':

        if '>' in x:

            start = int(x.strip('>').strip().strip('$').replace(',', ''))

            end = float('inf')

        elif '<' in x:

            start = 0

            end = int(x.strip('<').strip().strip('$').replace(',', ''))

        else:

            start, end = [int(part.strip('$').replace(',', '')) for part in x.split('-')]

        return start, end

    return -1, -1

    

df['salary'] = answers['Q10'].fillna('n/a')

df['has_salary_range'] = df['salary'].map(lambda x: 'yes' if x != 'n/a' else 'no')

df = pd.concat([df, df['salary'].map(parse_salary_range).apply(pd.Series).rename(columns={0: 'salary_from', 1: 'salary_to'})], axis=1)



salary_df = df.query('occupation not in ("Student", "Not employed")').groupby(['occupation', 'has_salary_range']).count().salary.reset_index()

g = sns.catplot(

    x="salary", y="occupation", col="has_salary_range",

    data=salary_df, kind="bar", height=5, aspect=1.4)

g.fig.suptitle('Occupation: Has a Respondent Reported The Salary Level?', y=1.1)

ax1, ax2 = g.axes.flat

for ax in (ax1, ax2):

    ax.grid(True, linestyle='--')

    ax.set_axisbelow(True)

    ax.set_xlabel('')

    ax.set_ylabel('')

ax1.set_title('No')

ax2.set_title('Yes')

n_total = salary_df.query("has_salary_range == 'no'").salary.sum()

y_total = salary_df.query("has_salary_range == 'yes'").salary.sum()

ax1.set_xlabel(f'Total: {n_total}');

ax2.set_xlabel(f'Total: {y_total}');
salary_df = df.query('occupation not in ("Student", "Not employed")').groupby(['degree', 'has_salary_range']).count().salary.reset_index()

g = sns.catplot(

    x="salary", y="degree", col="has_salary_range",

    data=salary_df, kind="bar", height=5, aspect=1.3)

g.fig.suptitle('Education Level: Has a Respondent Reported The Salary Level?', y=1.1)

ax1, ax2 = g.axes.flat

for ax in (ax1, ax2):

    ax.grid(True, linestyle='--')

    ax.set_axisbelow(True)

    ax.set_xlabel('')

    ax.set_ylabel('')

ax1.set_title('No')

ax2.set_title('Yes')

n_total = salary_df.query("has_salary_range == 'no'").salary.sum()

y_total = salary_df.query("has_salary_range == 'yes'").salary.sum()

ax1.set_xlabel(f'Total: {n_total}');

ax2.set_xlabel(f'Total: {y_total}');
from functools import partial

def q1(x): return np.quantile(x, q=.25)

def q2(x): return np.quantile(x, q=.50)

def q3(x): return np.quantile(x, q=.75)



def iqr_filter(data, group_col, salary_from='salary_from', salary_to='salary_to', var_name='Bound'):

    data = data[[group_col, salary_from, salary_to]].copy()

    data = data[~data[salary_to].map(np.isinf)]

    data = pd.melt(data, id_vars=[group_col], value_vars=[salary_from, salary_to], var_name=var_name)

    data[var_name] = data[var_name].map({salary_from: 'Lower', salary_to: 'Upper'})

    iqr = data.groupby([group_col, var_name]).aggregate(

        Q1=('value', q1),

        Q2=('value', q2),

        Q3=('value', q3),

    ).assign(

        IQR=lambda dataset: dataset['Q3'] - dataset['Q1'],

        IQR_lower=lambda dataset: dataset['Q1'] - 1.5*dataset['IQR'],

        IQR_upper=lambda dataset: dataset['Q3'] + 1.5*dataset['IQR']

    ).reset_index()

    joined = pd.merge(data, iqr, on=[group_col, var_name])

    return joined.query('IQR_lower <= value <= IQR_upper').reset_index()



def boxplot(data, group_col, var_name='Bound', title='Boxplot', figsize=(12, 8), text_wrap=None, ax=None):

    f, ax = create_axes_if_needed(ax, figsize=figsize)

    data = iqr_filter(data, group_col, var_name=var_name)[[group_col, var_name, 'value']]

    sns.boxplot(x='value', y=group_col, hue=var_name, data=data, ax=ax)

    ax.set_ylabel('')

    ax.set_xlabel('Amount in USD')

    ax.grid(True, linestyle='dotted')

    ax.set_axisbelow(True)

    ax.set_title(title)

    if text_wrap is not None:

        ax.set_yticklabels([

            '\n'.join(textwrap.wrap(t.get_text(), width=text_wrap))

            for t in ax.get_yticklabels()])

    return ax



salary_data = df.query('has_salary_range == "yes"')



boxplot(data=salary_data, group_col='degree', title='Salary vs. Education', figsize=(10, 6));
boxplot(data=salary_data, group_col='occupation', title='Salary vs. Occupation', text_wrap=18, figsize=(10, 8));
table = salary_data.groupby(['degree', 'occupation']).aggregate(

    salary_from_median=('salary_from', 'median'),

    salary_to_median=('salary_to', 'median')

).reset_index()

table_from = table.drop(columns=['salary_to_median']).pivot(index='degree', columns='occupation')

xticks = [x for _, x in table_from.columns.tolist()]

yticks = table_from.index.tolist()

color_map = truncate_colormap(get_cmap('Blues'), minval=0.1, maxval=0.9)

values = table_from.values.astype(float)

values_norm = values / values.max().max()

annots = np.array(['${:d}k'.format(int(v/1000)) for v in values.ravel()]).reshape(values_norm.shape)

backgrounds = np.array([color_map(v) for v in values_norm.ravel()]).reshape(values_norm.shape + (4,))

plot_heatmap(values, cm=color_map, background_colors=backgrounds,

             annots=annots, xticks=xticks, yticks=yticks, fmt='{:.2f}', 

             annot_threshold=30000, annot_light='#ffffff', 

             annot_dark='#000000', figsize=(12, 8), ticks_wrap=12);
table = salary_data.groupby(['degree', 'occupation']).aggregate(

    salary_from_median=('salary_from', 'median'),

    salary_to_median=('salary_to', 'median')

).reset_index()

table_from = table.drop(columns=['salary_from_median']).pivot(index='degree', columns='occupation')

xticks = [x for _, x in table_from.columns.tolist()]

yticks = table_from.index.tolist()

color_map = truncate_colormap(get_cmap('Reds'), minval=0.1, maxval=0.9)

values = table_from.values.astype(float)

values_norm = values / values.max().max()

annots = np.array(['${:d}k'.format(int(v/1000)) for v in values.ravel()]).reshape(values_norm.shape)

backgrounds = np.array([color_map(v) for v in values_norm.ravel()]).reshape(values_norm.shape + (4,))

plot_heatmap(values, cm=color_map, background_colors=backgrounds,

             annots=annots, xticks=xticks, yticks=yticks, fmt='{:.2f}', 

             annot_threshold=30000, annot_light='#ffffff', 

             annot_dark='#000000', figsize=(12, 8), ticks_wrap=12);
import matplotlib.ticker as ticker

stats = iqr_filter(salary_data, group_col='degree')

stats_long = pd.melt(stats, value_vars=['Q1', 'Q2', 'Q3'], id_vars=['degree'], var_name='Quantile')

f, ax = plt.subplots(1, 1, figsize=(12, 6))

sns.pointplot(x='degree', y='value', hue='Quantile', data=stats_long, ax=ax)

ax.set_xticklabels(ax.get_xticklabels(), rotation=15)

formatter = ticker.StrMethodFormatter('${x:,.0f}')

ax.yaxis.set_major_formatter(formatter)

ax.set_xlabel('')

ax.set_ylabel('')

ax.set_axisbelow(True)

ax.grid(True, linestyle='dotted')

ax.set_title('Salary Quantiles For Degree Levels');