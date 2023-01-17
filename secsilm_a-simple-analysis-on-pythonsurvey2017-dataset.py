import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import iplot, init_notebook_mode
%matplotlib inline
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (20, 6)
survey_df = pd.read_csv('../input/pythondevsurvey2017_raw_data.csv')
survey_df.columns = [c.lower() for c in survey_df.columns]
survey_df.head()
survey_df.shape
def find_cols(df, kws):
    '''找到 df 中含有 kws 的列'''
    return [item for item in df.columns if all ([w in item for w in kws])]
find_cols(df=survey_df, kws=['python', 'version'])
python_version = survey_df['which version of python do you use the most?']
python_version.describe()
python_version.value_counts(normalize=True, dropna=False)
python_version.value_counts(normalize=False, dropna=False)
python_version.value_counts(normalize=True, dropna=True)
python_version.value_counts(normalize=True, dropna=True).plot(kind='pie', 
                                                              figsize=(5, 5), 
                                                              startangle=90, 
                                                              autopct='%.0f%%', 
                                                              fontsize=14,
                                                              colors=sns.color_palette('rainbow')[:2])
plt.title('Python 2 VS Python 3', fontsize=18)
plt.ylabel('');
plt.tight_layout()
plt.savefig('python-version.png')
python_da_ml = survey_df[['machine learning:\xa0what do you use python for?', 'data analysis:\xa0what do you use python for?', 'which version of python do you use the most?']]
python_da_ml.dtypes
python_da = pd.crosstab(python_da_ml['which version of python do you use the most?'], python_da_ml['data analysis:\xa0what do you use python for?'], normalize=True)
python_da
python_ml = pd.crosstab(python_da_ml['which version of python do you use the most?'], python_da_ml['machine learning:\xa0what do you use python for?'], normalize=True)
pd.concat([python_da, python_ml], axis=1)
pd.concat([python_da, python_ml], axis=1).T.plot(kind='bar', figsize=(10, 5), color=sns.color_palette('rainbow'))
plt.xticks(rotation=0, fontsize=14)
plt.title('Data Analysis and Machine Learning VS Python version', fontsize=18)
plt.legend(title=None)
plt.tight_layout()
plt.savefig('data-analysis-machine-learning-vs-python-version.png')
cols = find_cols(survey_df, 'what framework(s) do you use in addition to python?')
cols
frameworks = survey_df[cols[1:]]
frameworks.head()
count_df = frameworks.count().sort_values(ascending=False)
count_df.index = [item.split(':')[0] for item in count_df.index]

count_df.plot(kind='bar', color=sns.color_palette('rainbow', frameworks.shape[1]))
plt.xticks(fontsize=14)
values = frameworks.count().sort_values(ascending=False).values
labels = [item.split(':')[0] for item in frameworks.count().sort_values(ascending=False).index]

plt.figure(figsize=(20, 17))
sns.barplot(x=values, y=labels, orient='h', palette=sns.color_palette("rainbow", 24))
plt.xticks(fontsize=14)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig('frameworks.png')
python_ver = survey_df['which version of python do you use the most?']
def process_col(col):
    return pd.crosstab(index=python_ver, columns=col).iloc[:, 0]
# process_col(frameworks['django:what framework(s) do you use in addition to python?'])
frameworks_pyver = frameworks.apply(lambda col: pd.crosstab(index=python_ver, columns=col).iloc[:, 0])
frameworks_pyver.columns = [item.split(':')[0] for item in frameworks.columns]
frameworks_pyver
frameworks_pyver_ratio = frameworks_pyver / frameworks_pyver.sum(axis=0)
frameworks_pyver_ratio.T.plot(kind='bar', color=sns.color_palette('rainbow'))
plt.xticks(rotation=90, fontsize=14)
df = frameworks_pyver_ratio.stack().reset_index()
df.columns=['pyver', 'framework', 'value']
df.head()
plt.figure(figsize=(20, 17))
sns.barplot(x='value', y='framework', hue='pyver', data=df, orient='h', palette=sns.color_palette('rainbow'))
plt.yticks(fontsize=18)
sns.distplot(frameworks_pyver_ratio.iloc[0, :], bins=5, color='b')
sns.stripplot(x='framework', y='value', hue='pyver', data=df, size=5, palette=sns.color_palette('rainbow'))
plt.xticks(rotation=90, fontsize=14);
plt.legend(title=None)
plt.figure(figsize=(7, 13))
sns.stripplot(x='value', y='framework', hue='pyver', data=df, orient='h', size=5, palette=sns.color_palette('rainbow'))
plt.yticks(fontsize=14)
plt.legend(title=None)
plt.figure(figsize=(13, 13))
sns.stripplot(x='value', y='framework', hue='pyver', 
              data=df, 
              order=frameworks_pyver_ratio.T['Python 3'].sort_values(ascending=False).index, 
              orient='h', 
              size=7, 
              palette=sns.color_palette('rainbow'))
plt.yticks(fontsize=14)
plt.xlabel('')
plt.ylabel('')
plt.legend(title=None, loc='upper center')
plt.tight_layout()
plt.savefig('frameworks-python-version.png')
cols = find_cols(survey_df, ['use', 'python', 'most'])
cols
uses = survey_df['what do you use python for the most?']
uses.head()
frameworks_uses = frameworks.apply(lambda col: pd.crosstab(index=uses, columns=col).iloc[:, 0])
frameworks_uses.columns = [item.split(':')[0] for item in frameworks_uses.columns]
frameworks_uses.head()
da_ml_frameworks_uses = frameworks_uses.loc[['Data analysis', 'Machine learning']]
da_ml_frameworks_uses.head()
da_ml_frameworks_uses.T.sort_values(by='Data analysis').plot.area(stacked=False, alpha=0.5, figsize=(20, 15), 
                                                                  color=sns.color_palette('rainbow')[:2])
plt.xticks(range(da_ml_frameworks_uses.shape[1]), 
           da_ml_frameworks_uses.T.sort_values(by='Data analysis').index, 
           rotation=90, fontsize=18);
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig('frameworks-data-analysis-machine-learning.png')
plt.figure(figsize=(10, 15))
df = da_ml_frameworks_uses.stack().reset_index()
df.columns = ['use', 'framework', 'value']
sns.barplot(x='value', y='framework', hue='use', 
            data=df, 
            orient='h', 
            order=da_ml_frameworks_uses.T.sort_values(by='Data analysis', ascending=False).index,
            palette=sns.color_palette('rainbow'))
plt.yticks(fontsize=16)
cols = find_cols(survey_df, ['how', 'many', 'people', 'project'])
cols
team_scale = survey_df[cols[0]]
team_scale.head()
team_scale.describe()
team_scale.isnull().sum()
team_pyver = pd.crosstab(team_scale, python_ver)
team_pyver = team_pyver.reindex(['2-7 people', '8-12 people', '13-20 people', '21-40 people', 'More than 40 people'])
team_pyver
team_pyver_sorted = team_pyver.div(team_pyver.sum(axis=1), axis=0).sort_values(by='Python 3', ascending=False)
team_pyver_sorted
team_pyver_sorted['Python 3'].plot(label='Python 3', marker='o', markersize=10, color='b', linewidth=3)
plt.xticks(range(5), team_pyver_sorted.index, fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('team scale', fontsize=16)
plt.ylabel('use ratio of python 3', fontsize=16)
plt.legend(fontsize=14)
plt.title('Team scale VS Use ratio of Python 3', fontsize=18)
plt.tight_layout()
plt.savefig('team-scale-python-3.png')
cols = find_cols(survey_df, ['age', 'range'])
cols
age = survey_df[cols[0]]
age.head()
age.describe()
age.isnull().sum()
survey_df.loc[age.isnull()]
age.unique()
age_pyver = pd.crosstab(index=age, columns=python_ver)
age_pyver
age_pyver.div(age_pyver.sum(axis=1), axis=0)['Python 3'].plot(label='Python 3',
                                                              marker='o', 
                                                              markersize=10, 
                                                              color='b', 
                                                              linewidth=3)
plt.xticks(range(age_pyver.shape[0]), age_pyver.index, fontsize=14);
plt.yticks(fontsize=14)
plt.xlabel('age', fontsize=16)
plt.ylabel('use ratio of python 3', fontsize=16)
plt.legend(fontsize=14)
plt.title("The developers' age VS The use ratio of Python 3", fontsize=18)
plt.tight_layout()
plt.savefig('age-python-3.png')
china = survey_df.loc[survey_df['what country do you live in?'] == 'China']
usa = survey_df.loc[survey_df['what country do you live in?'] == 'United States']
china.head()
country_age = pd.crosstab([survey_df['what country do you live in?'], survey_df['which version of python do you use the most?']], survey_df['could you tell us your age range?'])
country_age.index.names = ['country', 'pyver']
country_age.head()
country_age_total = country_age.sum(level=0)
country_age_total.head()
country_age_total['60 or older'].sum()
country_age_total['60 or older'].sort_values(ascending=False)[:10].plot(kind='bar', color=sns.color_palette('rainbow', 10))
plt.xticks(fontsize=14, rotation=0)
plt.xlabel('')
plt.ylabel('# of the developers whose age are 60+', fontsize=14)
plt.title('Top 10 countries of # of the developers whose age are 60+', fontsize=18)
plt.tight_layout()
plt.savefig('top10-60-or-older.png')

three_countries = country_age_total.loc[['United States', 'India', 'China']]
three_countries
three_countries_ratio = three_countries.div(three_countries.sum(axis=1), axis=0)
three_countries_ratio
three_countries_ratio.plot(kind='bar', figsize=(12, 6), color=sns.color_palette('rainbow', 7))
plt.xticks(range(3), ['United States', 'India', 'China'], rotation=0, fontsize=14);
plt.xlabel('')
plt.legend(title=None)
plt.title("Age distribution of the developers who're from USA, India and China", fontsize=18);
plt.tight_layout()
plt.savefig('age-distribution.png')
country_age.loc['United States']
41 / 56
cols = find_cols(survey_df, ['country', 'live'])
cols
countries = survey_df[cols[0]]
countries.head()
countries.describe()
countries.isnull().sum()
survey_df.loc[countries.isnull()]
sns.countplot(countries, order=countries.value_counts().index, palette=sns.color_palette('rainbow', 16))
plt.xlim(xmax=15.5);
# plt.xlabel('国家', fontsize=18, fontproperties=chinese);
# plt.ylabel('频数', fontsize=18, fontproperties=chinese);
plt.xticks(fontsize=14, rotation=15);
plt.xlabel('')
plt.ylabel('# of developers', fontsize=14)
plt.tight_layout()
plt.savefig('country-counts.png')
countries_pyver = pd.crosstab(index=countries, columns=python_ver)
countries_pyver.head()
top10_countries = countries_pyver.loc[countries.value_counts()[:10].index]
top10_countries = top10_countries.div(top10_countries.sum(axis=1), axis=0)
top10_countries
df = top10_countries.stack().reset_index()
df.columns = ['country', 'pyver', 'value']
df.head()
sns.barplot(x='country', y='value', hue='pyver', data=df, palette=sns.color_palette('rainbow'))
plt.xticks(fontsize=14)
plt.xlabel('')
plt.ylabel('use ratio', fontsize=14)
plt.legend(title=None, fontsize=12)
plt.tight_layout()
plt.savefig('country-pyver.png')
countries_pyver_ratio = countries_pyver.div(countries_pyver.sum(axis=1), axis=0)
countries_pyver_ratio.head()
plt.figure(figsize=(12, 6))
sns.distplot(countries_pyver_ratio['Python 3'], rug=True, color='b')
plt.xlabel('use ratio', fontsize=14)
plt.xticks(fontsize=12)
plt.title('Use ratio of Python 3 in the world', fontsize=18)
plt.tight_layout()
plt.savefig('use-ratio-of-python3-world.png')
plt.figure(figsize=(10, 3))
sns.stripplot(x='Python 3', data=countries_pyver_ratio)
import matplotlib.transforms as transforms
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(countries_pyver_ratio['Python 3'], c='b')
ax.axhline(countries_pyver_ratio['Python 3'].mean(), c='k', label='Mean')
plt.xticks([]);
xmin, _ = plt.xlim()
ax.text(0, countries_pyver_ratio['Python 3'].mean(), round(countries_pyver_ratio['Python 3'].mean(), 2), 
        transform=transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData), 
        rotation='horizontal', 
        horizontalalignment='right', 
        verticalalignment='center');
ax.legend()
plt.figure(figsize=(10, 5))
sns.violinplot(countries_pyver_ratio['Python 3'], palette=sns.color_palette('rainbow'))
countries_pyver_ratio['Python 3'].describe()
countries_pyver_ratio['Python 2'].describe()
init_notebook_mode(connected=True)
data = [ dict(
        type = 'choropleth',
        locations = countries_pyver_ratio.index,
        locationmode = 'country names', 
        z = countries_pyver_ratio['Python 3'] * 100,
        text = countries_pyver_ratio.index,
        colorscale = 'Set3',
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            ticksuffix = '%',
            title = 'Percent'),
      ) ]

layout = dict(
    title = 'Python 3 in the world',
    geo = dict(
        showframe = False,
        showcoastlines = False,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict( data=data, layout=layout )
iplot(fig, validate=False, filename='Python 3 in the world', image_height=1080, image_width=1920, show_link=False)