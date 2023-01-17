%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
import warnings
warnings.filterwarnings('ignore')
from statsmodels.api import OLS
df = pd.read_csv('../input/NBA_Players.csv', sep=',')
df.info()
# Basic data cleaning.
# Column names have space, trim it first.
df.columns = [i.strip() for i in df.columns.tolist()]

# Change data type.
df['AGE'] = df['AGE'].str.replace('-', '0')
df['AGE'] = df['AGE'].astype(int)
df['SALARY'] = df['SALARY'].str.replace('Not signed', '0')
df['SALARY'] = df['SALARY'].str.replace(',', '')
df['SALARY'] = df['SALARY'].astype(int)
df.head()
# Let's check the player's experience group.
fig, ax = plt.subplots(figsize=(10, 10))

experience_group = df['EXPERIENCE'].value_counts(bins=[0, 3, 5, 8, 10, 15, 25]).values
labels = ['0-3 years', '5-8 years', '3-5 years', '8-10 years', '10-15 years', 'more than 15 years']
explodes = (0, 0, 0, 0, 0.1, 0.2)
cmap = plt.get_cmap("tab20c")
colors = cmap(np.array([1, 2, 7, 8, 9, 12]))

ax = plt.pie(experience_group, labels=labels, autopct='%1.0f%%', colors=colors, 
             pctdistance=.55, explode=explodes, labeldistance=1.05)
plt.title('NBA Players Experience Group as of 18-19 Season', fontsize=12);
# Let's check how many players does each college contribute.
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)

ncount = len(df.COLLEGE)
orders = df['COLLEGE'].value_counts().iloc[:15].index.tolist()
ax = sns.countplot(x='COLLEGE', data=df, order=orders, palette="rainbow");
ax.grid(True)

ax.set_title('TOP Universties which contributed to NBA talents.', color='indianred', fontsize=12)
ax.set_xlabel('College Name')
ax.set_ylabel('Number of NBA Players')
# set text annotation for each bar.
for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 
                ha='center', va='bottom')
fig.autofmt_xdate()
fig, ax = plt.subplots(figsize=a4_dims, sharey=True)
# first subset teams according to each team's gross salary.
team_gross_salary = df.groupby(['TEAM'])['SALARY'].sum().values.tolist()
luxury_tax_teams = []
teams = df.groupby(['TEAM'])['SALARY'].sum().index.tolist()
bools = list(df.groupby(['TEAM'])['SALARY'].sum().values > 101900000)
for i in range(len(teams)):
    if not bools[i]:
        pass
    else:
        luxury_tax_teams.append(teams[i])
# plot two sets of data.
grouped = df.groupby(['TEAM'])['SALARY'].sum().sort_values(ascending=False)
# seperate two groups by color.
color_group = {True:'r', False:'g'}
color_scheme = [color_group[i] for i in list(grouped.index.isin(luxury_tax_teams))]

grouped.plot(kind='bar', ax=ax, color=color_scheme, label='Team Salary');
ax.axhline(y=101900000, color='blue', linewidth=.8, label='Salary Cap 2018-2019');
ax.set_title('NBA Teams Salary Overview', color='purple', fontsize=12);
ax.set_ylabel('Salary/100(million)');
ax.legend();

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}'.format(y/100000000), (x.mean(), y), ha='center', va='bottom')

fig.autofmt_xdate()
plt.savefig('NBA Teams Salary Overview.png', dpi=720, facecolor='w', edgecolor='w',
            orientation='landscape', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None)
fig, ax = plt.subplots(figsize=a4_dims)

nba_mean_age = df.AGE.mean()
teams = df.groupby(['TEAM'])['AGE'].mean().index.tolist()
team_mean_age = df.groupby(['TEAM'])['AGE'].mean().values.tolist()
bools = list(df.groupby(['TEAM'])['AGE'].mean().values > nba_mean_age)

old_teams = []
for i in range(len(teams)):
    if not bools[i]:
        pass
    else:
        old_teams.append(teams[i])

teams_grouped = df.groupby(['TEAM'])['AGE'].mean().sort_values(ascending=False)
colors = {True:'orange', False:'indianred'}
color_scheme = [colors[i] for i in teams_grouped.index.isin(old_teams)]

teams_grouped.plot(kind='bar', ax=ax, color=color_scheme, label='Team Average Age');
ax.axhline(y=nba_mean_age, color='green', linewidth=.8, label='NBA Plater Average Age 2018-2019');

ax.set_title('NBA Teams Player Average Age', fontsize=12)
ax.set_ylabel('Age')
ax.legend();

for p in ax.patches:
    x=p.get_bbox().get_points()[:,0]
    y=p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}'.format(y), (x.mean(), y), ha='center', va='bottom')

fig.autofmt_xdate()
fig, ax = plt.subplots(figsize=(12, 12))

data = df.loc[(df.POSITION!='F') & (df.POSITION!='G')]
position_salary_team = pd.pivot_table(data, values='SALARY', index='TEAM', columns=['POSITION'], aggfunc=np.mean)

sns.heatmap(position_salary_team, linewidth=.7, robust=True, annot=True, cmap='rainbow',
            annot_kws={'size':11}, fmt='.1e');
ax.set_title('Position Mean Salary in Each NBA Team 2018-2019 Season');
# Let's have a look at top earners in each team.
from matplotlib import rcParams
rcParams['axes.titlepad'] = 30
fig, ax = plt.subplots(figsize=(14, 12))
# Step 1: get all the top earners in each team
def func(group):
    return group.loc[group['SALARY'] == group['SALARY'].max()]

top_earners = df.groupby('TEAM', as_index=False).apply(func).reset_index(drop=True)
data = top_earners[['TEAM', 'NAME', 'SALARY', 'EXPERIENCE', 'AGE', 
                    'PPG_LAST_SEASON', 'PPG_CAREER', 'PER_LAST_SEASON']]
# Convert the salary to the smaller scale.
data['SALARY'] = data['SALARY'] / 1000000
data = data.set_index(['TEAM', 'NAME'])

sns.heatmap(data, linewidth=.7, robust=True, annot=True, cmap='tab20',
            annot_kws={'size':11}, fmt='.2f');
ax.set_title('Overview of 2018-2019 NBA Top Earners in Each Team', fontsize=12, fontweight="bold");
ax.tick_params(labeltop=True, labelsize=10, color='indianred');
# Let's have a look at top earners in each team.
# Step 1: get all the top earners in each team
def func(group):
    return group.loc[group['SALARY'] == group['SALARY'].max()]

top_earners = df.groupby('TEAM', as_index=False).apply(func).reset_index(drop=True)
g = sns.FacetGrid(top_earners, col='POSITION', col_wrap=3, hue='EXPERIENCE', height=4)
g.map(plt.scatter, 'SALARY', 'PER_LAST_SEASON', alpha=.7);
g.add_legend();

# Set figsize here
from matplotlib import rcParams
rcParams['axes.titlepad'] = 30

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,6))
cats = top_earners[['EXPERIENCE', 'POSITION', 'AGE', 'COLLEGE']]

# flatten axes for easy iterating
for i, ax in enumerate(axes.flatten()):
    sns.countplot(x= cats.iloc[:, i], orient='v', ax=ax)
    cats_names = cats.columns.tolist()
    ax.set_title('%s Distribution of 2018-2019 NBA Top Earners in Each Team' % cats_names[i],
                  fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of counts')
    for label in ax.get_xticklabels():
        if i==3:
            label.set_rotation(90) 

fig.tight_layout()
from matplotlib import rcParams
rcParams['axes.titlepad'] = 30

measures = ['EXPERIENCE', 'SALARY', 'PPG_LAST_SEASON', 'PER_LAST_SEASON', 
                        'APG_LAST_SEASON', 'PPG_CAREER', 'APG_CAREER', 'GP']
g = sns.PairGrid(top_earners, vars=measures, hue='POSITION', palette='Set2', height=2.5)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);
g.add_legend();
ax.tick_params(labeltop=True, color='indianred');
