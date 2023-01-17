import numpy as np
import pandas as pd
import seaborn as sns
import re
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

# read MCQ responses
df = pd.read_csv('../input/multipleChoiceResponses.csv')

# keep only columns needed in this analysis
needed_cols = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q17','Q20','Q22','Q24','Q25','Q26','Q32','Q37']
df = df[needed_cols]

#drop first row which include question details
df.drop(df.index[0], inplace=True)

# cleanup unneeded data
df = df[pd.notnull(df['Q9'])]
df = df[~df.Q9.isin(['I do not wish to disclose my approximate yearly compensation','500,000+'])]
df = df[df.Q6 != 'Student']
df = df[df.Q7 != 'I am a student']
df['Q3'] = df['Q3'].str.replace('(\(|,| or | and | of ).+$','', regex=True)
df['Q4'] = df['Q4'].str.replace('(/).+$','', regex=True)
df['Q5'] = df['Q5'].str.replace('(\(|,| or | and | of ).+$','', regex=True)

# represent compensation with range midpoint
def extract_avg_pay(compensation):
    result = re.split('-|,',compensation)
    return 1000*(int(result[0]) + int(result[1]))/2
    
df['pay'] = df['Q9'].apply(extract_avg_pay)

# plot formatting
sns.set(style='whitegrid')
comma_fmt = FuncFormatter(lambda x, p: format(int(x), ','))
pay_label = 'Average Yearly Compensation (USD)'
sns.set_palette(['#95d0fc','pink'])
px = df[df.Q1.isin(['Female','Male'])]['Q1'].value_counts(normalize=True).plot(kind='pie', autopct='%.1f%%', figsize=(16,7))
px.axis('equal')
px.get_yaxis().set_visible(False)
gender_palette = ['pink','#95d0fc']
sns.set_palette(gender_palette)
px = df[df.Q1.isin(['Female','Male'])].groupby(['Q1'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set(xlabel='Gender', ylabel=pay_label)
px.set_xticklabels(px.get_xticklabels(), rotation=0)
px.yaxis.set_major_formatter(comma_fmt)
sns.set_palette('husl')
px = df[df.Q3 != 'Other']['Q3'].value_counts().nlargest(15).plot(kind='bar', figsize=(16,7))
px.set_xticklabels(px.get_xticklabels(), rotation=30)
sns.set_palette(sns.light_palette("green",reverse=True, n_colors=10))
px = df.groupby(['Q3'])['pay'].mean().sort_values(ascending=False).head(10).plot(kind="bar", figsize=(16,7))
px.set(xlabel='Country', ylabel=pay_label)
px.set_xticklabels(px.get_xticklabels(), rotation=30)
px.yaxis.set_major_formatter(comma_fmt)
sns.set_palette('husl')
px = df['Q4'].value_counts(normalize=True).plot(kind='pie', autopct='%.1f%%', figsize=(16,7))
px.axis('equal')
px.get_yaxis().set_visible(False)
sns.set_palette('pastel')
px = df.groupby(['Q4'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set(xlabel='Highest Education level', ylabel=pay_label)
px.set_xticklabels(px.get_xticklabels(), rotation=30)
px.yaxis.set_major_formatter(comma_fmt)
sns.set_palette(gender_palette)
px = df[df.Q1.isin(['Male','Female'])].groupby(['Q4','Q1'])['pay'].mean().unstack().plot(kind="bar", figsize=(16,7))
px.set(xlabel='Highest Education', ylabel=pay_label)
px.legend().set_title('Gender')
px.set_xticklabels(px.get_xticklabels(), rotation=30)
px.yaxis.set_major_formatter(comma_fmt)
sns.set_palette('husl')
px = df['Q5'].value_counts(normalize=True).plot(kind='pie', autopct='%.1f%%', figsize=(16,7))
px.axis('equal')
px.get_yaxis().set_visible(False)
sns.set_palette('pastel')
px = df.groupby(['Q5'])['pay'].mean().sort_values().plot(kind="barh", figsize=(15,7))
px.set(xlabel=pay_label, ylabel='Major')
px.xaxis.set_major_formatter(comma_fmt)
sns.set_palette(gender_palette)
px = df[df.Q1.isin(['Male','Female'])].groupby(['Q5','Q1'])['pay'].mean().unstack().plot(kind="bar",figsize=(16,7))
px.legend().set_title('Gender')
px.set(xlabel='Major', ylabel=pay_label)
px.yaxis.set_major_formatter(comma_fmt)
dfz = df[~df['Q4'].isin(['Some college','I prefer not to answer'])].pivot_table(index='Q5', columns='Q4', values='pay', aggfunc=np.mean)
fig, ax = plt.subplots(figsize=(16,7))
px = sns.heatmap(dfz, annot=True, fmt=".1f", cmap="RdBu_r",ax=ax)
px.set(xlabel='Highest Education', ylabel='Major')
px.set_xticklabels(px.get_xticklabels(), rotation=0)

sns.set_palette('pastel')
px = df[df.Q6 != 'Other'].groupby(['Q6'])['pay'].mean().sort_values().plot(kind="barh",figsize=(15,7))
px.set(ylabel='Job Title', xlabel=pay_label)
px.xaxis.set_major_formatter(comma_fmt)
sns.set_palette(gender_palette)
px = df[(df.Q1.isin(['Male','Female'])) & (df.Q6 != 'Other')].groupby(['Q6','Q1'])['pay'].mean().unstack().plot(kind="bar",figsize=(16,7))
px.set(xlabel='Job Title', ylabel=pay_label)
px.legend().set_title('Gender')
px.yaxis.set_major_formatter(comma_fmt)
sns.set_palette('pastel')
px = df.groupby(['Q26'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set(xlabel='Are you Data scientist', ylabel=pay_label)
px.set_xticklabels(px.get_xticklabels(), rotation=0)
px.yaxis.set_major_formatter(comma_fmt)
px = df[df.Q7 != 'Other'].groupby(['Q7'])['pay'].mean().sort_values().plot(kind="barh", figsize=(14,7))
px.set(ylabel='Industry', xlabel=pay_label)
px.xaxis.set_major_formatter(comma_fmt)
px = df.groupby(['Q8'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set(xlabel='Years of Experience', ylabel=pay_label)
px.set_xticklabels(px.get_xticklabels(), rotation=0)
px.yaxis.set_major_formatter(comma_fmt)
sns.set_palette('pastel')
px = df.groupby(['Q17'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set(xlabel='Language', ylabel=pay_label)
px.set_xticklabels(px.get_xticklabels(), rotation=30)
px.yaxis.set_major_formatter(comma_fmt)
px = df[df.Q20 != 'Other'].groupby(['Q20'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set(xlabel='Library', ylabel=pay_label)
px.set_xticklabels(px.get_xticklabels(), rotation=30)
px.yaxis.set_major_formatter(comma_fmt)
px = df[df.Q22 != 'Other'].groupby(['Q22'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set(xlabel='Visualization', ylabel=pay_label)
px.set_xticklabels(px.get_xticklabels(), rotation=30)
px.yaxis.set_major_formatter(comma_fmt)
px = df[df.Q32 != 'Other Data'].groupby(['Q32'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set(xlabel='Type of Data', ylabel=pay_label)
px.set_xticklabels(px.get_xticklabels(), rotation=30)
px.yaxis.set_major_formatter(comma_fmt)
px = df[df.Q37 != 'Other'].groupby(['Q37'])['pay'].mean().sort_values().plot(kind="bar", figsize=(16,7))
px.set(xlabel='Learning Platform', ylabel='Average Yearly Pay in USD')
px.set_xticklabels(px.get_xticklabels(), rotation=30)
px.yaxis.set_major_formatter(comma_fmt)