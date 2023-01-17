# Import all required modules
import pandas as pd
import numpy as np

# Disable warnings
import warnings
warnings.filterwarnings("ignore")

# Import plotting modules
import seaborn as sns
sns.set()
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
%matplotlib inline
# Tune the visual settings for figures in `seaborn`
sns.set_context(
    "notebook", 
    font_scale=1.5,       
    rc={ 
        "figure.figsize": (11, 8), 
        "axes.titlesize": 18 
    }
)

from matplotlib import rcParams
rcParams['figure.figsize'] = 11, 8
df = pd.read_csv('../input/mlbootcamp5_train.csv')
print('Dataset size: ', df.shape)
df.head()
df_uniques = pd.melt(frame=df, value_vars=['gender','cholesterol', 
                                           'gluc', 'smoke', 'alco', 
                                           'active', 'cardio'])
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=12);
df_uniques = pd.melt(frame=df, value_vars=['gender','cholesterol', 
                                           'gluc', 'smoke', 'alco', 
                                           'active'], 
                     id_vars=['cardio'])
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 'value', 
                                              'cardio'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

sns.factorplot(x='variable', y='count', hue='value', 
               col='cardio', data=df_uniques, kind='bar', size=9);
for c in df.columns:
    n = df[c].nunique()
    print(c)
    if n <= 3:
        print(n, sorted(df[c].value_counts().to_dict().items()))
    else:
        print(n)
    print(10 * '-')
df[['gender', 'height']].groupby('gender')['height'].describe()[['count', 'mean']]
df['gender_cat'] = df['gender'].apply(lambda x: 'men' if x == 2 else 'women')
df[['gender_cat', 'alco']].groupby('gender_cat').describe()
df[(df['gender_cat'] == 'men') & 
   (df['smoke'] == 1)].shape[0] / df[(df['gender_cat'] == 'men')].shape[0] - \
df[(df['gender_cat'] == 'women') & 
   (df['smoke'] == 1)].shape[0] / df[(df['gender_cat'] == 'women')].shape[0]
round(df[df['smoke'] == 0]['age'].median() / 30 - df[(df['smoke'] == 1)]['age'].median() / 30)
df['age_years'] = df['age'].apply(lambda x: round(x / 365))
def level(x):
    if x == 1:
        return '4 mmol/l'
    if x == 2:
        return '5-7 mmol/l'
    if x == 3:
        return '8 mmol/l'
df['cholesterol-level'] = df['cholesterol'].apply(level)
def ap_hi_level(x):
    if x < 120:
        return 'group_120'
    if (x >= 160) & (x < 180):
        return 'group_160'
    else:
        return 'other'
df['ap_hi_level'] = df['ap_hi'].apply(ap_hi_level)
df_people = df[(df['age_years'] >= 60) & 
               (df['age_years'] <= 64)]
df_people[(df_people['gender_cat'] == 'men') &
          (df_people['smoke'] == 1) &
          (df_people['ap_hi_level'] == 'group_120')
                     ]['cardio'].mean()
df_people[(df_people['gender_cat'] == 'men') &
          (df_people['smoke'] == 1) &
          (df_people['ap_hi_level'] == 'group_160')
                     ]['cardio'].mean()
0.88/0.32
df['BMI'] = df['weight'] / df['height'].apply(lambda x: (x / 100)**2)
(df['BMI'].median() >= 18.5) & (df['BMI'].median() <= 25)
df[['BMI', 'gender_cat']].groupby('gender_cat').mean()
df[['cardio', 'BMI']].groupby('cardio').mean()
df[(df['cardio'] == 0) & (df['alco'] == 0)][['gender_cat',
                                             'BMI']].groupby(['gender_cat'])['BMI'].mean()
df_filtered = df.drop(df[df['ap_hi'] < df['ap_lo']].index)
df_filtered.drop(df_filtered[df_filtered['height'] < df.height.quantile(0.025)].index, inplace=True)
df_filtered.drop(df_filtered[df_filtered['height'] > df.height.quantile(0.975)].index, inplace=True)
df_filtered.drop(df_filtered[df_filtered['weight'] < df.weight.quantile(0.025)].index, inplace=True)
df_filtered.drop(df_filtered[df_filtered['weight'] > df.weight.quantile(0.975)].index, inplace=True)
round((df.shape[0] - df_filtered.shape[0]) * 100 / df.shape[0])
df.corr()
sns.heatmap(df.corr(), annot=True, fmt=".1f", linewidths=.5);
df_melt = pd.melt(df, value_vars=['height'], id_vars=['gender'])
sns.violinplot(y='value', 
               x='variable', 
               data=df_melt, 
               hue='gender',
              split=True,
              palette='muted',
              scale='count',
              scale_hue=False)
sns.heatmap(df_filtered.corr(method='spearman'), annot=True, fmt=".1f", linewidths=.5)
sns.heatmap(df.corr(method='spearman'), annot=True, fmt=".1f", linewidths=.5)
sns.countplot(x='age_years', hue='cardio', data=df)