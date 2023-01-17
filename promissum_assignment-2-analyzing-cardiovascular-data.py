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
df1 = df.copy()
gb = df1.groupby('gender')
a = gb['height'].agg([np.mean]).sort_values(by='mean', ascending=False).reset_index()
mix = a.loc[[0], ['gender']].gender[0]
wix = a.loc[[1], ['gender']].gender[1]
df1['gender'] = df1['gender'].map({mix: 'M', wix: 'W'})
gb = df1.groupby('gender')
#print(gb['height'].agg('mean'))
### .agg('mean') vs .agg([np.mean])
print(str(gb['gender'].size()['M']) + ' men and ' + str(gb['gender'].size()['W']) + ' women')
# 1. 45530 women and 24470 men
#gb = 
df1.groupby(['gender'])['alco'].agg([np.mean])
#value_counts(normalize=True) #agg([np.mean])
#pd.crosstab(df['gender'], df['alco']).T
a = df1.groupby(['gender'])['smoke'].value_counts(normalize=True)
(a.loc[('M',1)] - a.loc[('W',1)]) * 100  #.loc[['gender', 'smoke']]
q = (df1.groupby(['smoke'])['age'].agg([np.median]) / 30).reset_index()
#q[q['smoke'] == 0]['median'][0] - q[q['smoke'] == 1]['median'][1]
q['median'][0] - q['median'][1]
#q[q['smoke'] == 0]['median']# - q[1]
# You code here
df1['age_years'] = round(df1['age'] / 365)
df1['chol_1'] = df1['cholesterol'].map({1: 4, 2: 6, 3: 8})
df1.head()
# You code here
# You code here
# You code here
# You code here
# You code here