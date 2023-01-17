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
df.head()
df.groupby('gender')['height'].mean(), df.groupby('gender')['weight'].mean()
df.groupby('gender').describe()
df.groupby('gender')['alco'].mean()
percent_smokers = df.groupby('gender')['smoke'].mean()
percent_smokers[2]-percent_smokers[1]
med_age = df.groupby('smoke')['age'].median()/365
(med_age[0]-med_age[1])*12
df['age_years'] = (df['age'] / 365.25).round().astype('int')
df[(df['age_years']>=60) & (df['age_years']<=64)].head()
df['BMI'] = df['weight']/(df['height']/100)**2
df['BMI'].describe()
df.groupby('gender')['BMI'].mean()
df.groupby('cardio')['BMI'].mean()
healthy = df[(df['cardio']==0) & (df['alco']==0)]
healthy.groupby('gender')['BMI'].mean()

clean = df[(df['ap_lo']<=df['ap_hi']) &
           (df['height']>=df['height'].quantile(0.025))&
           (df['height']<=df['height'].quantile(0.975))&
           (df['weight']>=df['weight'].quantile(0.025))&
           (df['weight']<=df['weight'].quantile(0.975))
          ]
100-100*(clean.shape[0]/df.shape[0])
corr = df.corr()
sns.heatmap(corr)
df_melt = pd.melt(frame=df, value_vars=['height'], id_vars=['gender'])
plt.figure(figsize=(12, 10))
sns.violinplot( x='variable', 
    y='value', hue='gender',scale='count',split=True, data=df_melt)
spearman = df.corr('spearman')
sns.heatmap(spearman,annot=True, fmt='.1f')
sns.countplot(x='age_years',hue='cardio',data=df)