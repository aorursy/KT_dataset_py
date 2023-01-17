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
df_uniques
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 
                                              'value'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()

df_uniques
sns.factorplot(x='variable', y='count', hue='value', 
               data=df_uniques, kind='bar', size=12);
df_uniques = pd.melt(frame=df, value_vars=['gender','cholesterol', 
                                           'gluc', 'smoke', 'alco', 
                                           'active'], 
                     id_vars=['cardio'])
df_uniques
df_uniques = pd.DataFrame(df_uniques.groupby(['variable', 'value', 
                                              'cardio'])['value'].count()) \
    .sort_index(level=[0, 1]) \
    .rename(columns={'value': 'count'}) \
    .reset_index()
df_uniques
for c in df.columns:
    n = df[c].nunique()
    print(c)
    if n <= 3:
        print(n, sorted(df[c].value_counts().to_dict().items()))
    else:
        print(n)
    print(10 * '-')
df['height'].describe()
df['height'].value_counts()
df['height'].hist()
df['height'].mean()

#df['height'].hist()
sns.distplot(df['height'])
plt.show()
sns.countplot(x = 'height', data = df)
sns.countplot(x = 'gender', hue = 'alco', data = df)
d1 = df[(df['smoke'] == 1) & (df['gender'] == 1)].shape[0]/df[df['gender'] == 1].shape[0];
d2 = df[(df['smoke'] == 1) & (df['gender'] == 2)].shape[0]/df[df['gender'] == 2].shape[0];
np.abs(d1 - d2)
sns.countplot(x = 'gender', hue = 'smoke', data = df)
df['age']/365
df[df['smoke'] == 0]['age'].mean()/30 - df[df['smoke'] == 1]['age'].mean()/30

age_years = np.rint(df['age']/365)
#age_years = df[(df['age'] <= 64) & (df['age'] >= 60)]['age']
df.insert(loc = len(df.columns), column = 'age_years', value = age_years)
df[(df['age'] <= 64) & (df['age'] >= 60)]['cholesterol'] = df[(df['age'] <= 64) & (df['age'] >= 60)]['cholesterol'].map({'1' : '4 mmol/l', '2' : '1, 5-7 mmol/l', '3': '8 mmol/l'})
df[(df['age'] <= 64) & (df['age'] >= 60)]['cholesterol']

smoking_old_men = df[(df['gender'] == 2) & (df['smoke'] == 1) & (df['age_years'] < 65) & (df['age_years'] >= 60)]
smoking_old_men[(smoking_old_men['cholesterol'] == 1) & (smoking_old_men['ap_hi'] < 120)]['cardio'].mean()
smoking_old_men[(smoking_old_men['cholesterol'] == 3) & (smoking_old_men['ap_hi'] >= 160) & (smoking_old_men['ap_hi'] < 180)]['cardio'].mean()
bmi = df['weight']/df['height']**2
df.insert(loc = len(df.columns), column = 'bmi', value = bmi)
sns.distplot(df['bmi'])
sns.boxplot(y = 'bmi', x= 'gender', data = df)
df[df['gender'] == 2]['bmi'].median() - df[df['gender'] == 1]['bmi'].median()
df[df['cardio'] == 0]['bmi'].mean() - df[df['cardio'] == 1]['bmi'].mean()
df[(df['cardio'] == 0) & (df['alco'] == 0) & (df['gender'] == 1)]['bmi'].median() 
df[(df['cardio'] == 0) & (df['alco'] == 0) & (df['gender'] == 2)]['bmi'].median() 
df['bmi'].median()
df.drop(df[df['ap_lo'] > df['ap_hi']].index, inplace = True)
df.drop(df[df['height'] < df['height'].quantile(q = 0.025)].index, inplace = True)
df.drop(df[df['height'] > df['height'].quantile(q = 0.975)].index, inplace = True)
df.drop(df[df['weight'] < df['weight'].quantile(q = 0.025)].index, inplace = True)
df.drop(df[df['weight'] > df['weight'].quantile(q = 0.975)].index, inplace = True)
df
1 - 62784/70000
#mask:
correlation_matrix = df.corr(method = 'pearson')
mask = np.zeros_like(correlation_matrix,dtype = np.bool)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(correlation_matrix, mask = mask, vmax = 1, center=0, annot=True, fmt='.1f',
            square=False, linewidths=.5)
sns.violinplot(x = 'gender',y = 'height', data = df, scale = 'count')
# You code here
from scipy import stats 
stats.spearmanr(df)
# You code here
# You code here