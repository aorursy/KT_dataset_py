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
df.groupby(['gender'])['height'].mean()
df['gender'].value_counts()
df.groupby(['gender'])['alco'].mean()
df_smoke = df.groupby(['gender'])['smoke'].value_counts(normalize = True)*100
smoke_women = df_smoke.loc[(1, 1)]
smoke_men = df_smoke.loc[(2,1)]
print(df_smoke)
abs(smoke_women - smoke_men)
smoke_med = df.groupby(['smoke'])['age'].median()
non_smokers = smoke_med.loc[0]
smokers = smoke_med.loc[1]
abs(non_smokers - smokers)/30
# divide by 30 because age feature in dataset is in days not months
import math
df['age_years'] = df.apply(lambda col: math.ceil(col.age/360), axis=1) 

#creating a dataset with smoking men aged from 60 to 64.
df1 = pd.DataFrame()
df1 = df.loc[(df['age_years'] >= 60) & (df['age_years'] <= 64) & (df['smoke'] == 1) & (df['gender'] == 2)]
df1.head()
df_120 = df1.loc[df1['ap_hi'] < 120]
df_120[df_120['cholesterol'] == 1]['cardio'].value_counts(normalize = True)*100
#23.076923 % of men with cholestrol feature value 1 and systolic blood pressure lower than 120 have CVD
df_180 = df1.loc[(df1['ap_hi'] >= 160) & (df1['ap_hi'] < 180)]
df_180[df_180['cholesterol'] == 3]['cardio'].value_counts(normalize = True)*100
# 85.714286 % of men with cholestrol feature value 3 and systolic blood pressure  in the interval [160,180) have CVD
85.714286/23.076923  
df['BMI'] = df.apply(lambda col: round(col.weight/((col.height/100) ** 2), 1), axis=1) 
df.head()
df['BMI'].median()
# 1. statement is false (Normal BMI values are from 18.5 to 25. )
df.groupby(['gender'])['BMI'].mean()
# 2.statement is true
df.groupby(['cardio'])['BMI'].mean()
# 3.statement is false 
df.groupby(['cardio','alco','gender'])['BMI'].mean()
# 4.statement is true
df2 = pd.DataFrame()
df2 = df.drop(df[df.ap_lo > df.ap_hi].index)
df2 = df2.drop(df2[df2.height < df2.height.quantile(.025)].index)
df2 = df2.drop(df2[df2.height > df2.height.quantile(.975)].index)
df2 = df2.drop(df2[df2.weight < df2.weight.quantile(.025)].index)
df2 = df2.drop(df2[df2.weight > df2.weight.quantile(.975)].index)
df2.head()
#62784 rows (before - 70000 rows)
#before - 70000 rows; after - 62784 rows
# 10% of data was thrown away
(abs(62784- 70000)/70000)*100
corr_matrix = df.corr()
sns.heatmap(corr_matrix)
datafr = pd.melt(df, value_vars=['height'], id_vars=['gender'])
sns.violinplot(x='variable', y='value', hue='gender', data=datafr, scale = 'count')
plt.show()
sp_corr_matrix = df.corr(method = 'spearman')
sns.heatmap(sp_corr_matrix)
sns.countplot(x='age_years', hue='cardio', data=df)