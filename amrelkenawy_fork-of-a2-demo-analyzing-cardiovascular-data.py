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

df.pivot_table(['height'],['gender'],aggfunc=['mean']) # men are geneder 2
df['gender'].map({1:'Female',2:'Male'}).value_counts()

pd.crosstab(df['gender'].map({1:'Female',2:'Male'}), df['alco'] == 1)

men = len(df[(df['gender'] == 2)& (df['smoke']==1)] == True)/len(df[df['gender'] == 2] == True)*100
women = len(df[(df['gender'] == 1)& (df['smoke']==1)] == True)/len(df[df['gender'] == 1] == True)*100
men - women
((df[df['smoke'] == 0]['age'].median())-(df[df['smoke'] == 1]['age'].median()))/30
age_years = (df['age']/365).astype('int32')
df.insert(loc = len(df.columns), column = 'age_years', value= age_years)
df.head()
df.head()
segment1 = df[(df['ap_hi'] < 120) & 
              (df['age_years'] > 60) & (df['age_years'] < 64)]['age_years'].count()
segment1


segment2 = df[(df['ap_hi'] > 160) & (df['ap_hi'] < 180)& 
              (df['age_years'] > 60) & (df['age_years'] < 64)]['age_years'].count()
segment2
segment1 / segment2
np.power(df['height'], 2)
bmi = df['weight']/(np.power(df['height'], 2))*10000 #10,000 to convert from cm-sq to meter-sq
df.insert(loc = len(df.columns), column = 'bmi', value= bmi)
df.head()

print('Normal BMI values are said to be from 18.5 to 25 ')
if (df['bmi'].median() > 18.5) & (df['bmi'].median() < 25): print ('True')
else: print('False')




df[(df['gender'] == 1)]['bmi'].mean() > df[(df['gender'] == 2)]['bmi'].mean()  
df[(df['cardio'] == 0)]['bmi'].mean() > df[(df['cardio'] == 1)]['bmi'].mean() 
#men is gender 2 
df[((df['cardio'] == 0) & (df['alco'] == 0) & (df['gender'] == 2))]['bmi'].median() < df[
    ((df['cardio'] == 0) & (df['alco'] == 0) & (df['gender'] == 1))]['bmi'].median()

df.info()
df = df.drop(df[df['ap_hi']<df['ap_lo']].index)
df = df.drop(df[df['height']< df['height'].quantile(0.025)].index)
df = df.drop(df[df['height']>df['height'].quantile(0.975)].index)
df = df.drop(df[df['weight']< df['weight'].quantile(0.025)].index)
df = df.drop(df[df['weight']>df['weight'].quantile(0.975)].index)
df.info()


(70000 - 62784) / 70000 * 100
df = pd.read_csv('../input/mlbootcamp5_train.csv')
corr_matrix = df.corr()
sns.heatmap(corr_matrix);
import seaborn as sns
sns.set()
df = pd.melt(df, value_vars=['height', 'weight'], id_vars='gender')
df.info()
sns.violinplot(x='variable', y='value', hue='gender', split=True, data=df)

df = pd.read_csv('../input/mlbootcamp5_train.csv')
corr = df.corr(method='spearman')
from matplotlib import pyplot
a4_dims = (15, 13)
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.heatmap(corr,annot=True);
age_years = (df['age']/365).astype('int32')
df.insert(loc = len(df.columns), column = 'age_years', value= age_years)
df.head()
sns.countplot(x='age_years', hue='cardio',data=df)