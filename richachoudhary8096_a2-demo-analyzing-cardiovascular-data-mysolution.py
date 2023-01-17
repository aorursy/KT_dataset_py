import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



df= pd.read_csv('../input/mlbootcamp5_train.csv')

print(df.head(5))
df.shape

df.columns
type(df['gender'])
df['gender'].value_counts()
pd.crosstab(df['gender'], df['alco'])
pd.crosstab(df['gender'], df['smoke'])
813/44717
5356/19114
(0.28021345610547244-0.01818100498691773)*100
round(100 * (df.loc[df['gender'] == 2, 'smoke'].mean() - df.loc[df['gender'] == 1, 'smoke'].mean()))
#pd.crosstab(df['smoke'], df['age']).T
20000/365
(df[df['smoke']==1]['age'].median() - df[df['smoke']==0]['age'].median())/365.25*12
df['age_years'] = (df['age'] / 365.25).round().astype('int')

df['age_years'].max()
smoking_old_men = df[(df['gender'] == 2) & (df['age_years'] >= 60)

                    & (df['age_years'] < 65) & (df['smoke'] == 1)]

smoking_old_men.head()
smoking_old_men[(smoking_old_men['cholesterol'] == 1) &

               (smoking_old_men['ap_hi'] < 120)]['cardio'].mean()
smoking_old_men[(smoking_old_men['cholesterol'] == 3) & (smoking_old_men['ap_hi'] >=160) &

               (smoking_old_men['ap_hi'] < 180)]['cardio'].mean()
df['bmi']= df['weight']/(df['height']/100)**2
df['bmi'].median()
df.groupby('gender')['bmi'].mean()
df.groupby(['gender', 'alco', 'cardio'])['bmi'].median().to_frame()
###data cleaning 
filtered_df = df[(df['ap_lo'] <= df['ap_hi']) & 

                 (df['height'] >= df['height'].quantile(0.025)) &

                 (df['height'] <= df['height'].quantile(0.975)) &

                 (df['weight'] >= df['weight'].quantile(0.025)) & 

                 (df['weight'] <= df['weight'].quantile(0.975))]

print(filtered_df.shape[0] / df.shape[0])
df.corrwith(df['gender'])
corr = df[['id', 'age', 'height', 'weight', 

           'ap_hi', 'ap_lo', 'cholesterol', 

           'gluc']].corr(method='spearman')

corr
# Create a mask to hide the upper triangle of the correlation matrix (which is symmetric)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(12, 10))



# Plot the heatmap using the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, vmax=1, center=0, annot=True, fmt='.2f',

      square=True, linewidths=.5, cbar_kws={"shrink": .5});
sns.countplot(x="age_years", hue='cardio', data=df);