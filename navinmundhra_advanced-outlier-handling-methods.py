import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.rcParams['figure.figsize'] = (8,6) # setting a default preferred size for plots

plt.style.use('fivethirtyeight')

sns.set(style='darkgrid')

import scipy

from scipy import stats
df = pd.read_csv('../input/mobility/train.csv')

df
df.drop('Trip_ID', axis=1,inplace=True); target = 'Surge_Pricing_Type'

numfeat, catfeat = list(df.select_dtypes(include=np.number)), list(df.select_dtypes(exclude=np.number)); numfeat.remove(target)

df
df.info()
import missingno as msno; msno.bar(df, figsize=(16,6)); plt.show()
df[numfeat].skew()
for col in catfeat:

    df[col].fillna(df[col].mode()[0], inplace=True)

for col in numfeat:

    df[col].fillna(df[col].median(), inplace=True)
f, a = plt.subplots(2,4, figsize=(24,12))

a = a.flatten().T

for i, col in enumerate(df[numfeat].columns):

    sns.distplot(df[col],ax=a[i],kde=False).set_title('Skew: {:.4f}'.format(df[col].skew()))

plt.show()
f, a = plt.subplots(2,4, figsize=(24,12))

a = a.flatten().T

for i, col in enumerate(df[numfeat].columns):

    stats.probplot(df[col], plot=a[i])

    a[i].set_title(col)

plt.show()
## REMOVING OBSERVATIONS BASED ON CLASSIFICATION IN 'EXAMPLE OUTLIERS' FOR FEATURE 'Var1'

df.drop(df[df['Var1']>160].index).reset_index(drop=True)
## TRANSFORMING THE FEATURE TO TREAT OUTLIERS USING BOXCOX TRANSFORMATION FOR FEATURE 'Var3'

temp = pd.Series(stats.boxcox(df['Var3'],lmbda=stats.boxcox_normmax(df['Var3'])))

stats.probplot(temp, plot=plt); plt.show()
## TREATING OUTLIERS AS MISSING VALUES FOR FEATURE 'Life_Style_Index'

df['Life_Style_Index'].where(df['Life_Style_Index']<4.0, df['Life_Style_Index'].median())
sns.pairplot(df[numfeat]); plt.show()
sns.scatterplot('Var2', 'Var3', data=df, hue = target).set_title('Correlation: {:.4f}'.format(df['Trip_Distance'].corr(df['Life_Style_Index']))); plt.show()
from scipy.spatial import ConvexHull



extremes = df[['Var2', 'Var3']].to_numpy()



hull = ConvexHull(extremes)



# print(extremes[hull.vertices])



plt.plot(df["Var2"], df["Var3"], 'ok')

plt.plot(extremes[hull.vertices, 0], extremes[hull.vertices,1], 'r--', lw = 2)

plt.plot(extremes[hull.vertices, 0], extremes[hull.vertices,1], 'ro', lw = 2)

plt.show()