import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv").drop(['Unnamed: 32'],axis=1)

df.head(n=6).T
df.describe().T
print(df.dtypes)
plt.figure(figsize=(21,21))

plt.title("Pearson Correlation Heatmap")

corr = df.corr(method='pearson')

mask = np.tril(df.corr())

sns.heatmap(corr, 

           xticklabels=corr.columns.values,

           yticklabels=corr.columns.values,

           annot = True, # to show the correlation degree on cell

           vmin=-1,

           vmax=1,

           center= 0,

           fmt='0.2g', #

           cmap= 'coolwarm',

           linewidths=3, # cells partioning line width

           linecolor='white', # for spacing line color between cells

           square=False,#to make cells square 

           cbar_kws= {'orientation': 'vertical'}

           )



b, t = plt.ylim() 

b += 0.5  

t -= 0.5  

plt.ylim(b,t) 

plt.show()

################################################################################

plt.figure(figsize=(21,21))

plt.title("Spearman Correlation Heatmap")

corr = df.corr(method='spearman')

mask = np.tril(df.corr())

sns.heatmap(corr, 

           xticklabels=corr.columns.values,

           yticklabels=corr.columns.values,

           annot = True, # to show the correlation degree on cell

           vmin=-1,

           vmax=1,

           center= 0,

           fmt='0.2g', #

           cmap= 'coolwarm',

           linewidths=3, # cells partioning line width

           linecolor='white', # for spacing line color between cells

           square=False,#to make cells square 

           cbar_kws= {'orientation': 'vertical'}

           )



b, t = plt.ylim() 

b += 0.5  

t -= 0.5  

plt.ylim(b,t) 

plt.show()
#Sort dataframe columns by their mean value 

sorted_mean_df = pd.DataFrame(np.random.randn(5,32), columns=list(df.columns)).drop(['id'],axis=1)
plt.figure(figsize=(20,20))

sns.boxplot(data= sorted_mean_df.drop(['diagnosis','texture_se','perimeter_se','radius_se'],axis=1) ,width=0.3 , saturation=0.9,orient="h")
df.columns
plt.figure(figsize=(21,15))

sns.boxplot(data= sorted_mean_df[['radius_mean','symmetry_mean','compactness_se','concavity_se','concave points_se','texture_worst','smoothness_worst','compactness_worst','symmetry_worst', 'fractal_dimension_worst']],

                                   orient='h',

                                   width=0.2,

                                   saturation=0.9)
from scipy import stats

dd = pd.DataFrame()

dd,lambdaChanges = stats.boxcox(df['radius_mean'])

plt.figure(figsize=(21,5))

sns.boxplot(data = df['radius_mean'],orient='h',width=0.3)

plt.figure(figsize=(21,5))

sns.boxplot(data=dd,orient='h',width=0.3)
plt.figure(figsize=(15,5))

sns.distplot(df['radius_mean'])
from sklearn.preprocessing import StandardScaler

dataframe  = pd.DataFrame()

scaler = StandardScaler()

dataframe = scaler.fit_transform(df[['radius_mean','symmetry_mean','compactness_se','concavity_se','concave points_se','texture_worst','smoothness_worst','compactness_worst','symmetry_worst', 'fractal_dimension_worst']])

dataframe = pd.DataFrame(dataframe)

dataframe.columns = ['radius_mean','symmetry_mean','compactness_se','concavity_se','concave points_se','texture_worst','smoothness_worst','compactness_worst','symmetry_worst', 'fractal_dimension_worst']

dataframe.head()
plt.figure(figsize=(10,5))

sns.distplot(dataframe['radius_mean'])
plt.figure(figsize=(10,5))

sns.distplot(df['radius_mean'])