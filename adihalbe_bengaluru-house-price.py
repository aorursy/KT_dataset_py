# import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)





df = pd.read_csv('../input/Bengaluru_House_Data.csv')

df.head()
print("This dataset has {} rows and {} columns.".format(*df.shape))

print("This dataset contains {} duplicates.".format(df.duplicated().sum()))
df.isnull().sum()
new_df = df.fillna({'society': 'unknown',

                   'balcony': 0, 'bath': 0,

                   'size': 'unknown', 'location': 'unknown'})

new_df.head()
new_df.nunique()
# Check the types of data

new_df.dtypes
new_df.isnull().values.sum()
new_df.describe()
# Finding out the correlation between the features

correlations =new_df.corr()

correlations['price']
cor_target = abs(correlations['price'])



# Display features with correlation < 0.1

removed_features = cor_target[cor_target < 0.1]

removed_features
fig_1 = plt.figure(figsize=(10, 8))

new_correlations = new_df.corr()

sns.heatmap(new_correlations, annot=True, cmap='Accent', annot_kws={'size': 8})

plt.title('Pearson Correlation Matrix')

plt.show()