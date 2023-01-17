from string import ascii_letters

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df_govt = pd.read_csv("../input/GoodGovernment.csv")

df_continents = pd.read_csv("../input/ContinentCodes.csv", keep_default_na=False)
df_govt['density'] = df_govt['population'] / df_govt['surface area (Km2)']
df_merged = df_govt.merge(df_continents,

                         left_on=["ISO Country code"],

                         right_on=["ISO Code 3"])
df_merged.isnull().sum()
df_merged.dtypes
df_govt.columns.values
df_govt.dtypes
# Compute the correlation matrix

corr = df_govt.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# # Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}

           )
corr_strong = corr[(corr['human development index'] > 0.5) |

                  (corr['world happiness report score'] > 0.5) |

                  (corr['sustainable economic development assessment (SEDA)'] > 0.5)]



deletion = [column for column in corr.columns.values if column not in ['human development index',

                                                               'world happiness report score',

                                                               'sustainable economic development assessment (SEDA)']]

corr_strong = corr_strong.drop(deletion,axis=1)

corr_strong.head(100)
deletion = [column for column in corr.columns.values if column not in ['human development index',

                                                               'world happiness report score',

                                                               'sustainable economic development assessment (SEDA)']]

corr_all = corr.drop(deletion,axis=1)

corr_all.head(100)
df_govt_more_complete = df_govt.dropna()

df_govt_more_complete_positive = df_govt_more_complete[['human development index',

                                                        'world happiness report score', 'sustainable economic development assessment (SEDA)',

                                                       'GDP per capita (PPP)', 'government expenditure (% of GDP)', 

                                                        'health expenditure per person', 'education expenditure per person']]

g = sns.pairplot(df_govt_more_complete_positive, kind="reg")
corr_strong = corr[(corr['human development index'] < -0.5) |

                  (corr['world happiness report score'] < -0.5) |

                  (corr['sustainable economic development assessment (SEDA)'] < -0.5)]

corr_strong = corr_strong.drop(deletion,axis=1)

corr_strong.head(100)
sns.boxplot(x="sustainable economic development assessment (SEDA)", y="Continent Code", data=df_merged)
sns.boxplot(x="human development index", y="Continent Code", data=df_merged)
sns.boxplot(x="world happiness report score", y="Continent Code", data=df_merged)
aggregate_dictionary = {}

for column in df_merged.columns.values:

    if df_merged[column].dtype != object and column != 'ISO Number':

        aggregate_dictionary[column] = [min, max, np.mean, np.median]



df_continental = df_merged.groupby(['Continent Code']).agg(aggregate_dictionary)

df_continental.head()
df_merged['Continent Code'].value_counts()