import pandas as pd
import numpy as np
drinks_df = pd.read_csv("../input/drinks-by-country/drinksbycountry.csv")
drinks_df.head()
drinks_df.info()
drinks_df.describe()
drinks_df.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
%matplotlib inline
sns.pairplot(drinks_df, hue='continent', height=3, aspect=1)
sns.pairplot(drinks_df, hue='continent', height=3, diag_kind="hist");
drinks_df.hist(edgecolor='black', linewidth=1.2, figsize=(12, 8));
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
sns.violinplot(x='continent', y='beer_servings', data=drinks_df)

plt.subplot(2, 2, 2)
sns.violinplot(x='continent', y='spirit_servings', data=drinks_df)

plt.subplot(2, 2, 3)
sns.violinplot(x='continent', y='wine_servings', data=drinks_df)

plt.subplot(2, 2, 4)
sns.violinplot(x='continent', y='total_litres_of_pure_alcohol', data=drinks_df);
drinks_df.boxplot(by='continent', figsize=(12, 8));
pd.plotting.scatter_matrix(drinks_df, figsize=(12, 10));
sns.pairplot(drinks_df)
plt.figure(figsize=(12, 8))
sns.heatmap(drinks_df.corr(), linewidths=1, annot=True)