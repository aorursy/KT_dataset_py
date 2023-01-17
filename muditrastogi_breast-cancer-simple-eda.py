import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None 
data = pd.read_csv('../input/data.csv')
data.info()
data.head()
data.describe()
cnt_diagonis = data.diagnosis.value_counts()

plt.figure(figsize=(12,8))

sns.barplot(cnt_diagonis.index, cnt_diagonis.values, alpha=0.8, color=color[9])

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Diagnosis ', fontsize=12)

plt.title('Count of rows in each dataset', fontsize=15)

plt.xticks(rotation='vertical')

plt.show()
data.columns

datanew = data.drop('Unnamed: 32', axis =1)

datanew = datanew.drop('diagnosis', axis =1)
datanew.columns
corr = datanew.corr()

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=1, cbar_kws={"shrink": 1})
sns.set(style="white", palette="muted", color_codes=True)

rs = np.random.RandomState(10)



# Set up the matplotlib figure

f, axes = plt.subplots(5, 2, figsize=(10, 20), sharex=True)

sns.despine(left=True)

d = rs.normal(size=100)



sns.swarmplot(x="id", y="radius_mean", hue="diagnosis", data=data, ax=axes[0, 0]);

sns.swarmplot(x="id", y="texture_mean", hue="diagnosis", data=data, ax=axes[0, 1]);

sns.swarmplot(x="id", y="perimeter_mean", hue="diagnosis", data=data, ax=axes[1, 0]);

sns.swarmplot(x="id", y="area_mean", hue="diagnosis", data=data, ax=axes[1, 1]);

sns.swarmplot(x="id", y="smoothness_mean", hue="diagnosis", data=data, ax=axes[2, 0]);

sns.swarmplot(x="id", y="compactness_mean", hue="diagnosis", data=data, ax=axes[2, 1]);

sns.swarmplot(x="id", y="concavity_mean", hue="diagnosis", data=data, ax=axes[3, 0]);

sns.swarmplot(x="id", y="concave points_mean", hue="diagnosis", data=data, ax=axes[3, 1]);

sns.swarmplot(x="id", y="symmetry_mean", hue="diagnosis", data=data, ax=axes[4, 0]);

sns.swarmplot(x="id", y="fractal_dimension_mean", hue="diagnosis", data=data, ax=axes[4, 1]);



plt.setp(axes, yticks=[])

plt.tight_layout()
data.columns
sns.set(style="white", palette="muted", color_codes=True)

rs = np.random.RandomState(10)



# Set up the matplotlib figure

f, axes = plt.subplots(5, 2, figsize=(10, 20), sharex=True)

sns.despine(left=True)

d = rs.normal(size=100)



sns.swarmplot(x="id", y="radius_se", hue="diagnosis", data=data, ax=axes[0, 0]);

sns.swarmplot(x="id", y="texture_se", hue="diagnosis", data=data, ax=axes[0, 1]);

sns.swarmplot(x="id", y="perimeter_se", hue="diagnosis", data=data, ax=axes[1, 0]);

sns.swarmplot(x="id", y="area_mean", hue="diagnosis", data=data, ax=axes[1, 1]);

sns.swarmplot(x="id", y="smoothness_se", hue="diagnosis", data=data, ax=axes[2, 0]);

sns.swarmplot(x="id", y="compactness_se", hue="diagnosis", data=data, ax=axes[2, 1]);

sns.swarmplot(x="id", y="concavity_se", hue="diagnosis", data=data, ax=axes[3, 0]);

sns.swarmplot(x="id", y="concave points_se", hue="diagnosis", data=data, ax=axes[3, 1]);

sns.swarmplot(x="id", y="symmetry_se", hue="diagnosis", data=data, ax=axes[4, 0]);

sns.swarmplot(x="id", y="fractal_dimension_se", hue="diagnosis", data=data, ax=axes[4, 1]);



plt.setp(axes, yticks=[])

plt.tight_layout()
sns.set(style="white", palette="muted", color_codes=True)

rs = np.random.RandomState(10)



# Set up the matplotlib figure

f, axes = plt.subplots(5, 2, figsize=(10, 20), sharex=True)

sns.despine(left=True)

d = rs.normal(size=100)



sns.swarmplot(x="id", y="radius_worst", hue="diagnosis", data=data, ax=axes[0, 0]);

sns.swarmplot(x="id", y="texture_worst", hue="diagnosis", data=data, ax=axes[0, 1]);

sns.swarmplot(x="id", y="perimeter_worst", hue="diagnosis", data=data, ax=axes[1, 0]);

sns.swarmplot(x="id", y="area_worst", hue="diagnosis", data=data, ax=axes[1, 1]);

sns.swarmplot(x="id", y="smoothness_worst", hue="diagnosis", data=data, ax=axes[2, 0]);

sns.swarmplot(x="id", y="compactness_worst", hue="diagnosis", data=data, ax=axes[2, 1]);

sns.swarmplot(x="id", y="concavity_worst", hue="diagnosis", data=data, ax=axes[3, 0]);

sns.swarmplot(x="id", y="concave points_worst", hue="diagnosis", data=data, ax=axes[3, 1]);

sns.swarmplot(x="id", y="symmetry_worst", hue="diagnosis", data=data, ax=axes[4, 0]);

sns.swarmplot(x="id", y="fractal_dimension_worst", hue="diagnosis", data=data, ax=axes[4, 1]);



plt.setp(axes, yticks=[])

plt.tight_layout()