import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline
train_df = pd.read_csv("../input/train.csv")

train_df.head()
train_df.describe()
labels = []

values = []

for col in train_df.columns:

    labels.append(col)

    values.append(train_df[col].isnull().sum())

    print(col, values[-1])
train_copy = train_df

train_copy = train_copy.replace(-1, np.NaN)
import missingno as msno

msno.matrix(df=train_copy.iloc[:,2:42], figsize=(20, 14), color=(0.42, 0.1, 0.05))
plt.figure(figsize=(12,8))

sns.countplot(x="target", data=train_df, color=color[0])

plt.ylabel('Count', fontsize=12)

plt.xlabel('Target', fontsize=12)

plt.xticks(rotation='vertical')

plt.title("Frequency of targets (0 : Claim not filed; 1 : Claim filed)", fontsize=15)

plt.show()
train_float = train_df.select_dtypes(include=['float64'])

train_int = train_df.select_dtypes(include=['int64'])

colormap = plt.cm.inferno

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of continuous features', y=1.05, size=15)

sns.heatmap(train_float.corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)