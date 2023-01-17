import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import seaborn as sns # for data visualisation

import matplotlib.pyplot as plt # for data visualisation
df=pd.read_csv("../input/habermans-survival-data-set/haberman.csv")

df.head(5)
df=pd.read_csv("../input/habermans-survival-data-set/haberman.csv",names=['age','op_year','axil_nodes','sur_status'])

df.head(5)
print(df.shape)

print("Dataset has {} rows and {} columns".format(df.shape[0],df.shape[1]))
df.info()
df.describe()
df['sur_status'].value_counts()
df.nunique()
df['age'].unique()
df['op_year'].unique()
df['axil_nodes'].unique()
df['sur_status'].unique()
plt.figure(figsize=[16,16])

plt.subplot(221)

sns.distplot(df['age'])

plt.subplot(222)

sns.distplot(df['op_year'])

plt.subplot(223)

sns.distplot(df['axil_nodes'])

plt.subplot(224)

sns.distplot(df['sur_status'])

plt.show()
sns.boxplot(x='sur_status',y='age', data=df)

plt.show()
sns.boxplot(x='sur_status',y='axil_nodes', data=df)

plt.show()
sns.boxplot(x='sur_status',y='age', data=df)

plt.show()
sns.pairplot(df,hue="sur_status",height=3)
sns.jointplot(x='sur_status',y='age', data=df, kind="kde");
sns.jointplot(x='sur_status',y='op_year', data=df, kind="kde");
sns.jointplot(x='sur_status',y='axil_nodes', data=df, kind="kde");