from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd

import seaborn as sns
import missingno as msno
from scipy import stats
sns.set(color_codes=True)
import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/indian-lok-sabha-2019-election-candidates/LokSabha2019.csv")
print(df.shape)
df.head()
df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T
df.isnull().sum()
df["Criminal Cases"].value_counts()
df["City"].value_counts()[:10]
ax2 = df.plot.scatter(x='Education',y='Criminal Cases', colormap='viridis')
df["Education"].hist(edgecolor = "black")
plt.xticks(rotation=90)
ax = sns.barplot(x="Criminal Cases", y="Education", data=df)
sns.pairplot(df)
df.Education.value_counts()["Post Graduate"]
df_post_grad_cand = df.loc[df['Education'] == "Post Graduate"]
df_post_grad_cand.head()
ax = sns.barplot(x="Criminal Cases", y="Age", data=df_post_grad_cand)
df_edu_low = df.loc[df['Education'] == "Illiterate"]
df_edu_low.head()
ax = sns.barplot(x="Criminal Cases", y="Age", data=df_edu_low)
df_crime = df.loc[df['Criminal Cases'] == df["Criminal Cases"].max()]
df_crime
df_crime = df.loc[df['Criminal Cases'] == df["Criminal Cases"].min()]
df_crime.head()