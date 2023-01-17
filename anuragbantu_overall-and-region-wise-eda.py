import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import matplotlib.colors as mcolors
df = pd.read_csv('../input/videogamesales/vgsales.csv')
sns.set(rc={'figure.figsize':(15,10)})
sns.countplot(x = "Platform", data = df)
sns.countplot(x = "Genre", data = df)
df1 = df.head(100)
ax = sns.countplot(x="Publisher", data=df1)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
sns.set(rc={'figure.figsize':(15,10)})
sns.countplot(x = "Genre", data = df1)
df_na = df.sort_values(by = 'NA_Sales', ascending=False)

df_na100 = df_na.head(100)
sns.set(rc={'figure.figsize':(15,10)})
sns.countplot(x = "Genre", data = df_na100)
ax = sns.countplot(x="Publisher", data=df_na100)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
sns.set(rc={'figure.figsize':(15,10)})
sns.countplot(x = "Platform", data = df_na100)
df_eu = df.sort_values(by = 'EU_Sales', ascending=False)

df_eu100 = df_eu.head(100)
sns.countplot(x = "Genre", data = df_eu100)
ax = sns.countplot(x="Publisher", data=df_eu100)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
sns.countplot(x = "Platform", data = df_eu100)
df_jp = df.sort_values(by = 'JP_Sales', ascending=False)
df_jp100 = df_jp.head(100)
sns.countplot(x = "Genre", data = df_jp100)
ax = sns.countplot(x="Publisher", data=df_jp100)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
sns.countplot(x = "Platform", data = df_jp100)