import numpy as np
import pandas as pd
pd.set_option("display.precision", 2)
import seaborn as sns
sns.set_context(
    "notebook",
    font_scale = 1.5,
    rc = {
        "figure.figsize": (11,8),
    "axes.titlesize": 18
    }
)
import matplotlib.pyplot as plt

#To remove warnings
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/videogamesales/vgsales.csv')
print(df.head())
print(df.shape)
print(df.info())
df = df.dropna()
df['Year'] = df['Year'].astype('int64')
df.shape
df = df[df['Year'] <= 2016]
df.head()
df.describe()
df = df.sort_values(by='Global_Sales', ascending=False)
df['Rank'] = df.index + 1
df.head()
ax = sns.scatterplot(x = 'Rank', y = 'Global_Sales', data=df);
ax.set_ylabel("Global Sales");
ax = sns.lineplot(x='Year', y = 'Global_Sales', data=df, estimator = np.median, ci = False)
ax.set_ylabel("Median Global Sales (Millions)");
df_sales = df[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
ax = sns.lineplot(x = 'Year', y = 'value', data = pd.melt(df_sales,'Year'), hue = 'variable', 
             estimator = np.median, ci=False);
ax.set_ylabel("Median Sales (Millions)")
ax.legend().texts[0].set_text("Region");
ax = sns.barplot(x="Genre", y="Global_Sales", data=df, ci=False, estimator = np.median)
ax.set_ylabel("Global Sales (Millions)");
plt.xticks(rotation = 90)
plt.figure(figsize=(20,10))
ax = sns.lineplot(x='Year', y = 'Global_Sales', data=df, estimator = np.median, ci = False, hue = "Genre")
ax.set_ylabel("Median Global Sales (Millions)");
plt.figure(figsize=(20,10))
ax = sns.lineplot(x='Year', y = 'Global_Sales', data=df, estimator = np.median, ci = False, hue = "Genre")
ax.set_ylabel("Median Global Sales (Millions)");
ax.set(xlim=(2000,2016), ylim=(0,1))
plt.figure(figsize=(20,10))
ax = sns.barplot(x = "Platform", y = "Global_Sales", data = df, ci = False, estimator = np.median)
ax.set_ylabel("Median Global Sales (Millions)");
plt.figure(figsize=(20,10))
platform_group = df.groupby("Platform")
quantity_sold = platform_group.sum()['Global_Sales']
platforms = [platform for platform, df in platform_group]
plt.bar(platforms, quantity_sold)
plt.ylabel("Total Global Sales (Millions)")
plt.xlabel('Platform')
#plt.show()