import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



#Suppressing all warnings

warnings.filterwarnings("ignore")



%matplotlib inline
df = pd.read_csv('../input/windows-store/msft.csv')
df.head()
df.describe(include='all')
df.info()
df[df['Name'].isna()]
df.drop(5321, axis=0, inplace = True)
df["Name"].value_counts()[df["Name"].value_counts() > 1]
df.loc[df['Name'].isin(df["Name"].value_counts()[df["Name"].value_counts() > 1].index.values.tolist())].sort_values(by='Name')
sns.set(rc={'figure.figsize':(12,5)})

ax = sns.countplot(x="Category", data=df.sort_values(by='Category'), order=df.Category.value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.show()
len(df[df['Price']=='Free'])/len(df['Price'])*100
df.loc[~df["Price"].isin(['Free']), "Price"] = "Paid"

sns.set(rc={'figure.figsize':(12,5)})

ax = sns.countplot(x="Category", hue="Price", data=df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.show()
sns.set(rc={'figure.figsize':(12,5)})

ax = sns.countplot(x="Rating", data=df.sort_values(by='Rating'), order=df.Rating.value_counts().index)

#ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.show()
print("Mean Rating for Free Apps:", round(df[df['Price']=='Free'].Rating.mean(),2))

print("Mean Rating for Paid Apps:", round(df[df['Price']=='Paid'].Rating.mean(),2))

print("Overall Mean Rating:", round(df.Rating.mean(), 2))
df['Date'].dtype
df["Date"] = pd.to_datetime(df["Date"])
months=['January', 'February', 'March', 'April', 'May', 'June', 'July','August', 'September', 'October', 'November', 'December']

df['Launch Month']=[months[i.month-1] for i in df["Date"]]

df['Launch Year']=[i.year for i in df["Date"]]
sns.set(rc={'figure.figsize':(12,5)})

ax.set_title('Applications Launched Each Year')

ax = sns.countplot(x="Launch Year", data=df.sort_values(by='Launch Year'))

ax.axhline(df['Launch Year'].value_counts().mean(), color='green', linewidth=2)

ax.margins(0.05)

ax.annotate('Mean: {:0.2f}'.format(df['Launch Year'].value_counts().mean()), xy=(10.7, df['Launch Year'].value_counts().mean()+40),

            horizontalalignment='right', verticalalignment='center',

            )

#ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

plt.show()
sns.set(rc={'figure.figsize':(12,5)})

ax = sns.countplot(x="Launch Month", data=df.sort_values(by='Launch Month'), order=months)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

ax.axhline(df['Launch Month'].value_counts().mean(), color='green', linewidth=2)

ax.margins(0.05)

ax.annotate('Mean: {:0.2f}'.format(df['Launch Month'].value_counts().mean()), xy=(11.5, df['Launch Month'].value_counts().mean()+20),

            horizontalalignment='right', verticalalignment='center',

            )



plt.show()