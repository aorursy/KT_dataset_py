import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy import stats
# Importing the data from file:
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/googleplaystore.csv")

# Exploring the data set to better understand what're we facing
df.head()
df.shape
df.info()
df.describe()
df = df.drop(10472)
df.Category.unique()
df.Size.unique() # should be translated to ints
df.Installs.unique() # categorical, perhaps should be left as is
df.Price.unique() # should be translated to float 
df['Content Rating'].unique()
df['Last Updated'].unique() # should be converted to datetime
df['Android Ver'].unique()  # should be considered to convert 
df['Current Ver'] # should be considered to convert
df.Genres.unique() # seems like there's some repetition. 
# dropping duplicates from the datasets based on key fields 
df.drop_duplicates(subset=['App','Category','Current Ver','Last Updated'],inplace=True)

# changing last-update column to datetime
df['Last Updated'] =  pd.to_datetime(df['Last Updated'], format='%B %d, %Y')

# converting size to integers - 
#starting off by converting k and M to exponential based, removing "Varies with device", and converting it all to numeric:
df.Size=df.Size.str.replace('k','e+3')
df.Size=df.Size.str.replace('M','e+6')
df.Size=df.Size.replace('Varies with device',np.nan)
df.Size=pd.to_numeric(df.Size)

# converting reviews to integer - 
df.Reviews=pd.to_numeric(df.Reviews)

# converting price to float - 
df.Price=df.Price.str.replace('$','')
df.Price=pd.to_numeric(df.Price)

# converting installs to numeric (will be "at least X installed") - 
df.Installs = df.Installs.str.replace('+','')
df.Installs = df.Installs.str.replace(',','')
df.Installs = pd.to_numeric(df.Installs)

# to handle repetition in genres, we'll remove anything after ; and classify 
# based on what's coming before the ';'
df.Genres = df.Genres.str.split(';').str[0]
df.head()
sns.set(font_scale=1.5)
# Initial relationship view - pairplot on all variables
sns.pairplot(df, hue = 'Type')
# Free vs Paid apps:
sns.countplot(x="Type", data=df)
# Distribution downloads per type (free/paid)
sns.countplot(x="Installs", data=df, hue="Type")
fig, ax = plt.subplots()
# limiting the number of reviews to 50k, to make the histogram more readable. The vast majority of review counts is far less than 50k
sns.distplot(df[(df["Type"]=='Free') & (df["Reviews"]<50000)].Reviews,ax=ax,
                label="Free Apps",color='b',kde=False)
sns.distplot(df[(df["Type"]=='Paid') & (df["Reviews"]<50000)].Reviews,ax=ax,
                label="Paid Apps",color='g',kde=False)
ax.set(ylabel='Review Count')
plt.legend()
plt.show()
# Rating distribution:
sns.distplot(df['Rating'].dropna())
# Look at rating per genre:
temp0 = df.groupby(["Genres"]).mean().reset_index().sort_values('Rating', ascending=False)
sns.barplot(x="Genres", y="Rating", data=temp0.iloc[np.r_[0:5, -5:0]], palette="GnBu_d")
fig, ax = plt.subplots()
sns.distplot(df[df["Type"]=='Free'].Rating.dropna(),ax=ax,
                label="Free Apps",color='b',kde=False)
sns.distplot(df[df["Type"]=='Paid'].Rating.dropna(),ax=ax,
                label="Paid Apps",color='g',kde=False)
ax.set(ylabel='Rating Count')
plt.legend()
plt.show()
t, p = stats.ttest_ind(df[df["Type"]=='Free'].Rating.dropna(),
                       df[df["Type"]=='Paid'].Rating.dropna(), 
                       equal_var=False)
# start off by looking at size distribution
df.Size.isnull().values.any()
ax = sns.distplot((df['Size']/1e6).dropna(),kde=False, bins = 100)
ax.set(xlabel='Size in Mb',ylabel='Count')
ax1 = sns.jointplot(x='Size',y='Installs',data=df[df["Installs"]<10000001])

ax2 = sns.jointplot(x=(df[df["Reviews"]<50000]['Size']/1000000),
                   y=df[df["Reviews"]<50000].Reviews)
fig, (ax3, ax4) = plt.subplots(nrows=2, sharex=False)
temp1 = df.groupby(["Category"]).mean().reset_index().sort_values('Installs', ascending=False)
sns.barplot(x="Category", y="Installs", data=temp1[temp1.Installs>5000000], palette="BuGn_r", ax=ax3).set_title('Installs per Category')
temp2 = df.groupby(["Genres"]).mean().reset_index().sort_values('Installs', ascending=False)
sns.barplot(x="Genres", y="Installs", data=temp2[temp2.Installs>5000000], palette = "GnBu_d",ax=ax4).set_title('Installs per Genre')
plt.show()
sns.scatterplot(x="Installs",y="Rating", data =temp2[temp2.Installs>5000000], hue="Genres", palette="GnBu_d", s=500)
sns.boxplot(x="Installs", y="Reviews", data=df[(df["Installs"]>5000000)])
corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True)