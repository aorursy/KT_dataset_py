project_name = "GooglePlayStoreAnalysis"
# Imports
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')
googlestore_df =  pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
rows = googlestore_df.shape[0]
column = googlestore_df.shape[1]
print('There are {} Rows and {} Columns in the dataset'.format(rows, column))
googlestore_df.head(10)
googlestore_df.info()
googlestore_df.columns
googlestore_df.isnull().sum()
def printinfo():
    temp = pd.DataFrame(index=googlestore_df.columns)
    temp['data_type'] = googlestore_df.dtypes
    temp['null_count'] = googlestore_df.isnull().sum()
    temp['unique_count'] = googlestore_df.nunique()
    return temp
printinfo()
googlestore_df[googlestore_df.Rating.isnull()]
googlestore_df[googlestore_df.Type.isnull()]
googlestore_df['Type'].fillna("Free", inplace = True)
googlestore_df.isnull().sum()
googlestore_df[googlestore_df['Content Rating'].isnull()]
googlestore_df.loc[10468:10477, :]
googlestore_df.dropna(subset = ['Content Rating'], inplace=True)
googlestore_df.drop(['Current Ver','Last Updated', 'Android Ver'], axis=1, inplace=True)
googlestore_df.head()
modeValueRating = googlestore_df['Rating'].mode()
modeValueRating[0]
googlestore_df['Rating'].fillna(value=modeValueRating[0], inplace = True)
printinfo()
googlestore_df['Reviews'] = googlestore_df.Reviews.astype(int)
printinfo()
googlestore_df['Size'] = googlestore_df.Size.apply(lambda x: x.strip('+'))# Removing the + Sign
googlestore_df['Size'] = googlestore_df.Size.apply(lambda x: x.replace(',', ''))# For removing the `,`
googlestore_df['Size'] = googlestore_df.Size.apply(lambda x: x.replace('M', 'e+6'))# For converting the M to Mega
googlestore_df['Size'] = googlestore_df.Size.apply(lambda x: x.replace('k', 'e+3'))# For convertinf the K to Kilo
googlestore_df['Size'] = googlestore_df.Size.replace('Varies with device', np.NaN)
googlestore_df['Size'] = pd.to_numeric(googlestore_df['Size']) # Converting the string to Numeric type
printinfo()
googlestore_df.dropna(subset = ['Size'], inplace=True)
printinfo()
googlestore_df['Installs'] = googlestore_df.Installs.apply(lambda x: x.strip('+'))
googlestore_df['Installs'] = googlestore_df.Installs.apply(lambda x: x.replace(',', ''))
googlestore_df['Installs'] = pd.to_numeric(googlestore_df['Installs'])
printinfo()
googlestore_df['Price'].value_counts()
googlestore_df['Price'] = googlestore_df.Price.apply(lambda x: x.strip('$'))
googlestore_df['Price'] = pd.to_numeric(googlestore_df['Price'])
printinfo()
googlestore_df.describe()
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'
printinfo()
googlestore_df['Category'].value_counts()
y = googlestore_df['Category'].value_counts().index
x = googlestore_df['Category'].value_counts()
xsis = []
ysis = []
for i in range(len(x)):
    xsis.append(x[i])
    ysis.append(y[i])
plt.figure(figsize=(18,13))
plt.xlabel("Count")
plt.ylabel("Category")

graph = sns.barplot(x = xsis, y = ysis, palette= "husl")
graph.set_title("Top categories on Google Playstore", fontsize = 25);
x2 = googlestore_df['Content Rating'].value_counts().index
y2 = googlestore_df['Content Rating'].value_counts()

x2sis = []
y2sis = []
for i in range(len(x2)):
    x2sis.append(x2[i])
    y2sis.append(y2[i])
plt.figure(figsize=(12,10))
plt.bar(x2sis,y2sis,width=0.8,color=['#15244C','#FFFF48','#292734','#EF2920','#CD202D','#ECC5F2'], alpha=0.8);
plt.title('Content Rating',size = 20);
plt.ylabel('Apps(Count)');
plt.xlabel('Content Rating');
googlestore_df['Rating'].describe()
plt.figure(figsize=(15,9))
plt.xlabel("Rating")
plt.ylabel("Frequency")
graph = sns.kdeplot(googlestore_df.Rating, color="Blue", shade = True)
plt.title('Distribution of Rating',size = 20);
plt.figure(figsize=(10,10))
labels = googlestore_df['Type'].value_counts(sort = True).index
sizes = googlestore_df['Type'].value_counts(sort = True)
colors = ["blue","lightgreen"]
explode = (0.2,0)
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=0)
plt.title('Percent of Free Vs Paid Apps in store',size = 20)
plt.show()
highest_Installs_df = googlestore_df.groupby('Category')[['Installs']].sum().sort_values(by='Installs', ascending=False)
highest_Installs_df.head()
x2sis = []
y2sis = []

for i in range(len(highest_Installs_df)):
    x2sis.append(highest_Installs_df.Installs[i])
    y2sis.append(highest_Installs_df.index[i])

plt.figure(figsize=(18,13))

plt.xlabel("Installs")
plt.ylabel("Category")
graph = sns.barplot(x = x2sis, y = y2sis, alpha =0.9, palette= "viridis")
graph.set_title("Installs", fontsize = 25);
def findtop10incategory(str):
    str = str.upper()
    top10 = googlestore_df[googlestore_df['Category'] == str]
    top10apps = top10.sort_values(by='Installs', ascending=False).head(10)
    # Top_Apps_in_art_and_design
    plt.figure(figsize=(15,12))
    plt.title('Top 10 Installed Apps',size = 20);    
    graph = sns.barplot(x = top10apps.App, y = top10apps.Installs)
    graph.set_xticklabels(graph.get_xticklabels(), rotation= 45, horizontalalignment='right');
findtop10incategory('Sports')
top10PaidApps = googlestore_df[googlestore_df['Type'] == 'Paid'].sort_values(by='Price', ascending=False).head(11)
# top10PaidApps
top10PaidApps_df = top10PaidApps[['App', 'Installs']].drop(9934)
plt.figure(figsize=(15,12));
plt.pie(top10PaidApps_df.Installs, explode=None, labels=top10PaidApps_df.App, autopct='%1.1f%%', startangle=0);
plt.title('Top Expensive Apps Distribution',size = 20);
plt.legend(top10PaidApps_df.App, 
           loc="lower right",
           title="Apps",
           fontsize = "xx-small"
          );
Apps_with_Highest_rev = googlestore_df.sort_values(by='Reviews', ascending=False).head(20)
Apps_with_Highest_rev
topAppsinGenres = googlestore_df['Genres'].value_counts().head(50)
x3sis = []
y3sis = []

for i in range(len(topAppsinGenres)):
    x3sis.append(topAppsinGenres.index[i])
    y3sis.append(topAppsinGenres[i])
plt.figure(figsize=(15,9))
plt.ylabel('Genres(App Count)')
plt.xlabel('Genres')
graph = sns.barplot(x=x3sis,y=y3sis,palette="deep")
graph.set_xticklabels(graph.get_xticklabels(), rotation=90, fontsize=12)
graph.set_title("Top Genres in the Playstore", fontsize = 20);
Paid_Apps_df = googlestore_df[googlestore_df['Type'] == 'Paid']
earning_df = Paid_Apps_df[['App', 'Installs', 'Price']]
earning_df['Earnings'] = earning_df['Installs'] * earning_df['Price'];
earning_df_sorted_by_Earnings = earning_df.sort_values(by='Earnings', ascending=False).head(50)
earning_df_sorted_by_Price = earning_df_sorted_by_Earnings.sort_values(by='Price', ascending=False)
# PLot a bar chart of earning at y and app names at x
plt.figure(figsize=(15,9))
plt.bar(earning_df_sorted_by_Price.App, earning_df_sorted_by_Price.Earnings, width=1.1, label=earning_df_sorted_by_Price.Earnings)
plt.xlabel("Apps")
plt.ylabel("Earnings")
plt.tick_params(rotation=90)
plt.title("Top Earning Apps");
# jovian.commit(message="Completed")
