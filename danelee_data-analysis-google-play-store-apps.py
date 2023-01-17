from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
from collections import Counter
import re
import os

from pylab import *
import pandas as pd
import seaborn as sns
df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
df.head()
df.shape
df.count()
df.info()
df2 = df.copy()
df2[["App", "Installs"]]
df2[["App", "Installs"]].info()
Counter(df2['Installs'])
# change "Installs" column to numeric

def conv_Installs(n_installs):
    if n_installs == 'Free':
        val = 0
    else:
        if '+' in n_installs:
            val = int(n_installs[:-1].replace(',', ''))
        else:
            val = int(n_installs)
    return val

df2["Installs"] = df2["Installs"].apply(lambda n_installs: conv_Installs(n_installs))
# sort by 'Installs', 'Rating'

df2 = df2.sort_values(by=["Installs", "Rating"], ascending=False)
# select the first n most popular "Apps"

rank = 1
prev_apps = []
for app, installs, rating in df2.iloc[:40][["App", "Installs", "Rating"]].values:
    if app not in prev_apps:
        print(f"Ranking {rank} | {app} | install: {installs}+ | rating: {rating}")
        prev_apps.append(app)
        rank += 1
Counter(df2["Size"])
# remove rows with 'Varies with device'

indices_varies_with_device = (df2["Size"] != "Varies with device").values
df2 = df2[indices_varies_with_device]
# change 'Size' column to numerical values

def size_coverter(size):
    size = size.lower()
    num, eng = float(size[:-1].replace(',', '')), size[-1]
    
    eng2multiplier = {'k': 1000, 'm': 1000000, '+': 1}
    
    numerical_size = num * eng2multiplier[eng]
    return numerical_size

df2["Size"] = df2["Size"].apply(lambda s: size_coverter(s))
n = 0
prev_apps = []
for app, size in df2[["App", "Size"]].sort_values("Size", ascending=False).values:
    if app not in prev_apps:
        print(f"Ranking {n+1} | {app} | size: {int(size)}")
        n += 1
        prev_apps.append(app)
        
        if n == 10:
            break
import datetime
Counter(df2["Last Updated"])
def covt_date(unique_date):
    try:
        month, day, year = re.findall(r"(\w+) (\d+), (\d+)", unique_date)[0]
        pydate = datetime.datetime.strptime(f'{year}-{month}-{day}', '%Y-%B-%d')
    except IndexError:
        pydate = np.nan
    return pydate
df2["Last Updated"] = df2["Last Updated"].apply(lambda date: covt_date(date))
df2.shape
# remove rows with 'nan' 

df2 = df2.dropna(subset=["Last Updated"])
df2.shape
rank = 1
for app, last_updated in df2.sort_values(by="Last Updated").iloc[:10][["App", "Last Updated"]].values:
    print(f"Ranking {rank} | Last Updated: {last_updated} | App: {app}")
    rank += 1
Counter(df["Category"])
df["Category"].values.tolist().index('1.9')
df = df.drop([10472])  # removes typoed-row
Counter(df["Rating"])
df = df.dropna(subset=["Rating"])
# confirms that Nan is removed.

Counter(df["Rating"])  
Counter(df["Reviews"])
df["Reviews"] = df["Reviews"].astype(np.int)
Counter(df["Size"])
# remove rows with 'Varies with device'

indices_varies_with_device = (df["Size"] != "Varies with device").values
df = df[indices_varies_with_device]
# change 'Size' column to numerical values

def size_coverter(size):
    size = size.lower()
    num, eng = float(size[:-1].replace(',', '')), size[-1]
    
    eng2multiplier = {'k': 1000, 'm': 1000000, '+': 1}
    
    numerical_size = num * eng2multiplier[eng]
    return numerical_size

df["Size"] = df["Size"].apply(lambda s: size_coverter(s))
# change "Installs" column to numeric

def conv_Installs(n_installs):
    if n_installs == 'Free':
        val = 0
    else:
        if '+' in n_installs:
            val = int(n_installs[:-1].replace(',', ''))
        else:
            val = int(n_installs)
    return val

df["Installs"] = df["Installs"].apply(lambda n_installs: conv_Installs(n_installs))
df["Type"].unique()
df["Price"].unique()
def convt_price(price: str):
    price = price.replace("$", '')
    price = np.float(price)
    return price
df["Price"] = df["Price"].apply(lambda price: convt_price(price))
df["Content Rating"].unique()
df["Genres"].unique()
def covt_date(unique_date):
    try:
        month, day, year = re.findall(r"(\w+) (\d+), (\d+)", unique_date)[0]
        pydate = datetime.datetime.strptime(f'{year}-{month}-{day}', '%Y-%B-%d')
    except IndexError:
        pydate = np.nan
    return pydate
df["Last Updated"] = df["Last Updated"].apply(lambda date: covt_date(date))
# drop this column as 'Last Update' may represent this.
df = df.drop(columns=["Current Ver"])
df["Android Ver"].unique()
# drop rows with 'Varies with device'

df = df[df["Android Ver"] != "Varies with device"]
df.info()  # after cleaned up
category = pd.DataFrame()
category['category'] = Counter(df["Category"]).keys()
category['val'] = Counter(df["Category"]).values()
category = category.set_index("category")

category.plot.pie(y="val", figsize=(20, 5), );
plt.legend(loc="upper right", bbox_to_anchor=(2, 2));
sns.distplot(df.Rating);
# correlation

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)
sns.pairplot(df);
sns.catplot("Category", "Rating", data=df, kind="boxen", aspect=3);
plt.xticks(rotation=70);
sns.catplot("Type", "Rating", data=df, kind="boxen", aspect=0.7);
sns.catplot("Type", "Rating", data=df, kind="boxen", 
            hue="Content Rating",
            aspect=2);
sns.catplot("Genres", "Rating", data=df, kind="boxen", aspect=5)
plt.xticks(rotation=70);
sns.catplot("Genres", "Rating", data=df, kind="boxen", 
            col="Content Rating", col_wrap=1,
            aspect=5)
plt.xticks(rotation=70);
sns.relplot("Last Updated", "Rating", data=df, alpha=0.3);
sns.catplot("Android Ver", "Rating", data=df, kind="boxen", aspect=3)
plt.xticks(rotation=70);