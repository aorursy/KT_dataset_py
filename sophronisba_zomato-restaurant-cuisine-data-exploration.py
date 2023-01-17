import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# begin by loading file and merging with country code
df = pd.read_csv('../input/zomato.csv',encoding="ISO-8859-1")
country = pd.read_excel('../input/Country-Code.xlsx')
zomato_df = pd.merge(df, country, on='Country Code')
zomato_df.head()
zomato_df.describe()
nans = lambda df: df[df.isnull().any(axis=1)]
nans(zomato_df)
zomato_df = zomato_df.dropna()
nans(zomato_df)
zomato_df.hist("Aggregate rating", bins=25)
zeroes = zomato_df[zomato_df["Aggregate rating"] == 0]
zeroes["Rating color"].value_counts()
zeroes["Rating text"].value_counts()
zomato_df = zomato_df[zomato_df["Aggregate rating"] > 0]
zomato_df.describe()
zomato_df[zomato_df["Average Cost for two"] == 0]
zomato_df[zomato_df["Price range"] == 1]["Average Cost for two"].hist()
zomato_df = zomato_df[zomato_df["Average Cost for two"] > 0]
zomato_df.describe()
zomato_df.groupby('Currency')['Average Cost for two'].mean()
# I am not interested in all of these columns, so I am going to drop some
drop_columns = ["Restaurant ID", "Country Code", "Locality Verbose", "Latitude", "Longitude", "Rating color", "Currency", "Average Cost for two", "Address", "Locality", "Switch to order menu"]
zomato_df = zomato_df[zomato_df.columns.drop(drop_columns)]
zomato_df.head()
zomato_df["Cuisine count"] = zomato_df["Cuisines"].apply(lambda x: x.count(",") + 1)
zomato_df.head()
zomato_df.describe()
zomato_df["Cuisine count"].hist(bins = 8) # 8 bins because the cuisine count ranges from one to eight
zomato_df["Country"].value_counts()
zomato_df["Offers Indian food"] = zomato_df["Cuisines"].apply(lambda x: 'Yes' if x.find("Indian") >= 0 else 'No')
zomato_df["India"] = zomato_df["Country"].apply(lambda x: 'India' if x == "India" else 'outside India')
zomato_df["Multiple cuisines"] = zomato_df["Cuisine count"].apply(lambda x: 'Yes' if x > 1 else 'No')

avg_rating = zomato_df["Aggregate rating"].mean()
zomato_df["Relative rating"] = zomato_df["Aggregate rating"].apply(lambda x: 'Above average' if x > avg_rating else 'Average or below')
zomato_df.head()
sns.countplot(x = "Relative rating", hue = "Offers Indian food", data = zomato_df)
sns.countplot(x = "Price range", hue = "Offers Indian food", data = zomato_df)
sns.countplot(x = "Price range", hue = "Relative rating", data = zomato_df)
zomato_df.plot(x='Price range', y='Aggregate rating', kind='scatter', figsize=(12,6))
plt.title('Price range vs. Rating')
sns.countplot(x = "Relative rating", hue = "Multiple cuisines", data = zomato_df)
sns.countplot(x = "Relative rating", hue = "Cuisine count", data = zomato_df)
zomato_df["Multiple cuisines"] = zomato_df["Cuisine count"].apply(lambda x: str(x) if x <= 3 else '> 3')
cuisine_count_order = ['1', '2', '3', '> 3']
sns.countplot(x = "Relative rating", hue = "Multiple cuisines", data = zomato_df, hue_order = cuisine_count_order)
zomato_df.plot(x='Votes', y='Aggregate rating', kind='scatter', figsize=(12,6))
plt.title('Votes vs. Rating')
sns.countplot(x = "Relative rating", hue = "Has Table booking", data = zomato_df)
sns.countplot(x = "Relative rating", hue = "Has Online delivery", data = zomato_df)
sns.countplot(x = "Relative rating", hue = "Is delivering now", data = zomato_df)