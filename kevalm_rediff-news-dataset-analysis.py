import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv("../input/rediff_subset_with_cities.csv", header='infer')
data.head()
#Unique education values
data.source.unique()
#Unique crawl_time values
data.crawl_time.unique()
#Result displays almost unique time for every record
#Unique education values
data.cities.unique()
print(data.shape)
data.isnull().values.any() #checking NaN is in dataframe
data2 = data.dropna()
data2.isnull().values.any() #checking NaN is in dataframe
# #City slice plotting

# df_city = data2[data2['summary'].str.contains('^city\.[a-z]+$', regex=True)]
# df_city['city_name'] = df_city.summary.str[5:]

# city_list=df_city['city_name'].unique().tolist()
# print (city_list)

# #bar plot of all cities
# grp_city = df_city.groupby(['city_name'])['summary'].count().nlargest(50)
# ts = pd.Series(grp_city)
# ts.plot(kind='bar', figsize=(20,10),title='Articles per city')
# plt.show()
# #pie chart of top 40 cities
# grp_top_city = df_city.groupby(['city_name'])['summary'].count().nlargest(40)
# ts = pd.Series(grp_top_city)
# ts.plot(kind='pie', figsize=(10,10),title='Share of Top 40 Cities')
# plt.show()
data2.source.unique()
# le = LabelEncoder()
# source = le.fit_transform(data2['source'])
# le.fit(data2['source'])
#data2.source.astype("category").cat.codes
data2.source = pd.Categorical(data2.source).codes
data2.head()
data2['source'].value_counts().plot(kind='bar')

