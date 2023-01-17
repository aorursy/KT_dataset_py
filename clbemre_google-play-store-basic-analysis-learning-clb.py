import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
dataStore = pd.read_csv('../input/googleplaystore.csv') 
dataStore = dataStore.dropna(axis=0,how='any')
dataUserReview = pd.read_csv('../input/googleplaystore_user_reviews.csv')
dataStore.info()
dataStore.count()
dataStore.describe()
dataUserReview.info()
dataStore.groupby('Category').size()
dataStore.corr()
dataUserReview.corr()
f,ax = plt.subplots(figsize=(3,3))
sns.heatmap(dataStore.corr(), annot=True, linewidths=.5,fmt = '.2f', ax=ax)
plt.show()
f ,ax = plt.subplots(figsize=(2,2))
sns.heatmap(dataUserReview.corr(),annot=True,linewidths=1.0,fmt='.1f',ax=ax)
plt.show()
dataStore.head(3)
dataUserReview.head(12)
dataStore.columns
dataUserReview.columns
dataStore.tail()
dataStore.shape
dataUserReview.shape
dataStore.Rating.plot(kind='line',color='brown', label='Rating',linewidth=0.5)
plt.legend(loc='lower left') 
plt.title('Rating')
plt.show()
# Rating
dataStore.plot(kind='scatter' , x='Rating', y='Rating' , color = 'red')
plt.xlabel('Rating')
plt.ylabel('Rating')
plt.title('Info')
plt.show()
dataStore.Rating.plot(kind='hist',bins = 50,figsize=(5,5))
plt.show()
ratingSeries = dataStore['Rating']
ratingDataFrame = dataStore[['Rating']]

for index,value in dataStore[['Rating']][0:5].iterrows():
    print(index," : ",value)
filtre = dataStore['Rating'] > 4.5
dataStore[filtre]
dataStore[np.logical_and(dataStore['Rating'] > 4.7, dataStore['Category'] == 'ART_AND_DESIGN')]
category = iter(dataStore['Category'])
print(next(category))
print(*category)
dataStore["Degree"] = ["Very Good" if i > 4.5 else "Good" if i > 4.0 else "So-So"  for i in dataStore.Rating]
dataStore.loc[:100,["Degree","Rating"]]
print(dataStore['Category'].value_counts(dropna = False))
dataStore.boxplot(column='Rating', by='Price')
plt.show()
newDataStore = dataStore.head()
melted = pd.melt(frame=newDataStore, id_vars='App', value_vars=['Reviews','Rating'])
melted
melted.pivot(index='App', columns='variable', values='value')
data1 = dataStore.head()
data2 = dataStore.tail()
conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row
data1 = dataStore['Rating'].head()
data2 = dataStore['Reviews'].head()
conc_data_col = pd.concat([data1,data2],axis=1)
conc_data_col
dataStore['Reviews'] = dataStore['Reviews'].astype("float")
dataStore.info()
dataStore['Reviews'].value_counts(dropna=False)
assert dataStore['Reviews'].notnull().all() # true
assert dataStore.columns[3] == 'Reviews' # true
assert dataStore.columns[3] == 'Rating' # error
assert dataStore.Rating.dtypes == np.float # true
assert dataStore.Rating.dtypes == np.int # error
dataStore.plot(kind='hist',x='Reviews',y='Rating')
plt.show()
#close warning
import warnings
warnings.filterwarnings("ignore")
data3 = dataStore.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data3['_____date_____'] = datetime_object # _____date_____ fixed column for date
data3 = data3.set_index('_____date_____') 
data3
print(data3.loc["1992-03-10":"1993-03-16"])
data3.resample("A").mean() # A -> year
data3.resample("M").mean() # M -> Month
data3.resample("M").first().interpolate("linear") # fill
data3.resample("M").mean().interpolate("linear") # fill with mean
dataReview = pd.read_csv('../input/googleplaystore_user_reviews.csv')
# dataReview = dataReview.set_index("App")
dataReview.index.name = "index"
dataReview.head()






