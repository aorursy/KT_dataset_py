# importing the requiered packages
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
import seaborn as sns
# reading the playstore CSV and user reviews CSV into a pandas dataframe
data = pd.read_csv('../input/googleplaystore.csv')
print(data.columns)
data.head(3)
# Lets see how many null values are there in each of the colums
null_values = {}
for i in data.columns:
    null_values[i] = len(data[pd.isnull(data[i])])
print(null_values)
test = data.groupby('Category')['App']
test = pd.DataFrame(test.size().reset_index(name = "Count"))
test.sort_values(by = 'Count',axis=0,ascending=False,inplace=True)

#plotting the top 5 categories on the appstore
plt.figure(figsize=(15,9))
sns.barplot(x=test.Category[:8],y=test.Count[:8],data=test)
plt.show()

#importing playstore_user review data. This data as sentiment for each each of the use reviews for apps
review_data = pd.read_csv('../input/googleplaystore_user_reviews.csv')
review_data = review_data.groupby('App').mean()
review_data.reset_index(inplace=True)
review_data.head()

# Merging the sentiment_polarity and sentiment_subjectivity data with the data
final_data = pd.merge(data,review_data,how = 'inner',on='App')

# converting the Installs columns to integer
Installs_array = []
for i in final_data.Installs:
    Installs_array.append(int(i[:-1].replace(",","")))
final_data['Installs'] = np.array(Installs_array)

#removing NaN values from Sentiment_Polarity
final_data = final_data[~pd.isnull(final_data['Sentiment_Polarity'])]
final_data.sort_values(by = ['Installs','Sentiment_Polarity'],ascending=False,inplace=True)
final_data.Sentiment_Polarity = np.round(final_data.Sentiment_Polarity,decimals=1)

#Number of Apps
temp = final_data[['Installs','Sentiment_Polarity']]
temp = temp.groupby('Sentiment_Polarity',as_index=True).count()
temp.reset_index(inplace=True)

# Volumes of Installs
temp_1 = final_data[['Installs','Sentiment_Polarity']]
temp_1 = temp_1.groupby('Sentiment_Polarity',as_index=True).sum()
temp_1.reset_index(inplace=True)

#merging the two dataframes
temp = pd.merge(temp,temp_1,on='Sentiment_Polarity',how='inner')
temp.head()

#Plotting the distribution
f,ax = plt.subplots(figsize=(15,9))

sns.set_color_codes("pastel")
sns.barplot(x=temp.Sentiment_Polarity,y=temp.Installs_x,color='b',label = 'Total No of Apps')

sns.set_color_codes("muted")
sns.barplot(x=temp.Sentiment_Polarity,y=np.log(temp.Installs_y),color="b",label='Sum of Installs')
ax.legend(ncol=2, loc="upper right", frameon=True)
sns.despine(left=True,bottom=True)
plt.show()
rating_data = data[['Rating','App']]
rating_data['Rating_1'] = np.round(rating_data['Rating'],decimals=0)
rating_data = rating_data.groupby("Rating").count()
rating_data = rating_data.reset_index()
rating_data.sort_values(by = 'App',inplace=True,ascending=False)

plt.figure(figsize=(15,9))
sns.barplot(x="Rating",y="App",data=rating_data,hue="Rating_1",x_bins=50)
# sns.scatterplot()
plt.show()
