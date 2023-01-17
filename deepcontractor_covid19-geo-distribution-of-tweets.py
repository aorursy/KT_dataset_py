import pandas as pd
import numpy as np
import seaborn as sns
data = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')
data.head()
data.columns
data.describe()
data = data.dropna()
print('Total Number of Unique Locations Tweeting :',len(data['user_location'].unique()))

type(data['date'][2])
data['date']=pd.to_datetime(data['date'],dayfirst = True)
type(data['date'][2])
top10 = data['user_location'].value_counts()
top10 = top10[0:10]
top10 = top10.to_frame().reset_index()
top10.rename(columns = {'index':'location','user_location':'counts'}, inplace = True) 
sns.barplot(y="location", x="counts", data=top10)
sns.set(rc={'figure.figsize':(8,5)})
top20 = data['user_location'].value_counts()
top20 = top20[0:20]
top20 = top20.to_frame().reset_index()
top20.rename(columns = {'index':'location','user_location':'counts'}, inplace = True) 
sns.barplot(y="location", x="counts", data=top20)
sns.set(rc={'figure.figsize':(15,10)})

