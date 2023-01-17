# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Load Customer Ratings Information
df=pd.read_csv('/kaggle/input/netflix-prize-data/combined_data_1.txt',names=['Cust-Id','Ratings'],usecols=[0,1],header=None)             

df.head()
#Load the rest of the Customer Ratings Information
#df1=pd.read_csv('/kaggle/input/netflix-prize-data/combined_data_2.txt', names=['Cust-Id','Ratings'],usecols=[0,1])
#df2=pd.read_csv('/kaggle/input/netflix-prize-data/combined_data_3.txt', names=['Cust-Id','Ratings'],usecols=[0,1])
#df3=pd.read_csv('/kaggle/input/netflix-prize-data/combined_data_4.txt', names=['Cust-Id','Ratings'],usecols=[0,1])
df.head()
df.info()
#df=pd.concat([df,df1])
#df=pd.concat([df,df2])
#df=pd.concat([df,df3])


df.index=np.arange(0,len(df))
df.head()

p = df.groupby('Ratings')['Ratings'].agg(['count']) 
p

#Plotting Distribution of Movie Ratings
p.plot(kind='barh',legend = False)
#Get Rating Count
rating_count=df['Ratings'].count()
#Get Movie Count
movie_count=df.isnull().sum()

#Get Customer Count
customer_count=df['Cust-Id'].nunique()-movie_count

movie_count
rating_count
customer_count=customer_count[1]
df.info()
print("movie count:",movie_count)
print("rating count:",rating_count)
print("customer count:",customer_count)
#df_nan returns df with rows index that contain nan values
df_nan = pd.DataFrame(pd.isnull(df.Ratings))
df_nan = df_nan[df_nan['Ratings'] == True]
df_nan = df_nan.reset_index() #When reset_index is used, the old index becomes values in a column while the new index is sequential
df_nan

movie_np = []
movie_id = 1

for i,j in zip(df_nan['index'][1:],df_nan['index'][:-1]): # excludes 23057834 in df_na
    # numpy approach
    temp = np.full((1,i-j-1), movie_id) #i-j-1 because you want to know the number of rows in between 0 and 548. 
                                        #The number of rows between 0 and 548 correspond to the number of customer ratings
                                        #for movie one
    movie_np = np.append(movie_np, temp)
    movie_id += 1
#movie_np contains an array of movie id that can become a column to be appended next to a column of customer ratings
last_record = np.full((1,len(df) - df_nan.iloc[-1, 0] - 1),movie_id) #len(df) is the last customer rating for movie 4499 and df_nan.iloc[-1,0] is the 0th row for customer ratings for 4499 
movie_np = np.append(movie_np, last_record)
print('Movie numpy: {}'.format(movie_np))
print('Length: {}'.format(len(movie_np)))
#Remove movie id rows
df=df[pd.notnull(df['Ratings'])]
df.head()

df['Movie Id']=movie_np.astype(int)
df.head()
f = ['count','mean']

df_movie_summary = df.groupby('Movie Id')['Ratings'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)

df_movie_summary.head()
movie_benchmark=round(df_movie_summary['count'].quantile(0.7),0)
movie_list=df_movie_summary[df_movie_summary['count']>movie_benchmark].index
movie_list


df_customer_summary=df.groupby('Cust-Id')['Ratings'].agg(f)
df_customer_summary.head()
customer_benchmark=round(df_customer_summary['count'].quantile(0.7),0)
customer_benchmark
customer_list=df_customer_summary[df_customer_summary['count']>customer_benchmark].index
customer_list
#Slice df with customer_list and movie_list
df=df[~df['Movie Id'].isin(movie_list)]
df=df[~df['Cust-Id'].isin(customer_list)]
df=df.reset_index(drop=True)

df

df_p = pd.pivot_table(df,values='Ratings',index='Cust-Id',columns='Movie Id')
df_p
df_title = pd.read_csv('/kaggle/input/netflix-prize-data/movie_titles.csv', encoding = "ISO-8859-1",names = ['Movie_Id', 'Year', 'Name'])
df_title.set_index('Movie_Id',inplace=True)
df_title

#surprise library is used for building recommendation systems
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate


reader = Reader()

# get just top 100K rows for faster run time
data = Dataset.load_from_df(df[['Cust-Id', 'Movie Id', 'Ratings']][:100000], reader)
svd = SVD()

cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
df_count=df_p.count(axis='columns')
df_count.sort_values(ascending=False) # customer 2446318 has the most ratings
#To Establish the movies customer 2446318 
df_2446318=df[(df['Cust-Id']=='2446318') & (df['Ratings']==5)]
df_2446318=df_2446318.set_index('Movie Id')
df_2446318=df_2446318.join(df_title)
df_2446318
user_2446318 = df_title.copy()
user_2446318 = user_785314.reset_index()
user_2446318 = user_785314[~user_785314['Movie_Id'].isin(movie_list)]
user_2446318
data = Dataset.load_from_df(df[['Cust-Id', 'Movie Id', 'Ratings']], reader)
trainset = data.build_full_trainset()
svd.fit(trainset)
user_2446318['Estimate_Score'] = user_2446318['Movie_Id'].apply(lambda x: svd.predict(2446318, x).est)
user_2446318
user_2446318=user_2446318.sort_values(['Estimate_Score'],ascending=False)
user_2446318.head(10)

