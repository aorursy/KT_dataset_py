#import the libraries
#plot outputs will be  visible and stored within the notebook
%matplotlib inline 
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np #linear algebra
from scipy import stats #for optimization problem solving
import matplotlib.pyplot as plt #pyplot is matplotlib's plotting framework which import line from the module "matplotlib.pyplot" and binds that to the name "plt".
import ast #abstract syntax tree 

# load the dataset
df = pd.read_csv('../input/ted_main_tat.csv')# download the csv file in your local directory and play with it
df.columns
print(df)#print the data frame according to column 

df = df[['name', 'title', 'description', 'main_speaker', 'speaker_occupation', 'num_speaker', 'duration', 'event',  'comments', 'tags', 'languages', 'ratings', 'related_talks', 'url', 'views']]
print (df) #sorting data
#df.head(n) returns a DataFrame holding the first n rows of df
df.head()
len(df) 
#len shows what is the length of data   
pop_talks = df[['title', 'main_speaker', 'views']].sort_values('views', ascending=False)[:10]
pop_talks 
#sorting popular talks in decending order according to view
pop_talks['ranking_on_views']=df['views'].rank(ascending=False)
pop_talks
#ranking depending on views
df.iloc[1]['ratings']
#Purely integer-location based indexing for selection by position
df['ratings'] = df['ratings'].apply(lambda x:ast.literal_eval(x))
#Safely evaluate an expression node or a Unicode or Latin-1 encoded string containing a Python expression. 
#this syntax makes sure that only ratings datas will go as input
#lambda is used in conjunction with typical functional concepts like filter(), map() and reduce().
#filter() calls our lambda function for each element of the list, and returns a new list that contains only those elements for which the function returned "True".
# map() is used to convert our list
#The return value of the last call is the result of the reduce() construct.
df['jawdrop'] = df['ratings'].apply(lambda x: x[-3]['count'])
df['beautiful'] = df['ratings'].apply(lambda x: x[3]['count'])
df['confusing'] = df['ratings'].apply(lambda x: x[2]['count'])
df.head()
#sorting ted talks in decending order according to rating
beautiful = df[['title', 'main_speaker', 'views', 'beautiful']].sort_values('beautiful', ascending=False)[:10]
beautiful
#ranking depending on rating
beautiful['ranking_on_rating:beautiful']=df['beautiful'].rank(ascending=False)
beautiful
#sorting ted talks in decending order according to rating
jawdrop = df[['title', 'main_speaker', 'views',  'jawdrop']].sort_values('jawdrop', ascending=False)[:10]
jawdrop
#ranking depending on rating
jawdrop = df[['title', 'main_speaker', 'views',  'jawdrop']].sort_values('jawdrop', ascending=False)[:10]
jawdrop
#sorting ted talks in decending order according to rating
confusing = df[['title', 'main_speaker', 'views',  'confusing']].sort_values('confusing', ascending=False)[:10]
confusing
#ranking depending on rating
confusing['ranking_on_rating(confusing)']=df['confusing'].rank(ascending=False)
confusing
import tensorflow as tf
feature_columns = []
views = tf.feature_column.numeric_column('views')
feature_columns.append(views)
views_buckets = tf.feature_column.bucketized_column(
                        tf.feature_column.numeric_column('views'), 
                        boundaries = [10000000,20000000,30000000,40000000]

)
feature_columns.append(views_buckets)
print(views_buckets)