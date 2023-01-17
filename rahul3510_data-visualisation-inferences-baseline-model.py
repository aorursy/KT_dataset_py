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
import matplotlib.pyplot as plt
import seaborn as sns
#ignore warnings
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
#Lets take a look at some of the entries.
df.head()

df.describe(), df.info()
df.isnull().sum()
#Removing the columns which don't have affect on price.
df.drop(['id','host_name','last_review'], axis = 1,inplace=True) 

df.reviews_per_month.fillna(value=0,inplace=True)
plt.figure(figsize=(16, 6))
sns.barplot(df.neighbourhood_group,df.price,hue=df.room_type,ci=None)
plt.figure(figsize=(16, 6))
sns.countplot(df.neighbourhood_group,hue=df.room_type)
df.drop('price', axis=1).corrwith(df.price).plot.barh(figsize=(10, 8), 
                                                        title='Correlation with Response Variable',
                                                        fontsize=15)
plt.figure(figsize = (15, 10))
plt.style.use('seaborn-white')
ax=plt.subplot(221)
plt.boxplot(df['number_of_reviews'])
ax.set_title('Numer of Reviews')
ax=plt.subplot(222)
plt.boxplot(df['price'])
ax.set_title('Price')
ax=plt.subplot(223)
plt.boxplot(df['availability_365'])
ax.set_title('availability_365')
ax=plt.subplot(224)
plt.boxplot(df['reviews_per_month'])
ax.set_title('reviews_per_month')
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

oultier_remover = (df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 *IQR)
df=df.loc[oultier_remover]

Q1 = df['number_of_reviews'].quantile(0.25)
Q3 = df['number_of_reviews'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

oultier_remover = (df['number_of_reviews'] >= Q1 - 1.5 * IQR) & (df['number_of_reviews'] <= Q3 + 1.5 *IQR)
airbnb2=df.loc[oultier_remover]


Q1 = df['reviews_per_month'].quantile(0.25)
Q3 = df['reviews_per_month'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

oultier_remover = (df['reviews_per_month'] >= Q1 - 1.5 * IQR) & (df['reviews_per_month'] <= Q3 + 1.5 *IQR)
airbnb_new=df.loc[oultier_remover]
#extract the host_ids having high number of entries in the dataset.
host_with_most_listings=df.host_id.value_counts().head(13)

#extract the most popular neighbourhoods.
most_popular_neighbourhoods=df.neighbourhood.value_counts().head(13)
most_popular_neighbourhoods,host_with_most_listings
plt.figure(figsize=(16, 6))
host_with_most_listings.plot(kind='bar')
plt.figure(figsize=(16, 6))
most_popular_neighbourhoods.plot(kind='bar')
most_popular_neighbourhoods_df=df.loc[df.neighbourhood.isin(['Williamsburg','Bedford-Stuyvesant',   
'Harlem',                
'Bushwick',              
'Upper West Side',       
'Hell\'s Kitchen',        
'East Village',      
'Upper East Side',       
'Crown Heights',     
'Midtown',               
'East Harlem',           
'Greenpoint',            
'Chelsea' ])]
# host_with_most_listings_df=df.loc[df.host_id.isin([ '219517861',    
#  '107434423',    
#  '30283594',     
#  '137358866',    
#  '12243051',      
#  '16098958',      
#  '61391963',      
#  '22541573',      
#  '200380610',     
#  '7503643',       
# '1475015',       
#  '120762452',     
#  '2856748' ])]
plt.figure(figsize=(20, 6))
sns.catplot(x='neighbourhood', hue='neighbourhood_group', col='room_type', data=most_popular_neighbourhoods_df, kind='count').set_xticklabels(rotation=90)
plt.figure(figsize=(15, 6))
sns.scatterplot(x=df.neighbourhood_group,y=df.number_of_reviews,ci=False)
plt.figure(figsize=(15, 6))
sns.barplot(x=df.neighbourhood_group,y=df.calculated_host_listings_count,ci=False)
#Now lets take the name column and get some insights.
keywords=[]
#the basic motto is to get the individual tokens out of the dataframe column
for name in df.name:
    keywords.append(name)
def split_keywords(name):
    spl=str(name).split()
    return spl
keywords_filtered=[]
for x in keywords:
    for word in split_keywords(x):
        word=word.lower()
        keywords_filtered.append(word)
#These are some of the words that I have removed after working on data.
keywords_filtered=[word for word in keywords_filtered if not word in ['in','of','the','to','1','2','3','and','with','&']]
from collections import Counter
#Get the list of most frequent words.
freq_keywords=Counter(keywords_filtered).most_common()

freq_keywords_df=pd.DataFrame(freq_keywords)
freq_keywords_df.rename(columns={0:'Words', 1:'Count'}, inplace=True)
plt.figure(figsize=(15, 6))
#plotting the top ten
sns.barplot(x='Words',y='Count',data=freq_keywords_df[0:10])
plt.figure(figsize=(15, 6))
sns.barplot(x=df.neighbourhood_group,y=df.availability_365,hue=df.room_type,ci=False)
plt.figure(figsize=(15, 6))
sns.scatterplot(x=df.longitude,y=df.latitude,hue=df.neighbourhood_group)
plt.figure(figsize=(15, 6))
sns.scatterplot(x=df.longitude,y=df.latitude,hue=df.room_type)
df.info()
#Applying one hot encoding for categorical variables using get_dummies.
dummy_neighbourhood=pd.get_dummies(df['neighbourhood_group'], prefix='dummy')
dummy_roomtype=pd.get_dummies(df['room_type'], prefix='dummy')
df_new = pd.concat([df,dummy_neighbourhood,dummy_roomtype],axis=1)
#Removing the columns which are not helpful in predicting new prices.
df_new.drop(['neighbourhood_group','room_type','neighbourhood','name','longitude','latitude','host_id'],axis=1, inplace=True)
df_new
#Seperating the predictor and target variables.
y=df_new['price']
X=df_new.drop(['price'],axis=1)
from sklearn import preprocessing
#Only standardize the numerical colums and not the dummy variables.
X_scaled=preprocessing.scale(X.iloc[:,0:5])
X_scaled = pd.DataFrame(X_scaled, index=X.iloc[:,0:5].index, columns=X.iloc[:,0:5].columns)
X.drop(X.iloc[:,0:5],axis=1,inplace=True)
X=pd.concat([X_scaled,X],axis=1)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 50, random_state = 0) 
# As there is only option to calculate negative MAE.So lets make it positive.  
scores =  -1 * cross_val_score(regressor, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
print("Average MAE score (across experiments):",scores.mean())
#split the dataset.
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(X,y,random_state=1)
#Result using Random Forest Regressor
from sklearn.metrics import mean_absolute_error
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(train_X, train_y)
preds = model.predict(test_X)
mean_absolute_error(test_y, preds)
#Result using XGBoost.
from xgboost import XGBRegressor

model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
model.fit(train_X, train_y, 
             early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], 
             verbose=False)
predictions = model.predict(test_X)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, test_y)))