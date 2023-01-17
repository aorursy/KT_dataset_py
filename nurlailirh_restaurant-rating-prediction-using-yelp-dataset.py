# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Membaca data bisnis

biz=pd.read_csv('/kaggle/input/yelpversion6/yelp_business.csv')

biz.head()
biz.shape
#memisahkan masing-masing kategori ke kolom berbeda

df_category_split = biz['categories'].str.split(';', expand=True)[[0,1,2]]



# nama kolom yang baru

df_category_split.columns = ['category_1', 'category_2', 'category_3']

biz = pd.concat([biz, df_category_split], axis=1)



# menghapus kolom lama 'categories'

biz = biz.drop(['categories'], axis=1)
biz.head()
# Filter dataset, 'kategori: Restaurants, 'state': PA, dan 'is_open' : 1

resto = biz.loc[(biz['category_1'] == 'Restaurants') | (biz['category_2'] == 'Restaurants') | (biz['category_3'] == 'Restaurants')]

resto = resto.loc[(resto['state'] == 'PA')]

resto = resto.loc[(resto['is_open'] == 1)]

print(resto.shape)

#menghapus variabel yang tidak digunakan dan garbage collection

del biz



import gc

gc.collect()

resto.head()
#menghapus kolom yang tidak digunakan

resto=resto.drop(['name', 'neighborhood', 'address', 'city', 'state',

       'postal_code', 'latitude', 'longitude','is_open', 'category_1', 'category_2', 'category_3'],axis=1)

resto.reset_index(drop=True, inplace=True)
print(resto.info())

resto.head()
# Membaca data user

user=pd.read_csv('/kaggle/input/yelpversion6/yelp_user.csv')

user.head()

print(user.shape)
## Filter dataset



# Menghapus kolom 'name'

user=user.drop('name',axis=1)



# Memilih data user yang review_count nya >0

user = user.loc[(user['review_count'] > 0)]

print(user.shape)
print(user.info())

user.head()
# Membaca data review

reviews=pd.read_csv('yelp_review.csv')

reviews.head()
print(reviews.shape)

reviews.columns
reviews=reviews.drop('text',axis=1)
#join resto dan reviews

yelp_join=pd.merge(resto,reviews,on='business_id',how='inner')



#join resto, reviews dengan user

yelp_join=pd.merge(yelp_join,user,on='user_id',how='inner')

print(yelp_join.shape)

yelp_join.head()
#menghapus variabel yang tidak digunakan dan garbage collection

del resto

del reviews

del user



import gc

gc.collect()

yelp_join.head()
print(yelp_join.dtypes)



#tipe data datetime

yelp_join['date']=pd.to_datetime(yelp_join['date'])

yelp_join['yelping_since']=pd.to_datetime(yelp_join['yelping_since'])
#indexing id to number for simplicity

bizID = pd.Categorical((pd.factorize(yelp_join.business_id)[0] + 1))

userID = pd.Categorical((pd.factorize(yelp_join.user_id)[0] + 1))

reviewID = pd.Categorical((pd.factorize(yelp_join.review_id)[0] + 1))



bizID=bizID.astype(int)

userID=userID.astype(int)

reviewID=reviewID.astype(int)



yelp_join['business_id']=bizID

yelp_join['user_id']=userID

yelp_join['review_id']=reviewID
## Filter dataset

# Menghapus  incosistency: review date < yelping since

yelp_join = yelp_join.loc[((yelp_join['date'] > yelp_join['yelping_since']) == True)]

print(yelp_join.shape)
print(yelp_join.shape)

yelp_join.head()
print(yelp_join.business_id.nunique())

print(yelp_join.user_id.nunique())
yelp_join['user_id'].value_counts()
yelp_join['business_id'].value_counts()
min_resto_ratings = 1

filter_resto = yelp_join['business_id'].value_counts() > 1

filter_resto = filter_resto[filter_resto].index.tolist()



min_user_ratings = 1

filter_users = df_new['user_id'].value_counts() > min_user_ratings

filter_users = filter_users[filter_users].index.tolist()

df_new = yelp_join[(yelp_join['business_id'].isin(filter_resto)) & (yelp_join['user_id'].isin(filter_users))]

print('The original data frame shape:\t{}'.format(yelp_join.shape))

print('The new data frame shape:\t{}'.format(df_new.shape))
print(yelp_join.business_id.nunique())

print(yelp_join.user_id.nunique())
yelp_join.head()
cf=yelp_join
print(yelp_join.shape)

yelp_join.columns
#drop unused columns

#cf = cf.drop(['business_id'], axis=1)

cf = cf.drop(['stars_x'], axis=1)

cf = cf.drop(['review_count_x'], axis=1)

#cf = cf.drop(['review_id'], axis=1)

#cf = cf.drop(['user_id'], axis=1)

cf = cf.drop(['date'], axis=1)

cf = cf.drop(['useful_x'], axis=1)

cf = cf.drop(['funny_x'], axis=1)

cf = cf.drop(['cool_x'], axis=1)

cf = cf.drop(['review_count_y'], axis=1)

cf = cf.drop(['yelping_since'], axis=1)

cf = cf.drop(['friends'], axis=1)

cf = cf.drop(['cool_y'], axis=1)

cf = cf.drop(['useful_y'], axis=1)

cf = cf.drop(['funny_y'], axis=1)

cf = cf.drop(['fans'], axis=1)

cf = cf.drop(['elite'], axis=1)

cf = cf.drop(['average_stars'], axis=1)

cf = cf.drop(['compliment_hot'], axis=1)

cf = cf.drop(['compliment_more'], axis=1)

cf = cf.drop(['compliment_profile'], axis=1)

cf = cf.drop(['compliment_cute'], axis=1)

cf = cf.drop(['compliment_list'], axis=1)

cf = cf.drop(['compliment_note'], axis=1)

cf = cf.drop(['compliment_plain'], axis=1)

cf = cf.drop(['compliment_cool'], axis=1)

cf = cf.drop(['compliment_funny'], axis=1)

cf = cf.drop(['compliment_writer'], axis=1)

cf = cf.drop(['compliment_photos'], axis=1)

cf.columns
cf.to_csv('collaborative.csv')
import matplotlib.pyplot as plt
print(yelp_join.shape)

yelp_join.head()

yelp_join.dtypes
yelp_join=predictor
stars=yelp_join['rating'].value_counts()

stars=stars.to_frame().reset_index()

stars.columns=['rating','count']

print(stars)

stars.sort_values(by=['rating'],ascending=True).plot.bar(x='rating',y='count')
yr=yelp_join.groupby('yelping_since')[['user_id']].count()

yr
# df is defined in the previous example



# step 1: create a 'year' column

yelp_join['year_of_yelping'] = yelp_join['yelping_since'].map(lambda x: x.strftime('%Y'))



# step 2: group by the created columns

grouped_df = yelp_join.groupby('year_of_yelping')[['user_id']].count()



grouped_df

yr=grouped_df.reset_index()

yr
yr.plot.bar(x='year_of_yelping',y='user_id')
yr=yelp_join.groupby('date')[['review_id']].count()

yr
# df is defined in the previous example



# step 1: create a 'year' column

yelp_join['year_of_review'] = yelp_join['date'].map(lambda x: x.strftime('%Y'))



# step 2: group by the created columns

grouped_df = yelp_join.groupby('year_of_review')[['review_id']].count()



grouped_df

yr=grouped_df.reset_index()

yr
yr.plot.bar(x='year_of_review',y='review_id')
us=yelp_join.groupby('user_id')[['business_id']].count()

us
us=yelp_join.groupby('user_id')[['review_id']].count()

us
yelp_join.shape
# Only the last 3 years

#yelp_join=yelp_join.loc[(yelp_join['date'] >= '2005-10-01')]

sdfsd=yelp_join.loc[(yelp_join['date'] >= '2005-10-01')]

#yelp_join.shape
#x=yelp_join.loc[(yelp_join['date'] >= '2015-01-01')]

#x=x.loc[(x['yelping_since'] <'2015-02-01')]

print(sdfsd['date'].sort_values(ascending=True))
yelp_join=yelp_join.loc[(yelp_join['date'] < '2015-01-01')]

yelp_join.shape
yelp_join.shape
print(yelp_join.business_id.nunique())

print(yelp_join.user_id.nunique())
#x=yelp_join.loc[(yelp_join['yelping_since'] >= '2015-01-01')]

x=yelp_join.loc[(yelp_join['yelping_since'] <'2015-01-01')]

print(x['yelping_since'].sort_values(ascending=True))

print(x.shape)
yelp_join=yelp_join.loc[(yelp_join['yelping_since'] < '2015-02-01')]

yelp_join.shape
print(yelp_join['yelping_since'].sort_values(ascending=False))
yelp_join['no_friends']=0

yelp_join.loc[yelp_join['friends'] == 'None', ['no_friends']] = 1
yelp_join
yelp_join.shape
yelp_join['year_of_yelping']=yelp_join['year_of_yelping'].astype(int)

yelp_join['year_of_review']=yelp_join['year_of_review'].astype(int)

yelp_join.dtypes
#Check check



#check recent date

print(yelp_join[['date','yelping_since','review_id']].sort_values(by='date',ascending=False).head())

#print(yelp_join['date'].loc[yelp_join['index']==29524])
from datetime import datetime



d_base = datetime(2015, 1, 1)

print(d_base)
print(yelp_join['date'].loc[yelp_join['review_id']==53024])

days=(d_base-(yelp_join['date'].loc[yelp_join['review_id']==53024]))

print(days)
yelp_join.shape
yelp_join.columns
#derive columns

df = pd.DataFrame([])

for index, row in yelp_join.iterrows():

    #total friends

    number=row['friends'].count(",")+1

    #days been yelping since

    days=(d_base-row['yelping_since']).days

    #total compliments

    compnum=row['compliment_hot']+row['compliment_more']+row['compliment_cute']+row['compliment_note']+row['compliment_cool']+row['compliment_funny']+row['compliment_writer']+row['compliment_photos']

    #total votes per user

    votes=row['funny_y']+row['useful_y']+row['cool_y']

    #review age

    age=(d_base-row['date']).days

    print(days)

    df = df.append(pd.Series([row['review_id'],row['no_friends'],number,compnum,days,age,votes]),ignore_index=True)

    

df.columns=['review_id','no_friends','total_friends','total_compliments','days_since','review_age','total_votes']

df.shape

df.head()
df.shape
df.loc[df['no_friends'] == 1, ['total_friends']] = 0

df = df.drop(['no_friends'], axis=1)
yelp_join=pd.merge(yelp_join,df,on='review_id',how='inner')

yelp_join.shape
yelp_join.head()
print(yelp_join[['date','review_age','review_id']].sort_values(by='date',ascending=False).head())

print(yelp_join[['yelping_since','days_since','review_id']].sort_values(by='yelping_since',ascending=False).head())
yelp_join=yelp_join.rename(columns={"stars_x": "biz_avg_rating",

                          "review_count_x": "biz_total_rvw",

                          "stars_y": "rating",

                          "useful_x": "review_useful",

                          "funny_x": "review_funny",

                          "cool_x": "review_cool",

                          "review_count_y": "user_total_rvw",

                          "average_stars" : "user_avg_rating",

                         })
yelp_join['elite']
z=yelp_join.describe()

z

z.to_csv('descriptive-stats.csv')

#2019-07-27 17:36
skew=yelp_join.skew(axis=0,numeric_only=True)

skew.to_csv('skew.csv')

#2019-07-27 17:36
#export data to master file

yelp_join.to_csv('yelp_join_added_columns.csv')

#2019-07-27 17:36
yelp_join.head()
#Read data from file

yelp_join=pd.read_csv('yelp_join_added_columns.csv',index_col=0)

yelp_join.head()
yelp_join.business_id.nunique()
#for recent-ness column of item

bizpopularity=yelp_join.groupby('business_id')[['review_age']].mean()

bizpopularity
yelp_join=pd.merge(yelp_join,bizpopularity,on='business_id',how='inner')

yelp_join.head()

yelp_join.shape
yelp_join.columns
#review metadata columns for user feature

userreview=yelp_join[['review_id','user_id','review_useful','review_funny','review_cool','user_total_rvw']]

#c=userreview.groupby('user_id')[['cool_y','funny_y','useful_y','review_count_y']].mean()
userreview.sort_values(by='user_id')
yelp_join.business_id.nunique()
c=userreview.groupby('user_id').agg({'review_useful' : 'sum','review_funny' : 'sum','review_cool' : 'sum'})

c
yelp_join=pd.merge(yelp_join,c,on='user_id',how='inner')

yelp_join.head()

yelp_join.shape
yelp_join.columns

yelp_join.shape
yelp_join.columns
print(yelp_join.shape)

print(yelp_join.business_id.nunique())

print(yelp_join.user_id.nunique())
yelp_join.groupby(['business_id','user_id']).size()

#export

yelp_join.to_csv('resto_full.csv')

#2019-07-27 18:36
#import

import pandas as pd

yelp_join=pd.read_csv('resto_full.csv',index_col=0)
yelp_join.columns
yelp_join.shape
z=yelp_join.groupby('user_id').agg({'review_id' : 'count'})

z=z.sort_values(by='review_id',ascending=True)

z
z.to_csv('user-review-count.csv')
z.describe()
##Split test user, top 10%

from sklearn.model_selection import train_test_split

user_train, user_test = train_test_split(z,test_size=0.1,shuffle=False)
user_test=user_test.reset_index()
user_test.sort_values(by='user_id',ascending=True)
yelp_join.shape
forsample = yelp_join[(yelp_join['user_id'].isin(user_test['user_id']))]

forsample.shape
forsample.head()
#get 20% from the whole dataset

train, test = train_test_split(forsample,test_size=0.425,shuffle=True)
test
test.to_csv('test_dataset.csv')
yelp_join.shape
print(yelp_join.user_id.nunique())

print(yelp_join.business_id.nunique())
train = yelp_join[(~yelp_join['review_id'].isin(test['review_id']))]

train
#for training the neural network model

train.to_csv('train_dataset.csv')
yelp_join.business_id.nunique()
print(yelp_join.user_id.nunique())

print(yelp_join.business_id.nunique())
print(train.user_id.nunique())

print(train.business_id.nunique())
#import

import pandas as pd

yelp_join=pd.read_csv('resto_full.csv',index_col=0)

train=pd.read_csv('train_dataset.csv',index_col=0)

test=pd.read_csv('test_dataset.csv',index_col=0)
train.shape
train.sha
train['rating'].hist()
test.shape
test['rating'].hist()
print(yelp_join.shape)

print(train.shape)

print(test.shape)
yelp_join.user_id.nunique()
nuser=yelp_join.user_id.unique()

nuser
userset = pd.DataFrame({'user_id':nuser[:]})
userset
nbiz=yelp_join.business_id.unique()

nbiz
bizset = pd.DataFrame({'business_id':nbiz[:]})
bizset
userset['key'] = 0

bizset['key'] = 0



df_cartesian = userset.merge(bizset,on='key',how='outer')

df_cartesian = df_cartesian.drop(columns=['key'])

df_cartesian
iddata=train[['user_id','business_id']]

iddata
df_1_2 = df_cartesian.merge(iddata,on=['user_id','business_id'], how='left',indicator=True)

df_1_not_2 = df_1_2[df_1_2["_merge"] == "left_only"].drop(columns=["_merge"])

df_1_not_2
iddata=test[['user_id','business_id']]

iddata
df_1_2 = df_1_not_2.merge(iddata,on=['user_id','business_id'], how='left',indicator=True)

df_final = df_1_2[df_1_2["_merge"] == "left_only"].drop(columns=["_merge"])

df_final
bizf=yelp_join[['business_id','biz_avg_rating','biz_total_rvw','review_age_y']]

bizf.sort_values(by='business_id')

bizf=bizf.drop_duplicates()

bizf
df_final= df_final.merge(bizf,on='business_id', how='left')

df_final
userf=yelp_join[['user_id','fans','user_total_rvw','user_avg_rating','days_since','total_friends','total_compliments','total_votes','review_useful_y', 'review_funny_y', 'review_cool_y']]

userf.sort_values(by='user_id')

userf=userf.drop_duplicates()

userf
df_final= df_final.merge(userf,on='user_id', how='left')

df_final.head()
df_final.sort_values(by='user_id')
df_final.to_csv('full-mat-predict.csv')
import pandas as pd
df_final=pd.read_csv('full-mat-predict.csv')
df_final.shape
df_final=df_final.drop('Unnamed: 0',axis=1)
df_final.head()
yelp_join.columns
predictor=yelp_join[['rating','biz_avg_rating', 'biz_total_rvw',

       'review_age_y', 'fans', 'user_total_rvw', 'user_avg_rating',

       'days_since', 'total_friends', 'total_compliments', 'total_votes',

       'review_useful_y', 'review_funny_y', 'review_cool_y']]

predictor.shape
predictor.columns
predictor.shape
import pandas as pd

import numpy as np



rs = np.random.RandomState(0)

corr = predictor.corr()

corr.style.background_gradient(cmap='coolwarm')

# 'RdBu_r' & 'BrBG' are other good diverging colormaps
import pandas as pd

import numpy as np



rs = np.random.RandomState(0)

corr = predictor.corr()

corr.style.background_gradient(cmap='coolwarm')

# 'RdBu_r' & 'BrBG' are other good diverging colormaps
descdata=predictor.describe()

descdata.to_csv('predictor-descriptive.csv')
rating=predictor['rating']

pred = predictor.drop(['rating'], axis=1)



#export

pred.to_csv('predictor-new.csv')

rating.to_csv('target-new.csv',header=False)
import pandas as pd
#import

x=pd.read_csv('predictor-new.csv',index_col=0)

y=pd.read_csv('target-new.csv',index_col=0,header=None)
print(x.shape)

x.head()
print(y.shape)

y.head()
df_final.columns
df_final.shape
x
fullmatpred=df_final[['biz_avg_rating', 'biz_total_rvw',

       'review_age_y', 'fans', 'user_total_rvw', 'user_avg_rating',

       'days_since', 'total_friends', 'total_compliments', 'total_votes',

       'review_useful_y', 'review_funny_y', 'review_cool_y']]
test=test[['biz_avg_rating', 'biz_total_rvw',

       'review_age_y', 'fans', 'user_total_rvw', 'user_avg_rating',

       'days_since', 'total_friends', 'total_compliments', 'total_votes',

       'review_useful_y', 'review_funny_y', 'review_cool_y','rating']]
test
x_test=test[['biz_avg_rating', 'biz_total_rvw', 'review_age_y', 'fans',

       'user_total_rvw', 'user_avg_rating', 'days_since', 'total_friends',

       'total_compliments', 'total_votes', 'review_useful_y', 'review_funny_y',

       'review_cool_y']]
y_test=test[['rating']]
x_test
print(train.shape)

print(test.shape)
train.describe().to_csv('train-describe.csv')
test.describe().to_csv('test-describe.csv')
#encode target to 5 columns

# import preprocessing from sklearn

from sklearn import preprocessing

from tensorflow.python import keras

enc = preprocessing.LabelEncoder()



# 2. FIT

enc.fit(y)



# 3. Transform

labels = enc.transform(y)

labels.shape

y=keras.utils.to_categorical(labels)

# as you can see, you've the same number of rows 891

# but now you've so many more columns due to how we changed all the categorical data into numerical data



#encode target to 5 columns

# import preprocessing from sklearn

from sklearn import preprocessing

from tensorflow.python import keras

enc = preprocessing.LabelEncoder()



# 2. FIT

enc.fit(y_test)



# 3. Transform

labels = enc.transform(y_test)

labels.shape

y_test=keras.utils.to_categorical(labels)

# as you can see, you've the same number of rows 891

# but now you've so many more columns due to how we changed all the categorical data into numerical data



y.shape
y_test.shape
x.shape
##handling outliers

import numpy as np

import numpy.ma as ma

from scipy.stats import mstats



low = .05

high = .95

quant_df = x.quantile([low, high])

print(quant_df)
quant_df.head()
##handling outliers



# Winsorizing

x['biz_avg_rating']=mstats.winsorize(x['biz_avg_rating'], limits=[0.05, 0.05])

x['biz_total_rvw']=mstats.winsorize(x['biz_total_rvw'], limits=[0.05, 0.05])

x['review_age_y']=mstats.winsorize(x['review_age_y'], limits=[0.05, 0.05])

x['fans']=mstats.winsorize(x['fans'], limits=[0.05, 0.05])

x['user_total_rvw']=mstats.winsorize(x['user_total_rvw'], limits=[0.05, 0.05])

x['user_avg_rating']=mstats.winsorize(x['user_avg_rating'], limits=[0.05, 0.05])

x['days_since']=mstats.winsorize(x['days_since'], limits=[0.05, 0.05])

x['total_friends']=mstats.winsorize(x['total_friends'], limits=[0.05, 0.05])

x['total_compliments']=mstats.winsorize(x['total_compliments'], limits=[0.05, 0.05])

x['total_votes']=mstats.winsorize(x['total_votes'], limits=[0.05, 0.05])

x['review_useful_y']=mstats.winsorize(x['review_useful_y'], limits=[0.05, 0.05])

x['review_funny_y']=mstats.winsorize(x['review_funny_y'], limits=[0.05, 0.05])

x['review_cool_y']=mstats.winsorize(x['review_cool_y'], limits=[0.05, 0.05])
quant_df.head()
fullmatpred.loc[fullmatpred['biz_avg_rating'] < 2.5, 'biz_avg_rating'] = 2.5

fullmatpred.loc[fullmatpred['biz_avg_rating'] > 4.5, 'biz_avg_rating'] = 4.5

fullmatpred.loc[fullmatpred['biz_total_rvw'] < 15, 'biz_total_rvw'] = 15

fullmatpred.loc[fullmatpred['biz_total_rvw'] > 561, 'biz_total_rvw'] = 561

fullmatpred.loc[fullmatpred['review_age_y'] < 255, 'review_age_y'] = 255

fullmatpred.loc[fullmatpred['review_age_y'] > 1177, 'review_age_y'] = 1178

fullmatpred.loc[fullmatpred['fans'] < 0, 'fans'] = 0

fullmatpred.loc[fullmatpred['fans'] > 76, 'fans'] = 76

fullmatpred.loc[fullmatpred['user_total_rvw'] < 6, 'user_total_rvw'] = 6

fullmatpred.loc[fullmatpred['user_total_rvw'] > 862, 'user_total_rvw'] = 862

fullmatpred.loc[fullmatpred['user_avg_rating'] < 2.88, 'user_avg_rating'] = 2.88

fullmatpred.loc[fullmatpred['user_avg_rating'] > 4.43, 'user_avg_rating'] = 4.43

fullmatpred.loc[fullmatpred['days_since'] < 281, 'days_since'] = 281

fullmatpred.loc[fullmatpred['days_since'] > 2667, 'days_since'] = 2667

fullmatpred.loc[fullmatpred['total_friends'] < 0 , 'total_friends'] = 0

fullmatpred.loc[fullmatpred['total_friends'] > 589 , 'total_friends'] = 589

fullmatpred.loc[fullmatpred['total_compliments'] < 0 , 'total_compliments'] = 0

fullmatpred.loc[fullmatpred['total_compliments'] > 966 , 'total_compliments'] = 676

fullmatpred.loc[fullmatpred['total_votes'] < 0 , 'total_votes'] = 0

fullmatpred.loc[fullmatpred['total_votes'] > 4728 , 'total_votes'] = 4728

fullmatpred.loc[fullmatpred['review_useful_y'] > 201 , 'review_useful_y'] = 201

fullmatpred.loc[fullmatpred['review_funny_y'] > 78 , 'review_funny_y'] = 78

fullmatpred.loc[fullmatpred['review_cool_y'] > 78 , 'review_cool_y'] = 78

x_test.loc[x_test['biz_avg_rating'] < 2.5, 'biz_avg_rating'] = 2.5

x_test.loc[x_test['biz_avg_rating'] > 4.5, 'biz_avg_rating'] = 4.5

x_test.loc[x_test['biz_total_rvw'] < 15, 'biz_total_rvw'] = 15

x_test.loc[x_test['biz_total_rvw'] > 561, 'biz_total_rvw'] = 561

x_test.loc[x_test['review_age_y'] < 255, 'review_age_y'] = 255

x_test.loc[x_test['review_age_y'] > 1177, 'review_age_y'] = 1178

x_test.loc[x_test['fans'] < 0, 'fans'] = 0

x_test.loc[x_test['fans'] > 76, 'fans'] = 76

x_test.loc[x_test['user_total_rvw'] < 6, 'user_total_rvw'] = 6

x_test.loc[x_test['user_total_rvw'] > 862, 'user_total_rvw'] = 862

x_test.loc[x_test['user_avg_rating'] < 2.88, 'user_avg_rating'] = 2.88

x_test.loc[x_test['user_avg_rating'] > 4.43, 'user_avg_rating'] = 4.43

x_test.loc[x_test['days_since'] < 281, 'days_since'] = 281

x_test.loc[x_test['days_since'] > 2667, 'days_since'] = 2667

x_test.loc[x_test['total_friends'] < 0 , 'total_friends'] = 0

x_test.loc[x_test['total_friends'] > 589 , 'total_friends'] = 589

x_test.loc[x_test['total_compliments'] < 0 , 'total_compliments'] = 0

x_test.loc[x_test['total_compliments'] > 966 , 'total_compliments'] = 676

x_test.loc[x_test['total_votes'] < 0 , 'total_votes'] = 0

x_test.loc[x_test['total_votes'] > 4728 , 'total_votes'] = 4728

x_test.loc[x_test['review_useful_y'] > 201 , 'review_useful_y'] = 201

x_test.loc[x_test['review_funny_y'] > 78 , 'review_funny_y'] = 78

x_test.loc[x_test['review_cool_y'] > 78 , 'review_cool_y'] = 78

x.shape
fullmatpred.shape
#Normalization

from sklearn.preprocessing import MinMaxScaler

#Normalize data

scaler = MinMaxScaler()

# Fit only to the training data

x = scaler.fit_transform(x)

# Now apply the transformations to the data:

#x_test= scaler.transform(x_test)
fullmatpred
fullmatpred= scaler.transform(fullmatpred)
x_test= scaler.transform(x_test)
test.columns
##Split training and test set

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2,stratify=y)
x_test.shape
x_val.shape
x_test[6]
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";

 

# The GPU id to use, usually either "0" or "1";

os.environ["CUDA_VISIBLE_DEVICES"]="0";  

 

# Do other imports now...
from __future__ import absolute_import, division, print_function, unicode_literals



# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow.python import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation

from tensorflow.keras.optimizers import SGD,Adam

from tensorflow.keras import regularizers, initializers

from tensorflow.keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint

from sklearn.metrics import log_loss, confusion_matrix



# Helper libraries

import numpy as np

import matplotlib.pyplot as plt



print(tf.__version__)

print(keras.__version__)
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

print("GPU Available: ", tf.test.is_gpu_available())
csv_logger = CSVLogger('log-final-2.csv', append=True, separator=';')

from sklearn import metrics

from sklearn.metrics import log_loss, confusion_matrix
model = Sequential()

model.add(Dense(4,input_dim=13, activation='tanh',use_bias=True, kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.1))

model.add(Dense(4, activation='tanh',use_bias=True, kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.1))

model.add(Dense(8, activation='tanh',use_bias=True, kernel_regularizer=regularizers.l2(0.0001), bias_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.1))

model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])

# simple early stopping

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=50,restore_best_weights=True)

#checkpoint

# checkpoint

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')



history=model.fit(x_train, y_train,

          epochs=3000,batch_size=200,validation_data=(x_val, y_val),callbacks=[csv_logger,es,checkpoint]

          )



y_pred=model.predict(x_test)

matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

pd.DataFrame(matrix).to_csv("result-final-2.csv",header=False,index=False)



from tensorflow.keras.models import model_from_json

# serialize model to JSON

model_json = model.to_json()

with open("model-final-2.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model-final-2.h5")

print("Saved model to disk")

 

# later...
# load weights

model.load_weights("weights.best.hdf5")

# Compile model (required to make predictions)

model.compile(loss='categorical_crossentropy',

              optimizer=sgd,

              metrics=['accuracy'])

print("Created model and loaded weights from file")
y_pred_val=model.predict(x_val)
acc=metrics.accuracy_score(y_val.argmax(axis=1), y_pred_val.argmax(axis=1))

acc
matrix
len(history.history['loss'])
# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
import matplotlib.pyplot as plt

# list all data in history

print(history.history.keys())

# summarize history for accuracy

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# load json and create model

from tensorflow.keras.models import model_from_json



json_file = open('model-final-2.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

loaded_model = model_from_json(loaded_model_json)

# load weights into new model

loaded_model.load_weights("model-final-2.h5")

print("Loaded model from disk")

 

# evaluate loaded model on test data

loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

score = loaded_model.evaluate(x_test, y_test, verbose=0)

print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
y_pred=loaded_model.predict(x_test)
y_pred
test.rating.value_counts()
y_test.shape
from sklearn.metrics import mean_squared_error,mean_absolute_error

from math import sqrt



rms = sqrt(mean_squared_error(y_test.argmax(axis=1), y_pred.argmax(axis=1)))

print(rms)

mae = mean_absolute_error(y_test.argmax(axis=1), y_pred.argmax(axis=1))

print(mae)

conf=metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

print(conf)
df_final.shape
fmpred=loaded_model.predict(fullmatpred)
fmpredrating=fmpred.argmax(axis=1)

fmpredrating.min()
fmpredratingdf=pd.DataFrame({'col1':fmpred[:,0],'col2':fmpred[:,1],'col3':fmpred[:,2],'col4':fmpred[:,3],'col5':fmpred[:,4]})

fmpredratingdf
fmpredratingdf["rating"] = fmpredratingdf[["col1","col2","col3","col4","col5"]].max(axis=1)
fmpredratingdf
def get_status(df):

    if df['rating'] == df['col1']:

        return 1

    elif df['rating'] == df['col2']:

        return 2

    elif df['rating'] == df['col3']:

        return 3

    elif df['rating'] == df['col4']:

        return 4

    else:

        return 5



fmpredratingdf['star'] = fmpredratingdf.apply(get_status, axis = 1)
fmpredratingdf
full_matrix_id=df_final[['user_id','business_id']]

full_matrix_id
rat=fmpredratingdf[['star']]

rat=rat.rename({'star':'rating'},axis=1)

rat
fullpreddata=pd.concat([full_matrix_id, rat], axis=1)
fullpreddata
fullpreddata.to_csv('cf-predicted.csv',header=False,index=None)
train=pd.read_csv('train_dataset.csv',index_col=0)

train.head()
train_cf=train[['user_id','business_id','rating']]

train_cf.shape
fullpreddata.head()
test=pd.read_csv('test_dataset.csv',index_col=0)

test.head()
test_cf=test[['user_id','business_id','rating']]

test_cf.shape
full_total=fullpreddata.append(train_cf,sort=False)

full_total

full_total.to_csv('cf-full.csv',header=False,index=None)
full_total
##Import library

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.metrics import log_loss, confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.naive_bayes import BernoulliNB

from sklearn.tree import DecisionTreeClassifier
#model=MLPClassifier(hidden_layer_sizes=(4,4,8),max_iter=3000,solver='sgd',activation='tanh',alpha=0.01,learning_rate_init=0.0001,verbose=True)

#model=MLPClassifier(hidden_layer_sizes=(8,8,8),max_iter=3000,solver='adam',activation='tanh',verbose=True)

#model=KNeighborsClassifier(n_neighbors=1000)

#model=LogisticRegression(multi_class='auto',solver='sag')

#model=DecisionTreeClassifier()

model=MLPClassifier(hidden_layer_sizes=(100,100),max_iter=3000,verbose=True,activation='tanh')
model.fit(x_train,y_train)

#Test the model

y_pred = model.predict(x_test)

#Print final result

print(confusion_matrix(y_test,y_pred))
model.n_layers_
accuracy = model.score(x_test,y_test)

print(accuracy*100,'%')
accuracy = metrics.accuracy_score(y_test, y_pred)

accuracy