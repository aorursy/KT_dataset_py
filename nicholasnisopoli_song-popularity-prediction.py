# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics

from math import sqrt

import seaborn as sns

from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read the data

df=pd.read_csv("../input/19000-spotify-songs/song_data.csv")

df_info=pd.read_csv("../input/19000-spotify-songs/song_info.csv")



#remove the duplicates

df = df.drop_duplicates(subset=['song_name'])



# Join the categorical variables

names=df_info['artist_name']

playlist = df_info['playlist']

album = df_info['album_names']

df = df.join(names)

df = df.join(playlist)

df = df.join(album)
#delete records with popularity equal to zero (wrong values)

index_delete = df.index[df['song_popularity']==0]

df = df.drop(index_delete)



#check that there is no null value

df.isnull().sum()
leble_en=preprocessing.LabelEncoder()

df['artist_name']=leble_en.fit_transform(df['artist_name'])

df['playlist']=leble_en.fit_transform(df['playlist'])

df['album_names']=leble_en.fit_transform(df['album_names'])

df=pd.get_dummies(data=df,columns=['time_signature'])



df.head
f,ax = plt.subplots(figsize=(24, 10))

cor = df.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds, ax=ax)

plt.show()
# remove the features 

df = df.drop(['song_name','audio_mode','key','speechiness','album_names','song_duration_ms','energy'],axis=1)
sns.boxplot(x=df['liveness'])

plt.show()

sns.boxplot(x=df['loudness'])

plt.show()
sns.boxplot(x=df['tempo'])

plt.show()



index_delete = df.index[df['tempo']==0]

df = df.drop(index_delete)
scaler = preprocessing.MinMaxScaler()

df_scal = scaler.fit_transform(df)

df_scal = pd.DataFrame(df_scal, columns = df.columns)



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 5))

ax1.set_title('Before Scaling')

sns.kdeplot(df['loudness'], ax=ax1)

sns.kdeplot(df['danceability'], ax=ax1)

sns.kdeplot(df['tempo'], ax=ax1)

ax2.set_title('After Min-Max Scaling')

sns.kdeplot(df_scal['loudness'], ax=ax2)

sns.kdeplot(df_scal['danceability'], ax=ax2)

sns.kdeplot(df_scal['tempo'], ax=ax2)

plt.show()
# separate the target

x=df_scal.drop(['song_popularity'],axis=1)

y=df_scal['song_popularity']



# train, validation and test

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=20)

x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.20,random_state=20)



# range of the Hyperparameters

max_depth=[5,20, 50, 90]

min_samples_leaf = [1, 3, 5]

min_samples_split = [2, 6, 12]

# range of the parameter for the number of trees

n_estimators = [100,300,500]



best_err = 1

for num in n_estimators:

    for min_split in min_samples_split:

        for min_leaf in min_samples_leaf:

            for depth in max_depth:

                rf=RandomForestRegressor(n_estimators=num,max_depth=depth,max_features='sqrt',min_samples_leaf=min_leaf,min_samples_split=min_split, random_state=42)

                rf.fit(x_train,y_train)

                prediction=rf.predict(x_val)

                err = sqrt(metrics.mean_squared_error(y_val, prediction))

                if(err < best_err):

                    best_err = err

                    best_num = num

                    best_split = min_split

                    best_leaf = min_leaf

                    best_depth = depth

rf
new_x_train = x_train.append(x_val)

new_y_train = y_train.append(y_val)



randomForest = RandomForestRegressor(n_estimators=best_num,max_depth=best_depth,max_features='sqrt',min_samples_leaf=best_leaf,min_samples_split=best_split, random_state=42)

randomForest.fit(new_x_train, new_y_train)

test_prediction = randomForest.predict(x_test)



print('Root Mean Squared Error:', sqrt(metrics.mean_squared_error(y_test, test_prediction)))