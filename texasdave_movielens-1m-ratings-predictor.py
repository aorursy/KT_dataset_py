
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_m = pd.read_csv("../input/movielenssimplilearn/movies.dat", engine='python', sep='::', names=["MovieID", "Title", "Genres"])
df_m.head()
df_r = pd.read_csv("../input/movielenssimplilearn/ratings.dat", engine='python', sep='::', names=["UserID", "MovieID", "Rating", "Timestamp"])
df_r.head()
df_u = pd.read_csv("../input/movielenssimplilearn/users.dat", engine='python', sep='::', names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])
df_u.head()
df_merged1 = df_m.merge(df_r, how='outer')
df_merged1.head()
df_merged2 = df_u.merge(df_r, how='inner')
df_merged2.head()
df_merged3 = df_merged1.merge(df_merged2, how='inner')
df_merged3.head()
df_merged3.UserID = df_merged3.UserID.astype(int)
df_merged3.Rating = df_merged3.Rating.astype(int)
df_merged3.head()
df_merged3.shape
df_merged3.sort_values(by=['UserID'], ascending=True)
cols = list(df_merged3.columns.values)
print(cols)
master_data = df_merged3[['UserID', 'MovieID', 'Title', 'Rating', 'Genres', 'Zip-code', 'Gender', 'Age', 'Occupation', 'Timestamp']]
master_data.head()
master_data.sort_values(by=['MovieID'], ascending=True)
master_data.hist(column='Age')
bins_list = [1, 18, 25, 35, 45, 50, 56]
master_data.hist(column='Age', bins = bins_list)
master_data[master_data['MovieID'] == 1].hist(column='Rating')
master_data[master_data['MovieID'] == 1].count()
master_data[master_data['MovieID'] == 1].Rating.mean()
master_data.sort_values(by=['Rating'], ascending=False)
master_data[master_data['Rating'] == 5].Rating.count()
master_data[master_data['Rating'] == 5].Title.unique()
most_rated = master_data.groupby('Title').size().sort_values(ascending=False)[:25]
most_rated
user2696 = master_data[master_data['UserID'] == 2696]
user2696.shape
user2696
master_data['Genres_list'] = master_data['Genres'].str.split('|')
master_data.head()
master_data['Gender'].replace(['F','M'],[0,1],inplace=True)
master_data.head()
from sklearn.preprocessing import MultiLabelBinarizer

## assign a new series to the genres_list column that contains a list of categories for each movie
list2series = pd.Series(master_data.Genres_list)

mlb = MultiLabelBinarizer()

## use mlb to create a new dataframe of the genres from the list for each row from the original data

one_hot_genres = pd.DataFrame(mlb.fit_transform(list2series),columns=mlb.classes_,index=list2series.index)

print(one_hot_genres.head())
type(one_hot_genres)
md_small = master_data.iloc[:, [1,3,5,6,7,8]]
md_small.head()
md_small.dtypes
md_small['Zip-code'] = md_small['Zip-code'].str[:5]
pd.to_numeric(md_small['Zip-code'])
## skip the correlation of movieID

md_small[md_small.columns[1:]].corr()['Rating'][:]
master_features = pd.merge(md_small, one_hot_genres, left_index=True, right_index=True)
master_features.head()
X_feature = md_small.drop(['Zip-code'], axis=1)
X_feature.head()
X_feature_small = X_feature[X_feature['MovieID'] < 50] 
X_feature_small_trimmed = X_feature_small.drop(['MovieID','Rating'], axis=1)
X_feature_small_trimmed.shape
X_feature_small_trimmed.head()
Y_target = master_features['Rating'][master_features['MovieID']< 50]
Y_target.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_feature_small_trimmed,Y_target,random_state=1)
from sklearn.linear_model import LogisticRegression
#logreg = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=100000)
logreg = LogisticRegression(max_iter=100000)
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)
# print the first 30 true and predicted responses
print ('actual:    ', y_test.values[0:30])
print ('predicted: ', y_pred[0:30])
