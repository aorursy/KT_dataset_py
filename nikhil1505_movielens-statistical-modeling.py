# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier





import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
ratings = pd.read_table('/kaggle/input/movielens-case-study/ratings.dat', header = None, sep = '::', 

                        names=['UserId', 'MovieID', 'Rating', 'Timestamp'])



movies = pd.read_table('/kaggle/input/movielens-case-study/movies.dat', header = None, sep = '::', names=['MovieID', 'Title', 'Genres'])



users = pd.read_table('/kaggle/input/movielens-case-study/users.dat', header = None, sep = '::', 

                      names=['UserId', 'Gender', 'Age', 'Occupation', 'Zipcode'])
print(ratings.shape)

print(movies.shape)

print(users.shape)
merge_ratings_movies = pd.merge(ratings, movies, on='MovieID', how='left')

merge_three = pd.merge(merge_ratings_movies, users, on='UserId', how='left')
merge_three.shape
master_data = merge_three.drop(['Timestamp', 'Zipcode'], axis = 1)
master_data.head(6)
master_data.describe()
age_group = master_data.groupby('Age').size()

age_group
plt.figure(figsize=(8,6))

plt.hist(bins = 50, x = master_data.Age, data = age_group, color='orange')

plt.title('Distribution of users age')

plt.ylabel('count of users')

plt.xlabel('Age');
gender_group = master_data.groupby('Gender').size()

gender_group
plt.figure(figsize=(8,8))

gender_group.plot(kind='bar')
master_data.pivot_table('Rating', index = 'Genres', columns = 'Gender')
user_group = master_data.groupby(['UserId']).size()

user_group.head(10)
plt.figure(figsize=(25,10))

plt.hist(x=[master_data	.UserId], bins=1000, color='#F1948A')

plt.show()
### User rating of the movie “Toy Story”



toy_story_data = master_data.loc[master_data['Title'] == 'Toy Story (1995)']

toy_story_data.head(10)
toy_story_data.groupby('Rating').size()
plt.figure(figsize=(8,8))

plt.hist(x=toy_story_data['Rating'], color='#2FA39C')

plt.title('Ratings of Toy Story movie')

plt.ylabel('count of users')

plt.xlabel('Ratings');
print(toy_story_data.groupby('Age').size())



plt.figure(figsize=(12,8))

plt.hist(x = master_data['Age'], data=toy_story_data, bins=15, color = '#2FA353')

plt.xlabel("Age of viewers")

plt.ylabel("No of views")

plt.title("Viewership data of Toystory movie")

plt.show()

movie_rating = master_data.groupby(['MovieID'], as_index=False)

average_movie_ratings = movie_rating.agg({'Rating':'mean'})

top_25_movies = average_movie_ratings.sort_values('Rating', ascending=False).head(25)

top_25_movies
top_25_plot = pd.merge(top_25_movies, master_data, how='left', left_on=['MovieID'], right_on=['MovieID'])

top_25_plot.head(25)
### Ratings for all the movies reviewed by for a particular user of user id = 2696



user_id_data = master_data.loc[master_data['UserId'] == 2696]

user_id_data.head(20)
# plotting the above data

plt.figure(figsize=(12,7))

plt.scatter(x=user_id_data['MovieID'], y=user_id_data['Rating'])

plt.show()
master_data['Genres'].unique()
# Here I have made a change

master_data['Genres'] = master_data['Genres'].apply(lambda x : x.split('|')[0])
master_data.head()
master_data.isnull().sum()
# As we have the col MovieId which represents unique movies, we can remove ``Title`` col

master_data = merge_three.drop(['Title', 'Timestamp'], axis = 1)
master_data.head(4)
# Print the unique values of the categorical columns

print(master_data['Genres'].unique())



print()



print(master_data['Gender'].unique())
# From Sklearn library we will ues Label Encoder to encode cat features to numeric

label_encode = LabelEncoder()



master_data.iloc[:,3] = label_encode.fit_transform(master_data.iloc[:,3].values)

master_data.iloc[:,4] = label_encode.fit_transform(master_data.iloc[:,4].values)
# Print values of cat feature after conversion



print(master_data['Genres'].unique())



print()



print(master_data['Gender'].unique())
x = master_data[['Genres', 'Gender', 'Age', 'Occupation']].values

y = master_data.Rating.values
x
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],

                       remainder='passthrough')

X = ct.fit_transform(x)

X
print(X.shape)

print(y.shape)
# Splitting the data set into 80% Training & 20% Testing

train_X, test_X, train_y, test_y = train_test_split(X,y, test_size = 0.2, random_state = 42)

train_X.shape, test_X.shape, train_y.shape, test_y.shape
cardinality_cols = ['Gender', 'Genres']

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(train_X[cardinality_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(test_X[cardinality_cols]))
feature_scale = StandardScaler(with_mean=False)

train_X = feature_scale.fit_transform(train_X)

test_X = feature_scale.transform(test_X)
rf = RandomForestClassifier(n_estimators=50, criterion="entropy", max_depth=10, random_state=42)



rf.fit(train_X, train_y)



pred_rand_for = rf.predict(test_x)



rand_for_acc = accuracy_score(test_y, pred_rand_for)

print('Random Forest Accuracy:', rand_for_acc)
plt.figure(figsize=(8,5))

sns.set(font_scale=1.3)

sns.heatmap(confusion_matrix(pred_rand_for, test_y), annot = True, fmt = ".0f", cmap = "Accent")

plt.xlabel("Predicted Values")

plt.ylabel("Actual Values")

plt.title("Random Forest Confusion Matrix\n\n")

plt.show()