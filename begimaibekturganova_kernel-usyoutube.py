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
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/youtube-new/USvideos.csv')
print(df.columns)
category_data = [[1,"Film & Animation"],[2,"Autos & Vehicles"],[10,"Music"],[15,"Pets & Animals"],
       [17,"Sports"],[18,"Short Movies"],[19,"Travel & Events"],[20,"Gaming"],
       [21,"Vlog"],[22,"People & Blogs"],[23,"Comedy"],[24,"Entertainment"],
       [25,"News & Politics"],[26,"Howto & Style"],[27,"Education"],
       [28,"Science & Technology"],[29,"Nonprofits & Activism"],[30,"Movies"],
       [31,"Anime/Animation"],[32,"Action/Adventure"],[33,"Classics"],
       [34,"Comedy"],[35,"Documentary"],[36,"Drama"],[37,"Family"],
       [38,"Foreign"],[39,"Horror"],[40,"Sci-Fo/Fantasy"],[41,"Thriller"],
       [42,"Shorts"],[43,"Shows"],[44,"Trailers"]]

category_df = pd.DataFrame(category_data, columns=["category_id","category_name"])

cnt_video_per_category = df.groupby(["category_id"]).count().reset_index()
cnt_video_per_category = cnt_video_per_category.loc[:,['category_id','video_id']]
df_1 = pd.merge(cnt_video_per_category,category_df,left_on='category_id',right_on='category_id',how='left')
df_1 = df_1.sort_values(by='video_id', ascending = False)
df_1["Proportion"] = round((df_1["video_id"]/sum(df_1["video_id"]) * 100),2)
print(df_1)

sns.set(style="whitegrid")
plt.figure(figsize=(11, 10))
ax = sns.barplot(x="category_name",y="video_id", data=df_1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
# здесь мы проверяем рейтинг лайков, дислайков и комментариев
# Создаем 3 новые переменные
df["likes_rate"] = df["likes"] /df["views"] * 100
df["dislikes_rate"] = df["dislikes"] / df["views"] * 100
df["comment_rate"] = df["comment_count"] / df["views"] * 100

# группировка рейтинг лайков по категориям
cnt_likes_per_video_per_category = df.groupby("category_id").mean().reset_index()
cnt_likes_per_video_per_category = cnt_likes_per_video_per_category.loc[:,['category_id','likes_rate','dislikes_rate','comment_rate']]


df_2 = pd.merge(cnt_likes_per_video_per_category,category_df,left_on='category_id',right_on='category_id',how='left')
print(df_2)


df_2 = df_2.sort_values(by='likes_rate', ascending = False)
sns.set(style="whitegrid")
plt.figure(figsize=(11, 10))
plt.title("likes rate")
ax = sns.barplot(x="category_name",y="likes_rate", data=df_2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


df_2 = df_2.sort_values(by='dislikes_rate', ascending = False)
sns.set(style="whitegrid")
plt.figure(figsize=(11, 10))
plt.title("dislikes rate")
ax = sns.barplot(x="category_name",y="dislikes_rate", data=df_2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


df_2 = df_2.sort_values(by='comment_rate', ascending = False)
sns.set(style="whitegrid")
plt.figure(figsize=(11, 10))
plt.title("comments rate")
ax = sns.barplot(x="category_name",y="comment_rate", data=df_2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()
p_null= (len(df) - df.count())*100.0/len(df)
p_null
train = df[['views','likes','dislikes','comment_count','video_error_or_removed']]
df.isnull().any() 
train['video_error_or_removed'].fillna('False', inplace = True)
train.isnull().any()
train['video_error_or_removed'].interpolate(inplace = True)
train.isnull().any()
train['video_error_or_removed'].replace('True', 1, inplace = True)
train['video_error_or_removed'].replace('False', 0, inplace = True)
train.head()
sns.heatmap(train.corr(),cmap='coolwarm',annot=True)
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(train[['video_error_or_removed']])
train[['video_error_or_removed']] = pd.DataFrame(scaled)

train.head()
X_train, X_test, y_train, y_test = train_test_split(train[['views','likes','dislikes','comment_count']], train['video_error_or_removed'], test_size = 0.3)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
drugTree = DecisionTreeClassifier(criterion="gini")
drugTree.fit(X_train,y_train)
predTree = drugTree.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

nbc = GaussianNB()
nbc.fit(X_train,y_train)
y_pred = nbc.predict(X_test)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn import metrics

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
print(classification_report(y_test, predTree))
pd.DataFrame(
confusion_matrix(y_test, predTree),
columns=['Predicted No', 'Predicted Yes'],
index=['Actual No', 'Actual Yes']
)   
print("KNN's Accuracy: ", metrics.accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
pd.DataFrame(
confusion_matrix(y_test, pred),
columns=['Predicted No', 'Predicted Yes'],
index=['Actual No', 'Actual Yes']
)  
print("NB's Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
pd.DataFrame(
confusion_matrix(y_test, y_pred),
columns=['Predicted No', 'Predicted Yes'],
index=['Actual No', 'Actual Yes']
)
print("LR's Accuracy: ", metrics.accuracy_score(y_test, predictions ))
print(classification_report(y_test, predictions))
pd.DataFrame(
confusion_matrix(y_test, predictions),
columns=['Predicted No', 'Predicted Yes'],
index=['Actual No', 'Actual Yes']
)
target=df['video_error_or_removed']
target_count = target.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
