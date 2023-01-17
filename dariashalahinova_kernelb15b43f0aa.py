# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/OnlineNewsPopularityReduced.csv")

df.head().T
def outliers_indices(feature):



    mid = df[feature].mean()

    sigma = df[feature].std()

    return df[(df[feature] < mid - 3*sigma) | (df[feature] > mid + 3*sigma)].index

wrong_share = outliers_indices('shares')

wrong_vid = outliers_indices('num_videos')

wrong_img = outliers_indices('num_imgs')

wrong_content = outliers_indices('n_tokens_content')

wrong_title = outliers_indices('n_tokens_title')

out = set(wrong_share) | set(wrong_vid) | set(wrong_img) | set(wrong_content) | set(wrong_title)



df.drop(out, inplace=True)


rows = ['shares', 'n_tokens_title', 'n_tokens_content', 'num_hrefs', 'num_imgs', 'num_videos', 'num_keywords']



df[rows].corr(method='spearman')
rows = ['shares','kw_max_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max', 'kw_max_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg']

df[rows].corr(method='spearman')
y=df['shares']

df = df.drop('url', axis=1)

df = df.drop('shares', axis=1)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(df)

df_scaled = scaler.transform(df)

df_scaled




from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df_scaled, 

                                                    y, 

                                                    test_size=0.25, 

                                                    random_state=19)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=1)

knn.fit(x_train, y_train)

pred=knn.predict(x_test)

pred
from sklearn.metrics import mean_squared_error

temp = mean_squared_error(y_test, pred)

temp
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

knn = KNeighborsRegressor(n_neighbors=5)

scores = cross_val_score(knn, df, y, 

                         cv=kf, scoring='neg_mean_squared_error')

print(scores)

mean_score = scores.mean()

print(mean_score)
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 21)}

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='neg_mean_squared_error',

                        cv=5) # или cv=kf

a = knn_grid.fit(X_train, y_train)
