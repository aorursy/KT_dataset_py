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


columns = ['timedelta', 'n_tokens_title', 'n_tokens_content','num_hrefs', 'num_self_hrefs', 'num_imgs', 'num_videos',

           'data_channel_is_lifestyle', 'data_channel_is_entertainment', 'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech', 'data_channel_is_world',

          'weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend', 'shares',]

new_df = df[columns]

new_df.head()
from sklearn.model_selection import train_test_split

X = new_df.drop('shares', axis=1)

y = new_df['shares']

X_train, X_valid, y_train, y_valid = train_test_split( X, y, test_size=0.3, random_state=2019)
from sklearn.tree import DecisionTreeRegressor



tree = DecisionTreeRegressor(max_depth=3,random_state=2019)

tree.fit(X_train, y_train)
from sklearn.tree import export_graphviz



export_graphviz(tree, out_file='tree.dot')

print(open('tree.dot').read()) 
print(tree.score(X_valid,y_valid))
from sklearn.metrics import mean_squared_error

y_pred = tree.predict(X_valid)



m1=mean_squared_error(y_valid, y_pred)

m1
from sklearn.model_selection import GridSearchCV





tree_params = {'max_depth': list(range(2, 11)),

               'min_samples_leaf': list(range(2, 11))}





tree_grid = GridSearchCV(tree, 

                        tree_params, 

                        scoring='neg_mean_squared_error',

                        cv=5)

tree_grid.fit(X_train, y_train)
tree_grid.best_params_
tree_param1 = {'max_depth': list(range(2, 11))}



tree_grid1 = GridSearchCV(tree, 

                        tree_param1, 

                        scoring='neg_mean_squared_error',

                        cv=5) # или cv=kf

tree_grid1.fit(X_train, y_train)
tree_param2 = {'min_samples_leaf': list(range(2, 11))}



tree_grid2 = GridSearchCV(tree, 

                        tree_param2, 

                        scoring='neg_mean_squared_error',

                        cv=5) # или cv=kf

tree_grid2.fit(X_train, y_train)
# Отрисовка графиков

import matplotlib.pyplot as plt



fig, ax = plt.subplots(nrows=1, ncols=2) # 2 графика рядом с одинаковым масштабом по оси Оу



ax[0].plot(tree_param1['max_depth'], tree_grid1.cv_results_['mean_test_score']) # accuracy vs max_depth

ax[0].set_xlabel('max_depth')

ax[0].set_ylabel('Mean accuracy on test set')



ax[1].plot(tree_param2['min_samples_leaf'], tree_grid2.cv_results_['mean_test_score']) # accuracy vs min_samples_leaf

ax[1].set_xlabel('min_samples_leaf')

ax[1].set_ylabel('Mean accuracy on test set');
best_tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=2019)

best_tree.fit(X_train, y_train)

export_graphviz(best_tree, out_file='best_tree.dot')

print(open('best_tree.dot').read()) 
print(best_tree.score(X_valid,y_valid))
y_pred = best_tree.predict(X_valid)



m1=mean_squared_error(y_valid, y_pred)

m1
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=2019)

rf.fit(X_train,y_train)
print(rf.score(X_valid,y_valid))
y_pred = rf.predict(X_valid)



m1=mean_squared_error(y_valid, y_pred)

m1
from sklearn.model_selection import GridSearchCV





rf_params = {'max_depth': list(range(2, 11)),

               'min_samples_leaf': list(range(2, 11))

            }



print(rf_params)

rf_grid = GridSearchCV(rf, 

                        rf_params, 

                        scoring='neg_mean_squared_error',

                        cv=5)

rf_grid.fit(X_train, y_train)
rf_grid.best_params_
from sklearn.model_selection import GridSearchCV





rf_params = {

             'min_samples_split': list(range(2, 13))

            }



print(rf_params)

rf_grid = GridSearchCV(rf, 

                        rf_params, 

                        scoring='neg_mean_squared_error',

                        cv=5)

rf_grid.fit(X_train, y_train)
rf_grid.best_params_
rf_new = RandomForestRegressor(n_estimators=100, random_state=2019, max_depth=8, min_samples_leaf=12)

rf_new.fit(X_train,y_train)
print(rf_new.score(X_valid,y_valid))
y_pred = rf_new.predict(X_valid)



m1=mean_squared_error(y_valid, y_pred)

m1
import matplotlib.pyplot as plt



features = dict(zip(range(len(new_df.columns)-1), new_df.columns[:-1]))



# Важность признаков

importances = rf_new.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = max(10, len(new_df.columns[:-1]))

feature_indices = [ind for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")



for f in range(num_to_plot):

    print(f+1, features[feature_indices[f]], importances[indices[f]])



plt.figure(figsize=(25,10))

plt.title("Feature importances")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

               color=([str(i/float(num_to_plot+1)) for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features[i]) for i in feature_indices]);



# Самым весомым параметром является timedelta - кол-во дней между публикацией статьи и получением датасета.