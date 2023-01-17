# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# machine learning

from sklearn.ensemble import RandomForestClassifier
train_df= pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

test_df= pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



train_df['type'] = le.fit_transform(train_df['type'])



test_df['type'] = le.fit_transform(test_df['type'])
missing_count = train_df.isnull().sum()

missing_count[missing_count > 0]

train_df.dropna(subset=['feature3','feature4','feature5','feature8','feature9','feature10','feature11'], axis=0, inplace=True)



train_df.reset_index(drop=True, inplace=True)



f3 = test_df["feature3"].mean()

test_df["feature3"].fillna(value=f3, inplace=True)

f3 = test_df["feature4"].mean()

test_df["feature4"].fillna(value=f3, inplace=True)

f3 = test_df["feature5"].mean()

test_df["feature5"].fillna(value=f3, inplace=True)

f3 = test_df["feature8"].mean()

test_df["feature8"].fillna(value=f3, inplace=True)

f3 = test_df["feature9"].mean()

test_df["feature9"].fillna(value=f3, inplace=True)

f3 = test_df["feature10"].mean()

test_df["feature10"].fillna(value=f3, inplace=True)

f3 = test_df["feature11"].mean()

test_df["feature11"].fillna(value=f3, inplace=True)
corr = train_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(12, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



x_train = train_df.drop(['id', 'rating','feature5','feature9','type'], axis=1)

y_train = train_df['rating']

x_test = test_df.drop(['id','feature9','feature5','type'], axis=1)



from sklearn.model_selection import GridSearchCV

param_grid = {

    'bootstrap': [True],

    'max_depth': [80, 90, 100, 110],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}

rf = RandomForestClassifier()

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 2, n_jobs = -1, verbose = 2)
grid_search.fit(x_train,y_train)

grid_search.best_params_



random_forest = RandomForestClassifier(bootstrap=True,max_depth= 90,max_features= 2,min_samples_leaf= 3,min_samples_split= 8,n_estimators= 200)

random_forest.fit(x_train, y_train)

y_pred = random_forest.predict(x_test)
submission = pd.DataFrame({'id':test_df['id'],'rating':y_pred})

submission.head(5)
filename = 'best4.csv'

submission.to_csv(filename,index=False)