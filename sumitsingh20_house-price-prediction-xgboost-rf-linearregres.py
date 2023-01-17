import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from xgboost import XGBRegressor



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score



from sklearn.model_selection import RandomizedSearchCV



df = pd.read_csv('../input/home-data-for-ml-course/train.csv')

df.head()
df.describe()
df.info()
fig,ax = plt.subplots(figsize = (8,8))

ax = sns.distplot(df.SalePrice)
df['MSZoning'].value_counts()
fig,ax = plt.subplots(figsize = (8,8))

ax = sns.barplot(x = df.MSZoning,

                 y = df.SalePrice);
df['SaleCondition'].value_counts()
sns.countplot(df.SaleCondition)
fig,ax = plt.subplots(figsize = (8,8))

ax = sns.barplot(x = df.SaleCondition,

                 y = df.SalePrice);
fig,ax = plt.subplots(figsize = (8,8))

ax = sns.scatterplot(x = df.LotArea,

                     y = df.SalePrice);
df['SaleType'].value_counts()
fig,ax = plt.subplots(figsize = (10,10))

ax  = sns.barplot(x = df.SaleType,

                  y = df.SalePrice);
df['PoolArea'].value_counts()
sns.barplot(x = df.PoolArea,y = df.SalePrice)
corr = df.corr()

corr
fig,ax = plt.subplots(figsize = (20,15))

ax = sns.heatmap(corr,

                 annot = True,

                 linewidths = 1.0,

                 fmt = '.2f',

                 cmap = 'YlGnBu');
df.head()
df.info()
for label, content in df.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isna(content).sum():

            print(label)
for label, content in df.items():

    if pd.api.types.is_numeric_dtype(content):

        if pd.isna(content).sum():

            df[label] = content.fillna(content.median())
for label, content in df.items():

    if pd.api.types.is_string_dtype(content):

        print(label)
for label,content in df.items():

    if not pd.api.types.is_numeric_dtype(content):

        df[label] = content.astype('category').cat.as_ordered()

        df[label] = pd.Categorical(content).codes+1
df.info()
x = df.drop('SalePrice',axis = 1)

y = df['SalePrice']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)
def scores(model):

    train_preds = model.predict(x_train)

    val_preds = model.predict(x_test)

    scores = {"Training MAE": mean_absolute_error(y_train, train_preds),

              "Valid MAE": mean_absolute_error(y_test, val_preds),

              "Training R^2": model.score(x_train, y_train),

              "Valid R^2": model.score(x_test, y_test)}

    return scores
%%time

ran_model = RandomForestRegressor(n_estimators = 1000,random_state = 42)

ran_model.fit(x_train,y_train)
scores(ran_model)
linear_model = LinearRegression()

linear_model.fit(x_train,y_train)
scores(linear_model)
linear_model.score(x_test,y_test)
xg_model = XGBRegressor()

xg_model.fit(x_train,y_train)
scores(xg_model)
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
%%time

rf = RandomForestRegressor()

rf_model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5, verbose=2, random_state=42, n_jobs = -1)

rf_model.fit(x_train, y_train)
scores(rf_model)
rf_model.best_params_
%%time

tuned_model = RandomForestRegressor(n_estimators = 600,

                                    min_samples_split = 5,

                                    min_samples_leaf = 1,

                                    max_features = 'sqrt',

                                    max_depth = 60,

                                    bootstrap = False)

tuned_model.fit(x_train,y_train)
scores(tuned_model)
r2_scores = pd.DataFrame({'RandomForest': ran_model.score(x_test,y_test),

                       'LinearRegression': linear_model.score(x_test,y_test),

                       'XG Boost': xg_model.score(x_test,y_test),

                       'Tuned RandomForest': tuned_model.score(x_test,y_test)},

                        index = [0])
r2_scores.T.plot(kind = 'bar',

              figsize = (10,10))

plt.title('Scores of all Model')

plt.xlabel('Model Name')

plt.ylabel('Scores');