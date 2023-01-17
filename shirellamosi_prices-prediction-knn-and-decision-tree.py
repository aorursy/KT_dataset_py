import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



from sklearn.tree import DecisionTreeRegressor, export_graphviz

import pydot

from IPython.display import Image

from sklearn.externals.six import StringIO



from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import pairwise_distances



from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline



%matplotlib inline
def rmse(y_true, y_pred):

    se = (y_true - y_pred)**2

    mse = se.mean()

    return np.sqrt(mse)
df = pd.read_csv('../input/kc_house_data.csv', index_col='id', parse_dates=['date'])

pd.set_option("display.max_columns", 30)

df.head(3)
print(df.shape)

print()

df.info()
sns.pairplot(df)
ax = df.plot.scatter(x='long', y='lat',

                     c=df.price.sort_values(ascending=True), 

                     colormap='inferno', figsize=(15,7))

ax.xaxis.tick_top()
plt.figure(figsize=(16, 6))

ax = sns.distplot(df['price'], bins= 500, rug=True)

ax.set_xlim([0, 7000000])

ax.set_title('Price Distribution')

ax.set_ylabel('Frequency');
corrmat = df.corr()

cols = corrmat.nlargest(30, 'price')['price'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.5)

plt.figure(figsize=(15, 10))



hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.show()
df.corr(method='pearson').sort_values(['price'], ascending=False)['price'][1:10]
sns.set(rc={'figure.figsize':(10,5)})

sns.boxplot(data=df, y='price', x='grade')
sns.relplot(data=df, x='sqft_living', y='price',

                 legend='full', height=5, aspect=2)
sns.relplot(data=df, x='sqft_living', y='price',hue='grade',

                 legend='full', height=5, aspect=2)
plt.figure(figsize=(16,7))

sns.barplot(data = df, x='zipcode', y='price')

plt.xticks(rotation=55)

plt.title('Price Per Zipcode')
df[df.bedrooms > 30]
df.bedrooms.replace({33:3}, inplace=True)

print(df.shape)
df[df.sqft_living > 11500]
print(df.shape)

df = df[df.sqft_living < 11500]

print(df.shape)
df[df.sqft_lot > 1250000]
print(df.shape)

df = df[df.sqft_lot < 1250000]

print(df.shape)
df[df.sqft_lot15 > 800000]
print(df.shape)

df = df[df.sqft_lot15 < 800000]

print(df.shape)
df.bedrooms.sort_values().head()
print(df.shape)

df = df[df.bedrooms != 0]

print(df.shape)
df.bathrooms.sort_values().head()

#pay attention - there are houses with no bathroom! drop them?
print(df.shape)

df = df[df.bathrooms != 0.0]

print(df.shape)
df.price.sort_values(ascending=False).head(10)
(df.price>5000000).sum()
print(df.shape)

df = df[df['price']<5000000]

df.shape
sns.pairplot(df)
ax = df.plot.scatter(x='long', y='lat',c=df.price.sort_values(ascending=True),colormap='inferno', figsize=(15,8))

ax.xaxis.tick_top()
#interaction of correlated features:

df['sqft_living*grade'] = df['sqft_living']* df['grade']
df_knn = df.copy()
#New Boolean Feature:

def renovate(x):

    if x>0:

        return 1

    else:

        return 0
df['renovated'] = df['yr_renovated'].apply(renovate)
df['year'] = df.date.dt.year
def age(x):

    if x['yr_renovated'] == 0:

        age = x['year'] - x['yr_built']

    else:

        age = x['year'] - x['yr_renovated']

    return age
df['age'] = df.apply(age, axis=1)

df = df.drop(columns='year')

df_tree = df.copy()
X = df_tree.drop(columns=['price', 'date', 'sqft_above', 'sqft_lot', 'sqft_lot15', 

                          'floors', 'condition', 'view', 'sqft_basement', 'sqft_living15'])

y = df_tree.price



X.head(2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.27)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
X_train2 = X_train

X_test2 = X_test



parameters = {

    'max_depth': range (5, 20),

    'min_samples_leaf': [5, 10, 15, 20, 25]}



grid_search = GridSearchCV(DecisionTreeRegressor(), parameters, cv=10, return_train_score=False, scoring='neg_mean_squared_error', n_jobs=4)

grid_search.fit(X_train2, y_train)



grid_results = pd.DataFrame(grid_search.cv_results_)
grid_search.best_estimator_
model = grid_search.best_estimator_



y_train_pred = model.predict(X_train2)

print(f'Train rmse: {rmse((y_train_pred), (y_train)):.3f}')

y_test_pred = model.predict(X_test2)

print(f'Test rmse: {rmse((y_test_pred), (y_test)):.3f}')
dot_data = StringIO() 

export_graphviz(model, out_file=dot_data, feature_names=X.columns)  

graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 

graph.write_png("dec_tree.png")

# Image(graph.create_png(), width=1600) 
from IPython.display import Image

Image("dec_tree.png")

print(f'Std: {df.price.std():.3f}')

print(f'Mean: {df.price.mean():.3f}')
grid_results.sort_values(by='rank_test_score').head(5)
df_scores = grid_results.sort_values(by='rank_test_score')[['params', 'mean_test_score', 'std_test_score' ]]

df_scores[['mean_test_score',  'std_test_score']] = np.sqrt(df_scores[['mean_test_score',  'std_test_score']].abs())

df_scores.head()
grid_search.best_params_
df_scores[['mean_test_score',  'std_test_score']].plot(kind='scatter', x='mean_test_score', y='std_test_score', 

                                                       color='lightblue', figsize=(10,5))



P = [df_scores.iloc[0,1] , df_scores.iloc[0,2]]

plt.plot(P[0], P[1], marker='o', markersize=5, color="darkblue")

X = df_knn.drop(columns=['price','date','sqft_above', 'bedrooms',

                         'bathrooms','sqft_lot', 'floors','view', 'condition','sqft_basement', 

                         'yr_built', 'yr_renovated' ])

y=df_knn.price



X.head(2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666, test_size=0.27)



X_train.shape, X_test.shape, y_train.shape, y_test.shape
steps = [('scaler', MinMaxScaler()), 

         ('model', KNeighborsRegressor())]

pipe = Pipeline(steps)





params = {'model__n_neighbors': range(3, 10),

          'model__metric': ['minkowski', 'manhattan'], 

         'scaler': [MinMaxScaler(), MaxAbsScaler()]}        



gs = GridSearchCV(pipe, param_grid=params, cv=10, return_train_score=False, n_jobs=4, scoring = 'neg_mean_squared_error')



gs.fit(X_train, y_train)

gs_results = pd.DataFrame(gs.cv_results_)

gs.best_estimator_
gs.best_params_
y_pred_train = gs.predict(X_train)

y_pred_test = gs.predict(X_test)



train_rmse = rmse((y_train), (y_pred_train))

test_rmse = rmse((y_test),(y_pred_test))

print(f'Train RMSE: {train_rmse:.3f}')

print(f'Test RMSE: {test_rmse:.3f}')
gs_results.sort_values(by='rank_test_score').head(5)
df_scores = gs_results.sort_values(by='rank_test_score')[['params', 'mean_test_score', 'std_test_score' ]]

df_scores[['mean_test_score',  'std_test_score']] = np.sqrt(df_scores[['mean_test_score',  'std_test_score']].abs())

df_scores.head(5)
gs.best_params_
df_scores[['mean_test_score',  'std_test_score']].plot(kind='scatter', x='mean_test_score', y='std_test_score', 

                                                       color='lightblue', figsize=(10,5))



P = [df_scores.iloc[0,1] , df_scores.iloc[0,2]]

plt.plot(P[0], P[1], marker='o', markersize=5, color="darkblue")
