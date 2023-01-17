%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/board-games-prediction-data/games.csv')

print(data.shape)

data.head()
data_explore = data.copy()
drop_features = ['id', 'type', 'name', 'bayes_average_rating']

data_explore = data_explore.drop(columns=drop_features, axis=1)
data_explore.info()
data_explore.describe()
def plot_histogram(data):

    ax = plt.gca()

    counts, _, patches = ax.hist(data)

    for count, patch in zip(counts, patches):

        if count>0:

            ax.annotate(str(int(count)), xy=(patch.get_x(), patch.get_height()+5))

    if data.name:

        plt.xlabel(data.name)
plt.figure(figsize=(15, 25))

i=1

for col in data_explore.columns:

    plt.subplot(6, 3, i)

    plot_histogram(data_explore[col])

    i+=1
plt.title('Histogram of Average Ratings')

plot_histogram(data_explore['average_rating'])
data_explore_zero_ratings = data_explore[data_explore['average_rating']==0]

data_explore_zero_ratings.describe()
data_explore = data_explore[data_explore['average_rating']>0]

data = data[data['average_rating']>0] # making this change in orignal dataframe
plt.title('Histogram of Average Ratings')

plot_histogram(data_explore['average_rating'])
plt.title('Histogram of Average Weight')

plot_histogram(data_explore['average_weight'])
data_explore = data_explore[data_explore['average_weight']>0]

data = data[data['average_weight']>0] # making this change in orignal dataframe
plot_histogram(data_explore['yearpublished'])
data_explore = data_explore[data_explore['yearpublished']>0]

plot_histogram(data_explore['yearpublished'])
data_explore_1920 = data_explore.query('yearpublished > 1900 and yearpublished < 2000')

plot_histogram(data_explore_1920['yearpublished'])
data_explore = data_explore[data_explore['yearpublished']>1950]

data = data[data['yearpublished']>1950]

plot_histogram(data_explore['yearpublished'])
data_7585 = data_explore.query('yearpublished >= 1975 and yearpublished <= 1985')

data_0515 = data_explore.query('yearpublished >= 2005 and yearpublished <= 2015')



plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)

plt.scatter(data_7585['yearpublished'], data_7585['average_rating'], s=data_7585['average_weight']*10)

plt.title("1975-85 (Games {})".format(data_7585['yearpublished'].count()))

plt.xlabel('Published year')

plt.ylabel('Average rating')

plt.subplot(1, 2, 2)

plt.scatter(data_0515['yearpublished'], data_0515['average_rating'], s=data_0515['average_weight']*10)

plt.title("2005-15 (Games {})".format(data_0515['yearpublished'].count()))

plt.xlabel('Published year')

plt.show()
columns = ['minplaytime', 'maxplaytime', 'minplayers', 'maxplayers', 'users_rated']

plt.figure(figsize=(15, 8))

sns.boxplot(data=data_explore[columns])

plt.ylim(-100, 500)
Q1 = data_explore.quantile(0.25)

Q3 = data_explore.quantile(0.75)

IQR = Q3 - Q1

((data_explore < (Q1 - 1.5 * IQR)) | (data_explore > (Q3 + 1.5 * IQR))).sum()
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='median')
data_columns = data_explore.columns

data_explore = imputer.fit_transform(data_explore)

data_explore = pd.DataFrame(data=data_explore, columns=data_columns)
drop_features.append('playingtime')

data_explore = data_explore.drop(columns=['playingtime'], axis=1)
plt.figure(figsize=(15, 10))

corr_matrix = data_explore.corr()

sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), square=True, annot=True)
corr_matrix['average_rating'].sort_values(ascending=False)
data.shape
y = data[['average_rating']].copy()

X = data.drop(columns=['average_rating'], axis=1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
feature_columns =[ feature for feature in list(X.columns) if feature not in drop_features ]
from sklearn.compose import ColumnTransformer



drop_feature_cols = ColumnTransformer(transformers=[('drop_columns', 'drop', drop_features)], remainder='passthrough')
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
pre_process = Pipeline(steps=[('drop_features', drop_feature_cols),

                              ('imputer', SimpleImputer(strategy="median")),

                              ('scaler', StandardScaler())])
X_train_transformed = pre_process.fit_transform(X_train)

X_test_transformed = pre_process.transform(X_test)
from sklearn.model_selection import cross_val_score



def cv_results(model, X, y):

    scores = cross_val_score(model, X, y, cv = 7, scoring="neg_mean_squared_error", n_jobs=-1)

    rmse_scores = np.sqrt(-scores)

    print('CV Scores: ', rmse_scores)

    print('rmse: {},  S.D.:{} '.format(np.mean(rmse_scores), np.std(rmse_scores)))
from sklearn.linear_model import LinearRegression



linear_reg = LinearRegression()

linear_reg.fit(X_train_transformed, y_train)
coefs = list(zip(feature_columns, linear_reg.coef_[0]))

coefs.sort(key= lambda x:x[1], reverse=True)

coefs
print("Linear Regression Model Cross Validation Results")

cv_results(linear_reg, X_train_transformed, y_train)
from sklearn.decomposition import PCA



pca = PCA(n_components=0.95)   # Keeping variance 95% so that we will not loose much information.

X_train_reduced = pca.fit_transform(X_train_transformed)

X_test_reduced = pca.transform(X_test_transformed)

pca.n_components_, X_train_reduced.shape[1]
linear_reg.fit(X_train_reduced, y_train)
print("Linear Regression Model Cross Validation Results")

cv_results(linear_reg, X_train_reduced, y_train)
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(criterion='mse', random_state=42)

tree_reg.fit(X_train_transformed, y_train)
coefs = list(zip(feature_columns, tree_reg.feature_importances_))

coefs.sort(key= lambda x:x[1], reverse=True)

coefs
print("Decision Tree Regression Model Cross Validation Results")

cv_results(tree_reg, X_train_transformed, y_train)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(criterion='mse', random_state=42, n_jobs=-1)

forest_reg.fit(X_train_transformed, y_train.values.flatten())
coefs = list(zip(feature_columns, forest_reg.feature_importances_))

coefs.sort(key= lambda x:x[1], reverse=True)

coefs
print("Random Forest Regression Model Cross Validation Results")

cv_results(forest_reg, X_train_transformed, y_train.values.flatten())
from sklearn.model_selection import GridSearchCV



grid_parm=[{'n_estimators':[25, 50, 75, 100], 'max_depth':[4, 8, 12, 16]}]

grid_search = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), grid_parm, cv=5, scoring="neg_mean_squared_error", return_train_score=True, n_jobs=-1)

grid_search.fit(X_train_transformed, y_train.values.flatten())
cvres = grid_search.cv_results_

print("Results for each run of Random Forest Regression...")

for train_mean_score, test_mean_score, params in zip(cvres["mean_train_score"], cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-train_mean_score), np.sqrt(-test_mean_score), params)
grid_search.best_params_, -grid_search.best_score_
best_forest_reg = grid_search.best_estimator_

best_forest_reg.max_depth=12

best_forest_reg
print("Best Random Forest Cross Validation Results")

cv_results(best_forest_reg, X_test_transformed, y_test)
y_train_pred = best_forest_reg.predict(X_train_transformed)

y_test_pred = best_forest_reg.predict(X_test_transformed)
y_pred = np.concatenate((y_train_pred, y_test_pred), axis=0)

y_pred.shape
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

plt.title('Histogram of Observed Average Ratings')

plt.hist(data['average_rating'], bins=np.arange(1, 10), rwidth=0.85)

plt.subplot(1, 2, 2)

plt.title('Histogram of Predicted Average Ratings')

plt.hist(y_pred, bins=np.arange(1, 10), rwidth=0.85)

plt.show()
combine_data = pd.concat([X_train, X_test], axis=0)
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

plt.scatter(data['yearpublished'], data['average_rating'],  c='green')

plt.title('Distibution of Observed Average Rating')

plt.subplot(1, 2, 2)

plt.scatter(combine_data['yearpublished'], y_pred, c='red')

plt.title('Distibution of Predicted Average Rating')

plt.show()
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

plt.scatter(data['average_weight'], data['average_rating'],  c='green')

plt.title('Distibution of Observed Average Rating')

plt.ylabel('Average Rating')

plt.xlabel('Average Weight')

plt.subplot(1, 2, 2)

plt.scatter(combine_data['average_weight'], y_pred, c='red')

plt.title('Distibution of Predicted Average Rating')

plt.ylabel('Average Rating')

plt.xlabel('Average Weight')

plt.show()