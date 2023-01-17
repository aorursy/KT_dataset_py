import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error

from math import sqrt
game_df = pd.read_csv('/kaggle/input/rawg-game-dataset/game_info.csv')

game_df
percent_missing = game_df.isnull().sum() * 100 / len(game_df)

missing_value_df = pd.DataFrame({'column_name': game_df.columns,

                                 'percent_missing': percent_missing})

missing_value_df.sort_values(by=['percent_missing'], ascending=False).head()
game_df = game_df.drop(['website', 'tba', 'publishers', 'esrb_rating', 'metacritic',\

                        'platforms', 'developers', 'genres', 'slug', 'name', 'updated'], axis=1).dropna()
game_df['released'] = game_df['released'].apply(lambda x: str(x).split('-')[0]).astype('int')
f = plt.figure(figsize=(19, 15))

plt.matshow(game_df.corr(), fignum=f.number)

plt.xticks(range(game_df.shape[1]), game_df.columns, fontsize=14,rotation=45)

plt.yticks(range(game_df.shape[1]), game_df.columns, fontsize=14)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
def corrfunc(x, y, **kws):

    (r, p) = pearsonr(x, y)

    ax = plt.gca()

    ax.annotate("r = {:.2f} ".format(r),

                xy=(.1, .9), xycoords=ax.transAxes)

    ax.annotate("p = {:.3f}".format(p),

                xy=(.4, .9), xycoords=ax.transAxes)



matplotlib.rcParams["font.size"] = 16



graph = sns.pairplot(game_df, x_vars=["added_status_beaten", "added_status_dropped", \

                              "added_status_toplay", "rating", "suggestions_count", \

                              "game_series_count"], y_vars=["reviews_count"], height=5, aspect=.8, kind="reg");

graph.map(corrfunc)

plt.show()
game_df['logadded_status_beaten'] = np.log(game_df[['added_status_beaten']] + 1)

game_df['logadded_status_dropped'] = np.log(game_df[['added_status_dropped']] + 1)

game_df['logadded_status_toplay'] = np.log(game_df[['added_status_toplay']] + 1)

game_df['log_rating'] = np.log(game_df[['rating']] + 1)

game_df['log_suggestions_count'] = np.log(game_df[['suggestions_count']] + 1)

game_df['log_game_series_count'] = np.log(game_df[['game_series_count']] + 1)

game_df['logreviews_count'] = np.log(game_df[['reviews_count']] + 1)
graph = sns.pairplot(game_df, x_vars=["logadded_status_beaten", "logadded_status_dropped", \

                              "logadded_status_toplay", "log_rating", "suggestions_count", \

                              "log_game_series_count"], y_vars=["logreviews_count"], height=5, aspect=.8, kind="reg")

graph.map(corrfunc)

plt.show()
X = game_df[["logadded_status_beaten", "logadded_status_dropped", \

                              "logadded_status_toplay", "log_rating", "suggestions_count", \

                              "log_game_series_count"]]

y = game_df[["logreviews_count"]]



X_train, X_test, y_train, y_test = train_test_split( \

    X, y, test_size=0.30, random_state=42)



reg = LinearRegression().fit(X_train, y_train)

r2 = reg.score(X_train, y_train)
print("R-Squared: " + str(r2))

adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 

          (X_train.shape[0] - X_train.shape[1] - 1)))

print("Adjusted R-Squared: " + str(adj_r2))
X = game_df[["logadded_status_dropped", \

                              "logadded_status_toplay", "log_rating", "suggestions_count", \

                              "log_game_series_count"]]

y = game_df[["logreviews_count"]]



X_train, X_test, y_train, y_test = train_test_split( \

    X, y, test_size=0.30, random_state=42)



reg_test = LinearRegression().fit(X_train, y_train)

r2 = reg_test.score(X_train, y_train)



adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 

          (X_train.shape[0] - X_train.shape[1] - 1)))



print("Adjusted R-squared Dropping logadded_status_beaten: " + str(adj_r2))



X = game_df[["logadded_status_beaten", \

                              "logadded_status_toplay", "log_rating", "suggestions_count", \

                              "log_game_series_count"]]

y = game_df[["logreviews_count"]]



X_train, X_test, y_train, y_test = train_test_split( \

    X, y, test_size=0.30, random_state=42)



reg_test = LinearRegression().fit(X_train, y_train)

r2 = reg_test.score(X_train, y_train)



adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 

          (X_train.shape[0] - X_train.shape[1] - 1)))



print("Adjusted R-squared Dropping logadded_status_dropped: " + str(adj_r2))



X = game_df[["logadded_status_beaten", "logadded_status_dropped", \

                              "log_rating", "suggestions_count", \

                              "log_game_series_count"]]

y = game_df[["logreviews_count"]]



X_train, X_test, y_train, y_test = train_test_split( \

    X, y, test_size=0.30, random_state=42)



reg_test = LinearRegression().fit(X_train, y_train)

r2 = reg_test.score(X_train, y_train)



adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 

          (X_train.shape[0] - X_train.shape[1] - 1)))



print("Adjusted R-squared Dropping logadded_status_toplay: " + str(adj_r2))



X = game_df[["logadded_status_beaten", "logadded_status_dropped", \

                              "logadded_status_toplay", "suggestions_count", \

                              "log_game_series_count"]]

y = game_df[["logreviews_count"]]



X_train, X_test, y_train, y_test = train_test_split( \

    X, y, test_size=0.30, random_state=42)



reg_test = LinearRegression().fit(X_train, y_train)

r2 = reg_test.score(X_train, y_train)



adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 

          (X_train.shape[0] - X_train.shape[1] - 1)))



print("Adjusted R-squared Dropping log_rating: " + str(adj_r2))



X = game_df[["logadded_status_beaten", "logadded_status_dropped", \

                              "logadded_status_toplay", "log_rating", \

                              "log_game_series_count"]]

y = game_df[["logreviews_count"]]



X_train, X_test, y_train, y_test = train_test_split( \

    X, y, test_size=0.30, random_state=42)



reg_test = LinearRegression().fit(X_train, y_train)

r2 = reg_test.score(X_train, y_train)



adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 

          (X_train.shape[0] - X_train.shape[1] - 1)))



print("Adjusted R-squared Dropping suggestions_count: " + str(adj_r2))



X = game_df[["logadded_status_beaten", "logadded_status_dropped", \

                              "logadded_status_toplay", "log_rating", "suggestions_count"]]

y = game_df[["logreviews_count"]]



X_train, X_test, y_train, y_test = train_test_split( \

    X, y, test_size=0.30, random_state=42)



reg_test = LinearRegression().fit(X_train, y_train)

r2 = reg_test.score(X_train, y_train)



adj_r2 = (1 - (1 - r2) * ((X_train.shape[0] - 1) / 

          (X_train.shape[0] - X_train.shape[1] - 1)))



print("Adjusted R-squared Dropping log_game_series_count: " + str(adj_r2))
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



alphas = np.array([1,0.1,0.01,0.001,0.0001,0])



from sklearn.preprocessing import StandardScaler



X = game_df[["logadded_status_beaten", "logadded_status_dropped", \

                              "logadded_status_toplay", "log_rating", "suggestions_count", \

                              "log_game_series_count"]]

y = game_df[["logreviews_count"]]



X_train, X_test, y_train, y_test = train_test_split( \

    X, y, test_size=0.30, random_state=42)



model = Ridge()

grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas))

grid.fit(X_train, y_train)

print("Best Estimator value: " + str(grid.best_estimator_.alpha))
model = Ridge(alpha=1.0)



rreg = model.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print("Predictions: " + str(y_pred))

print("Actual: " + str(y_test['logreviews_count'].values))
rmse = sqrt(mean_squared_error(y_test, y_pred))

print("Model RMSE: " + str(rmse))
print("Model RMSE (Transformed Back): " + str(np.exp(rmse)))
y_pred = rreg.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred))

print("Regularized Model RMSE (Transformed Back): " + str(np.exp(rmse)))
y_test = y_test.reset_index()

max_index = list(y_test[y_test['logreviews_count'] == y_test['logreviews_count'].max()].index)[0]

print("Max test value: " + str(y_test['logreviews_count'][max_index]))

print("Max test value (Transformed back): " + str(np.ceil(np.exp(y_test['logreviews_count'][max_index]) - 1)))
print("Prediction on max test value: " + str(y_pred[max_index][0]))

print("Prediction on max test value (Transformed Back): " + str(np.ceil(np.exp(y_pred[max_index][0]) - 1)))