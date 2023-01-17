import pandas as pd
import matplotlib.pyplot as plt

# tratamento de dados
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# métodos de regressão
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor

# métricas de erro
from sklearn.metrics import mean_absolute_error
df = pd.read_csv('../input/imdb-5000-movie-dataset/movie_metadata.csv')
df.shape
df.head()
df.dtypes
df2 = df.drop(["actor_1_name", "actor_2_name", "actor_3_name", "genres", "movie_title", "plot_keywords", "movie_imdb_link", "language", "country", "director_name"], axis=1)
df2 = pd.get_dummies(df2, columns=["content_rating", "color"], drop_first=True)
df2.shape
plt.hist(df["imdb_score"])
fig = plt.figure()
ax = plt.gca()
ax.scatter(df["budget"], df["imdb_score"], alpha=0.1, edgecolors='none')
ax.set_xscale('log')
plt.xlabel("Budget")
plt.ylabel("IMDB")
y = df2["imdb_score"]
X = df2.drop("imdb_score", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=10)
print(y_train.shape, y_test.shape)
reg = make_pipeline(SimpleImputer(strategy="most_frequent"), StandardScaler(), RandomForestRegressor(n_estimators=200, random_state=10))
reg.fit(X_train, y_train)
y_train_pred = reg.predict(X_train)

print('Erro médio absoluto: ', mean_absolute_error(y_train, y_train_pred))
y_test_pred = reg.predict(X_test)
plt.scatter(y_test, y_test_pred, alpha=0.2, edgecolors='none')
plt.xlabel("IMDB (real)")
plt.ylabel("IMDB (predição)")
print('Erro médio absoluto: ', mean_absolute_error(y_test, y_test_pred))
err = y_test_pred - y_test

plt.hist(err)
plt.xlabel("Erro")
plt.ylabel("Quantidade")