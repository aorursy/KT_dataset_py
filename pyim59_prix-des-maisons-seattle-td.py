# Directive pour afficher les graphiques dans Jupyter (inutile si on utilise Spyder)
%matplotlib inline
# Pandas : librairie de manipulation de données
# NumPy : librairie de calcul scientifique
# MatPlotLib : librairie de visualisation et graphiques
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
#lecture du dataset
df = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
df.head(10)
df.info()
df.plot(kind="scatter", x="long", y="lat", c="price", cmap="rainbow", s=3, figsize=(12,12))


df.info()
df.count()
df['year'] = pd.DatetimeIndex(df['date']).year
df['month'] = pd.DatetimeIndex(df['date']).month
df.groupby(['year','month'])['price'].mean().plot(kind = 'bar', figsize=(12,8))
df.groupby(['year','month'])['price'].count().plot(kind = 'bar', figsize=(12,8))
df = df.drop(['id','date'], axis=1)
tabcorr = df.corr()     # on peut utiliser aussi bos.corr(method='pearson') par exemple
plt.figure(figsize=(12,12))
sns.heatmap(abs(tabcorr), cmap="coolwarm")
sns.clustermap(abs(tabcorr), cmap="coolwarm")
from scipy.cluster import hierarchy as hc

corr = 1 - df.corr()
corr_condensed = hc.distance.squareform(corr)
link = hc.linkage(corr_condensed, method='ward')
plt.figure(figsize=(12,12))
den = hc.dendrogram(link, labels=df.columns, orientation='left', leaf_font_size=10)
correlations = tabcorr.price
print(correlations)
correlations = correlations.drop(['price'],axis=0)
print(abs(correlations).sort_values(ascending=False))
continuous_features = ['sqft_living','sqrt_log','sqft_above','sqft_basement','sqft_living15','sqrt_lot15','lat','long']
discrete_features = ['month','year','bedrooms','bathrooms','floors','waterfront','view','condition',
                     'grade','yr_built','yr_renovated','zipcode']
df1 = df[df.price<1000000].drop(discrete_features, axis=1)
X = df1.drop(['price'], axis=1)
y = df1.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)            # apprentissage
y_pred = lm.predict(X_test)         # prédiction sur l'ensemble de test
plt.figure(figsize=(12,12))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], color='red', linewidth=3)
plt.xlabel("Prix")
plt.ylabel("Prediction de prix")
plt.title("Prix reels vs predictions")
sns.distplot(y_test-y_pred)
print(np.sqrt(mean_squared_error(y_test, y_pred)))
scoreR2 = r2_score(y_test, y_pred)
print(scoreR2)
lm.score(X_test,y_test)
X = df1.drop(['price'], axis=1)
y = df1.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
from sklearn import ensemble
rf = ensemble.RandomForestRegressor()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
print(rf.score(X_test,y_test))
plt.figure(figsize=(12,12))
plt.scatter(y_test, y_rf)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], color='red', linewidth=3)
plt.xlabel("Prix")
plt.ylabel("Prediction de prix")
plt.title("Prix reels vs predictions")
sns.distplot(y_test-y_rf)
print(np.sqrt(mean_squared_error(y_test, y_rf)))
import xgboost as XGB
xgb  = XGB.XGBRegressor()
xgb.fit(X_train, y_train)
y_xgb = xgb.predict(X_test)
print(xgb.score(X_test,y_test))

plt.figure(figsize=(12,12))
plt.scatter(y_test, y_xgb)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], color='red', linewidth=3)
plt.xlabel("Prix")
plt.ylabel("Prediction de prix")
plt.title("Prix reels vs predictions")

