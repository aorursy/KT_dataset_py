import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plote
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
df = pd.read_csv('../input/GDSChackathon.csv')
df.head()
plt.figure(figsize=(8,6))
plt.scatter(range(df.shape[0]), np.sort(df.imdb_score.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('imdb_score', fontsize=12)
plt.show()
plt.figure(figsize=(12,8))
sns.distplot(df.imdb_score.values, bins=50, kde=False)
plt.xlabel('imdb_score', fontsize=12)
plt.show()
df.describe()
df.info()
le = preprocessing.LabelEncoder()
encoding_list=['director_name','actor_2_name','genres','actor_1_name','movie_title','language','country','content_rating']
for i in encoding_list:
    df[i] = le.fit_transform(df[i].astype(str))
y = df['imdb_score']
x = df.drop('imdb_score', axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)
X_train.shape
X_test.shape
import xgboost as xgb
xgb = xgb.XGBRegressor()
xgb.fit(X_train,y_train)
predictions_xgb = xgb.predict(X_test)
error_xgb = metrics.mean_squared_error(y_test, predictions_xgb)
print(error_xgb)
importances = xgb.feature_importances_
feature_names = x.columns.values
data = pd.DataFrame({'x': feature_names,'importances':importances})
new_index = (data['importances'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)
plt.figure(figsize=(15,10))
ax= sns.barplot(x=sorted_data['x'], y=sorted_data['importances'])
plt.xticks(rotation= 90)
plt.xlabel('Features')
plt.ylabel('Importances')
plt.title('feature importances')
plt.show()
le = preprocessing.LabelEncoder()
encoding_list=['director_name','actor_2_name','genres','actor_1_name','movie_title','language','country','content_rating']
for i in encoding_list:
    df[i] = le.fit_transform(df[i].astype(str))
y = df['imdb_score']
x = df.drop(['imdb_score','color','plot_keywords','actor_3_name','movie_imdb_link','aspect_ratio'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)
X_train.shape
xgb.fit(X_train,y_train)
predictions_xgb = xgb.predict(X_test)
error_xgb = metrics.mean_squared_error(y_test, predictions_xgb)
print(error_xgb)
