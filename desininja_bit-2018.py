
import xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.pyplot as plote

from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor




data = pd.read_csv('../input/GDSChackathon.csv')

data.head()
data.info()
sns.heatmap(data.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')
#YELLOW LINES REPRESENT MISSING VALUES.
#mode
data['country'].fillna(data['country'].mode()[0], inplace=True)
data['language'].fillna(data['language'].mode()[0], inplace=True)
data['color'].fillna(data['color'].mode()[0], inplace=True)
data['director_name'].fillna(data['director_name'].mode()[0], inplace=True)
data['actor_2_name'].fillna(data['actor_2_name'].mode()[0], inplace=True)
data['actor_1_name'].fillna(data['actor_1_name'].mode()[0], inplace=True)
data['content_rating'].fillna(data['content_rating'].mode()[0], inplace=True)
data['movie_title'].fillna(data['movie_title'].mode()[0], inplace=True)
data['actor_3_name'].fillna(data['actor_3_name'].mode()[0], inplace=True)  
#
data['num_critic_for_reviews'].fillna(data['num_critic_for_reviews'].mode()[0], inplace=True) 
data['director_facebook_likes'].fillna(data['director_facebook_likes'].mode()[0], inplace=True) 
data['actor_3_facebook_likes'].fillna(data['actor_3_facebook_likes'].mode()[0], inplace=True) 
data['actor_1_facebook_likes'].fillna(data['actor_1_facebook_likes'].mode()[0], inplace=True) 
data['gross'].fillna(data['gross'].mode()[0], inplace=True) 
data['facenumber_in_poster'].fillna(data['facenumber_in_poster'].mode()[0], inplace=True) 
data['num_user_for_reviews'].fillna(data['num_user_for_reviews'].mode()[0], inplace=True) 
data['budget'].fillna(data['budget'].mode()[0], inplace=True) 
data['title_year'].fillna(data['title_year'].mode()[0], inplace=True) 
data['actor_2_facebook_likes'].fillna(data['actor_2_facebook_likes'].mode()[0], inplace=True) 
data['aspect_ratio'].fillna(data['aspect_ratio'].mode()[0], inplace=True) 
data['duration'].fillna(data['duration'].mode()[0], inplace=True) 


encoding_list = [ 'country','color','director_name','actor_2_name','content_rating','movie_title','actor_1_name','actor_3_name','language']
data[encoding_list] = data[encoding_list].apply(LabelEncoder().fit_transform)
scale_list = ['color', 'director_name', 'num_critic_for_reviews', 'duration',
       'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name',
       'actor_1_facebook_likes', 'gross', 'actor_1_name',
       'movie_title', 'num_voted_users', 'cast_total_facebook_likes',
       'actor_3_name', 'facenumber_in_poster', 
        'num_user_for_reviews', 'language', 'country',
       'content_rating', 'budget', 'title_year', 'actor_2_facebook_likes',
        'aspect_ratio', 'movie_facebook_likes']
sc = data[scale_list]
scaler = StandardScaler()
sc = scaler.fit_transform(sc)
data[scale_list] = sc
data[scale_list].head()
data.columns
X = data.drop(['genres','plot_keywords','imdb_score','movie_imdb_link'], axis=1)

Y = data['imdb_score']


X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.3)

logreg=LinearRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

y_pred
print(metrics.mean_squared_error(y_test, y_pred))
RMSE_LG = sqrt(0.9266430564468509)
print("rms of LR",RMSE_LG)
regressor = DecisionTreeRegressor( random_state = 0)
regressor.fit(X,Y)
y_pred1 = regressor.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred1 ))
DTR = sqrt(3.493303413734818e-32)
print(" rms value is ",DTR)
xgb = xgboost.XGBRegressor(n_estimators=25000, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)
                           
                           
xgb.fit(X_train,y_train)
predictions = xgb.predict(X_test)
print(metrics.mean_squared_error(y_test, predictions))
xgb = sqrt(0.5823674097245253)
print("rms value is",xgb)
rand = RandomForestRegressor(n_estimators = 100,random_state = 0)
rand.fit(X,Y)
y_pred2 = rand.predict(X_test)
print(metrics.mean_squared_error(y_test,y_pred2 ))
RFR = sqrt(0.08428359352280236)
print("rms value is",RFR)
print("rms value of RandomForestRegressor",RFR)
print("rms value xgboost",xgb)
print("rms value DecisionTreeRegressor ",DTR)
print("rms of LinearRegressor",RMSE_LG)
print('BEST MODEL')
print('RandomForestRegressor')
    
#RandomForestRegressor
plt.scatter(y_test,y_pred2)
#DECISION REGRESSOR
plt.scatter(y_test, predictions)
#xgbboost regressor
plt.scatter(y_test,y_pred1)
#LINEAR REGRESSION
plt.scatter(y_test,y_pred)


##LINEAR REGRESSION
sns.distplot((y_test-y_pred))
#xgbboost regressor
sns.distplot((y_test-y_pred1))
#decision tree regressor
sns.distplot((y_test-predictions))
sns.distplot((y_test-y_pred2))
bc = data.corr()
f,ax = plote.subplots(figsize = (18,18))
sns.heatmap(bc,cmap = 'coolwarm', linewidths= 1,linecolor = 'White',annot = True,fmt = '.1f',ax=ax)
X = data.drop(['genres','plot_keywords','imdb_score','movie_imdb_link','actor_1_facebook_likes'], axis=1)
Y = data['imdb_score']
X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.3)
logreg3=LinearRegression()
logreg3.fit(X_train,y_train)
y_pred3 = logreg3.predict(X_test)

y_pred3
print(metrics.mean_squared_error(y_test, y_pred))






