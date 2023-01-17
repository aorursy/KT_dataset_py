import pandas as pd # for dataframes

import numpy as np # for arrays & math functions



%matplotlib inline

import matplotlib.pyplot as plt # for plotting



import warnings

warnings.filterwarnings('ignore') # ignoring any warnings
PATH = '../input/tmdb-movie-metadata/tmdb_5000_movies.csv'



movies_df = pd.read_csv(PATH) # load data into a pandas dataframe
SEED = 2020 # for reproducability



movies_df.sample(3,random_state=SEED)
movies_df.info()
movies_df['popularity'].describe()
movies_df.sort_values(by='popularity',ascending=False)[:5]
movies_df.sort_values(by='popularity',ascending=True)[:5]
movies_df.sort_values(by='vote_average', ascending = False)[:5]
movies_df['original_language'].value_counts()
### Pie Chart



labels = np.array(['English','Other'])

sizes = np.array([4505, sum(movies_df['original_language'].value_counts()) - 4505])



plt.figure(figsize=(8,9))



plt.pie(sizes, labels=labels, autopct='%1.1f%%', explode=[0,0.08], startangle=90)

plt.title('Original Languages of Movies', fontdict={'fontsize': 14})

plt.axis('equal')
features_df = movies_df[['budget','genres','original_language','runtime','vote_average','vote_count']]

labels_df = movies_df['popularity']
features_df = features_df.dropna()
features_df = features_df[features_df['vote_count'] >= 10]
features_df = features_df[features_df['runtime'] != 0.0]
labels_df = labels_df[features_df.index]
features_df.describe()
from sklearn.model_selection import train_test_split



# split data into 80% training and 20% testing

x_train, x_test, y_train, y_test = train_test_split(features_df, labels_df, test_size=0.2, random_state = SEED)
# Plot histogram

plt.figure(figsize=(10,8))

plt.hist(y_train.values,bins=50)

plt.title('Movies popularity histogram')

plt.xlabel('Popularity')

plt.ylabel('# of movies')

plt.show()
y_train.skew()
y_train = np.log(y_train)

y_test = np.log(y_test)
# Plot histogram

plt.figure(figsize=(10,8))

plt.hist(y_train.values,bins=50)

plt.title('Movies popularity histogram')

plt.xlabel('Popularity')

plt.ylabel('# of movies')

plt.show()
print('Popularity skew:', y_train.skew())
### Plot histograms

plt.rcParams['figure.figsize'] = 12, 12

fig, axs = plt.subplots(2,2)

fig.suptitle('Numerical Feature Histograms',y=0.95,fontsize=16)



axs[0,1].hist(x_train['budget'].values,bins=30,color='salmon')

axs[0,1].set_title('Budget')

axs[0,1].set(xlabel='US dollars')

axs[0,0].hist(x_train['runtime'].values,bins=30,color='salmon')

axs[0,0].set_title('Runtime (min)')

axs[0,0].set(xlabel='Minutes')

axs[1,0].hist(x_train['vote_average'].values,bins=30,color='salmon')

axs[1,0].set_title('Vote Average')

axs[1,1].hist(x_train['vote_count'].values,bins=30,color='salmon')

axs[1,1].set_title('Vote Count')

plt.show()
print('Vote count skew:', x_train['vote_count'].skew())

print('Vote average skew:', x_train['vote_average'].skew())

print('Runtime skew:', x_train['runtime'].skew())

print('Budget skew:', x_train['budget'].skew())
x_train['vote_count'] = np.log(x_train['vote_count'].values)

x_train['budget'] = np.sqrt(x_train['budget'].values)



x_test['vote_count'] = np.log(x_test['vote_count'].values)

x_test['budget'] = np.sqrt(x_test['budget'].values)
### Histograms



plt.rcParams['figure.figsize'] = 12, 12

fig, axs = plt.subplots(2,2)

fig.suptitle('Numerical Feature Histograms',y=0.95,fontsize=16)



axs[0,1].hist(x_train['budget'].values,bins=30,color='salmon')

axs[0,1].set_title('Budget')

axs[0,1].set(xlabel='US dollars')

axs[0,0].hist(x_train['runtime'].values,bins=30,color='salmon')

axs[0,0].set_title('Runtime (min)')

axs[0,0].set(xlabel='Minutes')

axs[1,0].hist(x_train['vote_average'].values,bins=30,color='salmon')

axs[1,0].set_title('Vote Average')

axs[1,1].hist(x_train['vote_count'].values,bins=30,color='salmon')

axs[1,1].set_title('Vote Count')

plt.show()
print(x_train['vote_count'].skew())

print(x_train['vote_average'].skew())

print(x_train['runtime'].skew())

print(x_train['budget'].skew())
# Scatterplots



plt.figure(figsize=(12,12))



fig, axs = plt.subplots(2,2)

fig.suptitle('Correlation to target',y=0.95,fontsize=16)

axs[0,1].scatter(x_train['vote_average'].values, y_train.values,color='green')

axs[0,1].set(xlabel='Vote Average',ylabel='Populariy')

axs[0,0].scatter(x_train['budget'].values, y_train.values,color='green')

axs[0,0].set(xlabel='Budget (US dollars)',ylabel='Populariy')

axs[1,0].scatter(x_train['vote_count'].values, y_train.values,color='green')

axs[1,0].set(xlabel='Vote Count',ylabel='Populariy')

axs[1,1].scatter(x_train['runtime'].values, y_train.values,color='green')

axs[1,1].set(xlabel='Runtime (min)',ylabel='Populariy')

plt.show()
corr_matrix = pd.concat([x_train,y_train],axis=1).corr()



corr_matrix['popularity'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



scatter_matrix(x_train[['budget','runtime','vote_count','vote_average']], figsize=(12,12))
x_train.corr()
### Encode language 



x_train['Language'] = x_train['original_language'].apply(lambda x: 1 if 'en' == x else 0)

x_test['Language'] = x_test['original_language'].apply(lambda x: 1 if 'en' == x else 0)
genre_list = ['Action', 'Adventure', 'Fantasy', 'Science Fiction', 'Crime', 'Drama', 'Thriller', 'Animation',

 'Family', 'Western', 'Comedy', 'Romance', 'Horror', 'Mystery', 'History', 'War', 'Music']
# make column for each genre and encode it



for genre in genre_list:

  x_train[genre] = x_train['genres'].apply(lambda x: 1 if genre in x else 0)

  x_test[genre] = x_test['genres'].apply(lambda x: 1 if genre in x else 0)
x_train.describe()
corr_matrix = pd.concat([x_train,y_train],axis=1).corr()



corr_matrix['popularity'].sort_values(ascending=False)
x_train = x_train.drop(columns=['original_language','genres']) # drop encoded columns

x_test = x_test.drop(columns=['original_language','genres']) # drop encoded columns
from sklearn.preprocessing import StandardScaler, MinMaxScaler



scaler = MinMaxScaler()



x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train),index=x_train.index, columns=x_train.columns)

x_test_scaled = pd.DataFrame(scaler.transform(x_test),index=x_test.index, columns=x_test.columns)
# Verify



x_train_scaled.head()
from sklearn.model_selection import cross_val_score



rmse_list = []

std_list = []



def get_score(model):

  cv_score = cross_val_score(model, x_train_scaled, y_train, scoring = "neg_mean_squared_error", cv = 8)

  rmse = np.sqrt(-cv_score)

  print('Cross-Validation Root Mean Squared Error:', rmse)

  print('Average Root Mean Squared Error:', round(np.mean(rmse), 5))

  rmse_list.append(round(np.mean(rmse), 5))

  print('Standard deviation:', round(rmse.std(), 5))

  std_list.append(round(rmse.std(), 5))
from sklearn.linear_model import LinearRegression, Ridge, Lasso
### Linear Regression



model_1 = LinearRegression()



model_1.fit(x_train_scaled,y_train)



get_score(model_1)
### Ridge Regression



model_2 = Ridge(random_state=SEED)



model_2.fit(x_train_scaled,y_train)



get_score(model_2)
### Lasso Regression



model_3 = Lasso(random_state=SEED)



model_3.fit(x_train_scaled,y_train)



get_score(model_3)
from sklearn.ensemble import RandomForestRegressor
model_4 = RandomForestRegressor(random_state=SEED)



model_4.fit(x_train_scaled,y_train)



get_score(model_4)
from sklearn.svm import SVR
model_5 = SVR()



model_5.fit(x_train_scaled,y_train)



get_score(model_5)
from xgboost import XGBRegressor
model_6 = XGBRegressor(random_state=SEED,verbose=0,objective='reg:squarederror')



model_6.fit(x_train_scaled,y_train)



get_score(model_6)
import tensorflow as tf

import tensorflow.keras.layers as L

from keras.wrappers.scikit_learn import KerasRegressor
def get_tf_model():

    model = tf.keras.Sequential([

        L.Input(shape=(x_train_scaled.shape[1])),

        L.Dense(250, activation='relu'),

        L.BatchNormalization(),

        L.Dense(200, activation='relu'),

        L.BatchNormalization(),

        L.Dense(200, activation='relu'),

        L.BatchNormalization(),

        L.Dense(1)

    ])



    model.compile(

        optimizer='adam',

        loss = 'mse',

        metrics=['accuracy','mse']

    )

    

    return model
get_tf_model().summary()
model_7 = KerasRegressor(build_fn = get_tf_model, epochs = 10, verbose = 0, batch_size = 100)

model_7.fit(x_train_scaled,y_train.values)
get_score(model_7)
# For creating tables that render in Github

!pip install --upgrade plotly

!pip install -U kaleido
import plotly.graph_objects as go



# Create table



models_list = ['Linear Regression','Ridge Regression','Lasso Regression','Random Forest', 

               'Support Vector Regressor','XGBoost', 'Neural Network']



fig = go.Figure(data=[go.Table(header=dict(values=['Model', 'RMSE', 'Standard Deviation']),

                 cells=dict(values=[models_list, rmse_list, std_list]))

                     ])



fig.update_layout(

    title={

        'text': "Starting Model Cross Validation Scores",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show("png")
from sklearn.model_selection import GridSearchCV



# Enter model and parameter options and returns best model

def grid_search(model,params):

  search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')

  search.fit(x_train_scaled,y_train)

  return search.best_estimator_
model_3.get_params()
param_grid = [

              {'alpha': [0.1,0.05,0.01,0.005] , 

               "fit_intercept": [True, False], 

               'normalize': [True, False],

               "tol": [0.0005,.0001,0.00005]}

]



model_3_grid = grid_search(model_3,param_grid)



model_3_grid.get_params() # these will be our new parameters
get_score(model_3_grid)
# Support Vector Regression



model_5.get_params()
param_grid = [

              {'kernel': ['linear', 'rbf'],

               'tol': [0.015, 0.01],

               'epsilon': [0.2, 0.15] }

]



model_5_grid = grid_search(model_5,param_grid)



model_5_grid.get_params()
get_score(model_5_grid)
model_6.get_params()
param_grid = [

              {'gamma': [10,5],

               'max_depth': [7,5],

               'min_child_weight': [30,20],

               'learning_rate': [0.05,0.01]}

]



model_6_grid = grid_search(model_6,param_grid)



model_6_grid.get_params()
get_score(model_6_grid)
# Create table



models_list = ['Lasso Regression','Support Vector Regressor','XGBoost']



fig = go.Figure(data=[go.Table(header=dict(values=['Model', 'Original RMSE', 'Grid Search RMSE', 

                                                   'Original Standard Deviation', 'Grid Search Standard Deviation']),

                 cells=dict(values=[models_list, [rmse_list[2], rmse_list[4], rmse_list[5]], rmse_list[-3:],

                                    [std_list[2], std_list[4], std_list[5]], std_list[-3:]]))

                     ])



fig.update_layout(

    title={

        'text': "Grid Searched Model Comparisons",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})





fig.show("png")
from sklearn.metrics import mean_squared_error



predictions = []

final_scores = []



def get_results(preds):

  score = np.sqrt(mean_squared_error(preds,y_test.values))

  final_scores.append(round(score,5))
### Regression



preds = np.array(model_1.predict(x_test_scaled))

predictions.append(preds)

get_results(preds)





### Ridge



preds = np.array(model_2.predict(x_test_scaled))

predictions.append(preds)

get_results(preds)





### Lasso



preds = np.array(model_3.predict(x_test_scaled))

predictions.append(preds)

get_results(preds)





### Forest



preds = np.array(model_4.predict(x_test_scaled))

predictions.append(preds)

get_results(preds)





### SVR



preds = np.array(model_5.predict(x_test_scaled))

predictions.append(preds)

get_results(preds)





### XGBoost



preds = np.array(model_6.predict(x_test_scaled))

predictions.append(preds)

get_results(preds)





### Neural Network



preds = model_7.predict(x_test_scaled).reshape(len(x_test_scaled))

predictions.append(preds)

get_results(preds)





### Grid searched Lasso



preds = np.array(model_3_grid.predict(x_test_scaled))

predictions.append(preds)

get_results(preds)





### Grid searched SVR



preds = np.array(model_5_grid.predict(x_test_scaled))

predictions.append(preds)

get_results(preds)





### Grid searched XGBoost



preds = np.array(model_6_grid.predict(x_test_scaled))

predictions.append(preds)

get_results(preds)
# average all model predictions

ensemble_1 = np.mean(predictions,axis=0) 

get_results(ensemble_1)





# average last three model predictions

ensemble_2 = np.mean(predictions[-3:],axis=0) 

get_results(ensemble_2)





# average top three model predictions

ensemble_3 = np.mean([predictions[0], predictions[1],predictions[8]],axis=0) 

get_results(ensemble_3)
# Create table



models_list = ['Linear Regression','Ridge Regression','Lasso Regression','Random Forest','Support Vector Regressor',

               'XGBoost','Neural Network','Grid Search Lasso','Grid Search Support Vector Regressor',

               'Grid Search XGBoost','All Model Ensemble','Grid Search Model Ensemble', 'Top 3 Ensemble']



models_ranked_df = pd.DataFrame(data={'model': models_list, 'score': final_scores}).sort_values(by='score')



fig = go.Figure(data=[go.Table(header=dict(values=['Model', 'Final RMSE']),

                 cells=dict(values=[models_ranked_df.model, models_ranked_df.score ]))

                     ])



fig.update_layout(

    title={

        'text': "All Models Ranked",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show("png")
from yellowbrick.regressor import ResidualsPlot



visualizer = ResidualsPlot(model_6, is_fitted=True, train_color='b', test_color='g', size=(1080,720))

visualizer.fit(x_train_scaled, y_train)

visualizer.score(x_test_scaled,y_test)

visualizer.poof() 
# Create table



features_ranked_df = pd.DataFrame(data={'feature': x_test_scaled.columns, 

                                        'importance': model_6.feature_importances_}

                                  ).sort_values(by='importance', ascending = False)



fig = go.Figure(data=[go.Table(header=dict(values=['Feature', 'Importance']),

                 cells=dict(values=[features_ranked_df.feature, [round(x,5) for x in features_ranked_df.importance]]))

                     ])





fig.update_layout(

    title={

        'text': "Features Ranked",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show("png")
def score(model):

  return round(model.score(x_test_scaled, y_test),5)



# Create table



model_r2_list = [score(model_1), score(model_2), score(model_3), score(model_4), score(model_5),

                 score(model_6), score(model_7), score(model_3_grid), score(model_5_grid), score(model_6_grid)]



r2_ranked_df = pd.DataFrame(data={'model': models_list[:10], 'r2':model_r2_list}

                                  ).sort_values(by='r2', ascending = False)



fig = go.Figure(data=[go.Table(header=dict(values=['Model', 'R^2']),

                 cells=dict(values=[r2_ranked_df.model, r2_ranked_df.r2 ]))

                     ])



fig.update_layout(

    title={

        'text': "R^2 Values",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.show("png")