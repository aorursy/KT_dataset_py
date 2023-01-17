import pandas as pd

import numpy as np

import calendar

import matplotlib.pyplot as plt

import seaborn as sns

import folium

from folium.plugins import HeatMap

from IPython.display import IFrame



import warnings

warnings.filterwarnings('ignore')



pd.set_option('display.max_columns', 500)

%matplotlib inline
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head(10)
df.info()
df.isna().any().any()
df.isnull().any().any()
df['year_sale'] = df['date'].str[:4].astype(int)
df['month_sale_num'] = df['date'].str[4:6].astype(int)

df['month_sale_name'] = df['month_sale_num'].apply(lambda x: calendar.month_abbr[x])
df['year_month_day_sale'] = pd.to_datetime(df['date'].str[:4] +"-"+ df['date'].str[4:6] + "-" +  df['date'].str[6:8])
df['month_sale_name'].sample(10)
df['bathrooms'].unique()
df['bedrooms'].value_counts().sort_index()
df.loc[df['bedrooms'] >= 10]
df['bedrooms'] = df['bedrooms'].apply(lambda x: 8 if x>=8 else x)
df['floors'].value_counts().sort_index()
df['floors_int'] = df['floors'].round(0).astype('int')
df['waterfront'].value_counts()
df['view'].value_counts().sort_index()
df['condition'].value_counts().sort_index()
df['grade'].value_counts().sort_index()
def grades_to_categories(col):

    if col in [1,2,3,4,5,6]:

        return 1

    elif col in [7,8]:

        return 2

    elif col in [9,10,11]:

        return 3

    elif col in [12,13]:

        return 4
df['grade_category'] = df['grade'].apply(grades_to_categories)
df['grade_category'].value_counts().sort_index()
print("Year built:",df['yr_built'].value_counts().count(),"Year renovated:",df['yr_renovated'].value_counts().count())
plt.hist(df['yr_built'], bins=100)

plt.show()
plt.hist(df['yr_renovated'], bins=100)

plt.show()
df['built_after_ww2'] = df['yr_built'].map(lambda x: x>1945)

df['house_renovated'] = df['yr_renovated'].map(lambda x: x != 0)
df['years_since_construction'] = df['year_sale'] - df['yr_built']
plt.hist(df['years_since_construction'], bins=100)

plt.show()
print(df['zipcode'].unique())

print("Numer of unique districts:",df['zipcode'].unique().size)
fig, axes = plt.subplots(1, 2,figsize=(15,5))

sns.boxplot(y = df['price'], ax=axes[0])



sns.distplot(df['price'], ax=axes[1])

sns.despine(left=True, bottom=True)



axes[0].set(ylabel='Price')

axes[0].yaxis.tick_left()



axes[1].yaxis.set_label_position("left")

axes[1].yaxis.tick_left()

axes[1].set(xlabel='Price', ylabel='Distribution');



fig, axes = plt.subplots(1,2,figsize=(15,10))

sns.scatterplot(y = df['price'],x=df['sqft_living'], ax=axes[0])

sns.scatterplot(y = df['price'],x=df['sqft_lot'], ax = axes[1])

axes[0].set(xlabel = 'Square foot of living area',ylabel="Price")

axes[1].set(xlabel = 'Square foot of lot',ylabel="Price");
print(df.loc[df['price'] >=4000000].shape[0],df.loc[df['price'] >=3000000].shape[0],df.loc[df['price'] >=2000000].shape[0])
g = sns.FacetGrid(df, col = "zipcode", height=5,col_wrap=5)

g.map(plt.scatter, "price",'sqft_living', color = 'red');
g = sns.FacetGrid(df, col = "zipcode", height=5,col_wrap=5)

g.map(plt.scatter, "price",'sqft_lot', color = 'blue');
zipcode_list = [98004,98006,98007,98008,98033,98034,98039,98040,98056,98102,98103,98105,98106,98107,98108,98109,98112,98115,98116,

               98117,98118,98119,98122,98125,98126,98133,98136,98144,98146,98148,98155,98166,98168,98177,98178,98188,98198,98199]



df['urban_zipcode'] = df['zipcode'].map(lambda x: x in zipcode_list)
print("sqft_lot equal to zero:",df.loc[df['sqft_lot']==0].shape[0])
print("Min:",min(df['lat']), "Max:",max(df['lat']), "Difference:", max(df['lat']) - min(df['lat']))
print("Min:",min(df['long']), "Max:",max(df['long']), "Difference:",max(df['long']) - min(df['long']))
import folium

from folium.plugins import HeatMap





m = folium.Map(location=[df['lat'].mean(), df['long'].mean(),],

                        zoom_start=9.4,

                        tiles="CartoDB dark_matter")





HeatMap(data=df[['lat','long']].groupby(['lat','long']).sum().reset_index().values.tolist(),radius=11.5).add_to(m)





#m.save("map.html")



m
#IFrame(src='map.html', width=700, height=600)
fig, axe = plt.subplots(1, 1,figsize=(15,7))

reg = sns.regplot(y = df['price'],x = df['years_since_construction'], scatter_kws={"s": 0.3})

axes = reg.axes

axes.set_ylim(0,1500000)



axe.yaxis.set_label_position("left")

axe.yaxis.tick_left()

axe.set(xlabel='House age', ylabel='Price');

def plot_sqft_regplot(outlier_limit,features):

    

    df_copy = df.copy()

    df_copy = df_copy.loc[df_copy['price'] <= outlier_limit]

    

    fig, axes = plt.subplots(len(features), 1,figsize=(20,60))

    

    for i, feature in enumerate(features):

        

        reg = sns.regplot(x=df_copy[feature],y=df_copy['price'], ax=axes[i], fit_reg=True, scatter_kws={"s": 0.5})

        reg.tick_params(labelsize=15)

        ax = reg.axes

        ax.set_xlabel(feature, fontsize = 30)

        ax.set_ylabel('Price',fontsize= 30)

        ax.grid(True)

        



plot_sqft_regplot(3000000,['sqft_living','sqft_lot','sqft_above','sqft_basement','sqft_living15','sqft_lot15'])
print("sqft_basement equal to zero:",df.loc[df['sqft_basement']==0].shape[0])

print("sqft_lot15 equal to zero:",df.loc[df['sqft_lot15']==0].shape[0])
df['no_basement'] = df['sqft_basement'].map(lambda x: int(x==0))
def plot_categorical_features(outlier_limit):

    

    df_copy = df.copy()

    df_copy = df_copy.loc[df_copy['price'] <= outlier_limit]

    fig, axes = plt.subplots(5, 2,figsize=(17,45))



    sns.boxplot(x=df_copy['grade'],y=df_copy['price'], ax=axes[0][0])

    axes[0][0].set(xlabel='Grade', ylabel='Price')

    axes[0][0].yaxis.tick_left()

    axes[0][0].grid(True)



    sns.boxplot(x=df_copy['grade_category'],y=df_copy['price'], ax=axes[0][1])

    axes[0][1].yaxis.set_label_position("right")

    axes[0][1].yaxis.tick_right()

    axes[0][1].set(xlabel='Grade categorized', ylabel='Price')

    axes[0][1].grid(True)





    sns.boxplot(x=df_copy['view'],y=df_copy['price'], ax=axes[1][0])

    axes[1][0].yaxis.tick_right()

    axes[1][0].set(xlabel='View', ylabel='Price')

    axes[1][0].grid(True)



    sns.boxplot(x=df_copy['waterfront'],y=df_copy['price'], ax=axes[1][1])

    axes[1][1].yaxis.set_label_position("right")

    axes[1][1].yaxis.tick_right()

    axes[1][1].set(xlabel='Waterfront', ylabel='Price')

    axes[1][1].grid(True)





    sns.boxplot(x=df_copy['built_after_ww2'],y=df_copy['price'], ax=axes[2][0])

    axes[2][0].yaxis.tick_right()

    axes[2][0].set(xlabel='Built after WW2?', ylabel='Price')

    axes[2][0].grid(True)



    sns.boxplot(x=df_copy['house_renovated'],y=df_copy['price'], ax=axes[2][1])

    axes[2][1].yaxis.set_label_position("right")

    axes[2][1].yaxis.tick_right()

    axes[2][1].set(xlabel='House renovated?', ylabel='Price')

    axes[2][1].grid(True)



    sns.boxplot(x=df_copy['condition'],y=df_copy['price'], ax=axes[3][0])

    axes[3][0].yaxis.tick_right()

    axes[3][0].set(xlabel='Condition', ylabel='Price')

    axes[3][0].grid(True)



    sns.boxplot(x=df_copy['urban_zipcode'],y=df_copy['price'], ax=axes[3][1])

    axes[3][1].yaxis.set_label_position("right")

    axes[3][1].yaxis.tick_right()

    axes[3][1].set(xlabel='Has urban zipcode?', ylabel='Price')

    axes[3][1].grid(True)

    

    sns.boxplot(x=df_copy['month_sale_name'],y=df_copy['price'], ax=axes[4][0])

    axes[4][0].yaxis.tick_right()

    axes[4][0].set(xlabel='Month of sale', ylabel='Price')

    axes[4][0].grid(True)



    sns.boxplot(x=df_copy['no_basement'],y=df_copy['price'], ax=axes[4][1])

    axes[4][1].yaxis.set_label_position("right")

    axes[4][1].yaxis.tick_right()

    axes[4][1].set(xlabel='Has basement?', ylabel='Price')

    axes[4][1].grid(True)



    fig, axes = plt.subplots(3, 1,figsize=(17,25))



    sns.boxplot(x=df_copy['bathrooms'],y=df_copy['price'], ax=axes[0])

    axes[0].yaxis.tick_left()

    axes[0].set(xlabel='Bathrooms', ylabel='Price')

    axes[0].grid(True)



    sns.boxplot(x=df_copy['bedrooms'],y=df_copy['price'], ax=axes[1])

    axes[1].yaxis.tick_left()

    axes[1].set(xlabel='Bedrooms', ylabel='Price')

    axes[1].grid(True)

    

    sns.boxplot(x=df_copy['floors'],y=df_copy['price'], ax=axes[2])

    axes[2].yaxis.tick_left()

    axes[2].set(xlabel='Floors', ylabel='Price')

    axes[2].grid(True);



plot_categorical_features(2000000)
def divide_bathrooms_bedrooms(bathrooms,bedrooms):

    if bedrooms != 0:

        return bathrooms/bedrooms

    else:

        return 0
df['bathrooms/bedrooms'] = df.apply(lambda x: divide_bathrooms_bedrooms(x.bathrooms,x.bedrooms),axis=1)

df.loc[df['bedrooms'] == 0].head(1)
df['bathrooms*bedrooms'] = df['bathrooms']*df['bedrooms']
df['waterfront+view'] = df['waterfront'] + df['view']
df['over_one_floor'] = df['floors'].map(lambda x: int(x>1.))

df['over_two_floors'] = df['floors'].map(lambda x: int(x>2.))
df['view_over_zero'] = df['view'].map(lambda x: int(x>0))
df.columns
df_correlation = df[['price','bedrooms','bathrooms', 'over_one_floor','over_two_floors','view_over_zero',

                     'waterfront','view','condition', 'grade','house_renovated',

                    'grade_category', 'built_after_ww2','urban_zipcode','no_basement','waterfront+view']].copy()

plt.rcParams['figure.figsize']=(15,10)

sns.heatmap(df_correlation.corr(method='spearman'), vmax=1., vmin=-1., annot=True, linewidths=.8, cmap="YlGnBu");
df_correlation = df[['price','sqft_living','sqft_lot','sqft_above', 'sqft_basement','sqft_living15', 'sqft_lot15',

                    'year_sale','month_sale_num','years_since_construction','bathrooms/bedrooms',

                     'bathrooms*bedrooms', 'yr_built', 'yr_renovated','floors']].copy()

plt.rcParams['figure.figsize']=(15,10)

sns.heatmap(df_correlation.corr(), vmax=1., vmin=-1., annot=True, linewidths=.8, cmap="YlGnBu");
df_correlation = df[['price','sqft_living','sqft_lot','sqft_above', 'sqft_basement','sqft_living15', 'sqft_lot15',

                    'year_sale','month_sale_num','years_since_construction','bathrooms/bedrooms',

                     'bathrooms*bedrooms', 'yr_built', 'yr_renovated','floors',

                    'bedrooms','bathrooms', 'over_one_floor','over_two_floors','view_over_zero',

                     'waterfront','view','condition', 'grade','house_renovated',

                    'grade_category', 'built_after_ww2','urban_zipcode','no_basement','waterfront+view']].copy()

plt.rcParams['figure.figsize']=(15,10)

sns.heatmap(df_correlation.corr(), vmax=1., vmin=-1., annot=True, linewidths=.8, cmap="YlGnBu");
from sklearn.linear_model import LinearRegression, Ridge

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_curve, roc_auc_score

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

import xgboost as xgb



from functools import partial

from hyperopt import hp

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import random

from math import sqrt



random.seed(100)
y = df['price'].values
def train_validate(model, metric, X, y):

    skf = KFold(n_splits = 8, shuffle= True)

    

    scores_metric = []

    for train_idx, test_idx in skf.split(X,y):

        model.fit(X[train_idx],y[train_idx])

        y_pred = model.predict(X[test_idx])

        

        score = metric(y[test_idx],y_pred)

        

        scores_metric.append(score)



        

    result = np.mean(scores_metric)



    return result
def best_features(dataframe, model,metric, features, repeats = 20, min_features = 1, max_features = 18):



    best_score = 100000000000000000

    best_feats = []

    np.random.seed(2000)

    

    y = dataframe['price'].values

    

    if max_features > len(features):

        max_features = len(features)

        



    for i in range(min_features,max_features): 

        for a in range(repeats): # repeat n times for this number of features

            feats = np.random.choice(features,i,replace = False).tolist()

            X = dataframe[feats].values

            score = train_validate(model,metric,X,y)

            if score < best_score:

                best_score = score

                best_feats = feats

        print("Best score for {0} features is {1}".format(len(feats),best_score))

        print(feats)





    print('\n\nBest score is {0} with features: {1}'.format(best_score,best_feats))
df_copy = df.copy()

df_copy = df_copy.drop(columns=['id','date','month_sale_name','year_month_day_sale','zipcode'])

names = df_copy.columns



scaler = StandardScaler()

standarized_df = pd.DataFrame(scaler.fit_transform(df_copy), columns = names)

standarized_df.head()
standarized_df.columns
features = standarized_df.columns.to_list()

features.remove('price')
#function below is hashed because it takes several minutes to get the result.

#best_features(standarized_df, LinearRegression(),mean_squared_error, features, max_features=25, repeats = 35)
feats = ['bathrooms', 'sqft_above', 'over_one_floor', 'waterfront+view', 'bathrooms*bedrooms', 'lat', 

         'month_sale_num', 'year_sale', 'sqft_living15', 'years_since_construction', 'sqft_basement', 

         'condition', 'sqft_living', 'no_basement', 'urban_zipcode', 'grade', 'house_renovated', 'bedrooms', 

         'over_two_floors', 'waterfront', 'yr_built', 'long', 'floors']



y = standarized_df['price'].values

X = standarized_df[feats].values



lin_reg = LinearRegression()

lin_reg.fit(X,y)

lin_reg.score(X,y)
features = df.columns.to_list()



remove_list = ['price','id','month_sale_name','year_month_day_sale','date','zipcode']



for elem in remove_list:

    features.remove(elem)
#function below is hashed because it takes several minutes to get the result.

#best_features(df, LinearRegression(),mean_squared_error, features, max_features=25, repeats = 35)
feats = ['bathrooms', 'sqft_above', 'over_one_floor', 'waterfront+view', 'bathrooms*bedrooms', 'lat', 

         'month_sale_num', 'year_sale', 'sqft_living15', 'years_since_construction', 'sqft_basement', 

         'condition', 'sqft_living', 'no_basement', 'urban_zipcode', 'grade', 'house_renovated', 'bedrooms', 

         'over_two_floors', 'waterfront', 'yr_built', 'long', 'floors']



y = df['price'].values

X = df[feats].values



lin_reg = LinearRegression()

lin_reg.fit(X,y)

lin_reg.score(X,y)
rmse = round(sqrt(38082141181.18445),None)

rmse
ridge = Ridge()

parameters = {'alpha':[0,1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,2,3,5,10,15,20]}
ridge_regressor = GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error', cv=10)
ridge_regressor.fit(X,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
rmse = round(sqrt(38214893294.43146),None)

rmse
feats = ['lat','long','sqft_living']

X = df[feats].values
for i in range(1,16):



    KNR = KNeighborsRegressor(n_neighbors=i)

    score = train_validate(KNR,mean_squared_error,X,y)

    rmse = sqrt(score)

    print('Neighbors: {0}, MSE: {1}'.format(i,rmse))
features = df.columns.to_list()



remove_list = ['price','id','month_sale_name','year_month_day_sale','date','zipcode']



for elem in remove_list:

    features.remove(elem)
#function below is hashed because it takes several minutes to get the result.

#best_features(df, KNeighborsRegressor(n_neighbors=9),mean_squared_error, features, max_features=25, repeats = 35)
features = ['lat', 'grade', 'view_over_zero','long']



X = df[features].values

y = df['price']



KNR = KNeighborsRegressor(n_neighbors=9)

score = train_validate(KNR,mean_squared_error,X,y)

rmse = round(sqrt(score),None)

print('Neighbors: {0}, RMSE: {1}'.format(9,rmse))
feats = ['lat','long','sqft_living']

X = df[feats].values
# XGB

print(sqrt(train_validate(xgb.XGBRegressor(),mean_squared_error,X,y)))

# Random Forest Regressor

print(sqrt(train_validate(RandomForestRegressor(),mean_squared_error,X,y)))
feats = ['sqft_living','sqft_lot','sqft_living15', 'sqft_lot15','years_since_construction',

                     'bathrooms*bedrooms', 'yr_built', 'yr_renovated','floors_int',

                     'over_one_floor','over_two_floors', 'sqft_basement',

                     'waterfront','view','condition', 'grade','house_renovated',

                     'built_after_ww2','urban_zipcode','no_basement', 'lat','long']

X = df[feats].values

print(sqrt(train_validate(xgb.XGBRegressor(),mean_squared_error,X,y)))
#function below is hashed because it takes several minutes to get the result.

# def objective(space):

#     params = {

#         'eta':space['eta'],

#         'max_depth':int(space['max_depth']),

#         'min_child_weight':int(space['min_child_weight']),

        

#     }

    

#     model = xgb.XGBRegressor(**params)

    

#     score = sqrt(train_validate(model,mean_squared_error,X,y))

#     print('Score: {0}'.format(score))

#     return {'loss':score,'status':STATUS_OK}





# space = {

#     'eta':hp.uniform('eta',0.1,1),

#     'max_depth':hp.quniform('max_depth',1,70,1),

#     'min_child_weight':hp.quniform('min_child_weight',0,150,1)

# }





# trials = Trials()

# best_params = fmin(fn = objective,

#                   space = space,

#                   algo=partial(tpe.suggest, n_startup_jobs = 10),

#                   max_evals = 20,

#                   trials = trials)



# print('Best params: ', best_params)
feats = ['sqft_living','sqft_lot','sqft_living15', 'sqft_lot15','years_since_construction',

                     'bathrooms*bedrooms', 'yr_built', 'yr_renovated','floors_int',

                     'over_one_floor','over_two_floors', 'sqft_basement',

                     'waterfront','view','condition', 'grade','house_renovated',

                     'built_after_ww2','urban_zipcode','no_basement', 'lat','long']

X = df[feats].values

print(sqrt(train_validate(RandomForestRegressor(),mean_squared_error,X,y)))
#function below is hashed because it takes several minutes to get the result.

# def objective(space):

#     params = {

#         'max_depth':int(space['max_depth']),

#         'min_samples_split':int(space['min_samples_split']),

#         'max_features':int(space['max_features'])

#     }

    

#     model = RandomForestRegressor(**params)

    

#     score = sqrt(train_validate(model,mean_squared_error,X,y))

#     print('Score: {0}'.format(score))

#     return {'loss':score,'status':STATUS_OK}





# space = {

#     'max_depth':hp.quniform('max_depth',1,35,1),

#     'min_samples_split':hp.quniform('min_samples_split',2,100,1),

#     'max_features':hp.quniform('max_features',1,10,1)

# }





# trials = Trials()

# best_params = fmin(fn = objective,

#                   space = space,

#                   algo=partial(tpe.suggest, n_startup_jobs = 50),

#                   max_evals = 150,

#                   trials = trials)



# print('Best params: ', best_params)

def draw_feature_importances(model, features):

    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))

    plt.title("Feature importances")

    plt.bar(range(X.shape[1]), model.feature_importances_[indices],

           color="b", align="center")

    plt.xticks(range(X.shape[1]), [ features[x] for x in indices] )

    #plt.xticks(range(X.shape[1]), model.feature_importances_[indices])

    plt.xticks(rotation=90)

    plt.xlim([-1, X.shape[1]])

    plt.show()
feats = ['sqft_living','sqft_lot','sqft_living15', 'sqft_lot15','years_since_construction',

                     'bathrooms*bedrooms', 'yr_built', 'yr_renovated','floors_int',

                     'over_one_floor','over_two_floors', 'sqft_basement',

                     'waterfront','view','condition', 'grade','house_renovated',

                     'built_after_ww2','urban_zipcode','no_basement', 'lat','long']

X = df[feats].values
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size = 0.3, random_state = 2)
model_XGBoostRegressor = xgb.XGBRegressor(eta = 0.4392, max_depth = 6, min_child_weight = 30)

model_XGBoostRegressor.fit(train_X,train_y)
y_pred = model_XGBoostRegressor.predict(test_X)

print("RMSE error: {0}".format(sqrt(mean_squared_error(test_y,y_pred))))

print("MAE error: {0}".format(mean_absolute_error(test_y,y_pred)))
draw_feature_importances(model_XGBoostRegressor,feats)
model_RandomForestRegressor = RandomForestRegressor(max_depth=30, max_features='auto',min_samples_split=2)

model_RandomForestRegressor.fit(train_X,train_y)
y_pred = model_RandomForestRegressor.predict(test_X)

print("RMSE error: {0}".format(sqrt(mean_squared_error(test_y,y_pred))))

print("MAE error: {0}".format(mean_absolute_error(test_y,y_pred)))
draw_feature_importances(model_RandomForestRegressor,feats)
fig, axe = plt.subplots(1, 1,figsize=(15,7))

scatter = sns.scatterplot(x = y_pred,y = test_y)

axes = scatter.axes

plt.title('Random Forest Price actual vs predicted')

plt.grid(True)

axe.yaxis.set_label_position("left")

axe.yaxis.tick_left()

axe.set(xlabel= 'Price predicted', ylabel='Price actual');
