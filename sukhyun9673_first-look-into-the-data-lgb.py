# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
print ("Train set shape: ", df_train.shape)

print ("Test set shape: ", df_test.shape)
df_train.head()
df_test.head()
df_train.isnull().sum()
df_test.isnull().sum()
cor_mat = df_train.corr()

cor_mat
cor_mat["price"].sort_values(ascending = False)
fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(cor_mat, ax = ax, cmap="jet")

plt.show()
df_train["price"].hist(bins = 100, figsize = (10, 10))

np.log(df_train["price"]).hist(bins = 100, figsize = (10, 10))

df_train["price"] = np.log(df_train["price"])
df_train.plot(kind = "scatter", x = "long", y = "lat", alpha = 0.1, s = df_train["sqft_living"]*0.02, 

             label = "sqft_living", figsize = (10, 10), c = "price", cmap = plt.get_cmap("jet"), colorbar = True, sharex = False)

df_train.drop("id", axis = 1).hist(bins = 40, figsize = (20, 20))

plt.show()
f = ["sqft_living", "sqft_above", "sqft_living15", "bathrooms", "bedrooms", "grade", "price"]



feature = df_train[f]



#categorical

plt.figure(figsize = (10, 10))

sns.boxplot(x="grade", y="price", data=feature)

plt.show()
def grade_outliers(grade, quantile = 0.75):

    gd = feature.loc[feature["grade"] == grade]

    quant = gd["price"].quantile(quantile)

    print ("{} Grade Outliers, {} Quantile".format(grade, quantile))

    

    return df_train.loc[(df_train["price"]>quant) & (df_train["grade"] == grade)]
grade_outliers(6, 0.75)["price"].hist(bins =100)

grade_outliers(6, 0.9)["price"].hist(bins =100)

plt.show()
ids = []

df_train.loc[(df_train["price"]>13.8) & (df_train["grade"] == 6)]
for i in df_train.loc[(df_train["price"]>13.65) & (df_train["grade"] == 6)]["id"].values:

    ids.append(i)
grade_outliers(7, 0.75)["price"].hist(bins =100)

grade_outliers(7, 0.9)["price"].hist(bins =100)

plt.show()
df_train.loc[(df_train["price"]>14) & (df_train["grade"] == 7)]
for i in df_train.loc[(df_train["price"]>14) & (df_train["grade"] == 7)]["id"].values:

    ids.append(i)
grade_outliers(8, 0.75)["price"].hist(bins =100)

grade_outliers(8, 0.9)["price"].hist(bins =100)

plt.show()
df_train.loc[(df_train["price"]>14.5) & (df_train["grade"] == 8)]
for i in df_train.loc[(df_train["price"]>14.5) & (df_train["grade"] == 8)]["id"].values:

    ids.append(i)
grade_outliers(9, 0.75)["price"].hist(bins =100)

grade_outliers(9, 0.9)["price"].hist(bins =100)

plt.show()
grade_outliers(10, 0.75)["price"].hist(bins =100)

grade_outliers(10, 0.9)["price"].hist(bins =100)

plt.show()
grade_outliers(11, 0.75)["price"].hist(bins =100)

grade_outliers(11, 0.9)["price"].hist(bins =100)

plt.show()
df_train.loc[(df_train["price"]>15.6) & (df_train["grade"] == 11)]
for i in df_train.loc[(df_train["price"]>15.6) & (df_train["grade"] == 11)]["id"].values:

    ids.append(i)
df_train = df_train.loc[~df_train['id'].isin(ids)]
from pandas.plotting import scatter_matrix

scatter_matrix(feature.drop("grade", axis =1), figsize = (20, 20))

plt.show()
df_train.loc[df_train["sqft_living"]>10000]
df_train = df_train.loc[~df_train["id"].isin(df_train.loc[df_train["sqft_living"]>10000]["id"].values)]
df_train.head()
label = df_train["price"]

df_train.drop("price", axis = 1, inplace = True)



traindex = len(df_train)

df = pd.concat([df_train, df_test])

print(df.shape)
df["sqft_living"] = np.log(df["sqft_living"])

df["sqft_living15"] = np.log(df["sqft_living15"])

df["sqft_above"] = np.log(df["sqft_above"])
for i in ["sqft_living", "sqft_above", "sqft_living15"]:

    df[i].hist(bins = 100)

    plt.show()
df.loc[df["sqft_basement"]!=0]["sqft_basement"].hist(bins = 100)
df["sqft_basement"] = df["sqft_basement"].apply(lambda x : np.log(x) if x!=0 else 0)
np.log(df["sqft_lot"]).hist(bins = 100)
df["sqft_lot"] = np.log(df["sqft_lot"])

df["sqft_lot15"] = np.log(df["sqft_lot15"])
df.head()
print((sum(df["sqft_living"]==df["sqft_living15"])/len(df))*100, "Percent of sqft_living stays the same")

print((sum(df["sqft_lot"]==df["sqft_lot15"])/len(df))*100, "Percent of sqft_lot stays the same")
df["has_basement"] = df["sqft_basement"].apply(lambda x: 1 if x !=0 else 0)

df["renovated"] = df["yr_renovated"].apply(lambda x: 1 if x !=0 else 0)

df["sq_changed"] = np.array((df["sqft_living"]!=df["sqft_living15"]) | (df["sqft_lot"]!=df["sqft_lot15"]))



df["purchase_yr"] = df["date"].apply(lambda x: int(x[:4]))

df["yr_renovated"]=df["yr_renovated"].apply(lambda x: np.nan if x==0 else int(x))

df['yr_renovated'] = df['yr_renovated'].fillna(df['yr_built'])



df["time_after_renovation"] = df["purchase_yr"] - df["yr_renovated"]

df["sq_changed"] = df["sq_changed"].apply(lambda x: int(x))

df.head()

for i in ["has_basement", "waterfront", "renovated", "sq_changed"]:

    counts = df[i].value_counts()

    x = np.arange(len(counts))

    y = counts

    plt.bar(x, y)

    plt.xticks(x, np.array(counts.index))

    plt.title(i)

    plt.show()
features = pd.concat([df[:traindex], label], axis = 1)[["price", "waterfront", "renovated"]]



for f in ["waterfront", "renovated"]:

    print ("Portion of", f)

    for q in [0.25, 0.50, 0.75, 1]:

        quantile_feats = features.loc[(features["price"].quantile(q-0.25)<=features["price"]) & (features["price"]<features["price"].quantile(q))]

        print (q, "Quantile of ", f, "feature")



        counts = quantile_feats[f].value_counts()

        x = np.arange(len(counts))

        y = counts

        

        plt.figure(figsize=(5, 5))

        plt.bar(x, y)

        plt.title("{}, {} Quantile".format(f, q))

        plt.xticks(x, [0, 1])

        plt.show()
waterfront = features[features["waterfront"].apply(lambda x: x==1)]

non_waterfront = features[features["waterfront"].apply(lambda x: x==0)]



print ("Waterfront houses price mean: ", waterfront["price"].mean())

print ("Non Waterfront houses price mean: ",waterfront["price"].mean())
plt.boxplot(waterfront["price"])

plt.title("Waterfront_price_boxplot")

plt.show()



plt.boxplot(non_waterfront["price"])

plt.title("Non-Waterfront_price_boxplot")

plt.show()
renovated = features[features["renovated"].apply(lambda x: x==1)]

not_renovated = features[features["renovated"].apply(lambda x: x==0)]



print ("Renovated houses price mean: ", renovated["price"].mean())

print ("Not renovated houses price mean: ",not_renovated["price"].mean())
plt.boxplot(renovated["price"])

plt.title("Renovated_price_boxplot")

plt.show()



plt.boxplot(not_renovated["price"])

plt.title("Non-Renovated_price_boxplot")

plt.show()
df["how_old"] = df["purchase_yr"]-df["yr_built"]

df.head()
df["bedrooms_per_floor"] = df["bedrooms"]/df["floors"]

df["bathrooms_per_floor"] = df["bathrooms"]*df["bedrooms"]/df["floors"]



df["bedrooms_per_sqft"] = df["bedrooms"]/df["sqft_above"]*100



df["bathrooms_per_floor"] = df["bathrooms"]*df["bedrooms"]/df["floors"]

df["living_per_lot"] = df["sqft_living"] / df["sqft_lot"]

df["living_per_lot15"] = df["sqft_living15"] / df["sqft_lot15"]



df.head()
df = df.drop(["date", "id", "yr_built", "yr_renovated", "zipcode", "purchase_yr"], axis = 1)

df.head()
cor_mat = df.corr("spearman")



fig, ax = plt.subplots(figsize = (10, 10))

sns.heatmap(cor_mat, ax = ax, cmap = "jet")

plt.show()
df.hist(bins = 100, figsize = (20, 20))

plt.show()
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



x_train = df[:traindex].values

x_test = df[traindex:].values



y_train = label.values
import lightgbm as lgb

from sklearn.linear_model import Ridge

from sklearn.model_selection import KFold



SEED = 0

NFOLDS = 5

kf = KFold(n_splits=NFOLDS)
class SklearnHelper(object):

    def __init__(self, clf, seed=0, params=None, seed_bool = True):

        if(seed_bool == True):

            params['random_state'] = seed

        self.clf = clf(**params)



    def train(self, x_train, y_train):

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def feature_importances(self,x,y):

        return self.clf.fit(x,y).feature_importances_



def get_oof(clf, x_train, y, x_test):

    oof_train = np.zeros((len(x_train,)))

    oof_test = np.zeros((len(x_test,)))

    oof_test_skf = np.empty((NFOLDS, len(x_test)))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        print('\nFold {}'.format(i))

        x_tr = x_train[train_index]

        y_tr = y[train_index]

        x_te = x_train[test_index]



        clf.train(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
from sklearn.metrics import mean_squared_error

from math import sqrt



ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,

                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}



ridge = SklearnHelper(clf=Ridge, seed = SEED, params = ridge_params)

ridge_oof_train, ridge_oof_test = get_oof(ridge, x_train, y_train, x_test)

rms = sqrt(mean_squared_error(y_train, ridge_oof_train))

print('Ridge OOF RMSE: {}'.format(rms))
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,

GradientBoostingRegressor,ExtraTreesRegressor)
#Random Forest

rf_params = {

    "n_jobs": -1,

    "n_estimators": 500,

    #"warm_start": True,

    #"max_features":0.2,

    "max_depth":6,

    "min_samples_leaf": 2, 

    "max_features": "sqrt",

    "verbose":0

}



#Extra Trees

et_params = {

    "n_jobs": -1,

    "n_estimators": 500,

    #"max_features":0.5,

    "max_depth":8,

    "min_samples_leaf": 2, 

    "verbose":0

}



#AdaBoost

ada_params = {

    "n_estimators" : 500,

    "learning_rate" : 0.75

}



#Gradient Boosting

gb_params = {

    "n_estimators":500,

    #"max_features" : 0.2

    "max_depth" : 5,

    "min_samples_leaf" : 2,

    "verbose" : 0

}
rf = SklearnHelper(clf = RandomForestRegressor, seed = SEED, params = rf_params)

et = SklearnHelper(clf = ExtraTreesRegressor, seed = SEED, params = et_params)

ada = SklearnHelper(clf = AdaBoostRegressor, seed = SEED, params = ada_params)

gb = SklearnHelper(clf = GradientBoostingRegressor, seed = SEED, params = gb_params)
rf_oof_train, rf_oof_test = get_oof(rf, x_train, y_train, x_test)

et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test)

ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test)

gb_oof_train, gb_oof_test = get_oof(gb, x_train, y_train, x_test)
rms1 = sqrt(mean_squared_error(y_train, rf_oof_train))

print('Random Forest OOF RMSE: {}'.format(rms1))



rms2 = sqrt(mean_squared_error(y_train, et_oof_train))

print('Extra Trees OOF RMSE: {}'.format(rms2))



rms3 = sqrt(mean_squared_error(y_train, ada_oof_train))

print('AdaBoost OOF RMSE: {}'.format(rms3))



rms4 = sqrt(mean_squared_error(y_train, gb_oof_train))

print('GradientBoosting OOF RMSE: {}'.format(rms4))
rf_feature = rf.feature_importances(x_train,y_train)

et_feature = et.feature_importances(x_train, y_train)

ada_feature = ada.feature_importances(x_train, y_train)

gb_feature = gb.feature_importances(x_train,y_train)



cols = df[:traindex].columns.values

feature_df = pd.DataFrame({

    "features":cols,

    'Random Forest feature importances': rf_feature,

     'Extra Trees  feature importances': et_feature,

      'AdaBoost feature importances': ada_feature,

    'Gradient Boost feature importances': gb_feature

})



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



for i in list(feature_df.columns)[1:]:



    trace = go.Scatter(

        y = feature_df[i].values,

        x = feature_df['features'].values,

        mode='markers',

        marker=dict(

            sizemode = 'diameter',

            sizeref = 1,

            size = 25,

    #       size= feature_dataframe['AdaBoost feature importances'].values,

            #color = np.random.randn(500), #set color equal to a variable

            color = feature_df[i].values,

            colorscale='Portland',

            showscale=True

        ),

        text = feature_df['features'].values

    )

    data = [trace]



    layout= go.Layout(

        autosize= True,

        title= i,

        hovermode= 'closest',

    #     xaxis= dict(

    #         title= 'Pop',

    #         ticklen= 5,

    #         zeroline= False,

    #         gridwidth= 2,

    #     ),

        yaxis=dict(

            title= 'Feature Importance',

            ticklen= 5,

            gridwidth= 2

        ),

        showlegend= False

    )

    fig = go.Figure(data=data, layout=layout)

    py.iplot(fig,filename='scatter2010')
feature_df["mean"] = feature_df.mean(axis = 1)

data =[

    go.Bar(

    x =feature_df["features"].values,

    y = feature_df["mean"].values

    )

]



layout = go.Layout(

    title = "Feature Importance-Mean",

    yaxis =dict(

        title = "Importance", 

    )

)



fig = go.Figure(data = data, layout = layout)

py.iplot(fig, filename = "BAR")
x_train = np.concatenate((x_train, et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train), axis = 1)

x_test = np.concatenate((x_test, et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test), axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.20, random_state=2018)
print("Light Gradient Boosting Regressor")

lgbm_params = {'num_leaves': 31,

         'min_data_in_leaf': 30, 

         'objective':'regression',

         'max_depth': 1,

         'learning_rate': 0.01,

         "min_child_samples": 20,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9 ,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": 4,}



lgtrain = lgb.Dataset(X_train, y_train)

lgvalid = lgb.Dataset(X_valid, y_valid)
lgb_clf = lgb.train(lgbm_params,lgtrain,num_boost_round=10000,

        valid_sets=[lgtrain, lgvalid],

        valid_names=['train','valid'],

        early_stopping_rounds=100,

        verbose_eval=100)
print("Model Evaluation Stage")

print('RMSE:', np.sqrt(mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
lgpred = np.exp(lgb_clf.predict(x_test))
sub1 = pd.DataFrame({"id" : df_test["id"].values, "price": lgpred})

sub1.to_csv('lgb_0321.csv', index=False)