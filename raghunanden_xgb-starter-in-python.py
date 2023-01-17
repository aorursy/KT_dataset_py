import os

import sys

import operator

import numpy as np

import pandas as pd

from scipy import sparse

import xgboost as xgb

from sklearn import model_selection, preprocessing, ensemble

from sklearn.metrics import log_loss

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, num_rounds=1000):

    param = {}

    param['objective'] = 'multi:softprob'

    param['eta'] = 0.1

    param['max_depth'] = 6

    param['silent'] = 1

    param['num_class'] = 3

    param['eval_metric'] = "mlogloss"

    param['min_child_weight'] = 1

    param['subsample'] = 0.7

    param['colsample_bytree'] = 0.7

    param['seed'] = seed_val

    num_rounds = num_rounds



    plst = list(param.items())

    xgtrain = xgb.DMatrix(train_X, label=train_y)



    if test_y is not None:

        xgtest = xgb.DMatrix(test_X, label=test_y)

        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)

    else:

        xgtest = xgb.DMatrix(test_X)

        model = xgb.train(plst, xgtrain, num_rounds)



    pred_test_y = model.predict(xgtest)

    return pred_test_y, model
data_path = "../input/"

train_file = data_path + "train.json"

test_file = data_path + "test.json"

train_df = pd.read_json(train_file)

test_df = pd.read_json(test_file)

print(train_df.shape)

print(test_df.shape)
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]

train_df.columns
import datetime as dt

# count of photos #

train_df["num_photos"] = train_df["photos"].apply(len)

test_df["num_photos"] = test_df["photos"].apply(len)



## price per bedroom

## price per bathroom

train_df["price_be"] = train_df["price"]/train_df["bedrooms"]

train_df["price_ba"] = train_df["price"]/train_df["bathrooms"]



# count of "features" #

train_df["num_features"] = train_df["features"].apply(len)

test_df["num_features"] = test_df["features"].apply(len)



# count of words present in description column #

train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))

test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))



# convert the created column to datetime object so as to extract more features 

train_df["created"] = pd.to_datetime(train_df["created"])

test_df["created"] = pd.to_datetime(test_df["created"])



# Let us extract some features like year, month, day, hour from date columns #

train_df["created_year"] = train_df["created"].dt.year

test_df["created_year"] = test_df["created"].dt.year

train_df["created_month"] = train_df["created"].dt.month

test_df["created_month"] = test_df["created"].dt.month

train_df["created_day"] = train_df["created"].dt.day

test_df["created_day"] = test_df["created"].dt.day

train_df["created_hour"] = train_df["created"].dt.hour

test_df["created_hour"] = test_df["created"].dt.hour

train_df['created_weekday'] = train_df['created'].dt.weekday

train_df['created_week'] = train_df['created'].dt.week

train_df['created_quarter'] = train_df['created'].dt.quarter



train_df['created_weekend'] = ((train_df['created_weekday'] == 5) & (train_df['created_weekday'] == 6))

train_df['created_wd'] = ((train_df['created_weekday'] != 5) & (train_df['created_weekday'] != 6))

train_df['created'] = train_df['created'].map(lambda x: float((x - dt.datetime(1899, 12, 30)).days) + (float((x - dt.datetime(1899, 12, 30)).seconds) / 86400))



train_df['x5'] = train_df['latitude'].map(lambda x : round(x,5))

train_df['y5'] = train_df['longitude'].map(lambda x : round(x,5))

train_df['x4'] = train_df['latitude'].map(lambda x : round(x,4))

train_df['y4'] = train_df['longitude'].map(lambda x : round(x,4))

train_df['x3'] = train_df['latitude'].map(lambda x : round(x,3))

train_df['y3'] = train_df['longitude'].map(lambda x : round(x,3))

train_df['x2'] = train_df['latitude'].map(lambda x : round(x,2))

train_df['y2'] = train_df['longitude'].map(lambda x : round(x,2))



# adding all these new features to use list #

features_to_use.extend(["num_photos", "num_features", "num_description_words","created_year", "created_month", "created_day", "listing_id", "created_hour",'created_weekday'])

#features_to_use.extend(["created_week",'created_quarter','created','y2','x2','y3','x3','y4','x4','y5','x5'])
categorical = ["display_address", "manager_id", "building_id", "street_address"]

for f in categorical:

        if train_df[f].dtype=='object':

            #print(f)

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(train_df[f].values) + list(test_df[f].values))

            train_df[f] = lbl.transform(list(train_df[f].values))

            test_df[f] = lbl.transform(list(test_df[f].values))

            features_to_use.append(f)
#Our feature construction class will inherit from these two base classes of sklearn.

from sklearn.base import BaseEstimator

from sklearn.base import TransformerMixin



class manager_skill(BaseEstimator, TransformerMixin):

    """

    Adds the column "manager_skill" to the dataset, based on the Kaggle kernel

    "Improve Perfomances using Manager features" by den3b. The function should

    be usable in scikit-learn pipelines.

    

    Parameters

    ----------

    threshold : Minimum count of rental listings a manager must have in order

                to get his "own" score, otherwise the mean is assigned.



    Attributes

    ----------

    mapping : pandas dataframe

        contains the manager_skill per manager id.

        

    mean_skill : float

        The mean skill of managers with at least as many listings as the 

        threshold.

    """

    def __init__(self, threshold = 5):

        

        self.threshold = threshold

        

    def _reset(self):

        """Reset internal data-dependent state of the scaler, if necessary.

        

        __init__ parameters are not touched.

        """

        # Checking one attribute is enough, becase they are all set together

        # in fit        

        if hasattr(self, 'mapping_'):

            

            self.mapping_ = {}

            self.mean_skill_ = 0.0

        

    def fit(self, X,y):

        """Compute the skill values per manager for later use.

        

        Parameters

        ----------

        X : pandas dataframe, shape [n_samples, n_features]

            The rental data. It has to contain a column named "manager_id".

            

        y : pandas series or numpy array, shape [n_samples]

            The corresponding target values with encoding:

            low: 0.0

            medium: 1.0

            high: 2.0

        """        

        self._reset()

        

        temp = pd.concat([X.manager_id,pd.get_dummies(y)], axis = 1).groupby('manager_id').mean()

        temp.columns = ['low_frac', 'medium_frac', 'high_frac']

        temp['count'] = X.groupby('manager_id').count().iloc[:,1]

        

        

        temp['manager_skill'] = temp['high_frac']*2 + temp['medium_frac']

        

        mean = temp.loc[temp['count'] >= self.threshold, 'manager_skill'].mean()

        

        temp.loc[temp['count'] < self.threshold, 'manager_skill'] = mean

        

        self.mapping_ = temp[['low_frac', 'medium_frac', 'high_frac','manager_skill']]

        self.mean_skill_ = mean

            

        return self

        

    def transform(self, X):

        """Add manager skill to a new matrix.

        

        Parameters

        ----------

        X : pandas dataframe, shape [n_samples, n_features]

            Input data, has to contain "manager_id".

        """        

        X = pd.merge(left = X, right = self.mapping_, how = 'left', left_on = 'manager_id', right_index = True)

        X['manager_skill'].fillna(self.mean_skill_, inplace = True)

        

        return X
trans = manager_skill()

train_df = trans.fit_transform(train_df, train_df['interest_level'])

test_df = trans.transform(test_df)

#features_to_use.extend(['low_frac', 'medium_frac', 'high_frac','manager_skill','longitude', 'latitude',"price_be",'price_ba'])

features_to_use.extend(['low_frac','medium_frac','high_frac',"price_be",'price_ba'])
train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

print(train_df["features"].head())

tfidf = CountVectorizer(stop_words='english', max_features=200)

tr_sparse = tfidf.fit_transform(train_df["features"])

te_sparse = tfidf.transform(test_df["features"])
train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()

#test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()



target_num_map = {'high':0, 'medium':1, 'low':2}

train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

print(train_X.shape, test_X.shape)
cv_scores = []

kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2016)

for dev_index, val_index in kf.split(range(train_X.shape[0])):

        dev_X, val_X = train_X[dev_index,:], train_X[val_index,:]

        dev_y, val_y = train_y[dev_index], train_y[val_index]

        preds, model = runXGB(dev_X, dev_y, val_X, val_y)

        cv_scores.append(log_loss(val_y, preds))

        print(cv_scores)

        break
#preds, model = runXGB(train_X, train_y, test_X, num_rounds=400)

#out_df = pd.DataFrame(preds)

#out_df.columns = ["high", "medium", "low"]

#out_df["listing_id"] = test_df.listing_id.values

#out_df.to_csv("xgb_starter2.csv", index=False)

train_df[features_to_use].info()

#train_df['created_weekend']