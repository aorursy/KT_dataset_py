# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline
train_df = pd.read_json("/kaggle/input/two-sigma-connect-rental-listing-inquiries/train.json.zip")

train_df.head()
test_df = pd.read_json("/kaggle/input/two-sigma-connect-rental-listing-inquiries/test.json.zip")

print("Train Rows : ", train_df.shape[0])

print("Test Rows : ", test_df.shape[0])
target_level = train_df['interest_level'].value_counts()

plt.figure(figsize=(10,8))

sns.barplot(target_level.index,target_level.values,color=color[1])

plt.xlabel("Interset Level",fontsize=12)

plt.ylabel("Number of Occurence")

plt.show()

# Visualize the bathrooms

bathroom_bar = train_df['bathrooms'].value_counts()

plt.figure(figsize=(10,8))

sns.barplot(bathroom_bar.index,bathroom_bar.values,alpha=0.8,color=color[1])

plt.xlabel("Number of bathrooms")

plt.ylabel("Number of Occurence" )

plt.show()
# visulaize the bedroom features.

bedroom_bar = train_df['bedrooms'].value_counts()

plt.figure(figsize=(10,8))

sns.barplot(bedroom_bar.index,bedroom_bar.values,alpha=0.8,color=color[1])

plt.xlabel("Number of bedrooms")

plt.ylabel("Occurence of numbes")

plt.show()
plt.figure(figsize=(8,6))

sns.countplot(x='bedrooms', hue='interest_level', data=train_df)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('bedrooms', fontsize=12)

plt.show()
plt.figure(figsize=(8,6))

plt.scatter(range(train_df.shape[0]), np.sort(train_df.price.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('price', fontsize=12)

plt.show()
# Looks like there are some outliers in this feature. So let us remove them and then plot again.

ulimit = np.percentile(train_df.price.values, 99)

train_df['price'].loc[train_df['price']>ulimit] = ulimit

plt.figure(figsize=(8,6))

sns.distplot(train_df.price.values, bins=50, kde=True)

plt.xlabel('price', fontsize=12)

plt.show()
lower_limit = np.percentile(train_df.latitude.values, 1)

upper_limit = np.percentile(train_df.latitude.values, 99)

train_df['latitude'].loc[train_df['latitude']<lower_limit ] = lower_limit 

train_df['latitude'].loc[train_df['latitude']>upper_limit] = upper_limit



plt.figure(figsize=(8,6))

sns.distplot(train_df.latitude.values, bins=50, kde=False)

plt.xlabel('latitude', fontsize=12)

plt.show()
llimit = np.percentile(train_df.longitude.values, 1)

ulimit = np.percentile(train_df.longitude.values, 99)

train_df['longitude'].loc[train_df['longitude']<llimit] = llimit

train_df['longitude'].loc[train_df['longitude']>ulimit] = ulimit



plt.figure(figsize=(8,6))

sns.distplot(train_df.longitude.values, bins=50, kde=False)

plt.xlabel('longitude', fontsize=12)

plt.show()
train_df["created"] = pd.to_datetime(train_df["created"])

train_df["date_created"] = train_df["created"].dt.date

cnt_srs = train_df['date_created'].value_counts()





plt.figure(figsize=(12,4))

ax = plt.subplot(111)

ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)

ax.xaxis_date()

plt.xticks(rotation='vertical')

plt.show()
test_df["created"] = pd.to_datetime(test_df["created"])

test_df["date_created"] = test_df["created"].dt.date

cnt_srs = test_df['date_created'].value_counts()



plt.figure(figsize=(12,4))

ax = plt.subplot(111)

ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)

ax.xaxis_date()

plt.xticks(rotation='vertical')

plt.show()
train_df["hour_created"] = train_df["created"].dt.hour

cnt_srs = train_df['hour_created'].value_counts()



plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.show()
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
data_path = "/kaggle/input/two-sigma-connect-rental-listing-inquiries/"

train_file = data_path + "train.json.zip"

test_file = data_path + "test.json.zip"

train_df = pd.read_json(train_file)

test_df = pd.read_json(test_file)

print(train_df.shape)

print(test_df.shape)
features_to_use  = ["bathrooms", "bedrooms", "latitude", "longitude", "price"]
# count of photos #

train_df["num_photos"] = train_df["photos"].apply(len)

test_df["num_photos"] = test_df["photos"].apply(len)



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



# adding all these new features to use list #

features_to_use.extend(["num_photos", "num_features", "num_description_words","created_year", "created_month", "created_day", "listing_id", "created_hour"])
categorical = ["display_address", "manager_id", "building_id", "street_address"]



for f in categorical:

        if train_df[f].dtype=='object':

            #print(f)

            lbl = preprocessing.LabelEncoder()

            lbl.fit(list(train_df[f].values) + list(test_df[f].values))

            train_df[f] = lbl.transform(list(train_df[f].values))

            test_df[f] = lbl.transform(list(test_df[f].values))

            features_to_use.append(f)
train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

print(train_df["features"].head())

tfidf = CountVectorizer(stop_words='english', max_features=200)

tr_sparse = tfidf.fit_transform(train_df["features"])

te_sparse = tfidf.transform(test_df["features"])
train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()

test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()



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
preds, model = runXGB(train_X, train_y, test_X, num_rounds=400)

out_df = pd.DataFrame(preds)
out_df
out_df.columns = ["high", "medium", "low"]
out_df
out_df["listing_id"] = test_df.listing_id.values

out_df
out_df.to_csv("xgb_starter2.csv", index=False)