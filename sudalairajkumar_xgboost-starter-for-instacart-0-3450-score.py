### Import necessary modules ###

import numpy as np

import pandas as pd

import xgboost as xgb

from sklearn import metrics, model_selection
data_path = "../input/"

orders_df = pd.read_csv(data_path + "orders.csv", usecols=["order_id","user_id","order_number"])
# read the prior order file #

prior_df = pd.read_csv(data_path + "order_products__prior.csv")



# merge with the orders file to get the user_id #

prior_df = pd.merge(prior_df, orders_df, how="inner", on="order_id")



# get the products and reorder status of the latest purchase of each user #

prior_grouped_df = prior_df.groupby("user_id")["order_number"].aggregate("max").reset_index()

prior_df_latest = pd.merge(prior_df, prior_grouped_df, how="inner", on=["user_id", "order_number"])

prior_df_latest = prior_df_latest[["user_id", "product_id", "reordered"]]

prior_df_latest.columns = ["user_id", "product_id", "reordered_latest"]



# get the count of each product and number of reorders by the customer #

prior_df = prior_df.groupby(["user_id","product_id"])["reordered"].aggregate(["count", "sum"]).reset_index()

prior_df.columns = ["user_id", "product_id", "reordered_count", "reordered_sum"]



# merge the prior df with latest df #

prior_df = pd.merge(prior_df, prior_df_latest, how="left", on=["user_id","product_id"])

prior_df.head()
orders_df.drop(["order_number"],axis=1,inplace=True)



train_df = pd.read_csv(data_path + "order_products__train.csv", usecols=["order_id"])

train_df = train_df.groupby("order_id").aggregate("count").reset_index()

test_df = pd.read_csv(data_path + "sample_submission.csv", usecols=["order_id"])

train_df = pd.merge(train_df, orders_df, how="inner", on="order_id")

test_df = pd.merge(test_df, orders_df, how="inner", on="order_id")

print(train_df.shape, test_df.shape)
train_df = pd.merge(train_df, prior_df, how="inner", on="user_id")

test_df = pd.merge(test_df, prior_df, how="inner", on="user_id")

del prior_df, prior_grouped_df, prior_df_latest

print(train_df.shape, test_df.shape)
products_df = pd.read_csv(data_path + "products.csv", usecols=["product_id", "aisle_id", "department_id"])

train_df = pd.merge(train_df, products_df, how="inner", on="product_id")

test_df = pd.merge(test_df, products_df, how="inner", on="product_id")

del products_df

print(train_df.shape, test_df.shape)
train_y_df = pd.read_csv(data_path + "order_products__train.csv", usecols=["order_id", "product_id", "reordered"])

train_y_df = pd.merge(train_y_df, orders_df, how="inner", on="order_id")

train_y_df = train_y_df[["user_id", "product_id", "reordered"]]

#print(train_y_df.reordered.sum())

train_df = pd.merge(train_df, train_y_df, how="left", on=["user_id", "product_id"])

train_df["reordered"].fillna(0, inplace=True)

print(train_df.shape)

#print(train_df.reordered.sum())

del train_y_df
# target variable for train set #

train_y = train_df.reordered.values



# dataframe for test set predictions #

out_df = test_df[["order_id", "product_id"]]



# drop the unnecessary columns #

train_df = np.array(train_df.drop(["order_id", "user_id", "reordered"], axis=1))

test_df = np.array(test_df.drop(["order_id", "user_id"], axis=1))

print(train_df.shape, test_df.shape)
# function to run the xgboost model #

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0):

        params = {}

        params["objective"] = "binary:logistic"

        params['eval_metric'] = 'logloss'

        params["eta"] = 0.05

        params["subsample"] = 0.7

        params["min_child_weight"] = 10

        params["colsample_bytree"] = 0.7

        params["max_depth"] = 8

        params["silent"] = 1

        params["seed"] = seed_val

        num_rounds = 100

        plst = list(params.items())

        xgtrain = xgb.DMatrix(train_X, label=train_y)



        if test_y is not None:

                xgtest = xgb.DMatrix(test_X, label=test_y)

                watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]

                model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=10)

        else:

                xgtest = xgb.DMatrix(test_X)

                model = xgb.train(plst, xgtrain, num_rounds)



        pred_test_y = model.predict(xgtest)

        return pred_test_y
# run the xgboost model #

pred = runXGB(train_df, train_y, test_df)

del train_df, test_df



# use a cut-off value to get the predictions #

cutoff = 0.2

pred[pred>=cutoff] = 1

pred[pred<cutoff] = 0

out_df["Pred"] = pred

out_df = out_df.ix[out_df["Pred"].astype('int')==1]
# when there are more than 1 product, merge them to a single string #

def merge_products(x):

    return " ".join(list(x.astype('str')))

out_df = out_df.groupby("order_id")["product_id"].aggregate(merge_products).reset_index()

out_df.columns = ["order_id", "products"]
# read the sample csv file and populate the products from predictions #

sub_df = pd.read_csv(data_path + "sample_submission.csv", usecols=["order_id"])

sub_df = pd.merge(sub_df, out_df, how="left", on="order_id")



# when there are no predictions use "None" #

sub_df["products"].fillna("None", inplace=True)

sub_df.to_csv("xgb_starter_3450.csv", index=False)