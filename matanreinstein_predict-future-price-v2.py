import pandas as pd

import matplotlib

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from itertools import product

from sklearn.base import BaseEstimator, TransformerMixin

import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import LabelEncoder
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sales_train.head()
sales_train.hist(figsize=(30,15))
test.hist(figsize=(30,15))
plt.figure(figsize=(15,7))

plt.xlim(sales_train.item_cnt_day.min(), sales_train.item_cnt_day.max())

sns.boxplot(sales_train.item_cnt_day)
day_cnt_more_than_20 = sales_train[sales_train['item_cnt_day'] > 20].copy()
mean_item_cnts = sales_train.groupby(['item_id', 'date_block_num'])['item_cnt_day'].mean().to_frame('mean_item_cnt')
day_cnt_more_than_20 = day_cnt_more_than_20.merge(mean_item_cnts, how = 'left', on = ['item_id', 'date_block_num'])
day_cnt_more_than_20[day_cnt_more_than_20.item_cnt_day > 10 * day_cnt_more_than_20.mean_item_cnt].head()
del day_cnt_more_than_20
plt.xlim(sales_train.item_price.min(), sales_train.item_price.max())

sns.boxplot(sales_train.item_price)
sales_train[sales_train.item_price > 100000]
sales_train[sales_train.item_id == 6066]
items[items.item_id == 6066]
items[items.item_name.str.contains('Radmin')]
test[test.item_id == 6066]
sales_train[sales_train.item_id == 6065]
sales_train.drop(1163158, inplace=True)
sales_train[sales_train.item_id == 6066]
plt.xlim(sales_train.item_price.min(), sales_train.item_price.max())

sns.boxplot(sales_train.item_price)
sales_train[sales_train.item_price > 45000]
items[items.item_id == 11365]
sales_train[sales_train.item_id == 11365]
test[test.item_id == 11365]
sales_train.loc[885138,'item_price'] = sales_train[sales_train.item_id == 11365].drop(885138).item_price.mean()
items[items.item_name.str.contains('UserGate')]
test[test.item_id == 7241]
items[items.item_id == 13403]
sales_train[sales_train.item_id == 13403]
items.head()
items.info()
items[items.item_name.str.contains('ABBYY FineReader')].item_name
categories.info()
categories.head()
categories[categories.item_category_name.str.contains('Книги - Аудиокниги')]
similiar_cat_df = items[(items.item_category_id == 43) | (items.item_category_id == 44)]
items.head(100)
from fuzzywuzzy import fuzz

from fuzzywuzzy import process
item_name_list = items.item_name.unique()
#similiarities = np.array([[fuzz.ratio(w1,w2) for w2 in item_name_list] for w1 in item_name_list])
#similiarities[similiarities < 80] = 0
#from sklearn.cluster import AffinityPropagation

#aff_prop = AffinityPropagation(affinity='precomputed')

#aff_prop.fit(-1 * similiarities)
#aff_prop.labels_
#labels = np.array([(ind,label) for (ind,label) in enumerate(aff_prop.labels_)])
#labels
#sims = np.array([similiarities[ind][label] for (ind,label) in labels])
#sims.size
#labels[sims > 0]
#items[items.item_id == 18]
#items[items.item_id == 22]
sales_train['item_cnt_day'] = sales_train['item_cnt_day'].clip(0,20)
class MonthlyCountExtractor(BaseEstimator, TransformerMixin):

    def fit(self, df, y = None, **fit_params):

        return self

    def transform(self, df, **transform_params):

        items_per_month_df = df.groupby(['shop_id', 'item_id', 'date_block_num'])['item_cnt_day'].sum().to_frame('target').reset_index()

        valid_item_shop_month_triplets = []

        for date_block_num in df['date_block_num'].unique():

            monthly_entries = df[df['date_block_num'] == date_block_num]

            valid_item_shop_month_triplets.append(np.array([triplet for triplet in product([date_block_num], monthly_entries['shop_id'].unique(), monthly_entries['item_id'].unique())]))

        all_vals_df = pd.DataFrame(np.vstack(valid_item_shop_month_triplets), columns=['date_block_num', 'shop_id', 'item_id'])

        all_vals_df = all_vals_df.merge(items_per_month_df, on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')

        all_vals_df['target'] = all_vals_df['target'].fillna(0)

        return all_vals_df
monthly_vals_df = MonthlyCountExtractor().fit_transform(sales_train)
monthly_vals_df.head()
monthly_vals_df['target'].max()
monthly_vals_df.info()
len(monthly_vals_df)
len(sales_train)
len(sales_train['date_block_num'].unique()) * len(sales_train['item_id'].unique()) * len(sales_train['shop_id'].unique())
class MeanEncodingExtractor(BaseEstimator, TransformerMixin):

    

    def fit_transform(self, df, y = None, **fit_params):

        feature_to_encode = fit_params['feature_to_encode']

        encoded_feature = pd.DataFrame(columns= [feature_to_encode, feature_to_encode+'_mean_target_encoded', 'date_block_num'])

        for date_block_num in df['date_block_num'].unique():

            prev_months_df = df[df['date_block_num'] < date_block_num]

            curr_month_encoded_feature = prev_months_df.groupby([feature_to_encode])['target'].mean().to_frame(feature_to_encode+'_mean_target_encoded').reset_index()

            curr_month_encoded_feature['date_block_num'] = date_block_num

            encoded_feature = pd.concat([encoded_feature, curr_month_encoded_feature], sort= True)

        encoded_feature[feature_to_encode] = encoded_feature[feature_to_encode].astype(df[feature_to_encode].dtype)

        encoded_feature['date_block_num'] = encoded_feature['date_block_num'].astype(np.int8)

        return df.merge(encoded_feature, on = [feature_to_encode, 'date_block_num'], how = 'left')
mean_enc_df = MeanEncodingExtractor().fit_transform(monthly_vals_df,feature_to_encode = 'shop_id')
mean_enc_df.info()
mean_enc_df.tail()
class LagFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit_transform(self, df, y = None, **fit_params):

        lag = fit_params['lag']

        lag_df = df.copy()

        lag_df['date_block_num'] = lag_df['date_block_num'] + lag

        lag_df = lag_df.rename(columns = {'target' : 'target_lag_' + str(lag)})

        lag_df = lag_df[['shop_id', 'item_id', 'date_block_num', 'target_lag_' + str(lag)]]

        return df.merge(lag_df, on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')
lag_df = LagFeatureExtractor().fit_transform(mean_enc_df, lag = 12)
lag_df.head()
lag_df['target_lag_12'].isna().sum()
category_and_sub_category = categories['item_category_name'].str.split("-", n = 1, expand = True)

categories['category'] = category_and_sub_category[0].fillna('')

categories['category'] = LabelEncoder().fit_transform(categories['category'])

categories['sub_category'] = category_and_sub_category[1].fillna('')

categories['sub_category'] = LabelEncoder().fit_transform(categories['sub_category'])

categories.head()
shops['shop_city'] = shops['shop_name'].str.split(n=1, expand = True)[0]

shops['shop_city'] = shops['shop_city'].str.replace('\W', '')

shops.head()

shops['shop_city'] = LabelEncoder().fit_transform(shops['shop_city'])
class ShopItemCategoryExtractor(BaseEstimator, TransformerMixin):

    def fit(self, df, y = None, **fit_params):

        self.items_df = fit_params['items']

        self.categories_df = fit_params['categories']

        self.shops_df = fit_params['shops']

        return self

    def transform(self, df):

        df_with_category_and_item = df.merge(self.items_df, on = ['item_id'], how = 'left')

        df_with_category_and_item = df_with_category_and_item.merge(self.categories_df, on = 'item_category_id', how = 'left')

        df_with_category_and_item = df_with_category_and_item.merge(self.shops_df, on = 'shop_id', how = 'left')

        return df_with_category_and_item.drop(['item_category_name', 'shop_name', 'item_name'], axis = 'columns')
shop_category_df = ShopItemCategoryExtractor().fit_transform(monthly_vals_df, items = items, categories = categories, shops = shops)
shop_category_df.head()
test['date_block_num'] = 34
test.head()
class FeatureInteractionsExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, cols, interaction_type):

        self.cols = cols

        self.interaction_type = interaction_type

    def fit(self, df, y = None, **fit_params):

        return self

    def transform(self, df):

        interactions_df = pd.DataFrame()

        if not set(self.cols) <= set(df.columns):

            print('given Dataframe doesn''t contain all given columns: columns ',set(self.cols) - set(df.columns), 'missing')

            return interactions_df

        for i in range(len(self.cols)):

            for j in range(i+1,len(self.cols)):

                if self.interaction_type == 'multiplication':

                    interaction = df[self.cols[i]] * df[self.cols[j]]

                elif self.interaction_type == 'division':

                    interaction = df[self.cols[i]] / df[self.cols[j]]

                elif self.interaction_type == 'subtraction':

                    interaction = df[self.cols[i]] - df[self.cols[j]]

                else:

                    print('invalid interaction type given')

                    break

                interactions_df[self.cols[i] + '_' + self.cols[j] + '_' + self.interaction_type] = interaction

        return interactions_df
fie = FeatureInteractionsExtractor(['category','sub_category','shop_city'], 'subtraction')

inter_df = fie.fit_transform(shop_category_df)

inter_df.head()

del fie

del inter_df
def extract_features(sales_df):

    lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    features_df = MonthlyCountExtractor().fit_transform(sales_df)

    features_df['target'] = features_df['target'].clip(0,20)

    features_df = features_df.merge(test.drop(['ID'], axis = 'columns'), on = ['item_id', 'shop_id', 'date_block_num'], how = 'outer')

    features_df = ShopItemCategoryExtractor().fit_transform(features_df, items = items, categories = categories, shops = shops)

    features_df = MeanEncodingExtractor().fit_transform(features_df, feature_to_encode = 'shop_id')

    features_df = MeanEncodingExtractor().fit_transform(features_df, feature_to_encode = 'item_id')

    features_df = MeanEncodingExtractor().fit_transform(features_df, feature_to_encode = 'item_category_id')

    features_df = MeanEncodingExtractor().fit_transform(features_df, feature_to_encode = 'category')

    features_df = MeanEncodingExtractor().fit_transform(features_df, feature_to_encode = 'sub_category')

    features_df = MeanEncodingExtractor().fit_transform(features_df, feature_to_encode = 'shop_city')



    for lag in lags:

        features_df = LagFeatureExtractor().fit_transform(features_df, lag = lag)

    return features_df
features_df = extract_features(sales_train)
features_df.tail()
features_df.info()
sales_train.info()
def run_xgboost(xgboost_params, xgboost_num_rounds, train_df, val_df):

    train_matrix = xgb.DMatrix(data=train_df.drop(columns = ['target']), label=train_df['target'].values)

    model = xgb.train(xgboost_params, train_matrix, xgboost_num_rounds)

    train_prediction = model.predict(train_matrix).clip(0,20)

    train_error = mean_squared_error(train_prediction, train_df['target'].values.clip(0,20))

    

    validation_matrix = xgb.DMatrix(data=val_df.drop(columns = ['target']))

    validation_predictions = model.predict(validation_matrix).clip(0,20)

    validation_error = mean_squared_error(validation_predictions, val_df['target'].clip(0,20))

    

    print('XGBoost training error: ', train_error)

    print('XGBoost validation error: ' + str(validation_error))

    print('XGBoost r2 score: ', r2_score(val_df['target'].values.clip(0,20), validation_predictions))

    return model, validation_error
train_df = features_df[features_df['date_block_num'] < 33]

val_df = features_df[features_df['date_block_num'] == 33]

xgboost_params = {'max_depth':7, 

         'subsample':0.25,

         'eta':0.3, 

         'gamma': 1000,

         'min_child_weight': 5,

         'eval_metric':'rmse'}

xgboost_num_rounds = 5



xgboost_model, xgboost_val_error = run_xgboost(xgboost_params, xgboost_num_rounds, train_df, val_df)
xgb.plot_importance(xgboost_model, importance_type = 'gain')
submission_df = test.merge(features_df, on = ['item_id', 'shop_id', 'date_block_num'], how = 'left').drop(['ID'], axis = 'columns')

submission_df = submission_df[features_df.columns]

submission_df = submission_df.drop(['target'], axis = 'columns')
submission_matrix = xgb.DMatrix(data=submission_df)

submission_pred_xgboost = xgboost_model.predict(submission_matrix).clip(0,20)
submission = pd.DataFrame()

submission['ID'] = test['ID']

submission['item_cnt_month'] = pd.Series(submission_pred_xgboost)

submission.set_index('ID', inplace = True)

submission.to_csv('submission.csv')