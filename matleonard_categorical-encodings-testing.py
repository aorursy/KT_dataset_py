import pandas as pd

from sklearn.preprocessing import LabelEncoder



ks = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv',

                 parse_dates=['deadline', 'launched'])



# Drop live projects

ks = ks.query('state != "live"')



# Add outcome column, "successful" == 1, others are 0

ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))



# Timestamp features

ks = ks.assign(hour=ks.launched.dt.hour,

               day=ks.launched.dt.day,

               month=ks.launched.dt.month,

               year=ks.launched.dt.year)



# Label encoding

cat_features = ['category', 'currency', 'country']

encoder = LabelEncoder()

encoded = ks[cat_features].apply(encoder.fit_transform)



data_cols = ['goal', 'hour', 'day', 'month', 'year', 'outcome']

baseline_data = ks[data_cols].join(encoded)
# Defining  functions that will help us test our encodings

import lightgbm as lgb

from sklearn import metrics



def get_data_splits(dataframe, valid_fraction=0.1):

    valid_fraction = 0.1

    valid_size = int(len(dataframe) * valid_fraction)



    train = dataframe[:-valid_size * 2]

    # valid size == test size, last two sections of the data

    valid = dataframe[-valid_size * 2:-valid_size]

    test = dataframe[-valid_size:]

    

    return train, valid, test



def train_model(train, valid):

    feature_cols = train.columns.drop('outcome')



    dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])

    dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])



    param = {'num_leaves': 64, 'objective': 'binary', 

             'metric': 'auc', 'seed': 7}

    print("Training model!")

    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 

                    early_stopping_rounds=10, verbose_eval=False)



    valid_pred = bst.predict(valid[feature_cols])

    valid_score = metrics.roc_auc_score(valid['outcome'], valid_pred)

    print(f"Validation AUC score: {valid_score:.4f}")

    return bst
# Training a model on the baseline data

train, valid, _ = get_data_splits(baseline_data)

bst = train_model(train, valid)
import category_encoders as ce

cat_features = ['category', 'currency', 'country']

count_enc = ce.CountEncoder()

count_encoded = count_enc.fit_transform(ks[cat_features])



data = baseline_data.join(count_encoded.add_suffix("_count"))



# Training a model on the baseline data

train, valid, test = get_data_splits(data)

bst = train_model(train, valid)
import category_encoders as ce

cat_features = ['category', 'currency', 'country']



# Create the encoder itself

target_enc = ce.TargetEncoder(cols=cat_features)



train, valid, _ = get_data_splits(data)



# Fit the encoder using the categorical features and target

target_enc.fit(train[cat_features], train['outcome'])



# Transform the features, rename the columns with _target suffix, and join to dataframe

train = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))

valid = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))



train.head()

bst = train_model(train, valid)
cat_features = ['category', 'currency', 'country']

target_enc = ce.CatBoostEncoder(cols=cat_features)



train, valid, _ = get_data_splits(data)

target_enc.fit(train[cat_features], train['outcome'])



train = train.join(target_enc.transform(train[cat_features]).add_suffix('_cb'))

valid = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_cb'))



bst = train_model(train, valid)
from sklearn.decomposition import TruncatedSVD



# Use 3 components in the latent vectors

svd = TruncatedSVD(n_components=3)
train, valid, _ = get_data_splits(data)



# Create a sparse matrix with cooccurence counts

pair_counts = train.groupby(['country', 'category'])['outcome'].count()

pair_counts.head(10)
pair_matrix = pair_counts.unstack(fill_value=0)

pair_matrix
svd_encoding = pd.DataFrame(svd.fit_transform(pair_matrix))

svd_encoding.head(10)
encoded = svd_encoding.reindex(data['country']).set_index(data.index)

encoded.head(10)
# Join encoded feature to the dataframe, with info in the column names

data_svd = data.join(encoded.add_prefix("country_category_svd_"))

data_svd.head()
train, valid, _ = get_data_splits(data_svd)

bst = train_model(train, valid)