

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

data = ks[data_cols].join(encoded)



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

    bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid], 

                    early_stopping_rounds=10, verbose_eval=False)



    valid_pred = bst.predict(valid[feature_cols])

    valid_score = metrics.roc_auc_score(valid['outcome'], valid_pred)

    print(f"Validation AUC score: {valid_score:.4f}")
# Train a model (on the baseline data)

train, valid, test = get_data_splits(data)

train_model(train, valid)
import category_encoders as ce

cat_features = ['category', 'currency', 'country']



# Create the encoder

count_enc = ce.CountEncoder()



# Transform the features, rename the columns with the _count suffix, and join to dataframe

count_encoded = count_enc.fit_transform(ks[cat_features])

data = data.join(count_encoded.add_suffix("_count"))



# Train a model 

train, valid, test = get_data_splits(data)

train_model(train, valid)
# Create the encoder

target_enc = ce.TargetEncoder(cols=cat_features)

target_enc.fit(train[cat_features], train['outcome'])



# Transform the features, rename the columns with _target suffix, and join to dataframe

train_TE = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))

valid_TE = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))



# Train a model

train_model(train_TE, valid_TE)
# Create the encoder

target_enc = ce.CatBoostEncoder(cols=cat_features)

target_enc.fit(train[cat_features], train['outcome'])



# Transform the features, rename columns with _cb suffix, and join to dataframe

train_CBE = train.join(target_enc.transform(train[cat_features]).add_suffix('_cb'))

valid_CBE = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_cb'))



# Train a model

train_model(train_CBE, valid_CBE)