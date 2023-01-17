import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd 

import seaborn as sns
df=pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv", parse_dates=['deadline', 'launched'])
df.head()
df.groupby('state')['ID'].count()
df = df.drop(df[df.state == "live"].index)

df.groupby('state')['ID'].count()
mapping = {"canceled": 0 , "failed" : 0, "suspended" : 0, "undefined": 0, "successful": 1}

df["outcome"] = df["state"].map(mapping)
(df.groupby('outcome')['ID'].count())
df["outcome"].hist()
df.head()
from sklearn.preprocessing import LabelEncoder

cat_features = ['category', 'currency', 'country']

encoder = LabelEncoder()

encoded = df[cat_features].apply(encoder.fit_transform)

encoded.head()
df = df.drop(df[cat_features], axis = 1)

df.head()
df = pd.concat([df, encoded], axis = 1)


df = df.assign(hour=df.launched.dt.hour,

               day=df.launched.dt.day,

               month=df.launched.dt.month,

               year=df.launched.dt.year)

df.head()
df = df[['goal', 'hour', 'day', 'month', 'year', 'category', 'currency', 'country', 'outcome']]

df.head()
X = df[['goal', 'hour', 'day', 'month', 'year', 'category', 'currency', 'country']]

y = df["outcome"]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    stratify=y, 

                                                    test_size=0.25)

import lightgbm as lgb



dtrain = lgb.Dataset(X_train, label=y_train)

dvalid = lgb.Dataset(X_test, label=y_test)



param = {'num_leaves': 64, 'objective': 'binary'}

param['metric'] = 'auc'

num_round = 100

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
from sklearn import metrics

ypred = bst.predict(X_test)

score = metrics.roc_auc_score(y_test, ypred)



print(f"Test AUC score: {score}")
ypred.shape
from sklearn.metrics import roc_curve, auc , roc_auc_score

from matplotlib import pyplot



lr_fpr, lr_tpr, _ = roc_curve(y_test, ypred)
ns_probs = [0 for _ in range(len(y_test))]

ns_auc = roc_auc_score(y_test, ns_probs)

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)



pyplot.plot(ns_fpr, ns_tpr, marker='.', label='None')

pyplot.plot(lr_fpr, lr_tpr, marker='.', label='model')

# axis labels

pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')

# show the legend

pyplot.legend()

# show the plot

pyplot.show()
import category_encoders as ce
cat_features = ['category', 'currency', 'country']

count_enc = ce.CountEncoder(cols=cat_features)

count_enc.fit(X_train[cat_features])
train_encoded = X_train.join(count_enc.transform(X_train[cat_features]).add_suffix('_count'))

valid_encoded = X_test.join(count_enc.transform(X_test[cat_features]).add_suffix('_count'))
valid_encoded.head()
import lightgbm as lgb



dtrain = lgb.Dataset(train_encoded, label=y_train)

dvalid = lgb.Dataset(valid_encoded, label=y_test)



param = {'num_leaves': 64, 'objective': 'binary'}

param['metric'] = 'auc'

num_round = 100

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)
ypred = bst.predict(valid_encoded)

score = metrics.roc_auc_score(y_test, ypred)

print (score)
import category_encoders as ce

cat_features = ['category', 'currency', 'country']

# Create the encoder itself

target_enc = ce.TargetEncoder(cols=cat_features)

# Fit the encoder using the categorical features and target

target_enc.fit(X_train[cat_features], y_train)



train_encoded = X_train.join(target_enc.transform(X_train[cat_features]).add_suffix('_target'))

valid_encoded = X_test.join(target_enc.transform(X_test[cat_features]).add_suffix('_target'))



train_encoded.head()
import lightgbm as lgb



dtrain = lgb.Dataset(train_encoded, label=y_train)

dvalid = lgb.Dataset(valid_encoded, label=y_test)



param = {'num_leaves': 64, 'objective': 'binary'}

param['metric'] = 'auc'

num_round = 100

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)

ypred = bst.predict(valid_encoded)

score = metrics.roc_auc_score(y_test, ypred)

print (score)
import category_encoders as ce

cat_features = ['category', 'currency', 'country']

# Create the encoder itself

target_enc = ce.CatBoostEncoder(cols=cat_features)

# Fit the encoder using the categorical features and target

target_enc.fit(X_train[cat_features], y_train)



train_encoded = X_train.join(target_enc.transform(X_train[cat_features]).add_suffix('_cb'))

valid_encoded = X_test.join(target_enc.transform(X_test[cat_features]).add_suffix('_cb'))



train_encoded.head()
import lightgbm as lgb



dtrain = lgb.Dataset(train_encoded, label=y_train)

dvalid = lgb.Dataset(valid_encoded, label=y_test)



param = {'num_leaves': 64, 'objective': 'binary'}

param['metric'] = 'auc'

num_round = 100

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)

ypred = bst.predict(valid_encoded)

score = metrics.roc_auc_score(y_test, ypred)

print (score)
df=pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv", parse_dates=['deadline', 'launched'])
interactions = df['category'] + "_" + df['country']

print(interactions.head(10))
from sklearn.preprocessing import LabelEncoder

label_enc = LabelEncoder()

df = df.assign(category_country=label_enc.fit_transform(interactions))

df.head()
launched = pd.Series(df.index, index=df.launched, name="count_7_days").sort_index()

launched.head(20)
import matplotlib.pyplot as plt

count_7_days = launched.rolling('7d').count() - 1

print(count_7_days.head(20))



# Ignore records with broken launch dates the first 7 numbers..

plt.plot(count_7_days[7:]);

plt.title("Competitions in the last 7 days");
count_7_days.index = launched.values  # launched.values  are the index

count_7_days = count_7_days.reindex(df.index)

count_7_days.head(10)
df = df.join(count_7_days)

df.head()
def time_since_last_project(series):

    # Return the time in hours

    return series.diff().dt.total_seconds() / 3600.



df_temp = df[['category', 'launched']].sort_values('launched')

df_temp.head()


df_temp.groupby('category').count().head(10)
timedeltas = df_temp.groupby('category').transform(time_since_last_project)

timedeltas.head(20)
timedeltas = timedeltas.fillna(timedeltas.median()).reindex(df.index)

timedeltas.head(20)
df["competitionValue"] = timedeltas
df.head()
df = df.drop(df[df.state == "live"].index)

mapping = {"canceled": 0 , "failed" : 0, "suspended" : 0, "undefined": 0, "successful": 1}

df["outcome"] = df["state"].map(mapping)

from sklearn.preprocessing import LabelEncoder

cat_features = ['category', 'currency', 'country']

encoder = LabelEncoder()

encoded = df[cat_features].apply(encoder.fit_transform)

df = df.drop(df[cat_features], axis = 1)

df = pd.concat([df, encoded], axis = 1)

df = df.assign(hour=df.launched.dt.hour,

               day=df.launched.dt.day,

               month=df.launched.dt.month,

               year=df.launched.dt.year)

df.head()
X = df[['goal', 'hour', 'day', 'month', 'year', 'category', 'currency', 'country', 'category_country',

       'count_7_days', 'competitionValue']]

y = df["outcome"]



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    stratify=y, 

                                                    test_size=0.25)
import category_encoders as ce

cat_features = ['category', 'currency', 'country']

# Create the encoder itself

target_enc = ce.CatBoostEncoder(cols=cat_features)

# Fit the encoder using the categorical features and target

target_enc.fit(X_train[cat_features], y_train)



train_encoded = X_train.join(target_enc.transform(X_train[cat_features]).add_suffix('_cb'))

valid_encoded = X_test.join(target_enc.transform(X_test[cat_features]).add_suffix('_cb'))



train_encoded.head()
import lightgbm as lgb



dtrain = lgb.Dataset(train_encoded, label=y_train)

dvalid = lgb.Dataset(valid_encoded, label=y_test)



param = {'num_leaves': 64, 'objective': 'binary'}

param['metric'] = 'auc'

num_round = 100

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)

ypred = bst.predict(valid_encoded)

score = metrics.roc_auc_score(y_test, ypred)

print (score)
from sklearn.feature_selection import SelectKBest, f_classif

# Keep 5 features

selector = SelectKBest(f_classif, k=5)

X_new = selector.fit_transform(train_encoded, y_train)

X_new
# Get back the features we've kept, zero out all other features

selected_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=train_encoded.index, 

                                 columns=train_encoded.columns)

selected_features.head()
selected_columns = selected_features.columns[selected_features.var() != 0]

train_encoded[selected_columns].head()
import lightgbm as lgb



dtrain = lgb.Dataset(train_encoded[selected_columns], label=y_train)

dvalid = lgb.Dataset(valid_encoded[selected_columns], label=y_test)



param = {'num_leaves': 64, 'objective': 'binary'}

param['metric'] = 'auc'

num_round = 100

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)

ypred = bst.predict(valid_encoded[selected_columns])

score = metrics.roc_auc_score(y_test, ypred)

print (score)
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel



# Set the regularization parameter C=1

logistic = LogisticRegression(C=1, penalty="l1", random_state=7).fit(train_encoded, y_train)

model = SelectFromModel(logistic, prefit=True)



X_new = model.transform(train_encoded)
# Get back the features we've kept, zero out all other features

selected_features_2 = pd.DataFrame(model.inverse_transform(X_new), 

                                 index=train_encoded.index, 

                                 columns=train_encoded.columns)

selected_features_2.head()
selected_columns = selected_features_2.columns[selected_features_2.var() != 0] # var() --> variance of columns

train_encoded[selected_columns].head()
import lightgbm as lgb



dtrain = lgb.Dataset(train_encoded[selected_columns], label=y_train)

dvalid = lgb.Dataset(valid_encoded[selected_columns], label=y_test)



param = {'num_leaves': 64, 'objective': 'binary'}

param['metric'] = 'auc'

num_round = 100

bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)

ypred = bst.predict(valid_encoded[selected_columns])

score = metrics.roc_auc_score(y_test, ypred)

print (score)