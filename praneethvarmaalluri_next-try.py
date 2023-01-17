import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

%matplotlib inline
import os
print(os.listdir("../input"))
df_train = pd.read_csv("../input/train_V2.csv")
df_test = pd.read_csv("../input/test_V2.csv")
df_train.head()
df_train['matchType'].value_counts()
train_id = df_train.Id.values
test_id = df_test.Id.values
train_groupId = df_train.groupId.values
test_groupId = df_test.groupId.values
df_train['matchType'] = df_train['matchType'].replace('normal-solo', 'solo')
df_train['matchType'] = df_train['matchType'].replace('normal-duo', 'duo')
df_train['matchType'] = df_train['matchType'].replace('normal-squad', 'squad')
df_train['matchType'] = df_train['matchType'].replace('normal-solo-fpp', 'solo-fpp')
df_train['matchType'] = df_train['matchType'].replace('normal-duo-fpp', 'duo-fpp')
df_train['matchType'] = df_train['matchType'].replace('normal-squad-fpp', 'squad-fpp')

df_test['matchType'] = df_test['matchType'].replace('normal-solo', 'solo')
df_test['matchType'] = df_test['matchType'].replace('normal-duo', 'duo')
df_test['matchType'] = df_test['matchType'].replace('normal-squad', 'squad')
df_test['matchType'] = df_test['matchType'].replace('normal-solo-fpp', 'solo-fpp')
df_test['matchType'] = df_test['matchType'].replace('normal-duo-fpp', 'duo-fpp')
df_test['matchType'] = df_test['matchType'].replace('normal-squad-fpp', 'squad-fpp')
df_train['matchType'].value_counts()
df_train.dropna(subset=['winPlacePerc'], inplace=True)

df_train = df_train.drop(['matchId'], axis=1)
df_test = df_test.drop(['matchId'], axis=1)
df_train_solo = df_train[df_train.matchType == 'solo']
df_train_duo = df_train[df_train.matchType == 'duo']
df_train_squad = df_train[df_train.matchType == 'squad']
df_train_solo_fpp = df_train[df_train.matchType == 'solo-fpp']
df_train_duo_fpp = df_train[df_train.matchType == 'duo-fpp']
df_train_squad_fpp = df_train[df_train.matchType == 'squad-fpp']
df_train_flarefpp = df_train[df_train.matchType == 'flarefpp']
df_train_flaretpp = df_train[df_train.matchType == 'flaretpp']
df_train_crashfpp = df_train[df_train.matchType == 'crashfpp']
df_train_crashtpp = df_train[df_train.matchType == 'crashtpp']

df_train_solo = df_train_solo.drop(['matchType'], axis=1)
df_train_duo = df_train_duo.drop(['matchType'], axis=1)
df_train_squad = df_train_squad.drop(['matchType'], axis=1)
df_train_solo_fpp = df_train_solo_fpp.drop(['matchType'], axis=1)
df_train_duo_fpp = df_train_duo_fpp.drop(['matchType'], axis=1)
df_train_squad_fpp = df_train_squad_fpp.drop(['matchType'], axis=1)
df_train_flarefpp = df_train_flarefpp.drop(['matchType'], axis=1)
df_train_flaretpp = df_train_flaretpp.drop(['matchType'], axis=1)
df_train_crashfpp = df_train_crashfpp.drop(['matchType'], axis=1)
df_train_crashtpp = df_train_crashtpp.drop(['matchType'], axis=1)
df_test_solo = df_test[df_test.matchType == 'solo']
df_test_duo = df_test[df_test.matchType == 'duo']
df_test_squad = df_test[df_test.matchType == 'squad']
df_test_solo_fpp = df_test[df_test.matchType == 'solo-fpp']
df_test_duo_fpp = df_test[df_test.matchType == 'duo-fpp']
df_test_squad_fpp = df_test[df_test.matchType == 'squad-fpp']
df_test_flarefpp = df_test[df_test.matchType == 'flarefpp']
df_test_flaretpp = df_test[df_test.matchType == 'flaretpp']
df_test_crashfpp = df_test[df_test.matchType == 'crashfpp']
df_test_crashtpp = df_test[df_test.matchType == 'crashtpp']

df_test_solo = df_test_solo.drop(['matchType'], axis=1)
df_test_duo = df_test_duo.drop(['matchType'], axis=1)
df_test_squad = df_test_squad.drop(['matchType'], axis=1)
df_test_solo_fpp = df_test_solo_fpp.drop(['matchType'], axis=1)
df_test_duo_fpp = df_test_duo_fpp.drop(['matchType'], axis=1)
df_test_squad_fpp = df_test_squad_fpp.drop(['matchType'], axis=1)
df_test_flarefpp = df_test_flarefpp.drop(['matchType'], axis=1)
df_test_flaretpp = df_test_flaretpp.drop(['matchType'], axis=1)
df_test_crashfpp = df_test_crashfpp.drop(['matchType'], axis=1)
df_test_crashtpp = df_test_crashtpp.drop(['matchType'], axis=1)
# analysing solo

df_train_solo = df_train_solo.groupby('groupId',as_index=False).mean()
df_test_solo = df_test_solo.groupby('groupId',as_index=False).mean()

train_solo_groupId = df_train_solo.groupId.values
test_solo_groupId = df_test_solo.groupId.values

df_train_solo = df_train_solo.drop(['groupId'], axis=1)
df_test_solo = df_test_solo.drop(['groupId'], axis=1)

label_solo = df_train_solo['winPlacePerc']

df_train_solo = df_train_solo.drop(['winPlacePerc'], axis=1)

df_train_solo.describe()
df_train_solo = df_train_solo.drop(['DBNOs','revives'], axis=1)
df_test_solo = df_test_solo.drop(['DBNOs','revives'], axis=1)

x_train_solo = df_train_solo.values
x_test_solo = df_test_solo.values
y_true_solo = label_solo.values
# correlation heat map

corr_solo = df_train_solo.corr()
sns.heatmap(corr_solo, xticklabels=corr_solo.columns, yticklabels=corr_solo.columns)
# standard scaling

from sklearn.preprocessing import StandardScaler
scaler_solo = StandardScaler()
x_train_solo_scaled = scaler_solo.fit_transform(x_train_solo)
x_test_solo_scaled = scaler_solo.transform(x_test_solo)
# performing PCA

from sklearn.decomposition import PCA
pca_solo = PCA(0.95) 
x_train_solo_scaled = pca_solo.fit_transform(x_train_solo_scaled)
x_test_solo_scaled = pca_solo.transform(x_test_solo_scaled)
pca_solo.explained_variance_ratio_
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_solo_scaled)
dnn_reg_solo = tf.contrib.learn.DNNRegressor(hidden_units=[1000,1000,1000,1000], activation_fn = tf.nn.relu, feature_columns=feature_columns, config=config)
dnn_reg_solo = tf.contrib.learn.SKCompat(dnn_reg_solo) # to be compatible with sklearn
dnn_reg_solo.fit(x_train_solo_scaled, y_true_solo, batch_size=50, steps=50000)
# analysing duo

df_train_duo = df_train_duo.groupby('groupId',as_index=False).mean()
df_test_duo = df_test_duo.groupby('groupId',as_index=False).mean()

train_duo_groupId = df_train_duo.groupId.values
test_duo_groupId = df_test_duo.groupId.values

df_train_duo = df_train_duo.drop(['groupId'], axis=1)
df_test_duo = df_test_duo.drop(['groupId'], axis=1)

label_duo = df_train_duo['winPlacePerc']

df_train_duo = df_train_duo.drop(['winPlacePerc'], axis=1)

df_train_duo.describe()
x_train_duo = df_train_duo.values
x_test_duo = df_test_duo.values
y_true_duo = label_duo.values
# correlation heat map

corr_duo = df_train_duo.corr()
sns.heatmap(corr_duo, xticklabels = corr_duo.columns, yticklabels = corr_duo.columns)
# standard scaling

from sklearn.preprocessing import StandardScaler
scaler_duo = StandardScaler()
x_train_duo_scaled = scaler_duo.fit_transform(x_train_duo)
x_test_duo_scaled = scaler_duo.transform(x_test_duo)
# performing PCA

from sklearn.decomposition import PCA
pca_duo = PCA(0.95) 
x_train_duo_scaled = pca_duo.fit_transform(x_train_duo_scaled)
x_test_duo_scaled = pca_duo.transform(x_test_duo_scaled)
pca_duo.explained_variance_ratio_
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_duo_scaled)
dnn_reg_duo = tf.contrib.learn.DNNRegressor(hidden_units=[1000,1000,1000,1000], activation_fn = tf.nn.relu, feature_columns=feature_columns, config=config)
dnn_reg_duo = tf.contrib.learn.SKCompat(dnn_reg_duo) # to be compatible with sklearn
dnn_reg_duo.fit(x_train_duo_scaled, y_true_duo, batch_size=50, steps=50000)
# analysing squad

df_train_squad = df_train_squad.groupby('groupId',as_index=False).mean()
df_test_squad = df_test_squad.groupby('groupId',as_index=False).mean()

train_squad_groupId = df_train_squad.groupId.values
test_squad_groupId = df_test_squad.groupId.values

df_train_squad = df_train_squad.drop(['groupId'], axis=1)
df_test_squad = df_test_squad.drop(['groupId'], axis=1)

label_squad = df_train_squad['winPlacePerc']

df_train_squad = df_train_squad.drop(['winPlacePerc'], axis=1)

df_train_squad.describe()
x_train_squad = df_train_squad.values
x_test_squad = df_test_squad.values
y_true_squad = label_squad.values
# correlation heat map

corr_squad = df_train_squad.corr()
sns.heatmap(corr_squad, xticklabels = corr_squad.columns, yticklabels = corr_squad.columns)
# standard scaling

from sklearn.preprocessing import StandardScaler
scaler_squad = StandardScaler()
x_train_squad_scaled = scaler_squad.fit_transform(x_train_squad)
x_test_squad_scaled = scaler_squad.transform(x_test_squad)
# performing PCA

from sklearn.decomposition import PCA
pca_squad = PCA(0.95) 
x_train_squad_scaled = pca_squad.fit_transform(x_train_squad_scaled)
x_test_squad_scaled = pca_squad.transform(x_test_squad_scaled)
pca_squad.explained_variance_ratio_
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_squad_scaled)
dnn_reg_squad = tf.contrib.learn.DNNRegressor(hidden_units=[1000,1000,1000,1000], activation_fn = tf.nn.relu, feature_columns=feature_columns, config=config)
dnn_reg_squad = tf.contrib.learn.SKCompat(dnn_reg_squad) # to be compatible with sklearn
dnn_reg_squad.fit(x_train_squad_scaled, y_true_squad, batch_size=50, steps=50000)
# analysing solo fpp

df_train_solo_fpp = df_train_solo_fpp.groupby('groupId',as_index=False).mean()
df_test_solo_fpp = df_test_solo_fpp.groupby('groupId',as_index=False).mean()

train_solo_fpp_groupId = df_train_solo_fpp.groupId.values
test_solo_fpp_groupId = df_test_solo_fpp.groupId.values

df_train_solo_fpp = df_train_solo_fpp.drop(['groupId'], axis=1)
df_test_solo_fpp = df_test_solo_fpp.drop(['groupId'], axis=1)

label_solo_fpp = df_train_solo_fpp['winPlacePerc']

df_train_solo_fpp = df_train_solo_fpp.drop(['winPlacePerc'], axis=1)

df_train_solo_fpp.describe()
df_train_solo_fpp = df_train_solo_fpp.drop(['DBNOs','revives'], axis=1)
df_test_solo_fpp = df_test_solo_fpp.drop(['DBNOs','revives'], axis=1)

x_train_solo_fpp = df_train_solo_fpp.values
x_test_solo_fpp = df_test_solo_fpp.values
y_true_solo_fpp = label_solo_fpp.values
# correlation heat map

corr_solo_fpp = df_train_solo_fpp.corr()
sns.heatmap(corr_solo_fpp, xticklabels=corr_solo_fpp.columns, yticklabels=corr_solo_fpp.columns)
# standard scaling

from sklearn.preprocessing import StandardScaler
scaler_solo_fpp = StandardScaler()
x_train_solo_fpp_scaled = scaler_solo_fpp.fit_transform(x_train_solo_fpp)
x_test_solo_fpp_scaled = scaler_solo_fpp.transform(x_test_solo_fpp)
# performing PCA

from sklearn.decomposition import PCA
pca_solo_fpp = PCA(0.98)
x_train_solo_fpp_scaled = pca_solo_fpp.fit_transform(x_train_solo_fpp_scaled)
x_test_solo_fpp_scaled = pca_solo_fpp.transform(x_test_solo_fpp_scaled)
pca_solo_fpp.explained_variance_ratio_
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_solo_fpp_scaled)
dnn_reg_solo_fpp = tf.contrib.learn.DNNRegressor(hidden_units=[1000,1000,1000,1000], activation_fn = tf.nn.relu, feature_columns=feature_columns, config=config)
dnn_reg_solo_fpp = tf.contrib.learn.SKCompat(dnn_reg_solo_fpp) # to be compatible with sklearn
dnn_reg_solo_fpp.fit(x_train_solo_fpp_scaled, y_true_solo_fpp, batch_size=50, steps=50000)
# analysing duo fpp

df_train_duo_fpp = df_train_duo_fpp.groupby('groupId',as_index=False).mean()
df_test_duo_fpp = df_test_duo_fpp.groupby('groupId',as_index=False).mean()

train_duo_fpp_groupId = df_train_duo_fpp.groupId.values
test_duo_fpp_groupId = df_test_duo_fpp.groupId.values

df_train_duo_fpp = df_train_duo_fpp.drop(['groupId'], axis=1)
df_test_duo_fpp = df_test_duo_fpp.drop(['groupId'], axis=1)

label_duo_fpp = df_train_duo_fpp['winPlacePerc']

df_train_duo_fpp = df_train_duo_fpp.drop(['winPlacePerc'], axis=1)

df_train_duo_fpp.describe()
x_train_duo_fpp = df_train_duo_fpp.values
x_test_duo_fpp = df_test_duo_fpp.values
y_true_duo_fpp = label_duo_fpp.values
# correlation heat map

corr_duo_fpp = df_train_duo_fpp.corr()
sns.heatmap(corr_duo_fpp, xticklabels = corr_duo_fpp.columns, yticklabels = corr_duo_fpp.columns)
# standard scaling

from sklearn.preprocessing import StandardScaler
scaler_duo_fpp = StandardScaler()
x_train_duo_fpp_scaled = scaler_duo_fpp.fit_transform(x_train_duo_fpp)
x_test_duo_fpp_scaled = scaler_duo_fpp.transform(x_test_duo_fpp)
# performing PCA

from sklearn.decomposition import PCA
pca_duo_fpp = PCA(0.98)
x_train_duo_fpp_scaled = pca_duo_fpp.fit_transform(x_train_duo_fpp_scaled)
x_test_duo_fpp_scaled = pca_duo_fpp.transform(x_test_duo_fpp_scaled)
pca_duo_fpp.explained_variance_ratio_
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_duo_fpp_scaled)
dnn_reg_duo_fpp = tf.contrib.learn.DNNRegressor(hidden_units=[1000,1000,1000,1000], activation_fn = tf.nn.relu, feature_columns=feature_columns, config=config)
dnn_reg_duo_fpp = tf.contrib.learn.SKCompat(dnn_reg_duo_fpp) # to be compatible with sklearn
dnn_reg_duo_fpp.fit(x_train_duo_fpp_scaled, y_true_duo_fpp, batch_size=50, steps=50000)
# analysing squad fpp

df_train_squad_fpp = df_train_squad_fpp.groupby('groupId',as_index=False).mean()
df_test_squad_fpp = df_test_squad_fpp.groupby('groupId',as_index=False).mean()

train_squad_fpp_groupId = df_train_squad_fpp.groupId.values
test_squad_fpp_groupId = df_test_squad_fpp.groupId.values

df_train_squad_fpp = df_train_squad_fpp.drop(['groupId'], axis=1)
df_test_squad_fpp = df_test_squad_fpp.drop(['groupId'], axis=1)

label_squad_fpp = df_train_squad_fpp['winPlacePerc']

df_train_squad_fpp = df_train_squad_fpp.drop(['winPlacePerc'], axis=1)

df_train_squad_fpp.describe()
x_train_squad_fpp = df_train_squad_fpp.values
x_test_squad_fpp = df_test_squad_fpp.values
y_true_squad_fpp = label_squad_fpp.values
# correlation heat map

corr_squad_fpp = df_train_squad_fpp.corr()
sns.heatmap(corr_squad_fpp, xticklabels = corr_squad_fpp.columns, yticklabels = corr_squad_fpp.columns)
# standard scaling

from sklearn.preprocessing import StandardScaler
scaler_squad_fpp = StandardScaler()
x_train_squad_fpp_scaled = scaler_squad_fpp.fit_transform(x_train_squad_fpp)
x_test_squad_fpp_scaled = scaler_squad_fpp.transform(x_test_squad_fpp)
# performing PCA

from sklearn.decomposition import PCA
pca_squad_fpp = PCA(0.98)
x_train_squad_fpp_scaled = pca_squad_fpp.fit_transform(x_train_squad_fpp_scaled)
x_test_squad_fpp_scaled = pca_squad_fpp.transform(x_test_squad_fpp_scaled)
pca_squad_fpp.explained_variance_ratio_
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_squad_fpp_scaled)
dnn_reg_squad_fpp = tf.contrib.learn.DNNRegressor(hidden_units=[1000,1000,1000,1000], activation_fn = tf.nn.relu, feature_columns=feature_columns, config=config)
dnn_reg_squad_fpp = tf.contrib.learn.SKCompat(dnn_reg_squad_fpp) # to be compatible with sklearn
dnn_reg_squad_fpp.fit(x_train_squad_fpp_scaled, y_true_squad_fpp, batch_size=50, steps=50000)
# analysing flarefpp

df_train_flarefpp = df_train_flarefpp.groupby('groupId',as_index=False).mean()
df_test_flarefpp = df_test_flarefpp.groupby('groupId',as_index=False).mean()

train_flarefpp_groupId = df_train_flarefpp.groupId.values
test_flarefpp_groupId = df_test_flarefpp.groupId.values

df_train_flarefpp = df_train_flarefpp.drop(['groupId'], axis=1)
df_test_flarefpp = df_test_flarefpp.drop(['groupId'], axis=1)

label_flarefpp = df_train_flarefpp['winPlacePerc']

df_train_flarefpp = df_train_flarefpp.drop(['winPlacePerc'], axis=1)

df_train_flarefpp.describe()
df_train_flarefpp = df_train_flarefpp.drop(['killPoints','rankPoints','winPoints'], axis=1)
df_test_flarefpp = df_test_flarefpp.drop(['killPoints','rankPoints','winPoints'], axis=1)

x_train_flarefpp = df_train_flarefpp.values
x_test_flarefpp = df_test_flarefpp.values
y_true_flarefpp = label_flarefpp.values
# correlation heat map

corr_flarefpp = df_train_flarefpp.corr()
sns.heatmap(corr_flarefpp, xticklabels = corr_flarefpp.columns, yticklabels = corr_flarefpp.columns)
# standard scaling

from sklearn.preprocessing import StandardScaler
scaler_flarefpp = StandardScaler()
x_train_flarefpp_scaled = scaler_flarefpp.fit_transform(x_train_flarefpp)
x_test_flarefpp_scaled = scaler_flarefpp.transform(x_test_flarefpp)
# performing PCA

from sklearn.decomposition import PCA
pca_flarefpp = PCA(0.98)
x_train_flarefpp_scaled = pca_flarefpp.fit_transform(x_train_flarefpp_scaled)
x_test_flarefpp_scaled = pca_flarefpp.transform(x_test_flarefpp_scaled)
pca_flarefpp.explained_variance_ratio_
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_flarefpp_scaled)
dnn_reg_flarefpp = tf.contrib.learn.DNNRegressor(hidden_units=[1000,1000,1000,1000], activation_fn = tf.nn.relu, feature_columns=feature_columns, config=config)
dnn_reg_flarefpp = tf.contrib.learn.SKCompat(dnn_reg_flarefpp) # to be compatible with sklearn
dnn_reg_flarefpp.fit(x_train_flarefpp_scaled, y_true_flarefpp, batch_size=50, steps=50000)
# analysing flaretpp

df_train_flaretpp = df_train_flaretpp.groupby('groupId',as_index=False).mean()
df_test_flaretpp = df_test_flaretpp.groupby('groupId',as_index=False).mean()

train_flaretpp_groupId = df_train_flaretpp.groupId.values
test_flaretpp_groupId = df_test_flaretpp.groupId.values

df_train_flaretpp = df_train_flaretpp.drop(['groupId'], axis=1)
df_test_flaretpp = df_test_flaretpp.drop(['groupId'], axis=1)

label_flaretpp = df_train_flaretpp['winPlacePerc']

df_train_flaretpp = df_train_flaretpp.drop(['winPlacePerc'], axis=1)

df_train_flaretpp.describe()

x_train_flaretpp = df_train_flaretpp.values
x_test_flaretpp = df_test_flaretpp.values
y_true_flaretpp = label_flaretpp.values
# correlation heat map

corr_flaretpp = df_train_flaretpp.corr()
sns.heatmap(corr_flaretpp, xticklabels = corr_flaretpp.columns, yticklabels = corr_flaretpp.columns)
# standard scaling

from sklearn.preprocessing import StandardScaler
scaler_flaretpp = StandardScaler()
x_train_flaretpp_scaled = scaler_flaretpp.fit_transform(x_train_flaretpp)
x_test_flaretpp_scaled = scaler_flaretpp.transform(x_test_flaretpp)
# performing PCA

from sklearn.decomposition import PCA
pca_flaretpp = PCA(0.98)
x_train_flaretpp_scaled = pca_flaretpp.fit_transform(x_train_flaretpp_scaled)
x_test_flaretpp_scaled = pca_flaretpp.transform(x_test_flaretpp_scaled)
pca_flaretpp.explained_variance_ratio_
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_flaretpp_scaled)
dnn_reg_flaretpp = tf.contrib.learn.DNNRegressor(hidden_units=[1000,1000,1000,1000], activation_fn = tf.nn.relu, feature_columns=feature_columns, config=config)
dnn_reg_flaretpp = tf.contrib.learn.SKCompat(dnn_reg_flaretpp) # to be compatible with sklearn
dnn_reg_flaretpp.fit(x_train_flaretpp_scaled, y_true_flaretpp, batch_size=50, steps=50000)
# analysing crashfpp

df_train_crashfpp = df_train_crashfpp.groupby('groupId',as_index=False).mean()
df_test_crashfpp = df_test_crashfpp.groupby('groupId',as_index=False).mean()

train_crashfpp_groupId = df_train_crashfpp.groupId.values
test_crashfpp_groupId = df_test_crashfpp.groupId.values

df_train_crashfpp = df_train_crashfpp.drop(['groupId'], axis=1)
df_test_crashfpp = df_test_crashfpp.drop(['groupId'], axis=1)

label_crashfpp = df_train_crashfpp['winPlacePerc']

df_train_crashfpp = df_train_crashfpp.drop(['winPlacePerc'], axis=1)

df_train_crashfpp.describe()

df_train_crashfpp = df_train_crashfpp.drop(['killPoints','rankPoints','winPoints'], axis=1)
df_test_crashfpp = df_test_crashfpp.drop(['killPoints','rankPoints','winPoints'], axis=1)

x_train_crashfpp = df_train_crashfpp.values
x_test_crashfpp = df_test_crashfpp.values
y_true_crashfpp = label_crashfpp.values
 # correlation heat map

corr_crashfpp = df_train_crashfpp.corr()
sns.heatmap(corr_crashfpp, xticklabels = corr_crashfpp.columns, yticklabels = corr_crashfpp.columns)
# standard scaling

from sklearn.preprocessing import StandardScaler
scaler_crashfpp = StandardScaler()
x_train_crashfpp_scaled = scaler_crashfpp.fit_transform(x_train_crashfpp)
x_test_crashfpp_scaled = scaler_crashfpp.transform(x_test_crashfpp)
# performing PCA

from sklearn.decomposition import PCA
pca_crashfpp = PCA(0.98)
x_train_crashfpp_scaled = pca_crashfpp.fit_transform(x_train_crashfpp_scaled)
x_test_crashfpp_scaled = pca_crashfpp.transform(x_test_crashfpp_scaled)
pca_crashfpp.explained_variance_ratio_
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_crashfpp_scaled)
dnn_reg_crashfpp = tf.contrib.learn.DNNRegressor(hidden_units=[1000,1000,1000,1000], activation_fn = tf.nn.relu, feature_columns=feature_columns, config=config)
dnn_reg_crashfpp = tf.contrib.learn.SKCompat(dnn_reg_crashfpp) # to be compatible with sklearn
dnn_reg_crashfpp.fit(x_train_crashfpp_scaled, y_true_crashfpp, batch_size=50, steps=50000)
# analysing crashtpp

df_train_crashtpp = df_train_crashtpp.groupby('groupId',as_index=False).mean()
df_test_crashtpp = df_test_crashtpp.groupby('groupId',as_index=False).mean()

train_crashtpp_groupId = df_train_crashtpp.groupId.values
test_crashtpp_groupId = df_test_crashtpp.groupId.values

df_train_crashtpp = df_train_crashtpp.drop(['groupId'], axis=1)
df_test_crashtpp = df_test_crashtpp.drop(['groupId'], axis=1)

label_crashtpp = df_train_crashtpp['winPlacePerc']

df_train_crashtpp = df_train_crashtpp.drop(['winPlacePerc'], axis=1)

df_train_crashtpp.describe()
df_train_crashtpp = df_train_crashtpp.drop(['killPoints','rankPoints','winPoints'], axis=1)
df_test_crashtpp = df_test_crashtpp.drop(['killPoints','rankPoints','winPoints'], axis=1)
x_train_crashtpp = df_train_crashtpp.values
x_test_crashtpp = df_test_crashtpp.values
y_true_crashtpp = label_crashtpp.values
# correlation heat map

corr_crashtpp = df_train_crashtpp.corr()
sns.heatmap(corr_crashtpp, xticklabels = corr_crashtpp.columns, yticklabels = corr_crashtpp.columns)
# standard scaling

from sklearn.preprocessing import StandardScaler
scaler_crashtpp = StandardScaler()
x_train_crashtpp_scaled = scaler_crashtpp.fit_transform(x_train_crashtpp)
x_test_crashtpp_scaled = scaler_crashtpp.transform(x_test_crashtpp)
# performing PCA

from sklearn.decomposition import PCA
pca_crashtpp = PCA(0.98)
x_train_crashtpp_scaled = pca_crashtpp.fit_transform(x_train_crashtpp_scaled)
x_test_crashtpp_scaled = pca_crashtpp.transform(x_test_crashtpp_scaled)
pca_crashtpp.explained_variance_ratio_
# neural netowrk model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_crashtpp_scaled)
dnn_reg_crashtpp = tf.contrib.learn.DNNRegressor(hidden_units=[1000,1000,1000,1000], activation_fn = tf.nn.relu, feature_columns=feature_columns, config=config)
dnn_reg_crashtpp = tf.contrib.learn.SKCompat(dnn_reg_crashtpp) # to be compatible with sklearn
dnn_reg_crashtpp.fit(x_train_crashtpp_scaled, y_true_crashtpp, batch_size=50, steps=50000)
result_solo = dnn_reg_solo.predict(x_test_solo_scaled)
result_duo = dnn_reg_duo.predict(x_test_duo_scaled)
result_squad = dnn_reg_squad.predict(x_test_squad_scaled)
result_solo_fpp = dnn_reg_solo_fpp.predict(x_test_solo_fpp_scaled)
result_duo_fpp = dnn_reg_duo_fpp.predict(x_test_duo_fpp_scaled)
result_squad_fpp = dnn_reg_squad_fpp.predict(x_test_squad_fpp_scaled)
result_flarefpp = dnn_reg_flarefpp.predict(x_test_flarefpp_scaled)
result_flaretpp = dnn_reg_flaretpp.predict(x_test_flaretpp_scaled)
result_crashfpp = dnn_reg_crashfpp.predict(x_test_crashfpp_scaled)
result_crashtpp = dnn_reg_crashtpp.predict(x_test_crashtpp_scaled)
solo_groupId = pd.Series(test_solo_groupId,  name='groupId')
solo_winPlacePerc = pd.Series(result_solo["scores"], name='winPlacePerc')
df_solo = pd.concat([solo_groupId, solo_winPlacePerc], axis=1)

duo_groupId = pd.Series(test_duo_groupId,  name='groupId')
duo_winPlacePerc = pd.Series(result_duo["scores"], name='winPlacePerc')
df_duo = pd.concat([duo_groupId, duo_winPlacePerc], axis=1)

squad_groupId = pd.Series(test_squad_groupId,  name='groupId')
squad_winPlacePerc = pd.Series(result_squad["scores"], name='winPlacePerc')
df_squad = pd.concat([squad_groupId, squad_winPlacePerc], axis=1)

solo_fpp_groupId = pd.Series(test_solo_fpp_groupId,  name='groupId')
solo_fpp_winPlacePerc = pd.Series(result_solo_fpp["scores"], name='winPlacePerc')
df_solo_fpp = pd.concat([solo_fpp_groupId, solo_fpp_winPlacePerc], axis=1)

duo_fpp_groupId = pd.Series(test_duo_fpp_groupId,  name='groupId')
duo_fpp_winPlacePerc = pd.Series(result_duo_fpp["scores"], name='winPlacePerc')
df_duo_fpp = pd.concat([duo_fpp_groupId, duo_fpp_winPlacePerc], axis=1)

squad_fpp_groupId = pd.Series(test_squad_fpp_groupId,  name='groupId')
squad_fpp_winPlacePerc = pd.Series(result_squad_fpp["scores"], name='winPlacePerc')
df_squad_fpp = pd.concat([squad_fpp_groupId, squad_fpp_winPlacePerc], axis=1)

flarefpp_groupId = pd.Series(test_flarefpp_groupId,  name='groupId')
flarefpp_winPlacePerc = pd.Series(result_flarefpp["scores"], name='winPlacePerc')
df_flarefpp = pd.concat([flarefpp_groupId, flarefpp_winPlacePerc], axis=1)

flaretpp_groupId = pd.Series(test_flaretpp_groupId,  name='groupId')
flaretpp_winPlacePerc = pd.Series(result_flaretpp["scores"], name='winPlacePerc')
df_flaretpp = pd.concat([flaretpp_groupId, flaretpp_winPlacePerc], axis=1)

crashfpp_groupId = pd.Series(test_crashfpp_groupId,  name='groupId')
crashfpp_winPlacePerc = pd.Series(result_crashfpp["scores"], name='winPlacePerc')
df_crashfpp = pd.concat([crashfpp_groupId, crashfpp_winPlacePerc], axis=1)

crashtpp_groupId = pd.Series(test_crashtpp_groupId,  name='groupId')
crashtpp_winPlacePerc = pd.Series(result_crashtpp["scores"], name='winPlacePerc')
df_crashtpp = pd.concat([crashtpp_groupId, crashtpp_winPlacePerc], axis=1)
df2 = pd.concat([df_solo, df_duo, df_squad, df_solo_fpp, df_duo_fpp, df_squad_fpp, df_flarefpp, df_flaretpp, df_crashfpp, df_crashtpp], axis=0)
Id = pd.Series(test_id,  name='Id')
groupId = pd.Series(test_groupId, name='groupId')
df1 = pd.concat([Id, groupId], axis=1)
cols = ['groupId']
df = df1.join(df2.set_index(cols), on=cols)
df = df.drop(['groupId'], axis=1)
df.to_csv("predictions.csv", index=False)
