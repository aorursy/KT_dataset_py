import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns

%matplotlib inline
import os
print(os.listdir("../input"))
df_train = pd.read_csv("../input/train_V2.csv")
df_test = pd.read_csv("../input/test_V2.csv")
df_train.head()
df_train['matchType'].value_counts()
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
df_train.hist(bins = 50, figsize = (20,15))
corr_matrix = df_train.corr()
sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
corr_matrix["winPlacePerc"].sort_values(ascending=False)
sns.boxplot(x ='vehicleDestroys', y='winPlacePerc',data = df_train)
sns.boxplot(x ='teamKills', y='winPlacePerc',data = df_train)
sns.boxplot(x ='roadKills', y='winPlacePerc',data = df_train)
sns.boxplot(x ='assists', y='winPlacePerc',data = df_train)
df_train.dropna(subset=['winPlacePerc'], inplace=True)
df_train = df_train.drop(['Id', 'groupId', 'matchId'], axis=1)
df_test = df_test.drop(['Id','groupId', 'matchId'], axis=1)
df_train_solo = df_train[df_train.matchType == 'solo']
df_train_duo = df_train[df_train.matchType == 'duo']
df_train_squad = df_train[df_train.matchType == 'squad']
df_train_solo_fpp = df_train[df_train.matchType == 'solo-fpp']
df_train_duo_fpp = df_train[df_train.matchType == 'duo-fpp']
df_train_squad_fpp = df_train[df_train.matchType == 'squad-fpp']
df_train_flarefpp = df_train[df_train.matchType == 'flarefpp']
df_train_flaretpp = df_train[df_train.matchType == 'flaretpp']
df_train_crashfpp = df_train[df_train.matchType == 'crashfpp']
df_train_crashtpp = df_train[df_train.matchType == 'flaretpp']

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

label_solo = df_train_solo['winPlacePerc']
label_duo = df_train_duo['winPlacePerc']
label_squad = df_train_squad['winPlacePerc']
label_solo_fpp = df_train_solo_fpp['winPlacePerc']
label_duo_fpp = df_train_duo_fpp['winPlacePerc']
label_squad_fpp = df_train_squad_fpp['winPlacePerc']
label_flarefpp = df_train_flarefpp['winPlacePerc']
label_flaretpp = df_train_flaretpp['winPlacePerc']
label_crashfpp = df_train_crashfpp['winPlacePerc']
label_crashtpp = df_train_crashtpp['winPlacePerc']

df_train_solo = df_train_solo.drop(['winPlacePerc'], axis=1)
df_train_duo = df_train_duo.drop(['winPlacePerc'], axis=1)
df_train_squad = df_train_squad.drop(['winPlacePerc'], axis=1)
df_train_solo_fpp = df_train_solo_fpp.drop(['winPlacePerc'], axis=1)
df_train_duo_fpp = df_train_duo_fpp.drop(['winPlacePerc'], axis=1)
df_train_squad_fpp = df_train_squad_fpp.drop(['winPlacePerc'], axis=1)
df_train_flarefpp = df_train_flarefpp.drop(['winPlacePerc'], axis=1)
df_train_flaretpp = df_train_flaretpp.drop(['winPlacePerc'], axis=1)
df_train_crashfpp = df_train_crashfpp.drop(['winPlacePerc'], axis=1)
df_train_crashtpp = df_train_crashtpp.drop(['winPlacePerc'], axis=1)
x_train_solo = df_train_solo.values
x_train_duo = df_train_duo.values
x_train_squad = df_train_squad.values
x_train_solo_fpp = df_train_solo_fpp.values
x_train_duo_fpp = df_train_duo_fpp.values
x_train_squad_fpp = df_train_squad_fpp.values
x_train_flarefpp = df_train_flarefpp.values
x_train_flaretpp = df_train_flaretpp.values
x_train_crashfpp = df_train_crashfpp.values
x_train_crashtpp = df_train_crashtpp.values

x_test_solo = df_test_solo.values
x_test_duo = df_test_duo.values
x_test_squad = df_test_squad.values
x_test_solo_fpp = df_test_solo_fpp.values
x_test_duo_fpp = df_test_duo_fpp.values
x_test_squad_fpp = df_test_squad_fpp.values
x_test_flarefpp = df_test_flarefpp.values
x_test_flaretpp = df_test_flaretpp.values
x_test_crashfpp = df_test_crashfpp.values
x_test_crashtpp = df_test_crashtpp.values

y_true_solo = label_solo.values
y_true_duo = label_duo.values
y_true_squad = label_squad.values
y_true_solo_fpp = label_solo_fpp.values
y_true_duo_fpp = label_duo_fpp.values
y_true_squad_fpp = label_squad_fpp.values
y_true_flarefpp = label_flarefpp.values
y_true_flaretpp = label_flaretpp.values
y_true_crashfpp = label_crashfpp.values
y_true_crashtpp = label_crashtpp.values
# Analysing solo

df_train_solo = df_train_solo.drop(['DBNOs','revives'], axis=1)
df_test_solo = df_test_solo.drop(['DBNOs','revives'], axis=1)
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
pca_solo = PCA(0.98) 
x_train_solo_scaled = pca_solo.fit_transform(x_train_solo_scaled)
x_test_solo_scaled = pca_solo.transform(x_test_solo_scaled)
pca_solo.explained_variance_ratio_
x_train_solo_scaled.shape
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_solo_scaled)
dnn_reg_solo = tf.contrib.learn.DNNRegressor(hidden_units=[2500,2500,2000,1500,1000,500], activation_fn = tf.nn.relu,dropout = 0.2,feature_columns=feature_columns, config=config)
dnn_reg_solo = tf.contrib.learn.SKCompat(dnn_reg_solo) # to be compatible with sklearn
dnn_reg_solo.fit(x_train_solo_scaled, y_true_solo, batch_size=50, steps=50000)
# analysing duo

df_train_duo.describe()
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
pca_duo = PCA(0.98) 
x_train_duo_scaled = pca_duo.fit_transform(x_train_duo_scaled)
x_test_duo_scaled = pca_duo.transform(x_test_duo_scaled)
pca_duo.explained_variance_ratio_
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_duo_scaled)
dnn_reg_duo = tf.contrib.learn.DNNRegressor(hidden_units=[2500,2500,2000,1500,1000,500], activation_fn = tf.nn.relu,dropout = 0.2, feature_columns=feature_columns, config=config)
dnn_reg_duo = tf.contrib.learn.SKCompat(dnn_reg_duo) # to be compatible with sklearn
dnn_reg_duo.fit(x_train_duo_scaled, y_true_duo, batch_size=50, steps=50000)
# Analysing squad

df_train_squad.describe()
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
pca_squad = PCA(0.98) 
x_train_squad_scaled = pca_squad.fit_transform(x_train_squad_scaled)
x_test_squad_scaled = pca_squad.transform(x_test_squad_scaled)
pca_squad.explained_variance_ratio_
# neural network model

config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_squad_scaled)
dnn_reg_squad = tf.contrib.learn.DNNRegressor(hidden_units=[2500,2500,2000,1500,1000,500], activation_fn = tf.nn.relu,dropout = 0.2, feature_columns=feature_columns, config=config)
dnn_reg_squad = tf.contrib.learn.SKCompat(dnn_reg_squad) # to be compatible with sklearn
dnn_reg_squad.fit(x_train_squad_scaled, y_true_squad, batch_size=50, steps=50000)
# Analysing solo-fpp

df_train_solo_fpp = df_train_solo_fpp.drop(['DBNOs','revives'], axis=1)
df_test_solo_fpp = df_test_solo_fpp.drop(['DBNOs','revives'], axis=1)
# correlation heat map

corr_solo_fpp = df_train_solo_fpp.corr()
sns.heatmap(corr_solo_fpp, xticklabels = corr_solo_fpp.columns, yticklabels = corr_solo_fpp.columns)
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
config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_solo_fpp_scaled)
dnn_reg_solo_fpp = tf.contrib.learn.DNNRegressor(hidden_units=[2500,2500,2000,1500,1000,500], activation_fn = tf.nn.relu,dropout = 0.2, feature_columns=feature_columns, config=config)
dnn_reg_solo_fpp = tf.contrib.learn.SKCompat(dnn_reg_solo_fpp) # to be compatible with sklearn
dnn_reg_solo_fpp.fit(x_train_solo_fpp_scaled, y_true_solo_fpp, batch_size=50, steps=50000)
# Analysing duo-fpp

df_train_duo_fpp.describe()
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
config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_duo_fpp_scaled)
dnn_reg_duo_fpp = tf.contrib.learn.DNNRegressor(hidden_units=[2500,2500,2000,1500,1000,500], activation_fn = tf.nn.relu,dropout = 0.2, feature_columns=feature_columns, config=config)
dnn_reg_duo_fpp = tf.contrib.learn.SKCompat(dnn_reg_duo_fpp) # to be compatible with sklearn
dnn_reg_duo_fpp.fit(x_train_duo_fpp_scaled, y_true_duo_fpp, batch_size=50, steps=50000)
# Analysing squad-fpp

df_train_squad_fpp.describe()
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
config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_squad_fpp_scaled)
dnn_reg_squad_fpp = tf.contrib.learn.DNNRegressor(hidden_units=[2500,2500,2000,1500,1000,500], activation_fn = tf.nn.relu,dropout = 0.2, feature_columns=feature_columns, config=config)
dnn_reg_squad_fpp = tf.contrib.learn.SKCompat(dnn_reg_squad_fpp) # to be compatible with sklearn
dnn_reg_squad_fpp.fit(x_train_squad_fpp_scaled, y_true_squad_fpp, batch_size=50, steps=50000)
# Analysing flarefpp

df_train_flarefpp = df_train_flarefpp.drop(['killPoints','rankPoints','winPoints'], axis=1)
df_test_flarefpp = df_test_flarefpp.drop(['killPoints','rankPoints','winPoints'], axis=1)
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
config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_flarefpp_scaled)
dnn_reg_flarefpp = tf.contrib.learn.DNNRegressor(hidden_units=[2500,2500,2000,1500,1000,500], activation_fn = tf.nn.relu,dropout = 0.2, feature_columns=feature_columns, config=config)
dnn_reg_flarefpp = tf.contrib.learn.SKCompat(dnn_reg_flarefpp) # to be compatible with sklearn
dnn_reg_flarefpp.fit(x_train_flarefpp_scaled, y_true_flarefpp, batch_size=1, steps=5000)
# Analysing flaretpp

df_train_flaretpp.describe()
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
config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_flaretpp_scaled)
dnn_reg_flaretpp = tf.contrib.learn.DNNRegressor(hidden_units=[2500,2500,2000,1500,1000,500], activation_fn = tf.nn.relu,dropout = 0.2, feature_columns=feature_columns, config=config)
dnn_reg_flaretpp = tf.contrib.learn.SKCompat(dnn_reg_flaretpp) # to be compatible with sklearn
dnn_reg_flaretpp.fit(x_train_flaretpp_scaled, y_true_flaretpp, batch_size=1, steps=20000)
# Analysing crashfpp

df_train_crashfpp = df_train_crashfpp.drop(['killPoints','rankPoints','winPoints'], axis=1)
df_test_crashfpp = df_test_crashfpp.drop(['killPoints','rankPoints','winPoints'], axis=1)
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
config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_crashfpp_scaled)
dnn_reg_crashfpp = tf.contrib.learn.DNNRegressor(hidden_units=[2500,2500,2000,1500,1000,500], activation_fn = tf.nn.relu,dropout = 0.2, feature_columns=feature_columns, config=config)
dnn_reg_crashfpp = tf.contrib.learn.SKCompat(dnn_reg_crashfpp) # to be compatible with sklearn
dnn_reg_crashfpp.fit(x_train_crashfpp_scaled, y_true_crashfpp, batch_size=1, steps=50000)
# Analysing crashtpp

df_train_crashtpp.describe()
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
config = tf.contrib.learn.RunConfig(tf_random_seed=42) 

feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(x_train_crashtpp_scaled)
dnn_reg_crashtpp = tf.contrib.learn.DNNRegressor(hidden_units=[2500,2500,2000,1500,1000,500], activation_fn = tf.nn.relu,dropout = 0.2, feature_columns=feature_columns, config=config)
dnn_reg_crashtpp = tf.contrib.learn.SKCompat(dnn_reg_crashtpp) # to be compatible with sklearn
dnn_reg_crashtpp.fit(x_train_crashtpp_scaled, y_true_crashtpp, batch_size=1, steps=20000)
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
df_test_solo["winPlacePerc"] = result_solo['scores']
df_test_duo["winPlacePerc"] = result_duo['scores']
df_test_squad["winPlacePerc"] = result_squad['scores']
df_test_solo_fpp["winPlacePerc"] = result_solo_fpp['scores']
df_test_duo_fpp["winPlacePerc"] = result_duo_fpp['scores']
df_test_squad_fpp["winPlacePerc"] = result_squad_fpp['scores']
df_test_flarefpp["winPlacePerc"] = result_flarefpp['scores']
df_test_flaretpp["winPlacePerc"] = result_flaretpp['scores']
df_test_crashfpp["winPlacePerc"] = result_crashfpp['scores']
df_test_crashtpp["winPlacePerc"] = result_crashtpp['scores']
frames = [df_test_solo, df_test_duo, df_test_squad, df_test_solo_fpp, df_test_duo_fpp, df_test_squad_fpp, df_test_flarefpp, df_test_flaretpp, df_test_crashfpp, df_test_crashtpp]
result = pd.concat(frames)
result.sort_index(inplace=True)
result
df_sample = pd.read_csv("../input/sample_submission_V2.csv")
df_sample["winPlacePerc"] = result["winPlacePerc"]
df_sample["winPlacePerc"].dtype
df_sample.to_csv("predictions.csv", index=False)

