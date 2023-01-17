import pandas as pd

import numpy as np

from sklearn.metrics import log_loss

from sklearn.preprocessing import MinMaxScaler



import tensorflow as tf

from tensorflow.keras import layers as L

from tensorflow.keras.callbacks import *

from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential
train_df = pd.read_csv('../input/lish-moa/train_features.csv')

test_df = pd.read_csv('../input/lish-moa/test_features.csv')

train_target_df = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

sample_sub = pd.read_csv('../input/lish-moa/sample_submission.csv')



target_cols = train_target_df.columns[1:]

N_TARGETS = len(target_cols)
train_df
train_df.at[train_df['cp_type'].str.contains('ctl_vehicle'),train_df.filter(regex='-.*').columns] = 0.0



test_df.at[test_df['cp_type'].str.contains('ctl_vehicle'),test_df.filter(regex='-.*').columns] = 0.0
def preprocess_df(df, target=False):

    

    

    scaler = MinMaxScaler()

    df["cp_time"]=scaler.fit_transform(df["cp_time"].values.reshape(-1, 1))

    

    df["cp_dose"]=(df["cp_dose"]=="D1").astype(int)

    df["cp_type"]=(df["cp_type"]=="trt_cp").astype(int)

    

    return df
x_train = preprocess_df(train_df.drop(["sig_id"], axis=1))

y_train = train_target_df.drop(["sig_id"], axis=1)



x_test = preprocess_df(test_df.drop(["sig_id"], axis=1))
import seaborn as sns

import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 8))

# sns.heatmap(x_train[:30].corr())

# plt.title('Pairwise correlations of the first 50 gene features')

# plt.show()

#Correlation matrix for Variables

cell=train_df.loc[:, train_df.columns.str.startswith('c-')]

corr = cell.corr(method='pearson')

# corr

f, ax = plt.subplots(figsize=(25, 25))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.triu(np.ones_like(corr, dtype=np.bool))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.8, cbar_kws={"shrink": .5})



ax = sns.heatmap(corr,linewidths=0.8,cmap=cmap)
# Capping outliers

def cap_outliers(col):

    col[col>3]=3

    col[col<-3]=-3

    return col
from scipy import stats #Outlier Analysis & Removal



#Filtering all the numeric columns

numcols=x_train._get_numeric_data().columns

all_data_num=x_train.loc[:,numcols]

all_data_num=x_train.iloc[:,1:]



#z=np.abs(stats.zscore(all_data_num['g-0']))

#Calculate Z Scores for all the variables. 

all_data_num=x_train.apply(stats.zscore)



#Cap the outliers

all_data_num=all_data_num.apply(cap_outliers)

#all_data_num.describe()

#z
def pca_application(df,n_components,pattern):

    df_p=df.loc[:, df.columns.str.startswith(pattern)]

    x = StandardScaler().fit_transform(df_p)

    pca = PCA(n_components=n_components)

    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data = principalComponents)

    return principalDf,pca
from sklearn.preprocessing import StandardScaler #Scaling variables

from sklearn.decomposition import PCA #Dimensionality Reduction



#Calculate principal components separately for GE & CV columns

principalDf_g,pca_g=pca_application(x_train,200,'g-')

principalDf_c,pca_c=pca_application(x_train,30,'c-')



#principalDf_g
plt.plot(np.cumsum(pca_g.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.title('Cumulative Explained Variance for Gene Expression Variable PCAs')
plt.plot(np.cumsum(pca_c.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.title('Cumulative Explained Variance for Cell Viability Variable PCAs')
corr_matrix = x_train.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]
len(to_drop)
corr_matrix = y_train.corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop_y = [column for column in upper.columns if any(upper[column] > 0.90)]
len(to_drop_y)
xx_train = x_train.drop(x_train[to_drop], axis=1)

yy_train = y_train.drop(y_train[to_drop_y], axis=1)

import tensorflow_addons as tfa
def get_keras_model(input_dim=875, output_dim=206):

    

    model = Sequential()

    model.add(tfa.layers.WeightNormalization((L.Dense(512, input_dim=875, activation="elu"))))

    model.add(L.BatchNormalization())

    model.add(L.Dropout(0.5))

    model.add(tfa.layers.WeightNormalization(L.Dense(256, activation="elu")))

    model.add(L.BatchNormalization())

    model.add(L.Dropout(0.3))

    model.add(tfa.layers.WeightNormalization(L.Dense(256, activation="elu")))

    model.add(L.BatchNormalization())

    model.add(L.Dropout(0.3))

    model.add(tfa.layers.WeightNormalization(L.Dense(256, activation="elu")))

    model.add(L.BatchNormalization())

    model.add(L.Dropout(0.3))

    model.add(tfa.layers.WeightNormalization(L.Dense(206, activation="sigmoid")))

    return model
model = get_keras_model()

# model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),

             loss="binary_crossentropy",

             metrics=["accuracy"])
def multi_log_loss(y_true, y_pred):

    losses = []

    for col in y_true.columns:

        losses.append(log_loss(y_true.loc[:, col], y_pred.loc[:, col]))

    return np.mean(losses)
hist = model.fit(x_train, y_train, epochs=80)
loss = pd.DataFrame({"loss": hist.history['accuracy'], "val_loss": hist.history['loss'] })
# hist.history
loss.plot()
ps = model.predict(x_train)
ps_df = y_train.copy()

ps_df.iloc[:, : ] = ps



tr_score = multi_log_loss(y_train, ps_df)



print(f"Train score: {tr_score}")
test_preds = sample_sub.copy()

test_preds[target_cols] = 0



test_preds.loc[:,target_cols] = model.predict(x_test)



test_preds.loc[x_test['cp_type'] == 0, target_cols] = 0

test_preds.to_csv('submission.csv', index=False)