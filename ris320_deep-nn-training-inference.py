# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

import statsmodels.api as sm

import pylab

from scipy import stats

import tensorflow as tf

from sklearn.model_selection import train_test_split,KFold

from sklearn.metrics import log_loss

from sklearn.preprocessing import MinMaxScaler,PowerTransformer,StandardScaler

from sklearn.decomposition import PCA

import tensorflow_addons as tfa

# from sklearn.components import PCA

# from keras.utils import to_categorical

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_features=pd.read_csv("../input/lish-moa/train_features.csv")

train_targets=pd.read_csv("../input/lish-moa/train_targets_scored.csv")

test_features=pd.read_csv("../input/lish-moa/test_features.csv")

submission=pd.read_csv("../input/lish-moa/sample_submission.csv")
train_features.head(10)
test_features.head(10)
# Checking the Shape of Data

train_features.shape
test_features.shape
train_targets.head(10)
submission.head(10)
# Checking the Missing Values

train_features.isnull().sum()/len(train_features)
# Missing HeatMap

plt.figure(figsize=(12,8))

sns.heatmap(train_features.isnull(),cbar=False).set_title("Missing Values")
# Checking Feature Correlation

plt.figure(figsize=(12,7))

sns.heatmap(train_features[:10].corr())
# Checking CP Type Distribution

y=Counter(train_features.cp_type).most_common(train_features.cp_type.nunique())

cp_type=[i[0] for i in y]

cp_count=[i[1] for i in y]

plt.figure(figsize=(12,7))

sns.barplot(cp_count,cp_type).set_title("CP Type Distribution")

plt.xlabel("CP Count")

plt.ylabel("CP Type")
# Checking CP Dose Distribution

y=Counter(train_features.cp_dose).most_common(train_features.cp_dose.nunique())

cp_dose=[i[0] for i in y]

cp_count=[i[1] for i in y]

plt.figure(figsize=(12,7))

sns.barplot(cp_count,cp_dose).set_title("CP Dose Distribution")

plt.xlabel("Dose Count")

plt.ylabel("Dose Type")
plt.figure(figsize=(12,10))

plt.subplot(2,2,1)

sns.distplot(train_features['g-0'])

plt.subplot(2,2,2)

sns.distplot(train_features['g-7'])



plt.subplot(2,2,3)

sns.distplot(train_features['c-0'])

plt.subplot(2,2,4)

sns.distplot(train_features['c-7'])
# Q-Q Plot

plt.figure(figsize=(12,10))

sm.qqplot(train_features['g-0'], line='45')
y=Counter(train_features.cp_time).most_common(train_features.cp_time.nunique())

cp_time=[i[0] for i in y]

cp_count=[i[1] for i in y]

plt.figure(figsize=(12,7))

sns.barplot(cp_count,cp_time).set_title("CP Time Distribution")
train_targets
y=Counter(train_targets["5-alpha_reductase_inhibitor"]).most_common(train_targets["5-alpha_reductase_inhibitor"].nunique())

name=[i[0] for i in y]

count=[i[1] for i in y]

plt.figure(figsize=(12,7))

sns.barplot(name,count).set_title("5-alpha_reductase_inhibitor Distribution")
train_features=pd.get_dummies(train_features,columns=['cp_type'])

test_features=pd.get_dummies(test_features,columns=['cp_type'])

cp_dose_enc={'D1':0,'D2':1}

train_features['cp_dose']=train_features['cp_dose'].replace(cp_dose_enc)



test_features['cp_dose']=test_features['cp_dose'].replace(cp_dose_enc)
# Dropping Columns

train_features.drop(columns=['sig_id','cp_time','cp_type_ctl_vehicle'],inplace=True)

train_targets.drop(columns=['sig_id'],inplace=True)



test_features.drop(columns=['sig_id','cp_type_ctl_vehicle','cp_time'],inplace=True)
train_features.head(10)
trt_cp=train_features["cp_type_trt_cp"]

train_features.drop(labels=['cp_type_trt_cp'], axis=1,inplace = True)

train_features.insert(0, 'cp_type_trt_cp', trt_cp)
trt_cp=test_features["cp_type_trt_cp"]

test_features.drop(labels=['cp_type_trt_cp'], axis=1,inplace = True)

test_features.insert(0, 'cp_type_trt_cp', trt_cp)
train_features.head(10)
test_features
# train_features.iloc[:,0:2]=train_features.iloc[:,0:2].astype('category')

# test_features.iloc[:,0:2]=test_features.iloc[:,0:2].astype('category')
# train_features.dtypes
# # # Scaling the Features

# scaler=MinMaxScaler()

# num_cols = list(train_features.select_dtypes(include=['float64']).columns)

# train_features[num_cols] = scaler.fit_transform(train_features[num_cols])



# test_features[num_cols]=scaler.transform(test_features[num_cols])
# # Variance VS Components

# pca = PCA().fit(train_features)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))

# plt.xlabel('number of components')

# plt.ylabel('cumulative explained variance')
# # PCA

# pca=PCA(n_components=400)

# train_components=pca.fit_transform(train_features)

# test_components=pca.transform(test_features)
# Train Test Split

# x_train,x_val,y_train,y_val=train_test_split(train_components,train_targets,test_size=0.20)
# # Transforming Skewed Data

# pt=PowerTransformer()

# pt.fit(x_train.iloc[:,2:])

# x_train_pt=pd.DataFrame(pt.transform(x_train.iloc[:,2:]),columns=x_train.iloc[:,2:].columns).set_index(x_train.index)



# x_val_pt=pd.DataFrame(pt.transform(x_val.iloc[:,2:]),columns=x_val.iloc[:,2:].columns).set_index(x_val.index)



# test_features_pt=pd.DataFrame(pt.transform(test_features.iloc[:,2:]),columns=test_features.iloc[:,2:].columns).set_index(test_features.index)
# x_train.drop(columns=x_train.iloc[:,2:],inplace=True)

# x_val.drop(columns=x_val.iloc[:,2:],inplace=True)



# test_features.drop(columns=test_features.iloc[:,2:],inplace=True)



# x_train=pd.concat([x_train,x_train_pt],axis=1)

# x_val=pd.concat([x_val,x_val_pt],axis=1)



# test_features=pd.concat([test_features,test_features_pt],axis=1)
# x_train.skew(axis=0)
LR=0.001

BATCH_SIZE=16

EPOCHS=30


def build_model():

    inp=tf.keras.layers.Input(shape=(train_features.shape[1],))



    x=tfa.layers.WeightNormalization(tf.keras.layers.Dense(128,activation='relu'))(inp)

    x=tf.keras.layers.BatchNormalization()(x)

    x=tf.keras.layers.Dropout(0.5)(x)



    x=tfa.layers.WeightNormalization(tf.keras.layers.Dense(64,activation='relu'))(inp)

    x=tf.keras.layers.BatchNormalization()(x)

    x=tf.keras.layers.Dropout(0.25)(x)



    x=tfa.layers.WeightNormalization(tf.keras.layers.Dense(32,activation='relu'))(inp)

    x=tf.keras.layers.BatchNormalization()(x)

    x=tf.keras.layers.Dropout(0.25)(x)



    out=tf.keras.layers.Dense(train_targets.shape[1],activation="sigmoid")(x)



    model=tf.keras.models.Model(inputs=inp,outputs=out)

    

    return model
save_best=tf.keras.callbacks.ModelCheckpoint(filepath="best_model.h5",monitor='val_loss',save_best_only=True)

reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.6,min_lr_rate=0.000000001)

early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)
# # Custom Metric

# def log_loss(y_true,y_pred):

# #     y_true = tf.cast(y_true,tf.float32)

# #     y_pred = tf.cast(y_pred,tf.float32)

#     loss = ((y_true*tf.math.log(y_pred))+((1.0-y_true)*tf.math.log(1.0-y_pred)))

    

#     return loss

# Compiling the Model

model=build_model()

opt=tf.keras.optimizers.Adam(learning_rate=LR)

model.compile(optimizer=opt,loss="binary_crossentropy",metrics=[tf.metrics.AUC()])
# Model Summary

model.summary()
train_features=train_features.values
train_targets=train_targets.values
# K-fold Cross Validation model Training and Evaludation

# Define per-fold score containers <-- these are new

acc_per_fold = []

loss_per_fold = []

kfold=KFold(n_splits=10)

fold_no = 1



for train_index, val_index in kfold.split(train_features, train_targets):



    # Define the model architecture

    model=build_model()

    # Compiling the Model

    

    opt=tf.keras.optimizers.Adam(learning_rate=LR)

    

    model.compile(optimizer=opt,loss="binary_crossentropy",metrics=[tf.keras.metrics.AUC()])



    # Generate a print

    print('------------------------------------------------------------------------')

    print(f'Training for fold {fold_no} ...')



  # Fit data to model

    history = model.fit(train_features[train_index], train_targets[train_index],

              batch_size=BATCH_SIZE,

              epochs=EPOCHS,callbacks=[reduce_lr])





    # Generate generalization metrics

    scores = model.evaluate(train_features[val_index], train_targets[val_index], verbose=0)

    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

    acc_per_fold.append(scores[1] * 100)

    loss_per_fold.append(scores[0])



    model.save(f'model_{fold_no}.h5')

    

    # Increase fold number

    fold_no = fold_no + 1

    

    # == Provide average scores ==

    print('------------------------------------------------------------------------')

    print('Score per fold')

    for i in range(0, len(acc_per_fold)):

      print('------------------------------------------------------------------------')

      print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')

    print('------------------------------------------------------------------------')

    print('Average scores for all folds:')

    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')

    print(f'> Loss: {np.mean(loss_per_fold)}')

    print('------------------------------------------------------------------------')





# model.fit(x_train,y_train,batch_size=BATCH_SIZE,validation_data=(x_val,y_val),callbacks=[save_best,reduce_lr,early_stopping],epochs=EPOCHS)
# Loading the Model

model=tf.keras.models.load_model("./model_10.h5")
# Submitting the Predictions

sig_id=submission['sig_id']

submission.drop(columns=['sig_id'],inplace=True)

predictions=pd.DataFrame(model.predict(test_features),columns=submission.columns)

predictions.insert(0, 'sig_id', sig_id)

predictions.to_csv("submission.csv",index=False)