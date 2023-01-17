import numpy as np

import pandas as pd

import os,time,random,tqdm

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import sklearn

from sklearn.model_selection import KFold, cross_val_score

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.utils import shuffle



# from iterstrat.ml_stratifiers import MultilabelStratifiedKFold



import tensorflow as tf

import tensorflow_addons as tfa



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_features = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

train_targets = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')

# train_targets_nonscored = pd.read_csv('/kaggle/input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

sample_submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

# cells with cp_type = ctl_vehicle have targets = 0 everywhere



indices = test_features[test_features['cp_type'] == 'ctl_vehicle'].index
oh = OneHotEncoder()



oh.fit(train_features[['cp_time','cp_dose']])

oh.get_feature_names()



train_features[oh.get_feature_names()] = oh.transform(train_features[['cp_time','cp_dose']]).todense()

test_features[oh.get_feature_names()] = oh.transform(test_features[['cp_time','cp_dose']]).todense()
train_targets.drop(['sig_id'],axis=1,inplace=True)

train_features.drop(['sig_id','cp_type','cp_time','cp_dose'],axis=1,inplace=True)

test_features.drop(['sig_id','cp_type','cp_time','cp_dose'],axis=1,inplace=True)
def make_model(inp_shape=877,out_shape=206,bias_init=None):

    

    model = tf.keras.models.Sequential([

        tf.keras.layers.BatchNormalization(input_shape=(inp_shape,)),

        tf.keras.layers.Dropout(0.2),

        tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048)),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dropout(0.5),

        tfa.layers.WeightNormalization(tf.keras.layers.Dense(2048)),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dropout(0.5),

        tfa.layers.WeightNormalization(tf.keras.layers.Dense(1024)),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dropout(0.5),

        tfa.layers.WeightNormalization(tf.keras.layers.Dense(512)),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dropout(0.5),

        tfa.layers.WeightNormalization(tf.keras.layers.Dense(512)),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(out_shape,'sigmoid',bias_initializer=tf.keras.initializers.Constant(bias_init))

    ])

    

    opt = tfa.optimizers.Lookahead(

        tf.keras.optimizers.Adam(lr=1e-4)

    )

    

    

    model.compile(

        optimizer=opt,

        loss='binary_crossentropy',

        metrics=['accuracy']

    )

    

    return model
# temp_learning_rate_schedule=CustomSchedule()

# plt.plot(temp_learning_rate_schedule(tf.range(4500, dtype=tf.float32)))

# plt.show()
n_folds = 7

cv = KFold(n_folds,random_state=33,shuffle=True)

preds = np.zeros((len(test_features),206))





scores = []



for index,(train_index, test_index) in enumerate(cv.split(train_features)):

    X_train,X_test,y_train,y_test = train_features.loc[train_index],train_features.loc[test_index],train_targets.loc[train_index],train_targets.loc[test_index]



    bias_init = np.log((np.sum(y_train.values,axis=0)+0.000001)/(len(y_train) - np.sum(y_train.values,axis=0)))

    pipeline = make_pipeline(StandardScaler())

    X_train = pipeline.fit_transform(X_train)

    X_test = pipeline.transform(X_test)





    tf.keras.backend.clear_session()

        

    model = make_model(bias_init=bias_init)

        

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=3,verbose=1)

    m_ckpt = tf.keras.callbacks.ModelCheckpoint(

            '/checkpoint', monitor='val_loss', verbose=1, save_best_only=True,

            save_weights_only=True

            )

    es = tf.keras.callbacks.EarlyStopping(patience=5)

    



        

    model.fit(X_train,y_train,batch_size=128,epochs=45,validation_data=(X_test,y_test),verbose=-1,callbacks=[es,m_ckpt,reduce_lr])

    

        

    scores.append(m_ckpt.best)

    print(f"Best fold {index} score : {scores[-1]}")

        

    model.load_weights('/checkpoint')

    p = model.predict(pipeline.transform(test_features))

    preds += p
# # Prediction on 10 the same models with different seeds



# preds = np.zeros((len(test_features),206))

# X_train,y_train = train_features,train_targets

# bias_init = np.log(np.sum(y_train.values,axis=0)/(len(y_train) - np.sum(y_train.values,axis=0)))



# pipeline = make_pipeline(StandardScaler())

# X_train = pipeline.fit_transform(X_train)



# for i in range(10):

#     tf.keras.backend.clear_session()

        

#     model = make_model(bias_init=bias_init)

    

#     model.fit(X_train,y_train,batch_size=128,epochs=20,verbose=1)



#     p = model.predict(pipeline.transform(test_features))

    

#     preds+=p
preds/=n_folds

# preds/=10.0

preds[indices][:,:] = 0

sample_submission[sample_submission.columns.tolist()[1:]] = preds

sample_submission.to_csv('submission.csv',index=False)