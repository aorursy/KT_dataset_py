import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import matplotlib.pyplot as plt

from keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization, Input

from keras.models import Sequential, save_model

from keras.utils import np_utils

import tensorflow as tf

from keras.callbacks import EarlyStopping

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

from imblearn.combine import SMOTEENN

from sklearn.decomposition import PCA

from sklearn.utils import class_weight

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import log_loss
import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
moa_train_feat = pd.read_csv('../input/lish-moa/train_features.csv')

moa_train_targ_NS = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

moa_train_targ_S = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

moa_test = pd.read_csv('../input/lish-moa/test_features.csv')
moa_train_feat
####Drop ID's

moa_train_targ_S=moa_train_targ_S.drop(moa_train_targ_S.columns[0],axis=1)

moa_train_targ_S.head()
####Drop ID's

moa_train_feat =moa_train_feat.drop(moa_train_feat.columns[0],axis=1)

moa_train_feat.head() 
moa_train_targ_S.dtypes
####Drop ID's but Save Test ID's

test_id=moa_test['sig_id']

moa_test=moa_test.drop(moa_test.columns[0],axis=1)

moa_test.head()
test_id
moa_train_feat=moa_train_feat[moa_train_feat['cp_type'] != 'ctl_vehicle']

indexs_list2=moa_train_feat.index.values.tolist() 
moa_train_targ_S=moa_train_targ_S.iloc[indexs_list2]

#df = moa_train_targ_S.sum(axis=1)
#df_no_label=(df==0)

#df_no_label
####Make List of Indexs

#moa_train_targ_S[df_no_label]

#indexs_list2=moa_train_targ_S[df_no_label].index.values.tolist() 
#moa_train_targ_S = moa_train_targ_S.drop(indexs_list2) 
#moa_train_feat = moa_train_feat.drop(indexs_list2) 
#moa_train_targ_S
#moa_train_feat
####One Hot Code Train Columns: cp_type and cp_dose

dummies=moa_train_feat[['cp_type','cp_dose']]

cat_columns = ['cp_type','cp_dose']
dummies2=pd.get_dummies(dummies, prefix_sep="_",

                              columns=cat_columns)

dummies2
moa_train_feat['cp_type']=dummies2['cp_type_trt_cp']

moa_train_feat['cp_dose']=dummies2['cp_dose_D1']
###Remove Low Variance Features

print(moa_train_feat.shape)

from sklearn import feature_selection as fs

## Define the variance threhold and fit the threshold to the feature array. 

sel = fs.VarianceThreshold(threshold=.7)

moa_train_feat_vt = sel.fit_transform(moa_train_feat)



## Print the support and shape for the transformed features

print(sel.get_support())

print(moa_train_feat.shape)



moa_train_feat=moa_train_feat[moa_train_feat.columns[sel.get_support(indices=True)]] 
one_hot_moa_train_feat=moa_train_feat.copy()
#dummies2=dummies2.multiply(dummies2['cp_type_trt_cp'], axis=0)

#dummies2=dummies2.reset_index()
###Remove Categorical Columns

#moa_train_feat=moa_train_feat.drop(['cp_type','cp_dose'],axis=1)
###Insert Dummies

#dummies2=dummies2[['cp_dose_D1','cp_dose_D2']]

#one_hot_moa_train_feat=dummies2.join(moa_train_feat)

#one_hot_moa_train_feat
####One Hot Code Columns: cp_type and cp_dose

dummies3=moa_test[['cp_type','cp_dose']]
dummies4=pd.get_dummies(dummies3, prefix_sep="_",

                              columns=cat_columns)

dummies4
moa_test['cp_type']=dummies4['cp_type_trt_cp']

moa_test['cp_dose']=dummies4['cp_dose_D1']

test_control_group=moa_test['cp_type'] == 0
#dummies4=dummies4.multiply(dummies4['cp_type_trt_cp'], axis=0)

#dummies4
#moa_test=moa_test.drop(['cp_type','cp_dose','cp_time'],axis=1)

#moa_test
top_feats3=list(sel.get_support(indices=True))
sel.get_support(indices=True)
one_hot_moa_test
###Remove Same Variance Threshold Columns from Test Set

#moa_test=moa_test[moa_test.columns[sel.get_support(indices=True)]] 

#moa_test

one_hot_moa_test=moa_test.iloc[:, top_feats3]

one_hot_moa_test
#dummies4=dummies4[['cp_dose_D1','cp_dose_D2']]

#one_hot_moa_test=dummies4.join(moa_test)

#one_hot_moa_test
combined_x=one_hot_moa_train_feat.copy()
combined_y=moa_train_targ_S.copy()
filter_col_g = [col for col in combined_x if col.startswith('g-')]

genes=combined_x[filter_col_g]

genes.head()
filter_col_c = [col for col in combined_x if col.startswith('c-')]

cells=combined_x[filter_col_c]

cells.head()
filter_col_c_test = [col for col in one_hot_moa_test if col.startswith('c-')]

cells_test=one_hot_moa_test[filter_col_c_test]

cells_test.head()
filter_col_g_test = [col for col in one_hot_moa_test if col.startswith('g-')]

genes_test=one_hot_moa_test[filter_col_g_test]

genes_test.head()
###Add PCA Features###

pca_c = PCA(.9)

pca_g = PCA(.9)



#fit PCA on Training Set

pca_c.fit(cells)

pca_g.fit(genes)



### Apply PCA Mapping to Training and Test Set: Converts to a np.array

pca_cells_train = pca_c.transform(cells)

pca_genes_train = pca_g.transform(genes)

pca_cells_test = pca_c.transform(cells_test)

pca_genes_test = pca_g.transform(genes_test)



#####Create Dataframe of PCA Features

PCA_g_train=pd.DataFrame(pca_genes_train)

PCA_c_train=pd.DataFrame(pca_cells_train)

PCA_g_test=pd.DataFrame(pca_genes_test)

PCA_c_test=pd.DataFrame(pca_cells_test)
PCA_g_train = PCA_g_train.reset_index()

del PCA_g_train['index']



PCA_c_train = PCA_c_train.reset_index()

del PCA_c_train['index']



PCA_g_test = PCA_g_test.reset_index()

del PCA_g_test['index']



PCA_c_test = PCA_c_test.reset_index()

del PCA_c_test['index']

print(PCA_g_train.shape)

print(PCA_c_train.shape)

print(PCA_g_test.shape)

print(PCA_c_test.shape)

print(one_hot_moa_test.shape)

print(combined_x.shape)
PCA_train=pd.merge(PCA_g_train, PCA_c_train,right_index=True, left_index=True)

PCA_test=pd.merge(PCA_g_test, PCA_c_test,right_index=True, left_index=True)
one_hot_moa_test = one_hot_moa_test.reset_index()

del one_hot_moa_test['index']



combined_x = combined_x.reset_index()

del combined_x['index']
one_hot_moa_test=one_hot_moa_test.join(PCA_test)
combined_x=combined_x.join(PCA_train)
one_hot_moa_test
combined_x
#one_hot_moa_train_feat=pd.merge(PCA_g_train, PCA_c_train,right_index=True, left_index=True)

#one_hot_moa_train_feat=pd.merge(dummies_train,one_hot_moa_train_feat,right_index=True, left_index=True)

#one_hot_moa_test=pd.merge(PCA_g_test, PCA_c_test,right_index=True, left_index=True)

#one_hot_moa_test=pd.merge(dummies_test,one_hot_moa_test,right_index=True, left_index=True)
#top_feats2=list(np.array(top_feats))

#top_feats3=[1]+top_feats2

#top_feats3=top_feats3[:-1]

#print(len(top_feats3))

#top_feats3=top_feats

top_feats3=list(sel.get_support(indices=True))
#combined_x=combined_x.iloc[:, top_feats3]

#combined_x
combined_y
combined_x.iloc[:, 7:]
X=np.array(combined_x)

input_dim=X.shape[1]

X.shape
X.shape[1]
Y=np.array(combined_y)

num_classes=Y.shape[1]

Y.shape
import tensorflow_addons as tfa
def create_model(num_columns):

    model = Sequential()

    model.add(Input(num_columns))

    model.add( BatchNormalization() )

    model.add( Dropout(0.5))

    model.add(Dense(units=800, kernel_initializer='glorot_uniform', activation='swish'))

    model.add( BatchNormalization() )

    model.add( Dropout(0.5))

    model.add(Dense(units=400,activation='swish'))

    model.add( BatchNormalization() )

    model.add( Dropout(0.5) )

    model.add(Dense(units=num_classes,activation='sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=3e-3)

    model.compile( optimizer=opt, loss='binary_crossentropy')

    return model

    



#metrics=[tf.keras.metrics.AUC(name='auc')]

#tf.keras.metrics.AUC(name='auc')

#tf.keras.metrics.Recall(name='recall')

#tf.keras.metrics.Precision(name='precision')
combined_y
####Get Length of Test

l=len(one_hot_moa_test)-1

l
##Empty Predictions Set

ss = combined_y.copy()

ss = ss.reset_index()

del ss['index']

ss=ss.loc[0:l,:]

ss.loc[:, combined_y.columns] = 0

ss
##Empty Validation Set

res = combined_y.copy()

res = res.reset_index ()

res.loc[:, combined_y.columns] = 0

del res['index']

res
one_hot_moa_test.values[:, top_feats3].shape
combined_y.shape
combined_x.shape
N_STARTS = 4

import tensorflow as tf

tf.random.set_seed(42)



####This iterates through starts:



for seed in range(N_STARTS):

#####This iteraties through folds n, validation indexes te, and train indexes tr:    

    for n, (tr, te) in enumerate(MultilabelStratifiedKFold(n_splits=5, random_state=seed, shuffle=True).split(combined_y, combined_y)):

        print(f'Fold {n}')

    

        model = create_model(input_dim)

        #checkpoint_path = f'repeat:{seed}_Fold:{n}.hdf5'

        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, epsilon=1e-4, mode='min')

        #cb_checkpt = ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose = 0, save_best_only = True,

          #                           save_weights_only = True, mode = 'min')

        

####This fits the model to each fold and validation set. .values avoids creating a np array:



        model.fit(combined_x.values[tr],

                  combined_y.values[tr],

                  validation_data=(combined_x.values[te], combined_y.values[te]),

                  epochs=28, batch_size=128,

                  callbacks=[reduce_lr_loss], verbose=2

                 )

        

        #model.load_weights(checkpoint_path)

####Makes predictions for each fold & seed:

        test_predict = model.predict(one_hot_moa_test.values[:, :])

        val_predict = model.predict(combined_x.values[te])

####Sum Predictions for Each Epoch     

        ss.loc[:, combined_y.columns] += test_predict

        res.loc[te, combined_y.columns] += val_predict

        print('')

        

####After all summed, Divide summed predictions by the number of starts times the number of folds:     

ss.loc[:, combined_y.columns] /= ((n+1) * N_STARTS)

res.loc[:, combined_y.columns] /= N_STARTS
####Estimate Validation Loss of Averaged Results

def metric(y_true, y_pred):

    metrics = []

    for _target in combined_y.columns:

        metrics.append(log_loss(y_true.loc[:, _target], y_pred.loc[:, _target].astype(float), labels=[0,1]))

    return np.mean(metrics)
print(f'OOF Metric: {metric(combined_y, res)}')
####Set Controls to 0

ss.loc[(test_control_group), combined_y.columns] = 0

test_id=pd.DataFrame(test_id)

ss

test_id
ss=pd.merge(test_id, ss, how='inner', left_index=True, right_index=True)

ss=pd.DataFrame(ss)

ss
ss.dtypes
###Check for nulls

pd.DataFrame(ss.isnull().sum(axis = 0)).sum()
df =pd.DataFrame(ss.describe()).max(axis=1)

df
ss.describe()
ss.to_csv('submission.csv', index=False)