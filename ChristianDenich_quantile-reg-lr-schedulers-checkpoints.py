from IPython.display import HTML

HTML('<center><iframe width="560" height="315" src="https://www.youtube.com/embed/AfK9LPNj-Zo" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
import numpy as np

import random

import pandas as pd

import pydicom

import os

import matplotlib.pyplot as plt

from timeit import timeit

from tqdm import tqdm

from PIL import Image



from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold



#color

from colorama import Fore, Back, Style



import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow.keras.layers as Layers

import warnings

warnings.filterwarnings('ignore') #Ignore "future" warnings and Data-Frame-Slicing warnings.

def seed_everything(seed): 

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)
ROOT = '../input/osic-pulmonary-fibrosis-progression'



train_df = pd.read_csv(f'{ROOT}/train.csv')

print(f'Train data has {train_df.shape[0]} rows and {train_df.shape[1]} columnns and looks like this:')

train_df.sample(10)
train_unique_df = train_df.drop_duplicates(subset = ['Patient'], keep = 'first')

train_unique_df.head()
# CHECK FOR DUPLICATES & DEAL WITH THEM

# keep = False: All duplicates will be shown

dupRows_df = train_df[train_df.duplicated(subset = ['Patient', 'Weeks'], keep = False )]

dupRows_df.head()
train_df.drop_duplicates(subset=['Patient','Weeks'], keep = False, inplace = True)
print(f'So there are {dupRows_df.shape[0]} (= {dupRows_df.shape[0] / train_df.shape[0] * 100:.2f}%) duplicates.')
test_df = pd.read_csv(f'{ROOT}/test.csv')

print(f'Test data has {test_df.shape[0]} rows and {test_df.shape[1]} columnns, has no duplicates and looks like this:')

test_df.head()
## CHECK SUBMISSION FORMAT

sub_df = pd.read_csv(f"{ROOT}/sample_submission.csv")



print(f"The sample submission contains: {sub_df.shape[0]} rows and {sub_df.shape[1]} columns.")
sub_df.head()
# split Patient_Week Column and re-arrage columns

sub_df[['Patient','Weeks']] = sub_df.Patient_Week.str.split("_",expand = True)

sub_df =  sub_df[['Patient','Weeks','Confidence', 'Patient_Week']]
sub_df = sub_df.merge(test_df.drop('Weeks', axis = 1), on = "Patient")
# introduce a column to indicate the source (train/test) for the data

train_df['Source'] = 'train'

sub_df['Source'] = 'test'



data_df = train_df.append([sub_df])

data_df.reset_index(inplace = True)

data_df.head()
def get_baseline_week(df):

    # make a copy to not change original df    

    _df = df.copy()

    # ensure all Weeks values are INT and not accidentaly saved as string

    _df['Weeks'] = _df['Weeks'].astype(int)

    _df['min_week'] = _df['Weeks']

    # as test data is containing all weeks, 

    _df.loc[_df.Source == 'test','min_week'] = np.nan

    _df["min_week"] = _df.groupby('Patient')['Weeks'].transform('min')

    _df['baselined_week'] = _df['Weeks'] - _df['min_week']

    

    return _df   
data_df = get_baseline_week(data_df)

data_df.head()
def get_baseline_FVC_old(df):

    # copy the DF to not in-place change the original one

    _df = df.copy()

    # get only the rows containing the baseline (= min_weeks) and therefore the baseline FVC

    baseline = _df.loc[_df.Weeks == _df.min_week]

    baseline = baseline[['Patient','FVC']].copy()

    baseline.columns = ['Patient','base_FVC']      

    

    # fill the df with the baseline FVC values

    for idx in _df.index:

        patient_id = _df.at[idx,'Patient']

        _df.at[idx,'base_FVC'] = baseline.loc[baseline.Patient == patient_id, 'base_FVC'].iloc[0]

    _df.drop(['min_week'], axis = 1)

    

    return _df
def get_baseline_FVC(df):

    # same as above

    _df = df.copy()

    base = _df.loc[_df.Weeks == _df.min_week]

    base = base[['Patient','FVC']].copy()

    base.columns = ['Patient','base_FVC']

    

    # add a row which contains the cumulated sum of rows for each patient

    base['nb'] = 1

    base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

    

    # drop all except the first row for each patient (= unique rows!), containing the min_week

    base = base[base.nb == 1]

    base.drop('nb', axis = 1, inplace = True)

    

    # merge the rows containing the base_FVC on the original _df

    _df = _df.merge(base, on = 'Patient', how = 'left')    

    _df.drop(['min_week'], axis = 1)

    

    return _df
def get_baseline_FVC_new(df):

    _df = (

        df

        .loc[df.Weeks == df.min_week][['Patient','FVC']]

        .rename({'FVC': 'min_FVC'}, axis=1)

        .groupby('Patient')

        .first()

        .reset_index()

    )

    

    return _df
def old_baseline_FVC():

    return get_baseline_FVC_old(data_df)

    



def baseline_FVC():

    return get_baseline_FVC(data_df)



def baseline_FVC_new():

    return get_baseline_FVC_new(data_df)

    



duration_old = timeit(old_baseline_FVC, number = 3)

duration_baseline = timeit(baseline_FVC, number = 3)

duration_new = timeit(baseline_FVC_new, number = 3)



print(f"Taking the first, old & non-vectorized approach took {duration_old / 3:.2f} sec, while the 2nd, vectorized approach only took {duration_baseline / 3:.3f} sec. That's {duration_old/duration_baseline:.0f} times faster!" )

print(f"Taking the 3rd, newest, shortest and cleanest approach took {duration_new / 3:.3f} sec. That's {duration_old/duration_new:.0f} (!!) times faster!" )
data_df = get_baseline_FVC(data_df)

data_df.head()
# import the necessary Encoders & Transformers

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.compose import ColumnTransformer



# define which attributes shall not be transformed, are numeric or categorical

no_transform_attribs = ['Patient', 'Weeks', 'min_week']

num_attribs = ['FVC', 'Percent', 'Age', 'baselined_week', 'base_FVC']

cat_attribs = ['Sex', 'SmokingStatus']
from sklearn.base import BaseEstimator, TransformerMixin



class NoTransformer(BaseEstimator, TransformerMixin):

    """Passes through data without any change and is compatible with ColumnTransformer class"""

    def fit(self, X, y=None):

        return self



    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X
## GET TRANSFORMED DATAFRAME



# create an instance of the ColumnTransformer

datawrangler = ColumnTransformer(([

     # the No-Transformer does not change the data and is applied to all no_transform_attribs 

     ('original', NoTransformer(), no_transform_attribs),

     # Apply MinMax to the numerical attributes, here you can change to e.g. StdScaler()   

     ('MinMax', MinMaxScaler(), num_attribs),

     # OneHotEncoder all categorical attributes.   

     ('cat_encoder', OneHotEncoder(), cat_attribs),

    ]))



transformed_data_series = []

transformed_data_series = datawrangler.fit_transform(data_df)
# get column names for non-categorical data

new_col_names = no_transform_attribs + num_attribs



# extract possible values from the fitted transformer

categorical_values = [s for s in datawrangler.named_transformers_["cat_encoder"].get_feature_names()]

new_col_names += categorical_values



# create Dataframe based on the extracted Column-Names

train_sklearn_df = pd.DataFrame(transformed_data_series, columns=new_col_names)

train_sklearn_df.head()
def own_MinMaxColumnScaler(df, columns):

    """Adds columns with scaled numeric values to range [0, 1]

    using the formula X_scld = (X - X.min) / (X.max - X.min)"""

    for col in columns:

        new_col_name = col + '_scld'

        col_min = df[col].min()

        col_max = df[col].max()        

        df[new_col_name] = (df[col] - col_min) / ( col_max - col_min )
def own_OneHotColumnCreator(df, columns):

    """OneHot Encodes categorical features. Adds a column for each unique value per column"""

    for col in cat_attribs:

        for value in df[col].unique():

            df[value] = (df[col] == value).astype(int)
## APPLY DEFINED TRANSFORMATIONS

own_MinMaxColumnScaler(data_df, num_attribs)

own_OneHotColumnCreator(data_df, cat_attribs)



data_df[data_df.Source != "train"].head()
# get back original data split

train_df = data_df.loc[data_df.Source == 'train']

sub = data_df.loc[data_df.Source == 'test']
######## CONFIG ########

# be careful, the resulsts are very SEED-DEPENDEND!

seed_everything(1989)





### Features: choose which features you want to use

# you can exclude and include features by extending this feature list

features_list = ['baselined_week_scld', 'Age_scld', 'base_FVC_scld', 'Male', 'Female', 'Ex-smoker', 'Never smoked', 'Currently smokes']



### Basics for training:

EPOCHS = 1500

BATCH_SIZE = 128





### LOSS; set tradeoff btw. Pinball-loss and adding score

_lambda = 0.8 # 0.8 default





### Optimizers

# choose ADAM or SGD

optimizer = 'SGD'



### Learning Rate Scheduler

def get_lr_callback(batch_size = 64, plot = False):

    """Returns a lr_scheduler callback which is used for training.

    Feel free to change the values below!

    """

    LR_START   = 0.001

    LR_MAX     = 0.0001 * BATCH_SIZE # higher batch size --> higher lr

    LR_MIN     = 0.000001

    # 30% of all epochs are used for ramping up the LR and then declining starts

    LR_RAMP_EP = EPOCHS * 0.3

    # how many epochs shall L_RMAX be sustained

    LR_SUS_EP  = 0

    # rate of decay

    LR_DECAY   = 0.993



    def lr_scheduler(epoch):

            if epoch < LR_RAMP_EP:

                lr = (LR_MAX - LR_START) / LR_RAMP_EP * epoch + LR_START



            elif epoch < LR_RAMP_EP + LR_SUS_EP:

                lr = LR_MAX



            else:

                lr = (LR_MAX - LR_MIN) * LR_DECAY ** (epoch - LR_RAMP_EP - LR_SUS_EP) + LR_MIN



            return lr

    

    if plot == False:

        # get the Keras-required callback with our LR for training

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler,verbose = False)

        return lr_callback 

    

    else: 

        return lr_scheduler

    

# plot & check the LR-Scheulder for sanity-check

lr_scheduler_plot = get_lr_callback(batch_size = 64, plot = True)

rng = [i for i in range(EPOCHS)]

y = [lr_scheduler_plot(x) for x in rng]

plt.plot(rng, y)

print(f"Learning rate schedule: {y[0]:.3f} to {max(y):.3f} to {y[-1]:.3f}")





# logging & saving

LOGGING = True



# defining custom callbacks

class LogPrintingCallback(tf.keras.callbacks.Callback):

    

    # defining a class variable which is used in the following

    # to ensure LogPrinting, evaluation & saving is consitent

    # choose between 'val_score' and 'val_loss'

    optimization_variable = 'val_loss'

    

    def on_train_begin(self, logs = None):

        #print("Training started for this fold")

        self.val_loss = []

        self.val_score = []        

        

    def on_epoch_end(self, epoch, logs = None):

        self.val_loss.append(logs['val_loss']) 

        self.val_score.append(logs['val_score'])

        if epoch % 250 == 0 or epoch == (EPOCHS -1 ):

            print(f"The average val-loss for epoch {epoch} is {logs['val_loss']:.2f}"

                  f" and the score is {logs['val_score']}")

            

    def on_train_end(self, logs = None):

        # get index of best epoch

        monitored_variable = self.val_loss if LogPrintingCallback.optimization_variable == 'val_loss' else self.val_score

        best_epoch = np.argmin(monitored_variable)        

        # get score in best epoch

        best_result_loss = self.val_loss[best_epoch]

        best_result_score = self.val_score[best_epoch]

        print(f"Best model was found and saved in epoch {best_epoch + 1} with val_loss: {best_result_loss} and val_score: {best_result_score} ") 

        

        

def get_checkpoint_saver_callback(fold):

    checkpt_saver = tf.keras.callbacks.ModelCheckpoint(

        'fold-%i.h5'%fold,

        monitor = 'val_score',

        verbose = 0,

        save_best_only = True,

        save_weights_only = True,

        mode = 'min',

        save_freq = 'epoch')

    

    return checkpt_saver
# create constants for the loss function

C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")



# define competition metric

def score(y_true, y_pred):

    """Calculate the competition metric"""

    tf.dtypes.cast(y_true, tf.float32)

    tf.dtypes.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    sigma_clip = tf.maximum(sigma, C1)

    # Python is automatically broadcasting y_true with shape (1,0) to 

    # shape (3,0) in order to make this subtraction work

    delta = tf.abs(y_true[:, 0] - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype = tf.float32) )

    metric = (delta / sigma_clip) * sq2 + tf.math.log(sigma_clip * sq2)

    return K.mean(metric)



# define pinball loss

def qloss(y_true, y_pred):

    """Calculate Pinball loss"""

    # IMPORTANT: define quartiles, feel free to change here!

    qs = [0.2, 0.50, 0.8]

    q = tf.constant(np.array([qs]), dtype = tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q * e, (q-1) * e)

    return K.mean(v)



# combine competition metric and pinball loss to a joint loss function

def mloss(_lambda):

    """Combine Score and qloss"""

    def loss(y_true, y_pred):

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda) * score(y_true, y_pred)

    return loss
import tensorflow_addons as tfa



def get_model(optimizer = 'ADAM', lr = 0.01):

    "Creates and returns a model"

    # instantiate optimizer

    optimizer = tf.keras.optimizers.Adam(lr = lr) if optimizer == 'ADAM' else tf.keras.optimizers.SGD(lr)

    

    # create model    

    inp = Layers.Input((len(features_list),), name = "Patient")

    x = Layers.BatchNormalization()(inp)

    x = tfa.layers.WeightNormalization(Layers.Dense(160, activation = "elu", name = "d1"))(x)

    x = Layers.BatchNormalization()(x)

    x = Layers.Dropout(0.3)(x)

    x = tfa.layers.WeightNormalization(Layers.Dense(128, activation = "elu", name = "d2"))(x)

    x = Layers.BatchNormalization()(x)

    x = Layers.Dropout(0.25)(x)

    # predicting the 3 quantiles

    q1 = Layers.Dense(3, activation = "relu", name = "p1")(x)

    # generating another output for quantile adjusting the quantile predictions

    q_adjust = Layers.Dense(3, activation = "relu", name = "p2")(x)

    

    # adding the tf.cumsum of q_adjust to the output q1

    # to ensure increasing values [a < b < c]

    # tf.cumsum([a, b, c]) --> [a, a + b, a + b + c]

    preds = Layers.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis = 1), 

                     name = "preds")([q1, q_adjust])

    

    model = tf.keras.Model(inputs = inp, outputs = preds, name = "NeuralNet")

    model.compile(loss = mloss(_lambda), optimizer = optimizer, metrics = [score])

    

    return model
# create neural Network

neuralNet = get_model(optimizer, lr = 0.01)

neuralNet.summary()
## GET TRAINING DATA AND TARGET VALUE



# get target value

y = train_df['FVC'].values.astype(float)





# get training & test data

X_train = train_df[features_list].values

X_test = sub[features_list].values



# instantiate target arrays

train_preds = np.zeros((X_train.shape[0], 3))

test_preds = np.zeros((X_test.shape[0], 3))
## Non-Stratified GroupKFold-split (can be further enhanced with stratification!)

"""K-fold variant with non-overlapping groups.

The same group will not appear in two different folds: in this case we dont want to have overlapping patientIDs in TRAIN and VAL-Data!

The folds are approximately balanced in the sense that the number of distinct groups is approximately the same in each fold."""



NFOLDS = 5

gkf = GroupKFold(n_splits = NFOLDS)

# extract Patient IDs for ensuring 

groups = train_df['Patient'].values



OOF_val_score = []

fold = 0



for train_idx, val_idx in gkf.split(X_train, y, groups = groups):

    fold += 1

    print(f"FOLD {fold}:")

    

    # callbacks: logging & model saving with checkpoints each fold

    # callbacks = [get_lr_callback(BATCH_SIZE)]  # un-comment for using LRScheduler

    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss',

                                                          factor = 0.6,

                                                          patience = 150,

                                                          verbose = 1,

                                                          epsilon = 1e-4,

                                                          mode = 'min',

                                                          min_lr = 0.00001)

    

    callbacks = [reduce_lr_loss]

    

    if LOGGING == True:

        callbacks +=  [get_checkpoint_saver_callback(fold),                     

                     LogPrintingCallback()]



    # build and train model

    model = get_model(optimizer, lr = 0.005)

    history = model.fit(X_train[train_idx], y[train_idx], 

              batch_size = BATCH_SIZE, 

              epochs = EPOCHS, 

              validation_data = (X_train[val_idx], y[val_idx]), 

              callbacks = callbacks,

              verbose = 0) 

    

    # evaluate

    print("Train:", model.evaluate(X_train[train_idx], y[train_idx], verbose = 0, batch_size = BATCH_SIZE, return_dict = True))

    print("Val:", model.evaluate(X_train[val_idx], y[val_idx], verbose = 0, batch_size = BATCH_SIZE, return_dict = True))

    

    ## Load best model to make pred

    model.load_weights('fold-%i.h5'%fold)

    train_preds[val_idx] = model.predict(X_train[val_idx],

                                         batch_size = BATCH_SIZE,

                                         verbose = 0)

    

    # append OOF evaluation to calculate OFF_Score

    OOF_val_score.append(model.evaluate(X_train[val_idx], y[val_idx], verbose = 0, batch_size = BATCH_SIZE, return_dict = True)['score'])

    

    # predict on test set and average the predictions over all folds

    print("Predicting Test...")

    test_preds += model.predict(X_test, batch_size = BATCH_SIZE, verbose = 0) / NFOLDS
## PLOT results

# fetch results from history

score = history.history['score']

val_score = history.history['val_score']



loss = history.history['loss']

val_loss = history.history['val_loss']



epochs_range = range(EPOCHS)



# create subplots

plt.figure(figsize = (20,5))

plt.subplot(1, 2, 1)

plt.plot(epochs_range, score, label = 'Training Score')

plt.plot(epochs_range, val_score, label = 'Validation Score')

# limit y-values for better zoom-scale. Remember that roughly -4.5 is the best possible score

# plt.ylim(0.8 * np.mean(val_score), 1.2 * np.mean(val_score))

plt.legend(loc = 'lower right')

plt.title('Training and Validation Score')



plt.subplot(1, 2, 2)

plt.plot(epochs_range, loss, label = 'Training Loss')

plt.plot(epochs_range, val_loss, label = 'Validation Loss')

# limit y-values for beter zoom-scale

plt.ylim(0.3 * np.mean(val_loss), 1.8 * np.mean(val_loss))



plt.legend(loc = 'upper right')

plt.title('Training and Validation Loss')

plt.show()
np.mean(OOF_val_score)
## FIND OPTIMIZED STANDARD-DEVIATION

sigma_opt = mean_absolute_error(y, train_preds[:,1])

sigma_uncertain = train_preds[:,2] - train_preds[:,0]

sigma_mean = np.mean(sigma_uncertain)

print(sigma_opt, sigma_mean)
sub.head()
## PREPARE SUBMISSION FILE WITH OUR PREDICTIONS

sub['FVC1'] = test_preds[:, 1]

sub['Confidence1'] = test_preds[:,2] - test_preds[:,0]



# get rid of unused data and show some non-empty data

submission = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()

submission.loc[~submission.FVC1.isnull()].head(10)
submission.loc[~submission.FVC1.isnull(),'FVC'] = submission.loc[~submission.FVC1.isnull(),'FVC1']



if sigma_mean < 70:

    submission['Confidence'] = sigma_opt

else:

    submission.loc[~submission.FVC1.isnull(),'Confidence'] = submission.loc[~submission.FVC1.isnull(),'Confidence1']
submission.head()
submission.describe().T
org_test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')



for i in range(len(org_test)):

    submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'FVC'] = org_test.FVC[i]

    submission.loc[submission['Patient_Week']==org_test.Patient[i]+'_'+str(org_test.Weeks[i]), 'Confidence'] = 70
submission[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index = False)