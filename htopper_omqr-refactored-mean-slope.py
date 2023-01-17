import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None) # show all cols

import pydicom

import os

import random

import matplotlib.pyplot as plt

from tqdm import tqdm

from PIL import Image

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold



import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import tensorflow.keras.models as M
#env = 'local'

env = 'kaggle'



BATCH_SIZE=128
if env == 'local':

    ROOT = "../input"

else:

    ROOT = "../input/osic-pulmonary-fibrosis-progression" # kaggle
train = pd.read_csv(f"{ROOT}/train.csv")

train.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])



test = pd.read_csv(f"{ROOT}/test.csv")



submission = pd.read_csv(f"{ROOT}/sample_submission.csv")
def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    if env == 'local':

        tf.random.set_random_seed(seed) # local

    else:

        tf.random.set_seed(seed) # kaggle

    

seed_everything(42)
## evaluation metric function

def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values = False):

    """

    Calculates the modified Laplace Log Likelihood score for this competition.

    """

    sd_clipped = np.maximum(confidence, 70)

    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)

    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)



    if return_values:

        return metric

    else:

        return np.mean(metric)
all_patient_ids = train['Patient'].unique()



patient_slopes_df = pd.DataFrame()

patient_slopes_df['Patient'] = all_patient_ids



for i in ['Percent','FVC']:

    slopes = []

    intercepts = []

    for patient_id in all_patient_ids:

        patient_df = train[train['Patient']==patient_id]

        x = patient_df['Weeks'].to_numpy()

        y = patient_df[i].to_numpy()



        ## fit with polyfit

        m, b = np.polyfit(x, y, 1)

        slopes.append(m)

        intercepts.append(b)



    patient_slopes_df['Slope_'+i] = slopes

    patient_slopes_df['Intercept_'+i] = intercepts

    

mean_slope_percent = patient_slopes_df['Slope_Percent'].mean()

mean_slope_fvc = patient_slopes_df['Slope_FVC'].mean()



print('mean_slope_percent:',mean_slope_percent)

print('mean_slope_fvc:',mean_slope_fvc)

print('')

print(patient_slopes_df.shape)

patient_slopes_df.head(3)
submission['Patient'] = submission['Patient_Week'].apply(lambda x:x.split('_')[0])

submission['Weeks'] = submission['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))



submission =  submission[['Patient','Weeks','Confidence','Patient_Week']]



temp_test = test.copy()



temp_test.drop(columns=['Weeks'],inplace=True)



submission = pd.merge(submission,temp_test, how='left', on=["Patient"])



del temp_test

submission.head(3)
submission[submission['Patient']=='ID00419637202311204720264'].head()
temp_test = test.copy()



# intercept_percent based on mean_slope_percent

temp_test['intercept_percent'] = temp_test['Percent'] - (temp_test['Weeks'] * mean_slope_percent)



submission = pd.merge(submission,temp_test[['Patient','intercept_percent']], how='left', on=["Patient"])

del temp_test



submission['Percent_mean_slope'] =  submission['intercept_percent'] + (submission['Weeks'] * mean_slope_percent)



submission['Percent'] = submission['Percent_mean_slope']

submission.drop(columns=['intercept_percent','Percent_mean_slope'],inplace=True)
submission[submission['Patient']=='ID00419637202311204720264'].head()
train['WHERE'] = 'train'

test['WHERE'] = 'test'

submission['WHERE'] = 'submission'



data = train.append([test, submission])
print(train.shape, test.shape, submission.shape, data.shape)

print(train.Patient.nunique(), test.Patient.nunique(), submission.Patient.nunique(), data.Patient.nunique())



data.head(3)
data['min_week'] = data['Weeks']

data.loc[data.WHERE=='submission','min_week'] = np.nan

data['min_week'] = data.groupby('Patient')['min_week'].transform('min')
base = data.loc[data.Weeks == data.min_week]

base = base[['Patient','FVC']].copy()

base.columns = ['Patient','min_FVC']

base['nb'] = 1

base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

base = base[base.nb==1]

base.drop('nb', axis=1, inplace=True)
data = data.merge(base, on='Patient', how='left')

data['base_week'] = data['Weeks'] - data['min_week']

del base
COLS = ['Sex','SmokingStatus'] #,'Age'

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)

#=================
#

data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )

data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )

data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )

# WRONG? - does altered submission data give unrealistic percent range?

data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )

FE += ['age','percent','week','BASE']



## for later use in variant of "tr"

data_percent_min = data['Percent'].min()

data_percent_max = data['Percent'].max()
train = data.loc[data.WHERE=='train']

test = data.loc[data.WHERE=='test']

submission = data.loc[data.WHERE=='submission']

del data
train.shape, test.shape, submission.shape
submission.head()
# "weeks_0" : percent on patient week 0 for all weeks

# "avg_slope" : use globel avg slope to calc percent for all weeks

# "pred_slope" : use prediction of slope to calc percent for all weeks



percent_model = "avg_slope"
tr_adj = train[train['Weeks']>=0].groupby(['Patient'],as_index=False)['Weeks'].min()

tr_adj = pd.merge(tr_adj,train,how='left',on=['Patient','Weeks'])

# if multiple time FVC measured in week for patient, take the mean (Depends on notebook if this does anything)

tr_adj = tr_adj.groupby(['Patient','Weeks'],as_index=False)[['Percent','FVC']].mean()



# intercept_percent based on mean_slope_percent

tr_adj['intercept_percent'] = tr_adj['Percent'] - (tr_adj['Weeks'] * mean_slope_percent)



# rename before merge

tr_adj.rename(columns={'Weeks': 'Weeks_0', 'Percent': 'Percent_0','FVC':'FVC_0'}, inplace=True)

tr_adj.head()
train_variant = train.copy()

train_variant = pd.merge(train_variant,tr_adj,how='left',on=['Patient'])



train_variant['Percent_mean_slope'] =  train_variant['intercept_percent'] + (train_variant['Weeks'] * mean_slope_percent)



##

if percent_model == "weeks_0":

    train_variant['percent'] = (train_variant['Percent_0'] - data_percent_min) / (data_percent_max - data_percent_min)

##

if percent_model == "avg_slope":

    train_variant['percent'] = (train_variant['Percent_mean_slope'] - data_percent_min) / (data_percent_max - data_percent_min)



    

train_variant.drop(columns=['Percent_mean_slope','intercept_percent'],inplace=True)



print(train_variant.shape)

train_variant.head()
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

#=============================#

def score(y_true, y_pred):

    tf.dtypes.cast(y_true, tf.float32)

    tf.dtypes.cast(y_pred, tf.float32)

    sigma = y_pred[:, 2] - y_pred[:, 0]

    fvc_pred = y_pred[:, 1]

    

    #sigma_clip = sigma + C1

    sigma_clip = tf.maximum(sigma, C1)

    delta = tf.abs(y_true[:, 0] - fvc_pred)

    delta = tf.minimum(delta, C2)

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) )

    metric = (delta / sigma_clip)*sq2 + tf.math.log(sigma_clip* sq2)

    return K.mean(metric)

#============================#

def qloss(y_true, y_pred):

    # Pinball loss for multiple quantiles

    qs = [0.2, 0.50, 0.8]

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    return K.mean(v)

#=============================#

def mloss(_lambda):

    def loss(y_true, y_pred):

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*score(y_true, y_pred)

    return loss

#=================

def make_model(nh):

    z = L.Input((nh,), name="Patient")

    x = L.Dense(100, activation="relu", name="d1")(z)

    x = L.Dense(100, activation="relu", name="d2")(x)

    #x = L.Dense(100, activation="relu", name="d3")(x)

    p1 = L.Dense(3, activation="linear", name="p1")(x)

    p2 = L.Dense(3, activation="relu", name="p2")(x)

    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 

                     name="preds")([p1, p2])

    

    model = M.Model(z, preds, name="CNN")

    #model.compile(loss=qloss, optimizer="adam", metrics=[score])

    model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])

    return model
y_train = train['FVC'].values.astype(np.float32)

x_train = train[FE].values.astype(np.float32)



x_train_variant = train_variant[FE].values.astype(np.float32) # variant of train data, for oof



x_test = test[FE].values.astype(np.float32)

x_submission = submission[FE].values.astype(np.float32)



pred_test = np.zeros((x_test.shape[0], 3))

pred_train = np.zeros((x_train.shape[0], 3))

pred_submission = np.zeros((x_submission.shape[0], 3))
num_inputs = x_train.shape[1]



net = make_model(num_inputs)

print(net.summary())

print(net.count_params())
NFOLD = 5

repeats = 10
%%time



## oof

preds_df = pd.DataFrame()

conf_df = pd.DataFrame()

## test

preds_test_df = pd.DataFrame()

conf_test_df = pd.DataFrame()

## submisson

preds_submission_df = pd.DataFrame()

conf_submission_df = pd.DataFrame()



for random_state in range(repeats):

    print('repeat:',random_state)

    

#     pe = np.zeros((ze.shape[0], 3))

#     pred = np.zeros((z.shape[0], 3))

    

    pred_test = np.zeros((x_test.shape[0], 3))

    pred_train = np.zeros((x_train.shape[0], 3))

    pred_submission = np.zeros((x_submission.shape[0], 3))

    

    kf = KFold(n_splits=NFOLD,shuffle=True,random_state=random_state)

    cnt = 0

    EPOCHS = 800

    for tr_idx, val_idx in kf.split(x_train):

        cnt += 1

        print(f"FOLD {cnt}")

        net = make_model(num_inputs)

        net.fit(x_train[tr_idx], y_train[tr_idx], batch_size=BATCH_SIZE, epochs=EPOCHS, 

                validation_data=(x_train[val_idx], y_train[val_idx]), verbose=0) #

        print("train", net.evaluate(x_train[tr_idx], y_train[tr_idx], verbose=0, batch_size=BATCH_SIZE))

        print("val", net.evaluate(x_train[val_idx], y_train[val_idx], verbose=0, batch_size=BATCH_SIZE))

        print("predict train-val...")

        pred_train[val_idx] = net.predict(x_train_variant[val_idx], batch_size=BATCH_SIZE, verbose=0) # use variant for oof

        print("predict test...")

        pred_test += net.predict(x_test, batch_size=BATCH_SIZE, verbose=0) / NFOLD

        print("predict submission...")

        pred_submission += net.predict(x_submission, batch_size=BATCH_SIZE, verbose=0) / NFOLD

    #==============





    sigma_opt = mean_absolute_error(y_train, pred_train[:, 1])

    unc = pred_train[:,2] - pred_train[:, 0]

    sigma_mean = np.mean(unc)

    print('----------------------------')

    print(sigma_opt, sigma_mean)

    

    preds_df[random_state] = pred_train[:, 1]

    conf_df[random_state] = unc

    

    preds_test_df[random_state] = pred_test[:, 1]

    conf_test_df[random_state] = pred_test[:,2] - pred_test[:, 0]

    

    preds_submission_df[random_state] = pred_submission[:, 1]

    conf_submission_df[random_state] = pred_submission[:,2] - pred_submission[:, 0]
## oof

preds_df['mean'] = preds_df.iloc[:,0:repeats].mean(axis=1)

conf_df['mean'] = conf_df.iloc[:,0:repeats].mean(axis=1)

preds_df['median'] = preds_df.iloc[:,0:repeats].median(axis=1)

conf_df['median'] = conf_df.iloc[:,0:repeats].median(axis=1)

## test

preds_test_df['mean'] = preds_test_df.iloc[:,0:repeats].mean(axis=1)

conf_test_df['mean'] = conf_test_df.iloc[:,0:repeats].mean(axis=1)

preds_test_df['median'] = preds_test_df.iloc[:,0:repeats].median(axis=1)

conf_test_df['median'] = conf_test_df.iloc[:,0:repeats].median(axis=1)

## submission

preds_submission_df['mean'] = preds_submission_df.iloc[:,0:repeats].mean(axis=1)

conf_submission_df['mean'] = conf_submission_df.iloc[:,0:repeats].mean(axis=1)

preds_submission_df['median'] = preds_submission_df.iloc[:,0:repeats].median(axis=1)

conf_submission_df['median'] = conf_submission_df.iloc[:,0:repeats].median(axis=1)
scores = []

for i in range(repeats):

    score = laplace_log_likelihood(y_train, preds_df[i], conf_df[i])

    print('solution',i,':',score)

    scores.append(score)

print('----------------')

print('mean score:',np.mean(scores))
print('mean OOF lap_log:',laplace_log_likelihood(y_train, preds_df['mean'], conf_df['mean']))

print('median OOF lap_log:',laplace_log_likelihood(y_train, preds_df['median'], conf_df['median']))
temp = train_variant.copy()



temp['prediction'] = preds_df['mean']

temp['confidence'] = conf_df['mean']



## remove given rows before calc score

# remove rows with FVC "given"

temp['for_eval'] = temp['Weeks'] != temp['Weeks_0']

temp = temp[temp['for_eval']==True]



print('mean OOF lap_log:',laplace_log_likelihood(temp['FVC'], temp['prediction'], temp['confidence']))
p_oof = train_variant.copy()

p_oof['prediction'] = preds_df['mean']

p_oof['confidence'] = conf_df['mean']



p_oof.head(3)
train_adj = p_oof[p_oof['Weeks']>=0].groupby(['Patient'],as_index=False)['Weeks'].min()

train_adj = pd.merge(train_adj,p_oof,how='left',on=['Patient','Weeks'])



# if multiple time FVC measured in week for patient, take the mean

train_adj = train_adj.groupby(['Patient','Weeks'],as_index=False)[['Percent','FVC']].mean()



train_adj.rename(columns={'Weeks': 'First_week'}, inplace=True) 



train_adj.head(3)
p_oof = pd.merge(p_oof,train_adj[['Patient','First_week']],how='left',on=['Patient'])



p_oof_week_0 = p_oof[p_oof['Weeks']==p_oof['First_week']][['Patient','FVC','prediction']].copy()

p_oof_week_0.reset_index(inplace=True, drop=True)

p_oof_week_0['prediction_w0_delta'] = p_oof_week_0['FVC'] - p_oof_week_0['prediction']

p_oof_week_0.head(3)
## add week 0 delta data

p_oof = pd.merge(p_oof,p_oof_week_0[['Patient','prediction_w0_delta']],how='left',on=['Patient'])

## adjust prediction

p_oof['predition_adjusted'] =  p_oof['prediction'] + p_oof['prediction_w0_delta']
temp = p_oof.copy()



# remove rows with FVC "given"

temp['for_eval'] = (temp['Weeks'] != temp['First_week'])

temp = temp[temp['for_eval']==True] 



print('mean OOF lap_log:',laplace_log_likelihood(temp['FVC'], temp['predition_adjusted'], temp['confidence']))
save_oof = False



if save_oof:

    check_me = train.copy()

    check_me['prediction'] = preds_df['mean']

    check_me['confidence'] = conf_df['mean']



    check_me.head(3)

    check_me.to_csv("oof_name_here.csv", index=False)
print(submission.shape)

submission.head()
my_sub = submission.copy()

my_sub.reset_index(inplace=True, drop=True)

my_sub = my_sub[['Patient','Weeks']]

my_sub['FVC'] = preds_submission_df['mean'].values

my_sub['Confidence'] = conf_submission_df['mean'].values



my_sub.head()
# test_temp = test[['Patient','Weeks','FVC']].copy()

# test_temp.reset_index(inplace=True,drop=True)

# test_temp.rename(columns={'FVC':'W0_FVC'}, inplace=True)



# print(test_temp.shape)

# test_temp.head(3)
# test_temp = pd.merge(test_temp,my_sub,on=['Patient','Weeks'],how='left')

# test_temp['FVC_delta'] = test_temp['W0_FVC'] - test_temp['FVC']



# print(test_temp.shape)

# test_temp.head(3)
# my_sub = pd.merge(my_sub,test_temp[['Patient','FVC_delta']],on=['Patient'],how='left')

# my_sub['FVC_adjusted'] = my_sub['FVC'] + my_sub['FVC_delta']



# print(my_sub.shape)

# my_sub.head(10)
show_test_predictions = False



if show_test_predictions:

    all_patients = my_sub['Patient'].unique()

    print('num unique patients:',len(all_patients))



    for patient in all_patients[0:5]:

        temp = my_sub[my_sub['Patient']==patient].copy()

        print('patient:',patient)

        plt.plot(temp['Weeks'], temp['FVC_adjusted'], '-',label='prediction',color='purple')

        plt.axvline(x=0,color='gray',ls='--') # visual aid

        plt.legend()

        plt.show();

        print('--')
make_submission = True

if make_submission:

    #my_sub['FVC'] = my_sub['FVC_adjusted']

    

    my_sub['Weeks'] = my_sub['Weeks'].astype(str)

    my_sub['Patient_Week'] = my_sub['Patient'] + '_' + my_sub['Weeks']

    my_sub = my_sub[['Patient_Week','FVC','Confidence']]

    

    print(my_sub.shape)

    my_sub.head()



    my_sub.to_csv("submission.csv", index=False)
# print(my_sub.shape)

# my_sub.head()