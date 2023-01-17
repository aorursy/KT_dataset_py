# intended one

# me
import numpy as np

import pandas as pd

import seaborn as sns

#import pydicom

import os

import random

import matplotlib.pyplot as plt

from tqdm import tqdm

#from PIL import Image

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold, GroupKFold
import tensorflow as tf

import tensorflow.keras.backend as backend   #K  

import tensorflow.keras.layers as layers     #L

import tensorflow.keras.models as models     #M
# avoiding randomness to get the same result of the model asways

def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    return seed
# # CONFIGURE GPUs

# #os.environ["CUDA_VISIBLE_DEVICES"]="0"

# gpus = tf.config.list_physical_devices('GPU'); print(gpus)

# if len(gpus)==1: strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

# else: strategy = tf.distribute.MirroredStrategy()
# # ENABLE MIXED PRECISION for speed

# #tf.config.optimizer.set_jit(True)

# tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

# print('Mixed precision enabled')
path = "../input/osic-pulmonary-fibrosis-progression"



train = pd.read_csv(f"{path}/train.csv")                        #tr

test  = pd.read_csv(f"{path}/test.csv")                         #chunk

print('****training data head 1 value****\n')

print(train.head(1))

print('\n****test data head 1 value****\n')

print(test.head(1))
print('train_data_shape', train.shape)

print('test_data_shape', test.shape)

print('duplicates',train.duplicated().sum())



# for now we are leaving this command but it could be useful to see its effect on the accuracy

print('duplicates',train.duplicated(subset=['Patient','Weeks']).sum())

train.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
#reading the submission file

# columns 'Patient_Week', 'FVC', 'Confidence'

sub = pd.read_csv(f"{path}/sample_submission.csv")



# creating the new colomn patient and week from the Patient_Week column

# we are using the - as seprator key

sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])



sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))





# 'Patient_Week', 'FVC', 'Confidence', 'Patient', 'Weeks'

# we are droping the FVC column

sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
# we are merging the submission data with test data [before merging removing the weeks column from it]

submission = sub.merge(test.drop('Weeks', axis=1), on="Patient")

print('****submission_data head 1 value*****\n')

print(sub.head(1))

print('******test data head 1 value********')

print(test.head(1))



submission.head(1)
# adding new columns WHERE in train, test and submission data

train['WHERE'] = 'train'

test['WHERE'] = 'val'

submission['WHERE'] = 'test'

print('train data shape\n\n',train.shape)

print('\ntest data shape\n\n',test.shape)

print('\nsubmission data shape\n\n',submission.shape)



# we append the test and submission data on to

data = train.append([test, submission])

print('\ndata_shape',data.shape)

data.head(2)

# 1535+5+730 = 2270
# we know each patient have multiple entries so we are checking the unique entries

print(train.Patient.nunique(), data.Patient.nunique(), test.Patient.nunique(), submission.Patient.nunique())
# creating a new column min_week 

# min week means the first week of patient's observation

check_point_1 = data['min_week'] = data['Weeks']



# putting the min_week to NAN value

check_point_2 = data.loc[data.WHERE=='test','min_week'] = np.nan



check_point_3 = data['min_week'] = data.groupby('Patient')['min_week'].transform('min')



print('check_point_1\n',check_point_1.head(10))

print('\ncheck_point_2\n',check_point_2)

print('\ncheck_point_3\n',check_point_3)

data.head(10)
base = data.loc[data.Weeks == data.min_week]

a = base

base = base[['Patient','FVC']].copy()

b = base



base.columns = ['Patient','min_FVC']

base['nb'] = 1

c = base



base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

d = base



base = base[base.nb==1]

e = base



base.drop('nb', axis=1, inplace=True)

f = base



# a.head(1)

# b.head(1)

# c.head(5)

# d.head(5)

# e.head(8)

# f.head(5)
data = data.merge(base, on='Patient', how='left')



data['base_week'] = data['Weeks'] - data['min_week']





data.head(10)

del base
data.head(5)
# creating dummies of all categorical values

############## THIS METHORD IS BIT COMPLEX ONE ##############################3



COLS = ['Sex','SmokingStatus']

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        #print(FE)

        data[mod] = (data[col] == mod).astype(int)

        #print(data)

    #break
train = data.loc[data.WHERE=='train']

test = data.loc[data.WHERE=='val']

submission = data.loc[data.WHERE=='test']





# print('train data shape\n',train.shape)

# print('\ntest data shape\n',test.shape)

# print('\nsubmission data shape\n',submission.shape)



del data
train['age'] = (train['Age'] - train['Age'].min() ) / ( train['Age'].max() - train['Age'].min() )

train['BASE'] = (train['min_FVC'] - train['min_FVC'].min() ) / ( train['min_FVC'].max() - train['min_FVC'].min() )

train['week'] = (train['base_week'] - train['base_week'].min() ) / ( train['base_week'].max() - train['base_week'].min() )

train['percent'] = (train['Percent'] - train['Percent'].min() ) / ( train['Percent'].max() - train['Percent'].min() )



test['age'] = (test['Age'] - test['Age'].min() ) / ( test['Age'].max() - test['Age'].min() )

test['BASE'] = (test['min_FVC'] - test['min_FVC'].min() ) / ( test['min_FVC'].max() - test['min_FVC'].min() )

test['week'] = (test['base_week'] - test['base_week'].min() ) / ( test['base_week'].max() - test['base_week'].min() )

test['percent'] = (test['Percent'] - test['Percent'].min() ) / ( test['Percent'].max() - test['Percent'].min() )



submission['age'] = (submission['Age'] - submission['Age'].min() ) / ( submission['Age'].max() - submission['Age'].min() )

submission['BASE'] = (submission['min_FVC'] - submission['min_FVC'].min() ) / ( submission['min_FVC'].max() - submission['min_FVC'].min() )

submission['week'] = (submission['base_week'] - submission['base_week'].min() ) / ( submission['base_week'].max() - submission['base_week'].min() )

submission['percent'] = (submission['Percent'] - submission['Percent'].min() ) / ( submission['Percent'].max() - submission['Percent'].min() )



FE += ['age','percent','week','BASE']

#FE += ['age','percent','BASE']
#FE
SEED = seed_everything(42)

NFOLD      = 4

BATCH_SIZE = 128

EPOCHS     = 400
C1, C2 = tf.constant(70, dtype='float32'), tf.constant(1000, dtype="float32")

# print('C1 = ',C1)

# print('C2 = ',C2)



def Laplace_log_Likelihood_score(y_true, y_pred):

    

    tf.dtypes.cast(y_true, tf.float32)  # converting y_true in float values

    tf.dtypes.cast(y_pred, tf.float32)  # converting y_pred in float values

    sigma = y_pred[:, 2] - y_pred[:, 0] # calculating the standard deviation 

    fvc_pred = y_pred[:, 1]

    

    #sigma_clip = sigma + C1

    sigma_clip = tf.maximum(sigma, C1)  # clipping all standard deviation(sigma) the other values less than 70

    

    delta = tf.abs(y_true[:, 0] - fvc_pred) # |FVC_true - FVC_predicted| abs mean we need a +ve value as always

    delta = tf.minimum(delta, C2)           # clipping all values greater than 1000

    

    sq2 = tf.sqrt( tf.dtypes.cast(2, dtype=tf.float32) ) # calculating sqr root of 2

    

    metric = -(delta / sigma_clip)*sq2 - tf.math.log(sigma_clip* sq2) # calculating metric as given in OSIC  evaluation

    #print('matric value', metric)

    #print('backend.mean(metric) from log laplace transform', backend.mean(metric))

    #print('Calculating Laplace_log_Likelihood_score')

    #print('value = ', backend.mean(metric))

    return backend.mean(metric)
def qloss(y_true, y_pred):

    

    # Pinball loss for multiple quantiles

    qs = [0.2, 0.50, 0.8]

    q = tf.constant(np.array([qs]), dtype=tf.float32)

    e = y_true - y_pred

    v = tf.maximum(q*e, (q-1)*e)

    

    #print('backend.mean(v)', backend.mean)

    #print('Calculating qLoss')

    #print('value = ', (v))

    return backend.mean(v)
def mloss(_lambda):

    def loss(y_true, y_pred):

        return _lambda * qloss(y_true, y_pred) + (1 - _lambda)*Laplace_log_Likelihood_score(y_true, y_pred)

    #print('Calculating mLoss')

    return loss
def make_model():                          #backend   #K layers     #L models     #M

    z = layers.Input((9,), name="Patient")

    

    x = layers.Dense(100, activation="relu", name="d1")(z)

    #x = layers.Dropout(0.002)(x)

    x = layers.Dense(100, activation="relu", name="d2")(x)

    x = layers.Dropout(0.002)(x)

#     x = layers.Dense(100, activation="relu", name="d3")(x)

#     x = layers.Dense(100, activation="relu", name="d4")(x)

#     x = layers.Dense(100, activation="relu", name="d5")(x)

#     x = layers.Dense(100, activation="relu", name="d6")(x)

    

    p1 = layers.Dense(3, activation="linear", name="p1")(x)

    #x = layers.Dropout(0.05)(x)

    p2 = layers.Dense(3, activation="relu", name="p2")(x)

    #x = layers.Dropout(0.05)(x)

    preds = layers.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 

                     name="preds")([p1, p2])

    

    model = models.Model(z, preds, name="ANN")

    #model.compile(loss=qloss, optimizer="adam", metrics=[Laplace_log_Likelihood_score]) #.775

    model.compile(loss=mloss(0.8), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, 

                beta_2=0.999, epsilon=None, decay=0.009, amsgrad=False), metrics=[Laplace_log_Likelihood_score])

    return model
model = make_model()

print(model.summary())

print(model.count_params())
y = train['FVC'].values

z = train[FE].values

ze = submission[FE].values

pe = np.zeros((ze.shape[0], 3))

pred = np.zeros((z.shape[0], 3))

delta = np.zeros((z.shape[0], 3))
# kf = KFold(n_splits=NFOLD)

gkf = GroupKFold(n_splits=NFOLD) 

print(gkf)
%%time

temp_val = []

temp_train = []

cnt = 0

for train_idx, val_idx in gkf.split(z, groups=train['Patient']):





    cnt += 1

    print(f"FOLD {cnt}")

    

    model.fit(z[train_idx], y[train_idx], batch_size=BATCH_SIZE, epochs= EPOCHS, 

            validation_data=(z[val_idx], y[val_idx]), verbose=0) #

    

    

    training   = model.evaluate(z[train_idx], y[train_idx], verbose=0, batch_size=BATCH_SIZE)

    validation = model.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE)

    print('training', training)

    print('validation', validation)

    

    temp_train.append(training)

    

    

    temp_val.append(validation)

    

    #print("predict val...")

    pred[val_idx] = model.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)

    #print("predict test...")

    pe += model.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD

    delta += model.predict(z) / NFOLD
# Scoring



o_clipped = np.maximum(delta[:,2] - delta[:,0], 70)

delta = np.minimum(np.abs(delta[:, 1] - y), 1000)

sqrt = (np.sqrt((2)))

score = (-(sqrt * (delta))/(o_clipped)) - tf.math.log(sqrt * o_clipped)



logL_Score = np.mean(score)

print(np.mean(score))
# print(temp_train)

# print(temp_val)
# sigma_opt mean_absolute_error   sigma_mean  unc_mean 

mean_absolute_error = mean_absolute_error(y, pred[:, 1]) # find  UNC 

unc = pred[:,2] - pred[:, 0]

unc_mean = np.mean(unc)
# ########### RUN FIRST TIME ONLY ################

stats = pd.DataFrame()

index = 0
data = [[index, logL_Score, mean_absolute_error, unc.mean(), unc.min(),  unc.max(), (unc>=0).mean(), BATCH_SIZE, EPOCHS,  NFOLD,  SEED]]

columns = ['Run Kernal','logL_Score', 'mean_abs_err', 'unc.mean', 'unc.min', 'unc.max',  '(unc>=0).mean','batch_size', 'epochs', 'NFOLD','seed']

kernal_stats = pd.DataFrame(data, columns=columns)

# print("current kernal state")

kernal_stats
#temp = pd.read_csv('./kernal.csv')

# temp = stats.tail(1)

# temp_1 = temp['Run Kernal']

# print(temp_1.shape)

# print(temp_1)
stats = pd.concat([stats, kernal_stats])

stats.to_csv('kernal.csv', index = False)

index+=1



# print('kernal stats of every version')

stats
print('we are using fix seed value always to avoid RANDOMIZATION (NEED TO GET SAME RESULT)')

print('Seed value          =',SEED)

print('Number of folds     =',BATCH_SIZE)

print('Number of epochs    =',EPOCHS)



print('\nmean_absolute_error =',mean_absolute_error)

#print('unc_mean            =',unc_mean)



print('unc_mean            =',unc.mean())

print('unc_min             =',unc.min())

print('unc_max             =',unc.max())

print('unc_mean            =',(unc>=0).mean())
idxs = np.random.randint(0, y.shape[0], 100)

plt.plot(y[idxs], label="ground truth")

plt.plot(pred[idxs, 0], label="q25")

plt.plot(pred[idxs, 1], label="q50")

plt.plot(pred[idxs, 2], label="q75")

plt.legend(loc="best")

plt.show()
sns.distplot(unc)

#plt.(unc)

plt.title("uncertainty in prediction")

plt.savefig('sns{}.png'.format(index))

plt.show()
plt.hist(unc)

plt.title("uncertainty in prediction")

plt.savefig('plt{}.png'.format(index))

plt.show()
submission.head()
submission['FVC1'] = pe[:, 1]

submission['Confidence1'] = pe[:, 2] - pe[:, 0]
subm = submission[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
subm.loc[~subm.FVC1.isnull()].head(10)
# sigma_opt mean_absolute_error   sigma_mean  unc_mean 

subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']

if unc_mean<70:

    subm['Confidence'] = mean_absolute_error

else:

    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']
subm.head()
sns.distplot(subm.FVC)
sns.distplot(subm.Confidence)
subm.describe().T
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)
# !pip install jovian --upgrade --quiet

# jovian.commit(project='osic-new-era')

# import jovian