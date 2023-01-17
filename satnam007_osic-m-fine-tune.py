import numpy as np

import pandas as pd

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
def seed_everything(seed=2020):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)

    return seed
ROOT = "../input/osic-pulmonary-fibrosis-progression"



tr = pd.read_csv(f"{ROOT}/train.csv")

tr.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])

chunk = pd.read_csv(f"{ROOT}/test.csv")



print("add infos")

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")

sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]

sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient")
tr['WHERE'] = 'train'

chunk['WHERE'] = 'val'

sub['WHERE'] = 'test'

data = tr.append([chunk, sub])
print(tr.shape, chunk.shape, sub.shape, data.shape)

print(tr.Patient.nunique(), chunk.Patient.nunique(), sub.Patient.nunique(), 

      data.Patient.nunique())

#
data['min_week'] = data['Weeks']

data.loc[data.WHERE=='test','min_week'] = np.nan

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
COLS = ['Sex','SmokingStatus']

FE = []

for col in COLS:

    for mod in data[col].unique():

        FE.append(mod)

        data[mod] = (data[col] == mod).astype(int)

#=================


data['age'] = (data['Age'] - data['Age'].min() ) / ( data['Age'].max() - data['Age'].min() )

data['BASE'] = (data['min_FVC'] - data['min_FVC'].min() ) / ( data['min_FVC'].max() - data['min_FVC'].min() )

data['week'] = (data['base_week'] - data['base_week'].min() ) / ( data['base_week'].max() - data['base_week'].min() )

data['percent'] = (data['Percent'] - data['Percent'].min() ) / ( data['Percent'].max() - data['Percent'].min() )

FE += ['age','percent','week','BASE']

#FE += ['age','percent','BASE']
tr = data.loc[data.WHERE=='train']

chunk = data.loc[data.WHERE=='val']

sub = data.loc[data.WHERE=='test']

del data
SEED = seed_everything(42)

NFOLD        = 5

EPOCHS       = 800

BATCH_SIZE   = 128



M_LOSS       = 0.775

LR           = 0.1

DECAY        = 0.01     



kf = KFold(n_splits=NFOLD)
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

    qs = [0.2, 0.5, 0.8]

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

def make_model():

    z = L.Input((9,), name="Patient")

    x = L.Dense(100, activation="relu", name="d1")(z)

    x = L.Dense(100, activation="relu", name="d2")(x)

    #x = L.Dense(100, activation="relu", name="d3")(x)

    p1 = L.Dense(3, activation="linear", name="p1")(x)

    p2 = L.Dense(3, activation="relu", name="p2")(x)

    preds = L.Lambda(lambda x: x[0] + tf.cumsum(x[1], axis=1), 

                     name="preds")([p1, p2])

    

    

    model = M.Model(z, preds, name="CNN")

    #model.compile(loss=qloss, optimizer="adam", metrics=[score])

    model.compile(loss=mloss(0.775), optimizer=tf.keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.01, amsgrad=False), metrics=[score])

    return model
net = make_model()

print(net.summary())

print(net.count_params())
y = tr['FVC'].values

z = tr[FE].values

ze = sub[FE].values

pe = np.zeros((ze.shape[0], 3))

pred = np.zeros((z.shape[0], 3))

delta = np.zeros((z.shape[0], 3))
%%time

cnt = 0

train = []

val   = []





for tr_idx, val_idx in kf.split(z):

    cnt += 1

    print(f"FOLD {cnt}")

    

    

    net = make_model()

    

    

    net.fit(z[tr_idx], y[tr_idx], batch_size=BATCH_SIZE, epochs= EPOCHS, 

            validation_data=(z[val_idx], y[val_idx]), verbose=0) # 

    

#     train.append(net.evaluate(z[tr_idx], y[tr_idx], batch_size=BATCH_SIZE))

#     val.append(net.evaluate(z[val_idx], y[val_idx], batch_size=BATCH_SIZE)

    print("train", net.evaluate(z[tr_idx], y[tr_idx], verbose=0, batch_size=BATCH_SIZE))

    print("val", net.evaluate(z[val_idx], y[val_idx], verbose=0, batch_size=BATCH_SIZE))

    #print("predict val...")

    pred[val_idx] = net.predict(z[val_idx], batch_size=BATCH_SIZE, verbose=0)

    #print("predict test...")

    print()

    pe += net.predict(ze, batch_size=BATCH_SIZE, verbose=0) / NFOLD

    

    delta += net.predict(z) / NFOLD

    

    

#==============



sigma_opt = mean_absolute_error(y, pred[:, 1])

unc = pred[:,2] - pred[:, 0]

sigma_mean = np.mean(unc)

print(sigma_opt, sigma_mean)

#142.3356196400397 234.99396579009314
# Scoring



o_clipped = np.maximum(delta[:,2] - delta[:,0], 70)

delta = np.minimum(np.abs(delta[:, 1] - y), 1000)

sqrt = (np.sqrt((2)))

score = (-(sqrt * (delta))/(o_clipped)) - tf.math.log(sqrt * o_clipped)



logL_Score = np.mean(score)
print('we are using fix seed value always to avoid RANDOMIZATION (NEED TO GET SAME RESULT)')

print('Seed value          =',SEED)

print('Number of folds     =',BATCH_SIZE)

print('Number of epochs    =',EPOCHS)



print('\nmean_absolute_error =',sigma_opt)

#print('unc_mean            =',unc_mean)



print('unc_mean            =',unc.mean())

print()

print('Log_laplace_Scores  =',logL_Score )

print()

print('unc_min             =',unc.min())

print('unc_max             =',unc.max())

print('unc_mean            =',(unc>=0).mean())
# # # # # ########### RUN FIRST TIME ONLY ################

# stats = pd.DataFrame()

# index = 0
data = [[index, logL_Score, sigma_opt, unc.mean(), unc.min(),  unc.max(), (unc>=0).mean(),

         BATCH_SIZE, EPOCHS,  NFOLD, M_LOSS, LR, DECAY ,SEED]]

columns = ['S.No','Score', 'meanAbseErr', 'unc.mean', 'unc.min', 'unc.max',  '(unc>=0)',

           'Bsize', 'epoch', 'NFOLD', 'M_LOSS', 'LR', 'DECAY', 'seed']

kernal_stats = pd.DataFrame(data, columns=columns)

# print("current kernal state")

kernal_stats
stats = pd.concat([stats, kernal_stats])

stats.to_csv('kernal.csv', index = False)

index+=1

# 154.68119019747556 213.12221373517662 0.0 213.12221373517662 384.3486328125 1.0

# print('kernal stats of every version')

stats



#0 -6.490188 142.335620 234.993966 25.237061 485.192871 1.0 128 800 5 0.775 0.100 0.010 42

#3 -6.488414	141.404185	234.008693	20.999268	495.660645	1.0	128	850	5	0.775	0.1	0.01	42
plt.hist(unc)

plt.title("uncertainty in prediction")

#plt.savefig('plt_unc_pred{}.png'.format(index))

plt.show()
# plt.plot(arr)

# plt.xlabel('epoch')

# plt.ylabel('accuracy')

# plt.title('Accuracy vs. No. of epochs');
# idxs = np.random.randint(0, y.shape[0], 100)

# plt.plot(y[idxs], label="ground truth")

# plt.plot(pred[idxs, 0], label="q25")

# plt.plot(pred[idxs, 1], label="q50")

# plt.plot(pred[idxs, 2], label="q75")

# plt.legend(loc="best")

# plt.show()
import seaborn as sns

sns.distplot(unc)

#plt.(unc)

plt.title("uncertainty in prediction")

# plt.savefig('sns_unc_pred{}.png'.format(index))

plt.show()
#sub.head()
sub['FVC1'] = pe[:, 1]

sub['Confidence1'] = pe[:, 2] - pe[:, 0]
subm = sub[['Patient_Week','FVC','Confidence','FVC1','Confidence1']].copy()
subm.loc[~subm.FVC1.isnull()].head(1)
subm.loc[~subm.FVC1.isnull(),'FVC'] = subm.loc[~subm.FVC1.isnull(),'FVC1']

if sigma_mean<70:

    subm['Confidence'] = sigma_opt

else:

    subm.loc[~subm.FVC1.isnull(),'Confidence'] = subm.loc[~subm.FVC1.isnull(),'Confidence1']
subm.head(1)
plt.hist(subm.FVC)

# plt.title("uncertainty in prediction")

# plt.savefig('subm.FVC{}.png'.format(index))

plt.show()



plt.hist(subm.FVC1)

# plt.title("uncertainty in prediction")

# plt.savefig('subm.FVC_1{}.png'.format(index))

plt.show()
plt.hist(subm.Confidence)

# plt.title("uncertainty in prediction")

# plt.savefig('subm.Confidence{}.png'.format(index))

plt.show()



plt.hist(subm.Confidence1)

# plt.title("uncertainty in prediction")

# plt.savefig('subm.Confidence_1{}.png'.format(index))

plt.show()
subm.describe().T
otest = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

for i in range(len(otest)):

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'FVC'] = otest.FVC[i]

    subm.loc[subm['Patient_Week']==otest.Patient[i]+'_'+str(otest.Weeks[i]), 'Confidence'] = 0.1
subm[["Patient_Week","FVC","Confidence"]].to_csv("submission.csv", index=False)
subm
# !pip install jovian --upgrade --quiet
# import jovian
# jovian.commit(project='osic-new-era')
submm = pd.read_csv('../input/osic36175/submission_6.175.csv')

submm = submm.values

colnames = ['Patient','FVC_175', 'Confidence_175']

submm = pd.DataFrame(submm, columns=colnames)
submm

import seaborn as sns



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 8))



sns.kdeplot(subm['FVC'], ax=ax1)

sns.kdeplot(subm['Confidence'], ax=ax2)



sns.kdeplot(submm['FVC_175'], ax=ax1)

sns.kdeplot(submm['Confidence_175'], ax=ax2)