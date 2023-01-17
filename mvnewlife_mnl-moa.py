# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
from numpy import savez_compressed
import SimpleITK as sitk

from time import time

# Required Imports and loading up a scan for processing as presented by Guide Zuidhof

%matplotlib inline

import pydicom
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import zoom
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns 

from skimage import measure, morphology, segmentation
from skimage.transform import resize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from sklearn.model_selection import KFold,GroupKFold,TimeSeriesSplit,train_test_split, StratifiedKFold
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras as K
import tensorflow.keras.layers as L
import tensorflow.keras.backend as B
import tensorflow.keras.callbacks as C
from tensorflow_addons.optimizers import RectifiedAdam

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import History, EarlyStopping
from tensorflow.keras.regularizers import l1_l2,l2,l1
from tensorflow.keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, MaxPooling3D, BatchNormalization
from tensorflow.keras.layers import UpSampling2D, UpSampling3D, Flatten, Reshape, Conv3DTranspose,Conv3D
def set_seed(seed):
    '''
    from os import environ
    environ["PYTHONHASHSEED"] = '0'
    environ["CUDA_VISIBLE_DEVICES"]='-1'
    environ["TF_CUDNN_USE_AUTOTUNE"] ='0'
    '''

    from numpy.random import seed as np_seed
    np_seed(seed)
    import random
    random.seed(seed)
    from tensorflow import random
    random.set_seed(seed)
SEED = 11
set_seed(SEED)
testFeP = '../input/lish-moa/test_features.csv'
trainFeP = '../input/lish-moa/train_features.csv'
trainTaSP = '../input/lish-moa/train_targets_scored.csv'
trainTaNP = '../input/lish-moa/train_targets_nonscored.csv'
sample_submission_path = '../input/lish-moa/sample_submission.csv'
trainFe = pd.read_csv(trainFeP)
trainFe_col = trainFe.columns
trainFe.describe(),trainFe.info()
cols_list=trainFe.columns
cols_list
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
trainFe.drop(columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose']).max().max(),trainFe.drop(columns=['sig_id', 'cp_type', 'cp_time', 'cp_dose']).min().min()

# gaussian percent point function
from scipy.stats import norm
# define probability
p = 0.025
# retrieve value <= probability
value = norm.ppf(p)
print(value)
# confirm with cdf
p = norm.cdf(value)
print(p)
data = trainFe[cols_list[16]].copy()
std_data = data.std()
mean_data = data.mean()
print(data.mean(),data.std())
pyplot.hist(data, bins=100)
pyplot.show()
qqplot(data, line='s')
pyplot.show()
lam = 1e-3
for k in range(len(data)):
    if norm.cdf(data[k])> (1-lam):
        #print(k, data[k], norm.cdf(data[k]))
        data[k]= norm.ppf(1-lam)
    elif norm.cdf(data[k]) < lam:
        #print(k, data[k], norm.cdf(data[k]))
        data[k]=norm.ppf(lam)
print(data.mean(),data.std())
pyplot.hist(data, bins=100)
pyplot.show()
qqplot(data, line='s')
pyplot.show()
result = anderson(data)
print('Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
	sl, cv = result.significance_level[i], result.critical_values[i]
	if result.statistic < result.critical_values[i]:
		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
	else:
		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
# normality test
stat, p = normaltest(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
gaussian = np.zeros(len(cols_list))
std_list = np.zeros(len(cols_list))
alpha = 0.05
for k in range(4,len(cols_list)):
    data = trainFe[cols_list[k]]
    # normality test
    stat, p = normaltest(data)
    if p <= alpha: gaussian[k] = 1
    std_list[k]= data.std()
    '''
    print(data.mean(),data.std())
    qqplot(data, line='s')
    pyplot.show()
    '''
print('There are %i not gaussian' %gaussian.sum())
data = std_list[4:]
print(data.mean(),data.std())
pyplot.hist(data, bins=100)
pyplot.show()
from scipy import stats

correlation = trainFe.corr(method='pearson')
correlation.shape
trainFe.cp_type.unique()
lista= trainFe.loc[trainFe.cp_type == 'ctl_vehicle','sig_id']
lista
kk=0
for i in lista:
    #trainFe = trainFe.drop(trainFe.loc[trainFe.sig_id == i].index)
    kk+=1

print(kk)
testFe = pd.read_csv(testFeP)
testFe_col = testFe.columns
testFe.describe(),testFe.info()
lesta= testFe.loc[testFe.cp_type == 'ctl_vehicle','sig_id']
lesta
trainTaS = pd.read_csv(trainTaSP)
trainTaS_col = trainTaS.columns
trainTaS.describe(),trainTaS.info()
k=0
kk=0
for i in lista:
    app=trainTaS.loc[trainTaS.sig_id == i]
    k+=app.drop(columns=['sig_id']).max(axis=1).values[0]
    kk+=1
print(k,kk)
kk=0
for i in lista:
    #trainTaS = trainTaS.drop(trainTaS.loc[trainTaS.sig_id == i].index)
    kk+=1
print(kk)
app.columns
#s.cummax(skipna=False)
k=app.drop(columns=['sig_id']).max(axis=1)
k.values
trainFe.info(), trainTaS.info()
trainTaN = pd.read_csv(trainTaNP)
trainTaN_col = trainTaN.columns
trainTaN.describe(), trainTaN.info()
# it could be used to create new features
sub = pd.read_csv(sample_submission_path)
sub_col = sub.columns
sub.describe(), sub.info()
cols = sub.drop(columns=['sig_id']).columns
for i in lesta:
    sub.loc[sub.sig_id == i,cols]=0
sub.describe()
trainFe['cp_type'].unique()
# column 'cp_type' has only 2 values: 'trt_cp' and 'ctl_vehicle'

lenn = np.array(trainFe['cp_type'])
plt.hist(lenn, bins=3)
plt.show()
#print(lenn.std(),lenn.mean())
# column 'cp_dose' has only 2 values: 'D1' and 'D2'

lenn = np.array(trainFe['cp_dose'])
plt.hist(lenn, bins=3)
plt.show()
#print(lenn.std(),lenn.mean())
# column 'cp_time' has only 3 values: 24, 48, 72

lenn = np.array(trainFe['cp_time'])
plt.hist(lenn, bins=5)
plt.show()
print(lenn.std(),lenn.mean())
''' 
------ sostitusco valori ----------

appoggio['Sex']=appoggio['Sex'].replace(to_replace =['male','female'],value=[0,1]).astype(int)

------ normalizzo le feature ------

from sklearn.preprocessing import MinMaxScaler

features = ['Fare','Fare_B',"Age","family","SibSp","Parch","Pclass",'Embarked','Title','Cabin','Ticket',"Pre_TK","Post_TK"]
mms = MinMaxScaler()

to_norm = appoggio[features]
appoggio[features] = mms.fit_transform(to_norm)

------ creo dummy -----------------

appoggio = pd.get_dummies(appoggio,columns=["Sex"])

'''
trainFe.describe()
outliers_remuval = 2
if outliers_remuval == 1:
    lam = 5e-2
    alto = norm.ppf(1-lam)
    basso = norm.ppf(lam)
    for i in range(4,len(cols_list)):
        data = trainFe[cols_list[i]].copy()
        for k in range(len(data)):
            data[k]=max(min(data[k],alto),basso)
        trainFe[cols_list[i]] = data
        
        data = testFe[cols_list[i]].copy()
        for k in range(len(data)):
            data[k]=max(min(data[k],alto),basso)
        testFe[cols_list[i]] = data
elif outliers_remuval == 2:
    alto = 1
    basso = -1
    for i in range(4,len(cols_list)):
        data = trainFe[cols_list[i]].copy()
        data[data>1]=1
        data[data<-1]=-1
        trainFe[cols_list[i]] = data
        
        data = testFe[cols_list[i]].copy()
        data[data>1]=1
        data[data<-1]=-1
        testFe[cols_list[i]] = data
if outliers_remuval < 2:
    features = trainFe.drop(columns=['sig_id','cp_type','cp_time','cp_dose']).columns
    to_norm = trainFe[features]
    mms = MinMaxScaler()
    trainFe[features] = mms.fit_transform(to_norm)
if outliers_remuval < 2:
    features = testFe.drop(columns=['sig_id','cp_type','cp_time','cp_dose']).columns
    to_norm = testFe[features]
    mms = MinMaxScaler()
    testFe[features] = mms.fit_transform(to_norm)
trainFe.describe()
testFe.describe()
'''
trainFe['cp_dose'] = trainFe['cp_dose'].replace(to_replace = ['D1','D2'],value=[0,1]).astype(int)
trainFe['cp_type'] = trainFe['cp_type'].replace(to_replace = ['trt_cp','ctl_vehicle'],value=[0,1]).astype(int)
'''
trainFe = pd.get_dummies(trainFe, columns = ['cp_dose'])
trainFe = pd.get_dummies(trainFe, columns = ['cp_type'])
trainFe = pd.get_dummies(trainFe, columns = ['cp_time'])

trainFe.describe()
'''
testFe['cp_dose'] = testFe['cp_dose'].replace(to_replace = ['D1','D2'],value=[0,1]).astype(int)
testFe['cp_type'] = testFe['cp_type'].replace(to_replace = ['trt_cp','ctl_vehicle'],value=[0,1]).astype(int)
'''
testFe = pd.get_dummies(testFe, columns = ['cp_dose'])
testFe = pd.get_dummies(testFe, columns = ['cp_type'])
testFe = pd.get_dummies(testFe, columns = ['cp_time'])
testFe.describe()
print('trainFe:\n',trainFe.columns,'\ntestFe:\n',testFe.columns,'\ntrainTaS:\n',trainTaS.columns,'\ntrainTaN:\n',trainTaN.columns,'\nsub:\n',sub.columns)
'''
tra = pd.read_csv(f"{ROOT}/train.csv")
tra.drop_duplicates(keep=False, inplace=True, subset=['Patient','Weeks'])
chunk = pd.read_csv(f"{ROOT}/test.csv")


tr['WHERE'] = 'train'
chunk['WHERE'] = 'val'
sub['WHERE'] = 'test'
data = tr.append([chunk, sub])

sub = pd.read_csv(f"{ROOT}/sample_submission.csv")
sub['Patient'] = sub['Patient_Week'].apply(lambda x:x.split('_')[0])
sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))
sub =  sub[['Patient','Weeks','Confidence','Patient_Week']]
sub = sub.merge(chunk.drop('Weeks', axis=1), on="Patient") # appiccica a sub le colonne di Chunk a parte week

'''
'''
trainFe['WHERE'] = 'train' 
trainFe = trainFe.merge(trainTaS, on='sig_id')
trainFe = trainFe.merge(trainTaN, on='sig_id') 
trainFe.info()
testFe['WHERE'] = 'test' sub['WHERE'] = 'sub'
data = trainFe.append([testFe,sub])
'''
train = trainFe.drop(columns=['sig_id']).values
real = trainTaS.drop(columns=['sig_id']).values
test = testFe.drop(columns=['sig_id']).values
real.sum()
trainFe.info()
trainFe.describe(include=[object])  ,trainTaS.describe(include=[object])
trainTaS.describe()
testFe.info()
X_train = train#[:512]
y_real = real#[:512]
X_train.shape, y_real.shape
XT = np.append(X_train,test,axis=0)
X_train.shape, test.shape, XT.shape
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

M = mean(X_train.T, axis=1)
print(M.shape,'M\n',M)
Mt = mean(test.T, axis=1)
MT = mean(XT.T, axis=1)
# center columns by subtracting column means
C = X_train - M
print(C.shape,'C\n',C)
Ct = test - Mt
CT = XT- MT
# calculate covariance matrix of centered matrix, that is, the joint probability for two features shape: (features, features)
V = cov(C.T)
print(V.shape,'V = cov(C.T)\n',V)
Vt = cov(Ct.T)
VT = cov(CT.T)
# factorize covariance matrix
values, vectors = eig(V)
print(vectors.shape,'vectors\n',vectors)
values_t, vectors_t = eig(Vt)
values_T, vectors_T = eig(VT)
print(values.shape,'values\n',np.sort(values))
print(values_t.shape,'values_t\n',np.sort(values_t))
print(values_T.shape,'values_T\n',np.sort(values_T))
C.shape, Ct.shape, CT.shape
PCA_LIMIT = 1.5e-3

valori = values_T.copy()
k=0
for i in range(len(values_T)):
    if abs(values_T[i]) < PCA_LIMIT:
        
        CT=np.delete(CT,k+i,1)
        Ct=np.delete(Ct,k+i,1)
        C=np.delete(C,k+i,1)
        valori=np.delete(valori,k+i)
        vectors=np.delete(vectors,k+i,0)
        vectors=np.delete(vectors,k+i,1)
        
        vectors_t=np.delete(vectors_t,k+i,1)
        vectors_t=np.delete(vectors_t,k+i,0)

        vectors_T=np.delete(vectors_T,k+i,0)
        vectors_T=np.delete(vectors_T,k+i,1)
        
        k-=1
print(k)   
C.shape, Ct.shape, CT.shape, valori.shape, vectors.shape, vectors_t.shape, vectors_T.shape
P = vectors.T.dot(C.T) # P.T is a projection of C - same projection can be applied to a selection of features based on eigenvalues
print(P.shape,'P\n',P)
PT = P.T
print(PT.shape,'PT\n',PT) 
P_t = vectors_t.T.dot(Ct.T) # P.T is a projection of C - same projection can be applied to a selection of features based on eigenvalues
print(P_t.shape,'P_t\n',P_t)
P_tT = P_t.T
print(P_tT.shape,'P_tT\n',P_tT) 
PCA = False
if PCA == True:
    train = PT
    X_train = PT
    test = P_tT
    print(train.shape, X_train.shape, test.shape)
else:
    train = trainFe.drop(columns=['sig_id']).values
    X_train = train
    real = trainTaS.drop(columns=['sig_id']).values
    test = testFe.drop(columns=['sig_id']).values
    print(train.shape, X_train.shape, test.shape, real.shape)
L2 = 0
SEED = 23
set_seed(SEED)
INITIALIZER = tf.keras.initializers.GlorotUniform()
MOMENTUM = 0.9

try: 
    del model
    tf.keras.backend.clear_session()
    print('session cleared')
except Exception as OSError:
    pass

checkpoint_filepath = 'checkpointWeight'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
train.shape
def conv_model(init, regul):
    model_input = K.Input(shape = (train.shape[1],1), name="input")
    x = L.Conv1D(96,2,2,'same')(model_input)
    x = L.Conv1D(64,2,2,'same')(x)
    x = L.Conv1D(32,10,1,'valid')(x)
    x = L.Conv1D(1,1,1,'same')(x)
    model_output = L.Flatten()(x)
    model_conv = K.Model(model_input, model_output, name="output")
    
    return model_conv

model_conv = conv_model(tf.keras.initializers.GlorotUniform(),0)
K.utils.plot_model(model_conv, "conv_model.png", show_shapes=True)
model_conv.summary()
train.shape[1]
MOMENTUM = 0.9
# 'relu', activity_regularizer=l2(regul),
def madel(init, regul):
    model_input = K.Input(shape = (train.shape[1]), name="input")

    x = BatchNormalization(momentum=MOMENTUM)(model_input)
    x = Dense(2048,activation='swish', kernel_initializer=init)(x)
 
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Dropout(0.4)(x)
    x = Dense(1024,activation='swish', kernel_initializer=init)(x)
    
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Dropout(0.4)(x)
    x = Dense(512,activation='swish', kernel_initializer=init)(x)
    
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Dropout(0.4)(x)
    x = Dense(256,activation='swish', kernel_initializer=init)(x)
    
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Dropout(0.4)(x)
    model_output = Dense(206,activation='sigmoid', kernel_initializer=init)(x)

    model_base = K.Model(model_input, model_output, name="output")
    
    return model_base

def madelWN(init, regul):
    model_input = K.Input(shape = (train.shape[1]), name="input")

    x = BatchNormalization(momentum=MOMENTUM)(model_input)
 
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Dropout(0.5)(x)
    x = tfa.layers.WeightNormalization(Dense(1024,activation='swish', kernel_initializer=init))(x)
    
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Dropout(0.5)(x)
    x = tfa.layers.WeightNormalization(Dense(512,activation='swish', kernel_initializer=init))(x)
    
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Dropout(0.4)(x)
    x = tfa.layers.WeightNormalization(Dense(256,activation='swish', kernel_initializer=init))(x)
    
    x = BatchNormalization(momentum=MOMENTUM)(x)
    x = Dropout(0.4)(x)
    model_output = tfa.layers.WeightNormalization(Dense(206,activation='sigmoid', kernel_initializer=init))(x)

    model_base = K.Model(model_input, model_output, name="output")
    
    return model_base

model_base = madel(tf.keras.initializers.GlorotUniform(),0)
K.utils.plot_model(model_base, "madel.png", show_shapes=True)
model_base.summary()
def score_MoA(y_true, y_pred, dtype = 'float64'):
    #preda = tf.math.abs(y_pred)
    preda = tf.dtypes.cast(y_pred, 'float64')
    predb = tf.math.minimum(preda, tf.constant(1-1e-15,dtype='float64'))
    predb = tf.dtypes.cast(predb, 'float64')
    pred = tf.math.maximum(predb, tf.constant(1e-15,dtype='float64'))
    pred = tf.dtypes.cast(pred, 'float64')
    #print('pred',pred,'tf.math.log((1 - pred)',tf.math.log((1 - pred)))
    yt = tf.dtypes.cast(y_true, 'float64')
    m1 = tf.math.multiply(yt, tf.math.log(tf.math.maximum(tf.constant(1e-15,dtype='float64'),pred)))
    m2 = tf.math.multiply((1 - yt), tf.math.log(tf.math.maximum(tf.constant(1e-15,dtype='float64'),1-pred)))
    metric = m1+m2
    return -B.mean(B.mean(metric, axis = -1), axis=0)

def loss_MoA(dtype = 'float64'):
    def losss(y_true, y_pred, dtype = 'float64'):
        return score_MoA(y_true, y_pred)
    return losss
    
def rmse(y_true, y_pred):
    return B.sqrt(B.mean(B.square(y_pred - y_true), axis=-1))

EPOCHS = 200
RIF = 9
L2 = 0.0
LR_RA = 1e-2
MIN_LR = 1e-4
BATCH_SIZE = 256
logLoss = tf.keras.losses.BinaryCrossentropy()

myloss = logLoss # loss_MoA()

train_phase = False
SEED = 23
set_seed(SEED)

start_all_at = time()
if train_phase:
    k_folds = 5
    kf = KFold(n_splits=k_folds, random_state=2, shuffle=True)
    i = 1
    for train_index, test_index in kf.split(X_train):
        trainData = X_train[train_index]
        valData = X_train[test_index]
        trainLabels = y_real[train_index]
        valLabels = y_real[test_index]

        try: 
            del model
            tf.keras.backend.clear_session()
            print('session cleared')
        except Exception as OSError:
            pass

        SEED = 23
        set_seed(SEED)
        INITIALIZER = tf.keras.initializers.GlorotUniform()
        tot_steps = max(int(((len(trainData)//BATCH_SIZE))*RIF),1)
        opt = tfa.optimizers.RectifiedAdam(lr=LR_RA,total_steps=tot_steps, warmup_proportion=0.1, min_lr=MIN_LR) #total_steps=336000


        model = madelWN(INITIALIZER, L2)
        #model = conv_model(INITIALIZER, L2)
        model.compile(loss=myloss, optimizer=opt, metrics=[score_MoA])

        history = History()
        print('Iniizio addestramento fold',i)
        start_at = time()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=50, restore_best_weights=True)

        model.fit(trainData, trainLabels, epochs=EPOCHS, validation_data=(valData,valLabels), batch_size=BATCH_SIZE,
                  verbose=1, callbacks=[history, early_stopping])

        print(len(history.history['loss']))
        exec_time = time() - start_at
        print("\nTempo totale di addestramento fold: %i %d minuti e %d secondi" % (i, exec_time/60, exec_time%60),'\n')
        i +=1
else:
    trainData = X_train
    trainLabels = y_real

    try: 
        del model
        tf.keras.backend.clear_session()
        print('session cleared')
    except Exception as OSError:
        pass

    SEED = 23
    set_seed(SEED)
    INITIALIZER = tf.keras.initializers.GlorotUniform()
    tot_steps = max(int(((len(trainData)//BATCH_SIZE))*RIF),1)
    opt = tfa.optimizers.RectifiedAdam(lr=LR_RA,total_steps=tot_steps, warmup_proportion=0.1, min_lr=MIN_LR) #total_steps=336000


    model = madel(INITIALIZER, L2)
    model.compile(loss=myloss, optimizer=opt, metrics=[score_MoA])

    history = History()
    print('Iniizio addestramento')
    start_at = time()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=50, restore_best_weights=True)

    model.fit(trainData, trainLabels, epochs=EPOCHS, batch_size=BATCH_SIZE,
              verbose=1, callbacks=[history, early_stopping])

    print(len(history.history['loss']))
    exec_time = time() - start_at
    
exec_time = time() - start_all_at
print("\nTempo totale di addestramento: %d minuti e %d secondi" % (exec_time/60, exec_time%60),'\n')
def plot_model(model_history, epochs, metric):
    fraz = 2
    starting_point = int(epochs/fraz)
    plt.figure(figsize=(14,10))
    plt.title("Metric")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.plot(model_history[metric][epochs - starting_point:],color='green')
    try: plt.plot(model_history['val_' + metric][epochs - starting_point:],color='red')
    except: no=1

    plt.figure(figsize=(14,10))
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(model_history['loss'][epochs - starting_point:],color='green')
    try: plt.plot(model_history['val_loss'][epochs - starting_point:],color='red')
    except: no=1
# def plot_model(model_history, starting_point, metric):
plot_model(history.history, EPOCHS, 'score_MoA' )
trainData.shape
testFe.columns
testFe.describe(include=[object]),testFe.describe()
testFe.columns
#pred = model.predict(testFe.drop(columns=['sig_id']).values)
pred = model.predict(test)
x_tr, y_tr = train[512:1024], real[512:1024]
prad = model.predict(x_tr)
model.evaluate(x_tr, y_tr,verbose=0, batch_size=BATCH_SIZE)
'''
predbb=np.minimum(prad, 1-1e-15)
preddd=np.maximum(predbb, 1e-15)
m1=y_tr*np.log(preddd)+(1-y_tr)*np.log(1-preddd)
m2 = m1.mean(axis = -1)
scoreMoA = -m2.mean(axis = 0)
scoreMoA
'''
len(pred)
col = trainTaS.drop(columns=['sig_id']).columns
col
score_MoA(y_tr,prad)

predDF = pd.DataFrame(pred, columns=col)
pradDF = pd.DataFrame(prad, columns=col)
pradTaSDF = pd.DataFrame(y_tr, columns=col)
predDF.info()
pradDF.info()
predDF.describe()
pradDF.describe()
pradTaSDF.describe()
predDF.describe()
predDF.head(20)
output = pd.DataFrame({'sig_id': testFe.sig_id})
to_csv = pd.concat([output, predDF], axis = 1)
to_csv.head(20)
to_csv.describe()
cols = to_csv.drop(columns=['sig_id']).columns
for i in lesta:
    to_csv.loc[to_csv.sig_id == i,cols]=0
to_csv.describe()
# quelli con lesta vanno azzerati
trainTaS.describe()
if not train_phase:
    to_csv.to_csv('submission.csv', index=False)
    print("Your submission was successfully saved!")
