#load with pandas, manipulate with numpy, plot with matplotlib

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



#ML - we will classify using a naive xgb with stratified cross validation

import xgboost as xgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss







#filenames

inputFolder = "../input/"

trainSet = 'train.json'

#testSet = 'test.json'

subName = 'iceberg-svd-xgb-3fold.csv'

#load data

trainDF = pd.read_json(inputFolder+trainSet)

#testDF = pd.read_json(inputFolder+testSet)
trainDF.head(10)
#get numpy arrays for train/test data, prob there is a more pythonic approach

band1 = trainDF['band_1'].values

im1 = np.zeros((len(band1),len(band1[0])))

for j in range(len(band1)):

    im1[j,:]=np.asarray(band1[j])

    

band2 = trainDF['band_2'].values

im2 = np.zeros((len(band2),len(band2[0])))

for j in range(len(band2)):

    im2[j,:]=np.asarray(band2[j])

    

from sklearn.preprocessing import normalize

def distanc(X,Y):

    Z=X

    for yi in range(0,len(X)):

        Z[yi]=angle_between((X[yi],Y[yi],0),(1,0,0))

    return np.reshape(Z,(75,75))



def unit_vector(vector):

    """ Returns the unit vector of the vector.  """

    return vector / np.linalg.norm(vector)



def angle_between(v1, v2):

    """ Returns the angle in radians between vectors 'v1' and 'v2'::



            >>> angle_between((1, 0, 0), (0, 1, 0))

            1.5707963267948966

            >>> angle_between((1, 0, 0), (1, 0, 0))

            0.0

            >>> angle_between((1, 0, 0), (-1, 0, 0))

            3.141592653589793

    """

    v1_u = unit_vector(v1)

    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



fig, ax = plt.subplots(1,7)    

for xi in range(0,7):

    xi1=np.reshape(im1[xi,:],(75,75))

    xi2=np.reshape(im2[xi,:],(75,75))

    ax[xi].imshow(distanc(im1[xi,:],im2[xi,:])  )

    

def anglematrix(X,Y):

    Z=X

    for yi in range(0,len(X)):

        Z[yi]=angle_between((X[yi],Y[yi],0),(1,0,0))

    return Z



ima=im1

for xi in range(0,len(im1)):

    ima[xi]=anglematrix(im1[xi,:],im2[xi,:])  
Ua1,sa1,Va1 = np.linalg.svd(ima,full_matrices = 0)

U1,s1,V1  = np.linalg.svd(im1,full_matrices = 0)

U2,s2,V2  = np.linalg.svd(im2,full_matrices = 0)
plt.figure()

fraca1 = np.cumsum(sa1)/np.sum(sa1)

frac1 = np.cumsum(s1)/np.sum(s1)

frac2 = np.cumsum(s2)/np.sum(s2)

plt.plot(fraca1[:200])

plt.plot(frac1[:200])

plt.plot(frac2[:200])
nmodes = 20

Xt = np.hstack((U1[:,:nmodes],Ua1[:,:nmodes]))

X = np.hstack((Xt,U2[:,:nmodes]))

y = trainDF['is_iceberg'].values
#is there a native xgb way of doing it?

def logloss_xgb(preds, dtrain):

    labels = dtrain.get_label()

    score = log_loss(labels, preds)

    return 'logloss', score
nfolds = 3;

xgb_mdl=[None]*nfolds





xgb_params = {

        'objective': 'binary:logistic',

        'n_estimators':1000,

        'max_depth': 8,

        'subsample': 0.9,

        'colsample_bytree': 0.9 ,

     #   'max_delta_step': 1,

     #   'min_child_weight': 10,

        'eta': 0.01,

      #  'gamma': 0.5

        }





folds = list(StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=2016).split(X, y))



#d_test = xgb.DMatrix(X_test)



#preds = np.zeros((X_test.shape[0],nfolds))



for j, (train_idx, valid_idx) in enumerate(folds):

    X_train = X[train_idx]

    y_train = y[train_idx]

    

    X_valid = X[valid_idx]

    y_valid = y[valid_idx]

    

    d_train =  xgb.DMatrix(X_train,label=y_train)

    d_valid =  xgb.DMatrix(X_valid,label=y_valid)

    

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    

    xgb_mdl[j]=xgb.train(

            xgb_params, 

            d_train, 

            1600, watchlist, 

            early_stopping_rounds=70, 

            feval=logloss_xgb, 

            maximize=False, 

            verbose_eval=100)

#    preds[:,j] = xgb_mdl[j].predict(d_test)
y_pred = np.mean(preds,axis=1)

sub = pd.DataFrame()

sub['id'] = testDF['id']

sub['is_iceberg'] = y_pred

sub.to_csv(subName, index=False)
