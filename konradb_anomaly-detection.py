import pandas as pd

import numpy as np

from sklearn.model_selection import KFold

from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from joblib import dump, load

import pickle

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import log_loss, brier_score_loss, precision_score, recall_score, f1_score, roc_auc_score

from datetime import date

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA





import matplotlib.pyplot as plt



from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from sklearn.model_selection import train_test_split



from datetime import datetime



from sklearn.linear_model import LogisticRegression
data_folder = '../input/lish-moa/'

output_folder = ''



# fix the random seed 

xseed = 43



# number of folds for cv

nfolds = 5



# number of components to retain from PCA decomposition

nof_comp = 200



model_name = 'ad' + str(nof_comp)
xtrain = pd.read_csv(data_folder + 'train_features.csv')

xtest = pd.read_csv(data_folder + 'test_features.csv')

ytrain = pd.read_csv(data_folder + 'train_targets_scored.csv')
# due to small cardinality of all values, it's faster to handle categoricals that way,



# cp_time

xtrain['cp_time_24'] = (xtrain['cp_time'] == 24) + 0

xtrain['cp_time_48'] = (xtrain['cp_time'] == 48) + 0

xtest['cp_time_24'] = (xtest['cp_time'] == 24) + 0

xtest['cp_time_48'] = (xtest['cp_time'] == 48) + 0

xtrain.drop('cp_time', axis = 1, inplace = True)

xtest.drop('cp_time', axis = 1, inplace = True)



# cp_dose

print(set(xtrain['cp_dose']), set(xtest['cp_dose']) )

xtrain['cp_dose_D1'] = (xtrain['cp_dose'] == 'D1') + 0

xtest['cp_dose_D1'] = (xtest['cp_dose'] == 'D1') + 0

xtrain.drop('cp_dose', axis = 1, inplace = True)

xtest.drop('cp_dose', axis = 1, inplace = True)



# cp_type

xtrain['cp_type_control'] = (xtrain['cp_type'] == 'ctl_vehicle') + 0

xtest['cp_type_control'] = (xtest['cp_type'] == 'ctl_vehicle') + 0

xtrain.drop('cp_type', axis = 1, inplace = True)

xtest.drop('cp_type', axis = 1, inplace = True)
kf = KFold(n_splits = nfolds)



# separation

id_train = xtrain['sig_id']; id_test = xtest['sig_id']

ytrain.drop('sig_id', axis = 1, inplace = True) 

xtrain.drop('sig_id', axis = 1, inplace = True)

xtest.drop('sig_id', axis = 1, inplace = True)



# storage matrices for OOF / test predictions

prval = np.zeros(ytrain.shape)

prfull = np.zeros((xtest.shape[0], ytrain.shape[1]))
for (ff, (id0, id1)) in enumerate(kf.split(xtrain)):

     

    x0, x1 = xtrain.loc[id0], xtrain.loc[id1]

    y0, y1 = np.array(ytrain.loc[id0]), np.array(ytrain.loc[id1])

    

    for ii in range(0, ytrain.shape[1]):

        idx = np.where(y0[:,ii] == 0)[0]

        xtr = np.array(x0)[idx,:]

        pca = PCA()

        pca.fit(xtr)

        W1 = pca.components_[:, :nof_comp]



        # prval

        x1_proj = np.dot(x1, np.dot(W1, W1.T))

        prval[id1,ii] = (x1 - x1_proj).std(axis = 1)



        # prfull

        x1_proj = np.dot(xtest, np.dot(W1, W1.T))

        prfull[:,ii] += (xtest - x1_proj).std(axis = 1)/nfolds

        

    print(datetime.now())

    
column_list = ytrain.columns



prval_cal = np.zeros(ytrain.shape)

prfull_cal = np.zeros((xtest.shape[0], ytrain.shape[1]))







for (ff, (id0, id1)) in enumerate(kf.split(xtrain)):

     

    for ii in range(0, ytrain.shape[1]):

        

        xname = column_list[ii]

        

        x0, x1 = prval[id0,ii], prval[id1,ii]

        y0, y1 = np.array(ytrain)[id0,ii], np.array(ytrain)[id1,ii]

       

        if sum(y0) == 0:

            y0[0] = 1

            

        basemodel = LogisticRegression(C = 10)        

        basemodel.fit(x0.reshape(-1,1), y0)

        prv = basemodel.predict_proba(x1.reshape(-1,1))[:,1]

        prf = basemodel.predict_proba(np.array(prfull)[:,ii].reshape(-1,1))[:,1]

        

        prval_cal[id1, ii] = prv

        prfull_cal[:, ii] += prf/nfolds



    print(ff)
# compare performance pre- and post- calibration

metrics1 = []

metrics2 = []





for ii in range(0,ytrain.shape[1]):

    loss1 = log_loss(np.array(ytrain)[:, ii], prval[:, ii])

    metrics1.append(loss1)

    loss2 = log_loss(np.array(ytrain)[:, ii], prval_cal[:, ii])

    metrics2.append(loss2)

    

print('raw: ' + str(np.mean(metrics1)) )

print('cal: ' + str(np.mean(metrics2)))
prval_cal = pd.DataFrame(prval_cal)

prfull_cal = pd.DataFrame(prfull_cal)

prval_cal.columns = ytrain.columns

prfull_cal.columns = ytrain.columns



prval_cal['sig_id'] = id_train

prfull_cal['sig_id'] = id_test
metrics = []

for _target in ytrain.columns:

    metrics.append(log_loss(ytrain.loc[:, _target], prval_cal.loc[:, _target]))

print(f'OOF Metric: {np.round(np.mean(metrics),4)}')
xcols = list(ytrain.columns); xcols.insert(0, 'sig_id')

prval_cal = prval_cal[xcols]; prfull_cal = prfull_cal[xcols]





todate = date.today().strftime("%d%m")

print(todate)



# files for combination

prval_cal.to_csv(output_folder + 'prval_'+model_name+'_'+todate+'.csv', index = False)

prfull_cal.to_csv(output_folder + 'prfull_'+model_name+'_'+todate+'.csv', index = False)

# actual submission

prfull_cal.to_csv(output_folder + 'submission.csv', index = False)