import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['target'] = np.log1p(train['target'])
trainids = train.ID.ravel()
traintargets = train.target.ravel()
testids = test.ID.ravel()
train.drop(['ID','target'],inplace=True,axis=1)
test.drop(['ID'],inplace=True,axis=1)
# check and remove constant columns
colsToRemove = []
for col in train.columns:
    if train[col].std() == 0: 
        colsToRemove.append(col)
        
# remove constant columns in the training set
train.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
test.drop(colsToRemove, axis=1, inplace=True) 

print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))
colsToRemove = []
colsScaned = []
dupList = {}

columns = train.columns

for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
                
# remove duplicate columns in the training set
train.drop(colsToRemove, axis=1, inplace=True) 

# remove duplicate columns in the testing set
test.drop(colsToRemove, axis=1, inplace=True)
floatcolumns = []
intcolumns = []

for c in train.columns[2:]:
    s = train[c].dtype
    if(s=='float64'):
        floatcolumns.append(c)
    else:
        intcolumns.append(c)
train['sumofzeros'] = (train[intcolumns]==0).sum(axis=1)
test['sumofzeros'] = (test[intcolumns]==0).sum(axis=1)
n_comp = 11
alldata = pd.concat([train,test])

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd.fit(alldata)
tsvd_results_train = tsvd.transform(train)
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca.fit(alldata)
pca2_results_train = pca.transform(train)
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica.fit(alldata)
ica2_results_train = ica.transform(train)
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp.fit(alldata)
grp_results_train = grp.transform(train)
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp.fit(alldata)
srp_results_train = srp.transform(train)
srp_results_test = srp.transform(test)

# Append decomposition components to datasets
traindata = pd.DataFrame()
testdata = pd.DataFrame()
for i in range(0, n_comp):
    traindata['svdfeature' + str(i)] = tsvd_results_train[:,i-1]
    testdata['svdfeature' + str(i)] = tsvd_results_test[:, i-1]
for i in range(1, n_comp+1):    
    traindata['pcafeature' + str(i)] = pca2_results_train[:,i-1]
    testdata['pcafeature' + str(i)] = pca2_results_test[:, i-1]
for i in range(1, n_comp+1):    
    traindata['icafeature' + str(i)] = ica2_results_train[:,i-1]
    testdata['icafeature' + str(i)] = ica2_results_test[:, i-1]
for i in range(1, n_comp+1):    
    traindata['grpfeature' + str(i)] = grp_results_train[:,i-1]
    testdata['grpfeature' + str(i)] = grp_results_test[:, i-1]
for i in range(1, n_comp+1):    
    traindata['srpfeature' + str(i)] = srp_results_train[:,i-1]
    testdata['srpfeature' + str(i)] = srp_results_test[:, i-1]
    

traindata['target'] = traintargets
del alldata

gc.collect()
traindata = traindata.astype(float).round(6)
testdata = testdata.astype(float).round(6)
for c in testdata.columns:
    traindata[c] = np.sign(traindata[c])*np.log1p(np.abs(traindata[c]))
    testdata[c] = np.sign(testdata[c])*np.log1p(np.abs(testdata[c]))
ss = StandardScaler()
alldata = pd.concat([traindata[testdata.columns],testdata[testdata.columns]])
alldata[alldata.columns] = ss.fit_transform(alldata)
def Output(p):
    return 1./(1.+np.exp(-p))


def GP1(data):
    return Output(  0.100000*np.tanh((-1.0*((((((((data["pcafeature3"]) + (data["svdfeature3"]))) * ((11.14300060272216797)))) + (data["svdfeature1"])))))) +
                    0.100000*np.tanh(((((((data["pcafeature3"]) * 2.0)) - ((10.0)))) * (((((-3.0) * 2.0)) + ((((10.0)) * (data["pcafeature3"]))))))) +
                    0.100000*np.tanh(((np.where(data["pcafeature7"]>0, (9.71962261199951172), data["svdfeature3"] )) - (((data["svdfeature3"]) * ((8.0)))))) +
                    0.100000*np.tanh(((((np.where(data["svdfeature10"]>0, data["pcafeature3"], data["svdfeature3"] )) * (np.minimum(((-3.0)), ((data["svdfeature10"])))))) * 2.0)) +
                    0.100000*np.tanh(((((((((data["pcafeature7"]) * 2.0)) - (((((data["pcafeature3"]) * 2.0)) * 2.0)))) - (data["pcafeature3"]))) - (((data["svdfeature10"]) * 2.0)))) +
                    0.100000*np.tanh((((7.09501695632934570)) * ((((((data["svdfeature3"]) < (((((((data["svdfeature3"]) < ((13.60823345184326172)))*1.)) < ((8.58406448364257812)))*1.)))*1.)) - (data["svdfeature3"]))))) +
                    0.100000*np.tanh(((data["svdfeature9"]) * (np.minimum(((((np.minimum(((data["svdfeature9"])), (((-1.0*((data["svdfeature9"]))))))) + (-3.0)))), ((-3.0)))))) +
                    0.100000*np.tanh((((((-1.0*((data["svdfeature9"])))) - (data["svdfeature1"]))) - (((np.where(data["pcafeature3"]>0, data["svdfeature1"], data["svdfeature10"] )) + (data["pcafeature3"]))))) +
                    0.100000*np.tanh(((((data["pcafeature7"]) - (((data["svdfeature9"]) - (np.where(data["pcafeature7"]>0, (-1.0*((data["svdfeature10"]))), data["pcafeature7"] )))))) * 2.0)) +
                    0.100000*np.tanh(((((((((np.where(data["svdfeature1"]>0, data["pcafeature7"], (-1.0*((data["svdfeature1"]))) )) * 2.0)) * 2.0)) - (data["pcafeature3"]))) - (data["svdfeature10"]))) +
                    0.100000*np.tanh(((data["pcafeature7"]) + (((data["pcafeature5"]) + ((((((data["pcafeature8"]) - (data["svdfeature9"]))) + (((data["pcafeature8"]) - (data["svdfeature3"]))))/2.0)))))) +
                    0.100000*np.tanh(((data["pcafeature3"]) + ((((7.98993062973022461)) + (((np.where(data["pcafeature11"]>0, data["pcafeature4"], data["pcafeature11"] )) * ((13.70520782470703125)))))))) +
                    0.100000*np.tanh(((np.maximum(((np.maximum(((data["svdfeature10"])), ((data["svdfeature3"]))))), ((data["svdfeature10"])))) * (np.minimum(((data["pcafeature5"])), ((((data["svdfeature3"]) + (data["pcafeature3"])))))))) +
                    0.100000*np.tanh(np.where(data["svdfeature10"]>0, (((((4.0)) * (np.minimum(((data["pcafeature11"])), ((data["pcafeature3"])))))) - (-2.0)), ((data["pcafeature4"]) * 2.0) )) +
                    0.100000*np.tanh(np.minimum(((((data["pcafeature7"]) * (data["pcafeature7"])))), ((((np.minimum(((data["pcafeature7"])), ((((data["pcafeature7"]) * (data["svdfeature1"])))))) * (data["svdfeature1"])))))) +
                    0.100000*np.tanh(((data["svdfeature3"]) * ((((np.minimum(((((data["svdfeature1"]) * (data["pcafeature5"])))), ((data["pcafeature8"])))) + (((data["pcafeature5"]) + (data["pcafeature8"]))))/2.0)))) +
                    0.100000*np.tanh(np.minimum((((((data["pcafeature5"]) > (data["pcafeature8"]))*1.))), ((((data["pcafeature8"]) + ((((((data["pcafeature5"]) > (data["svdfeature10"]))*1.)) + (data["pcafeature8"])))))))) +
                    0.100000*np.tanh(np.where(((data["pcafeature5"]) - (data["pcafeature3"]))>0, ((data["pcafeature7"]) * ((-1.0*((data["svdfeature10"]))))), ((data["pcafeature7"]) * (data["svdfeature3"])) )) +
                    0.100000*np.tanh(((((((((((data["svdfeature7"]) * 2.0)) + (data["svdfeature3"]))) * 2.0)) + (data["icafeature8"]))) - (data["pcafeature7"]))) +
                    0.100000*np.tanh(((data["pcafeature4"]) * (np.maximum(((((data["svdfeature7"]) * (((data["svdfeature7"]) + (data["pcafeature4"])))))), ((data["pcafeature8"])))))) +
                    0.100000*np.tanh(np.where(data["pcafeature8"]>0, data["pcafeature4"], ((data["svdfeature7"]) * (np.where(data["svdfeature10"]>0, data["svdfeature3"], (((data["pcafeature4"]) + (data["svdfeature7"]))/2.0) ))) )) +
                    0.100000*np.tanh(((((((-1.0*((data["pcafeature7"])))) + (((((((-1.0*(((((-1.0*((data["pcafeature7"])))) / 2.0))))) / 2.0)) < (data["svdfeature9"]))*1.)))/2.0)) / 2.0)) +
                    0.099941*np.tanh((((((((data["svdfeature10"]) / 2.0)) < (((((((data["pcafeature4"]) > (data["pcafeature7"]))*1.)) > (data["pcafeature7"]))*1.)))*1.)) * (data["pcafeature4"]))) +
                    0.100000*np.tanh(np.where(((data["pcafeature8"]) - ((-1.0*((0.636620)))))>0, np.where(data["pcafeature4"]>0, data["svdfeature3"], ((0.636620) / 2.0) ), -3.0 )) +
                    0.100000*np.tanh(np.where(np.minimum(((data["pcafeature8"])), ((data["svdfeature10"])))>0, ((0.636620) - (data["pcafeature5"])), (((0.636620) < (np.tanh((data["pcafeature5"]))))*1.) )) +
                    0.100000*np.tanh(((data["svdfeature10"]) * (((data["svdfeature10"]) * (((data["svdfeature10"]) * ((-1.0*(((((data["svdfeature10"]) > (((1.570796) / 2.0)))*1.))))))))))) +
                    0.100000*np.tanh((((np.where(data["pcafeature5"]>0, data["pcafeature5"], 3.0 )) < ((((((((3.0) / 2.0)) / 2.0)) < (data["pcafeature5"]))*1.)))*1.)) +
                    0.100000*np.tanh((((((-1.0*((((((0.93733572959899902)) < (data["svdfeature10"]))*1.))))) * 2.0)) * 2.0)) +
                    0.089803*np.tanh(((np.where(data["pcafeature3"]>0, data["pcafeature3"], data["svdfeature10"] )) * (np.where(data["pcafeature5"]>0, (((0.0) > (data["pcafeature3"]))*1.), data["pcafeature8"] )))) +
                    0.100000*np.tanh((((2.0) < (np.where(data["svdfeature10"]>0, np.where(data["pcafeature7"]>0, 2.0, data["pcafeature4"] ), data["pcafeature4"] )))*1.)) +
                    0.100000*np.tanh(np.where(data["svdfeature7"]>0, 0.0, np.where(data["svdfeature7"]>0, 0.0, np.minimum(((((data["pcafeature4"]) * (data["pcafeature8"])))), ((data["pcafeature4"]))) ) )) +
                    0.100000*np.tanh((((np.maximum(((data["pcafeature7"])), ((data["pcafeature5"])))) < ((((data["pcafeature5"]) > (np.tanh(((-1.0*((-1.0)))))))*1.)))*1.)) +
                    0.099570*np.tanh(np.where((((data["pcafeature4"]) + (0.636620))/2.0)>0, (((((0.0) - (1.570796))) > (data["svdfeature10"]))*1.), ((-3.0) * 2.0) )) +
                    0.100000*np.tanh((((((-3.0) > (data["svdfeature1"]))*1.)) - (((((((data["svdfeature7"]) + (3.141593))/2.0)) < (data["svdfeature1"]))*1.)))) +
                    0.100000*np.tanh((((np.tanh((np.tanh(((1.96118521690368652)))))) < (np.minimum(((data["pcafeature5"])), ((((data["svdfeature3"]) + (data["svdfeature3"])))))))*1.)) +
                    0.100000*np.tanh((-1.0*((((((-1.0*((np.tanh((-1.0)))))) < (data["pcafeature3"]))*1.))))) +
                    0.099980*np.tanh(((((((np.where(data["svdfeature3"]>0, data["svdfeature3"], 1.570796 )) + (data["pcafeature5"]))/2.0)) < ((((data["svdfeature3"]) < (data["pcafeature5"]))*1.)))*1.)) +
                    0.099980*np.tanh(((np.minimum(((((data["svdfeature1"]) * (((data["svdfeature1"]) * (((data["svdfeature1"]) * (data["pcafeature8"])))))))), ((((data["svdfeature1"]) / 2.0))))) / 2.0)) +
                    0.100000*np.tanh(((((((((((data["svdfeature3"]) > (data["svdfeature10"]))*1.)) > ((((data["pcafeature7"]) > (((data["pcafeature7"]) / 2.0)))*1.)))*1.)) / 2.0)) / 2.0)) +
                    0.100000*np.tanh((((((-1.0*((((((((data["svdfeature3"]) > (data["pcafeature7"]))*1.)) < ((((data["svdfeature3"]) + (((data["svdfeature3"]) / 2.0)))/2.0)))*1.))))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((((((data["pcafeature7"]) < (-1.0))*1.)) * ((9.0)))) +
                    0.090545*np.tanh((((((((-1.0) > (np.minimum(((data["pcafeature8"])), ((data["pcafeature7"])))))*1.)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((-1.0*((((((((((((-2.0) / 2.0)) / 2.0)) < (data["pcafeature8"]))*1.)) < (np.minimum(((data["pcafeature5"])), ((data["pcafeature4"])))))*1.))))) +
                    0.100000*np.tanh((((((((np.tanh((np.tanh(((((5.0)) - (3.141593))))))) < (data["pcafeature5"]))*1.)) * (2.0))) * (data["svdfeature3"]))) +
                    0.100000*np.tanh(np.where(data["svdfeature10"]>0, (((data["pcafeature3"]) < ((-1.0*((((data["svdfeature10"]) * (data["pcafeature4"])))))))*1.), ((data["svdfeature10"]) * (0.0)) )) +
                    0.085603*np.tanh((-1.0*(((((1.0) < ((((np.maximum(((-1.0)), ((data["svdfeature10"])))) + ((((1.570796) + (data["svdfeature10"]))/2.0)))/2.0)))*1.))))) +
                    0.090408*np.tanh(np.tanh(((((np.minimum(((data["pcafeature7"])), ((data["svdfeature10"])))) > ((((((data["pcafeature3"]) > (((data["pcafeature8"]) * 2.0)))*1.)) - (data["pcafeature4"]))))*1.)))) +
                    0.100000*np.tanh(((((np.minimum((((((((data["pcafeature8"]) < (data["svdfeature9"]))*1.)) - (((data["pcafeature4"]) / 2.0))))), ((data["svdfeature9"])))) / 2.0)) / 2.0)) +
                    0.099980*np.tanh((((((np.where(data["pcafeature5"]>0, -1.0, ((data["svdfeature9"]) / 2.0) )) - (data["pcafeature8"]))) > ((((data["svdfeature10"]) > (0.636620))*1.)))*1.)) +
                    0.099980*np.tanh((-1.0*((((((1.69305002689361572)) < (np.maximum(((data["pcafeature7"])), ((((data["pcafeature3"]) * 2.0))))))*1.))))) +
                    0.100000*np.tanh(((((((data["pcafeature5"]) + (np.tanh((np.tanh((-2.0))))))/2.0)) > (((((((data["svdfeature1"]) < (data["pcafeature5"]))*1.)) < (data["svdfeature1"]))*1.)))*1.)) +
                    0.100000*np.tanh(((((((data["pcafeature3"]) > (((np.maximum(((data["pcafeature5"])), ((0.318310)))) * 2.0)))*1.)) + (((np.tanh((np.tanh((data["pcafeature7"]))))) / 2.0)))/2.0)) +
                    0.100000*np.tanh((((3.141593) < (3.141593))*1.)) +
                    0.100000*np.tanh((((np.minimum((((((np.minimum(((1.570796)), ((1.570796)))) > (1.570796))*1.))), ((data["pcafeature4"])))) > (((data["svdfeature10"]) + (1.570796))))*1.)) +
                    0.100000*np.tanh((((((((((((data["pcafeature3"]) * (data["pcafeature3"]))) > (3.0))*1.)) * (3.0))) * (data["pcafeature3"]))) * (data["pcafeature3"]))) +
                    0.099961*np.tanh(((((((data["svdfeature9"]) < (np.minimum(((-1.0)), ((data["pcafeature4"])))))*1.)) > ((((data["svdfeature3"]) < (data["pcafeature7"]))*1.)))*1.)) +
                    0.100000*np.tanh((((-1.0*((data["pcafeature4"])))) * (((((-1.0*((data["pcafeature8"])))) > (((((2.0) - (data["pcafeature5"]))) / 2.0)))*1.)))) +
                    0.100000*np.tanh((((data["pcafeature7"]) < (-1.0))*1.)) +
                    0.100000*np.tanh((((np.minimum(((data["pcafeature7"])), ((data["pcafeature4"])))) < (-1.0))*1.)) +
                    0.100000*np.tanh(np.minimum(((((1.0) - (data["svdfeature3"])))), (((((-1.0*(((((3.0) < (((data["svdfeature3"]) * 2.0)))*1.))))) * (3.0)))))) +
                    0.099980*np.tanh((((-1.0) > (np.minimum(((data["pcafeature7"])), (((((-3.0) > ((((-3.0) > (-3.0))*1.)))*1.))))))*1.)) +
                    0.099941*np.tanh((((((((np.maximum((((-1.0*((data["svdfeature3"]))))), ((data["pcafeature5"])))) < (np.tanh(((((data["svdfeature3"]) < (data["pcafeature5"]))*1.)))))*1.)) / 2.0)) / 2.0)) +
                    0.100000*np.tanh((((data["pcafeature7"]) < (np.where(-2.0>0, (((np.tanh((-2.0))) < (-2.0))*1.), -1.0 )))*1.)) +
                    0.099941*np.tanh((((((7.0)) - (np.maximum((((7.08447408676147461))), (((((7.08447408676147461)) / 2.0))))))) / 2.0)) +
                    0.100000*np.tanh((((((data["pcafeature5"]) > ((((data["pcafeature3"]) > (np.minimum((((((data["pcafeature8"]) > (data["pcafeature7"]))*1.))), ((data["pcafeature7"])))))*1.)))*1.)) / 2.0)) +
                    0.100000*np.tanh(np.minimum(((((1.570796) * (((((0.0) * (0.0))) * (((0.0) * (0.0)))))))), ((0.0)))) +
                    0.100000*np.tanh((-1.0*((((np.maximum(((0.0)), ((((data["pcafeature5"]) * (data["svdfeature1"])))))) * (((data["svdfeature1"]) * (data["svdfeature1"])))))))) +
                    0.099980*np.tanh(((((((((((-1.0) > (0.636620))*1.)) - (0.636620))) > (data["pcafeature4"]))*1.)) / 2.0)) +
                    0.098965*np.tanh((((data["pcafeature4"]) < (-1.0))*1.)) +
                    0.099980*np.tanh((-1.0*(((((((2.0) < (data["pcafeature7"]))*1.)) * ((((6.77608394622802734)) * 2.0))))))) +
                    0.100000*np.tanh((((0.0) > (((0.0) * ((((2.0) > (0.0))*1.)))))*1.)) +
                    0.087849*np.tanh(((((np.where(data["svdfeature1"]>0, np.where(data["pcafeature5"]>0, (((data["svdfeature1"]) < (data["pcafeature5"]))*1.), data["svdfeature10"] ), data["pcafeature4"] )) / 2.0)) / 2.0)) +
                    0.074937*np.tanh((-1.0*((np.tanh(((((3.0) < (((((data["svdfeature10"]) * 2.0)) * 2.0)))*1.))))))) +
                    0.091561*np.tanh(np.tanh(((((((0.636620) < (np.where(data["svdfeature10"]>0, data["pcafeature3"], ((data["pcafeature8"]) + (((data["pcafeature3"]) * 2.0))) )))*1.)) / 2.0)))) +
                    0.081403*np.tanh(((((data["pcafeature5"]) * ((((((((-1.0) * ((((data["pcafeature4"]) + (data["pcafeature5"]))/2.0)))) + (-1.0))/2.0)) / 2.0)))) / 2.0)) +
                    0.099980*np.tanh((-1.0*(((((((((((data["pcafeature3"]) > (np.tanh(((1.47271907329559326)))))*1.)) * ((9.69428157806396484)))) * ((8.62608623504638672)))) * (data["pcafeature3"])))))) +
                    0.100000*np.tanh((((data["pcafeature5"]) > (np.where(data["pcafeature4"]>0, np.where(data["pcafeature7"]>0, (((data["pcafeature5"]) > (data["pcafeature5"]))*1.), data["pcafeature4"] ), data["pcafeature5"] )))*1.)) +
                    0.100000*np.tanh(((((((((((data["svdfeature10"]) + (3.0))) + (data["svdfeature9"]))/2.0)) / 2.0)) < (0.0))*1.)) +
                    0.100000*np.tanh((((data["pcafeature4"]) < (-1.0))*1.)) +
                    0.100000*np.tanh((((-1.0) > (data["pcafeature4"]))*1.)) +
                    0.100000*np.tanh((((np.where(data["pcafeature8"]>0, data["pcafeature7"], data["pcafeature4"] )) < (np.tanh((((-3.0) + (data["pcafeature4"]))))))*1.)) +
                    0.100000*np.tanh(np.minimum(((((((0.0) * (0.0))) * (0.0)))), ((0.0)))) +
                    0.099980*np.tanh((((-3.0) > (0.0))*1.)) +
                    0.070756*np.tanh((((data["pcafeature7"]) < (np.minimum((((((-1.0) > ((((data["pcafeature7"]) < (data["pcafeature7"]))*1.)))*1.))), ((-1.0)))))*1.)) +
                    0.098652*np.tanh((((0.0) < ((0.0)))*1.)) +
                    0.098203*np.tanh(np.minimum((((((((data["svdfeature9"]) > (((((((0.0) > (data["pcafeature4"]))*1.)) > (data["pcafeature3"]))*1.)))*1.)) * (data["svdfeature10"])))), ((0.0)))) +
                    0.100000*np.tanh((((((0.0) * (0.636620))) < (0.0))*1.)) +
                    0.100000*np.tanh((((((((-3.0) / 2.0)) > (np.minimum(((data["pcafeature5"])), (((((-3.0) > (data["pcafeature5"]))*1.))))))*1.)) / 2.0)) +
                    0.099980*np.tanh(((np.minimum(((data["svdfeature9"])), ((data["icafeature8"])))) * (((((((data["svdfeature9"]) + (1.570796))/2.0)) < ((-1.0*((data["icafeature8"])))))*1.)))) +
                    0.099980*np.tanh((((((((((((0.318310) * 2.0)) / 2.0)) / 2.0)) / 2.0)) + ((((((data["pcafeature5"]) * 2.0)) > (1.570796))*1.)))/2.0)) +
                    0.099727*np.tanh((-1.0*((((((0.86650031805038452)) < (((((((0.636620) > ((((data["pcafeature3"]) < ((-1.0*((data["pcafeature8"])))))*1.)))*1.)) < (data["pcafeature3"]))*1.)))*1.))))) +
                    0.100000*np.tanh((((data["svdfeature9"]) > (((((((np.where(2.0>0, data["pcafeature8"], data["svdfeature9"] )) + (data["svdfeature9"]))/2.0)) + (2.0))/2.0)))*1.)) +
                    0.100000*np.tanh((((((data["pcafeature3"]) < (np.minimum(((((data["svdfeature1"]) - (3.141593)))), ((((3.141593) - (3.141593)))))))*1.)) * (data["svdfeature1"]))) +
                    0.099883*np.tanh((((0.0) < ((((-1.0) > ((((0.0) < ((((-1.0) > (data["pcafeature7"]))*1.)))*1.)))*1.)))*1.)) +
                    0.100000*np.tanh((((0.0) > ((((0.0) < (0.0))*1.)))*1.)) +
                    0.099961*np.tanh(((np.minimum(((data["pcafeature3"])), ((data["pcafeature4"])))) * ((((2.0) < (data["pcafeature8"]))*1.)))) +
                    0.100000*np.tanh((((3.141593) < (((data["pcafeature4"]) + (((((((data["pcafeature4"]) > (data["pcafeature7"]))*1.)) > (data["pcafeature7"]))*1.)))))*1.)) +
                    0.097187*np.tanh((-1.0*((((((((((0.0) * (3.141593))) < (1.570796))*1.)) < (np.where(data["pcafeature5"]>0, data["pcafeature4"], data["svdfeature10"] )))*1.))))) +
                    0.100000*np.tanh(((np.where(data["svdfeature1"]>0, 0.318310, ((((data["pcafeature5"]) / 2.0)) + (np.where(data["pcafeature5"]>0, data["pcafeature4"], data["pcafeature5"] ))) )) / 2.0)) +
                    0.100000*np.tanh(((0.0) * (0.0))) +
                    0.099980*np.tanh(((((((0.0)) - (data["svdfeature3"]))) > ((1.16795563697814941)))*1.)) +
                    0.099980*np.tanh((((np.tanh((3.141593))) < (np.where(data["pcafeature3"]>0, np.where(data["pcafeature7"]>0, data["svdfeature10"], data["pcafeature7"] ), np.tanh((data["svdfeature10"])) )))*1.)) +
                    0.100000*np.tanh((((data["pcafeature7"]) < (-1.0))*1.)) +
                    0.099961*np.tanh(np.where(data["svdfeature10"]>0, ((0.636620) - (data["svdfeature10"])), (((0.636620) < (((-1.0) - (data["svdfeature10"]))))*1.) )) +
                    0.100000*np.tanh(((((((np.where((((data["svdfeature10"]) + (0.636620))/2.0)>0, 0.636620, data["pcafeature3"] )) / 2.0)) / 2.0)) / 2.0)) +
                    0.099980*np.tanh((((((1.570796) < ((0.0)))*1.)) * (0.0))) +
                    0.100000*np.tanh((((0.0) > (0.0))*1.)) +
                    0.099980*np.tanh((((-1.0*((data["svdfeature3"])))) * ((((data["svdfeature1"]) > ((((3.0) + ((((data["pcafeature8"]) + ((-1.0*((data["pcafeature4"])))))/2.0)))/2.0)))*1.)))) +
                    0.100000*np.tanh((((-1.0*(((((np.tanh((((((1.570796) / 2.0)) * (1.570796))))) < (data["pcafeature3"]))*1.))))) * ((11.42416763305664062)))) +
                    0.099980*np.tanh(((np.minimum(((((((((0.0) > (data["pcafeature8"]))*1.)) > (data["svdfeature1"]))*1.))), ((((np.tanh((data["svdfeature1"]))) / 2.0))))) * (data["svdfeature10"]))) +
                    0.077496*np.tanh((((((((((((data["pcafeature3"]) / 2.0)) / 2.0)) > ((0.14671686291694641)))*1.)) / 2.0)) * ((((data["svdfeature1"]) > (((data["pcafeature8"]) / 2.0)))*1.)))) +
                    0.099980*np.tanh((((np.where(-1.0>0, (((-1.0) > (-1.0))*1.), -1.0 )) > (np.where(data["pcafeature7"]>0, data["pcafeature4"], data["pcafeature7"] )))*1.)) +
                    0.100000*np.tanh((((0.0) < (0.0))*1.)) +
                    0.099980*np.tanh(((((((((data["pcafeature3"]) + (np.tanh((1.570796))))/2.0)) > (np.tanh((1.570796))))*1.)) * (-3.0))) +
                    0.097441*np.tanh(((((((data["pcafeature7"]) < (-1.0))*1.)) > ((((0.0) < (0.0))*1.)))*1.)) +
                    0.100000*np.tanh((((((data["pcafeature11"]) * (((data["pcafeature3"]) * (data["svdfeature3"]))))) + ((((((data["pcafeature1"]) < (data["pcafeature11"]))*1.)) * (data["svdfeature3"]))))/2.0)) +
                    0.099961*np.tanh(((0.0) * (0.0))) +
                    0.096777*np.tanh(np.minimum(((((((1.0) - (data["svdfeature3"]))) * (np.maximum(((0.0)), ((data["svdfeature3"]))))))), ((((1.0) - (data["svdfeature3"])))))) +
                    0.083708*np.tanh((((0.0) > ((((((((0.0)) < (0.0))*1.)) < ((13.16501045227050781)))*1.)))*1.)) +
                    0.099980*np.tanh(((data["pcafeature7"]) * ((((-1.0*((np.minimum((((((data["pcafeature5"]) < (((data["pcafeature5"]) * (data["svdfeature3"]))))*1.))), ((0.318310))))))) / 2.0)))) +
                    0.099941*np.tanh((((((np.tanh((1.570796))) < (data["svdfeature10"]))*1.)) * ((((((data["pcafeature3"]) < (np.tanh((data["svdfeature10"]))))*1.)) * (data["pcafeature3"]))))) +
                    0.090760*np.tanh(((((((((1.570796) * (data["pcafeature8"]))) < (-1.0))*1.)) > (((data["pcafeature3"]) + ((((3.49713277816772461)) / 2.0)))))*1.)) +
                    0.100000*np.tanh(((((((1.570796) < (data["pcafeature11"]))*1.)) > ((((data["pcafeature7"]) > (((((((1.570796) < (data["pcafeature11"]))*1.)) > (1.570796))*1.)))*1.)))*1.)) +
                    0.099980*np.tanh((((np.minimum((((((1.0) < (0.0))*1.))), (((((0.0) > ((3.0)))*1.))))) < (0.0))*1.)) +
                    0.100000*np.tanh(((np.where(data["svdfeature7"]>0, np.where(data["pcafeature11"]>0, ((data["svdfeature7"]) * (0.0)), ((data["svdfeature7"]) / 2.0) ), data["pcafeature11"] )) / 2.0)) +
                    0.099531*np.tanh(((((((4.0)) * (0.0))) < (((0.0) * (0.0))))*1.)) +
                    0.100000*np.tanh((((np.minimum(((data["svdfeature3"])), (((((((-1.0) > (-1.0))*1.)) + (np.tanh((-1.0)))))))) > (data["pcafeature7"]))*1.)) +
                    0.086228*np.tanh(((((((np.tanh((((data["pcafeature5"]) / 2.0)))) * (data["pcafeature4"]))) * (data["pcafeature8"]))) * (data["svdfeature9"]))))

def GP2(data):
    return Output(  0.100000*np.tanh(((((((((((((data["pcafeature5"]) - (data["pcafeature7"]))) - (data["svdfeature1"]))) - (data["svdfeature1"]))) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((((((np.minimum(((((data["pcafeature3"]) * 2.0))), ((data["pcafeature5"])))) - (data["svdfeature1"]))) * 2.0)) + (data["svdfeature10"]))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((data["pcafeature3"]) + ((((((((((-1.0*((data["svdfeature1"])))) - (data["pcafeature7"]))) * 2.0)) * 2.0)) * 2.0)))) +
                    0.100000*np.tanh(((((((((np.minimum(((data["svdfeature10"])), ((((data["pcafeature5"]) - (data["pcafeature7"])))))) + (data["pcafeature3"]))) * 2.0)) * 2.0)) + (data["pcafeature3"]))) +
                    0.100000*np.tanh((((((((((((0.636620) < (np.tanh((data["pcafeature5"]))))*1.)) - (data["svdfeature1"]))) * 2.0)) * 2.0)) - (0.636620))) +
                    0.100000*np.tanh((((((data["pcafeature3"]) < ((((data["svdfeature3"]) + (data["pcafeature5"]))/2.0)))*1.)) - (((((data["pcafeature7"]) - (data["pcafeature5"]))) - (data["svdfeature3"]))))) +
                    0.100000*np.tanh(((((((((((((data["svdfeature10"]) * (data["svdfeature3"]))) - (data["pcafeature7"]))) - (data["pcafeature8"]))) - (data["svdfeature9"]))) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((np.minimum(((((data["pcafeature5"]) - (data["svdfeature9"])))), ((np.minimum(((data["pcafeature5"])), ((((data["svdfeature10"]) + (data["svdfeature10"]))))))))) - (data["svdfeature1"]))) +
                    0.100000*np.tanh(((((((data["svdfeature10"]) + (data["pcafeature8"]))) * ((-1.0*((data["pcafeature7"])))))) - (((data["svdfeature1"]) / 2.0)))) +
                    0.100000*np.tanh(((np.maximum(((data["pcafeature7"])), ((data["pcafeature5"])))) * (np.minimum(((data["pcafeature5"])), (((((data["svdfeature10"]) < (data["pcafeature5"]))*1.))))))) +
                    0.100000*np.tanh(((np.minimum(((data["svdfeature3"])), ((np.where(data["pcafeature5"]>0, data["pcafeature5"], data["svdfeature9"] ))))) - (data["svdfeature9"]))) +
                    0.100000*np.tanh((-1.0*((np.where(data["pcafeature4"]>0, np.maximum(((data["pcafeature11"])), ((data["pcafeature4"]))), ((data["pcafeature11"]) + (((data["pcafeature4"]) + (data["pcafeature3"])))) ))))) +
                    0.100000*np.tanh(np.where(data["pcafeature3"]>0, (((np.tanh((np.tanh((2.0))))) < (data["pcafeature5"]))*1.), (-1.0*((data["svdfeature10"]))) )) +
                    0.100000*np.tanh((((((((data["pcafeature11"]) < (data["pcafeature4"]))*1.)) + (((((data["pcafeature4"]) + (data["pcafeature3"]))) + (data["pcafeature4"]))))) * (data["pcafeature8"]))) +
                    0.100000*np.tanh(((((2.0) * ((((np.tanh((np.tanh((2.0))))) < (data["pcafeature5"]))*1.)))) + (((data["pcafeature7"]) * (data["pcafeature5"]))))) +
                    0.100000*np.tanh((-1.0*((np.where(data["svdfeature3"]>0, data["svdfeature1"], (((data["svdfeature1"]) < (((data["svdfeature1"]) * (((data["svdfeature1"]) * (data["svdfeature1"]))))))*1.) ))))) +
                    0.100000*np.tanh((((((((data["pcafeature5"]) > (np.tanh((np.tanh((((np.tanh((((data["pcafeature5"]) * 2.0)))) * 2.0)))))))*1.)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((3.141593) * ((-1.0*(((((data["svdfeature10"]) > ((((1.570796) > ((((0.0) > (-3.0))*1.)))*1.)))*1.))))))) +
                    0.100000*np.tanh(((((np.maximum(((data["pcafeature7"])), ((data["icafeature8"])))) * (((data["svdfeature7"]) - (np.maximum(((data["icafeature8"])), ((data["icafeature8"])))))))) * 2.0)) +
                    0.100000*np.tanh((((((((data["pcafeature5"]) > (np.tanh((np.tanh(((((((-2.0) < (data["svdfeature1"]))*1.)) * 2.0)))))))*1.)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((-1.0*((np.maximum(((((data["svdfeature10"]) * (data["pcafeature4"])))), (((((((data["svdfeature7"]) * (data["svdfeature10"]))) + (data["svdfeature3"]))/2.0)))))))) +
                    0.100000*np.tanh(np.where(data["pcafeature3"]>0, ((((((np.tanh((data["pcafeature5"]))) > (0.636620))*1.)) > (data["pcafeature5"]))*1.), ((data["svdfeature9"]) * (data["pcafeature5"])) )) +
                    0.099941*np.tanh((-1.0*((np.where(data["svdfeature9"]>0, data["pcafeature4"], np.where(data["pcafeature4"]>0, data["pcafeature7"], (((((data["pcafeature4"]) * 2.0)) < (data["svdfeature9"]))*1.) ) ))))) +
                    0.100000*np.tanh(np.minimum(((((data["pcafeature8"]) * (data["pcafeature4"])))), ((((((((data["pcafeature4"]) < (((data["pcafeature8"]) * (data["svdfeature3"]))))*1.)) + (data["pcafeature8"]))/2.0))))) +
                    0.100000*np.tanh((((((-3.0) > (((data["svdfeature1"]) + ((((data["svdfeature1"]) + ((((data["svdfeature1"]) > (-3.0))*1.)))/2.0)))))*1.)) * 2.0)) +
                    0.100000*np.tanh((-1.0*(((((np.minimum(((data["svdfeature10"])), ((data["pcafeature8"])))) > ((-1.0*(((((data["svdfeature10"]) > (np.tanh((1.570796))))*1.))))))*1.))))) +
                    0.100000*np.tanh(((((((data["pcafeature5"]) < ((((0.636620) < (np.tanh((data["pcafeature5"]))))*1.)))*1.)) > (((data["pcafeature5"]) * (np.tanh((data["pcafeature5"]))))))*1.)) +
                    0.100000*np.tanh((-1.0*((((((((data["svdfeature9"]) < ((((data["svdfeature10"]) < (np.tanh(((((-3.0) < ((9.0)))*1.)))))*1.)))*1.)) < (data["pcafeature5"]))*1.))))) +
                    0.089803*np.tanh(np.where((((((1.570796) + (data["pcafeature8"]))) > (1.0))*1.)>0, (((data["pcafeature8"]) < (1.0))*1.), data["pcafeature8"] )) +
                    0.100000*np.tanh(np.maximum(((np.where(data["svdfeature1"]>0, 0.0, np.where(data["pcafeature7"]>0, -2.0, data["pcafeature4"] ) ))), ((((-2.0) - (data["svdfeature1"])))))) +
                    0.100000*np.tanh(((((((((((data["svdfeature1"]) < (data["pcafeature5"]))*1.)) < ((((data["pcafeature5"]) + (((np.tanh((data["pcafeature5"]))) * 2.0)))/2.0)))*1.)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((((-1.0*(((((((((((data["pcafeature7"]) > (data["pcafeature8"]))*1.)) > (((data["pcafeature7"]) / 2.0)))*1.)) < (data["pcafeature5"]))*1.))))) * 2.0)) +
                    0.099570*np.tanh(((((((data["pcafeature4"]) < (2.0))*1.)) > (((((((data["pcafeature3"]) > ((((-1.0) > (data["pcafeature4"]))*1.)))*1.)) > (data["svdfeature10"]))*1.)))*1.)) +
                    0.100000*np.tanh((((-1.0*((((1.570796) * 2.0))))) * ((((((((1.570796) < (data["svdfeature1"]))*1.)) * 2.0)) * 2.0)))) +
                    0.100000*np.tanh(((((((-3.0) > (((data["svdfeature10"]) * 2.0)))*1.)) > ((((data["pcafeature5"]) > ((((-3.0) > (((data["svdfeature10"]) * 2.0)))*1.)))*1.)))*1.)) +
                    0.100000*np.tanh(((np.minimum((((((data["pcafeature3"]) + ((((data["svdfeature10"]) < (data["pcafeature5"]))*1.)))/2.0))), ((((data["svdfeature3"]) * (data["svdfeature10"])))))) / 2.0)) +
                    0.099980*np.tanh((-1.0*(((((np.tanh((1.570796))) < (data["svdfeature10"]))*1.))))) +
                    0.099980*np.tanh((((((((((data["svdfeature1"]) < (((-2.0) - (np.where(data["svdfeature1"]>0, data["pcafeature3"], data["pcafeature3"] )))))*1.)) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(np.minimum((((((data["pcafeature3"]) > ((((((data["pcafeature5"]) / 2.0)) > (data["pcafeature7"]))*1.)))*1.))), (((((data["pcafeature8"]) < (((data["pcafeature3"]) / 2.0)))*1.))))) +
                    0.100000*np.tanh((-1.0*(((((data["pcafeature5"]) > (((((((data["pcafeature5"]) > (np.tanh((((data["pcafeature7"]) / 2.0)))))*1.)) > (((data["pcafeature7"]) / 2.0)))*1.)))*1.))))) +
                    0.100000*np.tanh((((np.tanh((1.0))) < (np.where(((data["svdfeature10"]) * (data["pcafeature8"]))>0, data["pcafeature3"], data["pcafeature5"] )))*1.)) +
                    0.090545*np.tanh((((((((data["pcafeature5"]) > (data["pcafeature7"]))*1.)) * (((data["pcafeature8"]) * ((((((-1.0) / 2.0)) > (data["pcafeature7"]))*1.)))))) / 2.0)) +
                    0.100000*np.tanh(np.where((((np.tanh((-1.0))) + (data["pcafeature5"]))/2.0)>0, data["pcafeature3"], ((0.0) * ((((data["pcafeature3"]) > (-1.0))*1.))) )) +
                    0.100000*np.tanh(((np.minimum((((-1.0*(((((data["pcafeature5"]) > (1.0))*1.)))))), ((1.0)))) * 2.0)) +
                    0.100000*np.tanh((-1.0*(((((data["svdfeature10"]) > (np.where(data["svdfeature10"]>0, np.where(data["pcafeature4"]>0, ((1.570796) / 2.0), data["svdfeature10"] ), data["svdfeature10"] )))*1.))))) +
                    0.085603*np.tanh(((((((((data["pcafeature3"]) > (0.0))*1.)) < (data["pcafeature5"]))*1.)) * (((data["pcafeature4"]) * (((data["pcafeature5"]) + (data["pcafeature5"]))))))) +
                    0.090408*np.tanh(((0.318310) * (0.318310))) +
                    0.100000*np.tanh((-1.0*(((((((np.minimum(((data["svdfeature9"])), ((data["pcafeature7"])))) > (np.tanh((data["pcafeature8"]))))*1.)) * (((data["pcafeature4"]) * 2.0))))))) +
                    0.099980*np.tanh((-1.0*(((((((((((0.318310) < (-1.0))*1.)) > (data["pcafeature7"]))*1.)) < ((((1.570796) < ((-1.0*((data["pcafeature5"])))))*1.)))*1.))))) +
                    0.099980*np.tanh((((((((data["svdfeature3"]) + (data["svdfeature3"]))) * (data["svdfeature3"]))) < (np.where(data["pcafeature7"]>0, data["svdfeature3"], data["pcafeature4"] )))*1.)) +
                    0.100000*np.tanh(((((-3.0) + (-3.0))) * ((((((-3.0) + (data["svdfeature1"]))) > (data["svdfeature3"]))*1.)))) +
                    0.100000*np.tanh((((np.minimum(((data["pcafeature7"])), ((np.minimum(((data["pcafeature4"])), ((np.minimum(((data["pcafeature4"])), ((-1.0)))))))))) < (-1.0))*1.)) +
                    0.100000*np.tanh((-1.0*(((((((data["svdfeature1"]) * 2.0)) > (np.where(data["pcafeature5"]>0, (1.54018199443817139), np.where(data["svdfeature1"]>0, 3.141593, -3.0 ) )))*1.))))) +
                    0.100000*np.tanh((((data["pcafeature7"]) < (np.where(data["pcafeature7"]>0, ((data["pcafeature5"]) - (((data["pcafeature4"]) * (data["svdfeature10"])))), -1.0 )))*1.)) +
                    0.100000*np.tanh((((((((((data["svdfeature3"]) * (np.where(data["svdfeature3"]>0, data["svdfeature3"], -1.0 )))) < (0.318310))*1.)) * 2.0)) * 2.0)) +
                    0.099961*np.tanh(((3.141593) * (((3.141593) * ((((3.141593) < (((data["pcafeature4"]) + ((((data["pcafeature7"]) < (1.570796))*1.)))))*1.)))))) +
                    0.100000*np.tanh((-1.0*(((((2.0) < (data["pcafeature8"]))*1.))))) +
                    0.100000*np.tanh(((np.minimum((((((data["svdfeature10"]) > (data["pcafeature8"]))*1.))), (((((data["pcafeature5"]) > (np.tanh(((((data["svdfeature10"]) > (data["pcafeature7"]))*1.)))))*1.))))) * 2.0)) +
                    0.100000*np.tanh((((-1.0*(((5.0))))) * ((((np.where(data["svdfeature10"]>0, data["pcafeature8"], (-1.0*((0.636620))) )) < ((-1.0*((0.636620)))))*1.)))) +
                    0.100000*np.tanh((((((1.0) < (data["svdfeature10"]))*1.)) * ((((((((data["svdfeature10"]) - (data["pcafeature8"]))) + (data["svdfeature10"]))/2.0)) * (data["svdfeature3"]))))) +
                    0.099980*np.tanh((-1.0*((((((((-1.0*((data["pcafeature4"])))) - (np.minimum(((((data["svdfeature3"]) - (data["pcafeature4"])))), ((data["pcafeature4"])))))) < (data["svdfeature9"]))*1.))))) +
                    0.099941*np.tanh(((((((((data["pcafeature5"]) < ((((data["pcafeature3"]) < (np.tanh((data["pcafeature5"]))))*1.)))*1.)) + (((data["pcafeature3"]) / 2.0)))/2.0)) / 2.0)) +
                    0.100000*np.tanh((((((((((data["pcafeature5"]) > (np.tanh(((0.96179389953613281)))))*1.)) * (data["pcafeature3"]))) * ((((data["pcafeature4"]) < (1.0))*1.)))) * 2.0)) +
                    0.099941*np.tanh(np.tanh((np.tanh((((np.maximum(((((data["pcafeature8"]) * (data["pcafeature5"])))), ((data["pcafeature3"])))) * (((data["pcafeature4"]) / 2.0)))))))) +
                    0.100000*np.tanh(np.where(((1.570796) + (data["pcafeature5"]))>0, (-1.0*(((((((data["pcafeature5"]) + (data["pcafeature3"]))) > (1.570796))*1.)))), data["pcafeature3"] )) +
                    0.100000*np.tanh(((((((data["pcafeature8"]) < (((-1.0) / 2.0)))*1.)) < ((((0.318310) < (np.where(data["pcafeature3"]>0, -1.0, data["svdfeature10"] )))*1.)))*1.)) +
                    0.100000*np.tanh((((((data["svdfeature1"]) < ((-1.0*(((2.95451116561889648))))))*1.)) * (np.maximum((((-1.0*((data["svdfeature1"]))))), ((data["svdfeature1"])))))) +
                    0.099980*np.tanh((((9.74876594543457031)) * (((((((data["svdfeature10"]) + ((((0.636620) < (-1.0))*1.)))/2.0)) > (np.tanh((0.636620))))*1.)))) +
                    0.098965*np.tanh(((data["pcafeature5"]) * (((data["svdfeature3"]) * (((((-1.0*(((((data["pcafeature4"]) < (data["svdfeature7"]))*1.))))) > (data["svdfeature1"]))*1.)))))) +
                    0.099980*np.tanh((-1.0*((((((np.maximum(((np.minimum(((data["pcafeature5"])), ((data["svdfeature3"]))))), ((np.where(data["pcafeature5"]>0, data["pcafeature7"], 0.0 ))))) / 2.0)) / 2.0))))) +
                    0.100000*np.tanh((((((data["pcafeature5"]) > (np.tanh((np.tanh((3.141593))))))*1.)) * (((data["pcafeature5"]) - (data["svdfeature1"]))))) +
                    0.087849*np.tanh(((((((((-3.0) - (data["pcafeature5"]))) > (data["pcafeature5"]))*1.)) < (((-3.0) - (((data["pcafeature4"]) * (data["svdfeature10"]))))))*1.)) +
                    0.074937*np.tanh((-1.0*((np.where(data["svdfeature1"]>0, (((2.0) < (data["pcafeature7"]))*1.), ((((((data["svdfeature1"]) > (-3.0))*1.)) + (data["pcafeature7"]))/2.0) ))))) +
                    0.091561*np.tanh((((((((data["pcafeature3"]) + (((((0.0) * (0.0))) * (0.0))))/2.0)) / 2.0)) / 2.0)) +
                    0.081403*np.tanh((((1.570796) < ((((((np.tanh((0.318310))) + (data["pcafeature5"]))/2.0)) * (3.0))))*1.)) +
                    0.099980*np.tanh((((1.53350508213043213)) - (np.maximum(((np.maximum(((((data["pcafeature5"]) * 2.0))), ((np.maximum((((1.53350508213043213))), ((data["svdfeature10"])))))))), ((((data["svdfeature10"]) * 2.0))))))) +
                    0.100000*np.tanh(((((((((data["pcafeature3"]) / 2.0)) > ((0.16701583564281464)))*1.)) < (np.minimum(((np.where(data["pcafeature4"]>0, data["pcafeature8"], 0.0 ))), ((data["svdfeature10"])))))*1.)) +
                    0.100000*np.tanh((((np.tanh((np.where(data["svdfeature10"]>0, np.where(data["svdfeature7"]>0, data["svdfeature1"], 0.318310 ), data["svdfeature7"] )))) + (((data["svdfeature9"]) / 2.0)))/2.0)) +
                    0.100000*np.tanh((((-1.0*(((((((np.tanh((data["pcafeature5"]))) > (((((2.0)) + ((-1.0*((data["pcafeature5"])))))/2.0)))*1.)) * (data["svdfeature1"])))))) * 2.0)) +
                    0.100000*np.tanh((((np.minimum(((data["pcafeature5"])), ((np.maximum(((((data["svdfeature3"]) + (np.maximum(((data["pcafeature8"])), ((data["svdfeature10"]))))))), (((0.82600969076156616)))))))) > ((0.82600969076156616)))*1.)) +
                    0.100000*np.tanh((((((data["pcafeature7"]) > (2.0))*1.)) * (((((-1.0*((data["pcafeature7"])))) + ((-1.0*((data["pcafeature7"])))))/2.0)))) +
                    0.100000*np.tanh((((np.where(data["pcafeature7"]>0, np.tanh((data["pcafeature4"])), (((8.0)) / 2.0) )) < (np.tanh((np.minimum(((data["svdfeature3"])), ((data["svdfeature10"])))))))*1.)) +
                    0.099980*np.tanh(((((data["pcafeature8"]) * ((2.34932589530944824)))) * ((((data["pcafeature5"]) > ((((0.636620) + ((((data["svdfeature10"]) + (3.0))/2.0)))/2.0)))*1.)))) +
                    0.070756*np.tanh((-1.0*((((((((((data["pcafeature8"]) < (1.570796))*1.)) < (np.maximum(((((data["svdfeature3"]) + (data["svdfeature9"])))), ((data["svdfeature9"])))))*1.)) / 2.0))))) +
                    0.098652*np.tanh((((((data["svdfeature3"]) > (((((((data["pcafeature8"]) + (np.tanh((np.tanh((data["pcafeature7"]))))))/2.0)) + (3.0))/2.0)))*1.)) / 2.0)) +
                    0.098203*np.tanh(((((((data["pcafeature8"]) + (data["svdfeature10"]))/2.0)) < (np.minimum(((-1.0)), ((-1.0)))))*1.)) +
                    0.100000*np.tanh(((data["pcafeature8"]) * (np.maximum(((((((data["svdfeature9"]) / 2.0)) * ((-1.0*((data["svdfeature10"]))))))), ((0.0)))))) +
                    0.100000*np.tanh(((((((np.where((((data["pcafeature3"]) > ((-1.0*((data["pcafeature8"])))))*1.)>0, (-1.0*((data["pcafeature8"]))), data["pcafeature5"] )) / 2.0)) / 2.0)) / 2.0)) +
                    0.099980*np.tanh((((((((((((-1.0*((((((6.0)) < ((7.0)))*1.))))) > (data["svdfeature10"]))*1.)) + (data["svdfeature9"]))/2.0)) * (data["icafeature8"]))) / 2.0)) +
                    0.099980*np.tanh((-1.0*(((((((2.0) - (data["pcafeature7"]))) < (((((((3.0) + (1.570796))/2.0)) < (data["pcafeature4"]))*1.)))*1.))))) +
                    0.099727*np.tanh(((np.minimum(((data["pcafeature8"])), ((((((((data["pcafeature4"]) < (1.0))*1.)) + ((-1.0*((data["pcafeature8"])))))/2.0))))) + (0.636620))) +
                    0.100000*np.tanh(((np.where(data["pcafeature5"]>0, data["svdfeature9"], data["svdfeature9"] )) * ((((((data["pcafeature7"]) / 2.0)) > (np.tanh((1.570796))))*1.)))) +
                    0.100000*np.tanh(((((((data["svdfeature1"]) > (data["pcafeature7"]))*1.)) < ((((((((((data["pcafeature7"]) > (-1.0))*1.)) > (data["pcafeature5"]))*1.)) < (data["svdfeature1"]))*1.)))*1.)) +
                    0.099883*np.tanh((((((np.tanh((np.tanh((data["svdfeature10"]))))) > (data["pcafeature7"]))*1.)) - ((((data["svdfeature10"]) > (np.tanh((data["pcafeature7"]))))*1.)))) +
                    0.100000*np.tanh(((data["svdfeature7"]) * (((((((data["pcafeature5"]) > (np.tanh((np.tanh((2.0))))))*1.)) > ((((data["svdfeature9"]) < (data["svdfeature7"]))*1.)))*1.)))) +
                    0.099961*np.tanh((((data["pcafeature5"]) > ((((data["pcafeature5"]) > (np.tanh((((((((data["pcafeature5"]) > ((-1.0*((data["pcafeature7"])))))*1.)) < (data["pcafeature5"]))*1.)))))*1.)))*1.)) +
                    0.100000*np.tanh((((((((((data["svdfeature1"]) * 2.0)) < (-3.0))*1.)) - ((((0.318310) + (((((1.34650146961212158)) < (data["svdfeature1"]))*1.)))/2.0)))) / 2.0)) +
                    0.097187*np.tanh((((((((-1.0*(((((np.maximum(((((data["pcafeature5"]) / 2.0))), ((((data["svdfeature1"]) * 2.0))))) < (0.318310))*1.))))) * 2.0)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh((((((data["svdfeature10"]) * (np.tanh(((((3.0) + ((-1.0*((data["pcafeature4"])))))/2.0)))))) > (1.0))*1.)) +
                    0.100000*np.tanh((-1.0*(((((data["pcafeature7"]) > (np.where(data["svdfeature3"]>0, (((3.0) + (data["svdfeature10"]))/2.0), ((3.0) + (data["svdfeature3"])) )))*1.))))) +
                    0.099980*np.tanh((((((-2.0) > (((data["pcafeature4"]) * (np.where(data["svdfeature3"]>0, data["pcafeature7"], ((np.tanh((data["pcafeature7"]))) * 2.0) )))))*1.)) * 2.0)) +
                    0.099980*np.tanh((((((((np.minimum(((-1.0)), ((data["pcafeature5"])))) > (np.minimum(((np.minimum(((data["pcafeature4"])), ((data["pcafeature7"]))))), ((-1.0)))))*1.)) * 2.0)) * 2.0)) +
                    0.100000*np.tanh(((((((data["pcafeature4"]) > (data["pcafeature7"]))*1.)) < ((((((data["pcafeature8"]) < (data["svdfeature10"]))*1.)) - ((((data["pcafeature7"]) < (data["pcafeature8"]))*1.)))))*1.)) +
                    0.099961*np.tanh((((((-1.0*(((((((data["pcafeature3"]) * (data["pcafeature3"]))) > (3.0))*1.))))) * 2.0)) * (((data["pcafeature3"]) * (data["svdfeature10"]))))) +
                    0.100000*np.tanh(np.minimum((((((data["svdfeature10"]) > (0.636620))*1.))), (((-1.0*((((data["svdfeature10"]) - ((-1.0*((((data["svdfeature10"]) - (1.570796)))))))))))))) +
                    0.099980*np.tanh((((-1.0*(((((data["svdfeature9"]) > ((((2.23544645309448242)) / 2.0)))*1.))))) * ((((((data["svdfeature10"]) + ((2.23544645309448242)))/2.0)) + (data["pcafeature7"]))))) +
                    0.100000*np.tanh(((((data["svdfeature3"]) * ((((np.maximum(((data["pcafeature5"])), ((1.0)))) < (data["svdfeature10"]))*1.)))) * (np.maximum(((data["pcafeature4"])), ((data["svdfeature3"])))))) +
                    0.099980*np.tanh(((((-1.0*((data["svdfeature10"])))) > (np.maximum(((1.570796)), ((data["svdfeature10"])))))*1.)) +
                    0.100000*np.tanh((((((2.0) < (np.where((((-1.0) + (data["svdfeature10"]))/2.0)>0, data["pcafeature8"], 2.0 )))*1.)) * 2.0)) +
                    0.099980*np.tanh(((((((5.0)) < ((((((0.0) > ((5.0)))*1.)) - (data["svdfeature1"]))))*1.)) * ((5.0)))) +
                    0.077496*np.tanh((((((((2.52607297897338867)) < (((data["svdfeature10"]) * 2.0)))*1.)) + ((((((data["pcafeature4"]) + ((2.52607297897338867)))) < (((data["svdfeature10"]) * 2.0)))*1.)))/2.0)) +
                    0.099980*np.tanh((((3.0) < (np.where(data["pcafeature3"]>0, ((data["pcafeature5"]) * (np.where(data["pcafeature4"]>0, 3.0, (3.53686904907226562) ))), 3.0 )))*1.)) +
                    0.100000*np.tanh(((np.minimum((((((data["pcafeature7"]) + (0.636620))/2.0))), (((((((1.0) < (data["pcafeature5"]))*1.)) * (data["pcafeature7"])))))) * ((5.0)))) +
                    0.099980*np.tanh((((((((-1.0*(((((2.0) < (((data["svdfeature3"]) * (data["pcafeature3"]))))*1.))))) * 2.0)) * 2.0)) * 2.0)) +
                    0.097441*np.tanh(((np.tanh(((((np.tanh((np.tanh((0.318310))))) < (((data["pcafeature4"]) * (data["pcafeature7"]))))*1.)))) / 2.0)) +
                    0.100000*np.tanh((((((-1.0*((((((-1.0*((data["pcafeature11"])))) < (((data["pcafeature1"]) / 2.0)))*1.))))) / 2.0)) / 2.0)) +
                    0.099961*np.tanh((-1.0*((((((((3.70939588546752930)) / 2.0)) < (np.maximum(((data["pcafeature7"])), ((((data["pcafeature7"]) - (((data["svdfeature1"]) * (data["pcafeature7"])))))))))*1.))))) +
                    0.096777*np.tanh((((data["pcafeature4"]) > (((np.minimum(((((data["pcafeature7"]) - (data["pcafeature5"])))), ((data["pcafeature7"])))) + (3.141593))))*1.)) +
                    0.083708*np.tanh(np.where(data["pcafeature5"]>0, 0.0, np.tanh((np.tanh((np.where(((data["pcafeature3"]) + (data["pcafeature7"]))>0, 0.318310, data["pcafeature7"] ))))) )) +
                    0.099980*np.tanh((((((np.minimum(((data["pcafeature7"])), ((data["pcafeature7"])))) < (-1.0))*1.)) * 2.0)) +
                    0.099941*np.tanh(((data["svdfeature1"]) * (np.where(data["pcafeature7"]>0, ((((((data["svdfeature3"]) > (data["svdfeature1"]))*1.)) > (data["svdfeature1"]))*1.), (0.07243753969669342) )))) +
                    0.090760*np.tanh(np.minimum(((((3.141593) * (np.minimum(((((data["pcafeature7"]) + (0.636620)))), ((0.636620))))))), ((0.0)))) +
                    0.100000*np.tanh((((((((-2.0) * (data["pcafeature11"]))) > (np.maximum((((((data["icafeature8"]) + (3.0))/2.0))), ((np.tanh((3.141593)))))))*1.)) / 2.0)) +
                    0.099980*np.tanh((((-1.0) + (((((((0.81793087720870972)) * ((((((((0.81793087720870972)) > (data["pcafeature3"]))*1.)) > (data["svdfeature9"]))*1.)))) > (data["svdfeature10"]))*1.)))/2.0)) +
                    0.100000*np.tanh(((((data["svdfeature10"]) + (data["pcafeature11"]))) * ((((((data["pcafeature11"]) > (1.570796))*1.)) * (data["svdfeature10"]))))) +
                    0.099531*np.tanh((((((2.0) < (((data["pcafeature4"]) * (np.where(data["pcafeature3"]>0, data["pcafeature3"], ((data["pcafeature8"]) / 2.0) )))))*1.)) * (data["pcafeature4"]))) +
                    0.100000*np.tanh(((((data["svdfeature9"]) * 2.0)) * ((-1.0*((((data["pcafeature4"]) * (((((4.77691841125488281)) < (((((data["svdfeature9"]) * 2.0)) * 2.0)))*1.))))))))) +
                    0.086228*np.tanh((-1.0*(((((((((((data["svdfeature10"]) < (data["svdfeature10"]))*1.)) > (-1.0))*1.)) < (data["svdfeature10"]))*1.))))))


a = pd.DataFrame()
a['gp1'] = GP1(alldata[:train.shape[0]]).values
a['gp2'] = GP2(alldata[:train.shape[0]]).values
a = a.values
y = traindata.target.ravel()
from sklearn.model_selection import KFold
folds = KFold(n_splits=5, shuffle=True, random_state=42)
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(a)):
    print(n_fold)
    trn_x, trn_y = a[trn_idx], y[trn_idx]
    val_x, val_y = a[val_idx], y[val_idx]
    
       
    clf = KNeighborsRegressor(n_neighbors=100)
    clf.fit(trn_x,trn_y)
    print(np.sqrt(mean_squared_error(val_y,clf.predict(val_x))))
    
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
plt.figure(figsize=(10,10))
plt.scatter(GP1(alldata[:train.shape[0]]),GP2(alldata[:train.shape[0]]),s=1)
plt.figure(figsize=(10,10))
plt.scatter(GP1(alldata[train.shape[0]:]),GP2(alldata[train.shape[0]:]),s=1)