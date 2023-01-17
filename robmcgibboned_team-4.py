import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
path = "kaggle/input/"
signalData = pd.read_csv(path+"bjet_train.csv") # signal has mc_flavour = 5
backgroundData = pd.concat([pd.read_csv(path+"cjet_train.csv"), 
                            pd.read_csv(path+"ljet_train.csv")]) # background has mc_flavour != 5

signalData['nTrkRatio'] = signalData['nTrk']/signalData['nTrkJet']
backgroundData['nTrkRatio'] = backgroundData['nTrk']/backgroundData['nTrkJet']
signalData['mRatio'] = signalData['mCor']/signalData['m']
backgroundData['mRatio'] = backgroundData['mCor']/backgroundData['m']
signalData['mCorSig'] = signalData['mCor']/signalData['mCorErr']
backgroundData['mCorSig'] = backgroundData['mCor']/backgroundData['mCorErr']
# Try fdChi2 as log10, others as linear
toLogCol = ['fdChi2', 'PT', 'nTrk', 'nTrkJet']
linCol = ['ETA', 'drSvrJet', 'fdrMin', 
          'm', 'mCor', 'mCorErr', 'pt', 'ptSvrJet',
          'tau', 'ipChi2Sum'] # Note skip Id as that is not helpful

logCol = []
for l in toLogCol:
    logCol.append('log_'+l)
    signalData['log_'+l] = np.log10(signalData[l])
    backgroundData['log_'+l] = np.log10(backgroundData[l])

    
X = np.concatenate([signalData[logCol+linCol].values,
                         backgroundData[logCol+linCol].values])
y = np.concatenate([(signalData["mc_flavour"]==5).values.astype(np.int),
                         (backgroundData["mc_flavour"]==5).values.astype(np.int)])

p = np.random.permutation(len(X))
X = X[p]
y = y[p]
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sklearn.preprocessing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)


qt = sklearn.preprocessing.QuantileTransformer(n_quantiles=10)
X_train = qt.fit_transform(X_train)
X_test = qt.transform(X_test)
import sklearn.ensemble

# Parameters were found using random search
best_params = {'n_estimators': 800, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 89, 'bootstrap': True}
clf = sklearn.ensemble.RandomForestClassifier(**best_params, n_jobs=-1, class_weight='balanced')

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(accuracy_score(y_test, pred))
testData = pd.read_csv(path+"competitionData.csv")

testData['nTrkRatio'] = testData['nTrk']/testData['nTrkJet']
testData['mRatio'] = testData['mCor']/testData['m']
testData['mCorSig'] = testData['mCor']/testData['mCorErr']
## Transform data

logCol = []
for l in toLogCol:
    logCol.append('log_'+l)
    testData['log_'+l] = np.log10(testData[l])
x_comp = testData[logCol+linCol].values

x_comp = qt.transform(x_comp)

predMCFloat = clf.predict(x_comp)
predMC = (predMCFloat>0.5).astype(np.int)
testData["Prediction1"] = predMC

# solution to submit
display(testData[["Id","Prediction1"]]) # display 5 rows
# write to a csv file for submission
testData.to_csv("submit.csv.gz",index=False,columns=["Id","Prediction1"],compression="gzip") # Output a compressed csv file for submission: see /kaggle/working to the right
