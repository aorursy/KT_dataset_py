#!pip install modin
import numpy as np # linear algebra
#import modin.pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
import os
print(os.listdir("../input"))
print(os.listdir("../input/explanation-of-the-current-training-set-ipynb"))
print(os.listdir("../input/all-the-data-to-build-the-testfeaturetable"))
print(os.listdir("../input/fastreadandextractl460-500"))
# Any results you write to the current directory are saved as output.
trainDf=pd.read_csv("../input/explanation-of-the-current-training-set-ipynb/trainingSetToMatchCustomTestSet111518.csv")
trainDf.shape
testDfBegin=pd.read_csv("../input/all-the-data-to-build-the-testfeaturetable/Objects0Through3219999.csv")
print(testDfBegin.shape)
testDfEnd=pd.read_csv("../input/fastreadandextractl460-500/testFeaturesFrom3220000TO3492890.csv")
print(testDfEnd.shape)
testDfBegin=testDfBegin.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
print(testDfBegin.shape)
testDfBegin.columns
trainDf=trainDf.drop('Unnamed: 0', axis=1)
#print(testDfEnd.shape)

testDfEnd=testDfEnd.drop(['Unnamed: 0'], axis=1)
print(testDfEnd.shape)
testDf=testDfBegin.append(testDfEnd, sort=False)
testDf.shape

#missing values in:
#distmod
#hostgal_specz - 
#deltaDetect - NaN would mean nothing was detected treat as zero
trainDf['deltaDetect'].fillna(0,inplace=True)
testDf['deltaDetect'].fillna(0, inplace=True)
testDf.describe()
nullFilter=testDf.loc[:,'distmod'].isnull()
missingDm=testDf.loc[nullFilter]
missingDm.head()
missingDm.describe()

#tig is training intergalactic.  For test we'll just use ig
igdf=missingDm
egdf=testDf[nullFilter==False]

print(igdf.shape)
print(egdf.shape)
eobjdf=pd.DataFrame()
eobjdf.loc[:,'object_id']=egdf.loc[:,'object_id']
eobjdf.head()

iobjdf=pd.DataFrame()
iobjdf.loc[:,'object_id']=igdf.loc[:,'object_id']
iobjdf.head()
igdf.describe()
egdf.describe()
igdf.columns
#!pip install -U imbalanced-learn
#traindf=pd.read_csv('../input/trainingSetToMatchCustomTestSet111518.csv')
traindf=trainDf
print(traindf.shape)
print(traindf.head())

#hostgal_specz isn't in most rows of the test data
def dropUselessFeatures(df):
    print(df.shape)
    df=df.drop(['hephs','hepos','hepts','lephs','lepos','lepts','hostgal_specz',
                  'hmphs','hmpos','hmpts','lmphs','lmpos','lmpts',
                  'hlphs','hlpos','hlpts','llphs','llpos','llpts',
                'highEnergy_transitory_1.0_TF', 
                'highEnergy_transitory_1.5_TF',
                'lowEnergy_transitory_1.0_TF', 
                'lowEnergy_transitory_1.5_TF', 
               'object_id'], axis=1)
    
    #df.loc[:,'hMin']=np.min([df.loc[:,'heavg'], df.loc[:,'hmavg'],df.loc[:,'hlavg']])
    #df.loc[:,'hMax']=np.max([df.loc[:,'heavg'], df.loc[:,'hmavg'],df.loc[:,'hlavg']])
    #df.loc[:,'hVar']=np.average([df.loc[:,'hestd'], df.loc[:,'hmstd'],df.loc[:,'hlstd']])
    #df.loc[:,'hSpread']=(df.loc[:,'hMax']-df.loc[:,'hMin']) / df.loc[:,'hVar']
    
    #df.loc[:,'lMin']=np.min([df.loc[:,'leavg'], df.loc[:,'lmavg'],df.loc[:,'llavg']])
    #df.loc[:,'lMax']=np.max([df.loc[:,'leavg'], df.loc[:,'lmavg'],df.loc[:,'llavg']])
    #df.loc[:,'lVar']=np.average([df.loc[:,'lestd'], df.loc[:,'lmstd'],df.loc[:,'llstd']])
    #df.loc[:,'lSpread']=(df.loc[:,'lMax']-df.loc[:,'lMin']) / df.loc[:,'lVar']
    
    #df=df.drop(['hMin','hMax','hVar','lMin','lMax','lVar'],axis=1)
    
    print(df.shape)
    return df

traindf=dropUselessFeatures(traindf)


traindf.head()
traindf.columns
traindf.loc[:,'target']=traindf.loc[:,'target'].astype(str)

#from stacy's code
# move target to end
traindf = traindf[[c for c in traindf if c not in ['target']] + ['target']]
traindf.head()

#df[1].fillna(0, inplace=True)
traindf['deltaDetect'].fillna(0,inplace=True)
tigdf=traindf[traindf['hostgal_photoz']==0]
tegdf=traindf[traindf['hostgal_photoz']!=0]

print(tigdf.shape)
print(tegdf.shape)


tigdf=tigdf.drop(['hostgal_photoz', 'distmod', 'hostgal_photoz_err'], axis=1)
print(tigdf.shape)
igdf=dropUselessFeatures(igdf)
print(igdf.shape)
egdf=dropUselessFeatures(egdf)
print(egdf.shape)
igdf=igdf.drop(['hostgal_photoz', 'distmod', 'hostgal_photoz_err'], axis=1)
print(igdf.shape)
print('inter-galactic')
for theClass in tigdf.loc[:,'target'].unique():
    print('class ' + str(theClass) + ':')
    trueFilter=tigdf['target']==theClass
    print(trueFilter.sum())
print('extra-galactic')
for theClass in tegdf.loc[:,'target'].unique():
    print('class ' + str(theClass) + ':')
    trueFilter=tegdf['target']==theClass
    print(trueFilter.sum())
#https://www.kaggle.com/qianchao/smote-with-imbalance-data
#from sklearn.preprocessing import StandardScaler
Xig = np.array(tigdf.iloc[:, tigdf.columns != 'target'])
yig = np.array(tigdf.iloc[:, tigdf.columns == 'target'])
print('Shape of X: {}'.format(Xig.shape))
print('Shape of y: {}'.format(yig.shape))

Xeg = np.array(tegdf.iloc[:, tegdf.columns != 'target'])
yeg = np.array(tegdf.iloc[:, tegdf.columns == 'target'])
print('Shape of X: {}'.format(Xeg.shape))
print('Shape of y: {}'.format(yeg.shape))
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

Xig_train, Xig_test, yig_train, yig_test = train_test_split(Xig, yig, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", Xig_train.shape)
print("Number transactions y_train dataset: ", yig_train.shape)
print("Number transactions X_test dataset: ", Xig_test.shape)
print("Number transactions y_test dataset: ", yig_test.shape)
print("Before OverSampling, counts of label '92': {}".format(sum(yig_train=='92')))
print("Before OverSampling, counts of label '65': {} \n".format(sum(yig_train=='65')))
print("Before OverSampling, counts of label '16': {}".format(sum(yig_train=='16')))
print("Before OverSampling, counts of label '6': {} \n".format(sum(yig_train=='6')))
print("Before OverSampling, counts of label '53': {}".format(sum(yig_train=='53')))

sm = SMOTE(random_state=2)
Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(Xig_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(yig_train_res.shape))

print("After OverSampling, counts of label '92': {}".format(sum(yig_train_res=='92')))
print("After OverSampling, counts of label '65': {}".format(sum(yig_train_res=='65')))
print("After OverSampling, counts of label '16': {}".format(sum(yig_train_res=='16')))
print("After OverSampling, counts of label '6': {}".format(sum(yig_train_res=='6')))
print("After OverSampling, counts of label '53': {}".format(sum(yig_train_res=='53')))

def smoteAdataset(Xig, yig, test_size=0.2, random_state=0):
    
    Xig_train, Xig_test, yig_train, yig_test = train_test_split(Xig, yig, test_size=test_size, random_state=random_state)
    print("Number transactions X_train dataset: ", Xig_train.shape)
    print("Number transactions y_train dataset: ", yig_train.shape)
    print("Number transactions X_test dataset: ", Xig_test.shape)
    print("Number transactions y_test dataset: ", yig_test.shape)

    classes=[]
    for i in np.unique(yig):
        classes.append(i)
        print("Before OverSampling, counts of label " + str(i) + ": {}".format(sum(yig_train==i)))
        
    sm=SMOTE(random_state=2)
    Xig_train_res, yig_train_res = sm.fit_sample(Xig_train, yig_train.ravel())

    print('After OverSampling, the shape of train_X: {}'.format(Xig_train_res.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(yig_train_res.shape))
    
    for eachClass in classes:
        print("After OverSampling, counts of label " + str(eachClass) + ": {}".format(sum(yig_train_res==eachClass)))
        
    return Xig_train_res, yig_train_res, Xig_test, yig_test

Xeg_train_res, yeg_train_res, Xeg_test, yeg_test=smoteAdataset(Xeg, yeg)

#from HW5
from sklearn.ensemble import GradientBoostingClassifier

def getGbm(X_encoded_train, Y1, X_encoded_test, Y2,
           nTrees=100, max_depth=5, min_node_size=5, verbose=0, learning_rate=0.05):

    gbm_clf = GradientBoostingClassifier(n_estimators=nTrees, loss='deviance', learning_rate=learning_rate, max_depth=max_depth, \
                                        min_samples_leaf=min_node_size)
    gbm_clf.fit(X_encoded_train, Y1)
    
    
    Y_test_hat = gbm_clf.predict_proba(X_encoded_test)
    #Accuracy = [1 for i in range(len(Y_test_hat)) if Y2.iloc[i] == Y_test_hat[i]]
    #Accuracy = round(float(np.sum(Accuracy))/len(Y_test_hat)*100,2)
    #rocAuc=roc_auc_score(Y2Vals, Y_test_hat)
    
    #Y1Vals=np.array(Y1)
    
    #Y_train_hat = clf.predict(X_encoded_train)
    #trainAcc = [1 for i in range(len(Y_train_hat)) if Y1.iloc[i] == Y_train_hat[i]]
    #trainAcc = round(float(np.sum(trainAcc))/len(Y_train_hat)*100,2)
    #trainAuc=roc_auc_score(Y1Vals, Y_train_hat)
    
    
    return gbm_clf, Y_test_hat
tiggbm_clf, tigY_test_hat=getGbm(Xig_train_res, yig_train_res, Xig_test, yig_test,
                                 nTrees=100, max_depth=5, min_node_size=5, verbose=0, learning_rate=0.05)

print(tiggbm_clf.feature_importances_)
#print("Accuracy on Testing Data = %.2f%%"%Accuracy)
#print("AUC for ROC curve on Testing Data = %.2f"%rocAuc)
#print("Accuracy on Training Data = %.2f%%"%trainAcc)
#print("AUC for ROC curve on Training Data = %.2f"%trainAuc)
print(tigY_test_hat)
print(tiggbm_clf.classes_)
teggbm_clf, tegY_test_hat=getGbm(Xeg_train_res, yeg_train_res, Xeg_test, yeg_test,
                                 nTrees=100, max_depth=5, min_node_size=5, verbose=0, learning_rate=0.05)

print(teggbm_clf.feature_importances_)
print(tegY_test_hat)
print(teggbm_clf.classes_)
actualIgX=igdf.values
actualIgPredictions = tiggbm_clf.predict_proba(actualIgX)
#igPredictions = tiggbm_clf.predict_proba(X_encoded_test)
actualEgX=egdf.values
actualEgPredictions = teggbm_clf.predict_proba(actualEgX)
#igPredictions = tiggbm_clf.predict_proba(X_encoded_test)
actualIgPredictions.shape
actualEgPredictions.shape
iobjdf.index=range(iobjdf.shape[0])
iobjdf.head()

eobjdf.index=range(eobjdf.shape[0])
eobjdf.head()
igResultDf=pd.DataFrame(data=actualIgPredictions, columns=tiggbm_clf.classes_)
#igResultDf.loc[:,'object_id']=igdf.loc[:,'object_id']
igResultDf.loc[:,'object_id']=iobjdf.loc[:,'object_id']
igResultDf.head()
egResultDf=pd.DataFrame(data=actualEgPredictions, columns=teggbm_clf.classes_)
#igResultDf.loc[:,'object_id']=igdf.loc[:,'object_id']
egResultDf.loc[:,'object_id']=eobjdf.loc[:,'object_id']
egResultDf.head()
fulldf=igResultDf.append(egResultDf, sort=False)
print(fulldf.shape)
import copy
nobjdf=copy.deepcopy(fulldf)
nobjdf=nobjdf.drop('object_id', axis=1)
nobjdf['max_value'] = nobjdf.max(axis=1)
print(nobjdf.head())
#I chose 7000 arbitrarily because its about 1/5 of 1%

#arbGuess99=7000
#arb99Proba=1.00
#mysteryThresh=np.max(nobjdf.nsmallest(arbGuess99, 'max_value').loc[:,'max_value'])
#print(mysteryThresh)
#nobjdf['99']=0
#nobjdf.loc[nobjdf['max_value']<]

averages=[]
for columns in nobjdf.columns:
    nobjdf[columns].fillna(0, inplace=True)
    #print(columns)
    averages.append(np.average(nobjdf.loc[:,columns]))
    #print(np.average(nobjdf.loc[:,columns]))
    
nobjdf.loc[:,'g99']=1-nobjdf.loc[:,'max_value']**2
#print(np.min(averages))
#print(nobjdf.head())
normalizingConstant = np.average(nobjdf.loc[:,'g99'])/np.min(averages)
print(normalizingConstant)
fulldf.loc[:,'99']=nobjdf.loc[:,'g99']/normalizingConstant
fulldf.head()
for cindex in fulldf.columns:
    fulldf[cindex].fillna(0, inplace=True)
    
fulldf.describe()
def submitOrder(fulldf):
    
    deleteCols=[]
    for cindex in fulldf.columns:
        fulldf = fulldf.rename(columns={cindex: 'was'+str(cindex)})
        deleteCols.append('was'+str(cindex))
    
    newNames=[6,15, 16, 42, 52,53,62,
             64,65,67,88,90,92,95,99]
    
    #string column names seemed to cause a problem
    fulldf.loc[:,'object_id']=fulldf.loc[:,'wasobject_id']
    for name in newNames:
        fulldf.loc[:,'class_'+str(name)]=fulldf.loc[:,'was'+str(name)]
    
    fulldf=fulldf.drop(deleteCols, axis=1)
    return fulldf


submitdf=submitOrder(fulldf)
#for cindex in submitdf.columns:
#    if cindex != 'object_id':
#        submitdf.loc[:,cindex]=submitdf.loc[:,cindex].astype(float)
print(submitdf.shape)
print(submitdf.columns)
submitdf.head()
submitdf.to_csv('fastFeatsOnly111718.csv', index=False)