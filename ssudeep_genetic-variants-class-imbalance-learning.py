import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import re

import operator

import warnings

from collections import Counter

from itertools import chain

from time import time

# feature selection

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV,SelectFromModel

# classifiers

from xgboost import XGBClassifier

from sklearn.preprocessing import RobustScaler

from sklearn.svm import SVC

from sklearn.ensemble import IsolationForest

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import RandomizedSearchCV,cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import PolynomialFeatures

from sklearn.utils import resample,shuffle

import matplotlib.pyplot as plt
vardat = pd.read_csv('../input/clinvar_conflicting.csv',dtype={0:object,38:object,40:object})

# explicity define dtype for pandas dtype error

vardat.head()
print(vardat.shape)

print(vardat.columns)
vardat.describe()
sns.countplot(vardat['CLASS'])

plt.show()

print(Counter(vardat['CLASS'].values))
vardat.groupby(['CHROM','CLASS']).size()
Counter(vardat[['REF', 'ALT']].apply(lambda x: ':'.join(x), axis=1))
print(Counter(vardat['ORIGIN'].values))
vardat['ORIGIN'].fillna(0, inplace=True)

print(Counter(vardat['ORIGIN'].values))
cons = Counter(list(chain.from_iterable([str(v).split('&') for v in vardat['Consequence'].values])))

sorted(cons.items(),key=operator.itemgetter(1),reverse=True)
clnvc = Counter(vardat['CLNVC'].values)

sorted(clnvc.items(),key=operator.itemgetter(1),reverse=True)
clndn = Counter(list(chain.from_iterable([str(v).split('|') for v in vardat['CLNDN'].values])))

sorted(clndn.items(),key=operator.itemgetter(1),reverse=True)
var_vals = vardat[['AF_ESP','AF_EXAC','AF_TGP','CADD_PHRED','CADD_RAW','CLASS']].dropna()

print(var_vals.info())

print(var_vals.describe())

print(Counter(var_vals['CLASS']))
sns.pairplot(var_vals,hue='CLASS')

plt.show()
var_corr = vardat[['AF_ESP','AF_EXAC','AF_TGP','CADD_PHRED','CADD_RAW']].corr()

cmap = sns.diverging_palette(220, 20, n=7,as_cmap=True)

sns.heatmap(var_corr, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5,annot=True,cbar_kws={"shrink": .5})
# Below are some functions used to encode various columsn

def consequence_encoder(consequence,consdict):

    '''

    encoder for consequence data

    '''

    outmat = np.zeros((len(consequence),len(consdict)),dtype=np.int)

    for i,cons in enumerate(consequence):

        conslist = str(cons).split('&')

        cindex = np.zeros((len(consdict)),dtype=np.int)

        for c in conslist:

            if c in consdict:

                cindex[consdict[c]]=1

            else:

                continue

        outmat[i] = cindex

    return outmat



def get_base_dict(ref_alt,mincount=25):

    '''

    return all the Reference/Alternate bases with count>=mincount

    '''

    base_count = Counter(ref_alt)

    basedict = {}

    i = 0

    for b,c in base_count.items():

    #     print(a,c)

        if c<mincount:

            continue

        basedict[b]= i

        i+=1

    return basedict

    



def base_encoder(basedat,basedict):

    '''

    encoder for Reference/Alternate bases

    '''

    basemat = np.zeros((len(basedat),len(basedict)),dtype=np.int)

    for i,b in enumerate(basedat):

        bindex = np.zeros((len(basedict)),dtype=np.int)

        if b in basedict:

            bindex[basedict[b]] = 1

    return basemat



def CLNDISDB_count(clndisdb):

    '''

    return count of evidence ids

    '''

    clncount = np.zeros(shape=(len(clndisdb)),dtype=np.int)

    for i, cln in enumerate(clndisdb):

        clncount[i]=len(re.sub(pattern=r'\.\|',repl='',string=str(cln)).split('|'))

    return clncount



def CLNDN_count(clndn):

    '''

    return clinvar disease name

    '''

    clndncount = np.zeros(shape=(len(clndn)),dtype=np.int)

    for i, cln in enumerate(clndn):

        clndncount[i]=len(re.sub(pattern=r'\.\|',repl='',string=str(cln)).split('|'))

    return clndncount



def get_clndn_dict(clndn,mincount=25):

    '''

    return clinvar disease name dictionary, where each disease name must occur mincount times

    '''

    clndn_count = Counter(list(chain.from_iterable([str(dn).split('|') for dn in clndn])))

    clndn_dict = {}

    i = 0

    for dn,cn in clndn_count.items():

        if cn < mincount:

            continue

        clndn_dict[dn]=i

        i+=1

    return clndn_dict



def clndn_encoder(clndn,clndn_dict):

    '''

    encoder for clinvar disease names

    '''

    clndnmat = np.zeros((len(clndn),len(clndn_dict)),dtype=np.int)

    for i,dns in enumerate(clndn):

        dndat = str(dns).split('|')

        dnindex = np.zeros((len(clndn_dict)),dtype=np.int)

        for dn in dndat:

            if dn in clndn_dict:

                dnindex[clndn_dict[dn]] = 1

    return clndnmat
format_dat = vardat[['AF_ESP','AF_EXAC','AF_TGP','LoFtool']]

format_dat.fillna(0, inplace=True)

format_dat.isnull().values.any()
cons_set = set(list(chain.from_iterable([str(v).split('&') for v in vardat['Consequence'].values])))

consdict = dict(zip(cons_set,range(len(cons_set))))

# # CLNDISDB

clndb_count = CLNDISDB_count(vardat['CLNDISDB'].values)

format_dat =  np.concatenate((format_dat,clndb_count.reshape(-1,1)),axis=1)

# # CLNDN

clndn_count = CLNDN_count(vardat['CLNDN'].values)

format_dat =  np.concatenate((format_dat,clndn_count.reshape(-1,1)),axis=1)

# # Reference allele length

reflen = np.array([len(r) for r in vardat['REF'].values],dtype=np.int)

format_dat =  np.concatenate((format_dat,reflen.reshape(-1, 1)),axis=1)

# # allele length

allelelen = np.array([len(r) for r in vardat['Allele'].values],dtype=np.int)

format_dat =  np.concatenate((format_dat,allelelen.reshape(-1, 1)),axis=1)

# chromosome

chr_encoder = LabelEncoder()

chr_onehot = OneHotEncoder(sparse=False)

chr_ind = chr_encoder.fit_transform(vardat['CHROM'])

format_dat =  np.concatenate((format_dat,chr_onehot.fit_transform(chr_ind.reshape(-1, 1))),axis=1)

# # origin

origin_encoder = OneHotEncoder(sparse=False)

format_dat =  np.concatenate((format_dat,origin_encoder.fit_transform(vardat[['ORIGIN']])),axis=1)

# # CLNVC

cldn_encoder = LabelEncoder()

cldn_onehot = OneHotEncoder(sparse=False)

clndn_ind = cldn_encoder.fit_transform(vardat['CLNVC'])

format_dat =  np.concatenate((format_dat,cldn_onehot.fit_transform(clndn_ind.reshape(-1, 1))),axis=1)

# # impact 

imp_encoder = LabelEncoder()

imp_onehot = OneHotEncoder(sparse=False)

imp_ind = imp_encoder.fit_transform(vardat['IMPACT'])

format_dat =  np.concatenate((format_dat,imp_onehot.fit_transform(imp_ind.reshape(-1, 1))),axis=1)

# # Exon encoder

exon_encode = np.ones((vardat.shape[0]),dtype=np.int)

exon_encode[vardat['EXON'].isna()]=0

format_dat =  np.concatenate((format_dat,exon_encode.reshape(-1, 1)),axis=1)

# # clinical disease name

clndn_dict = get_clndn_dict(vardat['CLNDN'],100)

clndn_encode = clndn_encoder(vardat['CLNDN'],clndn_dict)

format_dat =  np.concatenate((format_dat,clndn_encode),axis=1)

# # consequence 

cons_encode = consequence_encoder(vardat['Consequence'].values,consdict)

format_dat =  np.concatenate((format_dat,cons_encode),axis=1)

# # base data

base_dict = get_base_dict(vardat[['REF', 'ALT']].apply(lambda x: ':'.join(x), axis=1),50)

base_encode = base_encoder(list(vardat[['REF', 'ALT']].apply(lambda x: ':'.join(x), axis=1)),base_dict)

format_dat =  np.concatenate((format_dat,base_encode),axis=1)

print(format_dat.shape)
dat_Xtrain,tmp_x,dat_Ytrain,tmp_y = train_test_split(format_dat,vardat['CLASS'],test_size=0.3,random_state=42)

dat_Xval,dat_Xtest,dat_Yval,dat_Ytest= train_test_split(tmp_x,tmp_y,test_size=0.5,random_state=42)
print('Training data stats')

print(dat_Xtrain.shape)

print(Counter(dat_Ytrain))

print('\nValidation data stats')

print(dat_Xval.shape)

print(Counter(dat_Yval))

print('\nTest data stats')

print(dat_Xtest.shape)

print(Counter(dat_Ytest))
warnings.simplefilter(action='ignore',category=FutureWarning)

rf_base = RandomForestClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(rf_base,dat_Xtrain,dat_Ytrain,cv=10))

rf_base.fit(dat_Xtrain,dat_Ytrain)

y_pred = rf_base.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
bc_base = BaggingClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(bc_base,dat_Xtrain,dat_Ytrain,cv=10))

bc_base.fit(dat_Xtrain,dat_Ytrain)

y_pred = bc_base.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
et_base = ExtraTreesClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(et_base,dat_Xtrain,dat_Ytrain,cv=10))

et_base.fit(dat_Xtrain,dat_Ytrain)

y_pred = et_base.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
abc = AdaBoostClassifier(random_state=42,n_estimators=100)

print(cross_val_score(abc,dat_Xtrain,dat_Ytrain,cv=10))

abc.fit(dat_Xtrain,dat_Ytrain)

y_pred = abc.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
xg_base = XGBClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(xg_base,dat_Xtrain,dat_Ytrain,cv=10))

xg_base.fit(dat_Xtrain,dat_Ytrain)

y_pred = xg_base.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
def upsample(x,y):

    '''

    upsample least represented class

    y should be the labels,up sample least represented class in y

    '''

    x = np.array(x)

    ycount = Counter(y)

    ymin = min(list(ycount.values()))

    ymax = max(list(ycount.values()))

    yind = {}

    rex = None

    rey = list()

    for yi,c in ycount.items():

        if c==ymax:

            ind = np.where(y==yi)[0]

            if rex is None:

                rex = x[ind]

            else:

                rex = np.concatenate((rex,x[ind]),axis=0)

            rey.extend([yi]*ymax)

        elif c==ymin:

            ind = np.where(y==yi)[0]

            tmp_dat = resample(x[ind],replace=True,n_samples=ymax,random_state=42)

            if rex is None:

                rex = tmp_dat

            else:

                rex = np.concatenate((rex,tmp_dat),axis=0)

            rey.extend([yi]*ymax)

    return shuffle(rex,np.array(rey),random_state=42,replace=False)



def downsample(x,y):

    '''

    downsample over represented class

    y should be the labels,up sample least represented class in y

    '''

    x = np.array(x)

    ycount = Counter(y)

    ymin = min(list(ycount.values()))

    ymax = max(list(ycount.values()))

    yind = {}

    rex = None

    rey = list()

    for yi,c in ycount.items():

        if c==ymin:

            ind = np.where(y==yi)[0]

            if rex is None:

                rex = x[ind]

            else:

                rex = np.concatenate((rex,x[ind]),axis=0)

            rey.extend([yi]*ymin)

        elif c==ymax:

            ind = np.where(y==yi)[0]

            tmp_dat = resample(x[ind],replace=False,n_samples=ymin,random_state=42)

            if rex is None:

                rex = tmp_dat

            else:

                rex = np.concatenate((rex,tmp_dat),axis=0)

            rey.extend([yi]*ymin)

    return shuffle(rex,np.array(rey),random_state=42,replace=False)
dat_Xup, dat_Yup = upsample(dat_Xtrain,dat_Ytrain)

dat_Xdown, dat_Ydown = downsample(dat_Xtrain,dat_Ytrain)

print('\nUpsample stats')

print(dat_Xup.shape)

print(Counter(dat_Yup))

print('\nDownsample stats')

print(dat_Xdown.shape)

print(Counter(dat_Ydown))
rf_up = RandomForestClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(rf_up,dat_Xup,dat_Yup,cv=10))

rf_up.fit(dat_Xup,dat_Yup)

y_pred = rf_up.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
rf_down = RandomForestClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(rf_down,dat_Xdown,dat_Ydown,cv=10))

rf_down.fit(dat_Xdown,dat_Ydown)

y_pred = rf_down.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
bc_up = BaggingClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(bc_up,dat_Xup,dat_Yup,cv=10))

bc_up.fit(dat_Xup,dat_Yup)

y_pred = bc_up.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
bc_down = BaggingClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(bc_down,dat_Xdown,dat_Ydown,cv=10))

bc_down.fit(dat_Xdown,dat_Ydown)

y_pred = bc_down.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
et_up = ExtraTreesClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(et_up,dat_Xup,dat_Yup,cv=10))

et_up.fit(dat_Xup,dat_Yup)

y_pred = et_up.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
et_down = ExtraTreesClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(et_down,dat_Xdown,dat_Ydown,cv=10))

et_down.fit(dat_Xdown,dat_Ydown)

y_pred = et_down.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
abc_up = AdaBoostClassifier(random_state=42,n_estimators=100)

print(cross_val_score(abc_up,dat_Xup,dat_Yup,cv=10))

abc_up.fit(dat_Xup,dat_Yup)

y_pred = abc_up.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
abc_down = AdaBoostClassifier(random_state=42,n_estimators=100)

print(cross_val_score(abc_down,dat_Xdown,dat_Ydown,cv=10))

abc_down.fit(dat_Xdown,dat_Ydown)

y_pred = abc_down.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
xg_up = XGBClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(xg_up,dat_Xup,dat_Yup,cv=10))

xg_up.fit(dat_Xup,dat_Yup)

y_pred = xg_up.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
xg_down = XGBClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(xg_down,dat_Xdown,dat_Ydown,cv=10))

xg_down.fit(dat_Xdown,dat_Ydown)

y_pred = xg_down.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
from imblearn.over_sampling import SMOTENC

# Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTE-NC).
sm = SMOTENC(random_state=42,n_jobs=6,categorical_features=np.arange(8,dat_Xtrain.shape[1],1))

dat_Xsmote,dat_Ysmote = sm.fit_resample(dat_Xtrain,dat_Ytrain)

print('\nSMOTE stats')

print(dat_Xsmote.shape)

print(Counter(dat_Ysmote))
rf_smote = RandomForestClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(rf_smote,dat_Xsmote,dat_Ysmote,cv=10))

rf_smote.fit(dat_Xsmote,dat_Ysmote)

y_pred = rf_smote.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
bc_smote = BaggingClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(bc_smote,dat_Xsmote,dat_Ysmote,cv=10))

bc_smote.fit(dat_Xsmote,dat_Ysmote)

y_pred = bc_smote.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
et_smote = ExtraTreesClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(et_smote,dat_Xsmote,dat_Ysmote,cv=10))

et_smote.fit(dat_Xsmote,dat_Ysmote)

y_pred = et_smote.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
abc_smote = AdaBoostClassifier(random_state=42,n_estimators=100)

print(cross_val_score(abc_smote,dat_Xsmote,dat_Ysmote,cv=10))

abc_smote.fit(dat_Xsmote,dat_Ysmote)

y_pred = abc_smote.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
xg_smote = XGBClassifier(random_state=42,n_jobs=6,n_estimators=100)

print(cross_val_score(xg_smote,dat_Xsmote,dat_Ysmote,cv=10))

xg_smote.fit(dat_Xsmote,dat_Ysmote)

y_pred = xg_smote.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
from imblearn.ensemble import BalancedRandomForestClassifier 
brf = BalancedRandomForestClassifier(random_state = 42,n_estimators=100)

print(cross_val_score(brf,dat_Xtrain,dat_Ytrain,cv=10))

brf.fit(dat_Xtrain,dat_Ytrain)

y_pred =brf.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
from imblearn.ensemble import BalancedBaggingClassifier 
bbc = BalancedBaggingClassifier(random_state = 42,n_estimators=100)

print(cross_val_score(bbc,dat_Xtrain,dat_Ytrain,cv=10))

bbc.fit(dat_Xtrain,dat_Ytrain)

y_pred =bbc.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
from imblearn.ensemble import EasyEnsembleClassifier
eec = EasyEnsembleClassifier(random_state = 42,n_estimators=100)

#print(cross_val_score(eec,dat_Xtrain,dat_Ytrain,cv=10))

eec.fit(dat_Xtrain,dat_Ytrain)

y_pred =eec.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))
from imblearn.ensemble import RUSBoostClassifier
rbc = RUSBoostClassifier(random_state = 42,n_estimators=100)

#print(cross_val_score(eec,dat_Xtrain,dat_Ytrain,cv=10))

rbc.fit(dat_Xtrain,dat_Ytrain)

y_pred =rbc.predict(dat_Xval)

print('Confusion matrix\n',confusion_matrix(dat_Yval,y_pred))

print('Accuracy\n',accuracy_score(dat_Yval,y_pred))

print('Classificaton report\n',classification_report(dat_Yval,y_pred))