from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score,classification_report

import seaborn as sns

import pandas as pd

import numpy as np 

submission = pd.read_csv("/kaggle/input/cat-in-the-dat/sample_submission.csv")

train = pd.read_csv("/kaggle/input/cat-in-the-dat/train.csv")

test = pd.read_csv("/kaggle/input/cat-in-the-dat/test.csv")



labels = train['target'].values

train_id = train.pop("id")

test_id = test.pop("id")
data = pd.concat([train.drop('target',axis=1), test])

totaal=train.append(test)
train.shape,test.shape,data.shape,totaal.shape
data.head()
# One Hot Encode target mean()

cols=[ci for ci in train.columns if ci not in ['id','index','target']]

coltype=train.dtypes

for ci in cols:

    

    if (coltype[ci]=="object"):

        #bin_3

        #l_enc = LabelEncoder()

        codes=totaal[[ci,'target']].groupby(ci).mean().sort_values("target")

        #print(codes)

        codesdict=codes.target.to_dict()



        #print(codesdict)

        #l_enc.fit(list(codes.index))

        totaal[ci]=totaal[ci].map(codesdict) #l_enc.transform(totaal[ci])

    #print('labelized',ci)

#prevent error in test, because nom_8 can have empties

totaal['id']=train_id.append(test_id)

totaal=totaal.fillna(0)
from sklearn.linear_model import LogisticRegressionCV,SGDClassifier,RidgeClassifierCV,LogisticRegression

lr = LogisticRegression( solver="lbfgs",max_iter=500,n_jobs=4)



lengte=100000



lr.fit(totaal[:lengte].drop('target',axis=1), labels[:lengte])



lr_pred = lr.predict(totaal[:lengte].drop('target',axis=1))

score = classification_report(labels[:lengte], lr_pred)



print("score: ", score)

import random

lr_pred = [ (random.random()>0.7)*1 for x in range(lengte) ]

score = classification_report(labels[:lengte], lr_pred)

print("Randum number generated versus label:")

print( score)

from sklearn.metrics import roc_auc_score

lr_pred = [ (random.random()>0.7)*1 for x in range(lengte) ]

score = roc_auc_score(labels[:lengte], lr_pred)

print("Randum number generated versus label:")

print('auc', score)

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import r2_score,mean_squared_error







def pcr(X,y,pc):

    ''' Principal Component Regression in Python'''

    ''' Step 1: PCA on input data'''

 

    # Define the PCA object

    pca = PCA(n_components=pc)

 

    ## Preprocessing (1): first derivative

    #d1X = savgol_filter(X, 25, polyorder = 5, deriv=1)

 

    # Preprocess (2) Standardize features by removing the mean and scaling to unit variance

    Xstd = StandardScaler().fit_transform(X[:,:])

 

    # Run PCA producing the reduced variable Xred and select the first pc components

    Xreg = pca.inverse_transform( pca.fit_transform(Xstd)[:,:pc] )

 

 

    ''' Step 2: regression on selected principal components'''

 

    # Create linear regression object

    regr = LogisticRegression(solver="lbfgs",max_iter=500,n_jobs=4)

    #regr = LogisticRegressionCV()

    # Fit

    regr.fit(Xreg, y)

 

    # Calibration

    y_c = regr.predict(Xreg)

 

    # Cross-validation

    y_cv = cross_val_predict(regr, Xreg, y, cv=10)

 

    # Calculate scores for calibration and cross-validation

    score_c = r2_score(y, y_c)

    score_cv = r2_score(y, y_cv)

 

    # Calculate mean square error for calibration and cross validation

    mse_c = mean_squared_error(y, y_c)

    mse_cv = mean_squared_error(y, y_cv)

    

    score = classification_report(y, y_cv)

    print("score: ", score)



    return(y_cv, score_c, score_cv, mse_c, mse_cv)



results=pd.DataFrame([])

for xi in range(1,30,10):

    yw,sc1,sc2,mse1,mse2= pcr(totaal[:lengte].values, labels[:lengte],xi) 

    results=results.append([sc1,sc2,mse1,mse2])

u,s,v=np.linalg.svd(totaal.drop('id',axis=1)[:lengte].values,full_matrices=False)

u.shape,s.shape,v.shape
u_,s_,v_=np.linalg.svd(totaal[:lengte].drop(['target','id'],axis=1).values,full_matrices=False)

u_.shape,s_.shape,v_.shape
pd.DataFrame( np.round( np.dot( u[:,:23]*s[:23],v_[:23,:23])-np.dot(u*s,v)[:,:23],1) )