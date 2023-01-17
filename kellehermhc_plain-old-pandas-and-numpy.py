# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import scipy as sp

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import  KFold

from sklearn.feature_selection import SelectKBest, VarianceThreshold

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv("../input/test.csv")

train.head()
train_not_float = train.loc[:,train.dtypes != "float64"]

train_not_float.info()
mask = train.columns != "wheezy-copper-turtle-magic"

mask[0] = False

mask[-1] = False

#mask[75:] = False

mask_test = test.columns != "wheezy-copper-turtle-magic"

mask_test[0] = False

#mask_test[75:] = False
train20 = train[train['wheezy-copper-turtle-magic'] == 20]

fig, axs = plt.subplots(2,2)

axs[0,0].scatter(train20.iloc[:,25], train20.iloc[:,100], c = train20.target, alpha = 0.5)

axs[0,1].scatter(train20.iloc[:,12], train20.iloc[:,100], c = train20.target, alpha = 0.5)

axs[1,0].scatter(train20.iloc[:,25], train20.iloc[:,125], c = train20.target, alpha = 0.5)

axs[1,1].scatter(train20.iloc[:,12], train20.iloc[:,125], c = train20.target, alpha = 0.5)

plt.show
def MAPProba(x, mu0, mu1, sigma0, sigma1, det0, det1, sigma0_pinv, sigma1_pinv):

    x = np.asarray(x)

    # Recenter the x's

    x_prime0 = x - mu0

    x_prime1 = x - mu1

    # Check that it is a solvable equation, and solve if it is

    if np.allclose(x_prime0 @ sigma0 @ sigma0_pinv, x_prime0):

        y0 = 1

        x0 =  x_prime0 @ sigma0_pinv 

        Q0 = np.dot(x0, np.transpose(x_prime0))

        Q0 = float(Q0)

    else:

        y0 = 0

    if np.allclose(x_prime1 @ sigma1 @ sigma1_pinv, x_prime1):

        y1 = 1

        x1 =  x_prime1 @ sigma1_pinv 

        Q1 = np.dot(x1, np.transpose(x_prime1))

        Q1 = float(Q1)

    else:

        y1 = 0

    # output nulls if both the equations weren't solveable

    if y1 == 0 & y0 == 0:

        return np.asarray([np.nan, np.nan]).reshape((2))

    # output probabilities if both the equations were

    elif y1 == 1 & y0 == 1:

        y1 = 1/(1+np.exp(-0.5*(Q0-Q1))*det1/det0)

        y0 = 1 - y1

        return np.asarray([y0, y1]).reshape((2))

    else:

        return np.asarray([y0, y1]).reshape((2))

    return
class NormalMAP:

    def __init__(self):

        self.mu0 = 0

        self.mu1 = 0

       # 

        self.sigma0 = np.empty((0,0))

        self.sigma1 = np.empty((0,0))

        self.shape = 0

        self.samples = 0

        #

        return

    

    def fit(self, X_tr, y_tr):

        self.samples, self.shape = X_tr.shape

        masker = np.asarray(y_tr == 0).reshape(self.samples)

        X_tr0 = X_tr[masker]

        X_tr1 = X_tr[~masker]



        self.mu0 = np.asarray(X_tr0.mean()).reshape(1,self.shape)

        self.mu1 = np.asarray(X_tr1.mean()).reshape(1,self.shape)



        self.sigma0 = np.asarray(np.cov(X_tr0, rowvar = False))

        self.sigma1 = np.asarray(np.cov(X_tr1, rowvar = False))



        self.sigma0_pinv = sp.linalg.pinv(self.sigma0)

        self.sigma1_pinv = sp.linalg.pinv(self.sigma1)



        self.w0, self.V0 = np.linalg.eig(self.sigma0)

        self.w1, self.V1 = np.linalg.eig(self.sigma1)



        self.det0 = np.real(np.sqrt(np.prod(self.w0[~np.isclose(self.w0, 0)])))

        self.det1 = np.real(np.sqrt(np.prod(self.w1[~np.isclose(self.w1, 0)])))

        return



    def predict_proba(self, X_te):

        x = np.asarray(X_te)

        

        y = np.apply_along_axis(lambda x: MAPProba(x, self.mu0, self.mu1, self.sigma0, self.sigma1, self.det0, self.det1, self.sigma0_pinv, self.sigma1_pinv),

                               1,

                               x)

        return y
# initialize dataframes

cv_scores = pd.DataFrame({'train_score' : [], 

                          'test_score' : [], 

                          'fold' : []})

submission = pd.DataFrame({"id" : [],

                           "target" : []})

#

#loop over different values of wheezy-copper-turtle-magic

for wheezy_value in range(512):

    #

    # Subset the data by conditioning on wheezy-copper-turtle-magic

    train_temp = train[train["wheezy-copper-turtle-magic"]==wheezy_value]

    #

    # Break into train and target sets

    X_mask = train_temp.iloc[:,mask]

    y_temp = train_temp.iloc[:,-1]

    #

    # Initialize k fold split

    kfold = KFold(n_splits = 32, shuffle = True, random_state = 42)

    #

    # Initialize test set

    test_temp = test[test["wheezy-copper-turtle-magic"] == wheezy_value]

    predictions =np.zeros((test_temp.shape[0],2))

    #

    # go a layer deeper

    for fold_, (trn_idx, val_idx) in enumerate(kfold.split(X_mask,y_temp)):

        #

        # Create Pipeline

        pp_pl = Pipeline([#("scaler", StandardScaler()),

                          ("selection", VarianceThreshold(threshold = 1.3))])

        #

        # Created Classifier

        nmap = NormalMAP()

        

        X_train, X_test = X_mask.iloc[trn_idx,:], X_mask.iloc[val_idx,:]

        y_train, y_test = y_temp.iloc[trn_idx], y_temp.iloc[val_idx]

    

        X_train_trans = pp_pl.fit_transform(X_train, y_train)

        X_test_trans = pp_pl.transform(X_test)

        X_train_trans = pd.DataFrame(X_train_trans)

        X_test_trans = pd.DataFrame(X_test_trans)

        #

        # log CV scores

        nmap.fit(X_train_trans, y_train)

        y_pred_train = nmap.predict_proba(X_train_trans)

        y_pred_test = nmap.predict_proba(X_test_trans) 

        y_pred_train = (y_pred_train[:,1]>0.5)

        y_pred_test = (y_pred_test[:,1]>0.5)

        cv_train = np.mean(y_pred_train == y_train)

        cv_test = np.mean(y_pred_test == y_test)

        cv_scores.loc["wheezy_value_%s_%d" % (wheezy_value, fold_),

                      ['train_score', 'fold']] = [cv_train, fold_]

        cv_scores.loc["wheezy_value_%s_%d" % (wheezy_value, fold_), 

                      ['test_score', 'fold']] = [cv_test, fold_]

        #

        # make predictions



        test_trans = pp_pl.transform(test_temp.iloc[:,mask_test])

        index_temp = test_temp.iloc[:,0]

        predictions += nmap.predict_proba(test_trans)/kfold.n_splits

        #

    # format submission

    submission_temp = pd.DataFrame()

    submission_temp["id"] = index_temp

    submission_temp['target'] = predictions[:,1]

    submission = pd.concat([submission,submission_temp])

    #
print(cv_scores.head(10))

print(cv_scores.mean())
submission.to_csv('submission.csv', index = False)

plt.hist(submission.target)

plt.show