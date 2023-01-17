import numpy as np, pandas as pd, os

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import roc_auc_score

from tqdm import tqdm, tqdm_notebook

from sklearn.covariance import EmpiricalCovariance

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import sympy 

from sklearn import svm, neighbors, linear_model, neural_network

from xgboost import XGBClassifier

from sklearn.covariance import *

from sklearn.utils.validation import check_random_state

from sklearn.mixture import *

from sklearn.cluster import *
%time



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')



train.head()
import warnings

warnings.filterwarnings('ignore')
def dist(array, centre):

    x=float(0)

    for i in range(len(array)):

        x += ((array[i]-centre[i])*(array[i]-centre[i]))

    return x



def get_c(data):

    data.drop('labels', axis=1, inplace=True)

    centre = [1]*data.shape[1]

    for i in range(data.shape[1]):

        if data[i].mean() < 0:

            centre[i] = -1

    return (centre)



def my_min(a, b, c):

#     return 1

    if a<b:

        if a<c: return a

        return c

    if b<c: return b

    return c

def classify(data, val, test, y):

    data = pd.DataFrame(train3)

    data['target'] = y

    zero = data[data['target'] == 1]

    one = data[data['target'] == 0]

    zero.drop('target', axis=1, inplace=True)

    one.drop('target', axis=1, inplace=True)

    

    clf = KMeans(n_clusters=3)

    labels = clf.fit_predict(zero)

    zero['labels'] = labels

    zero_0 = (zero[zero['labels'] == 0])

    zero_1 = (zero[zero['labels'] == 1])

    zero_2 = (zero[zero['labels'] == 2])

    

    clf = KMeans(n_clusters=3)

    labels = clf.fit_predict(one)

    one['labels'] = labels

    one_0 = (one[one['labels'] == 0])

    one_1 = (one[one['labels'] == 1])

    one_2 = (one[one['labels'] == 2])

    

    c_z_0 = get_c(zero_0)

    c_z_1 = get_c(zero_1)

    c_z_2 = get_c(zero_2)

    

    c_o_0 = get_c(one_0)

    c_o_1 = get_c(one_1)

    c_o_2 = get_c(one_2)

    

    pred_val = [0]*val.shape[0]

    for i in range(val.shape[0]):

        array = val.loc[i]

        dist0_0 = dist(array, c_z_0)

        dist0_1 = dist(array, c_z_1)

        dist0_2 = dist(array, c_z_2)

        

        dist1_0 = dist(array, c_o_0)

        dist1_1 = dist(array, c_o_1)

        dist1_2 = dist(array, c_o_2)



        

        aggr = (dist0_0+dist0_1+dist0_2  +dist1_0+dist1_1+dist1_2)/3

#         dist1 = my_min(dist1_0, dist1_1, dist1_2, dist1_3)

#         dist0 = my_min(dist0_0, dist0_1, dist0_2, dist0_3)

        

        dist1 = (dist1_0+dist1_1+dist1_2) / 3

        dist0 = (dist0_0+dist0_1+dist0_2) / 3

        pred_val[i] = 1-1/np.exp(dist1/aggr)

        

    pred_test = [0]*test.shape[0]

    for i in range(test.shape[0]):

        array = test.loc[i]

        dist0_0 = dist(array, c_z_0)

        dist0_1 = dist(array, c_z_1)

        dist0_2 = dist(array, c_z_2)

        

        dist1_0 = dist(array, c_o_0)

        dist1_1 = dist(array, c_o_1)

        dist1_2 = dist(array, c_o_2)



        

        aggr = (dist0_0+dist0_1+dist0_2  +dist1_0+dist1_1+dist1_2)/4

#         dist1 = my_min(dist1_0, dist1_1, dist1_2, dist1_3)

#         dist0 = my_min(dist0_0, dist0_1, dist0_2, dist0_3)

        

        dist1 = (dist1_0+dist1_1+dist1_2) / 3

        dist0 = (dist0_0+dist0_1+dist0_2) / 3

        

        pred_test[i] = 1-1/np.exp(dist1/aggr)



    return np.array(pred_val), np.array(pred_test)

oof = np.zeros(len(train))

preds = np.zeros(len(test))

cols = [c for c in train.columns if c not in ['id', 'target']]





for k in tqdm_notebook(range(512)):



    train2 = train[train['wheezy-copper-turtle-magic']==k]

    test2 = test[test['wheezy-copper-turtle-magic']==k]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])



    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]

    skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)



    for i, (train_index, test_index) in enumerate(skf.split(train3, train2['target'])):





        oof[idx1[test_index]], test_pred = classify(pd.DataFrame(train3[train_index,:]), pd.DataFrame(train3[test_index,:]), pd.DataFrame(test3), train2.loc[train_index]['target'])

        

        preds[idx2] += test_pred / skf.n_splits



        print(roc_auc_score(train2.loc[test_index]['target'], oof[idx1[test_index]]))

    

    if k==5: break



auc = roc_auc_score(train['target'], oof)

print(f'AUC: {auc:.5}')
