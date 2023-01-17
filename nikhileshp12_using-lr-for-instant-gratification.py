# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np, pandas as pd

from tqdm import tqdm

from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from matplotlib import pyplot as plt

import sys



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

sys.stdout.write('Libraries Successfully Loaded\n')

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
sys.stdout.write('Loading Train\n')

train = pd.read_csv('../input/train.csv')

sys.stdout.write('Loading Test\n')

test = pd.read_csv('../input/test.csv')

sys.stdout.write('Finished Loading Datasets\n')
cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
def instant_model(train, test, cols = cols, clf = KNeighborsClassifier(leaf_size=2000, n_neighbors = 11, p = 2, weights = 'distance'), selection = "PCA"):

    oof = np.zeros(len(train))

    preds = np.zeros(len(test))



    for i in tqdm(range(512)):



        train2 = train[train['wheezy-copper-turtle-magic'] == i]

        test2 = test[test['wheezy-copper-turtle-magic'] == i]

        idx1 = train2.index

        idx2 = test2.index



        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

        

        if selection == "variance":

            # StandardScaler & Variance selection

            data2 = StandardScaler().fit_transform(VarianceThreshold(threshold=2).fit_transform(data[cols]))

            train3 = pd.DataFrame(data2[:train2.shape[0]], index = idx1)

            test3 = pd.DataFrame(data2[train2.shape[0]:], index = idx2)

            

        elif selection == "PCA":

            # PCA

            pca = PCA(n_components = 40, random_state= 1234)

            pca.fit(data[:train2.shape[0]])

            train3 = pd.DataFrame(pca.transform(data[:train2.shape[0]]), index = idx1)

            test3 = pd.DataFrame(pca.transform(data[train2.shape[0]:]), index = idx2)

        

        elif selection == "poly":

            poly = PolynomialFeatures(degree=2)

            data2 = StandardScaler().fit_transform(poly.fit_transform(VarianceThreshold(threshold=2).fit_transform(data[cols])))

            train3 = pd.DataFrame(data2[:train2.shape[0]],index = idx1); 

            test3 = pd.DataFrame(data2[train2.shape[0]:],index = idx2);

            

            

        train3['target'] = train2['target']



        # Kfold

        skf = StratifiedKFold(n_splits=15, random_state=42)

        for train_index, test_index in skf.split(train3, train3['target']):

            # clf

            clf = clf

            X_train = train3.iloc[train_index, :].drop(["target"], axis = 1)

            X_test = train3.iloc[test_index, :].drop(["target"], axis = 1)

            y_train = train3.iloc[train_index, :]['target']

            y_test = train3.iloc[test_index, :]['target']

            clf.fit(X_train, y_train)



            # output

            train_prob = clf.predict_proba(X_train)[:,1]

            test_prob = clf.predict_proba(X_test)[:,1]

            oof[idx1[test_index]] = test_prob



            # bagging

            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

            # print("Chunk {0} Fold {1}".format(i, roc_auc_score(y_test, test_prob)))

        

        if i%64==0:

            print(i, 'AUC: ', round(roc_auc_score(train['target'][idx1], oof[idx1]), 5))

    auc = roc_auc_score(train['target'], oof)

    print(f'AUC: {auc:.5}')

    

    return oof, preds, auc
def get_newtrain(train, test, preds, oof):

    # get useful train set from train and test data

    # get useful test 

    test['target'] = preds

    test.loc[test['target'] > 0.985, 'target'] = 1

    test.loc[test['target'] < 0.015, 'target'] = 0

    usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]

    print("No. of test records added: {}".format(usefull_test.shape[0]))

    # get useful train 

    new_train = pd.concat([train, usefull_test]).reset_index(drop=True)

    new_train.loc[oof > 0.985, 'target'] = 1

    new_train.loc[oof < 0.015, 'target'] = 0

    return new_train
oof, preds, auc = instant_model(train, test, clf = LogisticRegression(solver='saga',penalty='l2',C=0.01,tol=0.001), selection="poly")
print('LR auc: ', round(roc_auc_score(train['target'], oof),5))

plt.hist(preds,bins=200)
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)