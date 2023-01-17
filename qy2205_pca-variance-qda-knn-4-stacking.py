import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.svm import NuSVC

from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from tqdm import tqdm_notebook

import warnings

import multiprocessing

from scipy.optimize import minimize  

warnings.filterwarnings('ignore')
train1 = pd.read_csv('../input/train.csv')

test1 = pd.read_csv('../input/test.csv')

cols1 = [c for c in train1.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
def instant_model(train, test, cols = cols1, clf = QuadraticDiscriminantAnalysis(0.5), selection = "PCA"):

    oof = np.zeros(len(train))

    preds = np.zeros(len(test))



    for i in tqdm_notebook(range(512)):



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

        

        train3['target'] = train2['target']



        # Kfold

        skf = StratifiedKFold(n_splits=11, random_state=42)

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



    auc = roc_auc_score(train['target'], oof)

    print(f'AUC: {auc:.5}')

    

    return oof, preds
def get_newtrain(train, test, preds, oof):

    # get useful train set from train and test data

    # get useful test 

    test['target'] = preds

    test.loc[test['target'] > 0.985, 'target'] = 1

    test.loc[test['target'] < 0.015, 'target'] = 0

    usefull_test = test[(test['target'] == 1) | (test['target'] == 0)]



    # get useful train 

    new_train = pd.concat([train, usefull_test]).reset_index(drop=True)

    new_train.loc[oof > 0.985, 'target'] = 1

    new_train.loc[oof < 0.015, 'target'] = 0

    return new_train
oof_temp, preds_temp = instant_model(train1, test1, selection = 'variance')

newtrain1 = get_newtrain(train1, test1, preds_temp, oof_temp)
oof_qda_var, preds_qda_var = instant_model(newtrain1, test1, selection = 'variance')

# oof_nusvm, preds_nusvm = instant_model(newtrain1, test1, clf = NuSVC(probability = True, kernel = 'poly'))

oof_knn_var, preds_knn_var = instant_model(newtrain1, test1, \

                                   clf = KNeighborsClassifier(n_neighbors = 7, p = 2, weights = 'distance'),\

                                   selection = 'variance')
oof_qda_pca, preds_qda_pca = instant_model(newtrain1, test1)

# oof_nusvm, preds_nusvm = instant_model(newtrain1, test1, clf = NuSVC(probability = True, kernel = 'poly'))

oof_knn_pca, preds_knn_pca = instant_model(newtrain1, test1, \

                                           clf = KNeighborsClassifier(n_neighbors = 7, p = 2, weights = 'distance'))
logit = LogisticRegression()

newX_train_stack = pd.DataFrame({"QDA_var": oof_qda_var, "QDA_pca": oof_qda_pca, \

                                 "KNN_var": oof_knn_var, "KNN_pca": oof_knn_pca})

newX_test_stack = pd.DataFrame({"QDA_var": preds_qda_var, "QDA_pca": preds_qda_pca, \

                                "KNN_var": preds_knn_var, "KNN_pca": preds_knn_pca})

newy_stack = newtrain1['target']

logit.fit(newX_train_stack, newy_stack)

pred_stack_train = logit.predict_proba(newX_train_stack)[:,1]

pred_stack_test = logit.predict_proba(newX_test_stack)[:,1]

print("ROC_AUC: {0}".format(roc_auc_score(newy_stack, pred_stack_train)))

stack_result = logit.predict_proba(newX_test_stack)[:,1]
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = stack_result

sub.to_csv('submission_4stack.csv',index=False)