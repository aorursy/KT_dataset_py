# source from https://www.kaggle.com/infinitewing/for-fun

import pandas as pd; import numpy as np

kernel_df = pd.read_csv('../input/for-fun/kernel_df.csv')

kernel_df['final_score'] = (kernel_df['kernel_plb_score'] 

                            + kernel_df['kernel_valid_oof_score']

                            + kernel_df['kernel_valid_preds_score']) / 3

kernel_df['plb_rank'] = kernel_df['kernel_plb_score'].rank(ascending=False)

kernel_df['valid_oof_rank'] = kernel_df['kernel_valid_oof_score'].rank(ascending=False)

kernel_df['valid_test_rank'] = kernel_df['kernel_valid_preds_score'].rank(ascending=False)

kernel_df['final_rank'] = kernel_df['final_score'].rank(ascending=False)

print(kernel_df)

kernel_df.to_csv('kernel_df.csv',index=False)
import pandas as pd

import numpy as np

from sklearn.feature_selection import VarianceThreshold

import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv('../input/instant-gratification/train.csv')

test = pd.read_csv('../input/instant-gratification/test.csv')

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

magic_turtles = len(train['wheezy-copper-turtle-magic'].unique())

print('wheezy-copper-turtle-magic has {} unique values.'.format(magic_turtles))



useful_cols_count = np.zeros(magic_turtles)

useful_cols_count = useful_cols_count.astype('int32')

for i in range(magic_turtles):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    train_data2 = VarianceThreshold(threshold=2).fit_transform(train2[cols])

    train2 = train2[cols].values

    

    useful_cols_count[i] = train_data2.shape[1]

    '''

    # check if the useful cols in test and train are the same

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    test_data2 = VarianceThreshold(threshold=2).fit_transform(test2[cols])

    test2 = test2[cols].values

    

    print(train2.shape[1])

    print(train_data2.shape[1])

    for _i in range(train_data2.shape[1]):

        for _j in range(train2.shape[1]):

            if(np.sum(train_data2[:,_i] - train2[:,_j]) == 0):

                if(np.sum(test_data2[:,_i] - test2[:,_j]) == 0):

                    print('train_data2[:,{}] == train2[:,{}]'.format(_i, _j))

                break

    '''
df = pd.DataFrame()

df['wheezy-copper-turtle-magic'] = np.array([i for i in range(magic_turtles)])

df['useful_cols_count'] = useful_cols_count

#print(df)

print(df['useful_cols_count'].describe())

    
df['total_train_rows'] = [0 for _ in range(magic_turtles)]

df['total_test_rows'] = [0 for _ in range(magic_turtles)]

for i in range(magic_turtles):

    total_train_rows = train[train['wheezy-copper-turtle-magic'] == i]['wheezy-copper-turtle-magic'].count()

    total_test_rows = test[test['wheezy-copper-turtle-magic'] == i]['wheezy-copper-turtle-magic'].count()

    df.loc[df['wheezy-copper-turtle-magic'] == i, ['total_train_rows']] = total_train_rows

    df.loc[df['wheezy-copper-turtle-magic'] == i, ['total_test_rows']] = total_test_rows

print(df.head(10))

print(df['total_train_rows'].sum() == train.shape[0])

print(df['total_test_rows'].sum() == test.shape[0])

print(df[['total_train_rows','total_test_rows']].sum())



from sklearn.datasets import make_classification



train_for_valid = False

test_for_valid = False

for i, row in df.iterrows():

    if(i%50 == 0): print(i)

    np.random.seed(520999+i)

    useful_cols_count = row['useful_cols_count']

    total_train_rows = row['total_train_rows']

    total_test_rows = row['total_test_rows']

    X, y = make_classification(n_samples=total_train_rows+total_test_rows, n_features=255, \

                               n_informative=useful_cols_count, n_redundant=0, \

                               n_clusters_per_class=3, \

                               random_state=3228+i, shuffle=True,  \

                               flip_y=0.05)

    Xy = np.zeros((total_train_rows+total_test_rows, 257))

    Xy[:,:-2] = X

    Xy[:,-2] = i # represent 'wheezy-copper-turtle-magic'

    Xy[:,-1] = y

    if(train_for_valid is False):

        train_for_valid = Xy[:total_train_rows,:]

        test_for_valid = Xy[total_train_rows:,:]

    else:

        train_for_valid = np.concatenate((train_for_valid, Xy[:total_train_rows,:]))

        test_for_valid = np.concatenate((test_for_valid, Xy[total_train_rows:,:]))

print(train_for_valid.shape)

print(test_for_valid.shape)
train_valid_df = pd.DataFrame(train_for_valid, columns=cols+['wheezy-copper-turtle-magic', 'target'])

test_valid_df = pd.DataFrame(test_for_valid, columns=cols+['wheezy-copper-turtle-magic', 'target'])



train_valid_df['wheezy-copper-turtle-magic'] = train_valid_df['wheezy-copper-turtle-magic'].astype('int32')

train_valid_df['target'] = train_valid_df['target'].astype('int32')

test_valid_df['wheezy-copper-turtle-magic'] = test_valid_df['wheezy-copper-turtle-magic'].astype('int32')

test_valid_df['target'] = test_valid_df['target'].astype('int32')

print(train_valid_df.head(2))

print(test_valid_df.head(2))
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



def qda(train, test):

    oof = np.zeros(len(train))

    preds = np.zeros(len(test))

    aucs = np.zeros(512)

    for i in range(512):

        #if(i%50 == 0): print(i)

        train2 = train[train['wheezy-copper-turtle-magic']==i]

        test2 = test[test['wheezy-copper-turtle-magic']==i]

        idx1 = train2.index; idx2 = test2.index

        train2.reset_index(drop=True,inplace=True)



        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

        data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])



        train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



        skf = StratifiedKFold(n_splits=11, random_state=42)

        for train_index, test_index in skf.split(train2, train2['target']):



            clf = QuadraticDiscriminantAnalysis(0.1)

            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

            oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        aucs[i] = roc_auc_score(train['target'][idx1], oof[idx1])

    auc = roc_auc_score(train['target'], oof)

    print('QDA AUC: {}'.format(round(auc,5)))

    aucs_df = pd.DataFrame(aucs,columns = ['auc'])

    print(aucs_df.describe())

    return oof, preds

def qda_pseudo(train, test):

    oof = np.zeros(len(train))

    preds = np.zeros(len(test))

    aucs = np.zeros(512)



    # BUILD 512 SEPARATE MODELS

    for k in range(512):

        # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

        train2 = train[train['wheezy-copper-turtle-magic']==k] 

        train2p = train2.copy(); idx1 = train2.index 

        test2 = test[test['wheezy-copper-turtle-magic']==k]



        # ADD PSEUDO LABELED DATA

        test2p = test2[ (test2['target']<=0.01) | (test2['target']>=0.99) ].copy()

        test2p.loc[ test2p['target']>=0.5, 'target' ] = 1

        test2p.loc[ test2p['target']<0.5, 'target' ] = 0 

        train2p = pd.concat([train2p,test2p],axis=0)

        train2p.reset_index(drop=True,inplace=True)



        # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

        sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])     

        train3p = sel.transform(train2p[cols])

        train3 = sel.transform(train2[cols])

        test3 = sel.transform(test2[cols])



        # STRATIFIED K FOLD

        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

        for train_index, test_index in skf.split(train3p, train2p['target']):

            test_index3 = test_index[ test_index<len(train3) ] # ignore pseudo in oof



            # MODEL AND PREDICT WITH QDA

            clf = QuadraticDiscriminantAnalysis(reg_param=0.5)

            clf.fit(train3p[train_index,:],train2p.loc[train_index]['target'])

            oof[idx1[test_index3]] = clf.predict_proba(train3[test_index3,:])[:,1]

            preds[test2.index] += clf.predict_proba(test3)[:,1] / skf.n_splits

        aucs[k] = roc_auc_score(train['target'][idx1], oof[idx1])

        #if k%64==0: print(k)



    # PRINT CV AUC

    auc = roc_auc_score(train['target'],oof)

    print('Pseudo Labeled QDA scores CV =',round(auc,5))

    aucs_df = pd.DataFrame(aucs,columns = ['auc'])

    print(aucs_df.describe())

    return oof, preds

oof, preds = qda(train, test)

test['target'] = preds

oof, preds = qda_pseudo(train, test)
oof, preds = qda(train_valid_df, test_valid_df)

original_target = test_valid_df['target'].values.copy()

test_valid_df['target'] = preds

oof, preds = qda_pseudo(train_valid_df, test_valid_df)

test_valid_df['target'] = original_target
# The original LB score is 0.9659

auc = roc_auc_score(test_valid_df['target'], preds)

print('AUC: {}'.format(round(auc,5)))


from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import LogisticRegression



def log(train, test):

    oof = np.zeros(len(train))

    preds = np.zeros(len(test))

    aucs = np.zeros(512)

    for i in range(512):

        train2 = train[train['wheezy-copper-turtle-magic']==i]

        test2 = test[test['wheezy-copper-turtle-magic']==i]

        idx1 = train2.index; idx2 = test2.index

        train2.reset_index(drop=True,inplace=True)



        # Adding quadratic polynomial features can help linear model such as Logistic Regression learn better

        poly = PolynomialFeatures(degree=2)

        sc = StandardScaler()

        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

        data2 = poly.fit_transform(sc.fit_transform(VarianceThreshold(threshold=2).fit_transform(data[cols])))

        train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



        # STRATIFIED K FOLD

        skf = StratifiedKFold(n_splits=11, random_state=42)

        for train_index, test_index in skf.split(train2, train2['target']):



            clf = LogisticRegression(solver='liblinear',penalty='l2',C=0.001,tol=0.0001,random_state=0,max_iter=1000,n_jobs=-1)

            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

            oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        aucs[i] = roc_auc_score(train['target'][idx1], oof[idx1])

    auc = roc_auc_score(train['target'], oof)

    print('LOG AUC: {}'.format(round(auc,5)))

    aucs_df = pd.DataFrame(aucs,columns = ['auc'])

    print(aucs_df.describe())

    return oof, preds
oof, preds = log(train, test)

oof, preds = log(train_valid_df, test_valid_df)



auc = roc_auc_score(test_valid_df['target'], preds)

print('AUC: {}'.format(round(auc,5)))
from sklearn import svm, neighbors, linear_model, neural_network

from sklearn.svm import NuSVC

from sklearn.decomposition import PCA



def nusvc(train, test):

    oof = np.zeros(len(train))

    preds = np.zeros(len(test))

    aucs = np.zeros(512)

    cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



    for i in range(512):

        train2 = train[train['wheezy-copper-turtle-magic']==i]

        test2 = test[test['wheezy-copper-turtle-magic']==i]

        idx1 = train2.index; idx2 = test2.index

        train2.reset_index(drop=True,inplace=True)



        data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

        data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=4).fit_transform(data[cols]))

        train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



        # STRATIFIED K FOLD (Using splits=25 scores 0.002 better but is slower)

        skf = StratifiedKFold(n_splits=5, random_state=42)

        for train_index, test_index in skf.split(train2, train2['target']):



            clf = NuSVC(probability=True, kernel='poly', degree=4, gamma='auto', random_state=4, nu=0.59, coef0=0.053)

            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

            oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

        aucs[i] = roc_auc_score(train['target'][idx1], oof[idx1])

    auc = roc_auc_score(train['target'], oof)

    print('NUSVC AUC: {}'.format(round(auc,5)))

    aucs_df = pd.DataFrame(aucs,columns = ['auc'])

    print(aucs_df.describe())

    return oof, preds
oof, preds = nusvc(train, test)

oof, preds = nusvc(train_valid_df, test_valid_df)



auc = roc_auc_score(test_valid_df['target'], preds)

print('AUC: {}'.format(round(auc,5)))
from sklearn.covariance import EmpiricalCovariance

from sklearn.covariance import GraphicalLasso

from sklearn.mixture import GaussianMixture

def get_mean_cov(x,y):

    model = GraphicalLasso()

    ones = (y==1).astype(bool)

    x2 = x[ones]

    model.fit(x2)

    p1 = model.precision_

    m1 = model.location_

    

    onesb = (y==0).astype(bool)

    x2b = x[onesb]

    model.fit(x2b)

    p2 = model.precision_

    m2 = model.location_

    

    ms = np.stack([m1,m2])

    ps = np.stack([p1,p2])

    return ms,ps



def gmm(train, test):

    oof = np.zeros(len(train))

    preds = np.zeros(len(test))

    aucs = np.zeros(512)

    cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

    # BUILD 512 SEPARATE MODELS

    for i in (range(512)):

        # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

        train2 = train[train['wheezy-copper-turtle-magic']==i]

        test2 = test[test['wheezy-copper-turtle-magic']==i]

        idx1 = train2.index; idx2 = test2.index

        train2.reset_index(drop=True,inplace=True)



        # FEATURE SELECTION (USE APPROX 40 OF 255 FEATURES)

        sel = VarianceThreshold(threshold=1.5).fit(train2[cols])

        train3 = sel.transform(train2[cols])

        test3 = sel.transform(test2[cols])



        # STRATIFIED K-FOLD

        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)

        for train_index, test_index in skf.split(train3, train2['target']):



            # MODEL AND PREDICT WITH QDA

            ms, ps = get_mean_cov(train3[train_index,:],train2.loc[train_index]['target'].values)

            gm = GaussianMixture(n_components=2, init_params='random', covariance_type='full', tol=0.001,reg_covar=0.001, max_iter=100, n_init=1,means_init=ms, precisions_init=ps)

            gm.fit(np.concatenate([train3[train_index,:],test3],axis = 0))

            oof[idx1[test_index]] = gm.predict_proba(train3[test_index,:])[:,0]

            preds[idx2] += gm.predict_proba(test3)[:,0] / skf.n_splits

        aucs[i] = roc_auc_score(train['target'][idx1], oof[idx1])

    auc = roc_auc_score(train['target'], oof)

    print('GMM AUC: {}'.format(round(auc,5)))

    aucs_df = pd.DataFrame(aucs,columns = ['auc'])

    print(aucs_df.describe())

    return oof, preds
oof, preds = gmm(train, test)

oof, preds = gmm(train_valid_df, test_valid_df)



auc = roc_auc_score(test_valid_df['target'], preds)

print('AUC: {}'.format(round(auc,5)))
train_valid_df.to_csv('train_valid_df.csv', index=False)

test_valid_df.to_csv('test_valid_df.csv', index=False)