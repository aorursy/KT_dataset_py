import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import matplotlib.pyplot as plt



# read datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test['y'] = 102  # to make append possible

print

y_train = train["y"]

totaal= train.append(test)

#yx0_train = totaal[['y','X0']]

#print(yx0_train.groupby('X0').mean().sort('y'))

#temp=yx0_train.groupby('X0').mean().sort('y')

#templ=temp.index

#print(templ)

#totaal['X0'].replace(to_replace=templ, value=[x for x in range(0,len(templ))], inplace=True, method='pad', axis=1)

print(totaal.head())

# process columns, apply LabelEncoder to categorical features

for c in totaal.columns:

    if totaal[c].dtype == 'object':

        tempt = totaal[['y',c]]

        temp=tempt.groupby(c).mean().sort('y')

        templ=temp.index

        print(templ)

        aant=len(templ)

        train[c].replace(to_replace=templ, value=[x/aant for x in range(0,aant)], inplace=True, method='pad', axis=1)

        test[c].replace(to_replace=templ, value=[x/aant for x in range(0,aant)], inplace=True, method='pad', axis=1)

         #test[c] = lbl.transform(list(test[c].values))



# shape  

print(train.head())

print(test.head())

print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))





##Add decomposed components: PCA / ICA etc.

from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import NMF, LatentDirichletAllocation,FactorAnalysis,MiniBatchDictionaryLearning

from sklearn.cluster import MiniBatchKMeans

n_comp = 2

#nmf

nmf = NMF(n_components=n_comp, random_state=1,          alpha=.1, l1_ratio=.5)

nmf_results_train=nmf.fit_transform(train.drop(['y'], axis=1))#,y=y_train)  #R2 +Y 0.44 #R2-Y 0.44

nmf_results_test=nmf.transform(test.drop(['y'], axis=1))





#lda

lda = LatentDirichletAllocation(n_topics=n_comp, max_iter=5,                                learning_method='online',                                learning_offset=50.,                                random_state=0).fit(train.drop(["y"], axis=1),y_train)

lda_results_train = lda.fit_transform(train.drop(["y"], axis=1),y_train ) #R2 +Y 0.40

lda_results_test = lda.transform(test.drop(['y'], axis=1))



# tSVD

tsvd = TruncatedSVD(n_components=n_comp, random_state=42)

tsvd_results_train = tsvd.fit_transform(train.drop(['y'], axis=1) ) #,y=y_train) #R2 +Y=0.47

tsvd_results_test = tsvd.transform(test.drop(['y'], axis=1))



# PCA

pca = PCA(n_components=n_comp, random_state=42)

pca2_results_train = pca.fit_transform(train.drop(['y'], axis=1))#,y=y_train) #R2 +Y 0.467

pca2_results_test = pca.transform(test.drop(['y'], axis=1))



# ICA

ica = FastICA(n_components=n_comp, random_state=42)

ica2_results_train = ica.fit_transform(train.drop(['y'], axis=1) ) #,y=y_train) #R2+y=0.439

ica2_results_test = ica.transform(test.drop(['y'], axis=1))



#FA

fa =FactorAnalysis(n_components=n_comp)

fa2_results_train = fa.fit_transform(train.drop(["y"], axis=1)) #,y=y_train) #R2 +y 0.412

fa2_results_test = fa.transform(test.drop(['y'], axis=1))



#MDB

mdb=MiniBatchDictionaryLearning(n_components=n_comp, alpha=0.1,n_iter=50, batch_size=3,random_state=42)

mdb_results_train = mdb.fit_transform(train.drop(['y'], axis=1)) #,y=y_train) #R2 +y 0.469

mdb_results_test = mdb.transform(test.drop(['y'], axis=1))



#mbk

mbk=MiniBatchKMeans(n_clusters=n_comp, tol=1e-3, batch_size=20,max_iter=50, random_state=42)

mbk_results_train = mbk.fit_transform(train.drop(['y'], axis=1)) #,y=y_train) #R2 +y 0.437

mbk_results_test = mbk.transform(test.drop(['y'], axis=1))



# Append decomposition components to datasets

for i in range(1, n_comp+1):

    train['mbk_' + str(i)] = mbk_results_train[:,i-1]

    test['mbk_' + str(i)] = mbk_results_test[:, i-1]

    

    train['mdb_' + str(i)] = mdb_results_train[:,i-1]

    test['mdb_' + str(i)] = mdb_results_test[:, i-1]

     

    train['fa_' + str(i)] = fa2_results_train[:,i-1]

    test['fa_' + str(i)] = fa2_results_test[:, i-1]

    

    train['pca_' + str(i)] = pca2_results_train[:,i-1]

    test['pca_' + str(i)] = pca2_results_test[:, i-1]

    

    train['ica_' + str(i)] = ica2_results_train[:,i-1]

    test['ica_' + str(i)] = ica2_results_test[:, i-1]

    

    train['tsvd_' + str(i)] = tsvd_results_train[:,i-1]

    test['tsvd_' + str(i)] = tsvd_results_test[:, i-1]



    train['lda_' + str(i)] = lda_results_train[:,i-1]

    test['lda_' + str(i)] = lda_results_test[:, i-1]    



    train['nmf_' + str(i)] = nmf_results_train[:,i-1]

    test['nmf_' + str(i)] = nmf_results_test[:, i-1]        

    





print('Shape with PCA train: {}\nShape test: {}'.format(train.shape, test.shape))



y_mean = np.mean(y_train)



print(test.head())



### Regressor

import xgboost as xgb



# prepare dict of params for xgboost to run with

xgb_params = {

    'n_trees': 500, 

    'eta': 0.0025,

    'max_depth': 4,

    'subsample': 0.95,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean, # base prediction = mean(target)

    'silent': 1

}





# form DMatrices for Xgboost training

dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)

dtest = xgb.DMatrix(test.drop('y', axis=1))





num_boost_rounds = 1500

# train model

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

fig, ax = plt.subplots(figsize=(12,15))

xgb.plot_importance(model, height=0.8, ax=ax, max_num_features=30)

plt.show()



# check f2-score (to get higher score - increase num_boost_round in previous cell)

from sklearn.metrics import r2_score

print(r2_score(model.predict(dtrain), dtrain.get_label()))



# make predictions and save results

y_pred = model.predict(dtest)

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})





plt.figure(figsize=(12,8))

sns.distplot(output.y.values, bins=50, kde=False)

plt.xlabel('Predicted AVG Time on Test platform', fontsize=12)

plt.show()

    

output.to_csv('submission_baseLine.csv', index=False)