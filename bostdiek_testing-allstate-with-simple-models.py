# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp



from scipy.special import erfinv

from scipy.stats import norm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



%matplotlib inline



import matplotlib.pyplot as plt
data_train_raw = pd.read_csv('../input/train.csv')

data_test_raw = pd.read_csv('../input/test.csv')

# print(data_train_raw.dtypes)
ylog=np.log(data_train_raw['loss'])

ymean=ylog.mean()

ystd=ylog.std()

data_train_raw['loss_g']=(ylog-ymean)/ystd
def ToUniform(y):

    z = norm.cdf(-y/np.sqrt(2))

    return z

def UniformToGauss(z):

    return -np.sqrt(2)*norm.ppf(z)*ystd+ymean

def BackToOriginal(z):

    z[z <= 0] = 0.001

    z[z >=1] = 1-0.001

    return np.exp(UniformToGauss(z))



data_train_raw['loss_u']=ToUniform(data_train_raw['loss_g'])
plt.figure(figsize=(8,8))



plt.subplot(3,2,1)

plt.hist(data_train_raw['loss'],100)

plt.title('loss')



plt.subplot(3,2,2)

plt.hist(ylog,100)

plt.title('log(loss) - Gauss')



plt.subplot(3,2,3)

plt.hist(data_train_raw['loss_g'],100)

plt.title('Normal')



plt.subplot(3,2,4)

plt.hist(ToUniform(data_train_raw['loss_g']),100)

plt.title('Uniform')



plt.subplot(3,2,5)

plt.hist(BackToOriginal(data_train_raw['loss_u']),100)

plt.title('..and Back')



plt.show()
#list(zip(BackToOriginal(data_train_raw['loss_u']),data_train_raw['loss']))
col_uniques=[]

for col in data_train_raw.columns:

    if (col.find('cat') !=-1):

        col_uniques.append([col, len(data_train_raw[col].unique())])

print(col_uniques)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



for col in data_train_raw.columns:

    if (col.find('cat') !=-1):

      #  print(col)

        data_train_raw[str(col+'_numerical')]=le.fit_transform(data_train_raw[col])

        data_test_raw[col] = data_test_raw[col].map(lambda s: '<unknown>' if s not in le.classes_ else s)

        le.classes_ = np.append(le.classes_, '<unknown>')

        data_test_raw[str(col+'_numerical')]=le.transform(data_test_raw[col])

print(data_train_raw.columns)
XCols =[0]

datacols=data_train_raw.columns

for c in range(len(datacols)):

    if(datacols[c].find('cont')!=-1 or datacols[c].find('numerical')!=-1):

        XCols.append(c)

X_total = data_train_raw[XCols]

Y_total = data_train_raw['loss_u']
XColst =[0]

datacols=data_test_raw.columns

for c in range(len(datacols)):

    if(datacols[c].find('cont')!=-1 or datacols[c].find('numerical')!=-1):

        XColst.append(c)

X_test = data_test_raw[XColst]
# data_test_raw[XColst]
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_total, Y_total, test_size=0.1, random_state=0)
stds=[1]

means=[0]

xcols=list(X_train.columns)





for c in range(1,len(xcols)):

    mm = X_train[xcols[c]].mean()

    ss = X_train[xcols[c]].std()

    

    means.append(mm)

    stds.append(ss)

    

#    print(xcols[c],r)
X_train = (X_train[xcols] - means) / stds

X_valid = (X_valid[xcols] - means) / stds

X_test = (X_test[xcols] - means) / stds
xcols.remove('id')



print("Train")

print(X_train[xcols[100]].describe())

print("Valid")

print(X_valid[xcols[100]].describe())

print("Test")

print(X_test[xcols[100]].describe())
from sklearn.cluster import KMeans

km = KMeans(n_clusters=50,n_jobs=-1)

km.fit(X_train[xcols])



X_train['km']=km.predict(X_train[xcols])

X_valid['km']=km.predict(X_valid[xcols])

X_test['km']=km.predict(X_test[xcols])

xcols.append('km')
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X_train[xcols])
X_train_transformed = pca.transform(X_train[xcols])
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.scatter(X_train_transformed[:,0],y_train)

plt.title('First Axis')

plt.ylabel('loss')



plt.subplot(1,2,2)

plt.scatter(X_train_transformed[:,1],y_train)

plt.title('Second Axis')

plt.show()
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_train_transformed[:,0], X_train_transformed[:,1], y_train)



plt.show()
xnp=X_train[xcols].as_matrix().T



covariantmatrix = 1/xnp.shape[1] * np.dot(xnp, xnp.T)
U, S, V = np.linalg.svd(covariantmatrix)
stotal=S.sum()

# Find how many features we need to keep 99% of the variance

for i in range(len(S)):

    if (S[:i+1].sum()/stotal > 0.99):

        print("Using {0} of the PCA columns gets {1:.2f}% of the variance.".format(i+1,S[:i+1].sum()/stotal * 100))

        print("This removes {0} columns and reduces the data by {1:.2f}%".format(len(S)-(i+1), float(len(S)-(i+1))/len(S) * 100 ))

        Ureduce = U[:,:i+1]

        break
print(Ureduce.shape)

print(xnp.shape)



Z_train_np=np.dot(xnp.T, Ureduce)

Z_train=pd.DataFrame(Z_train_np, columns=list('PCA_'+str(i) for i in range(Ureduce.shape[1])))



Z_test_np = np.dot(X_test[xcols].as_matrix(), Ureduce)

Z_test=pd.DataFrame(Z_test_np, columns=list('PCA_'+str(i) for i in range(Ureduce.shape[1])))



Z_valid_np = np.dot(X_valid[xcols].as_matrix(), Ureduce)

Z_valid=pd.DataFrame(Z_valid_np, columns=list('PCA_'+str(i) for i in range(Ureduce.shape[1])))



print(Z_train.head())
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
#minleafs=[1,2,5,10,20,50,100]

minleafs=[50]

bestmodel = RandomForestRegressor()

bestvalidscore=1e10

for msl in minleafs:

    print("Min leafsize:{0}".format(msl))

    rfr = RandomForestRegressor(n_estimators= 200, n_jobs=-1,min_samples_leaf = msl)

    rfr.fit(X_train[xcols],y_train)

    trainscore=mean_absolute_error(BackToOriginal(rfr.predict(X_train[xcols])),BackToOriginal(y_train))

    validscore=mean_absolute_error(BackToOriginal(rfr.predict(X_valid[xcols])),BackToOriginal(y_valid))

    if validscore-trainscore < bestvalidscore:

        bestvalidscore = validscore

        bestmodel = rfr

    print("\ttrain_score:{0:.1f}, valid_score:{1:.1f}, difference:{2:.2f}% \n".format(trainscore,validscore, -(trainscore-validscore)/trainscore * 100))



rfr = bestmodel
X_train['rfr']=BackToOriginal(rfr.predict(X_train[xcols]))

X_valid['rfr']=BackToOriginal(rfr.predict(X_valid[xcols]))

X_test['rfr']=BackToOriginal(rfr.predict(X_test[xcols]))
trainscore=mean_absolute_error(X_train['rfr'],BackToOriginal(y_train))

validscore=mean_absolute_error(X_valid['rfr'],BackToOriginal(y_valid))
plt.figure(figsize=(8,5))



plt.subplot(1,2,1)

plt.title(r'Training score='+str(trainscore))

plt.scatter(X_train['rfr'],BackToOriginal( y_train))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.subplot(1,2,2)

plt.title(r'Validation score='+str(validscore))

plt.scatter(X_valid['rfr'], BackToOriginal(y_valid))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.show()
rfrpred=pd.DataFrame(list(zip(X_test['id'],X_test['rfr'])),columns=('id','loss'))

rfrpred['id']=rfrpred['id'].astype('int')

                     
rfrpred.to_csv('submit_RFR_' +str(validscore) +'.csv', index=False)
list(enumerate(sorted(list(zip(xcols,rfr.feature_importances_)), key=lambda l:l[1], reverse=True)))[:20]
#minleafs=[1,2,5,10,20,50,100]

minleafs=[50]

bestmodel = RandomForestRegressor()

bestvalidscore=1e10

for msl in minleafs:

    print("Min leafsize:{0}".format(msl))

    rfrpca = RandomForestRegressor(n_estimators= 100, n_jobs=-1,min_samples_leaf = msl)

    rfrpca.fit(Z_train,y_train)

    trainscore=mean_absolute_error(BackToOriginal(rfrpca.predict(Z_train)),BackToOriginal(y_train))

    validscore=mean_absolute_error(BackToOriginal(rfrpca.predict(Z_valid)),BackToOriginal(y_valid))

    if validscore-trainscore < bestvalidscore:

        bestvalidscore = validscore

        bestmodel = rfrpca

    print("\ttrain_score:{0:.1f}, valid_score:{1:.1f}, difference:{2:.2f}% \n".format(trainscore,validscore, -(trainscore-validscore)/trainscore * 100))



rfrpca = bestmodel
X_train['rfr_pca']=BackToOriginal(rfrpca.predict(Z_train))

X_valid['rfr_pca']=BackToOriginal(rfrpca.predict(Z_valid))

X_test['rfr_pca']=BackToOriginal(rfrpca.predict(Z_test))



trainscore_rfr_pca=mean_absolute_error(X_train['rfr_pca'],BackToOriginal(y_train))

validscore_rfr_pca=mean_absolute_error(X_valid['rfr_pca'],BackToOriginal(y_valid))
plt.figure(figsize=(10,5))



plt.subplot(1,2,1)

plt.title(r'Training score='+str(trainscore_rfr_pca))

plt.scatter(X_train['rfr_pca'],BackToOriginal( y_train))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.subplot(1,2,2)

plt.title(r'Validation score='+str(validscore_rfr_pca))

plt.scatter(X_valid['rfr_pca'], BackToOriginal(y_valid))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.show()
rfrpred_PCA=pd.DataFrame(list(zip(X_test['id'],X_test['rfr_pca'])),columns=('id','loss'))

rfrpred_PCA['id']=rfrpred_PCA['id'].astype('int')



rfrpred_PCA.to_csv('submit_RFR_PCA_' +str(trainscore_rfr_pca) +'.csv', index=False)                     
list(enumerate(sorted(list(zip(Z_train.columns,rfrpca.feature_importances_)), key=lambda l:l[1], reverse=True)))[:20]
X_valid[['rfr','rfr_pca']].head(10)
from sklearn.linear_model import Ridge
rid = Ridge(alpha=1e-6, fit_intercept=True, normalize=False, 

                  copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
rid.fit(X_train[xcols],y_train)
X_train['rid'] = rid.predict(X_train[xcols])

X_valid['rid'] = rid.predict(X_valid[xcols])

X_test['rid'] = rid.predict(X_test[xcols])



X_train.loc[X_train['rid'] < 0,'rid'] = 0.001

X_valid.loc[X_valid['rid'] < 0,'rid'] = 0.001

X_test.loc[X_test['rid'] < 0,'rid'] = 0.001



X_train.loc[X_train['rid'] >=1,'rid'] = 0.999

X_valid.loc[X_valid['rid'] >=1,'rid'] = 0.999

X_test.loc[X_test['rid'] >=1,'rid'] = 0.999



#print(X_train['rid'].max(),X_valid['rid'].max(),X_test['rid'].max())

#print(X_train['rid'].min(),X_valid['rid'].min(),X_test['rid'].min())



X_train['rid']=BackToOriginal(X_train['rid'])

X_valid['rid']=BackToOriginal(X_valid['rid'])

X_test['rid']=BackToOriginal(X_test['rid'])



#print(X_train['rid'].max(),X_valid['rid'].max(),X_test['rid'].max())

#print(X_train['rid'].min(),X_valid['rid'].min(),X_test['rid'].min())



trainscore=mean_absolute_error(X_train['rid'],BackToOriginal(y_train))

testscore=mean_absolute_error(X_valid['rid'],BackToOriginal(y_valid))

print(trainscore,testscore)
plt.figure(figsize=(10,5))



plt.subplot(1,2,1)

plt.title(r'Training score='+str(trainscore))

plt.scatter(X_train['rid'], BackToOriginal(y_train))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.subplot(1,2,2)

plt.title(r'Validation score='+str(testscore))

plt.scatter(X_valid['rid'], BackToOriginal(y_valid))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.show()
ridpred=pd.DataFrame(list(zip(X_test['id'],X_test['rid'])),columns=('id','loss'))

ridpred['id']=ridpred['id'].astype('int')

ridpred.head()
ridpred.to_csv('submit_ridge_' +str(testscore) +'.csv', index=False)
ridPCA = Ridge(alpha=1e-6, fit_intercept=True, normalize=False, 

                  copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)



ridPCA.fit(Z_train,y_train)
X_train['rid_pca'] = BackToOriginal(ridPCA.predict(Z_train))

X_valid['rid_pca'] = BackToOriginal(ridPCA.predict(Z_valid))

X_test['rid_pca'] = BackToOriginal(ridPCA.predict(Z_test))
trainscoreridPCA=mean_absolute_error(X_train['rid_pca'],BackToOriginal(y_train))

testscoreridPCA=mean_absolute_error(X_valid['rid_pca'],BackToOriginal(y_valid))



plt.figure(figsize=(8,3))



plt.subplot(1,2,1)

plt.title(r'Training score='+str(trainscoreridPCA))

plt.scatter(X_train['rid_pca'], BackToOriginal(y_train))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.subplot(1,2,2)

plt.title(r'Validation score='+str(testscoreridPCA))

plt.scatter(X_valid['rid_pca'], BackToOriginal(y_valid))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.show()
ridpredPCA=pd.DataFrame(list(zip(X_test['id'],X_test['rid_pca'])),columns=('id','loss'))

ridpredPCA['id']=ridpredPCA['id'].astype('int')

ridpredPCA.head()



ridpredPCA.to_csv('submit_ridge_PCA_' +str(testscoreridPCA) +'.csv', index=False)
from sklearn.neural_network import MLPRegressor



mlpnnR = MLPRegressor(hidden_layer_sizes=(int(X_train.shape[1]/2),int(X_train.shape[1]/2), int(X_train.shape[1]/2),int(X_train.shape[1]/2)), 

                     #  activation='logistic', 

                       solver='adam', 

                       alpha=0.1, 

                       batch_size='auto',

                       learning_rate='adaptive',

                       learning_rate_init=0.0001,

                       power_t=0.5, max_iter=200,

                       shuffle=True, 

                       random_state=None, 

                       tol=0.00001, 

                       verbose=True,

                       warm_start=False,

                       momentum=0.9,

                       nesterovs_momentum=True, early_stopping=False, 

                       validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 

                       epsilon=1e-08)
mlpnnR.fit(X_train[xcols],y_train)
X_train['nn'] = mlpnnR.predict(X_train[xcols])

X_valid['nn'] = mlpnnR.predict(X_valid[xcols])

X_test['nn'] = mlpnnR.predict(X_test[xcols])



X_train.loc[X_train['nn'] < 0,'nn'] = 0.001

X_valid.loc[X_valid['nn'] < 0,'nn'] = 0.001

X_test.loc[X_test['nn'] < 0,'nn'] = 0.001



X_train.loc[X_train['nn'] >=1,'nn'] = 1-0.001

X_valid.loc[X_valid['nn'] >=1,'nn'] = 1-0.001

X_test.loc[X_test['nn'] >=1,'nn'] = 1-0.001



X_train['nn'] = BackToOriginal(X_train['nn'])

X_valid['nn'] = BackToOriginal(X_valid['nn'])

X_test['nn'] = BackToOriginal(X_test['nn'])
trainnnpred=X_train['nn']

validnnpred=X_valid['nn']



nnscoret=mean_absolute_error(trainnnpred,BackToOriginal(y_train))

nnscorev=mean_absolute_error(validnnpred,BackToOriginal(y_valid))



plt.figure(figsize=(10,5))



plt.subplot(1,2,1)

plt.title(r'Training score='+str(nnscoret))

plt.scatter(trainnnpred, BackToOriginal(y_train))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.subplot(1,2,2)

plt.title(r'Validation score='+str(nnscorev))

plt.scatter(validnnpred, BackToOriginal(y_valid))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.show()
mlpout=pd.DataFrame(

    list(zip(X_test['id'],X_test['nn'])),

    columns=('id','loss'))

mlpout['id']=mlpout['id'].astype('int')

mlpout.head()

mlpout.to_csv('submit_nnet_' +str(nnscorev) +'.csv', 

               index=False)
mlpnnR_PCA = MLPRegressor(hidden_layer_sizes=(int(X_train.shape[1]/2),int(X_train.shape[1]/2), int(X_train.shape[1]/2),int(X_train.shape[1]/2)), 

                     #  activation='logistic', 

                       solver='adam', 

                       alpha=0.1, 

                       batch_size='auto',

                       learning_rate='adaptive',

                       learning_rate_init=0.0001,

                       power_t=0.5, max_iter=200,

                       shuffle=True, 

                       random_state=None, 

                       tol=0.00001, 

                       verbose=True,

                       warm_start=False,

                       momentum=0.9,

                       nesterovs_momentum=True, early_stopping=False, 

                       validation_fraction=0.1, beta_1=0.9, beta_2=0.999, 

                       epsilon=1e-08)



mlpnnR_PCA.fit(Z_train,y_train)
X_train['nn_pca'] = BackToOriginal(mlpnnR_PCA.predict(Z_train))

X_valid['nn_pca'] = BackToOriginal(mlpnnR_PCA.predict(Z_valid))

X_test['nn_pca'] = BackToOriginal(mlpnnR_PCA.predict(Z_test))
trainnnpred=X_train['nn_pca']

validnnpred=X_valid['nn_pca']



nnscoret_pca=mean_absolute_error(trainnnpred,BackToOriginal(y_train))

nnscorev_pca=mean_absolute_error(validnnpred,BackToOriginal(y_valid))



plt.figure(figsize=(8,3))



plt.subplot(1,2,1)

plt.title(r'Training score='+str(nnscoret_pca))

plt.scatter(trainnnpred, BackToOriginal(y_train))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.subplot(1,2,2)

plt.title(r'Validation score='+str(nnscorev_pca))

plt.scatter(validnnpred, BackToOriginal(y_valid))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.show()
mlpoutPCA=pd.DataFrame(

    list(zip(X_test['id'],X_test['nn_pca'])),

    columns=('id','loss'))

mlpoutPCA['id']=mlpout['id'].astype('int')

mlpoutPCA.head()

mlpoutPCA.to_csv('submit_nnet_PCA_' +str(nnscorev) +'.csv', 

               index=False)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()



ensemblefeats=['rfr','nn','rfr_pca','nn_pca'] + xcols



lr.fit(X_train[ensemblefeats],BackToOriginal(y_train))
ave_predtrain=lr.predict(X_train[ensemblefeats])

ave_predvalid=lr.predict(X_valid[ensemblefeats])



ave_scoret=mean_absolute_error(ave_predtrain,BackToOriginal(y_train))

ave_scorev=mean_absolute_error(ave_predvalid,BackToOriginal(y_valid))



plt.figure(figsize=(10,5))



plt.subplot(1,2,1)

plt.title(r'Training score='+str(ave_scoret))

plt.scatter(ave_predtrain, BackToOriginal(y_train))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.subplot(1,2,2)

plt.title(r'Validation score='+str(ave_scorev))

plt.scatter(ave_predvalid, BackToOriginal(y_valid))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.show()
enembleout=pd.DataFrame(

    list(zip(X_test['id'],lr.predict(X_test[ensemblefeats]))),

    columns=('id','loss'))

enembleout['id']=enembleout['id'].astype('int')

print(enembleout.head())

enembleout.to_csv('submit_ensemble_' +str(ave_scorev) +'.csv', 

               index=False)
X_train['ens_a']=ave_predtrain

X_valid['ens_a']=ave_predvalid

X_test['ens_a']=enembleout['loss']
import xgboost as xgb
dtrain = xgb.DMatrix(X_train[ensemblefeats + ['ens_a']].values, label=BackToOriginal(y_train.values))



xgb_params = {

    'seed': 0,

    'colsample_bytree': 0.7,

    'subsample': 0.7,

    'learning_rate': 0.075,

    'objective': 'reg:linear',

    'max_depth': 6,

    'num_parallel_tree': 1,

    'min_child_weight': 1,

    'eval_metric': 'mae',

    'silent':0

}



num_round = 50

bst = xgb.train(xgb_params, dtrain, num_round)
d_valid=xgb.DMatrix(X_valid[ensemblefeats + ['ens_a']].values)

d_test=xgb.DMatrix(X_test[ensemblefeats + ['ens_a']].values)
xgpred_train = bst.predict(dtrain)

xgpred_valid = bst.predict(d_valid)

xgpred_test = bst.predict(d_test)
xgb_scoret=mean_absolute_error(xgpred_train,BackToOriginal(y_train))

xgb_scorev=mean_absolute_error(xgpred_valid,BackToOriginal(y_valid))



plt.figure(figsize=(10,5))



plt.subplot(1,2,1)

plt.title(r'Training score='+str(xgb_scoret))

plt.scatter(xgpred_train, BackToOriginal(y_train))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.subplot(1,2,2)

plt.title(r'Validation score='+str(xgb_scorev))

plt.scatter(xgpred_valid, BackToOriginal(y_valid))

plt.xlabel('Prediction')

plt.ylabel('Truth')



plt.show()
enembleout=pd.DataFrame(

    list(zip(X_test['id'],xgpred_test)),

    columns=('id','loss'))

enembleout['id']=enembleout['id'].astype('int')

print(enembleout.head())

enembleout.to_csv('submit_ensemble_XGB_' +str(ave_scorev) +'.csv', 

               index=False)
