import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy as sp

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from subprocess import check_output

train = pd.read_csv('../input/train.csv')[:700000]

test = pd.read_csv('../input/test.csv')



print (train.shape,train.columns)

print(test.shape,test.columns)

test['split']=1

train['split']=0

total=train.append(test)
# lat and long number comes from & credit to DrGuillermo: Animation

xlim = [-74.03, -73.77]

ylim = [40.63, 40.85]

total = total[(total.pickup_longitude> xlim[0]) & (total.pickup_longitude < xlim[1])]

total = total[(total.dropoff_longitude> xlim[0]) & (total.dropoff_longitude < xlim[1])]

total = total[(total.pickup_latitude> ylim[0]) & (total.pickup_latitude < ylim[1])]

total = total[(total.dropoff_latitude> ylim[0]) & (total.dropoff_latitude < ylim[1])]
from math import radians, cos, sin, asin, sqrt, atan2,degrees



def distance(row):

#     lon1, lat1, lon2, lat2):

    """

    Calculate the circle distance between two points in lat and lon

    on the earth (specified in decimal degrees)

    returning distance in miles

    """

    # need to convert decimal degrees to radians 

    # a unit of angle, equal to an angle at the center of a circle whose arc is equal in length to the radius.

    lon1, lat1, lon2, lat2 = row['pickup_longitude'], row['pickup_latitude'], row['dropoff_longitude'], row['dropoff_latitude']

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])



    # haversine formula 

    dlon = lon2 - lon1 

    dlat = lat2 - lat1 

    #a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2

    #c = 2 * asin(sqrt(a)) 

    c = abs(dlon)+abs(dlat)

    r = 6371 # Radius of earth in kilometers. Use 3956 for miles

    return c * r





#applying to the dataset

total['distance'] = total.apply(distance, axis=1)

total.dtypes
#total transformation time

total['pickup_datetime'] = pd.to_datetime(total['pickup_datetime'])

total["p_day"] = total["pickup_datetime"].dt.strftime('%u').astype(int)

total["p_hour"] = total["pickup_datetime"].dt.strftime('%H').astype(int)

total["p_min"] = total["pickup_datetime"].dt.strftime('%M').astype(int)

total["p_month"]= total["pickup_datetime"].dt.strftime('%m').astype(int)

total['p_day_hour'] = total['p_day'] * 24 + total['p_hour']

total['p_min'] = total['pickup_datetime'].dt.minute



#speed

total["log_speed"]= np.log(total['distance']/total['trip_duration']*3600+1)  #log( km/h  )

total["p_x"]=((total['pickup_longitude']+74.25)*110).round(0)   #2miles square

total["p_y"]=((total['pickup_latitude']-40.6)*110).round(0)

total["d_x"]=((total['dropoff_longitude']+74.25)*110).round(0)

total["d_y"]=((total['dropoff_latitude']-40.6)*110).round(0)

total['p_block']=total["p_x"]*1000+total["p_y"]        # qiving unique block numbers

total['d_block']=total["d_x"]*1000+total["d_y"]

total["manhat"]=(total["p_x"]-total["d_x"]).abs()+(total["p_y"]-total["d_y"]).abs()  # estimating block distance

total["log_manh_speed"]= np.log(total['manhat']/total['trip_duration']*3600+1)

total['log_trip_duration'] = np.log(total['trip_duration'].values + 1)

total['time_km']=total['log_trip_duration']/np.log(total['distance']+1)

total['time_km']=total['time_km'].replace([np.inf, -np.inf], np.nan).fillna(value=-1)

total['lon_dist']=( total['pickup_longitude']-total['dropoff_longitude'] ) * 110

total['lat_dist']=( total['pickup_latitude']-total['dropoff_latitude'] ) * 110

total['lon_speed']=total['lon_dist']/total['trip_duration']*3600

total['lat_speed']=total['lat_dist']/total['trip_duration']*3600

 # log trip duration is normal distribution





total['store_and_fwd_flag'] = 1 * (total.store_and_fwd_flag.values == 'Y') # flag Y >>>> 1

total.dtypes

total.head(10)

#split again

train=total[total['split']==0]

test=total[total['split']==1]



# the block distance shows 0,1,2,3 blocks distance are dominant

plt.hist(np.log(train['manhat']+1).values, bins=100,color='g')

plt.xlabel('log(block-trip_duration)')

plt.ylabel('number of train records')

plt.show()



# the block distance shows 0,1,2,3 blocks distance are dominant

plt.hist(np.log(train['distance']+1).values, bins=100,color='g')

plt.xlabel('log(block distance)')

plt.ylabel('number of train records')

plt.show()



# the block distance shows 0,1,2,3 blocks distance are dominant

plt.hist(train['log_speed'].values, bins=100,color='g')

plt.xlabel('log(speed)')

plt.ylabel('number of train records')

plt.show()



# the block distance shows 0,1,2,3 blocks distance are dominant

plt.hist(train['log_trip_duration'].values, bins=100,color='g')

plt.xlabel('log(trip duration)')

plt.ylabel('number of train records')

plt.show()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

train=total[total['split']==0]

groep=train.groupby(['p_day_hour','d_block'])['log_speed'].describe().fillna(method='bfill')

groep['eff']=groep['std']/groep['mean']

groep['eff2']=groep['eff']*groep['std']





def clust(x):

    kl=0

    if x<0.1:   # low variability cluster

        kl=1

    if x>0.1 and x<0.3: # moderate variability cluster, process with short adjustments

        kl=2

    if x>0.3: # high variability class, process times with long outages, failures of tests

        kl=4

    return kl



groep['clust']=groep['eff2'].map(clust)

print(groep)

groep.columns=['count','mean','min','25p','50p','75p','std','max','eff','eff2','clust']

#append data to total to take with analysis later

total=pd.merge(total,groep, how='outer', left_on=['p_day_hour','d_block'],suffixes=('', '_DH'), right_index=True)



groep=train.groupby(['p_x','p_y','d_x','d_y'])['log_speed'].describe()  #.fillna(method='bfill')

groep['eff']=groep['std']/groep['mean']

groep['eff2']=groep['eff']*groep['std']





def clust(x):

    kl=0

    if x<0.1:   # low variability cluster

        kl=1

    if x>0.1 and x<0.3: # moderate variability cluster, process with short adjustments

        kl=2

    if x>0.3: # high variability class, process times with long outages, failures of tests

        kl=4

    return kl



groep['clust']=groep['eff2'].map(clust)

#print(groep)

groep.columns=['count','mean','min','25p','50p','75p','std','max','eff','eff2','clust']

total=pd.merge(total,groep, how='outer', left_on=['p_x','p_y','d_x','d_y'],suffixes=('', '_PD'), right_index=True)



from sklearn.decomposition import PCA, FastICA,TruncatedSVD

from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection

from sklearn.cluster import KMeans

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from pandas.plotting import scatter_matrix

# INPUT df  (dataframe en welke kolommen je gebruikt om te klusteren)

# define 'clust' groep

# define drop colomns



def plot_results(results):

    results=pd.DataFrame(results[:1000])

    results['d_block']=labels    

    

    sns.set(style="ticks")

    sns.pairplot(results,hue='d_block')

    plt.show()

    # To getter a better understanding of interaction of the dimensions

    # plot the first three PCA dimensions

    fig = plt.figure(1, figsize=(12, 12))

    ax = Axes3D(fig, elev=-150, azim=110)

    ax.scatter(results[0], results[1], results[2], c=results['d_block'], cmap=plt.cm.Paired)

    ax.set_title("cluster 3D")

    ax.set_xlabel("1st eigenvector")

    ax.w_xaxis.set_ticklabels([])

    ax.set_ylabel("2nd eigenvector")

    ax.w_yaxis.set_ticklabels([])

    ax.set_zlabel("3rd eigenvector")

    ax.w_zaxis.set_ticklabels([])



    plt.show()

#-------------------------------------

train=total[total['split']==0]



labels =total['d_block']

drop_columns=['dropoff_datetime', 'id', 'split', 'pickup_datetime']

drop_columns=list(set(drop_columns))

#y values df_new['y']

# X = all the variables X10-X300 not dupl, not singular

X = total.drop(drop_columns,axis=1)

print(X.columns)

#ndex1=[t for t in range(0,len(train))]

#X.index=index1 #drop is moved up is already happened

X=X.replace([np.inf, -np.inf], np.nan).fillna(value=0)

#print(X)

n_comp = 5  #define number of clusters

#-------------------------------------







print('-------Sparse Random Projection---------')

# SRP

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)

#normalizer = Normalizer(copy=False)

#lsa = make_pipeline(srp, normalizer)

results = srp.fit_transform(X)

#plot_results(results)



#append best clusters

#Append decomposition components to datasets  # to do in next part

for i in range(1, n_comp + 1):

    total['srp_' + str(i)] = results[:,i - 1]
print(total)
from collections import Counter

def todrop_col(df,tohold):

    # use todrop_col(dataframe,['listtohold'])

    # Categorical features

    df.replace([np.inf, -np.inf], np.nan).fillna(value=-1)

    

    cat_cols = []

    for c in df.columns:

        if df[c].dtype == 'object':

            cat_cols.append(c)

        if df[c].dtype == 'datetime64[ns]':

            cat_cols.append(c)

    print('Categorical columns:', cat_cols)

    

    

    # Constant columns

    cols = df.columns.values    

    const_cols = []

    for c in cols:   

        if len(df[c].unique()) == 1:

            const_cols.append(c)

    print('Constant cols:', const_cols)

    

    

    # Dublicate features

    d = {}; done = []

    cols = df.columns.values

    for c in cols:

        d[c]=[]

    for i in range(len(cols)):

        if i not in done:

            for j in range(i+1, len(cols)):

                if df[cols[i]].dtype == df[cols[j]].dtype:

                    if all(df[cols[i]] == df[cols[j]]):

                        done.append(j)

                        d[cols[i]].append(cols[j])

    dub_cols = []

    for k in d.keys():

        if len(d[k]) > 0: 

            # print k, d[k]

            dub_cols += d[k]        

    print('Dublicates:', dub_cols)

    

    kolom=list(set(dub_cols+const_cols+cat_cols))

    kolom=[k for k in kolom if k not in tohold]

    

    return kolom



def tree_col(df,splitcol,splitval,groupcol):

    #use tree_col(dataframe,column that splits,vale to split, column that groups)

    #sklear feature selection

    import sklearn    

    from sklearn.svm import LinearSVC

    from sklearn.feature_selection import SelectFromModel

    from sklearn.ensemble import ExtraTreesClassifier

    

    tabel = df[df[splitcol]==splitval]

    label = tabel[groupcol].round(0)

    feat = df.columns  

    clf = ExtraTreesClassifier()

    clf = clf.fit(tabel[feat], label)

    model = SelectFromModel(clf, prefit=True)

    interesting_cols = model.transform(tabel[feat])

    print('Treeclassifier cols',interesting_cols.shape)

    tabel2=pd.DataFrame(interesting_cols,index=tabel.index)

    feat2=tabel2.columns

    feat3=[]

    for ci in feat:

        for cj in feat2:

            if all(tabel[ci] == tabel2[cj]):

                feat3.append(ci) 

    #print('interesting Treecolumns',feat3)

    

    return feat3



train1=total[total['split']==0]

print(train1.dtypes)

print(todrop_col(train1,['d_block']))



train1=train1.drop(['dropoff_datetime','pickup_datetime','id','25p', 'clust'],axis=1).replace([np.inf, -np.inf], np.nan).fillna(value=-1)



print(tree_col(train1,'split',0,'d_block'))
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans



train1=total[total['split']==0]

test1=total[total['split']==1]



#feature_names = ['dropoff_latitude', 'dropoff_longitude', 'pickup_latitude', 'compass', 'p_hour','p_day','mean', 'p_block', 'd_block', 'manhat','log_trip_duration','time_km','lon_dist','lat_dist','lon_speed','lat_speed']

#feature_names = ['dropoff_latitude', 'dropoff_longitude', 'distance','d_x', 'd_y', 'p_block', 'd_block', 'count_PD', 'mean_PD', 'min_PD', '50p_PD', '75p_PD', 'std_PD', 'max_PD', 'eff2_PD','lon_dist','lat_dist','lon_speed','lat_speed']

#feature_names = ['dropoff_latitude', 'dropoff_longitude', 'p_y', 'd_x', 'd_y', 'p_block', 'd_block', 'manhat', 'count_PD', 'mean_PD', 'min_PD', '25p_PD', '50p_PD', '75p_PD', 'std_PD', 'max_PD', 'eff_PD', 'eff2_PD','count_DH','50p_lat', 'id', 'count_lat', 'eff2', 'split', '50p_lon', 'dropoff_datetime', 'min_lon', 'eff2_lat', '25p', '75p_lon', 'clust_lat', 'std_lon', '75p_lat', 'std_lat', '25p_lon', 'eff2_lon', 'pickup_datetime', 'max_lat', 'clust', 'eff_lat', 'clust_lon', 'min_lat', 'count', 'std', '25p_lat', 'count_lon', 'max', 'eff_lon', 'eff', 'mean_lon', '75p', 'min', 'max_lon', '50p', 'mean_lat', 'mean']

#feature_names = ['dropoff_latitude', 'dropoff_longitude', 'p_y', 'd_x', 'd_y', 'p_block', 'd_block', 'manhat', 'count_PD', 'mean_PD', 'min_PD', '25p_PD', '50p_PD', '75p_PD', 'std_PD', 'max_PD', 'eff_PD', 'eff2_PD','count_DH','50p_lat', 'id', 'count_lat', 'eff2', 'split', '50p_lon', 'dropoff_datetime', 'min_lon', 'eff2_lat', '25p', '75p_lon', 'clust_lat', 'std_lon', '75p_lat', 'std_lat', '25p_lon', 'eff2_lon', 'pickup_datetime', 'max_lat', 'clust', 'eff_lat', 'clust_lon', 'min_lat', 'count', 'std', '25p_lat', 'count_lon', 'max', 'eff_lon', 'eff', 'mean_lon', '75p', 'min', 'max_lon', '50p', 'mean_lat', 'mean']

feature_names =['dropoff_latitude', 'dropoff_longitude', 'd_x', 'd_y', 'd_block', 'count', 'count_PD', 'mean_PD', 'std_PD', 'max_PD']

feature_names =['dropoff_latitude', 'dropoff_longitude', 'p_y', 'd_x', 'd_y', 'd_block', 'count', 'count_PD', 'mean_PD', '25p_PD', '50p_PD', '75p_PD', 'std_PD', 'max_PD', 'eff_PD', 'srp_3', 'srp_5']

print(np.setdiff1d(train1.columns, test1.columns))

do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration','pickup_date']

y = train['log_speed']

feature_names=[k for k in feature_names if k not in do_not_use_for_training]

Xtr, Xv, ytr, yv = train_test_split(train1[feature_names].values, y, test_size=0.2, random_state=2017)

dtrain = xgb.DMatrix(Xtr, label=ytr)

dvalid = xgb.DMatrix(Xv, label=yv)

dtest = xgb.DMatrix(test1[feature_names].values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]



# Try different parameters! My favorite is random search :)

xgb_pars = {'min_child_weight': 100, 'eta': 0.5, 'colsample_bytree': 0.3, 'max_depth': 10,

            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,

            'eval_metric': 'rmse', 'objective': 'reg:linear'}



# You could try to train with more epoch

model = xgb.train(xgb_pars, dtrain, 200, watchlist, early_stopping_rounds=50,

                  maximize=False, verbose_eval=50)
print('Modeling RMSLE %.5f' % model.best_score)

feature_importance_dict = model.get_fscore()

fs = ['f%i' % i for i in range(len(feature_names))]

f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()), 'importance': list(feature_importance_dict.values())})

f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})

feature_importance = pd.merge(f1, f2, how='right', on='f')

feature_importance[['feature_name', 'importance']].sort_values(by='importance', ascending=False)
ypred = model.predict(dvalid)

fig,ax = plt.subplots(ncols=2)

ax[0].scatter(ypred, yv, s=0.1, alpha=0.1)

ax[0].set_xlabel('log(prediction)')

ax[0].set_ylabel('log(ground truth)')

ax[1].scatter(np.exp(ypred), np.exp(yv), s=0.1, alpha=0.1)

ax[1].set_xlabel('prediction')

ax[1].set_ylabel('ground truth')

plt.show()
ytest = model.predict(dtest)

print('Test shape OK.') if test.shape[0] == ytest.shape[0] else print('Oops')

test['trip_duration'] = np.exp(ytest) - 1

test[['id', 'trip_duration']].to_csv('d:\input\paul_xgb_submission.csv.gz', index=False, compression='gzip')
