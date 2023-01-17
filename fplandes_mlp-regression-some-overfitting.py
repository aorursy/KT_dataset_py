from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
testLength = 14 #days
path_to_inputs = '/kaggle/input/covid19-granular-demographics-and-times-series/'

input_filename1="departments_static_data_divBySubPop.csv"

input_filename2="time_series_covid_incidence_divBySubPop.csv"

input_filenameNORM="population_used_for_normalization.csv"

# filename="departments_static_data.csv"
## static data 

input_filename1 = 'departments_static_data_divBySubPop.csv' ## this file may contain the same pre-proc done in the cell below

df1 = pd.read_csv(path_to_inputs+input_filename1, delimiter=',')

df1 # .describe()
# for col in df1.columns[2:]:

#     plt.figure()

#     plt.title(col)

#     plt.hist(df1[col], 30)

# dynamic data (to be predicted)

df2 = pd.read_csv(path_to_inputs+input_filename2)
df5 = pd.read_csv(path_to_inputs+input_filenameNORM)

df5
df1 = df1.fillna(df1.mean())
df2 = df2.fillna(0)
codes_dynamic_file = np.array(df2.iloc[::24,0])

dynamic_file_column_names = np.array(df2.iloc[:24,1])

codes_dynamic_file, dynamic_file_column_names
y = np.array(df2.iloc[:,2:])

ndepartments = 100

nvals = y.shape[0]//ndepartments

nvals
yr = y.reshape( (ndepartments, nvals, y.shape[1]))



reshaped = []

for n in range(ndepartments) :

    reshaped.append(yr[n].transpose().copy())

reshaped = np.array(reshaped)
ndep = 54

nval = 6

y[24*ndep:24*(ndep+1),nval], y.shape
yr[ndep,:,nval], yr.shape
reshaped[ndep,nval]
df1 = df1.sort_values(by=['code'])

Xs = np.array(df1.iloc[:,2:])

Xs
import sklearn.decomposition

Xs.shape



## standardize static data: as usual

mean_Xs = Xs.mean(axis=0)

std_Xs  = Xs.std (axis=0)

scaled_Xs = (Xs-mean_Xs)/std_Xs

Xstatic = scaled_Xs.copy()
def quick_look_at_pca(X, n_components):

    pca = sklearn.decomposition.PCA(n_components=n_components)

    pca.fit(X)

    Xp = pca.transform(X)

    plt.plot(pca.explained_variance_ratio_[:10], label="explained_variance_ratio")

    plt.legend()

    plt.xlabel("n_components (PCA)")

    plt.ylim([0,1])

    Xrecov = pca.inverse_transform(Xp)

    print("reconstruciton (Mean Absolute) Error: ", abs(X-Xrecov).mean())

    print("Xp.shape, pca.noise_variance_", Xp.shape, pca.noise_variance_)

    return Xp    
n_components = None

Xp = quick_look_at_pca(Xstatic, n_components)
def reconstruction_errors(X):

    recos = []

    for n_components in range(1,20, 1):

        pca = sklearn.decomposition.PCA(n_components=n_components)

        pca.fit(X)

        Xp = pca.transform(X)

        Xrecov = pca.inverse_transform(Xp)

        reconstruction_MAE = abs(X-Xrecov).mean()

        recos.append( (n_components, reconstruction_MAE,pca.noise_variance_) )

    return np.array(recos)

recos = reconstruction_errors(Xstatic)

plt.plot(recos[:,0],recos[:,1], marker='o', label="reconstruction error (MAE)")

plt.plot(recos[:,0],recos[:,2], marker='o', label="noise variance")

plt.xlabel("n_components (PCA)")

plt.legend()
n_components = 7

Xp = quick_look_at_pca(Xstatic, n_components)
Xs = np.array(df1.iloc[:,2:])

dynamicData = reshaped.copy() ## dynamic data (all of it)

Xd  = dynamicData[:, :-testLength ].copy() ## dynamic data used as features 

yd  = dynamicData[:,  -testLength:].copy() ## dynamic data used as ground truth labels/values (to be predicted)



## standardize static data: as usual

mean_Xs = Xs.mean(axis=0)

std_Xs  = Xs.std (axis=0)

scaled_Xs = (Xs-mean_Xs)/std_Xs

Xstatic = scaled_Xs.copy()



## we re-build the data-set ##

n_components = 9

pca = sklearn.decomposition.PCA(n_components=n_components)

Xpca = pca.fit_transform(Xstatic)

# pop = np.array(df5["Pop_sex=all_age=all_Population"])





## standardize dynamic data:

## we use the last 3 days of (available) data as typical value

mean_Xd = Xd[:,-3:].mean(axis=1).mean(axis=0) 

## TODO: try other scales

std_Xd  = Xd[:,-3:].std(axis=1).std(axis=0)

scaled_Xd = (Xd-mean_Xd)/std_Xd

ndep = scaled_Xd.shape[0]

nday_train = scaled_Xd.shape[1]

nvalues = scaled_Xd.shape[2]

scaled_Xd = scaled_Xd.reshape( (ndep, nday_train*nvalues) )



## standardize output with same scales as input

nday_test = testLength

scaled_yd = (yd-mean_Xd)/std_Xd ## we cannot know in advance the scaling factor of future data !!

scaled_yd = scaled_yd.mean(axis=1) ## we average the next two weeks to make it less noisy

nday_test = 1 

scaled_yd = scaled_yd.reshape( (ndep, nday_test*nvalues))

## 

y = scaled_yd.copy()



print(ndep, nday_train, nvalues, nday_test)
## combine static and dynamic input

## here this is a dumb way (nto accounting for the temporal specificity of our data)

X = np.concatenate( (Xpca, scaled_Xd), axis=1) ## using both 



X = Xpca  ## only static !



X = scaled_Xd ## only dynamic (no socio-demographics)

Xpca.shape, scaled_Xd.shape, scaled_yd.shape, X.shape, y.shape
## consider  also sklearn.model_selection.TimeSeriesSplit(n_splits=5, max_train_size=None)



def train_test_pop_split(X,y,test_ratio, seed):

    ## train-test split, KEEPING TRACK of the departmental populations ##

    rng = np.random.default_rng(seed)

    Nexamples = X.shape[0]

    indexes = np.arange(Nexamples, dtype=int)

    Ntest = int(Nexamples*test_ratio)

    test_indexes = rng.choice(indexes, size=Ntest, replace=False)

    train_indexes = []

    for ind in indexes:

        if ind not in test_indexes:

            train_indexes.append(ind)

    train_indexes = np.array(train_indexes)



    X_train= X[train_indexes]

    y_train= y[train_indexes]

#     y_train_pop = pop[train_indexes].reshape( (Nexamples-Ntest,1) )



    X_test = X[test_indexes]

    y_test = y[test_indexes]

#     y_test_pop = pop[test_indexes].reshape( (Ntest,1) )



    return X_train, X_test, y_train, y_test # , y_train_pop, y_test_pop
import sklearn.neural_network



X = np.concatenate( (Xpca, scaled_Xd), axis=1) ## using both 

# X = Xpca  ## only static !

# X = scaled_Xd ## only dynamic (no socio-demographics)



seed = 42

test_ratio=0.33

X_train, X_test, y_train, y_test = train_test_pop_split(X, y, test_ratio, seed)

#, y_train_pop, y_test_pop = train_test_pop_split(X, y, pop, test_ratio, seed)



## model (cheap) ##

# model = sklearn.linear_model.LinearRegression(normalize=False)

network_layers = (500,100)

# model = sklearn.neural_network.MLPRegressor(network_layers,  solver='lbfgs', max_iter=1000)

model = sklearn.neural_network.MLPRegressor(network_layers, learning_rate='adaptive', early_stopping=True, validation_fraction=0.2,n_iter_no_change=20)

# model = sklearn.svm.LinearSVR()

print(X_train.shape, y_train.shape)

model.fit(X=X_train, y=y_train)

## predictions

ypred = model.predict(X_test)

ypred_all = model.predict(X)

ytrue = y_test.copy()



# test_ndep = y_test.shape[0]

# nday_train = y_test.shape[1]

# test_nvalues = y_test.shape[2]

# test_nday_test = scaled_yd.shape[1]



ypred = ypred.reshape((y_test.shape[0], nday_test, nvalues))



ytrue = ytrue.reshape((y_test.shape[0], nday_test, nvalues))



# def raw_number(y, std_Xd, mean_Xd, pop):

#     ## re-scaled predictions

#     raw_number_y = (y*std_Xd+mean_Xd)*pop

#     return raw_number_y



# ypred = raw_number(ypred, std_Xd, mean_Xd, y_test_pop)

# ytrue = raw_number(ytrue, std_Xd, mean_Xd, y_test_pop)
24*67*72, 22512

ypred_all  = ypred_all.reshape((y.shape[0], nday_test, nvalues))
model.score(X,y, sample_weight=None), model.score(X_train,y_train, sample_weight=None), model.score(X_test,y_test, sample_weight=None)
# try again with:

# X = scaled_Xd ## only dynamic (no socio-demographics)



model.score(X,y, sample_weight=None), model.score(X_train,y_train, sample_weight=None), model.score(X_test,y_test, sample_weight=None)
diff = (ypred-ytrue)/ytrue

indicator =23

# for dep in range(33): 

#     plt.plot(diff[dep, :, indicator])
diff = (ypred-ytrue)/ytrue

indicator =23

codes_dynamic_file, dynamic_file_column_names

for indicator in range(24): 

    err = np.median(np.abs(diff[:, :, indicator]))

    print( err , dynamic_file_column_names[indicator])
diff = (ypred_all-y.reshape((y.shape[0], nday_test, nvalues)))/ypred_all

for indicator in range(24): 

    err = np.median(np.abs(diff[:, :, indicator]))

    print( err , dynamic_file_column_names[indicator])
## we show some departments, not all of the test set, for clarity

Nshow=15



import matplotlib.cm as cm

Ncolors = Nshow+1

gradient = cm.jet( np.linspace(0.0, 1.0, Ncolors+1 ) )

# color = tuple(gradient[dep])



for dep in range(Nshow):

    color = tuple(gradient[dep])

    plt.figure(1)

    plt.semilogy(ytrue [dep], ls='-', lw=3, color=color, label= "true")

    plt.plot(ypred [dep], ls=':', lw=2, color=color, label= "predicted")

    

    plt.figure(2)

    plt.loglog(ytrue [dep], ypred [dep], ls='', marker='x', markersize=5, color=color)

plt.figure(2)

plt.xlabel("ytrue")

plt.ylabel("ypred")

    