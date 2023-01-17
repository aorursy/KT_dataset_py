from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
testLength = 14 #days
path_to_inputs = '/kaggle/input/covid19-granular-demographics-and-times-series/'

for dirname, _, filenames in os.walk(path_to_inputs):

    for filename in filenames:

        print(os.path.join(dirname, filename))
input_filename1 = 'departments_static_data.csv'

# input_filename1 = 'departments_static_data_divByPop.csv' 

# input_filename1 = 'departments_static_data_divBySubPop.csv' ## this file may contain the same pre-proc done in the cell below

df1 = pd.read_csv(path_to_inputs+input_filename1, delimiter=',')

# df1 = df1.drop(columns=["regcode"])

df1.dataframeName = input_filename1

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')

df1.head(3)
input_filename2 = "TIME_serie_donnees-hospitalieres-Total_amout_of_deaths_at_the_hospital.csv"

dftime1 = pd.read_csv(path_to_inputs+input_filename2, delimiter=',')

dftime1.dataframeName = input_filename2

dftime1 = dftime1.drop(columns=["Unnamed: 0", "DEPARTMENT"])

dftime1 = dftime1.drop(index=  dftime1.index[ dftime1.code=='976' ]  ) ## we delete Mayotte, as it does not have enough data available in the statics data set

nRow, nCol = dftime1.shape

print(f'There are {nRow} rows and {nCol} columns')

dftime1.head(3)
## we take a look at the possible prefixes (starting of column names)

prefixes = []

for col in df1.columns:

    prefixes.append(col[:4])

print("all column names start with one of these words: ", set(prefixes))



def def_pop_cols(df1):

    ## get all columns which relate to population

    pop_cols = []

    for col in df1.columns[2:]:

        if col[:3] == "Pop":

            pop_cols.append(col)

    return pop_cols

pop_cols = def_pop_cols(df1)

print(pop_cols[:5])



def get_sex_age(col):

    ## reads the column names and extract the tags are integers

    if "sex=H" in col:

        sex=1

    elif "sex=F" in col:

        sex=2

    elif "sex=all" in col:

        sex=0

    else:

        print("weird: ", col)

        sex=0



    if "age" in col:

        if "agemin" in col:

            agemin = int(col.split("agemin=")[1].split("_")[0])

            if "agemax" in col:

                agemax = int(col.split("agemax=")[1].split("_")[0])

            else:

                print("VERY VERY weird !")

        elif "age=all" in col:

            agemin=0

            agemax=150

        else:

            print("weird: ", col)

            agemin=0

            agemax=150

    else:

        print("a little but not really weird: ", col)

        agemin=0

        agemax=150

    return sex, agemin, agemax

    

def get_sub_pop_corresponding_to_col(col, pop_cols):

    ## returns the corresponding sub-population (correct denominator) of any column.

    ## If no exact match is found, returns the global population

    sex, agemin, agemax = get_sex_age(col)

    token = False

    for pop_col in pop_cols:

        Psex, Pagemin, Pagemax = get_sex_age(pop_col)

        if Psex == sex :

            if Pagemin == agemin:

                if Pagemax == agemax:

                    pop = df1[pop_col] ## we use the appropriate sub-pop as denominator

                    token = True

                    break



    if token == True:

        pass

    else: 

        ## no match: we divide by the total pop of the dept (all sex, all age)

        pop = df1["Pop_sex=all_age=all_Population"]

    return pop

    

pop_cols = def_pop_cols(df1)



df3 = df1.copy() # pd.DataFrame()

for col in df1.columns[3:]:

    ## population columns are themselves divided by the global pop.

    if col[:3] == "Pop":

        pop = df1["Pop_sex=all_age=all_Population"]

        df3[col] = df1[col] / pop

        df3 = df3.rename(columns={col: "Rate"+col[3:]})

    

    ## Nbre columns are divided by the correct sub-pop.

    if col[:4] == "Nbre" :

        pop = get_sub_pop_corresponding_to_col(col, pop_cols)          

        df3[col] = df1[col] / pop

        df3 = df3.rename(columns={col: "Rate"+col[4:]})

        

df3.head()
dftime2 = dftime1.copy() # pd.DataFrame()

pop = df1["Pop_sex=all_age=all_Population"]

for col in dftime1.columns[1:]:

    dftime2[col] = dftime1[col] / pop

dftime2.head(3)
df4 = df3.fillna(df3.mean())

dftime4 = dftime2.fillna(dftime2.mean())



# df4 = df3.fillna( 0 ) # an other possibility
df4 = df4.sort_values(by=['code'])

df5 = df4.drop(columns=["code", "regcode"])
pop = np.array(df5["Pop_sex=all_age=all_Population"])

pop.sum() # French population (total)
dftime4 = dftime4.sort_values(by=['code'])

dftime5 = dftime4.drop(columns=["code"])
## pandas-style merging

df = pd.merge(df4, dftime4, on='code', how='outer')

print(df4.shape, dftime4.shape, df.shape)

pd_Xtrain = np.array(df.iloc[:, :-testLength ])

pd_ytrain = np.array(df.iloc[:,  -testLength:])



codes = pd_Xtrain[:, 0] ## original index

regcodes = pd_Xtrain[:, 1] 



pd_Xtrain = pd_Xtrain[:, 2:] ## only numerical data

# Xtrain[:,-1]
## numpy-style merging

np_Xstatic = np.array(df5)

np_Xdynamic = np.array(dftime5)

# np_testLength = np_Xstatic.shape[1] + np_Xdynamic.shape[1] - np_testLength

np_merge = np.concatenate( (np_Xstatic, np_Xdynamic), axis=1)

print(np_Xstatic.shape, np_Xdynamic.shape, np_merge.shape)

np_Xtrain = np_merge[:, :-testLength]

np_ytrain = np_merge[:,  -testLength:]

# np_Xtrain[:,-1]
(pd_Xtrain-np_Xtrain).std(), (pd_ytrain-np_ytrain).std() 
Xs = np.array(df5) ## static data

dynamicData = np.array(dftime5) ## dynamic data (all of it)

Xd  = dynamicData[:, :-testLength ].copy() ## dynamic data used as features 

yd  = dynamicData[:,  -testLength:].copy() ## dynamic data used as ground truth labels/values (to be predicted)
mean_Xs = Xs.mean(axis=0)

std_Xs  = Xs.std (axis=0)

scaled_Xs = (Xs-mean_Xs)/std_Xs



mean_Xd = Xd[:,-3:].mean()  ## we use the last 3 days of data as 

std_Xd  = Xd[:,-3:].std()

scaled_Xd = (Xd-mean_Xd)/std_Xd



scaled_yd = (yd-mean_Xd)/std_Xd ## we cannot know in advance the scaling factor of future data !!
X = np.concatenate( (scaled_Xs, scaled_Xd), axis=1)

y = scaled_yd.copy()

Xstatic  = scaled_Xs.copy()

Xdynamic = scaled_Xd.copy()
import sklearn.decomposition

Xstatic.shape
def quick_look_at_pca(X, n_components):

    pca = sklearn.decomposition.PCA(n_components=n_components)

    pca.fit(X)

    Xp = pca.transform(X)

    plt.semilogy(pca.explained_variance_ratio_[:10], label="explained_variance_ratio")

    plt.legend()

    plt.xlabel("n_components (PCA)")

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

plt.semilogy(recos[:,0],recos[:,1], marker='o', label="reconstruction error (MAE)")

plt.plot(recos[:,0],recos[:,2], marker='o', label="noise variance")

plt.xlabel("n_components (PCA)")

plt.legend()
n_components = 7

Xp = quick_look_at_pca(Xstatic, n_components)
import sklearn.linear_model

from sklearn.model_selection import KFold
def linreg_score(ypred,ytrue):

    ## our score function ##

    defaultVal = np.maximum(ytrue, np.ones(ytrue.shape)*np.mean(ytrue) )

    return (abs(ypred-ytrue)/defaultVal  ).mean()



def try_linreg(X,y,n_splits):    

    linreg = sklearn.linear_model.LinearRegression(normalize=False)

    

    kf = KFold(n_splits=n_splits)

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]

        linreg.fit(X_train,y_train)

        print("scores: (train, test) ", linreg_score(linreg.predict(X_train),y_train), \

              linreg_score(linreg.predict(X_test),y_test) ) 
X = np.concatenate( (scaled_Xs, scaled_Xd), axis=1)

y = scaled_yd.copy()

Xstatic  = scaled_Xs.copy()

Xdynamic = scaled_Xd.copy()



try_linreg(X,y,5)
n_components = 9

pca = sklearn.decomposition.PCA(n_components=n_components)

Xpca = pca.fit_transform(Xstatic)

X = np.concatenate( (Xpca, scaled_Xd), axis=1)

y = scaled_yd.copy()
try_linreg(X,y,5)
y.shape
## we re-build the data-set ##

n_components = 9

pca = sklearn.decomposition.PCA(n_components=n_components)

Xpca = pca.fit_transform(Xstatic)

X = np.concatenate( (Xpca, scaled_Xd), axis=1)

y = scaled_yd.copy()

pop = np.array(df5["Pop_sex=all_age=all_Population"])
def train_test_pop_split(X,y,pop,test_ratio, seed):

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

    y_train_pop = pop[train_indexes].reshape( (Nexamples-Ntest,1) )



    X_test = X[test_indexes]

    y_test = y[test_indexes]

    y_test_pop = pop[test_indexes].reshape( (Ntest,1) )



    return X_train, X_test, y_train, y_test, y_train_pop, y_test_pop
seed = 42

test_ratio=0.33

X_train, X_test, y_train, y_test, y_train_pop, y_test_pop = train_test_pop_split(X, y, pop, test_ratio, seed)



## model (cheap) ##

linreg = sklearn.linear_model.LinearRegression(normalize=False)

linreg.fit(X_train, y_train)



## predictions

ypred = linreg.predict(X_test)

ytrue = y_test.copy()



def raw_number(y, std_Xd, mean_Xd, pop):

    ## re-scaled predictions

    raw_number_y = (y*std_Xd+mean_Xd)*pop

    return raw_number_y



ypred = raw_number(ypred, std_Xd, mean_Xd, y_test_pop)

ytrue = raw_number(ytrue, std_Xd, mean_Xd, y_test_pop)
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

    