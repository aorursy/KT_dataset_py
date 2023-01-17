#different imports

import numpy as np 

import pandas as pd 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt

from scipy.stats import norm

import matplotlib.mlab as mlab

import seaborn as sns

import bokeh

import keras

import sklearn



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/properties_2016.csv')

df1 = pd.read_csv('../input/train_2016_v2.csv')

#merging the data

dfmerge =df.merge(df1, on='parcelid', how ='left')

dfmerge =dfmerge[np.isfinite(dfmerge['logerror'])]

dfmerge.head(n=10)



#training data split

trainx = dfmerge.drop(['logerror'],axis=1,inplace=False)

trainy= dfmerge['logerror']



#we need test data too#



#the amount of NAs in each of the variables#

#this is a very large chunk#

nan = dfmerge.isnull().sum()

nan
#seeing the missing count in chart form#

nan_sorted = nan.sort_values(ascending=False).to_frame().reset_index()

nan_sorted.columns = ['Column', 'Number of NaNs']



fig, ax = plt.subplots(figsize=(12, 25))

sns.barplot(x="Number of NaNs", y="Column", data=nan_sorted, color='orange', ax=ax);

ax.set(xlabel="Number of NaNs", ylabel="", title="Total Nimber of NaNs in each column");



#Variables that are numeric with many missing values we should ignore
#distribution plots of variables#

sns.set(style="white", color_codes=True)

sns.jointplot(x='bathroomcnt', y='logerror', data=dfmerge, size=10)

sns.jointplot(x='basementsqft', y='logerror', data=dfmerge, size=10)

sns.jointplot(x='bedroomcnt', y='logerror', data=dfmerge, size=10)

sns.jointplot(x='calculatedbathnbr', y='logerror', data=dfmerge, size=10)

sns.jointplot(x='finishedfloor1squarefeet', y='logerror', data=dfmerge, size=10)

sns.jointplot(x='numberofstories', y='logerror', data=dfmerge, size=10)
#correlation plot#

import seaborn as sns

#Identify numerical columns to produce a heatmap

catcols = ['calculatedfinishedsquarefeet','finishedsquarefeet12',

           'garagetotalsqft','structuretaxvaluedollarcnt','taxvaluedollarcnt',

           'taxamount','logerror']



numcols = [x for x in dfmerge.columns if x in catcols]



plt.figure(figsize = (12,8))

sns.heatmap(data=dfmerge[numcols].corr())

plt.show()

plt.gcf().clear()



#correlation values of the individual variables#

df = pd.DataFrame(dfmerge, columns =['calculatedfinishedsquarefeet','finishedsquarefeet12',

           'garagetotalsqft','structuretaxvaluedollarcnt','taxvaluedollarcnt',

           'taxamount','logerror'])

df.corr()

#plot the distribution histogram #

median_values = dfmerge.median(axis=0)

train_df_new = dfmerge.fillna(median_values, inplace=True)



plt.figure(figsize=(12,8))

sns.distplot(dfmerge.logerror.values, bins=30, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.show()
#basementsqft,bathroomcnt, bedroomcnt,calculatedbathnbr,finishedfloor1squarefeet,numberofstories



sns.distplot(train_df_new.bathroomcnt.values, bins=15, kde=False)

plt.xlabel('bathroom', fontsize=12)

plt.show()



sns.distplot(train_df_new.finishedfloor1squarefeet.values, bins=20, kde=False)

plt.xlabel('finishedfloor1squarefeet', fontsize=12)

plt.show()



sns.distplot(train_df_new.numberofstories.values, bins=20, kde=False)

plt.xlabel('numberofstories', fontsize=12)

plt.show()



sns.distplot(train_df_new.basementsqft.values, bins=40, kde=False)

plt.xlabel('basementsqft', fontsize=12)

plt.show()

#understand the types of data we have

dfmerge.dtypes

#we can drop the objects as a easy way to see how we perform#

#only three objects#

dfmerge = dfmerge.select_dtypes(exclude=[object])

trainx = dfmerge.drop(['logerror'],axis=1,inplace=False)

trainy= dfmerge['logerror']
dfmerge.dtypes
bins = [1885,1953,1970,1987,2017]

names = ['Cat1','Cat2','Cat3','Cat4']

categories = pd.cut(dfmerge['yearbuilt'],bins,labels = names)

dfmerge['categories'] = pd.cut(dfmerge['yearbuilt'],bins,labels = names)

dfmerge['scoresbinned'] = pd.cut(dfmerge['yearbuilt'],bins)

#variables selection#

#data normalization#

#convert categorical - numerical#

df_x = dfmerge[['calculatedfinishedsquarefeet','fullbathcnt','parcelid']]

print(df_x)
#regression forests - decision trees#

#encoding then this should work#

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



parameters = {'n_estimators':[5,10,15],'n_jobs':[-1],'oob_score':[False]}  # this can be extended

model = RandomForestRegressor()

grid = GridSearchCV(model,param_grid=parameters,scoring='mean_absolute_error',cv=3)  

grid.fit(trainx,trainy)





cv_results = pd.DataFrame(grid.cv_results_)

print(cv_results[["param_n_estimators","mean_test_score","std_test_score"]])



feat_imps = grid.best_estimator_.feature_importances_

fi = pd.DataFrame.from_dict({'feat':trainx.columns,'imp':feat_imps})

fi.set_index('feat',inplace=True,drop=True)

fi = fi.sort_values('imp',ascending=False)

fi.head(20).plot.bar()







#feature importances#

#https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn
#Regression

#we need to perform feature selection here#

#making model without too many variables - R^2 ?

#be careful of multicollinearity

#Random forest regressor ? - Deciscion Tree



from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(df_x, y_train)

pred_train= lm.predict(df_x)

MSEfull = np.mean(y_train - lm.predict(df_x)**2)

print (MSEfull)
# fill in with -1.0#

#y_training data 

dfmerge = dfmerge.fillna(-1.0)

y_train = dfmerge['logerror']
#x_training data

x_train = dfmerge.drop(['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc','propertycountylandusecode','fireplacecnt', 'fireplaceflag'], axis=1)

list(x_train)

list(y_train)
y_mean = np.mean(y_train)

print(x_train.shape, y_train.shape)
from sklearn.preprocessing import Imputer

imputer= Imputer()

imputer.fit(x_train.iloc[:, :])

x_train = imputer.transform(x_train.iloc[:, :])

imputer.fit(x_test.iloc[:, :])

x_test = imputer.transform(x_test.iloc[:, :])



#df.head(n=10)

#print(df.shape)

#print(df1.shape)

dfmerge =df.merge(df1, on='parcelid', how ='left')

dfmerge = dfmerge[np.isfinite(dfmerge['logerror'])]

dfmerge.head(n=10)

print(dfmerge.shape)

#Segment into X and Y variables#

from keras.models import Sequential

from keras.layers import Dense

import numpy

dfsamplex= dfmerge[['bathroomcnt','bedroomcnt']]

dfsamplex.head(n=10)

arrayx=dfsamplex.values

dfsampley = dfmerge[['logerror']]

arrayy=dfsampley.values

dfsampley.head(n=10)

dfsampley.to_csv('out.csv', sep='\t')





#make dataframe into index#

print(dfsampley.shape)



#modeling with Keras#

# Regression vs Keras - Deep Learning



import numpy

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from keras.constraints import maxnorm

from keras.layers import Dropout
def createmodel(neurons = 1):

    model = Sequential()

    model.add(Dense(neurons, input_dim=2, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(4)))

    model.add(Dropout(0.2))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model
# create model

model = KerasRegressor(build_fn=createmodel, epochs=100, batch_size=5, verbose=0)
#standard scalar#

estimators=[]

estimators.append(('standardise', StandardScaler()))

estimators.append(('mlp', KerasRegressor(build_fn= createmodel,

nb_epoch=100, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

kfold = KFold( n_splits=10)

results = cross_val_score(pipeline, X, Y, cv=kfold)

print((results.mean(), results.std()))

neurons = [5, 10, 15, 20, 25, 30]

param_grid = dict(neurons=neurons)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

grid_result = grid.fit(arrayx, arrayy)



bins = [1885,1953,1970,1987,2017]

names = ['Cat1','Cat2','Cat3','Cat4']

categories = pd.cut(dfmerge['yearbuilt'],bins,labels = names)

dfmerge['categories'] = pd.cut(dfmerge['yearbuilt'],bins,labels = names)

dfmerge['scoresbinned'] = pd.cut(dfmerge['yearbuilt'],bins)
