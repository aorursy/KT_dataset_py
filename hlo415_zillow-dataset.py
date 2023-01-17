# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

dfmerge =df.merge(df1, on='parcelid', how ='left')

dfmerge =dfmerge[np.isfinite(dfmerge['logerror'])]

dfmerge.head(n=10)

from sklearn.preprocessing import Imputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

import random

from sklearn.preprocessing import Imputer

import plotly.plotly as py

import matplotlib.mlab as mlab

from matplotlib import pyplot



##new code##

#trainx1 = dfmerge.drop(['logerror','transactiondate'],axis=1,inplace=False)



trainx = dfmerge.drop(['logerror','parcelid','architecturalstyletypeid','basementsqft'

                      ,'buildingclasstypeid','calculatedbathnbr','decktypeid','finishedfloor1squarefeet'

                      ,'finishedsquarefeet13'

                      ,'finishedsquarefeet15','finishedsquarefeet50','finishedsquarefeet6'

                      ,'fips','fullbathcnt','garagecarcnt','hashottuborspa'

                      ,'latitude','longitude','lotsizesquarefeet','poolsizesum'

                      ,'pooltypeid10','pooltypeid2','pooltypeid7','propertycountylandusecode'

                      ,'propertylandusetypeid','propertyzoningdesc','rawcensustractandblock'

                      ,'regionidzip','storytypeid','threequarterbathnbr','typeconstructiontypeid',

                      'yardbuildingsqft17','yardbuildingsqft26','fireplaceflag','assessmentyear'

                      ,'landtaxvaluedollarcnt','taxdelinquencyflag','taxdelinquencyyear'

                      ,'censustractandblock','transactiondate'],axis=1,inplace=False)



imputer1 = Imputer(missing_values='NaN', strategy ='mean', axis=0)

imputed_df1 = pd.DataFrame(imputer1.fit_transform(trainx))

print(imputed_df1)



trainy= dfmerge['logerror']



parameters = {'n_estimators':[5,10,15],'n_jobs':[-1],'oob_score':[False]}

model = RandomForestRegressor()

model.fit(imputed_df1,trainy)

#grid = GridSearchCV(model,param_grid=parameters,scoring='neg_mean_absolute_error',cv=3)

#grid.fit(imputed_df1,trainy)



#cv_results = pd.DataFrame(grid.cv_results_)

#print(cv_results[["param_n_estimators","mean_test_score","std_test_score"]])

print(model.feature_importances_)

pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)

pyplot.show()



#feat_imps = grid.best_estimator_.feature_importances_

#fi = pd.DataFrame.from_dict({'feat':imputed_df1.columns,'imp':feat_imps})

#fi.plot.bar()

#fi.set_index('feat',inplace=True,drop=True)

#fi = fi.sort_values('imp',ascending=False)

#fi.head(20).plot.bar()

#modeling with Keras#

# Regression vs Keras - Deep Learning

import numpy

from sklearn.model_selection import GridSearchCV

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from keras.constraints import maxnorm

from keras.layers import Dropout
def create_model(neurons = 1):

    model = Sequential()

    model.add(Dense(neurons, input_dim=20, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(4)))

    model.add(Dropout(0.2))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model
#the amount of NAs in each of the variables#

#this is a very large chunk#

nan = dfmerge.isnull().sum()

nan







#summary statistics#

#From categorical#

nan_sorted = nan.sort_values(ascending=False).to_frame().reset_index()

nan_sorted.columns = ['Column', 'Number of NaNs']



fig, ax = plt.subplots(figsize=(12, 25))

sns.barplot(x="Number of NaNs", y="Column", data=nan_sorted, color='Sienna', ax=ax);

ax.set(xlabel="Number of NaNs", ylabel="", title="Total Number of NaNs in each column");
#dfmergetwo= pd.get_dummies('airconditioningtypeid','architecturalstyletypeid')#,#'architecturalstyletypeid'

#print (dfmergetwo)
color = sns.color_palette()

#creating normal distributions#

dfmerge['transaction_month'] = dfmerge['transactiondate']

cnt_srs = dfmerge['transaction_month'].value_counts()

plt.figure(figsize=(12,6))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])

plt.xticks(rotation='vertical')

plt.xlabel('Month of transaction', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()

#print(dfmerge.describe())

dfmerge.hist()
import seaborn as sns

#Identify numerical columns to produce a heatmap

catcols = ['basementsqft','bathroomcnt','bedroomcnt','calculatedbathnbr','finishedfloor1squarefeet','calculatedfinishedsquarefeet','finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15','finishedsquarefeet50','finishedsquarefeet6','fireplacecnt','fullbathcnt','garagecarcnt','garagetotalsqft','lotsizesquarefeet','poolsizesum','roomcnt','threequarterbathnbr','unitcnt','yardbuildingsqft17','yardbuildingsqft26','numberofstories','structuretaxvaluedollarcnt','taxvaluedollarcnt','landtaxvaluedollarcnt','taxamount']

numcols = [x for x in dfmerge.columns if x in catcols]

plt.figure(figsize = (12,8))

sns.heatmap(data=dfmerge[numcols].corr())

plt.show()

plt.gcf().clear()



#correlation plot#

import seaborn as sns

#Identify numerical columns to produce a heatmap

catcols = ['airconditioningtypeid','architecturalstyletypeid','buildingqualitytypeid',

'buildingclasstypeid','decktypeid','fips','hashottuborspa','heatingorsystemtypeid',

'pooltypeid10','pooltypeid2','pooltypeid7','propertycountylandusecode',

'propertylandusetypeid','propertyzoningdesc','rawcensustractandblock',

'regionidcity','regionidcounty','regionidneighborhood','regionidzip',

'storytypeid','typeconstructiontypeid','yearbuilt','taxdelinquencyflag']



numcols = [x for x in dfmerge.columns if x not in catcols]



plt.figure(figsize = (12,8))

sns.heatmap(data=dfmerge[numcols].corr())

plt.show()

plt.gcf().clear()
#variables selection#

#data normalization#

#convert categorical - numerical#



df_x = dfmerge[['calculatedfinishedsquarefeet','fullbathcnt','parcelid']]

print(df_x)
#Regression





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
def create_model(neurons = 1):

    model = Sequential()

    model.add(Dense(neurons, input_dim=2, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(4)))

    model.add(Dropout(0.2))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


# create model

model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=5, verbose=0)
neurons = [5, 10, 15, 20, 25, 30]

param_grid = dict(neurons=neurons)

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

grid_result = grid.fit(imputed_df1, trainy)