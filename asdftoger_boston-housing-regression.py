# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sys

import os

from time import time

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
#train.info()
#Threshold dropping of features, 

t = 0.8

train = train.dropna(thresh = t * len(train),axis = 1)

#train.info()
#train.info()
Y=train.iloc[:,-1]

del train['SalePrice']

del train['Id']

sns.distplot(Y)

plt.title('Distribution of Sale Prices')

plt.show()
train_numeric=train.select_dtypes(include=['int64','float64'])

#Use this for submission

cols_num=train_numeric.columns

#train_numeric.info()
train_numeric=train_numeric.fillna(train_numeric.median())

#train_numeric.info()
#Extracts the features with the Ex-Gd-...-NA

train_object=train.select_dtypes(include=['object'])

cols=train_object.columns

print(cols)

qual=[]

for feat in cols:

    if train_object[feat].str.contains('Ex').any():

        qual.append(feat)

        pass

    pass

print(qual)

train_qual=train_object[qual]

qual_letter=['Ex','Gd','TA','Fa','Po','NA']

qual_map={qual_letter[i]:i+1 for i in range(len(qual_letter))}

train_qual=train_qual.replace(qual_map)

train_qual=train_qual.fillna(train_qual.median())

train_qual.info()
X=pd.concat([train_numeric,train_qual],1)

#X.info()
trainX,testX,trainY,testY = train_test_split(X,Y,test_size = 0.1,random_state = 1)
#A few useful functions

def RMSE_CV(model,X,Y):

  scoring = 'neg_mean_squared_error'

  rmse_score = np.sqrt(-cross_val_score(model,X,Y,cv = 5,scoring=scoring))

  

  return rmse_score



def Kaggle_score(predY,testY):

  assert(len(predY)==len(testY))

  n = len(predY)

  predY = np.log(predY)

  testY = np.log(testY)

  t = np.sum((predY-testY)**2)

#  print(t)

  score = np.sqrt(t/n)

  return score

  

def report(results, n_top=2):

  for i in range(1, n_top + 1):

    candidates = np.flatnonzero(results['rank_test_score'] == i)

    for candidate in candidates:

      print("Model with rank: {0}".format(i))

      print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

            results['mean_test_score'][candidate],

            results['std_test_score'][candidate]))

      print("Parameters: {0}".format(results['params'][candidate]))

      print("")

            

  pass
#Random forest regression, commented out

regress = RandomForestRegressor(n_estimators=100)

regress.fit(trainX,trainY)



param_dist = {

              'max_depth':[i for i in range(10,50,4)]+[None],

              "max_features": ['sqrt','log2']

              }

#Parameters: {'max_depth': 22, 'max_features': 'sqrt'}

regress_grid = GridSearchCV(regress, param_grid=param_dist,cv=5,scoring = 'neg_mean_squared_error',verbose=1)

print('Starting GridSearchCV')

start = time()

regress_grid.fit(X,Y)

print("GridSearchCV took %.2f seconds." % (time() - start))

report(regress_grid.cv_results_)



regress.set_params(**regress_grid.best_params_)

regress.fit(trainX,trainY)
predY=regress.predict(testX)

score = Kaggle_score(predY,testY)

print('Expected Kaggle score(Lower is better): ', score)

feats=pd.DataFrame(regress.feature_importances_,index=trainX.columns)

sort_feats=feats.sort_values(0,ascending=True)

sort_feats.plot(kind='barh',title='Feature importances')

plt.tight_layout()

plt.show()
print(trainX.columns)

print(test.columns)

#Fill NA values with train values NOT Sub or test

#Extract numeric

sub_numeric=test[cols_num]

sub_numeric=sub_numeric.fillna(train_numeric.median())

#Extract quality

sub_qual=test[qual]

sub_qual=sub_qual.replace(qual_map)

sub_qual=sub_qual.fillna(train_qual.median())
subX=pd.concat([sub_numeric,sub_qual],1)

subX.info()
subY = regress.predict(subX)

subY[subY<=0]= 1

#Exporting to CSV,DONT DELETE

print('Exporting to CSV')

EXP = pd.DataFrame(subY,index = [i for i in range(1461,2919+1)],columns=['SalePrice'])

EXP.index.name = 'Id'

filename = 'Boston_RFRAllD.csv'

EXP.to_csv(filename)

print('Finished, fname: ',filename )