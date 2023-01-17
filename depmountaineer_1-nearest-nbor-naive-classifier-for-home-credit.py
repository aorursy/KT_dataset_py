# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
######################
#Load Test Data
######################
dftest = pd.read_csv('../input/application_test.csv')#.head(1000) #reduce for testing and development
######################
#Load Training Data
######################
dftrain = pd.read_csv('../input/application_train.csv')#.head(3000) #reduce for testing and development
######################
#Remove NA values in numerical data by replacing them with means
######################
dftest=dftest.fillna(dftest.mean())
dftrain=dftrain.fillna(dftrain.mean())
######################
#Convert categorical values in test data to numeric
######################
cat_columns = dftest.select_dtypes(['object']).columns
dftest[cat_columns]=dftest[cat_columns].apply(lambda x:x.astype('category'))
dftest[cat_columns]=dftest[cat_columns].apply(lambda x:x.cat.codes)
######################
#Convert categorical values in training data to numeric
######################
cat_columns = dftrain.select_dtypes(['object']).columns
dftrain[cat_columns]=dftrain[cat_columns].apply(lambda x:x.astype('category'))
dftrain[cat_columns]=dftrain[cat_columns].apply(lambda x:x.cat.codes)
######################
#Prepair training data for classifier by separating into features and target
######################
col = dftrain.columns
xcol=col.drop('TARGET')
x=dftrain[xcol]
y=dftrain['TARGET']

######################
# Run classifier.  Since outputs are 0 and 1, just test counts
######################

clf = KNeighborsClassifier(n_neighbors=1)
clf = clf.fit(x,y)
ytrainpred = clf.predict(x)
print("count predicted",sum(ytrainpred))
print("count actual",sum(y))
######################
#This one seems to overfit.  Also, only outputs 0 or 1, no probabilities
######################

######################
#Now run the classifier on the test data
######################

xx = dftest[xcol]
yy = clf.predict(xx)
print(sum(yy))
######################
#Convert result to data frame of the type the submission requires
######################

dfpred = pd.DataFrame([a for a in zip([x for x in xx.SK_ID_CURR], yy)], columns=['SK_ID_CURR','TARGET'])
dfpred.head()
######################
#Output the result
######################

dfpred.to_csv('submission.csv',index=False)