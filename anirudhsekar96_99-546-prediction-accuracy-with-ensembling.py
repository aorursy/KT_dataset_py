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
import numpy as np

import pandas as pd
import sklearn
data = pd.read_csv("../input/glass.csv")

data.head()
data.fillna(0,inplace=True)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

ss.fit_transform(data)
X = data.drop(['Type'],axis=1).values
from sklearn.decomposition import PCA

pca = PCA()

pca.fit_transform(X)
y = data.Type
import time

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_predict,cross_val_score

from sklearn.metrics import mean_squared_error



start_time = time.time()



clf_RF_L1 = RandomForestClassifier(verbose=2)

#clf_GBC.fit(X_train,y_train)



print("Classifier Created")



score_RF_L1 = cross_val_score(clf_RF_L1,X,y,cv=16,scoring= 'mean_absolute_error',n_jobs=-1)

print("Score Created")

print(score_RF_L1)





y_pred_RF_L1 = cross_val_predict(clf_RF_L1,X,y,cv=16,n_jobs=-1)

print("Training data for next level Created")





print("--- %s seconds ---" % (time.time() - start_time))

#print score

print(100-mean_absolute_error(y,y_pred_RF_L1))
from sklearn.neighbors import KNeighborsClassifier

start_time = time.time()

start_time = time.time()

clf_KN_L1 = KNeighborsClassifier()

#clf_GBC.fit(X_train,y_train)



print("Classifier Created")



score_KN_L1 = cross_val_score(clf_KN_L1,X,y,cv=16,scoring= 'mean_absolute_error',n_jobs=-1)

print("Score Created")

print(score_KN_L1)



y_pred_KN_L1 = cross_val_predict(clf_KN_L1,X,y,cv=16,n_jobs=-1)

print("Training data for next level Created")





print("--- %s seconds ---" % (time.time() - start_time))

#print score

print(100-mean_absolute_error(y,y_pred_KN_L1))
from sklearn.naive_bayes import GaussianNB

clf_NB_L1 = GaussianNB()

#clf_GBC.fit(X_train,y_train)

start_time = time.time()

print("Classifier Created")



score_NB_L1 = cross_val_score(clf_NB_L1,X,y,cv=16,scoring= 'mean_absolute_error',n_jobs=-1)

print("Score Created")

print(score_NB_L1)



y_pred_NB_L1 = cross_val_predict(clf_NB_L1,X,y,cv=16,n_jobs=-1)

print("Training data for next level Created")





print("--- %s seconds ---" % (time.time() - start_time))

#print score

print(100-mean_absolute_error(y,y_pred_NB_L1))
from sklearn.svm import SVC

start_time = time.time()

clf_SVC_L1 = SVC()

#clf_GBC.fit(X_train,y_train)



print("Classifier Created")



score_SVC_L1 = cross_val_score(clf_SVC_L1,X,y,cv=16,scoring= 'mean_absolute_error',n_jobs=-1)

print("Score Created")

print(score_SVC_L1)



y_pred_SVC_L1 = cross_val_predict(clf_SVC_L1,X,y,cv=16,n_jobs=-1)

print("Training data for next level Created")





print("--- %s seconds ---" % (time.time() - start_time))

#print score

print(100-mean_absolute_error(y,y_pred_SVC_L1))
from sklearn.neural_network import MLPClassifier

start_time = time.time()

clf_MLP_L1 = MLPClassifier()

#clf_GBC.fit(X_train,y_train)



print("Classifier Created")



score_MLP_L1 = cross_val_score(clf_MLP_L1,X,y,cv=16,scoring= 'mean_absolute_error',n_jobs=-1)

print("Score Created")

print(score_MLP_L1)



y_pred_MLP_L1 = cross_val_predict(clf_MLP_L1,X,y,cv=16,n_jobs=-1)

print("Training data for next level Created")





print("--- %s seconds ---" % (time.time() - start_time))

#print score

print(100-mean_absolute_error(y,y_pred_MLP_L1))
from sklearn.linear_model import SGDClassifier

start_time = time.time()

clf_SGD_L1 = SGDClassifier()

#clf_GBC.fit(X_train,y_train)



print("Classifier Created")



score_SGD_L1 = cross_val_score(clf_SGD_L1,X,y,cv=16,scoring= 'mean_absolute_error',n_jobs=-1)

print("Score Created")

print(score_SGD_L1)



y_pred_SGD_L1 = cross_val_predict(clf_SGD_L1,X,y,cv=16,n_jobs=-1)

print("Training data for next level Created")





print("--- %s seconds ---" % (time.time() - start_time))

#print score

print(100-mean_absolute_error(y,y_pred_SGD_L1))
from sklearn.linear_model import RidgeClassifier

start_time = time.time()

clf_RC_L1 = RidgeClassifier()

#clf_GBC.fit(X_train,y_train)



print("Classifier Created")



score_RC_L1 = cross_val_score(clf_RC_L1,X,y,cv=16,scoring= 'mean_absolute_error',n_jobs=-1)

print("Score Created")

print(score_RC_L1)



y_pred_RC_L1 = cross_val_predict(clf_RC_L1,X,y,cv=16,n_jobs=-1)

print("Training data for next level Created")





print("--- %s seconds ---" % (time.time() - start_time))

#print score

print(100-mean_absolute_error(y,y_pred_RC_L1))

print(100-mean_absolute_error(y,y_pred_RF_L1))

print(100-mean_absolute_error(y,y_pred_KN_L1))

print(100-mean_absolute_error(y,y_pred_NB_L1))

print(100-mean_absolute_error(y,y_pred_MLP_L1))

print(100-mean_absolute_error(y,y_pred_SVC_L1))

print(100-mean_absolute_error(y,y_pred_SGD_L1))

print(100-mean_absolute_error(y,y_pred_RC_L1))
print(100-mean_squared_error(y,y_pred_RF_L1))

print(100-mean_squared_error(y,y_pred_KN_L1))

print(100-mean_squared_error(y,y_pred_NB_L1))

print(100-mean_squared_error(y,y_pred_MLP_L1))

print(100-mean_squared_error(y,y_pred_SVC_L1))

print(100-mean_squared_error(y,y_pred_SGD_L1))

print(100-mean_squared_error(y,y_pred_RC_L1))
print (np.correlate(y,y_pred_RF_L1))

print (np.correlate(y,y_pred_KN_L1))

print (np.correlate(y,y_pred_NB_L1))

print (np.correlate(y,y_pred_MLP_L1))

print (np.correlate(y,y_pred_SVC_L1))

print (np.correlate(y,y_pred_SGD_L1))

print (np.correlate(y,y_pred_RC_L1))
k = {'rf':y_pred_RF_L1,'kn':y_pred_KN_L1,'nb':y_pred_NB_L1,'mlp':y_pred_MLP_L1,'svc':y_pred_SVC_L1,'sgd':y_pred_SGD_L1,'rc':y_pred_RC_L1}

df = pd.DataFrame(data=k)

arr2 =  np.concatenate((data.drop(['Type'],axis=1).values,df.values),axis=1)

pca2 = PCA()

pca2.fit_transform(arr2)


from xgboost import XGBClassifier

start_time = time.time()



clf_XG_L2_f = XGBClassifier()

#clf_GBC.fit(X_train,y_train)



print("Classifier Created")



score_XG_L2_f = cross_val_score(clf_XG_L2_f,arr2,y,cv=16,scoring= 'mean_absolute_error',n_jobs=-1)

print("Score Created")

print(score_XG_L2_f)





y_pred_XG_L2_f = cross_val_predict(clf_XG_L2_f,arr2,y,cv=16,n_jobs=-1)

print("Training data for next level Created")





print("--- %s seconds ---" % (time.time() - start_time))

#print score

print(100-mean_absolute_error(y,y_pred_XG_L2_f))
min_x = ['nb', 'rc']

arr = np.concatenate((data.drop(['Type'],axis=1).values,df[min_x].values),axis=1)

pca3 = PCA()

pca3.fit_transform(arr)



from xgboost import XGBClassifier

start_time = time.time()



clf_XG_L2 = XGBClassifier()

#clf_GBC.fit(X_train,y_train)



print("Classifier Created")



score_XG_L2 = cross_val_score(clf_XG_L2,arr,y,cv=16,scoring= 'mean_absolute_error',n_jobs=-1)

print("Score Created")

print(score_XG_L2)





y_pred_XG_L2 = cross_val_predict(clf_XG_L2,arr,y,cv=16,n_jobs=-1)

print("Training data for next level Created")





print("--- %s seconds ---" % (time.time() - start_time))

#print score

print(100-mean_absolute_error(y,y_pred_XG_L2))