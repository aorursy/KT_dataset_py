# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

import matplotlib.pyplot as plt

from sklearn.cross_decomposition import PLSRegression,PLSSVD

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

from sklearn import model_selection

import os
data=pd.read_csv("../input/youtube-new/USvideos.csv")
print(data.head())
data.describe().T
data=data[["views","likes","dislikes","comment_count"]]
data.reset_index(drop=True,inplace=True)
print(data)
data=data.dropna()
data.info()
data.describe().T
y=data["comment_count"]
X=data[["views","likes","dislikes"]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
print("X_train",X_train.shape)

print("X_test",X_test.shape)

print("y_train",y_train.shape)

print("y_test",y_test.shape)
data.shape
pca=PCA()
X_reduced_train=pca.fit_transform(scale(X_train))
X_reduced_train[0:1,:]
np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)[0:5]
lm=LinearRegression()
pcr_model=lm.fit(X_reduced_train,y_train)
pcr_model.intercept_
pcr_model.coef_
y_pred=pcr_model.predict(X_reduced_train)
y_pred[0:5]
np.sqrt(mean_squared_error(y_train,y_pred))
data["views"].mean()
r2_score(y_train,y_pred) # we can explain 0.83% of data 
pca2=PCA()
X_reduced_test=pca2.fit_transform(scale(X_test))
y_pred=pcr_model.predict(X_reduced_test)
np.sqrt(mean_squared_error(y_test,y_pred))
cv_10=model_selection.KFold(n_splits=10,shuffle=True,random_state=1) #10k validation preparing
lm=LinearRegression()
RMSE=[]
for i in np.arange(1,X_reduced_train.shape[1]+1):

    score=np.sqrt(-1*model_selection.cross_val_score(lm,X_reduced_train[:,:i],

                                                     y_train.ravel(),

                                                     cv=cv_10,

                                                     scoring='neg_mean_squared_error').mean())

    RMSE.append(score)
plt.plot(RMSE,'-v')

plt.xlabel('Component Values')

plt.ylabel('RMSE Values')

plt.title('Comment Count Estimation Model with PCR')
lm=LinearRegression()
pcr_model=lm.fit(X_reduced_train[:,0:2],y_train)
y_pred=pcr_model.predict(X_reduced_train[:,0:2])
print(np.sqrt(mean_squared_error(y_train,y_pred))) #Trying to trainset control after validation
y_pred=pcr_model.predict(X_reduced_test[:,0:2])
print(np.sqrt(mean_squared_error(y_test,y_pred))) # Validation answered the purpose :)