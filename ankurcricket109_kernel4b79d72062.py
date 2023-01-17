# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dtypes = {
        'Id'                : 'uint32',
        'groupId'           : 'uint32',
        'matchId'           : 'uint16',
        'assists'           : 'uint8',
        'boosts'            : 'uint8',
        'damageDealt'       : 'float16',
        'DBNOs'             : 'uint8',
        'headshotKills'     : 'uint8', 
        'heals'             : 'uint8',    
        'killPlace'         : 'uint8',    
        'killPoints'        : 'uint8',    
        'kills'             : 'uint8',    
        'killStreaks'       : 'uint8',    
        'longestKill'       : 'float16',    
        'maxPlace'          : 'uint8',    
        'numGroups'         : 'uint8',    
        'revives'           : 'uint8',    
        'rideDistance'      : 'float16',    
        'roadKills'         : 'uint8',    
        'swimDistance'      : 'float16',    
        'teamKills'         : 'uint8',    
        'vehicleDestroys'   : 'uint8',    
        'walkDistance'      : 'float16',    
        'weaponsAcquired'   : 'uint8',    
        'winPoints'         : 'uint8', 
        'winPlacePerc'      : 'float16' 
}
df=pd.read_csv('../input/train.csv', dtype=dtypes)
df1=pd.read_csv('../input/test.csv', dtype=dtypes)
df1.head()
X=df.iloc[0:100000, 0:-1].values
X1=df1.iloc[:, :].values
Y=df.iloc[0:100000,-1].values
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
X1=sc_X.transform(X1)
# Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=5)
X = pca.fit_transform(X)
explained_variance = pca.explained_variance_ratio_
X1=pca.transform(X1)
explained_variance
X
# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y)
Y_pred=regressor.predict(X1)
Y_pred.to_csv('csv_to_submit.csv', index = False)
Y_pred
my_submission = pd.DataFrame({'Id': df1.Id, 'winPlacePerc': Y_pred})
my_submission.to_csv("submission.csv", index=False)
