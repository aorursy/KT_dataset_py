# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
df=df.drop('gameId',1)
from sklearn.model_selection import train_test_split
labels=df.iloc[:,0]
train=df.iloc[:,1:]
X_train, X_test, y_train, y_test = train_test_split(train, labels,

                                                    stratify=labels, 

                                                    test_size=0.25)

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



X_train = StandardScaler().fit_transform(X_train)

pca = PCA()

pca_x = pca.fit_transform(X_train)

pca_x_format=pd.DataFrame(pca_x)



X_test=StandardScaler().fit_transform(X_test)

pca_test = PCA()

pca_test = pca_test.fit_transform(X_test)

pca_test_format=pd.DataFrame(pca_test)
explained_variance = pca.explained_variance_ratio_

corr_list=explained_variance.cumsum()
eigenvector_counter=0

for index in range(0,len(corr_list)):

    if corr_list[index] < 0.99:

        eigenvector_counter=eigenvector_counter+1
for index in range(eigenvector_counter,len(pca_x_format.columns)-1):

    pca_x_format=pca_x_format.drop([index], axis=1)

    pca_test_format=pca_test_format.drop([index], axis=1)
from sklearn.metrics import roc_auc_score



mein_classifier = LinearDiscriminantAnalysis()



mein_classifier.fit(X_train,y_train)



prediction=mein_classifier.predict(X_test)



print(round(roc_auc_score(y_test, prediction),5))

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp =confusion_matrix(y_test,prediction).ravel()
print("Accuracy: ",tp / (tp + tn))