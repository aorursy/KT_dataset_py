# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory





# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd 

from sklearn import svm 

from sklearn.decomposition import PCA 

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import LinearSVC

%matplotlib inline 

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_df = pd.read_csv("../input/train.csv")

test_df  = pd.read_csv("../input/test.csv")

X_train = train_df.drop("label",axis=1)

y_train = train_df['label']



print (X_train,y_train)
pca= PCA(n_components=50)

pca.fit(X_train)

X_train_pca = pca.transform(X_train)

X_test_pca  = pca.transform(test_df)

predict= OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_train_pca,y_train).predict(X_test_pca)
print(predict[0:200])