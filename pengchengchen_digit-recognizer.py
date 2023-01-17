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
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

print(df.head())

print('=============================================')

print(df_test.head())
print(df.info())

print(df_test.info())
# scale the data

from sklearn import preprocessing as pre

label = df['label'].copy()

df = df.drop(['label'], axis=1)

df = pd.DataFrame(data=pre.scale(df), columns=df.columns)

df_test = pd.DataFrame(data=pre.scale(df_test), columns=df_test.columns)

print(df.head())
# use the PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=150)

X_r = pca.fit_transform(df.values)

test_r = pca.transform(df_test.values)

# print(pca.explained_variance_ratio_)
# use the svm

# from sklearn.model_selection import cross_val_score

from sklearn import svm

clf = svm.SVC(kernel='linear', C=1)

clf.fit(X_r, label)

result = clf.predict(test_r)

submission = pd.DataFrame({'ImageId': df_test.index,

                           'Label': result})

submission.to_csv('output.csv', index=False)



# scores = cross_val_score(clf, X_r, label)

# print(scores)