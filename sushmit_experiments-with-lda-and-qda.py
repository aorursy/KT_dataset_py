# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import bokeh

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# Any results you write to the current directory are saved as output.
df_train = pd.read_json("../input/train.json")

df_test = pd.read_json("../input/test.json")
clf = LinearDiscriminantAnalysis(solver="svd")
#X_train = df_train[['bathrooms','bedrooms','price','latitude','longitude']]

X_train = df_train[['bathrooms','bedrooms','price']]

Y_train = df_train[['interest_level']]

#X_test = df_test[['bathrooms','bedrooms','price','latitude','longitude']]

X_test = df_test[['bathrooms','bedrooms','price']]

clf = LinearDiscriminantAnalysis(solver="svd")

clf.fit(X_train, Y_train)

pred =clf.predict(X_test)
pred