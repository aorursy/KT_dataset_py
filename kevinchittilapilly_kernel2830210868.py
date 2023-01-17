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
train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
sample_submission=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

print(test.shape)


train.describe()
test.head()
yy=train['SalePrice']

yy = yy.values.astype('float32')

y = pd.DataFrame(np.reshape(yy, (len(yy),1)))
print(y.head())
print(y.head())
X=train.iloc[:,0:80]
print(X.head())
X.describe()
print(X.shape)

def Cat_conversion(cols):
    for i in cols:
        X[i]=X[i].astype("category").cat.codes
Cat_conversion(X)


def Cat_conversion1(cols):
    for i in cols:
        test[i]=test[i].astype("category").cat.codes
Cat_conversion1(test)
test.shape
from sklearn.preprocessing import StandardScaler
x = X.values

# Standardizing the features
x = StandardScaler().fit_transform(x)
y=StandardScaler().fit_transform(y)
test=test.values
test=StandardScaler().fit_transform(test)

test.shape
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
y=pd.DataFrame(y)
finalDf = pd.concat([principalDf, y], axis = 1)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(test)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


principalDf.head()
xtrain=finalDf.iloc[:,0:2]
xlabel=finalDf.iloc[:,-1]
xtrain.head()

from sklearn import preprocessing
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y)
print(training_scores_encoded.shape)

Y = pd.DataFrame(np.reshape(training_scores_encoded, (len(training_scores_encoded),1)))
print(Y.head())

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(xtrain, Y)

principalDf.head()
y_pred=clf.predict(principalDf)
print(y_pred)
test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
print(test.shape)
