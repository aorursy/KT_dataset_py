# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import Imputer

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.


train=pd.read_csv("../input/train.csv")

train.head(2)
train.describe()
#train['intercept'] = 1.0

#train.columns

#cols_retain=['PassengerId', 'Pclass','Sex', 'Age', 'SibSp','Parch','Fare', 'Cabin', 'Embarked','intercept']

var=['PassengerId', 'Pclass','Sex', 'Age', 'SibSp','Parch','Fare', 'Cabin', 'Embarked']
def getFeatures(value, featureList, idx):

    features = {}

    if idx%10000==0: print(idx)

    for f in featureList:

        features[f] = value[f][idx]

    return features
X_all = train[var]

    # size_train, size_test = len(train), len(test)

#This step would convert the dataframe into List of feature - value mapping (dict like objects).. type =List

X_all1 = [getFeatures(X_all, var, idx) for idx in X_all.index]

#len(X_all1) : length of dict list
from sklearn.feature_extraction import DictVectorizer

# This would This transformer turns lists of mappings (dict-like objects) of feature 

#names to feature values into Numpy arrays or scipy.sparse matrices for use with scikit-learn estimators

vec = DictVectorizer()

X_all2 = vec.fit_transform(X_all1)
#X_all2.shape

X_all2
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)



X_all3 = imp.fit_transform(X_all2)

print ("size of X_all", X_all3.shape)

from scipy.sparse import csr_matrix

X_all4 = csr_matrix(X_all3)

X_all5 = X_all4.todense()

Y_all5 = np.array(train['Survived'])

Y_all5.shape
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_all5, Y_all5 , test_size = 0.3, random_state = 398)

from sklearn.linear_model import LogisticRegression
rt_lm = LogisticRegression()


rt_lm.fit(X_train,Y_train)
y_pred_test = rt_lm.predict(X_test)
Y_test.sha