# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory saved as output.
train_df = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv', index_col='id')

test_df = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv', index_col='id')

print(len(train_df))

print(len(test_df))
y = train_df['label']



train_df = train_df.drop('label', axis=1)



y.head()

#X = train_df.append(test_df, ignore_index=True, sort=False)

#print(len(X))

#X.head()
#train_df = train_df.drop(train_df.loc[:, 'time':'label'].columns, axis = 1)
#from sklearn.feature_selection import SelectKBest

#from sklearn.feature_selection import f_regression

#from sklearn.feature_selection import VarianceThreshold



#train_df = train_df.drop('label', axis = 1)

#train_df = pd.DataFrame(SelectKBest(f_regression, k=50).fit_transform(train_df, y), index=list(range(1, train_df.shape[0] + 1)))

#selector = VarianceThreshold()

#X = selector.fit_transform(X)

#print(len(X[0]))
#from sklearn.decomposition import PCA

#pca = PCA(n_components=50)



#train_df = pca.fit_transform(X[:161168,:])

#test_df = pca.transform(X[161168:,:])

#train_df = X[:161168,:]

#test_df = X[161168:,:]
#from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()



#scaler = scaler.fit(train_df)

#train_df = pd.DataFrame(scaler.transform(train_df))

#test_df = pd.DataFrame(scaler.transform(test_df))

#train_df.head()
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(train_df, y, train_size=0.8)
print(train_df.shape)

print(test_df.shape)
from sklearn.linear_model import RidgeCV, LassoCV

from sklearn.svm import LinearSVR

from sklearn.ensemble import RandomForestRegressor, StackingRegressor

from  sklearn.tree import DecisionTreeRegressor



RANDOM_SEED=42



clf = RandomForestRegressor()

#clf = StackingRegressor(estimators = [

#    ('lcr', DecisionTreeRegressor(random_state=RANDOM_SEED)),

#    ('rfr', RidgeCV())

#],final_estimator=LassoCV(random_state=RANDOM_SEED))



#clf.fit(X_train, y_train)

clf.fit(train_df, y)
#feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)

feat_importances = pd.Series(clf.feature_importances_, index=train_df.columns)

selected = feat_importances.nlargest(45).index.tolist()
import os

os.chdir(r'/kaggle/working')



feat_importances.to_csv('importances.csv')
#X_train = X_train[selected]

#X_val = X_val[selected]

train_df = train_df[selected]

test_df = test_df[selected]
from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

train_df = scaler.fit_transform(train_df)

test_df = scaler.transform(test_df)

#X_train = scaler.fit_transform(X_train)

#X_val = scaler.transform(X_val)
print(train_df.shape)

y.head()
from sklearn.ensemble import RandomForestRegressor



clf = RandomForestRegressor()



#clf.fit(X_train, y_train)

clf.fit(train_df, y)
del train_df, y
y_val_predicted = clf.predict(X_val)



from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score



R2_score = r2_score(y_val, y_val_predicted)

MSE = mean_squared_error(y_val, y_val_predicted)

EVE = explained_variance_score(y_val, y_val_predicted)

print("R2 score: ", R2_score)

print("MSE: ", MSE)

print("EVE: ", EVE)
del X_train, X_val, y_train, y_val
y_test = clf.predict(test_df)

del test_df
y_test = y_test.tolist()
print(y_test)
final_df = pd.DataFrame([[i + 1, y_test[i]] for i in range(len(y_test))], columns=['id', 'label'])

print(final_df.head())



import os

os.chdir(r'/kaggle/working')



final_df.to_csv(r'final.csv', index=False)

    

from IPython.display import FileLink

FileLink(r'final.csv')