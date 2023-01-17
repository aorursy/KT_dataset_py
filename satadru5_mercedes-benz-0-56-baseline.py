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
import numpy as np # linear algebra

import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

import matplotlib

from matplotlib import pyplot

import xgboost as xgb





# read datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# process columns, apply LabelEncoder to categorical features

for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder() 

        lbl.fit(list(train[c].values) + list(test[c].values)) 

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))



# shape        

print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))

##Add decomposed components: PCA / ICA etc.

from sklearn.decomposition import PCA, FastICA

n_comp = 10



# PCA

pca = PCA(n_components=n_comp, random_state=42)

pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))

pca2_results_test = pca.transform(test)



# ICA

ica = FastICA(n_components=n_comp, random_state=42)

ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))

ica2_results_test = ica.transform(test)



# Append decomposition components to datasets

for i in range(1, n_comp+1):

    train['pca_' + str(i)] = pca2_results_train[:,i-1]

    test['pca_' + str(i)] = pca2_results_test[:, i-1]

    

    train['ica_' + str(i)] = ica2_results_train[:,i-1]

    test['ica_' + str(i)] = ica2_results_test[:, i-1]

    

y_train = train["y"]

y_mean = np.mean(y_train)
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)

dtest = xgb.DMatrix(test)





# grid search

model = XGBRegressor()

n_estimators = [100, 200, 300, 400, 500]

learning_rate = [0.0001, 0.001, 0.01, 0.1]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)

#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)

grid_search = GridSearchCV(model, param_grid, scoring="r2", n_jobs=-1, cv=10)

grid_result = grid_search.fit(train.drop('y', axis=1), y_train)

# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

	print("%f (%f) with: %r" % (mean, stdev, param))
# plot results

scores = np.array(means).reshape(len(learning_rate), len(n_estimators))

for i, value in enumerate(learning_rate):

    pyplot.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))

pyplot.legend()

pyplot.xlabel('n_estimators')

pyplot.ylabel('r2')
model_1 = XGBRegressor(n_estimators=500,learning_rate=0.01,max_depth=4)
model_1.fit(train.drop('y', axis=1), y_train)
y_pred=model_1.predict(test)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})

output.to_csv('submission_baseLine_2.csv', index=False)
output.head(5)