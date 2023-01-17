from sklearn.decomposition import PCA

from sklearn import linear_model, decomposition, datasets

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
claim_ids = train_data[ 'id']

claim_ids_test = test_data['id']



# Seperating X and y for training 

y = train_data['loss']

train_data.drop('loss', axis =1, inplace=True)

train_data.drop('id', axis =1, inplace=True)

X = train_data 



# Seperating X 

test_data.drop('id', axis = 1, inplace=True)

X_test = test_data
#create a dataframe with only continuous features

split = 116 

continous_data = X.iloc[:,split:] 

X_cv = pd.DataFrame(continous_data)



test_continous_data = X_test.iloc[:, split:]

X_test_cv = pd.DataFrame(test_continous_data)





print(X_cv.shape)

print(X_test_cv.shape)



linear = linear_model.LinearRegression()

pca = PCA()

pipe = Pipeline(steps=[('pca', pca), ('linear', linear)])

n_components = list(range(1, 14))



pca.fit(X_cv)

plt.figure(1, figsize=(4, 3))

plt.clf()

plt.plot(pca.explained_variance_, linewidth=2)

plt.xlabel('n_components')

plt.ylabel('explained_variance_')

plt.show()



estimator = GridSearchCV(pipe, dict(pca__n_components=n_components))

estimator.fit(X_cv, y)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,

            linestyle=':', label='n_components chosen')

plt.legend(prop=dict(size=12))

plt.show()





## Preform the Dimensionality Reduction

pca.n_components=12

pca.fit(X_cv)

X_pca = pca.transform(X_cv)





pca.fit(X_cv)

X_pca = pca.transform(X_cv)

print(X_pca.shape)



pca.fit(X_test_cv)

X_test_pca = pca.transform(X_test_cv)

print(X_test_pca.shape)





linear.fit(X_pca,y)

y_pred = linear.predict(X_test_pca)

print(y_pred.shape, claim_ids_test.shape, type(y_pred), type(claim_ids_test.as_matrix()))





ids = np.array((claim_ids_test.as_matrix()))

print(claims)



y_prd = np.array((y_pred))

print(y_prd)