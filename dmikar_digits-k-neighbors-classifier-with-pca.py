import numpy as np

import pandas as pd

train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test_data =  pd.read_csv("/kaggle/input/digit-recognizer/test.csv")



y_train = train_data["label"]

X_train = train_data.drop(["label"], axis=1)
X_train.shape
from sklearn.decomposition import PCA

pca_transformer = PCA(n_components = 20)

X_train_reduced = pca_transformer.fit_transform (X_train)

X_train_reduced.shape

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV



kn_params = {'n_neighbors':np.arange(1,7), 'weights':['uniform', 'distance']}

kn_classifier = KNeighborsClassifier ()



kn_search = GridSearchCV(kn_classifier, kn_params, cv=5)



kn_search.fit (X_train_reduced, y_train)

kn_search.best_score_
from sklearn.pipeline import Pipeline

pca_kn_pipeline = Pipeline ([('pca', PCA()), ('clf',kn_search.best_estimator_)])

pca_kn_params = {'pca__n_components':np.arange(10,50,5)}



pca_kn_search = GridSearchCV(pca_kn_pipeline, pca_kn_params, cv=5)

pca_kn_search.fit (X_train, y_train)

pca_kn_search.best_score_
pca_kn_search.best_params_
y_test = pca_kn_search.best_estimator_.fit(X_train, y_train).predict(test_data)

submission_df = pd.DataFrame({'ImageId':np.arange(1,y_test.size+1), 'Label':y_test})

submission_df.to_csv('submission.csv', index = False)