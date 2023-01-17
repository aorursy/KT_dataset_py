# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")



Y = train["label"]

X = train.drop(labels = ["label"], axis = 1) 

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3)

X_test = test
covar_matrix = PCA()

scaler = StandardScaler()

X_new = covar_matrix.fit_transform(scaler.fit_transform(X))



# Percentage of variance explained by each of the selected components.

variance_ratio = covar_matrix.explained_variance_ratio_

cumulative_var_ratio = np.cumsum(np.round(variance_ratio, decimals=2))
%matplotlib inline

plt.ylabel('% Variance Explained')

plt.xlabel('# of Features')

plt.title('PCA Analysis')

plt.plot(cumulative_var_ratio)
# Pipeline with: StandardScaler | PCA with n_components set to 50, quadratic SVM

model = SVC(kernel='poly', degree=2, C=1.0, gamma='scale', coef0=1.0)

covar_matrix = PCA(n_components=50)



pipeline = Pipeline(steps=[

        ('scale', StandardScaler()),

        ('reduce_dim', covar_matrix),

        #('preprocessor', preprocessor),

        ('model', model)])
# pipeline.fit -> scale | reduce_dim| preprocess | fit

pipeline.fit(X_train, Y_train)



#n_features_to_test = np.arange(1, 11)

#params = {'reduce_dim__n_components': n_features_to_test}

#gridsearch = GridSearchCV(pipeline, params, cv=3, verbose=1).fit(X_train, Y_train)

#print('Final score is: ', gridsearch.score(X_val, Y_val))



# pipeline.score ->  scale | preprocess | predict | score

print('Final score with 50 scaled features: ', pipeline.score(X_val, Y_val))
# pipeline.predict -> scale | reduce_dim| preprocess | predict

preds_test = pipeline.predict(X_test)
# Save test predictions to file

output = pd.DataFrame({'ImageId': X_test.index+1,

                       'Label': preds_test})

output.to_csv('submission.csv', index=False)