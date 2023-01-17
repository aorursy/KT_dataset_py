import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
df = pd.read_csv('../input/heart.csv')

df.head()
df['age_grp'] = pd.qcut(df['age'], q=5, labels=[1, 2, 3, 4, 5])

df = df.drop('age', axis=1)

df.head()
nominal_cols = ['cp', 'restecg', 'slope', 'thal']

df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

df.head()
df.head()
X_cols = list(df.columns)

X_cols.remove('target')

X = df[X_cols]

y = df['target']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
lmodel = LogisticRegression()

lmodel.fit(train_X, train_y)
preds = lmodel.predict(test_X)
accuracy_score(test_y, preds)
print(classification_report(test_y, preds))
print(confusion_matrix(test_y, preds))
import h2o

h2o.init()
df = h2o.import_file(path="../input/heart.csv")

df.head()
response = 'target'

df[response] = df[response].asfactor()
train, test = df.split_frame(

    ratios=[0.75], 

    destination_frames=['train.hex', 'test.hex']

)
predictors = set(df.columns) - {'target'}

predictors
from h2o.estimators.gbm import H2OGradientBoostingEstimator

gbm = H2OGradientBoostingEstimator()

gbm.train(x=predictors, y=response, training_frame=train, validation_frame=test)
print(gbm)
from h2o.estimators.gbm import H2OGradientBoostingEstimator

gbm = H2OGradientBoostingEstimator(ntrees=100, nfolds=5, learn_rate_annealing = 0.99,

                                   stopping_rounds = 10, stopping_tolerance = 1e-4)
hyper_params = {

    'max_depth' : [5, 10, 20],

    'learn_rate': [0.1, 0.01],

    'sample_rate' : [0.8, 1.0],

    'col_sample_rate' : [0.8, 1.0],

}
from h2o.grid import H2OGridSearch

grid = H2OGridSearch(gbm, hyper_params=hyper_params)

grid.train(x=predictors, y=response, training_frame=train)
print(grid)
grid_df = grid.get_grid(sort_by='auc',decreasing=True)

grid_df
best_model = grid_df[0]

best_model
best_model.model_performance(test_data=test)