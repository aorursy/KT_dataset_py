

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

from sklearn import metrics, model_selection



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(train[c].values) + list(test[c].values))

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))
X = train.drop(['ID', 'y'], axis=1)

y = train['y']
feature_labels = X.columns #This will give us a list of all of the anonymized features
forest = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)

# 1000 trees sounds like a lot but setting n_jobs=-1 makes the process a lot faster since the

# trees will be constructed in parallel
forest.fit(X, y)    # Fits the Random Forest Regressor to the entire data set.

importances = forest.feature_importances_  # Sets importances equal to the feature importances of the model
indices = np.argsort(importances)[::-1]

order_features = []

order_importances = []

for f in range(X.shape[1]):

    print("%2d) %-*s %f" % (f+1, 30, feature_labels[f], importances[indices[f]]))

    order_features.append(feature_labels[f])

    order_importances.append(importances[indices[f]])
plt.figure(figsize=(10,10))

top_50_importances = order_importances[:50] #This will give us the top 50 features by importance

plt.title('Top 50 Features By Importance')

plt.bar(range(X.shape[1]-326), top_50_importances, color='lightblue', align='center')

plt.xticks(range(X.shape[1]-326), order_features[:50], rotation=90)

plt.xlim([-1, X.shape[1]-326])

plt.show()
test = test.drop(order_features[250:], axis=1)

train = train.drop(order_features[250:], axis=1) # Modify train to only take in the top 100 features and the target column y
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin

import xgboost as xgb

from sklearn.linear_model import ElasticNetCV, LassoLarsCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline, make_union

from sklearn.utils import check_array

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import r2_score
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import ElasticNetCV, LassoLarsCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline, make_union

from sklearn.utils import check_array

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection

from sklearn.decomposition import PCA, FastICA

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import r2_score







class StackingEstimator(BaseEstimator, TransformerMixin):

    

    def __init__(self, estimator):

        self.estimator = estimator



    def fit(self, X, y=None, **fit_params):

        self.estimator.fit(X, y, **fit_params)

        return self

    def transform(self, X):

        X = check_array(X)

        X_transformed = np.copy(X)

        # add class probabilities as a synthetic feature

        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):

            X_transformed = np.hstack((self.estimator.predict_proba(X), X))



        # add class prodiction as a synthetic feature

        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))



        return X_transformed
n_comp = 12



# tSVD

tsvd = TruncatedSVD(n_components=n_comp, random_state=420)

tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))

tsvd_results_test = tsvd.transform(test)



# PCA

pca = PCA(n_components=n_comp, random_state=420)

pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))

pca2_results_test = pca.transform(test)



# ICA

ica = FastICA(n_components=n_comp, random_state=420)

ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))

ica2_results_test = ica.transform(test)



# GRP

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)

grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))

grp_results_test = grp.transform(test)



# SRP

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)

srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))

srp_results_test = srp.transform(test)



#save columns list before adding the decomposition components



usable_columns = list(set(train.columns) - set(['y']))
for i in range(1, n_comp + 1):

    train['pca_' + str(i)] = pca2_results_train[:, i - 1]

    test['pca_' + str(i)] = pca2_results_test[:, i - 1]



    train['ica_' + str(i)] = ica2_results_train[:, i - 1]

    test['ica_' + str(i)] = ica2_results_test[:, i - 1]



    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]

    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]



    train['grp_' + str(i)] = grp_results_train[:, i - 1]

    test['grp_' + str(i)] = grp_results_test[:, i - 1]



    train['srp_' + str(i)] = srp_results_train[:, i - 1]

    test['srp_' + str(i)] = srp_results_test[:, i - 1]

y_train = train['y'].values

y_mean = np.mean(y_train)

id_test = test['ID'].values

#finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 

finaltrainset = train[usable_columns].values

finaltestset = test[usable_columns].values
'''Train the xgb model then predict the test data'''



xgb_params = {

    'n_trees': 520, 

    'eta': 0.0045,

    'max_depth': 4,

    'subsample': 0.93,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean, # base prediction = mean(target)

    'silent': 1

}





dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)

dtest = xgb.DMatrix(test)



num_boost_rounds = 1250

# train model

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)

'''Train the stacked models then predict the test data'''



stacked_pipeline = make_pipeline(

    StackingEstimator(estimator=ElasticNetCV(normalize=True)),

    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=5, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7, n_estimators=200)),

    ElasticNetCV()



)



stacked_pipeline.fit(finaltrainset, y_train)

results = stacked_pipeline.predict(finaltestset)



'''R2 Score on the entire Train data when averaging'''



print('R2 score on train data:')

print(r2_score(y_train,stacked_pipeline.predict(finaltrainset)*0.2855 + model.predict(dtrain)*0.7145))







'''Average the predictions test data of both models then save it on a csv file'''



print('Cross Validation')

print('................')



n_folds = 5

kf = model_selection.StratifiedKFold(n_splits=n_folds, random_state=1, shuffle=True)



X = train.drop('y', axis=1).values

y = train['y'].values





fold = 0

for train_index, test_index in kf.split(X, y):

    fold += 1

    

    X_training, X_valid = X[train_index], X[test_index]

    y_training, y_valid = y[train_index], y[test_index]

    

    finaltrainset = train[usable_columns].values

    final_train, final_valid = finaltrainset[train_index], finaltrainset[test_index]

    

    print("Fold", fold, X_training.shape, X_valid.shape)

    

    print('Fitting XGBoost for Fold {}'.format(fold))

    dtrain = xgb.DMatrix(X_training, y_training)

    dtest = xgb.DMatrix(X_valid)

    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

    

    print('Fitting stacked pipeline for Fold {}'.format(fold))

    stacked_pipeline.fit(final_train, y_training)

    

    print(r2_score(y_valid,stacked_pipeline.predict(final_valid)*0.2855 + model.predict(dtest)*0.7145))



    

sub = pd.DataFrame()

sub['ID'] = id_test

sub['y'] = y_pred*0.75 + results*0.25

sub.to_csv('stacked-models.csv', index=False)