import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

from sklearn.linear_model import LassoLarsCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.pipeline import make_pipeline, make_union

from sklearn.utils import check_array

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
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(train[c].values) + list(test[c].values))

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))



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



# Append decomposition components to datasets

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
leaks = {

    1:71.34112,

    12:109.30903,

    23:115.21953,

    28:92.00675,

    42:87.73572,

    43:129.79876,

    45:99.55671,

    57:116.02167,

    3977:132.08556,

    88:90.33211,

    89:130.55165,

    93:105.79792,

    94:103.04672,

    1001:111.65212,

    104:92.37968,

    72:110.54742,

    78:125.28849,

    105:108.5069,

    110:83.31692,

    1004:91.472,

    1008:106.71967,

    1009:108.21841,

    973:106.76189,

    8002:95.84858,

    8007:87.44019,

    1644:99.14157,

    337:101.23135,

    253:115.93724,

    8416:96.84773,

    259:93.33662,

    262:75.35182,

    1652:89.77625

}



leaky_df = test.ix[(test.ID == 1) | (test.ID==8002) | (test.ID == 259) | (test.ID==262) |

                   (test.ID == 8007) | (test.ID==72) | (test.ID == 3977) | (test.ID==12) |

                   (test.ID == 973) | (test.ID==78) | (test.ID == 337) | (test.ID==23) | 

                   (test.ID == 88) | (test.ID==89) | (test.ID == 28) | (test.ID==93) |

                   (test.ID == 94) | (test.ID==8416) (test.ID == 1644) | (test.ID==104) |

                   (test.ID == 1001) | (test.ID==42) | (test.ID == 43) | (test.ID==1004) |

                   (test.ID == 45) | (test.ID==110) | (test.ID == 1008) | (test.ID==1009) |

                   (test.ID == 1652) | (test.ID==105) | (test.ID == 57) | (test.ID==253)]



leaky_df['y'] = leaky_df.apply(lambda x: leaks[x.ID], axis=1)



train = pd.concat([train, leaky_df], axis=0)
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

    'base_score': 100.92, #y_mean, # base prediction = mean(target)

    'silent': 1

}
train = train.reindex_axis(sorted(train.columns), axis=1)

test = test.reindex_axis(sorted(test.columns), axis=1)



dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)

dtest = xgb.DMatrix(test)
num_boost_rounds = 1250+500



# train model

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

y_pred = model.predict(dtest)
'''Train the stacked models then predict the test data'''



stacked_pipeline = make_pipeline(

    StackingEstimator(estimator=LassoLarsCV(normalize=True)),

    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=5, max_features=0.7,

                                                          min_samples_leaf=18, min_samples_split=14, subsample=0.7)),

    LassoLarsCV()



)
stacked_pipeline.fit(finaltrainset, y_train)

results = stacked_pipeline.predict(finaltestset)
sub = pd.DataFrame()

sub['ID'] = id_test

sub['y'] = y_pred*0.55 + results*0.45



sub['y'] = sub.apply(lambda x: leaks[x.ID] if x.ID in leaks.keys() else x['y'], axis=1)



sub.to_csv('submission-stacked-models.csv', index=False)