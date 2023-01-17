import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV 
from scipy import stats

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

y_train = np.log1p(train["target"].values)
id1 = test.ID.values

PERC_TRESHOLD = 0.96   ### Percentage of zeros in each feature ###
N_COMP = 20            ### Number of decomposition components ###

target = np.log1p(train['target']).values
cols_to_drop = [col for col in train.columns[2:]
                    if [i[1] for i in list(train[col].value_counts().items()) 
                    if i[0] == 0][0] >= train.shape[0] * PERC_TRESHOLD]

print("Define training features...")
exclude_other = ['ID', 'target']
train_features = []
for c in train.columns:
    if c not in cols_to_drop \
    and c not in exclude_other:
        train_features.append(c)
print("Number of featuress for training: %s" % len(train_features))

train, test = train[train_features], test[train_features]
print("\nTrain shape: {}\nTest shape: {}".format(train.shape, test.shape))

print("\nStart decomposition process...")
print("PCA")
pca = PCA(n_components=N_COMP, random_state=17)
pca_results_train = pca.fit_transform(train)
pca_results_test = pca.transform(test)

print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=17)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

print("ICA")
ica = FastICA(n_components=N_COMP, random_state=17)
ica_results_train = ica.fit_transform(train)
ica_results_test = ica.transform(test)

print("GRP")
grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=17)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

print("SRP")
srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=17)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

print("Append decomposition components to datasets...")
for i in range(1, N_COMP + 1):
    train['pca_' + str(i)] = pca_results_train[:, i - 1]
    test['pca_' + str(i)] = pca_results_test[:, i - 1]

    train['ica_' + str(i)] = ica_results_train[:, i - 1]
    test['ica_' + str(i)] = ica_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))

params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 12**2,
        "learning_rate" : 0.02,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 42,
        "verbosity" : 1,
        "seed": 42
    }

X_train = train
X_test = test

param_grid = {}
y_pred = []

rand_list = {'C': 2.209602254061174, 'gamma': 0.8670701646824878}

model = lgb.LGBMRegressor(**params)

mod = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=5, verbose=4)
mod.fit(X_train, y_train)
yp1 = np.expm1(mod.predict(X_test))
y_pred.append(yp1)

mod1 = RandomizedSearchCV(model, param_distributions = rand_list, n_iter = 20, n_jobs = 4, 
                                 cv = 5, random_state = 2017)
mod1.fit(X_train, y_train)
yp2 = np.expm1(mod1.predict(X_test))
y_pred.append(yp2)

yp_avg = np.average(y_pred, axis=0)

yp_max = np.amax(y_pred, axis=0)

'''
output = pd.DataFrame({'ID': id1, 'target': yp_avg})
output.to_csv("lgb_avg.csv", index=False)

output = pd.DataFrame({'ID': id1, 'target': yp2})
output.to_csv("D:/Santander/lgb_cv_rand.csv", index=False)

output = pd.DataFrame({'ID': id1, 'target': yp1})
output.to_csv("lgb.csv", index=False)'''

output = pd.DataFrame({'ID': id1, 'target': yp_max})
output.to_csv("lgb_max.csv", index=False)
