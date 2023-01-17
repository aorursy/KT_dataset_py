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
import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FastICA
rng = np.random.RandomState(42)

S = rng.standard_t(1.5, size=(20000, 2))
S
S[:, 0] *=2.

S
A = np.array([[1, 1], [0, 2]])

A
X = np.dot(S, A.T)

X
pca = PCA()

S_pca = pca.fit(X).transform(X)

S_pca
ica = FastICA(random_state=rng)

S_ica = ica.fit(X).transform(X)
S_pca.shape, S_ica.shape
S_ica /= S_ica.std(axis=0)
def plot_samples(S, axis_list=None):

    plt.scatter(S[:, 0], S[:, 1], s=2, marker='o', zorder=10,

                color='steelblue', alpha=0.5)

    if axis_list is not None:

        colors = ['orange', 'red']

        for color, axis in zip(colors, axis_list):

            axis /= axis.std()

            x_axis, y_axis = axis

            # Trick to get legend to work

            plt.plot(0.1 * x_axis, 0.1 * y_axis, linewidth=2, color=color)

            plt.quiver(0, 0, x_axis, y_axis, zorder=11, width=0.01, scale=6,

                       color=color)



    plt.hlines(0, -3, 3)

    plt.vlines(0, -3, 3)

    plt.xlim(-3, 3)

    plt.ylim(-3, 3)

    plt.xlabel('x')

    plt.ylabel('y')
plt.figure()

plt.subplot(2, 2, 1)

plot_samples(S / S.std())

plt.title('True Independent Sources')



axis_list = [pca.components_.T, ica.mixing_]

plt.subplot(2, 2, 2)

plot_samples(X / np.std(X), axis_list=axis_list)

legend = plt.legend(['PCA', 'ICA'], loc='upper right')

legend.set_zorder(100)



plt.title('Observations')



plt.subplot(2, 2, 3)

plot_samples(S_pca / np.std(S_pca, axis=0))

plt.title('PCA recovered signals')



plt.subplot(2, 2, 4)

plot_samples(S_ica / np.std(S_ica))

plt.title('ICA recovered signals')



plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.36)

plt.show()
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder()

        lbl.fit(list(train[c].values) + list(test[c].values))

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))
print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))
from sklearn.decomposition import PCA, FastICA

n_comp = 10



pca = PCA(n_components=n_comp, random_state=42)

pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))

pca2_results_test = pca.transform(test)
ica = FastICA(n_components=n_comp, random_state=42)

ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))

ica2_results_test = ica.transform(test)
for i in range(1, n_comp+1):

    train['pca_' + str(i)] = pca2_results_train[:, i-1]

    test['pca_' + str(i)] = pca2_results_test[:, i-1]

    

    train['ica_' + str(i)] = ica2_results_train[:, i-1]

    test['ica_' + str(i)] = ica2_results_test[:, i-1]



y_train = train["y"]

y_mean = np.mean(y_train)
col_list = list(train)

for x in ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8', 'ID', 'y', 

          'pca_1', 'pca_2','pca_3','pca_4','pca_5','pca_6','pca_7','pca_8','pca_9','pca_10',

          'ica_1', 'ica_2','ica_3','ica_4','ica_5','ica_6','ica_7','ica_8','ica_9','ica_10']:

    col_list.remove(x)

train['n_features'] = train[col_list].sum(axis=1)



col_list = list(test)

for x in ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8', 'ID',

          'pca_1', 'pca_2','pca_3','pca_4','pca_5','pca_6','pca_7','pca_8','pca_9','pca_10',

          'ica_1', 'ica_2','ica_3','ica_4','ica_5','ica_6','ica_7','ica_8','ica_9','ica_10']:

    col_list.remove(x)

test['n_features'] = test[col_list].sum(axis=1)
import xgboost as xgb



xgb_params = {

    'n_trees': 500,

    'eta': 0.005,

    'max_depth': 4,

    'subsample': 0.95,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'base_score': y_mean,

    'silent': 1

}
dtrain = xgb.DMatrix(train.drop('y', axis=1), y_train)

dtest = xgb.DMatrix(test)
cv_results = xgb.cv(xgb_params,

                    dtrain,

                    num_boost_round=700,

                    early_stopping_rounds=50,

                    verbose_eval=10,

                    show_stdv=False

                   )



num_boost_rounds = len(cv_results)

print(num_boost_rounds)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
from sklearn.metrics import r2_score

print(r2_score(model.predict(dtrain), dtrain.get_label()))
y_pred = model.predict(dtest)

output = pd.DataFrame({"id": test['ID'].astype(np.int32), 'y': y_pred})

output.to_csv('xgboost-depth{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)