import numpy as np

np.random.seed(42)

# To read csv file

import pandas as pd

# To Split data into train and cv data

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold,cross_val_score

# For plot AUROC graph

import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV

# For heatmap

import seaborn as sns

# To ignore warninga

import warnings

warnings.filterwarnings('ignore')

# To stndardize the data

from sklearn.preprocessing import StandardScaler

import tqdm

from sklearn.decomposition import TruncatedSVD
ls
train = pd.read_csv("../input/datarar/train.csv")

train.head()
test = pd.read_csv("../input/datarar/test.csv")

test.head(5)
def feature_engg(df, if_test = False):

    if if_test:

        temp = df.drop(['id'], axis=1)

    else:

        temp = df.drop(['id','target'], axis=1)

    # Mean and Std FE

    df['mean'] = np.mean(temp, axis=1)

    df['std'] = np.std(temp, axis=1)

    # Trigometric FE

    sin_temp = np.sin(temp)

    cos_temp = np.cos(temp)

    tan_temp = np.tan(temp)

    df['mean_sin'] = np.mean(sin_temp, axis=1)

    df['mean_cos'] = np.mean(cos_temp, axis=1)

    df['mean_tan'] = np.mean(tan_temp, axis=1)

    # Hyperbolic FE

    sinh_temp = np.sinh(temp)

    cosh_temp = np.cosh(temp)

    tanh_temp = np.tanh(temp)

    df['mean_sinh'] = np.mean(sin_temp, axis=1)

    df['mean_cosh'] = np.mean(cos_temp, axis=1)

    df['mean_tanh'] = np.mean(tan_temp, axis=1)

    # Exponents FE

    exp_temp = np.exp(temp)

    expm1_temp = np.expm1(temp)

    exp2_temp = np.exp2(temp)

    df['mean_exp'] = np.mean(exp_temp, axis=1)

    df['mean_expm1'] = np.mean(expm1_temp, axis=1)

    df['mean_exp2'] = np.mean(exp2_temp, axis=1)

    # Polynomial FE

    # X**2

    df['mean_x2'] = np.mean(np.power(temp,2), axis=1)

    # X**3

    df['mean_x3'] = np.mean(np.power(temp,3), axis=1)

    # X**4

    df['mean_x4'] = np.mean(np.power(temp,4), axis=1)

    return df
df_train = feature_engg(train)

df_train.head(5)
df_test = feature_engg(test, True)

df_test.head(5)
# Take separate for features value

tr_X = df_train.drop(['id','target'], axis=1)

# Take separate for class value

tr_y = df_train['target'].values

# Take test feature value

ts_X = df_test.drop(['id'], axis=1)
exp_rat = []

for i in range(2,min(tr_X.shape[0],tr_X.shape[1])):

    trunsvd = TruncatedSVD(n_components=i)

    trunsvd.fit(tr_X,tr_y)

    exp_rat.append(np.sum(trunsvd.explained_variance_ratio_))
plt.plot(np.arange(2,min(tr_X.shape[0],tr_X.shape[1])),exp_rat)

plt.grid()

plt.show()
# Fit and transform on train data and transform on cv and test data

trunsvd = TruncatedSVD(n_components=175)

tr_X = trunsvd.fit_transform(tr_X,tr_y)

ts_X = trunsvd.transform(ts_X)
from imblearn.over_sampling import SMOTE

smt = SMOTE(random_state=87,n_jobs=-1)

tr_X, tr_y = smt.fit_sample(tr_X, tr_y)
stand_vec = StandardScaler()

tr_X = stand_vec.fit_transform(tr_X)

pd.DataFrame(tr_X).head(5)
ts_X = stand_vec.transform(ts_X)

pd.DataFrame(ts_X).head(5)
def hyperparameter_model(models, params):

    str_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

    # Find the right hyperparameter for the model

    grid_clf = GridSearchCV(models, params, cv=str_cv, return_train_score=True,scoring='roc_auc',verbose = 1)

    # Fit on train data

    grid_clf.fit(tr_X, tr_y)

    return grid_clf
from sklearn.neighbors import KNeighborsClassifier
params = {'n_neighbors':np.arange(3,51,2).tolist(), 'algorithm': ['kd_tree','brute']}

# Instance of knn model

knn_model = KNeighborsClassifier()

# Call hyperparameter for find the best params as possible

knn_clf = hyperparameter_model(knn_model, params)
cv_pvt = pd.pivot_table(pd.DataFrame(knn_clf.cv_results_),values='mean_test_score', index='param_n_neighbors', columns='param_algorithm')
tr_pvt = pd.pivot_table(pd.DataFrame(knn_clf.cv_results_),values='mean_train_score', index='param_n_neighbors',columns='param_algorithm')

plt.title('Train Hyperparameter')

sns.heatmap(tr_pvt, annot=True)

plt.show()

plt.title('CV Hyperparameter')

sns.heatmap(cv_pvt, annot=True)

plt.show()
print(knn_clf.best_params_)

clf = CalibratedClassifierCV(knn_clf, cv=3)

clf.fit(tr_X,tr_y)
# Create a submssion format to make submission in Kaggle

temp_id = df_test['id']

knn_csv = clf.predict_proba(ts_X)[:,1]

knn_df = pd.DataFrame(np.column_stack((temp_id,knn_csv)),columns=['id','target'])

knn_df['id'] = knn_df['id'].astype('int32')
knn_df.head()
knn_df.to_csv('submission.csv', index=False)