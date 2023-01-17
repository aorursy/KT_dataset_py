import numpy as np

import pandas as pd

from sklearn.metrics import confusion_matrix, average_precision_score, classification_report

from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_predict, train_test_split, StratifiedKFold, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression



import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")
working_folder = '/kaggle/input/predict-employee-quiting/'
training_data = pd.read_csv(working_folder + 'train_data.csv')
training_data = training_data[training_data.columns[1:]]
training_data.describe().T
training_data.info()
training_data.head()
training_data['Attrition'].value_counts()
356 / 1849
def show_unique(df):

    unique = {}

    

    for col in df.columns:

        values = np.sort(df[col].unique())

        unique[col] = [len(values), values]

    

    for col in unique:

        print('Column {} has {} unique values: {} \n'.format(col, unique[col][0], unique[col][1]))
show_unique(training_data)
BusinessTravel_map = {

    'Non-Travel': 0,

    'Travel_Rarely': 1,

    'Travel_Frequently': 2

}
training_data['BusinessTravel'] = training_data['BusinessTravel'].map(BusinessTravel_map)
training_data.drop('StandardHours', axis=1, inplace=True)
training_data.isnull().sum()
training_median = training_data.median()

training_data.fillna(training_median, inplace=True)
training_data.isnull().sum()
for col in training_data.columns:

    if (training_data[col].dtype != 'object') & (len(training_data[col].unique()) > 10):

        training_data[col].plot(kind='box')

        plt.show()
training_data = pd.get_dummies(training_data, prefix_sep='_', drop_first=True)
training_data
training_data.info()
pca = PCA(n_components=2, svd_solver= 'full')
X_train_PCA = pca.fit_transform(training_data)

X_train_PCA = pd.DataFrame(X_train_PCA)

X_train_PCA.index = training_data.index
X_train_PCA
def cov_matrix(data, verbose=False):

    covariance_matrix = np.cov(data, rowvar=False)

    if is_pos_def(covariance_matrix):

        inv_covariance_matrix = np.linalg.inv(covariance_matrix)

        if is_pos_def(inv_covariance_matrix):

            return covariance_matrix, inv_covariance_matrix

        else:

            print("Error: Inverse of Covariance Matrix is not positive definite!")

    else:

        print("Error: Covariance Matrix is not positive definite!")
def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):

    inv_covariance_matrix = inv_cov_matrix

    vars_mean = mean_distr

    diff = data - vars_mean

    md = []

    for i in range(len(diff)):

        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))

    return md
def MD_detectOutliers(dist, extreme=False, verbose=False):

    k = 3. if extreme else 2.

    threshold = np.mean(dist) * k

    outliers = []

    for i in range(len(dist)):

        if dist[i] >= threshold:

            outliers.append(i)

    return np.array(outliers)
def MD_threshold(dist, extreme=False, verbose=False):

    k = 3. if extreme else 2.

    threshold = np.mean(dist) * k

    return threshold
def is_pos_def(A):

    if np.allclose(A, A.T):

        try:

            np.linalg.cholesky(A)

            return True

        except np.linalg.LinAlgError:

            return False

    else:

        return False
data_train = np.array(X_train_PCA.values)

cov_matrix, inv_cov_matrix  = cov_matrix(data_train)

mean_distr = data_train.mean(axis=0)

dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)

threshold = MD_threshold(dist_train, extreme = True)
threshold
plt.figure()

sns.distplot(np.square(dist_train),

             bins = 10, 

             kde= False);

plt.xlim([0.0,15])
plt.figure()

sns.distplot(dist_train,

             bins = 10, 

             kde= True, 

            color = 'green');

plt.xlim([0.0,5])

plt.xlabel('Mahalanobis dist')
threshold = 3.2
anomaly_train = pd.DataFrame()

anomaly_train['Mob dist']= dist_train

anomaly_train['Thresh'] = threshold

anomaly_train['Anomaly'] = anomaly_train['Mob dist'] > anomaly_train['Thresh']

anomaly_train.index = X_train_PCA.index
anomaly_train['Anomaly'].value_counts()
training_data = training_data[~anomaly_train['Anomaly']]
training_data.shape
target_variable_name = 'Attrition'
training_values = training_data[target_variable_name]
training_values.shape
training_points = training_data.drop(target_variable_name, axis=1)
training_points.shape
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = xgb.XGBClassifier(eval_metric='auc', objective='binary:logistic', alpha=1)
eclf = VotingClassifier(estimators=[ ('lr', clf1), ('rf', clf2),('xgb', clf3)], voting='soft')
eclf_params = {

    'lr__C': np.linspace(1, 3, 10),

    'lr__penalty':['l1', 'l2'],

    

    'rf__n_estimators' : list(range(50, 101, 10)),

    'rf__max_features': ['auto', 'log2'],

    

    'xgb__subsample': [0.8, 0.9],

    'xgb__colsample_bytree': np.linspace(0.6, 1, 5),

    'xgb__eta': np.linspace(0.001, 0.5, 20),

    'xgb__n_estimators': [100, 250, 500],

    'xgb__max_depth': [4, 5, 6],

    'xgb__min_child_weight': [2, 3, 4],

}
voting_clf = RandomizedSearchCV(eclf, eclf_params, random_state=0, scoring='roc_auc', cv=skfold)
%%time

voting_search = voting_clf.fit(training_points, training_values)
voting_search.best_score_
voting_search.best_params_
best_model = voting_search.best_estimator_
test_data = pd.read_csv(working_folder + 'test_data.csv')
test_data = test_data.drop('Unnamed: 0', axis = 1)
test_data.describe().T
test_data['BusinessTravel'] = test_data['BusinessTravel'].map(BusinessTravel_map)

test_data.drop('StandardHours', axis=1, inplace=True)

test_data.fillna(training_median, inplace=True)

test_data = pd.get_dummies(test_data, prefix_sep='_', drop_first=True)
id_variable_name = 'index'

ids = test_data[id_variable_name]

test_points = test_data.drop(id_variable_name, axis=1)
test_points.shape
test_predictions = best_model.predict_proba(test_points)[:, 1]
result = pd.DataFrame(columns=['index', 'Attrition'])
result['index'] = ids

result['Attrition'] = test_predictions
result.to_csv('result.csv', index=False)