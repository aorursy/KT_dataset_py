import scipy
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import scale
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer

from skopt import BayesSearchCV

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import scikitplot as skplt
%matplotlib inline
# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# Get the combined data
total_df = pd.concat([train_df.drop('target', axis=1), test_df], axis=0).drop('ID', axis=1)
# Columns to drop because there is no variation in training set
zero_std_cols = train_df.drop("ID", axis=1).columns[train_df.std() == 0]
total_df.drop(zero_std_cols, axis=1, inplace=True)
print(f">> Removed {len(zero_std_cols)} constant columns")

# Removing duplicates
# Taken from: https://www.kaggle.com/scirpus/santander-poor-mans-tsne
colsToRemove = []
colsScaned = []
dupList = {}
columns = total_df.columns
for i in tqdm(range(len(columns)-1)):
    v = train_df[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, train_df[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
colsToRemove = list(set(colsToRemove))
total_df.drop(colsToRemove, axis=1, inplace=True)
print(f">> Dropped {len(colsToRemove)} columns")
# Go through the columns one at a time (can't do it all at once for this dataset)
for col in tqdm(total_df.columns):
    
    # Detect outliers in this column
    data = total_df[col].values
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers = [x for x in data if x < lower or x > upper]
    
    # If there are crazy high values, do a log-transform
    if len(outliers) > 0:
        non_zero_idx = data != 0
        total_df.loc[non_zero_idx, col] = np.log(data[non_zero_idx])
    
    # Scale non-zero column values
    nonzero_rows = total_df[col] != 0
    total_df.loc[nonzero_rows, col] = scale(total_df.loc[nonzero_rows, col])
COMPONENTS = 10

# List of decomposition methods to use
methods = [
    TruncatedSVD(n_components=COMPONENTS),
    PCA(n_components=COMPONENTS),
    FastICA(n_components=COMPONENTS),
    GaussianRandomProjection(n_components=COMPONENTS, eps=0.1),
    SparseRandomProjection(n_components=COMPONENTS, dense_output=True)    
]

# Run all the methods
embeddings = []
for method in methods:
    name = method.__class__.__name__    
    embeddings.append(
        pd.DataFrame(method.fit_transform(total_df), columns=[f"{name}_{i}" for i in range(COMPONENTS)])
    )
    print(f">> Ran {name}")
    
# Put all components into one dataframe
components_df = pd.concat(embeddings, axis=1).reset_index(drop=True)
aggregate_df = pd.DataFrame()

# V1 Features
aggregate_df['mean'] = total_df.mean(axis=1)
aggregate_df['median'] = total_df.median(axis=1)
aggregate_df['std'] = total_df.std(axis=1)
aggregate_df['min'] = total_df.min(axis=1)
aggregate_df['max'] = total_df.max(axis=1)
aggregate_df['number_of_different'] = total_df.nunique(axis=1)
aggregate_df['non_zero_count'] = total_df.astype(bool).sum(axis=1) 
aggregate_df['sum_zeros'] = (total_df == 0).astype(int).sum(axis=1)
aggregate_df['geometric_mean'] = total_df.apply(
    lambda x: np.exp(np.log(x[x>0]).mean()), axis=1
)
aggregate_df.reset_index(drop=True, inplace=True)
print(">> Created features for; mean, median, std, min, max, number_of_different, non_zero_count, sum_zeros, geometric_mean")
# Concate and split dataset
X = pd.concat([components_df, aggregate_df], axis=1).fillna(0)
X_train = X.iloc[0:len(train_df)]
X_test = X.iloc[len(train_df):]
y = train_df['target']
# Create forward feature selector
selector = SFS(
    ElasticNet(alpha=1, l1_ratio=0.5),
    k_features=(10,20),
    forward=True,
    floating=True,
    scoring='neg_mean_squared_error',
    cv=10,
    n_jobs=-1, 
    verbose=1
)

# Fit model and get best features
selector.fit(X_train.values, np.log1p(y))

# Let the user know which features were selected and how many
print(f">> Selected the following {len(selector.k_feature_idx_)} features: {X_train.columns[list(selector.k_feature_idx_)]}")
# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = ElasticNet(),
    search_spaces = {
        'alpha': (0.5, 500, 'log-uniform'),
        'l1_ratio': (0.25, 0.75),
        'fit_intercept': [True, False]
    },    
    scoring = 'neg_mean_squared_error',
    cv = KFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 1,
    n_iter = 40,   
    verbose = 0,
    refit = True,
    random_state = 42
)

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest mean_squared_error: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")

# Fit the model - uncomment to run
result = bayes_cv_tuner.fit(
    X_train.values[:, selector.k_feature_idx_], 
    np.log1p(y), 
    callback=status_print
)
# Fit model on full training set
print(">> Fitting with parameters: ", bayes_cv_tuner.best_params_)
regr = ElasticNet(**bayes_cv_tuner.best_params_)
regr.fit(X_train.values[:, selector.k_feature_idx_], np.log(y))

# Get submission file
submission_file = pd.read_csv('../input/sample_submission.csv')
submission_file.target = np.exp(
    regr.predict(X_test.values[:, selector.k_feature_idx_])
)

print(">> Saving submission file")
submission_file.to_csv('submission_baseline_elasticnet.csv', index=False)