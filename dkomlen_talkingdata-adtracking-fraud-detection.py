%matplotlib inline
%autosave 30

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import multiprocessing as mp
from tqdm import tqdm_notebook as tqdm

pd.set_option('display.max_rows', 10)
np.random.seed(42)
def read_dataset(path, nrows=None):
    """Reads dataset from CSV file, with optional number of rows"""
    
    df = pd.read_csv(path, nrows=nrows)
    df['click_time'] = pd.to_datetime(df['click_time'])
    df = df.set_index('click_time')
    df.sort_index(inplace=True)
    
    return df

# Training data set
dataset_train = read_dataset('../input/train_sample.csv')

# Benchmark test set, same as used in benchmark model: https://rpubs.com/el16/410747
dataset_test = read_dataset('../input/train.csv', nrows=1000000)

# Larger test set, used for Kaggle Leaderboard evaluation
dataset_test_all = read_dataset('../input/test.csv')
dataset_train.describe()
ser = dataset_train.nunique()
pd.DataFrame({'attribute':ser.index, 'unique values':ser.values})
dataset_train.hist(bins=20, figsize=(10,10));
corr = dataset_train.corr()
corr.style.background_gradient()
dataset_train[(dataset_train.is_attributed == 1)].hist(bins=20, figsize=(10,10));
top_apps = dataset_train.groupby(['ip'])['is_attributed'].agg(
    {"is_attributed": sum}).sort_values(
    "is_attributed", ascending=False).head(10).reset_index()
top_apps
top_apps = dataset_train.groupby(['app','channel'])['is_attributed'].agg(
    {"is_attributed": sum}).sort_values(
    "is_attributed", ascending=False).head(10).reset_index()
top_apps
dataset_train.apply(lambda x: 1 if x['is_attributed'] == 1 else 0, axis=1).plot(figsize=(20, 5));
dataset_train.apply(lambda x: 1 if x['app'] == 19 else 0, axis=1).plot(figsize=(20, 5));
dataset_train.apply(lambda x: 1 if x['channel'] == 213 and x['app'] == 19 else 0, axis=1).plot(figsize=(20, 5));
dataset_test.head()
benchmark_const = dataset_test_all[['click_id']].copy()
benchmark_const['is_attributed'] = 0
display(benchmark_const)
benchmark_rand = dataset_test_all[['click_id']].copy()
benchmark_rand['is_attributed'] = np.random.uniform(size=len(benchmark_rand))
display(benchmark_rand)
from sklearn import preprocessing

def prepare_column(job):
    """Generate new features from target column"""
    
    df = job['df']
    col = job['feat']
    
    print ('Preparing column: ', col)
    def calculate_last(row, attr, attr_map):
        attr_val = row[attr]
        if attr_val in attr_map:
            st = attr_map.get(attr_val)
            et = row['click_time']
            val = min((et - st).total_seconds(), 86400)
        else:
            val = 86400
        attr_map[attr_val] = row['click_time']
        return val

    new_col = 'last_'+col
    df[new_col] = df.apply(calculate_last, axis=1, attr=col, attr_map={})
    x = df[[new_col]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[new_col] = x_scaled
    return df[new_col]

def prepare_dataset(df):
    """Preprocess raw input dataset to enhanced dataset with additional features"""
    
    df['hour'] = df.index.hour
    df['click_time'] = df.index
    pool = mp.Pool(maxtasksperchild=1000)

    jobs = [{'df':df,'feat':feat} for feat in ['ip', 'app', 'os', 'channel', 'device']]
    with tqdm(total=len(jobs), desc="Preparing features") as pbar:
        for feat in pool.imap(prepare_column, jobs):
            df = pd.concat([df, feat], axis=1)
            pbar.update()
            
    pool.close()
    pool.join()

    return df.drop(labels=['is_attributed', 'attributed_time', 'click_time'], axis=1, errors='ignore')
print('Preparing train dataset')
train_y = dataset_train['is_attributed']
%time train_X = prepare_dataset(dataset_train)

print('Preparing test dataset')
test_y = dataset_test['is_attributed']
%time test_X = prepare_dataset(dataset_test)
train_X['last_app'].plot(figsize=(20, 5));
train_X.tail()
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

def evaluate_model(clf, tstX=test_X, tsty=test_y, verbose=True):
    """Evaluate model using AUC score on given test data"""
    
    predicted_proba_y = clf.predict_proba(tstX)[:,1]
    predicted_y = clf.predict(tstX)
    clf_name = str(clf).split('(')[0]
    if verbose:
        display(clf)
        print(clf_name + ' confusion matrix:')
        display(pd.DataFrame(confusion_matrix(tsty, predicted_y)))
    return [{'clf': clf_name,
            'auc': roc_auc_score(tsty, predicted_proba_y)}]

# Initial model evaluation using fixed random state
results = []
for clf in [LogisticRegression(random_state=42), 
            GaussianNB(), 
            tree.DecisionTreeClassifier(random_state=42),
            GradientBoostingClassifier(random_state=42),
            RandomForestClassifier(random_state=42)]:
    clf.fit(train_X, train_y)
    results += evaluate_model(clf)
    
results = pd.DataFrame(results).sort_values(by=['auc'])

display(results)
results.plot.bar(x='clf');
%%time

# https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
from sklearn.externals import joblib

def optimize_model(estimator, params, cv):
    scorer = make_scorer(roc_auc_score)
    gs = GridSearchCV(estimator=estimator, param_grid=params, 
                      scoring=scorer, cv=cv, verbose=3,
                      n_jobs=-1)
    return gs.fit(train_X, train_y)

cv = StratifiedKFold(n_splits=5, shuffle=True)
model = optimize_model(GradientBoostingClassifier(random_state=42), {
        'loss' : ['deviance', 'exponential'],
        'learning_rate': [0.1, 0.2, 0.3],
        'n_estimators': [50, 100, 200],
    }, cv=cv)

joblib.dump(model, 'final_model.joblib') 
display(pd.DataFrame(evaluate_model(model)))
model.estimator
dataset_test_sens = dataset_test.copy()

time_delta = pd.Series([ pd.Timedelta(minutes=np.random.randint(-120, 121)) for i in range(len(test_X)) ])
dataset_test_sens.index = dataset_test_sens.index + time_delta
dataset_test_sens['click_time'] = dataset_test_sens.index
dataset_test_sens.sort_index(inplace=True)
display(dataset_test_sens.head())
display(dataset_test.head())
print('Preparing modified test dataset')
test_sens_y = dataset_test_sens['is_attributed']
%time test_sens_X = prepare_dataset(dataset_test_sens)

test_sens_X.head()
from sklearn.externals import joblib

model = joblib.load('final_model.joblib')

display(pd.DataFrame(evaluate_model(model, test_sens_X, test_sens_y)))
from sklearn.externals import joblib
from sklearn.utils import resample

n_bootstraps = 1000
bootstraps = []
model = joblib.load('final_model.joblib')

def bootstrap_score(i):
    sample_X, sample_y = resample(test_X, test_y)
    res = evaluate_model(model, sample_X, sample_y, False)
    return res[0]['auc']

with tqdm(total=n_bootstraps, desc="Preparing bootstraps") as pbar:
    pool = mp.Pool(maxtasksperchild=1000)

    for bootstrap in pool.imap(bootstrap_score, range(n_bootstraps)):
        bootstraps.append(bootstrap)
        pbar.update()
    pool.close()
    pool.join()
        
pd.DataFrame(bootstraps).hist();

alpha = 0.95
p = ((1.0 - alpha) / 2.0) * 100
lower = max(0.0, np.percentile(bootstraps, p))
p = (alpha + ((1.0 - alpha) / 2.0)) * 100
upper = min(1.0, np.percentile(bootstraps, p))
print('%.1f confidence interval %.2f%% and %.2f%%' % (alpha*100, lower*100, upper*100))