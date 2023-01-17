from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import brier_score_loss, log_loss, average_precision_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

%matplotlib inline
# load dataset
df = pd.read_csv('../input/creditcard.csv')
X = df.copy().drop('Class', axis=1)
y = df.copy()['Class']
models = {
    'logreg': {'model': LogisticRegression(), 'params': {'solver': ['saga'], 'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]}},
    'rf': {'model': RandomForestClassifier(), 'params': {'n_estimators': [30], 'min_samples_split': [2, 4, 8, 12]}}
}

def get_model(model_dict, X, y, sample_type):
    if sample_type not in model_dict:
        if model_dict['params'] is not None:
            grid_model = GridSearchCV(model_dict['model'], model_dict['params'], cv=3)
            grid_model.fit(X, y)
            model_dict[sample_type] = grid_model.best_estimator_
            print('%s: %s' % (sample_type, grid_model.best_params_))
        else:
            # if a param grid isn't provide, use the model provided as is
            model_dict[sample_type] = model_dict['model']
    return model_dict[sample_type]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
sm = SMOTE(random_state=42, sampling_strategy='minority') # used for downsampling
nm = NearMiss(random_state=42) # used for upsampling

# put all this in a method so we don't have to type it out three times
def train_test_score(model, model_name, sample_type, X_train, y_train, X_test, y_test, scores):
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)
    pred_df = pd.DataFrame({'pred': [p[1] for p in pred], 'actual': y_test}, index=X_test.index).sort_values('pred', ascending=False)
    scores['logloss'][model_name][sample_type]['scores'].append(brier_score_loss(pred_df['actual'], pred_df['pred']))
    scores['AP'][model_name][sample_type]['scores'].append(average_precision_score(pred_df['actual'], pred_df['pred']))
    return scores

scores = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

for train, test in kf.split(X):
    X_train, X_test, y_train, y_test = X.iloc[train, :], X.iloc[test, :], y.iloc[train], y.iloc[test]
    for model_name, model_dict in models.items():
        model = get_model(model_dict, X_train, y_train, 'raw')
        scores = train_test_score(model, model_name, 'raw', X_train, y_train, X_test, y_test, scores)
        
        X_up, y_up = sm.fit_resample(X_train, y_train)
        model = get_model(model_dict, X_up, y_up, 'up')
        scores = train_test_score(model, model_name, 'up', X_up, y_up, X_test, y_test, scores)
        
        X_down, y_down = nm.fit_resample(X_train, y_train)
        model = get_model(model_dict, X_down, y_down, 'down')
        scores = train_test_score(model, model_name, 'down', X_down, y_down, X_test, y_test, scores)
# put the results into a dataframe
d = {'metric': [], 'model': [], 'sample': [], 'score': [], 'std': []}
for metric, scores1 in scores.items():
    for m, scores2 in scores1.items():
        for stype, scores3 in scores2.items():
            if len(scores3['scores']) > 0:
                d['metric'].append(metric)
                d['model'].append(m)
                d['sample'].append(stype)
                d['score'].append(np.mean(scores3['scores']))
                d['std'].append(np.std(scores3['scores']))
results_df = pd.DataFrame(d)
# plot average precision results
plot_ap_df = results_df.loc[results_df['metric']=='AP', ['model', 'sample', 'score', 'std']]
error_df = plot_ap_df.pivot(index='sample', columns='model', values='std')
plot_ap_df.pivot(index='sample', columns='model', values='score')\
    .plot(kind='bar', title='AP', yerr=error_df)
# plot logloss results
plot_logloss_df = results_df.loc[results_df['metric']=='logloss', ['model', 'sample', 'score', 'std']]
error_logloss_df = plot_logloss_df.pivot(index='sample', columns='model', values='std')
plot_logloss_df.pivot(index='sample', columns='model', values='score')\
    .plot(kind='bar', title='logloss', yerr=error_logloss_df)
# let's try plotting again without the downsampled results
plot_logloss_no_down_df = results_df.loc[
    (results_df['metric']=='logloss') & (results_df['sample']!='down'),
    ['model', 'sample', 'score', 'std']
]
error_logloss_df = plot_logloss_no_down_df.pivot(index='sample', columns='model', values='std')
plot_logloss_no_down_df.pivot(index='sample', columns='model', values='score')\
    .plot(kind='bar', title='logloss', yerr=error_logloss_df)
