# Data engineering libraries
import numpy as np
import pandas as pd

# Python default libraries
import os
import warnings
import itertools
import joblib

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Statistical functions
from scipy import stats

# imbalanced learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Model selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

# Machine Learning libraries
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier

# Metrics libraries
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

# Interpretability libraries
from sklearn.inspection import permutation_importance

# Notebook settings
%config Completer.use_jedi = False
%matplotlib inline
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
# Set random state to allow reproducible results
rs = 42
# Create function to divide the dataset into different train-test splits and assess the average performance of the model
def mean_performance(features_dataset, target_dataset, number_of_splits, model, percentage_test=0.2, labels=None):
    # Define empty lists to serve as output
    labels = [0, 1] if labels is None else sorted(labels)
    metrics = ['precision', 'recall', 'f1', 'auc']
    output_dict = dict()
    for label in labels:
        for m in metrics:
            output_dict[m + '_class_' + str(label)] = list()
        
    # initialize model
    model = model.estimator.set_params(**model.best_params_)
    
    for n in range(1, number_of_splits + 1):
        # Random train test split
        X_train, X_test, y_train, y_test = train_test_split(features_dataset, target_dataset, test_size=percentage_test, stratify=target_dataset)
        
        # Take the best model and fit it to the training set
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        probas = model.predict_proba(X_test)

        # Calculate performance metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, labels=labels)
        
        # Break the metrics by class and append results
        dummies = pd.get_dummies(y_test)
        for label in labels:
            if label not in dummies.columns:
                dummies[label] = 0
            auc = roc_auc_score(dummies[label], probas[:, label])
            
            output_dict['precision_class_' + str(label)].append(precision[label])
            output_dict['recall_class_' + str(label)].append(recall[label])
            output_dict['f1_class_' + str(label)].append(f1[label])
            output_dict['auc_class_' + str(label)].append(auc)
       
    # calculate the average results
    avg = dict()
    for label in labels:
        for m in metrics:
            avg['avg_' + m + '_class_' + str(label)] = np.mean(output_dict[m + '_class_' + str(label)])
    
    return avg
# Create function to calculate the permutated feature importance across different train-test splits
def permutated_feature_importance(features_dataset, target_dataset, input_columns, number_of_splits, 
                                  model, score, percentage_test=0.2):
    # Define output dataframe
    output_feature = pd.DataFrame()
    std_feature = pd.DataFrame()
    
    # Create column with the name of the feature
    output_feature['FEATURE'] = input_columns
    std_feature['FEATURE'] = input_columns
    
    # initialize model
    model = model.estimator.set_params(**model.best_params_)
    
    for n in range(1, number_of_splits + 1):
        # Stratified train test split
        X_train, X_test, y_train, y_test = train_test_split(features_dataset, target_dataset, test_size=percentage_test,
                                                    stratify=target_dataset)
        # Take the best model and fit it to the training set
        model.fit(X_train, y_train)
        
        # Dictionary of permutation
        permutated_dict = permutation_importance(model, features_dataset, target_dataset, scoring=score, random_state=rs)
        output_feature[n] = permutated_dict['importances_mean']
        std_feature[n] = permutated_dict['importances_std']
    
    # Calculate average importance and its std across all splits
    output_feature['AVG_IMPORTANCE_' + str(number_of_splits)] = output_feature.mean(axis=1)
    std_feature['AVG_STD_' + str(number_of_splits)] = std_feature.mean(axis=1)
    
    # Merge std avg with final output frame
    output_feature = output_feature.merge(std_feature[['FEATURE', 'AVG_STD_' + str(number_of_splits)]], on=['FEATURE'])
    
    # Sort values by average importance
    output_feature = output_feature.sort_values(by=['AVG_IMPORTANCE_' + str(number_of_splits)], ascending=False)
        
    return output_feature
def get_est_grid(grid, m):
    """
    Get the equivalent grid dictionary for a One vs All estimator
    :param grid: (dict) original grid dictioanry
    :param m   : (string) model name
    return: (dict) grid for One vs All
    """
    grid2 = dict()
    for k in grid:
        if m in k:
            grid2[k.replace(m + '__', m + '__estimator__')] = grid[k]
        else:
            grid2[k] = grid[k]
    return grid2
# function to calculate correlation between categorical variables
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2)/(n - 1)
    kcorr = k - ((k - 1) ** 2)/(n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
raw_covid = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
raw_covid.sample(10)
raw_covid.shape
missing = (raw_covid.isnull().sum()/raw_covid.shape[0]*100).sort_values(ascending=False).reset_index()
missing.columns = ['VARIABLE', 'PERC_MISSING']
missing.head(10)
missing['QUANTILE_MISSING'] = pd.cut(missing['PERC_MISSING'], 10)
sns.countplot(y='QUANTILE_MISSING', data=missing)
msno.matrix(raw_covid, sparkline=False)
plt.xticks(rotation='vertical')
raw_covid.dtypes
numerical_columns = raw_covid.select_dtypes(exclude='object').columns
print('There are', len(numerical_columns), 'numerical columns in the dataset')

raw_covid[numerical_columns].describe()
categorical_columns = raw_covid.select_dtypes(include='object').columns
print('There are', len(categorical_columns), 'categorical columns in the dataset')

raw_covid[categorical_columns].describe()
plt.figure(figsize = (12, 5))
sns.countplot(x = 'SARS-Cov-2 exam result', data = raw_covid)
raw_covid['SARS-Cov-2 exam result'].value_counts(normalize = True)*100
for col in [c for c in raw_covid.columns if 'Patient ID' not in c]:
    print(col, ':', raw_covid[col].nunique())
corr = raw_covid.corr()
plt.figure(figsize=(20, 8))

# Heatmap of correlations
sns.heatmap(corr, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=False, vmax=0.8)
# melt the matrix columns to get pair-wise correlation table
corr = pd.melt(corr.reset_index(), id_vars='index', value_name='corr')
    
# remove rows with the same variable
corr = corr[corr['index'] != corr['variable']]
    
# drop duplicates on pairwise columns
corr = corr.loc[
    pd.DataFrame(
        np.sort(corr[['index', 'variable']], 1), index=corr.index
    ).drop_duplicates(keep='first').index
]

# sort values by absolute correlation values
corr['corr'] = corr['corr'].abs()
corr.sort_values(by='corr', ascending=False, inplace=True)

# show top n rows
corr.head(60)
# list of columns to exclude from the correlation analysis
exclude_columns = [
    'Patient ID', 
    'SARS-Cov-2 exam result',
    'Patient addmited to regular ward (1=yes, 0=no)',
    'Patient addmited to semi-intensive unit (1=yes, 0=no)',
    'Patient addmited to intensive care unit (1=yes, 0=no)',
    
    # raises an error because of the number of null values
    'Urine - Esterase', 'Urine - Aspect', 'Urine - pH','Urine - Hemoglobin',
    'Urine - Bile pigments', 'Urine - Ketone Bodies', 'Urine - Nitrite', 
    'Urine - Density', 'Urine - Urobilinogen', 'Urine - Protein', 'Urine - Sugar',
    'Urine - Leukocytes', 'Urine - Crystals', 'Urine - Red blood cells', 
    'Urine - Hyaline cylinders', 'Urine - Granular cylinders', 'Urine - Yeasts',
    'Urine - Color'
]

cols = [c for c in categorical_columns if c in raw_covid.columns and c not in exclude_columns]

# calculate the correlation for each pair of variables
corr = list()
for c1, c2 in list(itertools.combinations(cols, 2)):
    corr.append([c1, c2, cramers_v(raw_covid[c1].values, raw_covid[c2].values)])

# show the table of correlations    
corr = pd.DataFrame(data=corr, columns=['col1', 'col2', 'corr'])
corr.sort_values(by='corr', ascending=False)
# columns to exclude from the process
exclude_columns = [
    'Patient ID',
    'SARS-Cov-2 exam result',
    'Patient addmited to regular ward (1=yes, 0=no)',
    'Patient addmited to semi-intensive unit (1=yes, 0=no)',
    'Patient addmited to intensive care unit (1=yes, 0=no)'
]
cleaned_covid = raw_covid.copy()
# Urine - pH
cleaned_covid.loc[cleaned_covid['Urine - pH'] == 'Não Realizado', 'Urine - pH'] = '-9'
cleaned_covid.loc[cleaned_covid['Urine - pH'] == 'not_done', 'Urine - pH'] = '-9'
cleaned_covid['Urine - pH'] = cleaned_covid['Urine - pH'].astype(np.float)

# Urine - Leukocytes
# We will input the <1000 entries with the midpoint 500. Note that there is no entry with this specific value
cleaned_covid.loc[cleaned_covid['Urine - Leukocytes'] == '<1000', 'Urine - Leukocytes'] = '500'
cleaned_covid.loc[cleaned_covid['Urine - Leukocytes']=='not_done', 'Urine - Leukocytes'] = '-9'
cleaned_covid['Urine - Leukocytes'] = cleaned_covid['Urine - Leukocytes'].astype(float)

# Urine - Crystals
rep = {'á': 'a', '-': 'Minus', '+': 'Plus'}
for key, value in rep.items():
    cleaned_covid['Urine - Crystals'] = cleaned_covid['Urine - Crystals'].str.replace(key, value)
# List of columns related to tests
tests_columns = [c for c in cleaned_covid.columns if c not in exclude_columns and c not in ['Patient age quantile']]

# Dataframe of patients without any test result
cleaned_covid_no_test = cleaned_covid[cleaned_covid[tests_columns].isnull().all(axis=1)].copy()
cleaned_covid_no_test.dropna(how='all', axis='columns', inplace=True)

# Dataframe of patients with test results
cleaned_covid_test = cleaned_covid[~(cleaned_covid['Patient ID'].isin(cleaned_covid_no_test['Patient ID']))].copy()
cleaned_covid_test['SARS-Cov-2 exam result'].value_counts()
cleaned_covid.dropna(how='all', axis=1, inplace=True)
# plot the amount of columns dropped by the fill percentage required
task1_ratio = cleaned_covid_test.count() / cleaned_covid_test.shape[0]
task2_ratio = cleaned_covid_test[cleaned_covid_test['SARS-Cov-2 exam result'] == 'positive']
task2_ratio = task2_ratio.count() / task2_ratio.shape[0]
ratio = pd.concat([task1_ratio, task2_ratio], axis='columns')
res = list()
for thrs in np.linspace(0.01, 0.3, 50):
    res.append([thrs, ratio[(ratio[0] < thrs) & (ratio[1] < thrs)].shape[0]])
fig = plt.figure(figsize=(16,4))
ax = fig.gca()
ax.set_xticks(np.linspace(0.01, 0.3, 20))
plt.scatter(np.array(res)[:, 0], np.array(res)[:, 1])
plt.grid()
# apply column removal
thrs = 0.05
task1_ratio = cleaned_covid_test.count() / cleaned_covid_test.shape[0]
task2_ratio = cleaned_covid_test[cleaned_covid_test['SARS-Cov-2 exam result'] == 'positive']
task2_ratio = task2_ratio.count() / task2_ratio.shape[0]
ratio = pd.concat([task1_ratio, task2_ratio], axis='columns')
cleaned_covid_test.drop(columns=ratio[(ratio[0] < thrs) & (ratio[1] < thrs)].index, inplace=True)
corr = cleaned_covid.drop(columns=exclude_columns).corr()
corr = pd.melt(corr.reset_index(), id_vars='index', value_name='corr')
corr = corr[corr['index'] != corr['variable']]
corr = corr.loc[pd.DataFrame(np.sort(corr[['index', 'variable']], 1), index=corr.index).drop_duplicates(keep='first').index]
corr['corr'] = corr['corr'].abs()
res = list()
for thrs in np.linspace(0.7, 1, 32):
    x = corr[corr['corr'] > thrs]
    drop_corr = list()
    while x.shape[0] > 0:
        for c in x['index'].append(x['variable']).unique():
            v1 = x[(x['variable'] == c) | (x['index'] == c)]
            
            if v1.shape[0] > 0:
                c2 = v1['index'].values[0] if v1['index'].values[0] != c else v1['variable'].values[0]
                v2 = x[(x['variable'] == c2) | (x['index'] == c2)]
                
                drop = c if (v1.shape[0] > v2.shape[0]) and (cleaned_covid[c].count() < cleaned_covid[c2].count()) else c2
                drop_corr.append(drop)
                x = x[(x['index'] != drop)& (x['variable'] != drop)]
    res.append((thrs, cleaned_covid.shape[1] - len(drop_corr)))
plt.plot(np.array(res)[:, 0], np.array(res)[:, 1])
# stablish correlation threshold to drop variables
thrs = 0.9

# calculate correlation matrix
corr = cleaned_covid_test.drop(columns=exclude_columns).corr()

# transform columns into rows
corr = pd.melt(corr.reset_index(), id_vars='index', value_name='corr')

# remove correlation of same variable
corr = corr[corr['index'] != corr['variable']]
corr['corr'] = corr['corr'].abs()

# drop pair-wise duplicates
corr = corr.loc[
    pd.DataFrame(
        np.sort(corr[['index', 'variable']], 1), index=corr.index
    ).drop_duplicates(keep='first').index
]

# select pairs above threshold
x = corr[corr['corr'] > thrs].sort_values('corr', ascending=False)
drop_corr = list()

# for each pair
while x.shape[0] > 0:
    for c in x['index'].append(x['variable']).unique():
        # verify number of ocurrencces of a feature in the table
        v1 = x[(x['variable'] == c) | (x['index'] == c)]
        
        if v1.shape[0] > 0:
            c2 = v1['index'].values[0] if v1['index'].values[0] != c else v1['variable'].values[0]
            v2 = x[(x['variable'] == c2) | (x['index'] == c2)]

            # if the first features have more occurences than the second, that means that the first variable
            # can be explained by multiple variables, therefore we will prefer the second which adds more
            # variance to the dataset
            drop = c if (v1.shape[0] > v2.shape[0]) and (cleaned_covid[c].count() < cleaned_covid[c2].count()) else c2
            drop_corr.append(drop)
            x = x[(x['index'] != drop)& (x['variable'] != drop)]

# drop the selected columns
cleaned_covid_test.drop(drop_corr, axis=1, inplace=True)
drop_corr
# List of columns with null entries
null_columns = [c for c in cleaned_covid_test.columns if cleaned_covid_test[c].isnull().sum() != 0]

# List of columns to be filled with -9
columns_num_fill = [c for c in null_columns if c in cleaned_covid_test.select_dtypes(exclude='object').columns]

# List of columns to be filled with "not_done"
columns_cat_fill = [c for c in null_columns if c in cleaned_covid_test.select_dtypes(include='object').columns]

# Impute missing values
for col in columns_num_fill:
    cleaned_covid_test[col].fillna(-9, inplace=True)
    
for col in columns_cat_fill:
    cleaned_covid_test[col].fillna('not_done', inplace=True)
# SARS-Cov-2 exam result
cleaned_covid_no_test.loc[cleaned_covid_no_test['SARS-Cov-2 exam result']=='negative', 'SARS-Cov-2 exam result'] = 0
cleaned_covid_no_test.loc[cleaned_covid_no_test['SARS-Cov-2 exam result']=='positive', 'SARS-Cov-2 exam result'] = 1
cleaned_covid_no_test['SARS-Cov-2 exam result'] = cleaned_covid_no_test['SARS-Cov-2 exam result'].astype(np.int)

cleaned_covid_test.loc[cleaned_covid_test['SARS-Cov-2 exam result']=='negative', 'SARS-Cov-2 exam result'] = 0
cleaned_covid_test.loc[cleaned_covid_test['SARS-Cov-2 exam result']=='positive', 'SARS-Cov-2 exam result'] = 1
cleaned_covid_test['SARS-Cov-2 exam result'] = cleaned_covid_test['SARS-Cov-2 exam result'].astype(np.int)

# One hot encode categorical columns
categorical = [c for c in cleaned_covid_test.select_dtypes(include='object').columns if 'Patient ID' not in c]
encoded_covid_test = pd.get_dummies(cleaned_covid_test[categorical], drop_first=True)
cleaned_covid_test.drop(categorical, axis=1, inplace=True)
cleaned_covid_test = pd.concat([cleaned_covid_test, encoded_covid_test], axis=1)
# Create features and target for dataframe with no tests
features_no_test_task1 = cleaned_covid_no_test.drop(exclude_columns, axis = 1)
target_no_test_task1 = cleaned_covid_no_test['SARS-Cov-2 exam result']

# Create features and target for dataframe with tests
features_covid_task1 = cleaned_covid_test.drop(exclude_columns, axis = 1)
target_covid_task1 = cleaned_covid_test['SARS-Cov-2 exam result']
units = cleaned_covid_test[cleaned_covid_test['SARS-Cov-2 exam result'] == 1]
features_covid_task2 = units.drop(exclude_columns, axis = 1)

target_cols = [
    'Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 
    'Patient addmited to intensive care unit (1=yes, 0=no)'
]
target_covid_task2 = units[target_cols[0]] + units[target_cols[1]] * 2 + units[target_cols[2]] * 3
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source
from IPython.display import SVG
dt = Pipeline([('smt', SMOTE(0.5, random_state=rs)), ('dt', DecisionTreeClassifier(random_state=rs, max_depth=3))])
dt.fit(features_covid_task1, target_covid_task1)
dt_feat = pd.DataFrame(dt.named_steps['dt'].feature_importances_, index=features_covid_task1.columns, columns=['feat_importance'])
dt_feat.sort_values('feat_importance').tail(8).plot.barh()
plt.show()
graph = Source(export_graphviz(dt.named_steps['dt'], out_file=None, feature_names=features_covid_task1.columns, filled=True))
SVG(graph.pipe(format='svg'))
graph.render()
# Define param grid
smt_grid = {
    'smt__sampling_strategy': [0.2, 0.3, 0.4, 0.5],
    'smt__k_neighbors' : [2, 3, 5],
    'smt__random_state': [rs]
}

rf_grid = {
    'rf__n_estimators' : [int(x) for x in np.linspace(100, 2000, 20)],
    'rf__max_features' : ['auto', 'sqrt', 'log2'],
    'rf__min_samples_leaf': [2, 5, 10], # we set the minimum of samples leaf to minimize overfit 
    'rf__min_samples_split': [5, 10, 15], # we set the minimum of samples split to minimize overfit
    'rf__max_depth': [5, 8, 15], # we limited the max depth to 15 given the risk of overfit
    'rf__random_state' : [rs]
}
rf_grid.update(smt_grid)

xgb_grid = {
    'xgb__loss' : ['ls', 'lad', 'huber', 'quantile'],
    'xgb__n_estimators': [int(x) for x in np.linspace(100, 2000, 20)],
    'xgb__learning_rate': np.linspace(0.01, 0.1, 10),
    'xgb__max_depth': range(3, 10),
    'xgb__subsample': [0.8, 0.85, 0.9, 0.95, 1],
    'xgb__max_features': ['auto', 'sqrt'],
    'xgb__min_samples_leaf': [2, 5, 10], # we set the minimum of samples split to minimize overfit
    'xgb__min_samples_split': [5, 10, 15, 30], # we limited the max depth to 15 given the risk of overfit
    'xgb__random_state': [rs]
}
xgb_grid.update(smt_grid)

lgbm_grid ={
    'lgbm__num_leaves': stats.randint(6, 50), 
    'lgbm__min_child_samples': stats.randint(100, 500), 
    'lgbm__min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
    'lgbm__subsample': stats.uniform(loc=0.2, scale=0.8), 
    'lgbm__colsample_bytree': stats.uniform(loc=0.4, scale=0.6),
    'lgbm__reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
    'lgbm__reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
    'lgbm__random_state': [rs]
}
lgbm_grid.update(smt_grid)

# list models to run
# NOTE: Since the lgbm model is much faster to run, we are going to use it for now, but those lines should be uncommented
# when thinking of a model deployment
models_task1 = [
    ('RF', 'rf', RandomForestClassifier, rf_grid),
    ('XGB', 'xgb', XGBClassifier, xgb_grid),
    ('LGBM', 'lgbm', LGBMClassifier, lgbm_grid),
]

# dictionary of results
results_task1 = dict()

# set the metric to select the best model
MODEL_SELECTION_METRIC_TASK1 = 'recall'
# go through the models
for m, n, c, g in models_task1:
    print(m)

    # instantiate pipeline
    p = Pipeline([('smt', SMOTE(random_state=rs)), (n, c())])

    # run a grid search
    results_task1[m] = RandomizedSearchCV(
        p, g, cv=10, scoring=MODEL_SELECTION_METRIC_TASK1, verbose=1, n_jobs=-1, n_iter=300, random_state=rs, refit=False
    )
    results_task1[m].fit(features_covid_task1.values, target_covid_task1.values)

    # print out the model performance
    print('Best %s %s score:' % (m, MODEL_SELECTION_METRIC_TASK1), results_task1[m].best_score_)
    print('\n')
print('-' * 100)

# select the model with the highest score
models = [results_task1[m] for m in results_task1]
scores = [results_task1[m].best_score_ for m in results_task1]
i = np.argmax(scores)
best_model_task1 = models[i]

avg = mean_performance(
    features_covid_task1.values, target_covid_task1.values, 500, 
    model=best_model_task1, percentage_test=0.1,
)
for k in avg:
    print(k + ':', '%.2f' % avg[k])
print('-' * 100)
print('\n')
interp_task1 = permutated_feature_importance(
    features_covid_task1.values, target_covid_task1, features_covid_task1.columns, 50, best_model_task1, score = MODEL_SELECTION_METRIC_TASK1
) 
sns.barplot(x='AVG_IMPORTANCE_50', y='FEATURE', data=interp_task1.head(10))
# list models to run
# NOTE: Since the lgbm model is much faster to run, we are going to use it for now, but those lines should be uncommented
# when thinking of a model deployment
models_task2 = [
    ('OVA_RF', 'rf', lambda: OneVsRestClassifier(RandomForestClassifier()), get_est_grid(rf_grid, 'rf')),
    ('RF', 'rf', RandomForestClassifier, rf_grid),
    ('OVA_XGB', 'xgb', lambda: OneVsRestClassifier(XGBClassifier()), get_est_grid(xgb_grid, 'xgb')),
    ('XGB', 'xgb', XGBClassifier, xgb_grid),
    ('OVA_LGBM', 'lgbm', lambda: OneVsRestClassifier(LGBMClassifier()), get_est_grid(lgbm_grid, 'lgbm')),
    ('LGBM', 'lgbm', LGBMClassifier, lgbm_grid)
]

# dictionary of results
results_task2 = dict()

# set the metric to select the best model
MODEL_SELECTION_METRIC_TASK2 = 'recall_weighted'
# go through the models
for m, n, c, g in models_task2:
    print(m)

    # instantiate pipeline
    p = Pipeline([('smt', SMOTE(random_state=rs)), (n, c())])

    # we change the sampling strategy based on the fact that we have a multi-class classification problem
    g['smt__sampling_strategy'] = ['not majority']
    
    # we also limit the number of knn given that we may not have enough sample depending on the random split
    g['smt__k_neighbors'] = [2]
    
    # run a grid search
    # NOTE: we reduced the number of iterations just to deploy the notebook faster. But looking forward this number should be 200~300
    results_task2[m] = RandomizedSearchCV(
        p, g, cv=10, scoring=MODEL_SELECTION_METRIC_TASK2, verbose=1, n_jobs=-1, n_iter=300, random_state=rs, refit=False
    )
    results_task2[m].fit(features_covid_task2.values, target_covid_task2.values)

    # print out the model performance
    print('Best %s %s score:' % (m, MODEL_SELECTION_METRIC_TASK2), results_task2[m].best_score_)
    print('\n')
print('-' * 100)

# select the model with the highest score
models = [results_task2[m] for m in results_task2]
scores = [results_task2[m].best_score_ for m in results_task2]
i = np.argmax(scores)
best_model_task2 = models[i]

avg = mean_performance(
    features_covid_task2.values, target_covid_task2.values, 300, 
    model=best_model_task2, percentage_test=0.1, labels=list(target_covid_task2.unique())
)
for k in avg:
    print(k + ':', '%.2f' % avg[k])
print('-' * 100)
print('\n')
interp_task2 = permutated_feature_importance(
    features_covid_task2.values, target_covid_task2, features_covid_task2.columns, 50, best_model_task2, score = MODEL_SELECTION_METRIC_TASK2
) 
sns.barplot(x='AVG_IMPORTANCE_50', y='FEATURE', data=interp_task2.head(10))
n_estimators = 400
max_samples = 0.9

# instantiate the bagging classifier
model1 = best_model_task1.estimator.set_params(**best_model_task1.best_params_)
bag1 = BaggingClassifier(model1, n_estimators=n_estimators, max_samples=max_samples)

model2 = best_model_task2.estimator.set_params(**best_model_task2.best_params_)
bag2 = BaggingClassifier(model2, n_estimators=n_estimators, max_samples=max_samples)

# train and export model
bag1.fit(features_covid_task1.values, target_covid_task1.values)
joblib.dump(bag1, 'model_task1.pkl', compress=9)

# train and export model
# NOTE: We had to comment the second model because there are very few positive examples depending on the class
# because of that we cannot run the bag with SMOTE. We either have to change our sampler or increase the number of data points
# bag2.fit(features_covid_task2.values, target_covid_task2.values)
# joblib.dump(bag2, 'model_task2.pkl', compress=9)