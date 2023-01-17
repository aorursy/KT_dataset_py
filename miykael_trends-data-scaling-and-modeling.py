import numpy as np

import pandas as pd

from os.path import join as opj

from tqdm.notebook import tqdm
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

sns.set_context('notebook')
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge

from sklearn.svm import SVR
path = '/kaggle/input/trends-feature-exploration-engineering/datasets'
# Load target values and corresponding scaler

import joblib

scaler_targets = joblib.load(opj(path, 'targets_scaler.pkl'))



targets = pd.read_hdf(opj(path, 'targets.h5'))

targets.head()
def load_dataset_and_scale(target, dataset_id='merge', scale_values=[1, 1, 1, 1]):



    # Load dataset

    X_tr = pd.read_hdf(opj(path, '%s_train.h5' % dataset_id))

    X_te = pd.read_hdf(opj(path, '%s_test.h5' % dataset_id))



    # Specify target

    y = pd.read_hdf(opj(path, 'targets.h5'))

    y_tr = y.loc[X_tr.index, target]



    # Remove missing values

    missval = y_tr.isnull().values

    idx = ~missval

    X_tr = X_tr[idx]

    X_te = X_te

    y_tr = y_tr[idx]

    print('Removing %s missing values from target dataset.' % missval.sum())

    

    # Centralize corr_coef features

    median_offset = X_tr.iloc[:, X_tr.columns.str.contains('corr_coef')].median().mean()

    X_tr.iloc[:, X_tr.columns.str.contains('corr_coef')] -= median_offset

    X_te.iloc[:, X_te.columns.str.contains('corr_coef')] -= median_offset



    # Establish masks for different kinds of data

    mask_ids = []

    for m in ['IC_', '_vs_', 'corr_coef', '^c[0-9]+_c[0-9]+']:

        mask_ids.append(X_tr.columns.str.contains(m))

    mask_ids = np.array(mask_ids)



    # Data scaling

    for i, m in enumerate(mask_ids):



        if m.sum()==0:

            continue

        

        # Apply Scale

        scale_value = scale_values[i]

        unify_mask_scale = np.percentile(X_tr.iloc[:, m].abs(), 90)



        X_te.iloc[:, m] /= unify_mask_scale

        X_tr.iloc[:, m] /= unify_mask_scale

    

        X_te.iloc[:, m] *= scale_value

        X_tr.iloc[:, m] *= scale_value



    # Drop irrelevant measurements

    X_tr.dropna(axis=1, inplace=True)

    X_te.dropna(axis=1, inplace=True)

    

    # Drop duplicate rows

    X_tr = X_tr.T.drop_duplicates().T

    X_te = X_te.T.drop_duplicates().T

    

    print('Size of dataset (train/test): ', X_tr.shape, X_te.shape)

    

    X_tr = X_tr.values

    X_te = X_te.values

    y_tr = y_tr.values

    

    return X_tr, X_te, y_tr
# Create scorer function

from sklearn.metrics import make_scorer

def model_metric(y_true, y_pred, scaler=None, tidx=0):

    

    # List of power transformations

    pow_age = 1.0

    pow_d1v1 = 1.5

    pow_d1v2 = 1.5

    pow_d2v1 = 1.5

    pow_d2v2 = 1.5

    pow_d21 = 1.5

    pow_d22 = 1.0



    powers = [pow_age, pow_d1v1, pow_d1v2, pow_d2v1, pow_d2v2, pow_d21, pow_d22]

    

    # Invert scaler

    t_true = scaler.inverse_transform(np.transpose([y_true] * 7))[:, tidx]

    t_pred = scaler.inverse_transform(np.transpose([y_pred] * 7))[:, tidx]

    

    # Assign closest value from training set

    unique_values = np.unique(t_true)

    for i, a in enumerate(t_pred):

        t_pred[i] = unique_values[np.argmin(np.abs(a-unique_values))]



    # Invert power transformation

    t_true = np.power(t_true, 1./powers[tidx])

    t_pred = np.power(t_pred, 1./powers[tidx])

    

    # Compute the score

    score = np.mean(np.sum(np.abs(t_true - t_pred), axis=0) / np.sum(t_true, axis=0))

    return score
def create_grid(model_metric, alphas=[0.1, 1, 10], estimator=None,

                cv=5, scaler_targets=None, tidx=0):



    # Create Pipeline

    pipeline = Pipeline([

        ('scaler', None),

        ('estimator', estimator),

    ])



    # Define parameter grid

    param_grid = [{'scaler': [None, RobustScaler()],

                   'estimator__alpha': alphas,

                  }]



    # Create grid search object

    f_scorer = make_scorer(model_metric, greater_is_better=False,

                           scaler=scaler_targets, tidx=tidx)

    grid = GridSearchCV(pipeline,

                        cv=cv,

                        param_grid=param_grid,

                        scoring=f_scorer,

                        return_train_score=True,

                        verbose=5,

                        n_jobs=-1)



    return grid
# Select the target (name and index)

target = 'age'

tidx = 0



# Let's select which features to use and how to scale them

scale_values = [1,        # IC features

                1,        # FNC features

                np.nan,   # intra features

                np.nan]   # inter features



X_tr, X_te, y_tr =  load_dataset_and_scale(target, scale_values=scale_values)
# Define estimator

estimator = Ridge(tol=1e-3)



# Create grid search object

alphas = np.logspace(0, 4, 21)

grid = create_grid(model_metric, alphas=alphas, estimator=estimator,

                   cv=5, scaler_targets=scaler_targets, tidx=tidx)



# Run grid search

_ = grid.fit(X_tr, y_tr)



# Provide some insights into the models top performance

print('Dataset scales used: ', scale_values)

print("Best score at: %f using %s" % (grid.best_score_, grid.best_params_))
def extract_predictions(X_tr, X_te, grid, y_tr):



    # Store predictions in dictionary

    res = {}

    res['tr'] = grid.predict(X_tr)

    res['te'] = grid.predict(X_te)

    

    # Assign closest value from training set 

    unique_values = np.unique(y_tr)

    for t in ['tr', 'te']:

        for i, a in enumerate(res[t]):

            res[t][i] = unique_values[np.argmin(np.abs(a-unique_values))]



    return res['tr'], res['te']
# Extract the predictions for the training and the test set

pred_tr, pred_te = extract_predictions(X_tr, X_te, grid, y_tr)
def plot_predictions(pred_tr, pred_te, y_tr):



    # Plot prediction descrepancy on training and test set

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))



    ax[0].set_title('Prediction X_te: %s' % target)

    ax[0].plot(pred_te, '.', alpha=0.5, markersize=5)

    ax[0].plot(pred_tr, '.', alpha=0.3, markersize=5)

    ax[0].legend(['Train', 'Test'])

    ax[0].set_ylabel('target value')

    ax[0].set_xlabel('Sample')



    ax[1].set_title('Prediction X_tr: %s' % target)

    sns.regplot(x=pred_tr, y=y_tr, marker='.', ax=ax[1], scatter_kws={'s':10})

    ax[1].set_xlim([-3.5, 3.5])

    ax[1].set_ylim([-3.5, 3.5])

    ax[1].set_ylabel('True value')

    ax[1].set_xlabel('Predicted value')

    

    plt.show()
# Plot the predictions with respect to each other and to the target

plot_predictions(pred_tr, pred_te, y_tr)
def create_df_pred(grid):



    # Store grid search parameters and outcomes in dataframe

    df_pred = pd.DataFrame(grid.cv_results_)

    columns = [c for c in df_pred.columns if 'time' not in c

               and 'split' not in c

               and 'rank' not in c

               and c!='params']

    df_pred = df_pred[columns].sort_values('mean_test_score', ascending=False)

    df_pred['param_estimator__alpha'] = df_pred['param_estimator__alpha'].astype('float')

    df_pred['param_scaler'] = df_pred['param_scaler'].astype('str')

    

    return df_pred
# Creates dataframe about grid point's performance

df_pred = create_df_pred(grid).sort_values('mean_test_score')

df_pred.head()
def plot_hyperparam_fitting(df_pred):



    # Plot the model fit information

    for s in df_pred['param_scaler'].unique():



        df_plot = df_pred[np.prod([df_pred['param_scaler']==s],

                                  axis=0).astype('bool')]



        df_plot = df_plot.sort_values('param_estimator__alpha')



        # Extract relevant modelling metrics

        train_scores = df_plot['mean_train_score']

        valid_scores = df_plot['mean_test_score']

        std_tr = df_plot['std_train_score']

        std_va = df_plot['std_test_score']



        plt.figure(figsize=(12, 4))

        alphas = df_plot['param_estimator__alpha']

        plt.semilogx(alphas, train_scores, label='Training Set')

        plt.semilogx(alphas, valid_scores, label='Validation Set')



        # Add marker and text for best score

        max_id = np.argmax(valid_scores)

        x_pos = alphas.iloc[max_id]

        y_pos = valid_scores.iloc[max_id]

        txt = '{:0.4f}'.format(y_pos)

        plt.scatter(x_pos, y_pos, marker='x', c='red', zorder=10)

        plt.text(x_pos, y_pos, txt, fontdict={'size': 18})



        # Quantify variance with ±std curves

        plt.fill_between(alphas, train_scores-std_tr, train_scores+std_tr, alpha=0.3)

        plt.fill_between(alphas, valid_scores-std_va, valid_scores+std_va, alpha=0.3)

        plt.ylabel('Performance metric')

        plt.xlabel('Model parameter')



        # Adjust x-lim, y-lim, add legend and adjust layout

        plt.legend()

        plt.title('Scaler: %s' % s)

        plt.show()
# Plot prediction behaviour

plot_hyperparam_fitting(df_pred)
# Select the target (name and index)

target = 'age'

tidx = 0



# Let's select which features to use and how to scale them

scale_values = [1,        # IC features

                1,        # FNC features

                1,        # intra features

                1]        # inter features



X_tr, X_te, y_tr =  load_dataset_and_scale(target, scale_values=scale_values)
# Define estimator

estimator = Ridge(tol=1e-3)



# Create grid search object

alphas = np.logspace(1, 5, 21)

grid = create_grid(model_metric, alphas=alphas, estimator=estimator,

                   cv=5, scaler_targets=scaler_targets, tidx=tidx)



# Run grid search

_ = grid.fit(X_tr, y_tr)



# Provide some insights into the models top performance

print('Dataset scales used: ', scale_values)

print("Best score at: %f using %s" % (grid.best_score_, grid.best_params_))
# Extract the predictions for the training and the test set

pred_tr, pred_te = extract_predictions(X_tr, X_te, grid, y_tr)
# Plot the predictions with respect to each other and to the target

plot_predictions(pred_tr, pred_te, y_tr)
# Creates dataframe about grid point's performance

df_pred = create_df_pred(grid).sort_values('mean_test_score', ascending=False)

df_pred.head()
# Plot prediction behaviour

plot_hyperparam_fitting(df_pred)
def run_prediction(model_metric, estimator=None, alphas=[0.1, 1, 10], cv=5,

                   scaler_targets=None, target='age', tidx=0, scale_values=None):

    

    # Extract dataset

    X_tr, X_te, y_tr = load_dataset_and_scale(target, scale_values=scale_values)

    

    # Create grid search object

    grid = create_grid(model_metric, alphas=alphas, estimator=estimator,

                       cv=cv, scaler_targets=scaler_targets, tidx=tidx)

    

    # Run grid search

    _ = grid.fit(X_tr, y_tr)

    

    # Provide some insights into the models top performance

    print('Dataset scales used: ', scale_values)

    print("Best score at: %f using %s" % (grid.best_score_, grid.best_params_))



    # Extract the predictions for the training and the test set

    pred_tr, pred_te = extract_predictions(X_tr, X_te, grid, y_tr)



    # Plot the predictions with respect to each other and to the target

    plot_predictions(pred_tr, pred_te, y_tr)



    # Creates dataframe about grid point's performance

    df_pred = create_df_pred(grid)

    display(df_pred.sort_values('mean_test_score', ascending=False).head())



    # Plot prediction behaviour

    plot_hyperparam_fitting(df_pred)

    

    return df_pred, pred_tr, pred_te, grid, y_tr
# Select the target (name and index)

target = 'age'

tidx = 0
# Let's select which features to use and how to scale them

scale_values = [0.25,     # Feature: IC

                0.04,     # Feature: FNC

                np.nan,   # Feature: Intra Corr

                np.nan,   # Feature: Inter Corr

               ]
# Define model parameters

estimator = Ridge(tol=1e-3)

alphas = np.logspace(-2, 4, 31)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=alphas, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Select the target (name and index)

target = 'age'

tidx = 0
# Let's select which features to use and how to scale them

scale_values = [0.25,     # Feature: IC

                0.04,     # Feature: FNC

                0.087,    # Feature: Intra Corr

                np.nan,   # Feature: Inter Corr

               ]
# Define model parameters

estimator = Ridge(tol=1e-3)

alphas = np.logspace(-2, 4, 31)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=alphas, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Select the target (name and index)

target = 'age'

tidx = 0
# Let's select which features to use and how to scale them

scale_values = [0.25,     # Feature: IC

                0.04,     # Feature: FNC

                0.087,    # Feature: Intra Corr

                0.025,    # Feature: Inter Corr

               ]
# Define model parameters

estimator = Ridge(tol=1e-3)

alphas = np.logspace(-2, 4, 31)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=alphas, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Collect test predictions

predictions_ridge = {}



# Save predictions for age in output variable

predictions_ridge[target] = pred_te
# Select the target (name and index)

target = 'domain1_var1'

tidx = 1
# Let's select which features to use and how to scale them

scale_values = [0.25,    # Feature: IC

                0.01,    # Feature: FNC

                0.032,   # Feature: Intra Corr

                0.019,   # Feature: Inter Corr

               ]
# Define model parameters

estimator = Ridge(tol=1e-3)

alphas = np.logspace(-1, 5, 31)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=alphas, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Save predictions for domain1_var1 in output variable

predictions_ridge[target] = pred_te
# Select the target (name and index)

target = 'domain1_var2'

tidx = 2
# Let's select which features to use and how to scale them

scale_values = [0.121,   # Feature: IC

                0.019,   # Feature: FNC

                0.032,   # Feature: Intra Corr

                0.025,   # Feature: Inter Corr

               ]
# Define model parameters

estimator = Ridge(tol=1e-3)

alphas = np.logspace(-1, 5, 31)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=alphas, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Save predictions for domain1_var2 in output variable

predictions_ridge[target] = pred_te
# Select the target (name and index)

target = 'domain2_var1'

tidx = 3
# Let's select which features to use and how to scale them

scale_values = [0.25,    # Feature: IC

                0.008,   # Feature: FNC

                0.012,   # Feature: Intra Corr

                0.012,   # Feature: Inter Corr

               ]
# Define model parameters

estimator = Ridge(tol=1e-3)

alphas = np.logspace(-1, 5, 31)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=alphas, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Save predictions for domain2_var1 in output variable

predictions_ridge[target] = pred_te
# Select the target (name and index)

target = 'domain2_var2'

tidx = 4
# Let's select which features to use and how to scale them

scale_values = [0.261,   # Feature: IC

                0.025,   # Feature: FNC

                0.052,   # Feature: Intra Corr

                0.022,   # Feature: Inter Corr

               ]
# Define model parameters

estimator = Ridge(tol=1e-3)

alphas = np.logspace(-1, 5, 31)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=alphas, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Save predictions for domain2_var2 in output variable

predictions_ridge[target] = pred_te
def create_grid(model_metric, alphas=[0.1, 1, 10], estimator=None,

                cv=5, scaler_targets=None, tidx=0):



    # Create Pipeline

    pipeline = Pipeline([

        ('estimator', estimator),

    ])



    # Define parameter grid

    param_grid = [{'estimator__C': alphas}]



    # Create grid search object

    f_scorer = make_scorer(model_metric, greater_is_better=False,

                           scaler=scaler_targets, tidx=tidx)

    grid = GridSearchCV(pipeline,

                        cv=cv,

                        param_grid=param_grid,

                        scoring=f_scorer,

                        return_train_score=True,

                        verbose=5,

                        n_jobs=-1)



    return grid
def create_df_pred(grid):



    # Store grid search parameters and outcomes in dataframe

    df_pred = pd.DataFrame(grid.cv_results_)

    columns = [c for c in df_pred.columns if 'time' not in c

               and 'split' not in c

               and 'rank' not in c

               and c!='params']

    df_pred = df_pred[columns].sort_values('mean_test_score', ascending=False)

    df_pred['param_estimator__C'] = df_pred['param_estimator__C'].astype('float')



    return df_pred
def plot_hyperparam_fitting(df_pred):



    # Plot the model fit information

    df_plot = df_pred.copy()



    df_plot = df_plot.sort_values('param_estimator__C')



    # Extract relevant modelling metrics

    train_scores = df_plot['mean_train_score']

    valid_scores = df_plot['mean_test_score']

    std_tr = df_plot['std_train_score']

    std_va = df_plot['std_test_score']



    plt.figure(figsize=(12, 4))

    Cs = df_plot['param_estimator__C']

    plt.semilogx(Cs, train_scores, label='Training Set')

    plt.semilogx(Cs, valid_scores, label='Validation Set')



    # Add marker and text for best score

    max_id = np.argmax(valid_scores)

    x_pos = Cs.iloc[max_id]

    y_pos = valid_scores.iloc[max_id]

    txt = '{:0.4f}'.format(y_pos)

    plt.scatter(x_pos, y_pos, marker='x', c='red', zorder=10)

    plt.text(x_pos, y_pos, txt, fontdict={'size': 18})



    # Quantify variance with ±std curves

    plt.fill_between(Cs, train_scores-std_tr, train_scores+std_tr, alpha=0.3)

    plt.fill_between(Cs, valid_scores-std_va, valid_scores+std_va, alpha=0.3)

    plt.ylabel('Performance metric')

    plt.xlabel('Model parameter')



    # Adjust x-lim, y-lim, add legend and adjust layout

    plt.legend()

    plt.show()
# Select the target (name and index)

target = 'age'

tidx = 0
# Let's select which features to use and how to scale them

scale_values = [0.25,      # Feature: IC

                0.015,     # Feature: FNC

                0.042,     # Feature: Intra Corr

                0.014,     # Feature: Inter Corr

               ]
# Define model parameters

estimator = SVR(gamma='scale', epsilon=0.2, tol=1e-3, max_iter=5000)

Cs = np.logspace(-1, 1, 11)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=Cs, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Collect test predictions

predictions_svr = {}



# Save predictions for age in output variable

predictions_svr[target] = pred_te
# Select the target (name and index)

target = 'domain1_var1'

tidx = 1
# Let's select which features to use and how to scale them

scale_values = [0.25,      # Feature: IC

                0.012,     # Feature: FNC

                0.032,     # Feature: Intra Corr

                0.018,     # Feature: Inter Corr

               ]
# Define model parameters

estimator = SVR(gamma='scale', epsilon=0.2, tol=1e-3, max_iter=5000)

Cs = np.logspace(-1, 1, 11)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=Cs, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Save predictions for age in output variable

predictions_svr[target] = pred_te
# Select the target (name and index)

target = 'domain1_var2'

tidx = 2
# Let's select which features to use and how to scale them

scale_values = [0.18,      # Feature: IC

                0.01,      # Feature: FNC

                np.nan,    # Feature: Intra Corr (no improvement)

                0.025,     # Feature: Inter Corr

               ]
# Define model parameters

estimator = SVR(gamma='scale', epsilon=0.2, tol=1e-3, max_iter=5000)

Cs = np.logspace(-1.5, 0.5, 11)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=Cs, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Save predictions for age in output variable

predictions_svr[target] = pred_te
# Select the target (name and index)

target = 'domain2_var1'

tidx = 3
# Let's select which features to use and how to scale them

scale_values = [0.25,      # Feature: IC

                0.025,     # Feature: FNC

                0.036,     # Feature: Intra Corr (no improvement)

                0.023,     # Feature: Inter Corr

               ]
# Define model parameters

estimator = SVR(gamma='scale', epsilon=0.2, tol=1e-3, max_iter=5000)

Cs = np.logspace(-1, 1, 11)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=Cs, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Save predictions for age in output variable

predictions_svr[target] = pred_te
# Select the target (name and index)

target = 'domain2_var2'

tidx = 4
# Let's select which features to use and how to scale them

scale_values = [0.189,      # Feature: IC

                0.025,      # Feature: FNC

                0.050,    # Feature: Intra Corr (no improvement)

                0.022,     # Feature: Inter Corr

               ]
# Define model parameters

estimator = SVR(gamma='scale', epsilon=0.2, tol=1e-3, max_iter=5000)

Cs = np.logspace(-1, 1, 11)

cv = 5
df_pred, pred_tr, pred_te, grid, y_tr = run_prediction(

    model_metric, estimator=estimator, alphas=Cs, cv=cv,

    scaler_targets=scaler_targets, target=target, tidx=tidx,

    scale_values=scale_values)
# Save predictions for age in output variable

predictions_svr[target] = pred_te
# Load sample submission file

submission = pd.read_csv(opj('/kaggle', 'input', 'trends-assessment-prediction', 'sample_submission.csv')).set_index('Id')

submission.head()
def back_transform(y_test, unique_values, scaler=None, tidx=0):

    

    # List of power transformations

    pow_age = 1.0

    pow_d1v1 = 1.5

    pow_d1v2 = 1.5

    pow_d2v1 = 1.5

    pow_d2v2 = 1.5

    pow_d21 = 1.5

    pow_d22 = 1.0



    powers = [pow_age, pow_d1v1, pow_d1v2, pow_d2v1, pow_d2v2, pow_d21, pow_d22]

    

    # Assign closest value from training set

    for i, a in enumerate(y_test):

        y_test[i] = unique_values[np.argmin(np.abs(a-unique_values))]

    

    # Invert scaler

    y_test = scaler.inverse_transform(np.transpose([y_test] * 7))[:, tidx]

    

    # Invert power transformation

    y_test = np.power(y_test, 1./powers[tidx])



    return y_test
# Fill up the submission file with the ridge predictions

prediction_dict = predictions_ridge

for i, t in enumerate(['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']):

    unique_values = sorted(targets[t].dropna().unique())

    pred_values = back_transform(prediction_dict[t], unique_values, scaler=scaler_targets, tidx=i)

    submission.iloc[submission.index.str.contains(t), 0] = pred_values



# Let's visualize a few points from the submission file

display(submission.head(10))



# Store predictions in CSF file

submission.to_csv('submission_ridge.csv')
# Fill up the submission file with the ridge predictions

prediction_dict = predictions_svr

for i, t in enumerate(['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']):

    unique_values = sorted(targets[t].dropna().unique())

    pred_values = back_transform(prediction_dict[t], unique_values, scaler=scaler_targets, tidx=i)

    submission.iloc[submission.index.str.contains(t), 0] = pred_values



# Let's visualize a few points from the submission file

display(submission.head(10))



# Store predictions in CSF file

submission.to_csv('submission_svr.csv')
for c in predictions_ridge.keys():

    plt.figure(figsize=(6, 6))

    plt.scatter(predictions_ridge[c], predictions_svr[c], s=2, alpha=0.5)

    plt.title(c)

    plt.xlabel('Ridge Prediction')

    plt.ylabel('SVR Prediction')

    plt.show()