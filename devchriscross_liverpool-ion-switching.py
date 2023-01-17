import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv

import os
import gc
def add_features(dataset, window_size, rev=False, per_batch=True):
    if per_batch:
        rolling_signal = dataset.groupby(["batch_group"]).signal.rolling(window_size)
    else:
        rolling_signal = dataset.signal.rolling(window_size)
        
    dataset["rolling_sum_" + str(window_size)] = rolling_signal.sum().reset_index(drop=True)
    dataset["rolling_mean_" + str(window_size)] = rolling_signal.mean().reset_index(drop=True)
    dataset["rolling_min_" + str(window_size)] = rolling_signal.min().reset_index(drop=True)
    dataset["rolling_max_" + str(window_size)] = rolling_signal.max().reset_index(drop=True)
    dataset["rolling_std_" + str(window_size)] = rolling_signal.std().reset_index(drop=True)
    dataset["rolling_var_" + str(window_size)] = rolling_signal.var().reset_index(drop=True)
    dataset["rolling_skew_" + str(window_size)] = rolling_signal.skew().reset_index(drop=True)
    dataset["rolling_kurt_" + str(window_size)] = rolling_signal.kurt().reset_index(drop=True)
    if rev:
        dataset["rev_rolling_sum_" + str(window_size)] = rolling_signal.sum().shift(-window_size).reset_index(drop=True)
        dataset["rev_rolling_mean_" + str(window_size)] = rolling_signal.mean().shift(-window_size).reset_index(drop=True)
        dataset["rev_rolling_min_" + str(window_size)] = rolling_signal.min().shift(-window_size).reset_index(drop=True)
        dataset["rev_rolling_max_" + str(window_size)] = rolling_signal.max().shift(-window_size).reset_index(drop=True)
        dataset["rev_rolling_std_" + str(window_size)] = rolling_signal.std().shift(-window_size).reset_index(drop=True)
        dataset["rev_rolling_var_" + str(window_size)] = rolling_signal.var().shift(-window_size).reset_index(drop=True)
        dataset["rev_rolling_skew_" + str(window_size)] = rolling_signal.skew().shift(-window_size).reset_index(drop=True)
        dataset["rev_rolling_kurt_" + str(window_size)] = rolling_signal.kurt().shift(-window_size).reset_index(drop=True)
    
    return dataset
def add_lagged_signals(dataset, windows):
    for window in windows:
        dataset["lagged_" + str(window)] = dataset.groupby(["batch_group"]).signal.shift(window)
        
    return dataset
def create_batch_group(df, size):
    num_list = np.empty([10, size])
    num_list.fill(0)
    for i in range(10):
        num_list[i,:].fill(i)
    num_list = num_list.reshape(-1,1)
    batch_group = pd.DataFrame(num_list).rename(columns={0: "batch_group"})
    return pd.concat([df.reset_index(drop=True), batch_group.reset_index(drop=True)], axis=1)
def load_dataset_with_features(window_sizes, lagged_windows, return_transform=True, rev=False, fn=None):
    df_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

    discrete_batch = 500000
    train_size = int(0.70 * discrete_batch)
    valid_size = int(discrete_batch - train_size)
    train_set = pd.DataFrame()
    valid_set = pd.DataFrame()

    print("Training size:", train_size*10)
    print("Validation size:", valid_size*10)
    
    for i in range(0, len(df_train)+1, discrete_batch)[1:]:
        train_set = pd.concat([train_set, df_train[i-discrete_batch:i].head(train_size)], axis=0)
        valid_set = pd.concat([valid_set, df_train[i-discrete_batch:i].tail(valid_size)], axis=0)
    
    train_set = create_batch_group(train_set, train_size)
    valid_set = create_batch_group(valid_set, valid_size)
        
    del df_train
    gc.collect()

    if fn:
        train_set = fn(train_set)
        valid_set = fn(valid_set)
    else:
        train_set = add_lagged_signals(train_set, lagged_windows)
        valid_set = add_lagged_signals(valid_set, lagged_windows)
        
        for window in window_sizes:
            train_set = add_features(train_set, window, rev)
            valid_set = add_features(valid_set, window, rev)
            gc.collect()
    
    train_set = train_set.drop(columns=["batch_group"])
    valid_set = valid_set.drop(columns=["batch_group"])
    
    train_set["signal_2"] = train_set["signal"]**2
    train_set["signal_3"] = train_set["signal"]**3
    valid_set["signal_2"] = valid_set["signal"]**2
    valid_set["signal_3"] = valid_set["signal"]**3
    
    if return_transform:
        y_train = train_set.open_channels
        y_valid = valid_set.open_channels

        scaler = MinMaxScaler()
        train_set = train_set.drop(columns=["time", "open_channels"])
        X_train = scaler.fit_transform(train_set)
        X_valid = scaler.transform(valid_set.drop(columns=["time", "open_channels"]))

        X_train = np.nan_to_num(X_train, nan=0)
        X_valid = np.nan_to_num(X_valid, nan=0)
        return X_train, X_valid, y_train, y_valid, train_set.columns
    else:
        return train_set, valid_set
gc.collect()
X_train, X_valid, y_train, y_valid, X_cols = load_dataset_with_features([5000, 6000, 7000, 8000], [1000,2000,3000,4000],
                                                                        return_transform=True, rev=True)
# catboost = CatBoostRegressor(loss_function="RMSE", random_seed=1, eval_metric="RMSE")
# catboost.fit(X_train, y_train, eval_set=Pool(X_valid, y_valid), early_stopping_rounds=100, plot=True)
cboost = CatBoostRegressor(loss_function="RMSE", random_seed=1, eval_metric="RMSE")
cboost.fit(X_train, y_train, eval_set=Pool(X_valid, y_valid), early_stopping_rounds=100, plot=True)
importance_list = list(zip(X_cols, cboost.feature_importances_))
sorted(importance_list, key = lambda x: x[1], reverse = True)
from sklearn.utils.class_weight import compute_class_weight
c_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
c_weights
cboost2 = CatBoostClassifier(loss_function="MultiClass", random_seed=1, eval_metric="MultiClass", class_weights=c_weights, task_type="GPU")
cboost2.fit(X_train, y_train, eval_set=Pool(X_valid, y_valid), early_stopping_rounds=100, plot=True)
importance_list2 = list(zip(X_cols, cboost2.feature_importances_))
sorted(importance_list2, key = lambda x: x[1], reverse = True)
from functools import partial
from sklearn.metrics import f1_score
import scipy as sp
class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize F1 (Macro) score
    # https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
    """
    def __init__(self):
        self.coef_ = 0

    def _f1_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        return -f1_score(y, X_p, average = 'macro')

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._f1_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']
cboost2.best_iteration_
cboost2.best_score_
cboost2.best_iteration_
cboost2.best_score_
optR = OptimizedRounder()
optR.fit(cboost.predict(X_train).reshape(-1,), y_train)
coefficients = optR.coefficients()
print(coefficients)
def round_preds(model, X_valid, y_valid, classification=False):
    y_preds = model.predict(X_valid)
        
    if y_valid is not None:
        X_round = pd.DataFrame(np.concatenate([y_preds.reshape(-1, 1), y_valid.to_numpy().reshape(-1, 1)], axis=1))
    else:
        X_round = pd.DataFrame(y_preds.reshape(-1, 1))
        
    if classification: 
        return X_round.rename(columns={0: "open_channels"})

    # [0.50865957 1.5679399  2.52020058 3.50749837 4.44309614 5.47364305
    #  6.51207558 7.54953974 8.43190615 9.36119176]
    coefficients1 = [0.50865957, 1.5679399, 2.52020058, 3.50749837, 4.44309614, 5.47364305, 6.51207558, 7.54953974, 8.43190615, 9.36119176] # 0.4173822
    coefficients2 = [0.51868838, 1.51346738, 2.48039854, 3.52332079, 4.4799399,  5.4619901, 6.49833485, 7.50395204, 8.4661238,  9.44524934]  # 0.3757953
    coefficients3 = [0.51713402, 1.50215723, 2.50622843, 3.50676172, 4.49122719, 5.43780087, 6.53170168, 7.52116961, 8.48203458, 9.44349437]   # 0.3461139853866367
    coefficients4 = [0.51627947, 1.50882392, 2.47761787, 3.51451859, 4.48488742, 5.485062, 6.51204066, 7.49428795, 8.49180723, 9.45132125]  #  0.3459922
    optR = OptimizedRounder()

#     ceil_10 = 11.259509127623923
    ceil_10 = 11.234590256740304  #  0.3459922

    X_round["open_channels"] = 0
    X_round.loc[X_round[0] < 0, "open_channels"] = 0
    X_round.loc[(X_round[0] > 10) & (X_round[0] <= ceil_10), "open_channels"] = 10
    X_round.loc[X_round[0] > ceil_10, "open_channels"] = np.round(X_round.loc[X_round[0] > ceil_10, 0])

    within_class_range = (X_round[0] > 0) & (X_round[0] < 10)
    X_round.loc[within_class_range, "open_channels"] = optR.predict(X_round.loc[within_class_range, 0], coefficients4)
    X_round["open_channels"] = X_round["open_channels"].astype("int64")

#     print(X_round.head())
    return X_round
from sklearn.metrics import classification_report
# X_round = round_preds(cboost, X_valid, y_valid)
# print(classification_report(y_valid, X_round[["open_channels"]]))
X_round = round_preds(cboost2, X_valid, y_valid, classification=True)
print(classification_report(y_valid, X_round["open_channels"]))
X_round = round_preds(cboost2, X_valid, y_valid, classification=True)
print(classification_report(y_valid, X_round["open_channels"]))
del X_train, X_valid, y_train, y_valid, X_cols
gc.collect()
def train_entire_dataset(iterations, gpu=False, classification=False):
    X_train = pd.read_csv("../input/liverpool-ion-switching/train.csv")
    X_train = create_batch_group(X_train, 500000)
    
    X_train = add_lagged_signals(X_train, lagged_window)
    for window in window_sizes:
        X_train = add_features(X_train, window, rev=True)
        
    X_train = X_train.drop(columns=["batch_group"])
    
    scaler = MinMaxScaler()
    y_train = X_train.open_channels
    X_train = X_train.drop(columns=["time", "open_channels"])
    X_train = scaler.fit_transform(X_train)
    X_train = np.nan_to_num(X_train, nan=0)
    
    task_type = "GPU" if gpu else "CPU"
    print("Model will be trained on", task_type)
    
    if classification:
        catboost = CatBoostClassifier(loss_function="MultiClass", random_seed=1, eval_metric="MultiClass", iterations=iterations, class_weights=c_weights, task_type=task_type)
        catboost.fit(X_train, y_train, plot=True)
    else:
        catboost = CatBoostRegressor(loss_function="RMSE", random_seed=1, eval_metric="RMSE", iterations=iterations, task_type=task_type)
        catboost.fit(X_train, y_train, plot=True)
    
    return catboost, scaler
def predict_test_dataset(model, scaler, classification=False):
    X_test = pd.read_csv("../input/liverpool-ion-switching/test.csv")
    X_test = create_batch_group(X_test, 500000)
    
    X_test = add_lagged_signals(X_test, lagged_window)
    for window in window_sizes:
        X_test = add_features(X_test, window, rev=True, per_batch=True)
        
    X_test = X_test.drop(columns=["batch_group"])  
    
    X_time = X_test["time"]
    X_test = X_test.drop(columns=["time"])
    X_test = scaler.transform(X_test)
    X_test = np.nan_to_num(X_test, nan=0)
    
    y_preds = round_preds(model, X_test, None, classification=classification)
    X_test = pd.concat([X_time, y_preds["open_channels"]], axis=1)
    
    return X_test
# iterations = 144
# iterations = 237
# iterations = 550  # 0.3459922
# iterations = 531  # classification 0.31083008985387717
iterations = cboost2.best_iteration_  #745
lagged_window = [1000, 2000]
window_sizes = [5000, 6000, 7000, 8000]
model, scaler = train_entire_dataset(iterations, gpu=True, classification=True)
submission = predict_test_dataset(model, scaler, classification=True)
submission.iloc[:2000000]["open_channels"].value_counts()
submission.iloc[:2000000].to_csv('submission.csv', float_format='%0.4f', index=False)
