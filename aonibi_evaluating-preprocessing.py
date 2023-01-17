import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 400)
train_df = pd.read_csv('../input/train.csv', index_col='Id')
test_df = pd.read_csv('../input/test.csv', index_col='Id')
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

def fill_lot_frontage(filled_df):
    nan_mask = filled_df['LotFrontage'].isnull()
    le = LabelEncoder()
    df = filled_df.copy()
    str_cols = df.dtypes[df.dtypes == object].index
    for col in str_cols:
        df[col] = le.fit_transform(df[col].fillna('nan'))
    train = df[~nan_mask]
    X = train.drop('LotFrontage', axis=1)
    y = train['LotFrontage']
    X_nan = df[nan_mask].drop('LotFrontage', axis=1)

    rf = RandomForestRegressor(n_estimators=200, random_state=0)
    rf.fit(X, y)
    pred = rf.predict(X_nan)
    filled_df.loc[X_nan.index, 'LotFrontage'] = pred
    return filled_df
from sklearn.ensemble import RandomForestRegressor

def fill_na(train_df, test_df, fill_lot_frontage_func):
    train_data = train_df.drop('SalePrice', axis=1)
    df = pd.concat([train_data, test_df])
    df.loc[[2121, 2189], 
        ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'TotalBsmtSF']] = 0
    df.loc[1380, 'Electrical'] = 'SBrkr'
    df.loc[2593, 'GarageYrBlt'] = 2007
    df.loc[2152, ['Exterior1st', 'Exterior2nd']] = 'VinylSd'
    df.loc[[2217, 2474], 'Functional'] = 'Typ'
    df.loc[2577, ['GarageType','GarageArea', 'GarageCars']] = [np.nan, 0, 0]
    df.loc[1556, 'KitchenQual'] = 'TA'
    df.loc[[1916, 2217, 2251, 2905], 'MSZoning'] = 'RL'
    df.loc[2409, 'SaleType'] = 'WD'
    df.loc[[1916, 1946], 'Utilities'] = 'AllPub'
    nan_ix = df[df['GarageYrBlt'].isnull()].index
    df.loc[nan_ix, 'GarageYrBlt'] = df.loc[nan_ix, 'YearRemodAdd']
    df.loc[:, 'GarageYrBlt'] = df.loc[:, 'GarageYrBlt'].astype('int64')
    nan_ix = df[df['MasVnrType'].isnull()].index
    for i, mv_area in zip(nan_ix,  df.loc[nan_ix, 'MasVnrArea']):
        if (not np.isnan(mv_area)) and mv_area > 0.0:
            df.loc[i, 'MasVnrType'] = 'BrkFace'

    none_ix = df[df['MasVnrType'] == 'None'].index
    df.loc[none_ix, 'MasVnrType'] = np.nan

    nan_ix = df[df['MasVnrArea'].isnull()].index
    df.loc[nan_ix, 'MasVnrArea'] = 0.0

    df = fill_lot_frontage_func(df)
    return df
filled_df = fill_na(train_df, test_df, fill_lot_frontage)
dummied_df = pd.get_dummies(filled_df, columns=['MSSubClass', 'MoSold', 'YrSold'])
dummied_df = pd.get_dummies(dummied_df)

X_test = dummied_df.loc[1461:, :]
X_train = dummied_df.loc[:1460, :]
y_train = train_df.loc[:, 'SalePrice']

from sklearn.preprocessing import LabelEncoder

label_encoded_df = filled_df.copy()
le = LabelEncoder()
str_cols = label_encoded_df.dtypes[label_encoded_df.dtypes == object].index
for col in str_cols:
    label_encoded_df[col] = le.fit_transform(label_encoded_df[col].fillna('nan'))
X_train_le = label_encoded_df.loc[:1460, :]
X_test_le = label_encoded_df.loc[1461:, :]

usable_data = ['dummied_df', 'label_encoded_df',
               'X_train', 'X_test', 'y_train', 'X_train_le', 'X_test_le']
print("'filled_df' contains nan and non-numerical data")
print("Usable data: {}".format(usable_data))
print("dummied_df -> (X_train + X_test)   label_encoded_df -> (X_train_le + X_test_le)\n")
gst = globals()
for x in usable_data:
    print((x + ".isnull().any().any() -> {}").format(gst[x].isnull().any().any()))
    print(("(" + x + " < 0).any().any() -> {}").format((gst[x] < 0).any().any()))


import inspect
from sklearn.model_selection import train_test_split

class PP:

    def __init__(self, model, pre_funcs=None, post_funcs=None):
        self.model = model
        self.pre_funcs = [] if pre_funcs is None else pre_funcs
        self.post_funcs = [] if post_funcs is None else post_funcs
    
    def __str__(self):
        return "PP(\nmodel: {}\nfpre_funcs: {}\npost_funcs: {}\n)".format(self.model, self.pre_funcs, self.post_funcs)

    @staticmethod
    def compred(X, y, pp_list, score_func, iters=20, lf=5, decimal_places=4, pp_listeners=None):
        PP._check_XyX(X, y)
        pp_count = len(pp_list)
        if pp_count == 0:
            raise ValueError("pp_list is empty!")
        for pp in pp_list:
            if pp.model is None:
                raise ValueError("model is None!  pp -> {}".format(pp))
        listeners = [PPScorer(lf, decimal_places)] if pp_listeners is None else pp_listeners
        for listener in listeners:
            if hasattr(listener, "start_pred"):
                listener.start_pred(X, y, pp_list, score_func, iters)
        for i in range(iters):
            data_list = []
            data_list.append(list(train_test_split(X, y)))
            for listener in listeners:
                if hasattr(listener, "notify_data") and not listener.notify_data(*data_list[0]):
                    print("##### compred function is terminated by listner #####")
                    return
            for _ in range(pp_count - 1):
                data_list.append([x.copy() for x in data_list[0]])
            for pp, data in zip(pp_list, data_list):
                pred = pp._predict(data[0], data[2], data[1])
                for listener in listeners:
                    if hasattr(listener, "notify_pred") and not listener.notify_pred(*data, pred, pp):
                        print("##### compred function is terminated by listner #####")
                        return
                score = score_func(data[3], pred)
                for listener in listeners:
                    if hasattr(listener, "notify_score") and not listener.notify_score(*data, pred, score, pp):
                        print("##### compred function is terminated by listner #####")
                        return
        for listener in listeners:
            if hasattr(listener, "end_pred"):
                listener.end_pred()
    
    def pre_process(self, X_train, y_train, X_test):
        PP._check_XyX(X_train, y_train, X_test)
        return self._pre_process(X_train.copy(), y_train.copy(), X_test.copy())

    def post_process(self, pred):
        return self._post_process(pred.copy())
    
    def _post_process(self, pred):
        for func in self.post_funcs:
            pred = func(pred)
        return pred
    
    def _pre_process(self, X_train, y_train, X_test):
        X1, y, X2 = X_train, y_train, X_test
        for func in self.pre_funcs:
            if (type(func) is list):
                X = pd.concat([X1, X2])
                for process_X in func:
                    X = process_X(X)
                    PP._check_XyX(X, y, pp_func=process_X)
                idx_X2 = len(X1)
                X1, X2 = X[:idx_X2], X[idx_X2:]
            else:
                if len(inspect.getargspec(func).args) < 3:
                    X1, y, X2 = func(X1), y, func(X2)
                else:
                    X1, y, X2 = func(X1, y, X2)
                PP._check_XyX(X1, y, X2, func)
        return (X1, y, X2)

    def _predict(self, X_train, y_train, X_test):
        X1, y, X2 = self._pre_process(X_train, y_train, X_test)
        self.model.fit(X1, y)
        pred = self.model.predict(X2)
        pred = self._post_process(pred)
        return pred
    
    @staticmethod
    def _check_XyX(X1, y, X2=None, pp_func=None):
        message = "" if pp_func is None else str(pp_func) + " returns illegal value: "
        if not isinstance(X1, pd.DataFrame) or (X2 is not None and not isinstance(X2, pd.DataFrame)):
            raise ValueError(message + "X must be pandas.DataFrame!")
        if not isinstance(y, pd.Series):
            raise ValueError(message + "y must be pandas.Series!")

class PPScorer:
    def __init__(self, lf=5, decimal_places=4):
        self.lf = lf
        self.placeholder = "{:." + str(decimal_places) + "f}, "
        
    
    def start_pred(self, X, y, pp_list, score_func, iters):
        self.pp_count = len(pp_list)
        self.iters = iters
        self.iters_count = 0
        self.scores = np.zeros(self.pp_count)
        self.total_scores = np.zeros(self.pp_count)
        self.scores_index = 0
        self.result_text = "("
    
    def notify_score(self, X_train, X_test, y_train, y_test, pred, score, pp):
        self.scores[self.scores_index] = score
        self.total_scores[self.scores_index] += score
        self.result_text += self.placeholder
        self.scores_index += 1
        if self.scores_index == self.pp_count:
            self.scores_index = 0
            self.iters_count += 1
            self.result_text = self.result_text[:-2] + ") "
            print(self.result_text.format(*self.scores), end='')
            if self.iters_count % self.lf == 0:
                print()
            if self.iters_count != self.iters:
                self.result_text = "("
        return True
    
    def end_pred(self):
        self.total_scores /= self.iters
        print(("\nmean: " + self.result_text).format(*self.total_scores))

from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.3)

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
def pp_log_data(X_train, y_train, X_test):
    return np.log1p(X_train), np.log1p(y_train), np.log1p(X_test)

def pp_expm_pred(pred):
    return np.expm1(pred)

pp1 = PP(elastic_net) # ValueError occurs when the predictions contain negative values
pp2 = PP(elastic_net, pre_funcs=[pp_log_data], post_funcs=[pp_expm_pred]) 
PP.compred(X_train, y_train, [pp1, pp2], rmsle, iters=5)
from sklearn.preprocessing import StandardScaler

def pp_standard_scale_fit_train(X_train, y_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train_scaled, y_train, X_test_scaled

# pre_funcs=[[pp_standard_scale]] -> X is pd.concat([X_train, X_test])
def pp_standard_scale(X):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

pp1 = PP(elastic_net, pre_funcs=[pp_log_data], post_funcs=[pp_expm_pred])
pp2 = PP(elastic_net, pre_funcs=[pp_log_data, pp_standard_scale_fit_train], post_funcs=[pp_expm_pred])
pp3= PP(elastic_net, pre_funcs=[pp_log_data, [pp_standard_scale]], post_funcs=[pp_expm_pred])
PP.compred(X_train, y_train, [pp1, pp2, pp3], rmsle, iters=18, lf=3)
def plot_outliers(y, pred, z, outliers):
    def plot_scatter(ax, x1, y1, x2, y2, y_label):
        ax.scatter(x1, y1, marker='.')
        ax.scatter(x2, y2, c='r')
        ax.legend(['Accepted', 'Outlier'])
        ax.set_xlabel('y')
        ax.set_ylabel(y_label);
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_scatter(axes[0], y, pred, y.loc[outliers], pred.loc[outliers], 'pred')
    plot_scatter(axes[1], y, y - pred, y.loc[outliers], y.loc[outliers] - pred.loc[outliers], 'y - pred')

    axes[2].hist(z, bins=50)
    axes[2].hist(z.loc[outliers], color='r', bins=50)
    axes[2].legend(['Accepted', 'Outlier'])
    axes[2].set_xlabel('z')

from sklearn.linear_model import Ridge

sigma = 2.8
pp0 = PP(model=None, pre_funcs=[pp_log_data])
X, y, _ = pp0.pre_process(X_train, y_train, X_test)
pred = Ridge().fit(X, y).predict(X)
error = y - pred
mean_error = error.mean()
std_error = error.std()
z = (error - mean_error) / std_error
outliers = z[abs(z) > sigma].index
plot_outliers(y, pd.Series(pred, index=y.index), z, outliers)
print("outliers in X_train: {}".format(outliers.values))
from sklearn.linear_model import Ridge
# model -> Linear model
def new_pp_drop_outliers(model=None, sigma=3):
    def pp_drop_outliers(X_train, y_train, X_test):
        estimator = Ridge() if model is None else model
        pred = estimator.fit(X_train, y_train).predict(X_train)
        error = y_train - pred
        mean_error = error.mean()
        std_error = error.std()
        z = (error - mean_error) / std_error
        outliers = z[abs(z) > sigma].index
        X = X_train.drop(outliers)
        y = y_train.drop(outliers)
        return X, y, X_test
    return pp_drop_outliers

pp_drop_outliers = new_pp_drop_outliers(sigma=2.8)
pp1 = PP(elastic_net, pre_funcs=[pp_log_data, [pp_standard_scale]], post_funcs=[pp_expm_pred])
pp2 = PP(elastic_net, pre_funcs=[pp_log_data, pp_drop_outliers, [pp_standard_scale]], post_funcs=[pp_expm_pred])
PP.compred(X_train, y_train, [pp1, pp2], rmsle)
def pp_sf(X):
    X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
    return X

pp_drop_outliers = new_pp_drop_outliers(sigma=2.8)
pp1 = PP(elastic_net, pre_funcs=[pp_log_data, pp_drop_outliers, [pp_standard_scale]], post_funcs=[pp_expm_pred])
pp2 = PP(elastic_net, pre_funcs=[pp_sf, pp_log_data, pp_drop_outliers, [pp_standard_scale]], post_funcs=[pp_expm_pred])
PP.compred(X_train, y_train, [pp1, pp2], rmsle)
sigmas = np.arange(1.8, 2.9, 0.1)
pp_drop_outliers_list = [new_pp_drop_outliers(sigma=x) for x in sigmas]
pp_list = [PP(elastic_net, pre_funcs=[pp_sf, pp_log_data, x, [pp_standard_scale]], post_funcs=[pp_expm_pred]) 
           for x in pp_drop_outliers_list]
scorer = PPScorer(lf=1)
PP.compred(X_train, y_train, pp_list, rmsle, pp_listeners=[scorer])
best_sigma = sigmas[np.argmin(scorer.total_scores)]
print("best_sigma: {}".format(best_sigma))
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.3)
pp_drop_outliers = new_pp_drop_outliers(sigma=best_sigma)
pp = PP(None, pre_funcs=[pp_sf, pp_log_data, pp_drop_outliers, [pp_standard_scale]], post_funcs=[pp_expm_pred])
X, y, test = pp.pre_process(X_train, y_train, X_test)
elastic_net.fit(X, y)
pred = elastic_net.predict(test)
pred = pp.post_process(pred)

result = pd.DataFrame({'Id': X_test.index, 'SalePrice': pred})
result.to_csv('submission.csv',index=False)


def plot_num_data(X_train, y_train, num_cols, X_test=None, hist_bins=30, ks_dict=None, rejected=0.05):
    def plot_scatter(ax, values, col):
        ax.scatter(values, y_train)
        ax.set_title(col + " - y_train")

    def plot_hist_and_box(ax1, ax2, values, col, is_train=True):
        title_color = plt.rcParams['axes.labelcolor']
        if is_train:
            title1 = title2 = col + "(X_train)"
        else:
            if ks_dict is None:
                title1 = title2 = col + "(X_test)"
            else:
                ks_result = ks_dict[col]
                if ks_result[1] < rejected:
                    title_color = 'red'
                title1 = "(X_test) ks stat:" + str(round(ks_result[0], 4))
                title2 = "(X_test) p-value:" + str(round(ks_result[1], 4))
            
        ax1.hist(values, bins=hist_bins)
        ax2.boxplot(values, vert=False)
        ax1.set_title(title1, color=title_color)
        ax2.set_title(title2, color=title_color)
        
    rows = len(num_cols)
    ax_count = 3 if X_test is None else 5
    total = rows * ax_count
    height = 5 if X_test is None else 4
    fig, all_axes = plt.subplots(rows, ax_count, figsize=(18, rows * height))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    axes = []
    for i in range(ax_count):
        axes.append([ax for j, ax in zip(range(total), all_axes.ravel()) if j % ax_count == i])
    
    X1 = X_train[num_cols]
    if X_test is None:
        for ax0, ax1, ax2, col in zip(*axes, num_cols):
            values = X1[col].values
            plot_scatter(ax0, values, col)
            plot_hist_and_box(ax1, ax2, values, col);
    else:
        X2 = X_test[num_cols]
        for ax0, ax1, ax2, ax3, ax4, col in zip(*axes, num_cols):
            valuesX1 = X1[col].values
            plot_scatter(ax0, valuesX1, col)
            plot_hist_and_box(ax1, ax2, valuesX1, col);
            plot_hist_and_box(ax3, ax4, X2[col].values, col, False);


def plot_obj_data(X_train, obj_cols, X_test=None, ks_dict=None, rejected=0.05):
    if X_test is None:
        rows = np.ceil(len(obj_cols) / 2).astype(np.int)
        fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 5))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for ax, col in zip(axes.ravel(), obj_cols):
            X_train.loc[:, col].value_counts(dropna=False).plot.bar(ax=ax)
            ax.set_title(col)
    else:
        rows = len(obj_cols)
        fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 5))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for ax, col in zip(axes, obj_cols):
            X_train.loc[:, col].value_counts(dropna=False).plot.bar(ax=ax[0])
            X_test.loc[:, col].value_counts(dropna=False).plot.bar(ax=ax[1])
            ax[0].set_title(col + "(X_train)")
            if ks_dict is None:
                ax[1].set_title(col + "(X_test)")
            else:
                ks_result = ks_dict[col]
                title = "(X_test) ks stat:{}  p-value:{}".format(
                    str(round(ks_result[0], 4)), str(round(ks_result[1], 4)))
    
                if ks_result[1] < rejected:
                    ax[1].set_title(title, color='red')
                else:
                    ax[1].set_title(title)

obj_cols = filled_df.columns[filled_df.dtypes == object]
num_cols = filled_df.columns.drop(obj_cols)

from scipy import stats
def compare_distribution(train_index, test_index):
    X_filled_train = filled_df.loc[train_index, :]
    X_filled_test = filled_df.loc[test_index, :]
    X_le_train = label_encoded_df.loc[train_index, :]
    X_le_test = label_encoded_df.loc[test_index, :]
    ks_dict ={x: stats.ks_2samp(X_le_train[x], X_le_test[x]) for x in X_train_le.columns}
    print("KS-test statistic mean: {}".format(np.mean([x[0] for x in ks_dict.values()])))
    plot_num_data(X_filled_train, y_train[train_index], num_cols, X_filled_test, ks_dict=ks_dict)
    plot_obj_data(X_filled_train, obj_cols, X_filled_test, ks_dict=ks_dict)

from sklearn.model_selection import train_test_split
X1, X2, y1, y2 = train_test_split(X_train, y_train)
compare_distribution(X1.index, X2.index)
train_index = filled_df.loc[:1460, :].index
test_index = filled_df.loc[1461:, :].index
compare_distribution(train_index, test_index)