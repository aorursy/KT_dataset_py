# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import sklearn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trainDF = pd.read_csv("../input/train.tsv", sep = '\t')

testDF = pd.read_csv("../input/test.tsv", sep = '\t')
#pd.get_dummies(df['brand_name'].sample(100), prefix_sep = 'brand_name',  dummy_na = False, sparse = True)

class oneHotEncoder:



    def __init__(self, threshold):

        self.threshold = threshold

        

    @staticmethod

    def binary_variance(p):

        return p * (1 - p)

    

    def dum_sign(self, df, col, threshold=0.01):

        dummy_col = df[col].fillna('')

        dummy_col = dummy_col.astype(str)

        p = dummy_col.value_counts() / dummy_col.shape[0]

        mask = dummy_col.isin(p[self.binary_variance(p) >= threshold].index)

        dummy_col[~mask] = np.nan

        res = pd.get_dummies(dummy_col, prefix=col, dummy_na=False)

        return res

    

    def one_hot_encoding(self, X, threshold):

        dfs = []

        for col in X.columns:

            if type(threshold) == float:

                t = threshold

            elif col in threshold:

                t = threshold[col]

            else:

                t = 0.0

            df = self.dum_sign(X, col, t)

            dfs.append(df)

        res = pd.concat(dfs, axis=1)

        return res

    

    def fit_transform(self, df):

        res = self.one_hot_encoding(df, self.threshold)

        self.columns = res.columns

        return res

    

    def transform(self, df):

        res = self.one_hot_encoding(df, self.threshold)

        return res.reindex(columns = self.columns, fill_value=0)



import numpy as np

import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold

from itertools import product



class MeanEncoder:

    def __init__(self, categorical_features, n_splits=5, target_type='classification', prior_weight_func=None):

        """

        :param categorical_features: list of str, the name of the categorical columns to encode



        :param n_splits: the number of splits used in mean encoding



        :param target_type: str, 'regression' or 'classification'



        :param prior_weight_func:

        a function that takes in the number of observations, and outputs prior weight

        when a dict is passed, the default exponential decay function will be used:

        k: the number of observations needed for the posterior to be weighted equally as the prior

        f: larger f --> smaller slope

        """



        self.categorical_features = categorical_features

        self.n_splits = n_splits

        self.learned_stats = {}



        if target_type == 'classification':

            self.target_type = target_type

            self.target_values = []

        else:

            self.target_type = 'regression'

            self.target_values = None



        if isinstance(prior_weight_func, dict):

            self.prior_weight_func = eval('lambda x: 1 / (1 + np.exp((x - k) / f))', dict(prior_weight_func, np=np))

        elif callable(prior_weight_func):

            self.prior_weight_func = prior_weight_func

        else:

            self.prior_weight_func = lambda x: 1 / (1 + np.exp((x - 2) / 1))



    @staticmethod

    def mean_encode_subroutine(X_train, y_train, X_test, variable, target, prior_weight_func):

        X_train = X_train[[variable]].copy()

        X_test = X_test[[variable]].copy()



        if target is not None:

            nf_name = '{}_pred_{}'.format(variable, target)

            X_train['pred_temp'] = (y_train == target).astype(int)  # classification

        else:

            nf_name = '{}_pred'.format(variable)

            X_train['pred_temp'] = y_train  # regression

        prior = X_train['pred_temp'].mean()



        col_avg_y = X_train.groupby(by=variable, axis=0)['pred_temp'].agg({'mean': 'mean', 'beta': 'size'})

        col_avg_y['beta'] = prior_weight_func(col_avg_y['beta'])

        col_avg_y[nf_name] = col_avg_y['beta'] * prior + (1 - col_avg_y['beta']) * col_avg_y['mean']

        col_avg_y.drop(['beta', 'mean'], axis=1, inplace=True)



        nf_train = X_train.join(col_avg_y, on=variable)[nf_name].values

        nf_test = X_test.join(col_avg_y, on=variable).fillna(prior, inplace=False)[nf_name].values



        return nf_train, nf_test, prior, col_avg_y



    def fit_transform(self, X, y):

        """

        :param X: pandas DataFrame, n_samples * n_features

        :param y: pandas Series or numpy array, n_samples

        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features

        """

        X_new = X.copy()

        if self.target_type == 'classification':

            skf = StratifiedKFold(self.n_splits)

        else:

            skf = KFold(self.n_splits)



        if self.target_type == 'classification':

            self.target_values = sorted(set(y))

            self.learned_stats = {'{}_pred_{}'.format(variable, target): [] for variable, target in

                                  product(self.categorical_features, self.target_values)}

            for variable, target in product(self.categorical_features, self.target_values):

                nf_name = '{}_pred_{}'.format(variable, target)

                X_new.loc[:, nf_name] = np.nan

                for large_ind, small_ind in skf.split(y, y):

                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(

                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, target, self.prior_weight_func)

                    X_new.iloc[small_ind, -1] = nf_small

                    self.learned_stats[nf_name].append((prior, col_avg_y))

        else:

            self.learned_stats = {'{}_pred'.format(variable): [] for variable in self.categorical_features}

            for variable in self.categorical_features:

                nf_name = '{}_pred'.format(variable)

                X_new.loc[:, nf_name] = np.nan

                for large_ind, small_ind in skf.split(y, y):

                    nf_large, nf_small, prior, col_avg_y = MeanEncoder.mean_encode_subroutine(

                        X_new.iloc[large_ind], y.iloc[large_ind], X_new.iloc[small_ind], variable, None, self.prior_weight_func)

                    X_new.iloc[small_ind, -1] = nf_small

                    self.learned_stats[nf_name].append((prior, col_avg_y))

        return X_new



    def transform(self, X):

        """

        :param X: pandas DataFrame, n_samples * n_features

        :return X_new: the transformed pandas DataFrame containing mean-encoded categorical features

        """

        X_new = X.copy()



        if self.target_type == 'classification':

            for variable, target in product(self.categorical_features, self.target_values):

                nf_name = '{}_pred_{}'.format(variable, target)

                X_new[nf_name] = 0

                for prior, col_avg_y in self.learned_stats[nf_name]:

                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[

                        nf_name]

                X_new[nf_name] /= self.n_splits

        else:

            for variable in self.categorical_features:

                nf_name = '{}_pred'.format(variable)

                X_new[nf_name] = 0

                for prior, col_avg_y in self.learned_stats[nf_name]:

                    X_new[nf_name] += X_new[[variable]].join(col_avg_y, on=variable).fillna(prior, inplace=False)[

                        nf_name]

                X_new[nf_name] /= self.n_splits



        return X_new
#one hot

X = df['name item_condition_id category_name brand_name shipping'.split()]

Y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.01, random_state=0)



oneHotEnc = oneHotEncoder(0.002)

X_train_onehot = oneHotEnc.fit_transform(X_train)

X_test_onehot = oneHotEnc.transform(X_test)
X = trainDF['name item_condition_id category_name brand_name shipping'.split()]

Y = trainDF['price']

X_train, X_validation, y_train, y_validation = train_test_split(X, Y, test_size=0.05, random_state=0)





meanENC = MeanEncoder(X_validation.columns.tolist(), 5, 'regression', {"k":100, "f":100})

X_train_new = meanENC.fit_transform(X_train, y_train)

X_validataion_new = meanENC.transform(X_validation)



new_columns = []

for c in X_train_new.columns.tolist():

    if(c.find("pred") != -1):

        new_columns.append(c)

new_columns





X_train_new = X_train_new[new_columns]

X_validataion_new = X_validataion_new[new_columns]





minMaxScaler = sklearn.preprocessing.MinMaxScaler()

X_train_scaler = minMaxScaler.fit_transform(X_train_new)

X_validation_scaler = minMaxScaler.transform(X_validataion_new)

# ridge regression

def ridgeRegression(x_train, y_train, x_test, y_test):

    from sklearn import linear_model

    from sklearn import metrics

    reg = linear_model.Ridge(alpha = 1)

    reg.fit(x_train, y_train)

    pre_train = reg.predict(x_train)

    pre_test = reg.predict(x_test)

    print ("Train-RMSE:", np.sqrt(metrics.mean_squared_error(y_train, pre_train)))

    print ("Test-RMSE:", np.sqrt(metrics.mean_squared_error(y_test, pre_test)))

    print ("Train-MAPE:", metrics.mean_absolute_error(y_train, pre_train))

    print ("Test-MAPE:", metrics.mean_absolute_error(y_test, pre_test))

    return reg



reg = ridgeRegression(X_train_scaler, y_train, X_validation_scaler, y_validation)

testDF_meanENC = meanENC.transform(testDF['name item_condition_id category_name brand_name shipping'.split()])[new_columns]

testDF_meanENC_scaler = minMaxScaler.transform(testDF_meanENC)

pre_test = reg.predict(testDF_meanENC_scaler)

testDF['price'] = pre_test

m = testDF.pre_test.mean()

testDF['price'] = testDF.price.apply(lambda x: x if(x > 0) else m)



sub = testDF['test_id price'.split()]

sub.to_csv('sub_submission.csv', index = False)
# ridge regression

from sklearn import linear_model

from sklearn import metrics

reg = linear_model.Ridge(alpha = 1)

reg.fit(X_train_onehot, y_train)



pre_train = reg.predict(X_train_onehot)

pre_test = reg.predict(X_test_onehot)

print ("Train-RMSE:", np.sqrt(metrics.mean_squared_error(y_train, pre_train)))

print ("Test-RMSE:", np.sqrt(metrics.mean_squared_error(y_test, pre_test)))

print ("Train-MAPE:", metrics.mean_absolute_error(y_train, pre_train))

print ("Test-MAPE:", metrics.mean_absolute_error(y_test, pre_test))



X_test = oneHotEnc.transform(df_test['name item_condition_id category_name brand_name shipping'.split()])

pre_test = reg.predict(X_test)

df_test['price'] = pre_test
df_test['price'] = df_test.price.apply(lambda x: x if(x > 0) else 26.74936370209377)
sub = df_test['test_id price'.split()]

sub.to_csv('sub_submission.csv', index = False)
df_test[df_test.price<0]