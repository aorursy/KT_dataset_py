# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



class CSVReader:

    def getTrainingData(self):

        print(os.getcwd())

        pd.set_option('display.float_format',

                      lambda x: '{:.3f}'.format(x))  # Limiting floats output to 3 decimal points

        print(check_output(["ls", "/kaggle/input/train.csv"]).decode("utf8"))  # check the files available in the directory

        train = pd.read_csv('/kaggle/input/train.csv', engine='python')

        # print("train : " + str(train.shape))

        return train



    def getTestData(self):

        # print(os.getcwd())

        pd.set_option('display.float_format',

                      lambda x: '{:.3f}'.format(x))  # Limiting floats output to 3 decimal points

        print(check_output(["ls", "/kaggle/input/test.csv"]).decode("utf8"))  # check the files available in the directory

        test = pd.read_csv('/kaggle/input/test.csv', engine='python')

        # print("test : " + str(test.shape))

        return test



# Any results you write to the current directory are saved as output.
class DataPreprocessing():



    def process(self, all_data):

        self.all_data = all_data

        self.numericTranformation(self)

        self.catagoricalTransformation(self)



    def numericTranformation(self):

        # log transform skewed numeric features:

        numeric_features = self.all_data.dtypes[self.all_data.dtypes != "object"].index



        # remove missing values and finding skeweness value of complete dataframe

        # compute skewness fo numeric columns after dropping missing values.

        feature_skew_val = self.all_data[numeric_features].apply(lambda x: skew(x.dropna()))



        print("=====================")

        print(feature_skew_val)

        print("=====================")

        # extracting dataframe whose skewness is greater than 75%

        skewed_features = feature_skew_val[abs(feature_skew_val) > 0.75]

        print(skewed_features)

        print("======================")



        skewed_features_column_index = skewed_features.index

        print(skewed_features_column_index)

        print("======================")



        # log1p to all skewed numerical columns

        self.all_data[skewed_features_column_index] = np.log1p(self.all_data[skewed_features_column_index])

    def catagoricalTransformation(self):

        # Adding one hot encoding to all categorical data

        self.all_data = pd.get_dummies(self.all_data)

        # filling NA's with the mean of the column:

        self.all_data = self.all_data.fillna(self.all_data.mean())
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

class SimpleModelAssembler(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models



    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models = [clone(model) for model in self.models]

        # for cloned_model in self.models:

        #     self.models_.append(clone(cloned_model))



        # Train cloned base models

        for model in self.models:

            model.fit(X, y)



        return self



    # Now we do the predictions for cloned models and average them

    def predict(self, testDataSet):

        trained_models = []

        for model in self.models:

            trained_models.append(model.predict(testDataSet))

        predictions = np.column_stack(trained_models)

        # below line is one line solution of above loop

        # predictions = np.column_stack([model.predict(testDataSet) for model in self.models])

        return np.mean(predictions, axis=1)



# process method of DataPreprocessing class excepts

# panda's dataframe to be passed for process.
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone

from sklearn.model_selection import KFold





class StackingAveragedModel(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds



    # We again fit the data on clones of the original models

    def fit(self, X, y):

        """Fit all the models on the given dataset"""

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)



        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                instance.fit(X[train_index], y[train_index])

                self.base_models_[i].append(instance)

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred



        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self



    # Do the predictions of all base models on the test data and use the averaged predictions as

    # meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_])

        return self.meta_model_.predict(meta_features)



# https://gist.github.com/TannerGilbert/765fbf450454fb89ef7e83b59139316e

# Need to explore

# import lightgbm as lgb

# import xgboost as xgb

import os



import numpy as np

import pandas as pd

# from StackingAveragedModel import StackingAveragedModel

from scipy.stats import skew

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from subprocess import check_output



class ModelAssemblerEntryPoint:

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0015, random_state=1))

    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                       max_depth=4, max_features='sqrt',

                                       min_samples_leaf=15, min_samples_split=10,

                                       loss='huber', random_state=5)

    n_folds = 10



    def multiLayerAssembelModels(self):

        self.dataProcessing(self)

        # stacking model selection

        base_models = (self.ENet, self.GBoost, self.KRR, self.lasso)

        # models = (self.ENet, self.GBoost, self.KRR, self.lasso)

        stacked_averaged_models = StackingAveragedModel(base_models, meta_model=self.lasso)

        score = self.rmsle_cv(self, stacked_averaged_models, self.X_train, self.train_y)

        print(score)

        print("\n Mean Score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



    def assembelModels(self):

        self.dataProcessing()

        # stacking model selection

        models = (self.ENet, self.GBoost, self.KRR, self.lasso)

        # models = (self.KRR)

        modelsAssembler = SimpleModelAssembler(models)

        score = self.rmsle_cv(modelsAssembler, self.X_train, self.train_y)

        print(score)

        print("\n Mean Score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))







    # rmsle_cv has cross_val_score method which internally calls fit and predict method of model to train

    # and make prediction

    # root mean square logrithmic error

    def rmsle_cv(self, model, X_train, y_train):

        #todo: change

        #kf = KFold(self.n_folds, shuffle=True, random_state=42).get_n_splits(X_train.values)

        kf = KFold(self.n_folds, shuffle=True, random_state=42).get_n_splits(X_train)

        rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=kf))

        return (rmse)



    def rmsle(y, y_pred):

        return np.sqrt(mean_squared_error(y, y_pred))



    def dataProcessing(self):

        csvObj = CSVReader

        train = csvObj.getTrainingData(csvObj)

        test = csvObj.getTestData(csvObj)



        # Concat all the columns except index and target in all_data

        all_data = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'], test.loc[:, 'MSSubClass':'SaleCondition']))

        train["SalePrice"] = np.log1p(train["SalePrice"])



        dataProcess = DataPreprocessing

        dataProcess.process(dataProcess, all_data)



        # creating matrices for sklearn:

        self.X_train = dataProcess.all_data[:train.shape[0]].values

        self.train_y = np.log1p(train.SalePrice).values

class EstimatorEntryPoint:



    def simpleModelKickOff(self):

        print("AI started thinking")

        modelObj = ModelAssemblerEntryPoint()

        modelObj.assembelModels()



    def advanceModelKickOff(self):

        print("AI started thinking")

        modelObj = ModelAssemblerEntryPoint

        modelObj.multiLayerAssembelModels(modelObj)





if __name__ == '__main__':

    import time

    start_time = time.process_time()

    point = EstimatorEntryPoint()

    point.simpleModelKickOff()

    point.advanceModelKickOff()

    print("Total time taken:", time.process_time() - start_time)

    print("Its time to take some rest AI :) ")





# SimpleModelKickOff()

#     Results for now

# [0.00835614 0.01040651 0.01010513 0.00846736 0.01127534]

#  Mean Score: 0.0097 (0.0011)



# without log1p self.all_data[skewed_features_column_index] = np.log1p(self.all_data[skewed_features_column_index])

# [0.01170166 0.04439337 0.01385752 0.01011304 0.01753346]

#  Mean Score: 0.0195 (0.0127)