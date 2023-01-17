# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# y_train_log = np.log1p(self.y_train)

something = CustomClassifiers(X_train, y_train,X_train, y_train)

something.baseline_models()
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(n_estimators=4000,alpha=0.01); ### Test 0.41

yLabelsLog = np.log1p(y_train)

gbm.fit(X_train,yLabelsLog)

preds = gbm.predict(X= X_train)

print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
from xgboost import XGBClassifier

gbm = XGBClassifier(n_estimators=400,alpha=0.01); ### Test 0.41

yLabelsLog = np.log1p(y_train)

gbm.fit(X_train,yLabelsLog)

preds = gbm.predict(X= x_train)

print ("RMSLE Value For Gradient Boost: ",rmsle(np.exp(yLabelsLog),np.exp(preds),False))
train['year'].dtype
import asyncio
train['year'] = train['datetime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').year).astype('int')

train['month'] = train['datetime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').month).astype('int')

train['day'] = train['datetime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').day).astype('int')

train['hour'] = train['datetime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').hour).astype('int')



drop_features  = ['datetime', 'count', 'casual', 'registered', 'temp', 'windspeed']

categoricalFeatureNames = ["season","holiday","workingday","weather","month","year","hour"]

for var in categoricalFeatureNames:

    train[var] = train[var].astype("int")

    

test['year'] = test['datetime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').year).astype('int')

test['month'] = test['datetime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').month).astype('int')

test['day'] = test['datetime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').day).astype('int')

test['hour'] = test['datetime'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d %H:%M:%S').hour).astype('int')



test_drop_features  = ['datetime', 'temp', 'windspeed']

categoricalFeatureNames = ["season","holiday","workingday","weather","month","year","hour"]

for var in categoricalFeatureNames:

    test[var] = test[var].astype("int")



X_train, X_test, y_train, y_test = train_test_split(train.drop(drop_features, axis=1), train['count'], test_size = 0.3, random_state = 100)


something.baseline_results


def rmsle(y, y_, convertExp=True):

    if convertExp:

        y = np.exp(y),

        y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))



import warnings



import numpy as np

from lightgbm import LGBMClassifier

from sklearn import metrics

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,

                              ExtraTreesClassifier, GradientBoostingClassifier,

                              GradientBoostingRegressor,

                              RandomForestClassifier, RandomForestRegressor,

                              VotingClassifier)

from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDClassifier

from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,

                             r2_score, recall_score, mean_squared_error,

                             roc_auc_score)

from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier



# from Statistica.utils.metrics import rmsle





class CustomClassifiers():

    '''

    Class to Create and Run multiple Classifiers from Sklearn

    ---------------------------------------------------------

    Currently Supports:

    - Linear Models:

        - Linear Regression

        - Logistic Regression

        - LDA



    - Non Linear Models:

        - KNN

        - SVM

        - Naive Bayes

        - SGDClassifier



    - Regularization Models:

        - Ridge

        - Lasso



    - Ensemble Models:

        - Random Forest

        - Gradient Boost

    '''



    def __init__(

        self,

        x_train,

        y_train,

        x_test,

        y_test,

        # classifier,

        problem_type='Classification',

        n_classifiers=100,

        range_values=False,

        exec_all=False,

        classifiers_list=None

    ):

        '''

        Initiaizing the CustomClassifier Class:



        Parameters:

        ----------------------------------------

        x_train: pd.DataFrame

            Features Data.

        y_train: pd.DataFrame

            Target Data.

        x_test: pd.DataFrame

            Test Features Data.

        y_Test: pd.DataFrame

            Test Target Data.

        n_classifiers: int

            n_classifiers argumnet for sklearn Classifiers.

        range_values: bool

            Get model fits on range of values in range 0..n_classifiers.

        exec_all: bool

            Executes all the implemented models.

        classfiers_list: list

            List of the Models to be executed.



        '''

        self.x_train = x_train

        self.y_train = y_train

        self.x_test = x_test

        self.y_test = y_test

        self.n_classifiers = n_classifiers

        self.range_value = range_values

        self.exec_all = exec_all

        self.classifiers_list = classifiers_list



        if self.classifiers_list is None:

            self.exec_all = True



        self.baseline_results = []



        self.many_results = []

        self.problem_type = problem_type

        self.best = None



    def set_classifier(self, classifier):

        '''

        Function returns the Classifier from the Same string name.

        ----------------------------------------------------------

        Parameters:

        !INFO: The names of Classifiers are the same as the OG imports in lowercase.!

        - self: CustomClassifiers Object

            Refernece to the Instance Variable.

        - classifier: string

            Name of the Classifier



        Returns:

            Required Classifier

        '''



        if classifier == 'LinearRegression':

            return LinearRegression

        elif classifier == 'LogisticRegression':

            return LogisticRegression

        elif classifier == 'KNN':

            return knn

        elif classifier == 'SVM':

            return svm

        elif classifier == 'LDA':

            return lda

        elif classifier == 'DecisionTreeClassifier':

            return decisiontreeclassifier

        elif classifier == 'RandomForestClassifier':

            return randomforestclassifier

        elif classifier == 'Lasso':

            return Lasso

        elif classifier == 'Ridge':

            return ridge

        elif classifier == 'SGDClassifier':

            return sgdclassifier

        elif classifier == 'ExtraTreesClassifier':

            return ExtraTreesClassifier

        elif classifier == 'BaggedDecisionTreeClassifier':

            return BaggedDecisionTreeClassifier

        elif classifier == 'AdaBoostClassifier':

            return AdaBoostClassifier

        # ! UnderDev

        elif classifier == 'ViolinThing':

            return violinthing



        elif classifier == 'XGBClassifier':

            return XGBClassifier

        elif classifier == 'LGBMClassifier':

            return LGBMClassifier

        elif classifier == "GradientBoostingRegressor":

            return GradientBoostingRegressor

        else:

            raise NotImplementedError('{} Is not yet implemented'.format(classifier))



    def set_metric(self, metric=None):

        if metric:

            if metric == 'accuracy_score':

                return accuracy_score

            elif metric == "confusion_matrix":

                return confusion_matrix

            elif metric == "precision_score":

                return precision_score

            elif metric == "mean_squared_error":

                return mean_squared_error

            elif metric == "recall_score":

                return recall_score

            elif metric == "r2_score":

                return r2_score

            elif metric == "roc_auc_score":

                return roc_auc_score

        else:

            if self.problem_type == "Classification":

                return rmsle

            elif self.problem_type == "Regression":

                # options = [Mean Absolute Error. Mean Squared Error. R^2.]

                return accuracy_score

            else:

                raise NotImplementedError



    def baseline_models(self, ret=False):

        model_list = [

            'Lasso',

            'GradientBoostingRegressor',

            # 'AdaBoost'

        ]

        for model in model_list:

            clf = self.set_classifier(model)()

            metric = self.set_metric()

            if metric.__name__ == 'rmsle':

                y_train_log = np.log1p(self.y_train)

                clf.fit(self.x_train, y_train_log)

                pred = clf.predict(self.x_test)

                acc = self.rmsle(np.exp(y_train_log), np.exp(pred), False)

                self.baseline_results.append([

                    model, acc

                ])

        for result in self.baseline_results:

            print(f"[{result[0]}] Accuracy Score: {result[1]}")

        return

        acc = self.set_metric()(self.y_test, pred)

        # try:

        #     clf.fit(self.x_train, self.y_train)

        #     pred = clf.predict(self.x_test)

        #     acc = accuracy_score(self.y_test, pred)

        # except ValueError:

        #     yLabelsLog = np.log1p(self.y_train)

        #     clf.fit(self.x_train, self.y_train)

        #     pred = clf.predict(self.x_test)

        #     # acc = self.set_metric()(self.y_test, pred)

        #     acc = self.rmsle(np.exp(yLabelsLog), np.exp(pred), False)

        self.baseline_results.append([

            model, acc

        ])

        for result in self.baseline_results:

            print(f"[{result[0]}] Accuracy Score: {result[1]}")



    def rmsle(self, y, y_, convertExp=True):

        if convertExp:

            y = np.exp(y),

            y_ = np.exp(y_)

        log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

        log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

        calc = (log1 - log2) ** 2

        return np.sqrt(np.mean(calc))



    # TODO: Change the Accuracy thing for this

    def lr(self):

        clf = LinearRegression()

        # clf = np.log1p(self.y_train)

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(self.x_test)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])

        # Cant use with Regression

        print("The AccuracyScore for Linear Regersion: ", )



    def ridge(self):

        ridge_m_ = Ridge()

        ridge_params_ = {'max_iter': [3000], 'alpha': [0.1, 1, 2, 3, 4, 10, 30,

                                                       100, 200, 300, 400, 800,

                                                       900, 1000]}

        rmsle_scorer = metrics.make_scorer(accuracy_score,

                                           greater_is_better=False)

        grid_ridge_m = GridSearchCV(ridge_m_, ridge_params_,

                                    scoring=rmsle_scorer, cv=5)

        yLabelsLog = np.log1p(self.y_train)

        grid_ridge_m.fit(self.x_train, yLabelsLog)

        pred = grid_ridge_m.predict(X=self.x_train)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])

        print("RMSLE Value For Extra Trees is: ",

              accuracy_score(self.y_test, pred))



    def lasso(self):

        lasso_m_ = Lasso()



        alpha = 1 / np.array([0.1, 1, 2, 3, 4, 10, 30,

                              100, 200, 300, 400, 800,

                              900, 1000])

        lasso_params_ = {'max_iter': [3000], 'alpha': alpha}

        rmsle_scorer = metrics.make_scorer(self.rmsle, greater_is_better=False)

        grid_lasso_m = GridSearchCV(lasso_m, lasso_params,

                                    scoring=rmsle_scorer, cv=5)



        yLabelsLog = np.log1p(self.y_train)

        grid_lasso_m.fit(self.x_train, yLabelsLog)

        pred = grid_lasso_m.predict(X=self.x_train)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])

        print("RMSLE Value For Extra Trees is: ",

              accuracy_score(self.y_test, pred))



    def randomT(self):

        clf = RandomForestRegressor(n_estimators=100)

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(X=self.x_train)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])



        print("RMSLE Value For Extra Trees is: ",

              accuracy_score(self.y_test, pred))



    def gradB_r(self):

        clf = GradientBoostingRegressor(n_estimators=4000, alpha=0.01)

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(X=self.x_train)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])



        print("RMSLE Value For Extra Trees is: ",

              accuracy_score(self.y_test, pred))



    def sgd(self):

        clf = SGDClassifier()

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(self.x_test)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])



        print("RMSLE Value For SGD: ",

              accuracy_score(self.y_test, pred))



    def exT(self):

        clf = ExtraTreesClassifier(n_jobs=-1, n_estimators=100)

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(self.x_test)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])



        print("RMSLE Value For Extra Trees is: ",

              accuracy_score(self.y_test, pred))



    def lda(self):

        clf = LinearDiscriminantAnalysis()

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(self.x_test)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])



        print("RMSLE Value For LDA is: ",

              accuracy_score(self.y_test, pred))



    def knn(self, n_neighbors=100):

        clf = KNeighborsClassifier(n_jobs=-1, n_neighbors=n_neighbors)

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(self.x_test)

        self.baseline_results.append(['lr',

                                      accuracy_score(self.y_test, pred)])



        print("RMSLE Value For KNN is: ",

              accuracy_score(self.y_test, pred))



    def naive_bayes(self):

        clf = GaussianNB()

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(self.x_test)

        print("RMSLE Value For Naive_Bayes is: ",

              accuracy_score(self.y_test, pred))



    def SVM(self, c):

        clf = SVC(C=c)

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(self.x_test)

        self.baseline_results.append(['lr',

                                      accuracy_score(self.y_test, pred)])



        print("RMSLE Value For SVM is: ",

              accuracy_score(self.y_test, pred))



    def bgt(self):

        base_estimator = DecisionTreeClassifier(max_depth=13)

        n_list = [100]

        for n_estimators in n_list:

            clf = BaggingClassifier(n_jobs=-1, base_estimator=base_estimator,

                                    n_estimators=n_estimators)

            clf.fit(self.x_train, self.y_train)

            pred = clf.predict(self.x_test)

            self.baseline_results.append(['lr',

                                          accuracy_score(self.y_test, pred)])



            print("RMSLE Value For BaggedDecisionTrees is: ",

                  accuracy_score(self.y_test, pred))



    def randomT_bagging(self):

        n_list = [100]

        for n_estimators in n_list:

            clf = RandomForestClassifier(n_jobs=-1, n_estimators=n_estimators)

            clf.fit(self.x_train, self.y_train)

            pred = clf.predict(self.x_test)

            self.baseline_results.append(

                ['lr', accuracy_score(self.y_test, pred)])



            print("RMSLE Value For RandomTreesBaggin is: ",

                  accuracy_score(self.y_test, pred))



    def abaBoost(self, n_estimators=100):

        clf = AdaBoostClassifier(n_estimators=n_estimators)

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(self.x_test)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])



        print("RMSLE Value For adaBoost is: ",

              accuracy_score(self.y_test, pred))



    def gradB_c(self, max_depth=100):

        clf = GradientBoostingClassifier(max_depth=max_depth)

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(self.x_test)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])



        print("RMSLE Value For GradientBoostingClassifier is: ",

              accuracy_score(self.y_test, pred))



    # ! Still needs some work BOIS

    def voting_classifier(self):

        list_estimators = []



        estimators = []

        model1 = ExtraTreesClassifier(n_jobs=-1, n_estimators=100)

        estimators.append(('et', model1))

        model2 = RandomForestClassifier(n_jobs=-1, n_estimators=100)

        estimators.append(('rf', model2))

        from sklearn.ensemble import BaggingClassifier

        from sklearn.tree import DecisionTreeClassifier

        base_estimator = DecisionTreeClassifier(max_depth=13)

        model3 = BaggingClassifier(n_jobs=-1, base_estimator=base_estimator,

                                   n_estimators=100)

        estimators.append(('bag', model3))



        list_estimators.append(['Voting', estimators])



        for name, estimators in list_estimators:

            clf = VotingClassifier(estimators=estimators, n_jobs=-1)

            clf.fit(self.x_train, self.y_train)

            pred = clf.predict(self.x_test)

            print("RMSLE Value For VotingClassifier is: ",

                  accuracy_score(self.y_test, pred))



    def xgboosht(self, n_estimators):

        clf = XGBClassifier(n_estimators=n_estimators, subsample=0.25)

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(self.x_test)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])



        print(

            "RMSLE Value For XgBoost is: ",

            accuracy_score(

                self.y_test,

                pred))



    def lgbm(self):

        clf = LGBMClassifier(random_state=17)

        clf.fit(self.x_train, self.y_train)

        pred = clf.predict(self.x_test)

        self.baseline_results.append(['lr', accuracy_score(self.y_test, pred)])



        print(

            "RMSLE Value For XgBoost is: ",

            accuracy_score(

                self.y_test,

                pred))



    def print_shit(self):

        raise NotImplementedError



    def executer(self):

        if self.exec_all:

            self.exT()

            self.gradB_c()

            self.gradB_r()

            self.knn()

            self.lasso()

            self.lr()

            self.randomT_bagging()

            self.abaBoost()

            self.bgt()

            self.naive_bayes()



        elif self.classifiers_list:

            for classifier in self.classifiers_list:

                # calling the classifier here

                pass
