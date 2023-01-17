import numpy as np

import pandas as pd

from hyperopt import hp, tpe, Trials

from hyperopt.fmin import fmin

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

import hyperopt

from sklearn.preprocessing import LabelEncoder
!ls ../input
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df_train.head()
df_train.shape
df_test.shape
df_train["train"] = 1

df_test["train"] = 0

df_all = pd.concat([df_train, df_test], sort=False)

df_all.head()
df_all.tail()
y = df_train["Survived"]

y.head()
def parse_cabin_type(x):

    if pd.isnull(x):

        return None

    #print("X:"+x[0])

    #cabin id consists of letter+numbers. letter is the type/deck, numbers are cabin number on deck

    return x[0]
def parse_cabin_number(x):

    if pd.isnull(x):

        return -1

#        return np.nan

    cabs = x.split()

    cab = cabs[0]

    num = cab[1:]

    if len(num) < 2:

        return -1

        #return np.nan

    return num
def parse_cabin_count(x):

    if pd.isnull(x):

        return np.nan

    #a typical passenger has a single cabin but some had multiple. multiple cabin ids are space separated

    cabs = x.split()

    return len(cabs)
df_train.dtypes
cabin_types = df_all["Cabin"].apply(lambda x: parse_cabin_type(x))

cabin_types = cabin_types.unique()

#drop the nan value from list of cabin types

cabin_types = np.delete(cabin_types, np.where(cabin_types == None))

cabin_types
df_all["cabin_type"] = df_all["Cabin"].apply(lambda x: parse_cabin_type(x))

df_all["cabin_num"] = df_all["Cabin"].apply(lambda x: parse_cabin_number(x))

df_all["cabin_count"] = df_all["Cabin"].apply(lambda x: parse_cabin_count(x))

df_all["cabin_num"] = df_all["cabin_num"].astype(int)

df_all[["Cabin", "cabin_type", "cabin_count", "cabin_num"]].head()
embarked_dummies = pd.get_dummies(df_all["Embarked"], prefix="embarked_", dummy_na=True)

#TODO: see if imputing embardked makes a difference

df_all = pd.concat([df_all, embarked_dummies], axis=1)

df_all[[col for col in df_all.columns if 'embarked_' in col]].head()
cabin_type_dummies = pd.get_dummies(df_all["cabin_type"], prefix="cabin_type_", dummy_na=True)

df_all = pd.concat([df_all, cabin_type_dummies], axis=1)

df_all[[col for col in df_all.columns if 'cabin_type_' in col]].head()
l_enc = LabelEncoder()

df_all["sex_label"] = l_enc.fit_transform(df_all["Sex"])

df_all[["Sex", "sex_label"]].head()
df_all["family_size"] = df_all["SibSp"] + df_all["Parch"] + 1

df_all[["SibSp", "Parch", "family_size"]].head()
# Cleaning name and extracting Title

for name_string in df_all['Name']:

    df_all['Title'] = df_all['Name'].str.extract('([A-Za-z]+)\.', expand=True)

df_all[["Name", "Title"]].head()
df_all["Title"].unique()
# Replacing rare titles 

mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs', 'Major': 'Other', 

           'Col': 'Other', 'Dr' : 'Other', 'Rev' : 'Other', 'Capt': 'Other', 

           'Jonkheer': 'Royal', 'Sir': 'Royal', 'Lady': 'Royal', 

           'Don': 'Royal', 'Countess': 'Royal', 'Dona': 'Royal'}

           

df_all.replace({'Title': mapping}, inplace=True)

#titles = ['Miss', 'Mr', 'Mrs', 'Royal', 'Other', 'Master']
titles = df_all["Title"].unique()

titles
title_dummies = pd.get_dummies(df_all["Title"], prefix="title", dummy_na=True)

df_all = pd.concat([df_all, title_dummies], axis=1)

df_all.head()
df_all['Age'].isnull().sum()
df_all["Age"].value_counts().count()
titles = list(titles)

# Replacing missing age by median age for title 

for title in titles:

    age_to_impute = df_all.groupby('Title')['Age'].median()[titles.index(title)]

    df_all.loc[(df_all['Age'].isnull()) & (df_all['Title'] == title), 'Age'] = age_to_impute
df_all['Age'].isnull().sum()
df_all["Age"].value_counts().count()
df_all.groupby('Pclass').agg({'Fare': lambda x: x.isnull().sum()})
df_all[df_all["Fare"].isnull()]
df_all.loc[152]
p3_median_fare = df_all[df_all["Pclass"] == 3]["Fare"].median()

p3_median_fare
df_all["Fare"].fillna(p3_median_fare, inplace=True)
df_all.loc[152]
#name col seems to be in format "last name, first names". 

#so split by comma and take first item in resulting list should give last name..

df_all['Last_Name'] = df_all['Name'].apply(lambda x: str.split(x, ",")[0])

df_all[['Name', 'Last_Name']].head()
#this would be the default value if no family member is found

DEFAULT_SURVIVAL_VALUE = 0.5

df_all['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

whatisthis1 = None

whatisthis2 = None

for grp, grp_df in df_all[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):



    if (len(grp_df) != 1):

        # A Family group is found.

        for ind, row in grp_df.iterrows():

            if ind == 312: #570

                print("t1")

                whatisthis1 = grp_df

            smax = grp_df.drop(ind)['Survived'].max()

            smin = grp_df.drop(ind)['Survived'].min()

            passID = row['PassengerId']

            if (smax == 1.0):

                df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 1

            elif (smin == 0.0):

                df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 0



for _, grp_df in df_all.groupby('Ticket'):

    if grp_df.iloc[0]["Ticket"] == "LINE":

        continue

    if (len(grp_df) != 1):

        for ind, row in grp_df.iterrows():

            if ind == 312:

                print("t2")

                whatisthis2 = grp_df

                

            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

                smax = grp_df.drop(ind)['Survived'].max()

                smin = grp_df.drop(ind)['Survived'].min()

                passID = row['PassengerId']

                if (smax == 1.0):

                    df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 1

                elif (smin == 0.0):

                    df_all.loc[df_all['PassengerId'] == passID, 'Family_Survival'] = 0
#whatisthis1
#whatisthis2
df_all.isnull().sum()
df_all['cabin_type'].unique()
df_all['Embarked'] = df_all['Embarked'].fillna('S')

df_all['cabin_type'] = df_all['cabin_type'].fillna('unknown')

df_all['Cabin'] = df_all['Cabin'].fillna('unknown')

df_all['cabin_count'] = df_all['cabin_count'].fillna(0)

df_train = df_all[df_all["train"] == 1]

df_test = df_all[df_all["train"] == 0]
X_cols = set(df_train.columns)



X_cols -= set(['PassengerId', 'Survived', 'Sex', 'Name', 'Ticket', 'Cabin', 

               'Embarked', 'cabin_type', 'Title', 'train', 'Last_Name'])

X_cols = list(X_cols)

X_cols

df_train[X_cols].head()
class OptimizerResult:

    avg_accuracy = None,

    misclassified_indices = None,

    misclassified_expected = None,

    misclassified_actual = None,

    oof_predictions = None,

    predictions = None,

    df_misses = None,

    all_accuracies = None,

    all_losses = None,

    all_params = None,

from sklearn.model_selection import cross_val_score, StratifiedKFold

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score



def stratified_test_prediction_avg_vote(clf, X_train, X_test, y, n_folds, n_classes, fit_params):

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=69)

    #N columns, one per target label. each contains probability of that value

    sub_preds = np.zeros((X_test.shape[0], n_classes))

    oof_preds = np.zeros((X_train.shape[0]))

    use_eval_set = fit_params.pop("use_eval_set")

    score = 0

    acc_score = 0

    acc_score_total = 0

    misclassified_indices = []

    misclassified_expected = []

    misclassified_actual = []

    for i, (train_index, test_index) in enumerate(folds.split(X_train, y)):

        print('-' * 20, i, '-' * 20)



        X_val, y_val = X_train.iloc[test_index], y[test_index]

        if use_eval_set:

            clf.fit(X_train.iloc[train_index], y[train_index], eval_set=([(X_val, y_val)]), **fit_params)

        else:

            #random forest does not know parameter "eval_set" or "verbose"

            clf.fit(X_train.iloc[train_index], y[train_index], **fit_params)

        #could directly do predict() here instead of predict_proba() but then mismatch comparison would not be possible

        oof_preds[test_index] = clf.predict_proba(X_train.iloc[test_index])[:,1].flatten()

        #we predict on whole test set, thus split by n_splits, not n_splits - 1

        sub_preds += clf.predict_proba(X_test) / folds.n_splits

#        sub_preds += clf.predict(X_test) / folds.n_splits

#        score += clf.score(X_train.iloc[test_index], y[test_index])

        preds_this_round = oof_preds[test_index] >= 0.5

        acc_score = accuracy_score(y[test_index], preds_this_round)

        acc_score_total += acc_score

        print('accuracy score ', acc_score)

        if hasattr(clf, 'feature_importances_'):

            importances = clf.feature_importances_

            features = X_train.columns



            feat_importances = pd.Series(importances, index=features)

            feat_importances.nlargest(30).sort_values().plot(kind='barh', color='#86bf91', figsize=(10, 8))

            plt.show()

        else:

            print("classifier has no feature importances: skipping feature plot")



        missed = y[test_index] != preds_this_round

        misclassified_indices.extend(test_index[missed])

        m1 = y[test_index][missed]

        misclassified_expected.append(m1)

        m2 = oof_preds[test_index][missed].astype("int")

        misclassified_actual.append(m2)



    print(f"acc_score: {acc_score}")

    sub_sub = sub_preds[:5]

    print(f"sub_preds: {sub_sub}")

    avg_accuracy = acc_score_total / folds.n_splits

    print('Avg Accuracy', avg_accuracy)

    result = OptimizerResult()

    result.avg_accuracy = avg_accuracy

    result.misclassified_indices = misclassified_indices

    result.misclassified_expected = misclassified_expected

    result.misclassified_actual = misclassified_actual

    result.oof_predictions = oof_preds

    result.predictions = sub_preds

    return result

#check if given parameter can be interpreted as a numerical value

def is_number(s):

    if s is None:

        return False

    try:

        float(s)

        return True

    except ValueError:

        return False



#convert given set of paramaters to integer values

#this at least cuts the excess float decimals if they are there

def convert_int_params(names, params):

    for int_type in names:

        #sometimes the parameters can be choices between options or numerical values. like "log2" vs "1-10"

        raw_val = params[int_type]

        if is_number(raw_val):

            params[int_type] = int(raw_val)

    return params



#convert float parameters to 3 digit precision strings

#just for simpler diplay and all

def convert_float_params(names, params):

    for float_type in names:

        raw_val = params[float_type]

        if is_number(raw_val):

            params[float_type] = '{:.3f}'.format(raw_val)

    return params

import numpy as np

import pandas as pd

from hyperopt import hp, tpe, Trials

from hyperopt.fmin import fmin

import catboost

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

import hyperopt

from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score, log_loss



# run n_folds of cross validation on the data

# averages fold results

def fit_cv(X, y, params, fit_params, n_classes, classifier, max_n, n_folds, print_summary, verbosity):

    # cut the data if max_n is set

    if max_n is not None:

        X = X[:max_n]

        y = y[:max_n]



    fit_params = fit_params.copy()

    use_eval = fit_params.pop("use_eval_set")

    score = 0

    acc_score = 0

    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=69)



    if print_summary:

        print(f"Running {n_folds} folds...")

    oof_preds = np.zeros((X.shape[0], n_classes))

    for i, (train_index, test_index) in enumerate(folds.split(X, y)):

        if verbosity > 0:

            print('-' * 20, f"RUNNING FOLD: {i}/{n_folds}", '-' * 20)



        X_train, y_train = X.iloc[train_index], y[train_index]

        X_test, y_test = X.iloc[test_index], y[test_index]

        #https://catboost.ai/docs/concepts/python-reference_parameters-list.html#python-reference_parameters-list

        #clf = catboost.CatBoostClassifier(**params)

        clf = classifier(**params)

        # verbose = print loss at every "verbose" rounds.

        #if 100 it prints progress 100,200,300,... iterations

        if use_eval:

            clf.fit(X_train, y_train, eval_set=(X_test, y_test), **fit_params)

        else:

            clf.fit(X_train, y_train, **fit_params)

        oof_preds[test_index] = clf.predict_proba(X.iloc[test_index])

        #score += clf.score(X.iloc[test_index], y[test_index])

        acc_score += accuracy_score(y[test_index], oof_preds[test_index][:,1] >= 0.5)

        # print('score ', clf.score(X.iloc[test_index], y[test_index]))

        #importances = clf.feature_importances_

        features = X.columns

    #accuracy is calculated each fold so divide by n_folds.

    #not n_folds -1 because it is not sum by row but overall sum of accuracy of all test indices

    total_acc_score = acc_score / n_folds

    logloss = log_loss(y, oof_preds)

    if print_summary:

        print(f"total acc: {total_acc_score}, logloss={logloss}")

    return total_acc_score, logloss
def create_misclassified_dataframe(result, y):

    oof_series = pd.Series(result.oof_predictions[result.misclassified_indices])

    oof_series.index = y[result.misclassified_indices].index

    miss_scale_raw = y[result.misclassified_indices] - result.oof_predictions[result.misclassified_indices]

    miss_scale_abs = abs(miss_scale_raw)

    df_miss_scale = pd.concat([miss_scale_raw, miss_scale_abs, oof_series, y[result.misclassified_indices]], axis=1)

    df_miss_scale.columns = ["Raw_Diff", "Abs_Diff", "Prediction", "Actual"]

    result.df_misses = df_miss_scale

class CatboostOptimizer:

    # how many CV folds to do on the data

    n_folds = 5

    # max number of rows to use for X and y. to reduce time and compare options faster

    max_n = None

    # max number of trials hyperopt runs

    n_trials = 200

    #verbosity in LGBM is how often progress is printed. with 100=print progress every 100 rounds. 0 is quite?

    verbosity = 0

    #if true, print summary accuracy/loss after each round

    print_summary = False



    all_accuracies = []

    all_losses = []

    all_params = []



    def objective_sklearn(self, params):

        int_types = ["depth"]

        params = convert_int_params(int_types, params)

        params["iterations"] = 1000

        params["early_stopping_rounds"] = 10

        if params['bootstrap_type'].lower() != "bayesian":

            #catboost gives error if bootstrap option defined with bootstrap disabled

            del params['bagging_temperature']



    #    n_classes = params["num_class"]

        n_classes = params.pop("num_class")

        

        score, logloss = fit_cv(self.X, self.y, params, self.fit_params, n_classes, catboost.CatBoostClassifier, 

                                self.max_n, self.n_folds, self.print_summary, verbosity=self.verbosity)

        self.all_params.append(params)

        self.all_accuracies.append(score)

        self.all_losses.append(logloss)

        if self.verbosity == 0:

            if self.print_summary:

                print("Score {:.3f}".format(score))

        else:

            print("Score {:.3f} params {}".format(score, params))

        #using logloss here for the loss but uncommenting line below calculates it from average accuracy

    #    loss = 1 - score

        loss = logloss

        result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}

        return result



    def optimize_catboost(self, n_classes, max_n_search):

        # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst

        #https://indico.cern.ch/event/617754/contributions/2590694/attachments/1459648/2254154/catboost_for_CMS.pdf

        space = {

            #'shrinkage': hp.loguniform('shrinkage', -7, 0),

            'depth': hp.quniform('depth', 2, 10, 1),

            'rsm': hp.uniform('rsm', 0.5, 1),

            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),

            'border_count': hp.qloguniform('border_count', np.log(32), np.log(255), 1),

            #'ctr_border_count': hp.qloguniform('ctr_border_count', np.log(32), np.log(255), 1),

            'l2_leaf_reg': hp.quniform('l2_leaf_reg', 0, 5, 1),

            'leaf_estimation_method': hp.choice('leaf_estimation_method', ['Newton', 'Gradient']),

            'bootstrap_type': hp.choice('bootstrap_type', ['Bayesian', 'Bernoulli', 'No']), #Poisson also possible for GPU

            'bagging_temperature': hp.loguniform('bagging_temperature', np.log(1), np.log(3)),

            'use_best_model': True

            #'gradient_iterations': hp.quniform('gradient_iterations', 1, 100, 1),

        }



        self.max_n = max_n_search



        if n_classes > 2:

            space['objective'] = "multiclass"

            space["num_class"] = n_classes

            space["eval_metric"] = "multi_logloss"

        else:

            space['objective'] = "Logloss"

            space["num_class"] = 2

            #space["eval_metric"] = ["Logloss"]

            #space["num_class"] = 1



        trials = Trials()

        best = fmin(fn=self.objective_sklearn,

                    space=space,

                    algo=tpe.suggest,

                    max_evals=self.n_trials,

                   trials=trials)



        # find the trial with lowest loss value. this is what we consider the best one

        idx = np.argmin(trials.losses())

        print(idx)



        print(trials.trials[idx])



        params = trials.trials[idx]["result"]["params"]

        print(params)

        return params



    # run a search for binary classification

    def classify_binary(self, X_cols, df_train, df_test, y_param):

        self.y = y_param



        self.X = df_train[X_cols]

        self.X_test = df_test[X_cols]



        self.fit_params = {'verbose': self.verbosity, 

                         'use_eval_set': True}



        # use 2 classes as this is a binary classification

        # the second param is the number of rows to use for training

        params = self.optimize_catboost(2, 5000)

        print(params)



        clf = catboost.CatBoostClassifier(**params)



        search_results = stratified_test_prediction_avg_vote(clf, self.X, self.X_test, self.y,

                                                             n_folds=self.n_folds, n_classes=2, fit_params=self.fit_params)

        search_results.all_accuracies = self.all_accuracies

        search_results.all_losses = self.all_losses

        search_results.all_params = self.all_params

        return search_results

import numpy as np

from sklearn.model_selection import StratifiedKFold

import lightgbm as lgbm

from hyperopt import hp, tpe, Trials

from hyperopt.fmin import fmin

import pandas as pd

from sklearn.preprocessing import LabelEncoder

import hyperopt



class LGBMOptimizer:

    # how many CV folds to do on the data

    n_folds = 5

    # max number of rows to use for training (from X and y). to reduce time and compare options faster

    max_n = None

    # max number of trials hyperopt runs

    n_trials = 200

    #verbosity in LGBM is how often progress is printed. with 100=print progress every 100 rounds. 0 is quite?

    verbosity = 0

    #if true, print summary accuracy/loss after each round

    print_summary = False



    from sklearn.metrics import accuracy_score, log_loss



    all_accuracies = []

    all_losses = []

    all_params = []



    def create_fit_params(self, params):

        using_dart = params['boosting_type'] == "dart"

        if params["objective"] == "binary":

            # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst

            fit_params = {"eval_metric": ["binary_logloss", "auc"]}

        else:

            fit_params = {"eval_metric": "multi_logloss"}

        if using_dart:

            n_estimators = 2000

        else:

            n_estimators = 15000

            fit_params["early_stopping_rounds"] = 100

        params["n_estimators"] = n_estimators

        fit_params['use_eval_set'] = True

        fit_params['verbose'] = self.verbosity

        return fit_params



    # this is the objective function the hyperopt aims to minimize

    # i call it objective_sklearn because the lgbm functions called use sklearn API

    def objective_sklearn(self, params):

        int_types = ["num_leaves", "min_child_samples", "subsample_for_bin", "min_data_in_leaf"]

        params = convert_int_params(int_types, params)



        # Extract the boosting type

        params['boosting_type'] = params['boosting_type']['boosting_type']

        #    print("running with params:"+str(params))



        fit_params = self.create_fit_params(params)

        if params['objective'] == "binary":

            n_classes = 2

        else:

            n_classes = params["num_class"]



        score, logloss = fit_cv(self.X, self.y, params, fit_params, n_classes, lgbm.LGBMClassifier, 

                                self.max_n, self.n_folds, self.print_summary, verbosity=self.verbosity)

        self.all_params.append(params)

        self.all_accuracies.append(score)

        self.all_losses.append(logloss)

        if self.verbosity == 0:

            if self.print_summary:

                print("Score {:.3f}".format(score))

        else:

            print("Score {:.3f} params {}".format(score, params))

    #using logloss here for the loss but uncommenting line below calculates it from average accuracy

    #    loss = 1 - score

        loss = logloss

        result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}

        return result



    def optimize_lgbm(self, n_classes):

        # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst

        # https://indico.cern.ch/event/617754/contributions/2590694/attachments/1459648/2254154/catboost_for_CMS.pdf

        space = {

            #this is just piling on most of the possible parameter values for LGBM

            #some of them apparently don't make sense together, but works for now.. :)

            'class_weight': hp.choice('class_weight', [None, 'balanced']),

            'boosting_type': hp.choice('boosting_type',

                                       [{'boosting_type': 'gbdt',

    #                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)

                                         },

                                        {'boosting_type': 'dart',

    #                                     'subsample': hp.uniform('dart_subsample', 0.5, 1)

                                         },

                                        {'boosting_type': 'goss'}]),

            'num_leaves': hp.quniform('num_leaves', 30, 150, 1),

            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),

            'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),

            'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),

            'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1), #alias "subsample"

            'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),

            'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),

            'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),

            'verbose': -1,

            #the LGBM parameters docs list various aliases, and the LGBM implementation seems to complain about

            #the following not being used due to other params, so trying to silence the complaints by setting to None

            'subsample': None, #overridden by bagging_fraction

            'reg_alpha': None, #overridden by lambda_l1

            'reg_lambda': None, #overridden by lambda_l2

            'min_sum_hessian_in_leaf': None, #overrides min_child_weight

            'min_child_samples': None, #overridden by min_data_in_leaf

            'colsample_bytree': None, #overridden by feature_fraction

    #        'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),

            'min_child_weight': hp.loguniform('min_child_weight', -16, 5), #also aliases to min_sum_hessian

    #        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),

    #        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),

    #        'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),

        }

        if n_classes > 2:

            space['objective'] = "multiclass"

            space["num_class"] = n_classes

        else:

            space['objective'] = "binary"

            #space["num_class"] = 1



        trials = Trials()

        best = fmin(fn=self.objective_sklearn,

                    space=space,

                    algo=tpe.suggest,

                    max_evals=self.n_trials,

                    trials=trials,

                   verbose= 1)



        # find the trial with lowest loss value. this is what we consider the best one

        idx = np.argmin(trials.losses())

        print(idx)



        print(trials.trials[idx])



        # these should be the training parameters to use to achieve the best score in best trial

        params = trials.trials[idx]["result"]["params"]

        max_n = None



        print(params)

        return params



    # run a search for binary classification

    def classify_binary(self, X_cols, df_train, df_test, y_param):

        self.y = y_param



        self.X = df_train[X_cols]

        self.X_test = df_test[X_cols]



        # use 2 classes as this is a binary classification

        params = self.optimize_lgbm(2)

        print(params)



        clf = lgbm.LGBMClassifier(**params)



        fit_params = self.create_fit_params(params)



        search_results = stratified_test_prediction_avg_vote(clf, self.X, self.X_test, self.y,

                                                             n_folds=self.n_folds, n_classes=2, fit_params=fit_params)

        search_results.all_accuracies = self.all_accuracies

        search_results.all_losses = self.all_losses

        search_results.all_params = self.all_params

        return search_results
import numpy as np

import pandas as pd

from hyperopt import hp, tpe, Trials

from hyperopt.fmin import fmin

import lightgbm as lgbm

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

import hyperopt

from sklearn.metrics import accuracy_score, log_loss



class RFOptimizer:

    # how many CV folds to do on the data

    n_folds = 5

    # max number of rows to use for training (from X and y). to reduce time and compare options faster

    max_n = None

    # max number of trials hyperopt runs

    n_trials = 200

    #verbosity 0 in RF is quite, 1 = print epoch, 2 = print within epoch

    #https://stackoverflow.com/questions/31952991/what-does-the-verbosity-parameter-of-a-random-forest-mean-sklearn

    verbosity = 0

    #if true, print summary accuracy/loss after each round

    print_summary = False



    all_accuracies = []

    all_losses = []

    all_params = []



    def objective_sklearn(self, params):

        int_types = ["n_estimators", "min_samples_leaf", "min_samples_split", "max_features"]

        n_classes = params.pop("num_class")

        params = convert_int_params(int_types, params)

        score, logloss = fit_cv(self.X, self.y, params, self.fit_params, n_classes, RandomForestClassifier, 

                                self.max_n, self.n_folds, self.print_summary, verbosity=self.verbosity)

        self.all_params.append(params)

        self.all_accuracies.append(score)

        self.all_losses.append(logloss)



        loss = logloss

        result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}

        return result



    def optimize_rf(self, n_classes):

        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

        space = {

            'criterion': hp.choice('criterion', ["gini", "entropy"]),

            # 'scale': hp.choice('scale', [0, 1]),

            # 'normalize': hp.choice('normalize', [0, 1]),

            'bootstrap': hp.choice('bootstrap', [True, False]),

            # nested choice: https://medium.com/vooban-ai/hyperopt-tutorial-for-optimizing-neural-networks-hyperparameters-e3102814b919

            'max_depth': hp.choice('max_depth', [None, hp.quniform('max_depth_num', 10, 100, 10)]),

            'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None, hp.quniform('max_features_num', 1, 5, 1)]),

            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 4, 1),

            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 4),

            'class_weight': hp.choice('class_weight', ["balanced", None]),

            'n_estimators': hp.quniform('n_estimators', 200, 2000, 200),

            'n_jobs': -1,

            'num_class': n_classes,

            'verbose': self.verbosity

        }

        # save and reload trials for hyperopt state: https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/blob/master/hyperopt_optimize.py



        trials = Trials()

        best = fmin(fn=self.objective_sklearn,

                    space=space,

                    algo=tpe.suggest,

                    max_evals=self.n_trials,

                   trials=trials)



        idx = np.argmin(trials.losses())

        print(idx)



        print(trials.trials[idx])



        params = trials.trials[idx]["result"]["params"]

        print(params)

        return params



    # run a search for binary classification

    def classify_binary(self, X_cols, df_train, df_test, y_param):

        self.y = y_param



        self.X = df_train[X_cols]

        self.X_test = df_test[X_cols]



        self.fit_params = {'use_eval_set': False}



        # use 2 classes as this is a binary classification

        params = self.optimize_rf(2)

        print(params)



        clf = RandomForestClassifier(**params)



        search_results = stratified_test_prediction_avg_vote(clf, self.X, self.X_test, self.y,

                                                             n_folds=self.n_folds, n_classes=2, fit_params=self.fit_params)

        search_results.all_accuracies = self.all_accuracies

        search_results.all_losses = self.all_losses

        search_results.all_params = self.all_params

        return search_results
import numpy as np

import pandas as pd

from hyperopt import hp, tpe, Trials

from hyperopt.fmin import fmin

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

import hyperopt

from sklearn.metrics import accuracy_score, log_loss



class XGBOptimizer:

    # how many CV folds to do on the data

    n_folds = 5

    # max number of rows to use for training (from X and y). to reduce time and compare options faster

    max_n = None

    # max number of trials hyperopt runs

    n_trials = 200

    #verbosity 0 in RF is quite, 1 = print epoch, 2 = print within epoch

    #https://stackoverflow.com/questions/31952991/what-does-the-verbosity-parameter-of-a-random-forest-mean-sklearn

    verbosity = 0

    #if true, print summary accuracy/loss after each round

    print_summary = False



    all_accuracies = []

    all_losses = []

    all_params = []



    def objective_sklearn(self, params):

        int_params = ['max_depth']

        params = convert_int_params(int_params, params)

        float_params = ['gamma', 'colsample_bytree']

        params = convert_float_params(float_params, params)

        n_classes = params.pop("num_class")



        score, logloss = fit_cv(self.X, self.y, params, self.fit_params, n_classes, xgb.XGBClassifier, 

                                self.max_n, self.n_folds, self.print_summary, verbosity=self.verbosity)

        self.all_params.append(params)

        self.all_accuracies.append(score)

        self.all_losses.append(logloss)



        loss = logloss

        result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}

        return result



    def optimize_xgb(self, n_classes):

        #https://indico.cern.ch/event/617754/contributions/2590694/attachments/1459648/2254154/catboost_for_CMS.pdf

        space = {

            'max_depth': hp.quniform('max_depth', 2, 10, 1),

            #removed gblinear since it does not support early stopping and it was getting tricky

            'booster': hp.choice('booster', ['gbtree', 'dart']),

            #'booster': hp.choice('booster', ['gbtree', 'gblinear', 'dart']),

            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),

            #nthread defaults to maximum so not setting it

            'subsample': hp.uniform('subsample', 0.75, 1.0),

            'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),

            'colsample_bylevel': hp.uniform('colsample_bylevel', 0.3, 1.0),

            #'gamma': hp.uniform('gamma', 0.0, 0.5),

            'min_child_weight': hp.loguniform('min_child_weight', -16, 5),

            'alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),

            'lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),

            'gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)]),

            'num_class': n_classes,

            'verbose': self.verbosity

            #'n_estimators': 1000   #n_estimators = n_trees -> get error this only valid for gbtree

            #https://github.com/dmlc/xgboost/issues/3789

        }



        trials = Trials()

        best = fmin(fn=self.objective_sklearn,

                    space=space,

                    algo=tpe.suggest,

                    max_evals=self.n_trials,

                   trials=trials)



        idx = np.argmin(trials.losses())

        print(idx)



        print(trials.trials[idx])



        params = trials.trials[idx]["result"]["params"]

        print(params)

        return params 



    # run a search for binary classification

    def classify_binary(self, X_cols, df_train, df_test, y_param):

        self.y = y_param



        self.X = df_train[X_cols]

        self.X_test = df_test[X_cols]



        self.fit_params = {'use_eval_set': False}



        # use 2 classes as this is a binary classification

        params = self.optimize_xgb(2)

        print(params)



        clf = xgb.XGBClassifier(**params)



        search_results = stratified_test_prediction_avg_vote(clf, self.X, self.X_test, self.y,

                                                             n_folds=self.n_folds, n_classes=2, fit_params=self.fit_params)

        search_results.all_accuracies = self.all_accuracies

        search_results.all_losses = self.all_losses

        search_results.all_params = self.all_params

        return search_results
xgb_opt = XGBOptimizer()

xgb_opt.n_trials = 200

xgb_results = xgb_opt.classify_binary(X_cols, df_train, df_test, y)

create_misclassified_dataframe(xgb_results, y)

lgopt = LGBMOptimizer()

lgopt.n_trials = 200

lgbm_results = lgopt.classify_binary(X_cols, df_train, df_test, y)

create_misclassified_dataframe(lgbm_results, y)

copt = CatboostOptimizer()

copt.n_trials = 200

cb_results = copt.classify_binary(X_cols, df_train, df_test, y)

create_misclassified_dataframe(cb_results, y)

rf_opt = RFOptimizer()

rf_opt.n_trials = 200

rf_results = rf_opt.classify_binary(X_cols, df_train, df_test, y)

create_misclassified_dataframe(rf_results, y)
df_losses = pd.DataFrame()

df_losses["lgbm_loss"] = lgbm_results.all_losses

df_losses["lgbm_accuracy"] = lgbm_results.all_accuracies

df_losses["cb_loss"] = cb_results.all_losses

df_losses["cb_accuracy"] = cb_results.all_accuracies

df_losses["rf_loss"] = rf_results.all_losses

df_losses["rf_accuracy"] = rf_results.all_accuracies

df_losses["xgb_loss"] = xgb_results.all_losses

df_losses["xgb_accuracy"] = xgb_results.all_accuracies

df_losses.plot(figsize=(14,8))
df_losses.drop("rf_loss", axis=1).plot(figsize=(14,8))
df_losses.sort_values(by="lgbm_accuracy", ascending=False)[["lgbm_loss", "lgbm_accuracy"]].head(10)
df_losses.sort_values(by="cb_accuracy", ascending=False)[["cb_loss", "cb_accuracy"]].head(10)
df_losses.sort_values(by="rf_accuracy", ascending=False)[["rf_loss", "rf_accuracy"]].head(10)
df_losses.sort_values(by="xgb_accuracy", ascending=False)[["xgb_loss", "xgb_accuracy"]].head(10)
df_losses.sort_values(by="lgbm_loss", ascending=True)[["lgbm_loss", "lgbm_accuracy"]].head(10)
df_losses.sort_values(by="cb_loss", ascending=True)[["cb_loss", "cb_accuracy"]].head(10)
df_losses.sort_values(by="rf_loss", ascending=True)[["rf_loss", "rf_accuracy"]].head(10)
df_losses.sort_values(by="xgb_loss", ascending=True)[["xgb_loss", "xgb_accuracy"]].head(10)
ss = pd.read_csv('../input/gender_submission.csv')

# predicting only true values, so take column 1 (0 is false column)

np_preds = np.array(cb_results.predictions)[: ,1]

ss["Survived"] = np.where(np_preds > 0.5, 1, 0)

ss.to_csv('catboost.csv', index=False)

ss.head(10)
ss = pd.read_csv('../input/gender_submission.csv')

# predicting only true values, so take column 1 (0 is false column)

np_preds = np.array(lgbm_results.predictions)[: ,1]

ss["Survived"] = np.where(np_preds > 0.5, 1, 0)

ss.to_csv('lgbm.csv', index=False)

ss.head(10)
ss = pd.read_csv('../input/gender_submission.csv')

# predicting only true values, so take column 1 (0 is false column)

np_preds = np.array(rf_results.predictions)[: ,1]

ss["Survived"] = np.where(np_preds > 0.5, 1, 0)

ss.to_csv('rf.csv', index=False)

ss.head(10)
ss = pd.read_csv('../input/gender_submission.csv')

# predicting only true values, so take column 1 (0 is false column)

np_preds = np.array(xgb_results.predictions)[: ,1]

ss["Survived"] = np.where(np_preds > 0.5, 1, 0)

ss.to_csv('xgb.csv', index=False)

ss.head(10)
print(len(lgbm_results.misclassified_indices))

print(len(cb_results.misclassified_indices))

print(len(rf_results.misclassified_indices))

print(len(xgb_results.misclassified_indices))
df_top_misses_lgbm = lgbm_results.df_misses.sort_values(by="Abs_Diff", ascending=False)

df_top_misses_lgbm.head()
df_top_misses_cb = cb_results.df_misses.sort_values(by="Abs_Diff", ascending=False)

df_top_misses_cb.head()
df_top_misses_rf = rf_results.df_misses.sort_values(by="Abs_Diff", ascending=False)

df_top_misses_rf.head()
df_top_misses_xgb = xgb_results.df_misses.sort_values(by="Abs_Diff", ascending=False)

df_top_misses_xgb.head()
#capture the probabilities of True (1) classification for each classifier, to use as inputs for ensembling:

ensemble_input_df = pd.DataFrame()

ensemble_input_df["lgbm"] = lgbm_results.predictions[:,1]

ensemble_input_df["xgb"] = xgb_results.predictions[:,1]

ensemble_input_df["catboost"] = cb_results.predictions[:,1]

ensemble_input_df["randomforest"] = rf_results.predictions[:,1]

ensemble_input_df.head()
ensemble_input_df["avg"] = (ensemble_input_df["lgbm"]+ensemble_input_df["xgb"]+ensemble_input_df["catboost"]+ensemble_input_df["randomforest"])/4

ensemble_input_df.head()
ensemble_input_df["lgbm_01"] = np.where(ensemble_input_df["lgbm"] > 0.5, 1, 0)

ensemble_input_df["xgb_01"] = np.where(ensemble_input_df["xgb"] > 0.5, 1, 0)

ensemble_input_df["cat_01"] = np.where(ensemble_input_df["catboost"] > 0.5, 1, 0)

ensemble_input_df["rf_01"] = np.where(ensemble_input_df["randomforest"] > 0.5, 1, 0)

ensemble_input_df.head()
ensemble_input_df.shape
from scipy.stats import mode



#TODO: majority, avg, stacked ensemble methods



data = [ensemble_input_df["lgbm_01"].values, ensemble_input_df["xgb_01"], ensemble_input_df["cat_01"], ensemble_input_df["rf_01"]]

majority = mode(data, axis=0)

#the "majority" variable is actually now a list of two lists. majority[0][0] is the mode (1 or 0, the majority class), and majority[0][1] is how many times the "mode" appears

#majority
len(majority[0][0])
ensemble_input_df["majority"] = majority[0][0]

ensemble_input_df.head()
ss = pd.read_csv('../input/gender_submission.csv')

ss["Survived"] = np.where(ensemble_input_df["avg"] > 0.5, 1, 0)

#ss["Survived"] = ensemble_input_df["avg"]

ss.to_csv('avg.csv', index=False)

ss.head(10)
ss = pd.read_csv('../input/gender_submission.csv')

ss["Survived"] = ensemble_input_df["majority"]

ss.to_csv('majority.csv', index=False)

ss.head(10)
oof_df = pd.DataFrame()

oof_df["lgbm"] = lgbm_results.oof_predictions

oof_df["xgb"] = xgb_results.oof_predictions

oof_df["cat"] = cb_results.oof_predictions

oof_df["rf"] = rf_results.oof_predictions

oof_df["target"] = y

oof_df.head()
import numpy as np

import pandas as pd

from hyperopt import hp, tpe, Trials

from hyperopt.fmin import fmin

import lightgbm as lgbm

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

import hyperopt

from sklearn.metrics import accuracy_score, log_loss

from sklearn.linear_model import LogisticRegression



class LogRegOptimizer:

    # how many CV folds to do on the data

    n_folds = 5

    # max number of rows to use for training (from X and y). to reduce time and compare options faster

    max_n = None

    # max number of trials hyperopt runs

    n_trials = 200

    # ?

    verbosity = 0

    #if true, print summary accuracy/loss after each round

    print_summary = False



    all_accuracies = []

    all_losses = []

    all_params = []



    def objective_sklearn(self, params):

        #print(params)

        params.update(params["solver_params"]) #pop nested dict to top level

        del params["solver_params"] #delete the original nested dict after pop (could pop() above too..)

        if params["penalty"] == "none":

            del params["C"]

            del params["l1_ratio"]

        elif params["penalty"] != "elasticnet":

            del params["l1_ratio"]

        if params["solver"] == "liblinear":

            params["n_jobs"] = 1

        n_classes = params.pop("num_class")

#        params = convert_int_params(int_types, params)

        score, logloss = fit_cv(self.X, self.y, params, self.fit_params, n_classes, LogisticRegression, 

                                self.max_n, self.n_folds, self.print_summary, verbosity=self.verbosity)

        self.all_params.append(params)

        self.all_accuracies.append(score)

        self.all_losses.append(logloss)



        loss = logloss

        result = {"loss": loss, "score": score, "params": params, 'status': hyperopt.STATUS_OK}

        return result



    def optimize_logreg(self, n_classes):

        # https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

        space = {

            'solver_params': hp.choice('solver_params', [

                {'solver': 'newton-cg', 

                 'penalty': hp.choice('penalty-ncg', ["l2", 'none'])}, #also multiclass loss supported

                 {'solver': 'lbfgs', 

                 'penalty': hp.choice('penalty-lbfgs', ["l2", 'none'])},

                 {'solver': 'liblinear',

                 'penalty': hp.choice('penalty-liblin', ["l1", "l2"])},

                 {'solver': 'sag',

                 'penalty': hp.choice('penalty-sag', ["l2", 'none'])},

                 {'solver': 'saga',

                 'penalty': hp.choice('penalty-saga', ["elasticnet", "l1", "l2", 'none'])},

            ]),

            'C': hp.uniform('C', 1e-5,10),

            'tol': hp.uniform('tol', 1e-5, 10),

            'fit_intercept': hp.choice("fit_intercept", [True, False]),

            'class_weight': hp.choice("class_weight", ["balanced", None]),

            #multi-class jos ei bianry

            'l1_ratio': hp.uniform('l1_ratio', 0.00001, 0.99999), #vain jos elasticnet penalty

            'n_jobs': -1,

            'num_class': n_classes,

            'verbose': self.verbosity

        }

        # save and reload trials for hyperopt state: https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100/blob/master/hyperopt_optimize.py



        trials = Trials()

        best = fmin(fn=self.objective_sklearn,

                    space=space,

                    algo=tpe.suggest,

                    max_evals=self.n_trials,

                   trials=trials)



        idx = np.argmin(trials.losses())

        print(idx)



        print(trials.trials[idx])



        params = trials.trials[idx]["result"]["params"]

        print(params)

        return params



    # run a search for binary classification

    def classify_binary(self, X_cols, df_train, df_test, y_param):

        self.y = y_param



        self.X = df_train[X_cols]

        self.X_test = df_test[X_cols]



        self.fit_params = {'use_eval_set': False}



        # use 2 classes as this is a binary classification

        params = self.optimize_logreg(2)

        print(params)



        clf = LogisticRegression(**params)



        search_results = stratified_test_prediction_avg_vote(clf, self.X, self.X_test, self.y,

                                                             n_folds=self.n_folds, n_classes=2, fit_params=self.fit_params)

        search_results.all_accuracies = self.all_accuracies

        search_results.all_losses = self.all_losses

        search_results.all_params = self.all_params

        return search_results

df_stack_train = oof_df.drop("target", axis=1)

df_stack_train.head()
X_stack_cols = df_stack_train.columns

df_stacked_X = ensemble_input_df[["lgbm", "xgb", "catboost", "randomforest"]]

df_stacked_X.columns = X_stack_cols

logreg_opt = LogRegOptimizer()

lr_results = logreg_opt.classify_binary(X_stack_cols, df_stack_train, df_stacked_X, y)

create_misclassified_dataframe(lr_results, y)
from sklearn.linear_model import LogisticRegression



ss = pd.read_csv('../input/gender_submission.csv')

# predicting only true values, so take column 1 (0 is false column)

np_preds = np.array(lr_results.predictions)[: ,1]

ss["Survived"] = np.where(np_preds > 0.5, 1, 0)

ss.to_csv('stacked.csv', index=False)

ss.head(10)
#https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/
all_misses = pd.concat([df_top_misses_lgbm,df_top_misses_cb,df_top_misses_xgb,df_top_misses_rf], axis=0)

all_misses.head()
all_misses_count = all_misses.index.value_counts()

all_misses_count.head()
all_misses_count.value_counts()
all_misses.head()
#group by index, count sum of abs
all_misses = all_misses.sort_index()



miss_counts = all_misses.groupby(all_misses.index)['Abs_Diff'].sum().sort_values(ascending=False)

miss_counts.head(10)
df_train[df_train["Ticket"] == "LINE"]
top10 = miss_counts.head(10).index

df_train.iloc[top10]

miss_counts.tail(10)
top10 = miss_counts.tail(10).index

df_train.iloc[top10]