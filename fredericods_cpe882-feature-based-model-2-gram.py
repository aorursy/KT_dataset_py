import pandas as pd

import numpy as np

import seaborn as sns

from sklearn import model_selection

from sklearn.metrics import roc_auc_score

from sklearn.feature_extraction.text import TfidfVectorizer

import xgboost as xgb

from sklearn.linear_model import LogisticRegression

import pickle

from joblib import dump, load

import time



sns.set()



KAGGLE_PATH = '/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/'



train = pd.read_csv(KAGGLE_PATH + 'train.csv')

test = pd.read_csv(KAGGLE_PATH + 'test_private_expanded.csv')

test['target'] = test.toxicity



# Make sure all comment_text values are strings

train['comment_text'] = train['comment_text'].astype(str)

test['comment_text'] = test['comment_text'].astype(str)



# List all identities

identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'

]



# Convert taget and identity columns to booleans

for col in ['target'] + identity_columns:

    train[col] = np.where(train[col] >= 0.5, True, False)

    test[col] = np.where(test[col] >= 0.5, True, False)

    

# Creating folds (Stratified Split, K = 5)

train["kfold"] = -1

train = train.sample(frac=1).reset_index(drop=True) # shuffle dataframe

y = train.target.values

kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True)

for fold_, (train_idx, test_idx) in enumerate(kf.split(X=train, y=y)):

    train.loc[test_idx, "kfold"] = fold_
class JigsawEvaluator:



    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25):

        self.y = (y_true >= 0.5).astype(int)

        self.y_i = (y_identity >= 0.5).astype(int)

        self.n_subgroups = self.y_i.shape[1]

        self.power = power

        self.overall_model_weight = overall_model_weight



    @staticmethod

    def _compute_auc(y_true, y_pred):

        try:

            return roc_auc_score(y_true, y_pred)

        except ValueError:

            return np.nan



    def _compute_subgroup_auc(self, i, y_pred):

        mask = self.y_i[:, i] == 1

        return self._compute_auc(self.y[mask], y_pred[mask])



    def _compute_bpsn_auc(self, i, y_pred):

        mask = self.y_i[:, i] + self.y == 1

        return self._compute_auc(self.y[mask], y_pred[mask])



    def _compute_bnsp_auc(self, i, y_pred):

        mask = self.y_i[:, i] + self.y != 1

        return self._compute_auc(self.y[mask], y_pred[mask])



    def compute_bias_metrics_for_model(self, y_pred):

        records = np.zeros((3, self.n_subgroups))

        for i in range(self.n_subgroups):

            records[0, i] = self._compute_subgroup_auc(i, y_pred)

            records[1, i] = self._compute_bpsn_auc(i, y_pred)

            records[2, i] = self._compute_bnsp_auc(i, y_pred)

        return records



    def _calculate_overall_auc(self, y_pred):

        return roc_auc_score(self.y, y_pred)



    def _power_mean(self, array):

        total = sum(np.power(array, self.power))

        return np.power(total / len(array), 1 / self.power)



    def get_final_metric(self, y_pred):

        bias_metrics = self.compute_bias_metrics_for_model(y_pred)

        bias_score = np.average([

            self._power_mean(bias_metrics[0]),

            self._power_mean(bias_metrics[1]),

            self._power_mean(bias_metrics[2])

        ])

        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)

        bias_score = (1 - self.overall_model_weight) * bias_score



        return overall_score + bias_score
def train_fold_i(fold_i, tfv):

    # Transform tfv object to train, validation and test sets

    xtrain_tfv = tfv.transform(train[train.kfold!=fold_i].comment_text.values)

    xvalid_tfv = tfv.transform(train[train.kfold==fold_i].comment_text.values)

    xtest_tfv = tfv.transform(test.comment_text.values)



    # Fit model

    model = LogisticRegression(random_state=0, n_jobs=-1, solver='saga')

    model.fit(xtrain_tfv,train[train.kfold!=fold_i].target.values)



    # Making predictions

    preds_train = model.predict_proba(xtrain_tfv)[:,1]

    preds_valid = model.predict_proba(xvalid_tfv)[:,1]

    preds_test = model.predict_proba(xtest_tfv)[:,1]



    # Calculate bias auc

    bias_auc_train = JigsawEvaluator(train[train.kfold!=fold_i].target.values, train[train.kfold!=fold_i][identity_columns].values).get_final_metric(preds_train)

    bias_auc_valid = JigsawEvaluator(train[train.kfold==fold_i].target.values, train[train.kfold==fold_i][identity_columns].values).get_final_metric(preds_valid)

    bias_auc_test = JigsawEvaluator(test.target.values, test[identity_columns].values).get_final_metric(preds_test)



    auc_train = roc_auc_score(train[train.kfold!=fold_i].target.values, preds_train)

    auc_valid = roc_auc_score(train[train.kfold==fold_i].target.values, preds_valid)

    auc_test = roc_auc_score(test.target.values, preds_test)



    # Print results

    print('Bias AUC | Train: {}'.format(bias_auc_train))

    print('Bias AUC | Valid: {}'.format(bias_auc_valid))

    print('Bias AUC |  Test: {}'.format(bias_auc_test))



    return {

        'auc_train': auc_train,

        'auc_valid': auc_valid,

        'auc_test': auc_test,

        'bias_auc_train': bias_auc_train,

        'bias_auc_valid': bias_auc_valid,

        'bias_auc_test': bias_auc_test,

        'preds_train': preds_train,

        'preds_valid': preds_valid,

        'preds_test': preds_test,

        'model': model

    }



def train_all_folds(saved_filename, n, max_feat):

    

    print('\n------------------ START ------------------\n')

    

    start = time.time()

    

    print('Start: fitting tf-idf')

    #tfv = TfidfVectorizer(min_df=5, token_pattern=r'\w{1,}', ngram_range=(1, n), sublinear_tf=1, stop_words = 'english')

    tfv = TfidfVectorizer(max_features=max_feat, token_pattern=r'\w{1,}', ngram_range=(1, n), sublinear_tf=1, stop_words = 'english')

    tfv.fit(train.comment_text.values)

    print('End: fitting tf-idf')

    

    dict_results = {}

    for i in [0,1,2,3,4]:

        print('\nStart: fitting fold {}'.format(int(i)))

        results_i = train_fold_i(i, tfv)

        dict_results['Fold_{}'.format(i)] = results_i

        print('End: fitting fold {}'.format(int(i)))

    

    dict_results['tfv'] = tfv

    

    end = time.time()

    

    dict_results['time'] = end - start

    

    dump(dict_results, saved_filename + ".joblib")

    

    print('\n------------------ END ------------------\n')

        

    return dict_results
train_all_folds('model_2_gram_500k', n=2, max_feat=500000)
train_all_folds('model_2_gram_250k', n=2, max_feat=250000)