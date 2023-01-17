# Standard dependencies

import os

import numpy as np

import random as rn

import pandas as pd

import matplotlib.pyplot as plt

from keras.callbacks import Callback



# Scipy's implementation of Spearman's Rho 

from scipy.stats import spearmanr



# Set seed for reproducability

seed = 1234

rn.seed(seed)

np.random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)



# Paths for easy data access

BASE_PATH = "../input/google-quest-challenge/"

TRAIN_PATH = BASE_PATH + "train.csv"

TEST_PATH = BASE_PATH + "test.csv"

SUB_PATH = BASE_PATH + "sample_submission.csv"
# File sizes and specifications

print('\n# Files and file sizes')

for file in os.listdir(BASE_PATH):

    print('{}| {} MB'.format(file.ljust(30), 

                             str(round(os.path.getsize(BASE_PATH + file) / 1000000, 2))))
# All 30 targets

target_cols = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible', 'answer_relevance',

       'answer_satisfaction', 'answer_type_instructions',

       'answer_type_procedure', 'answer_type_reason_explanation',

       'answer_well_written']
# Read in training data

df = pd.read_csv(TRAIN_PATH)
print("Target variables:")

df[target_cols].head()
def spearmans_rho(y_true, y_pred, axis=0):

    """

        Calculates the Spearman's Rho Correlation between ground truth labels and predictions 

    """

    return spearmanr(y_true, y_pred, axis=axis)
def _get_ranks(arr: np.ndarray) -> np.ndarray:

    """

        Efficiently calculates the ranks of the data.

        Only sorts once to get the ranked data.

        

        :param arr: A 1D NumPy Array

        :return: A 1D NumPy Array containing the ranks of the data

    """

    temp = arr.argsort()

    ranks = np.empty_like(temp)

    ranks[temp] = np.arange(len(arr))

    return ranks



def spearmans_rho_custom(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:

    """

        Efficiently calculates the Spearman's Rho correlation using only NumPy

        

        :param y_true: The ground truth labels

        :param y_pred: The predicted labels

    """

    # Get ranked data

    true_rank = _get_ranks(y_true)

    pred_rank = _get_ranks(y_pred)

    

    return np.corrcoef(true_rank, pred_rank)[1][0] 
class SpearmanRhoCallback(Callback):

    def __init__(self, training_data, validation_data, patience, model_name):

        self.x = training_data[0]

        self.y = training_data[1]

        self.x_val = validation_data[0]

        self.y_val = validation_data[1]

        

        self.patience = patience

        self.value = -1

        self.bad_epochs = 0

        self.model_name = model_name



    def on_train_begin(self, logs={}):

        return



    def on_train_end(self, logs={}):

        return



    def on_epoch_begin(self, epoch, logs={}):

        return



    def on_epoch_end(self, epoch, logs={}):

        y_pred_val = self.model.predict(self.x_val)

        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])

        if rho_val >= self.value:

            self.value = rho_val

        else:

            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:

            print("Epoch %05d: early stopping Threshold" % epoch)

            self.model.stop_training = True

            #self.model.save_weights(self.model_name)

        print('\rval_spearman-rho: %s' % (str(round(rho_val, 4))), end=100*' '+'\n')

        return rho_val



    def on_batch_begin(self, batch, logs={}):

        return



    def on_batch_end(self, batch, logs={}):

        return
# Sample two times from distributions that are highly correlated

samp_size = 1000000

norm_num = np.arange(samp_size) + np.random.normal(0, 10, samp_size)

norm_num2 = np.arange(samp_size) + np.random.normal(0, 100000, samp_size)
%%timeit

spearmanr(norm_num, norm_num2)[0]
%%timeit

spearmans_rho_custom(norm_num, norm_num2)
corrs = []

# Make random predictions

for col in target_cols:

    naive_preds = np.random.rand(len(df))

    corr = spearmans_rho_custom(naive_preds, df[col])

    corrs.append(corr)

rand_baseline = np.mean(corrs)

print(f"Spearman's Rho Score for random uniform predictions: {round(rand_baseline, 6)}")
corrs = []

# Predict the mean and a small amount of noise to avoid division by zero

for col in target_cols:

    probs = df[col].value_counts().values / len(df)

    vals = list(df[col].value_counts().index)

    naive_preds = df[col].mean() + np.random.normal(0, 1e-15, len(df))

    corr = spearmans_rho_custom(naive_preds, df[col])

    corrs.append(corr)

mean_baseline = np.mean(corrs)

print(f"Spearman's Rho Score for predicting the mean with some noise: {round(mean_baseline, 6)}")
corrs = []

# Calculate probability of some prediction and sample according to those probabilities

for col in target_cols:

    probs = df[col].value_counts().values / len(df)

    vals = list(df[col].value_counts().index)

    naive_preds = np.random.choice(vals, len(df), p=probs)

    corr = spearmanr(naive_preds, df[col])[0]

    corrs.append(corr)

dist_baseline = np.mean(corrs)

print(f"Spearman's Rho Score for sampling from calculated distribution: {round(dist_baseline, 6)}")
# Read in sample submission file

sub_df = pd.read_csv(SUB_PATH)



# Make random predictions

for col in target_cols:

    naive_preds = np.random.rand(len(sub_df))

    sub_df[col] = naive_preds.round(6)

    

sub_df.to_csv('submission.csv', index=False)
print('Final predictions:')

sub_df.head(2)