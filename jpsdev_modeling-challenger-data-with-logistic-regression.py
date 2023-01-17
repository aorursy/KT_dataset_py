import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import tensorflow.compat.v2.feature_column as fc

import statsmodels.api as sm

import scipy.stats as stats

from scipy.stats import ttest_ind

from sklearn import linear_model

from IPython.display import clear_output

from six.moves import urllib



# to import from Kaggle

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load training and testing data 

df_train = pd.read_csv("/kaggle/input/league-of-legends-challenger-ranked-games2020/Challenger_Ranked_Games.csv")

df_test = pd.read_csv("/kaggle/input/league-of-legends-challenger-ranked-games2020/Master_Ranked_Games.csv")

df_train.head()
# clean training data

df_train_clean = df_train.filter(like="blue")

gameDuration = df_train.pop("gameDuraton")

df_train_clean.insert(0, "gameDuration", gameDuration, True)

y_train = df_train_clean.pop("blueWins")

df_train_clean.head()
# clean testing data

df_test_clean = df_train.filter(like="blue")

df_test_clean.drop(["blueFirstBlood", "blueTotalHeal"], axis = 1)

y_test = df_test_clean.pop("blueWins")

df_test_clean.head()
# Separate winning and losing stats

win_stats = df_train_clean.loc[df_train["blueWins"] == 1]

loss_stats = df_train_clean.loc[df_train["blueWins"] == 0]
# summary stats of team that won

win_stats.describe()
# summary stats of team that lost

loss_stats.describe()
def draw_histograms(df, variables, n_rows, n_cols):

    fig=plt.figure(figsize=(15,15))

    for i, var_name in enumerate(variables):

        ax=fig.add_subplot(n_rows,n_cols,i+1)

        df[var_name].hist(bins=20,ax=ax)

        ax.set_title(var_name+" Distribution")

        plt.tight_layout()

    

    plt.show()
numerical_col_names = [c for c in df_test_clean.columns if c.lower()[4:9] != "first"]

# numerical_col_names = [c for c in numerical_col_names if c not in ["blueDragonKills", "blueBaronKills", "blueTowerKills", "blueInhibitorKills"]]

numerical_stats = df_test_clean[numerical_col_names]

histogram_all = df_test_clean[numerical_col_names]

histogram_data_win = win_stats[numerical_col_names]

histogram_data_lose = loss_stats[numerical_col_names]

histogram_all.head()
from matplotlib import pyplot

def compare_histograms(df1, df2, variables, n_rows, n_cols):

        fig=plt.figure(figsize=(15,15))

        for i, var_name in enumerate(variables):

            ax=fig.add_subplot(n_rows,n_cols,i+1)

            df1[var_name].hist(bins=20, ax=ax, label="Won") # for histogram

            df2[var_name].hist(bins=20, ax=ax, label="Lost") # for histogram

            ax.set_title(var_name+" Distribution")

            pyplot.legend(loc="upper right")

            plt.tight_layout()

        

        plt.show()
compare_histograms(histogram_data_win, histogram_data_lose, histogram_data_win, 9, 2)
draw_histograms(histogram_all, histogram_all.columns, 9, 2)
import pylab 

def draw_qqplots(df, variables, n_rows, n_cols):

    fig=plt.figure(figsize=(15,15))

    for i, var_name in enumerate(variables):

        

        ax=fig.add_subplot(n_rows,n_cols,i+1)

        stats.probplot(df[var_name], dist="norm", plot=pylab)

        # df[var_name].hist(bins=20,ax=ax)

        ax.set_title(var_name+" qq plot")

        plt.tight_layout()

    

    plt.show()
draw_qqplots(histogram_all, histogram_all.columns, 9, 2)
piechart_col_names = [c for c in df_test_clean.columns if c.lower()[4:9] == "first"]

# piechart_col_names = piechart_col_names + ["blueDragonKills", "blueBaronKills", "blueTowerKills", "blueInhibitorKills"]

piechart_all = df_train_clean[piechart_col_names]

piechart_all.head()
piechart_win = win_stats[piechart_col_names]

piechart_loss = loss_stats[piechart_col_names]
def draw_piecharts(df, variables, n_rows, n_cols):

    fig=plt.figure(figsize=(15,15))

    for i, var_name in enumerate(variables):

        ax=fig.add_subplot(n_rows,n_cols,i+1)

        percentages = list(df[var_name].value_counts(normalize=True, sort=False) * 100)

        labels = sorted(df[var_name].unique())

        explode = ([0.02] * len(percentages))

        ax.pie(percentages,shadow=True, explode=explode, labels=labels, autopct='%1.1f%%')

        ax.set_title(var_name)



        ax.legend(

          title="# of occurances",

          loc="center left",

          bbox_to_anchor=(1, 0, 0.5, 1),

          labels=['%s, %1.1f %%' % (l, s) for l, s in zip(labels, percentages)]

          )

        plt.tight_layout()

    

    plt.show()
draw_piecharts(piechart_all, piechart_all.columns, 3, 3)
draw_piecharts(piechart_win, piechart_win.columns, 3 ,3)
draw_piecharts(piechart_loss, piechart_loss.columns, 3 ,3)
histogram_all.skew() # from scipy.stats library
from sklearn import preprocessing

def normalize_data(df):

  df = df+1

  scaler = preprocessing.PowerTransformer(method="box-cox", standardize=True).fit(df)

  norm_df = pd.DataFrame(scaler.transform(df),columns=df.columns)

  return norm_df
norm = normalize_data(histogram_all)

norm.head()
norm.skew()
draw_histograms(norm, norm.columns, 6, 3)
draw_qqplots(norm, norm.columns, 6, 3)
from statsmodels.stats.outliers_influence import variance_inflation_factor



def calc_vif(X):



    # Calculating VIF

    vif = pd.DataFrame()

    vif["variables"] = X.columns

    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]



    return(vif)
vif_init = calc_vif(norm)

vif_init
norm2 = norm.drop(["blueTotalLevel", "blueAvgLevel", "blueTotalGold", "blueKillingSpree", "blueAssist"], axis=1)

vif_norm2 = calc_vif(norm2)

vif_norm2
import seaborn as sn

corr_matrix_norm = norm2.corr()

mask = np.triu(corr_matrix_norm)

f, ax = plt.subplots(figsize=(15,15))

ax = sn.heatmap(corr_matrix_norm, annot=True, mask=mask, cmap="coolwarm", square=True)

plt.show()
norm_X_train = piechart_all.join(norm2) # join categorical with numerical variables

X = norm_X_train.to_numpy()

Y = y_train

model = sm.Logit(Y,X).fit()

predictions = model.predict(X)

print_model = model.summary(xname=list(norm_X_train.columns))

print(print_model)
logit_insig = ["blueFirstTower", "blueFirstBaron", "blueWardkills", "blueChampionDamageDealt", "blueTotalMinionKills", "blueTotalHeal", "blueObjectDamageDealt"]

X = norm_X_train.drop(list(logit_insig), axis = 1)

Y = y_train

model = sm.Logit(Y,X).fit()

predictions = model.predict(X)

print_model = model.summary()

print(print_model)
# odd ratios of each predictor variable, with raw data



# getting the names of significant variables

sig_vars = []

for key, val in model.params.iteritems():

    sig_vars.append(key)



raw_num_model_df = piechart_all.join(histogram_all) # join categorical with numerical variables

X_raw = raw_num_model_df[sig_vars]

Y = y_train

raw_num_model = sm.Logit(Y,X_raw).fit()

np.exp(raw_num_model.params)
blue_sig_stats = norm_X_train



blue_sig_stats["blueWins"] = y_train

blue_sig_stats.head()
blue_wins = blue_sig_stats.where(df_train["blueWins"] == 1)

blue_wins = blue_wins.dropna()

blue_wins.head()
blue_loses = blue_sig_stats.where(df_train["blueWins"] == 0)

blue_loses = blue_loses.dropna()

blue_loses.head()
SIG_VARS = sig_vars

SIG_BIN_CAT_VARS = [v for v in list(piechart_all.columns) if v in SIG_VARS]

SIG_NUM_VARS = [v for v in list(histogram_all.columns) if v in SIG_VARS]

ALL_SIG_VARS = SIG_BIN_CAT_VARS + SIG_NUM_VARS

print(SIG_BIN_CAT_VARS)

print(SIG_NUM_VARS)
# INSIG_VARS = logit_insig

# SIG_BIN_CAT_VARS = [v for v in list(piechart_all.columns) if v not in INSIG_VARS]

# SIG_NUM_VARS = [v for v in list(histogram_all.columns) if v not in INSIG_VARS]

# ALL_SIG_VARS = SIG_BIN_CAT_VARS + SIG_NUM_VARS
def format_test_data(df):

    df = df.head(26904)

    bin_data = df[SIG_BIN_CAT_VARS]

    # normalize numerical variables

    num_data = df[SIG_NUM_VARS]

    norm_num_data = normalize_data(num_data)

    win_loss_col = df["blueWins"]

    test_data = pd.concat([bin_data, norm_num_data, win_loss_col], axis=1)

    return test_data
def format_train_data(df):

    df = df.head(26904)

    bin_data = df[SIG_BIN_CAT_VARS]

    # normalize numerical variables

    num_data = df[SIG_NUM_VARS]

    norm_num_data = normalize_data(num_data)

    test_data = pd.concat([bin_data, norm_num_data], axis=1)

    return test_data
clean_train_data = format_train_data(df_train)

clean_test_data = format_test_data(df_test)

clean_test_data.head()
feature_columns = []

y_test_clean = clean_test_data.pop("blueWins")



for feature_name in SIG_BIN_CAT_VARS:

  vocabulary = clean_train_data[feature_name].unique()

  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))



for feature_name in SIG_NUM_VARS:

  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=100):

  def input_function():  # inner function, this will be returned

    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label

    if shuffle:

      ds = ds.shuffle(1000)  # randomize order of data

    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs

    return (ds)  # return a batch of the dataset

  return input_function  # return a function object for use



train_input_fn = make_input_fn(clean_train_data, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model

eval_input_fn = make_input_fn(clean_test_data, y_test_clean, num_epochs=1, shuffle=False)

ds = make_input_fn(clean_train_data, y_train, batch_size=10)()

for feature_batch, label_batch in ds.take(1):

  print('Some feature keys:', list(feature_batch.keys()))

  print()

  print('A batch of blueDragonKills:', feature_batch['blueDragonKills'].numpy())

  print()

  print('A batch of Labels:', label_batch.numpy())
linear_est = tf.estimator.LinearClassifier(feature_columns= feature_columns)

# We create a linear estimator by passing the feature columns we created earlier
linear_est.train(train_input_fn)  # train

result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data



clear_output()  # clears console output

print(result)  # the result variable is simply a dict of stats about our model
pred_dicts = list(linear_est.predict(eval_input_fn))

probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])



probs.plot(kind='hist', bins=20, title='predicted probabilities')
from sklearn.metrics import roc_curve

from matplotlib import pyplot as plt



fpr, tpr, _ = roc_curve(y_test_clean, probs)

plt.plot(fpr, tpr)

plt.title('ROC curve')

plt.xlabel('false positive rate')

plt.ylabel('true positive rate')

plt.xlim(0,)

plt.ylim(0,)