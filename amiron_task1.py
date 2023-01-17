# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print(os.listdir("../input/test"))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from h2o.automl import H2OAutoML
import h2o
h2o.init()
df_train = pd.read_csv("../input/train/train.csv")
def regression_h2o(X_train, X_test, y_name, project_name):
#     max_runtime_secs = 7200 # 0.25 kappa
#    max_runtime_secs = 360 # 0.26 kappa
    max_runtime_secs = 180 # 0.24 kappa
    aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=1, project_name=project_name)
    aml.train(y=y_name, training_frame=X_train, leaderboard_frame=X_test)
    return aml

number_cols = df_train.describe().columns
data = df_train[number_cols]
X_train, X_test = train_test_split(data, test_size=0.1)

X_train = h2o.H2OFrame(X_train.dropna())
X_test = h2o.H2OFrame(X_test.dropna())
y_name = 'AdoptionSpeed'

model = regression_h2o(X_train, X_test, y_name, "automl_h2o")
print(model.leaderboard.head())
y_pred = model.predict(X_test).as_data_frame()['predict'].values
y_test = X_test.as_data_frame()[y_name].values
# mse 1.15   0.23752238131720194 on test
# mse 0.52   0.5998542141335967 on test

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def Cmatrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = Cmatrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)
y_pred_int = [int(x) for x in y_pred]
print("kappa:", quadratic_weighted_kappa(y_test, y_pred), "mse", mean_squared_error(y_test, y_pred))
print("kappa:", quadratic_weighted_kappa(y_test, y_pred), "mse", mean_squared_error(y_test, y_pred_int))
# kappa: 0.6105202419115187 mse 0.5302551125784477 float 360 sec
# kappa: 0.6105202419115187 mse 0.8639637236596426 int 360 sec
df_test = pd.read_csv("../input/test/test.csv")
df_test.fillna(0, inplace=True)
X_submission = df_test[list(set(number_cols) - {'AdoptionSpeed'})]
y_submission =  model.predict(h2o.H2OFrame(X_submission)).as_data_frame()['predict'].values
y_submission_int = [int(x) for x in y_submission]
df_sample_submisson = pd.read_csv("../input/test/sample_submission.csv")
df_sample_submisson['AdoptionSpeed'] = y_submission_int
df_sample_submisson.to_csv("submission.csv", index=False)