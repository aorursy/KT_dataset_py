import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Reading train

train_X = pd.read_csv(

    '../input/training_text', sep="\|\|", engine='python', header=None, skiprows=1, 

    names=["ID","Text"])

train_y = pd.DataFrame.from_csv("../input/training_variants")

print('Train Text')

print(train_X.head())

print("train classes")

print(train_y.head())

print("train classes probability")

train_y.Class.value_counts(normalize=True)

def multi_log_loss(y_true: np.array, y_pred: np.array):  # score function for CV

    # Handle all zeroes

    all_zeros = np.all(y_pred == 0, axis=1)

    y_pred[all_zeros] = 1/9

    # Normalise sum of row probabilities to one

    row_sums = np.sum(y_pred, axis=1)

    y_pred /= row_sums.reshape((-1, 1))

    # Calculate score

    n_rows = y_true.size

    y_true = y_true - 1  # classes start from 1 where columns start from zero

    score_sum = 0

    for i in range(y_true.size):

        score_sum -= np.log(y_pred[i, y_true[i]])

    score = score_sum / n_rows

    return score

        
# Gives every class 1/9 probability

predictions = np.repeat(1/9, train_y.size*9).reshape(train_y.size,9)

benchmark_blind = multi_log_loss(train_y.Class.values, predictions)

print("The score for equal probability per each class is:")

benchmark_blind
def class_probability_list(train_y_series):

    class_probability_series = train_y_series.value_counts(normalize=True)

    probability_list = []

    for i in range(1, 10):

        probability_list.append(class_probability_series.at[i])

    return probability_list
# Gives every class its precentange - Overfitting full train (Bayesian)

predictions = np.repeat(

    [class_probability_list(train_y.Class)], train_y.size, axis=0)

benchmark_probability_blind = multi_log_loss(train_y.Class.values, predictions)

print("The score for Bayesian probabilities using the whole train is:")

print(benchmark_probability_blind)
# Bayesian - without overfitting

# Generate stratified cv

from sklearn.model_selection import KFold



benchmark_probability_blind_no_overfit = []



n_cv = 4

skf = KFold(n_splits=n_cv, shuffle=True, random_state=1)

for indices_train, indices_test in skf.split(X=train_X.values):

    predictions = class_probability_list(train_y.iloc[indices_train].Class)

    predictions = np.repeat([predictions], indices_test.size, axis=0)

    benchmark_probability_blind_no_overfit.append(

        multi_log_loss(train_y.iloc[indices_test].Class.values, predictions))

print("The score for Bayesian probabilities with Kfold is:")

print(np.mean(benchmark_probability_blind_no_overfit))
print("Gene exploration")

train_y.Gene.value_counts(normalize=True)
