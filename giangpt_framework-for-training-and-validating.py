import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
input_data = "../input/forcedataset/force-ai-well-logs/train.csv"
TARGET_1 = "FORCE_2020_LITHOFACIES_LITHOLOGY"
TARGET_2 = "FORCE_2020_LITHOFACIES_CONFIDENCE"
WELL_NAME = 'WELL'
#read csv data
df = pd.read_csv(input_data, sep=';')
wells = np.unique(df['WELL'].values)
#number: rock types dictionary
lithology_keys = {30000: 'Sandstone',
                 65030: 'Sandstone/Shale',
                 65000: 'Shale',
                 80000: 'Marl',
                 74000: 'Dolomite',
                 70000: 'Limestone',
                 70032: 'Chalk',
                 88000: 'Halite',
                 86000: 'Anhydrite',
                 99000: 'Tuff',
                 90000: 'Coal',
                 93000: 'Basement'}
unused_columns = ['RSHA', 'SGR', 'NPHI', 'BS', 'DTS', 'DCAL', 'RMIC', 'ROPA', 'RXO']
unused_columns += [WELL_NAME, 'GROUP', 'FORMATION']
# ADD two target columns into unused columns
unused_columns += [TARGET_1, TARGET_2]
all_columns = list(df.columns)

use_columns = [c for c in all_columns if c not in unused_columns]
for c in use_columns:
    df[c].fillna(df[c].mean(), inplace=True)

train_wells = list(np.unique(df['WELL'].values))[:70]
# Use this condition to find out which rows in the data is select for training
train_mask = df[WELL_NAME].isin(train_wells)
X_train = df[train_mask][use_columns].values
y_train = df[train_mask][TARGET_1].values
print(X_train.shape, y_train.shape)
X_valid = df[~train_mask][use_columns].values
y_valid = df[~train_mask][TARGET_1].values
print(X_valid.shape, y_valid.shape)
rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
predict_y = rf_clf.predict(X_valid)
# plot confusion matrix
def draw_confusion_matrix(model, X_valid, y_valid):
    fig, ax = plt.subplots(figsize=(6,6))
    disp = plot_confusion_matrix(model, X_valid, y_valid, normalize = None, xticks_rotation = 'vertical', ax = ax)
    disp.ax_.set_title("Plot Confusion Matrix Not Normalized")
    fig1, ax1 = plt.subplots(figsize=(6,6))
    disp1 = plot_confusion_matrix(model, X_valid, y_valid, normalize = 'true', values_format = ".2f", xticks_rotation = 'vertical', ax = ax1)
    disp1.ax_.set_title("Plot Confusion Matrix Normalized")
    plt.show()
# accuracy calculation
def calculate_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true = y_true, y_pred = y_pred)
    tp = 0
    for i in range(len(cm)):
        tp += cm[i][i]
    accuracy = 1.0 * tp / np.sum(cm)
    return accuracy

cm_rf = confusion_matrix(y_true = y_valid, y_pred = predict_y)
cm_rf
draw_confusion_matrix(rf_clf, X_valid, y_valid)
accuracy_RandomForest = calculate_accuracy(y_valid, predict_y)
penalty_matrix = np.load("../input/penalty-matrix/penalty_matrix.npy")
# Position of each type of rock in the penalty_matrix
penalty_dict = {"Sandstone": 0,
                "Sandstone/Shale": 1,
                "Shale": 2, 
                "Marl": 3,
                "Dolomite": 4,
                "Limestone": 5,
                "Chalk": 6,
                "Halite": 7,
                "Anhydrite": 8,
                "Tuff": 9,
                "Coal": 10,
                "Basement": 11}
# Used for getting the right "rock number" from confusion matrix index
cm_rock_idx = np.unique(df[TARGET_1].values)
# penalty calculation according to FORCE metrics.
def calculate_penalty(cm = None, penalty_matrix = None, lithology_dict = None, penalty_dict = None, cm_rock_idx = None):
    sum_penalty = 0
    for i in range(len(cm)):
        for j in range(len(cm)):
            rock_i = lithology_dict[cm_rock_idx[i]]
            rock_j = lithology_dict[cm_rock_idx[j]]
            penalty_i = penalty_dict[rock_i]
            penalty_j = penalty_dict[rock_j]
            sum_penalty += cm[i][j] * penalty_matrix[penalty_i][penalty_j]
    return -1.0 * sum_penalty / np.sum(cm)
penalty_rf = calculate_penalty(cm_rf, penalty_matrix, lithology_keys, penalty_dict, cm_rock_idx)
penalty_rf