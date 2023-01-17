import numpy as np

from sklearn.metrics import auc

from sklearn import metrics

import matplotlib.pyplot as plt

import pickle
with open("../input/alaska2-efficientnet-on-tpus-valid-preds/valid.pkl", "rb") as f:

    [y_valid, y_true] = pickle.load(f)
def calculate_metric(y_true, y_pred):

    tpr_thresholds = [0.0, 0.4, 1.0]

    weights =        [       2, 1]

    

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    normalization = np.dot(areas, weights)



    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)

    competition_metric = 0



    for idx, weight in enumerate(weights):

        y_min = tpr_thresholds[idx]

        y_max = tpr_thresholds[idx + 1]

        mask = (y_min < tpr) & (tpr < y_max)



        x_padding = np.linspace(fpr[mask][-1], 1, 100)



        x = np.concatenate([fpr[mask], x_padding])

        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])

        y = y - y_min # normalize such that curve starts at y=0

        score = auc(x, y)

        submetric = score * weight

        best_subscore = (y_max - y_min) * weight

        competition_metric += submetric

        

    competition_metric = competition_metric / normalization

    return competition_metric
def show_auc_plots(y_true, y_pred):

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid)

    

    plt.scatter(fpr, tpr)

    plt.title("ROC Curve")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.plot(np.linspace(0, 1), [0.4] *50, color="green")

    plt.show()

    

    tpr_thresholds = [0.0, 0.4, 1.0]

    

    for idx in [1, 0]:

        

        y_min = tpr_thresholds[idx]

        y_max = tpr_thresholds[idx + 1]

        mask = (y_min < tpr) & (tpr < y_max)



        x_padding = np.linspace(fpr[mask][-1], 1, 100)



        x = np.concatenate([fpr[mask], x_padding])

        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])

        y = y - y_min # normalize such that curve starts at y=0

        score = auc(x, y)



        fig, ax = plt.subplots(1, 1)

        ax.fill_between(x, 0, y)

        ax.scatter(x, y)

        plt.xlim(0, 1)

        plt.ylim(0, y_max - y_min)

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title(f"AUC curve between {y_min} and {y_max}. AUC: {score:.3f}, Best AUC: {y_max - y_min}")

        plt.show()

calculate_metric(y_true, y_valid)
show_auc_plots(y_true, y_valid)