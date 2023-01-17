%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve, auc

from scipy import interp

import warnings

warnings.filterwarnings("ignore")
def run(x_c0, x_c1):

    # Create Target Variable

    y = np.array([1 if v < x_c0.size else 0 for v in range(x_c0.size + x_c1.size)])



    fig, axs = plt.subplots(1, 2, figsize = (16,8))



    # Plot the distributions

    ax = sns.kdeplot(x_c0, ax=axs[0])

    ax = sns.kdeplot(x_c1, ax=axs[0])



    # Prepare classification

    X = np.append(x_c0, x_c1)

    X = X.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y)



    # Run a simple linear classifier    

    clf = LogisticRegression(solver='lbfgs')

    clf.fit(X_train, y_train)

    y_preds = clf.predict_proba(X_test)



    # Calculate AUC

    fpr, tpr, thresholds = roc_curve(y_test, y_preds[:,1])    

    roc_auc = auc(fpr, tpr)



    # Plot ROC

    axs[1].plot(fpr, tpr, lw=1, label='(AUC = %0.2f)' % (roc_auc))

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")
run(np.random.normal(-7, 2, 100), np.random.normal(7, 2, 100))
run(np.random.normal(-7, 5, 100), np.random.normal(7, 5, 100))
run(np.random.normal(-7, 14, 100), np.random.normal(7, 14, 100))