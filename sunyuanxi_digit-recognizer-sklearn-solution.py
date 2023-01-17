import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, roc_curve, auc

from sklearn.preprocessing import label_binarize

# from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_predict

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from itertools import cycle

from scipy import interp
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
X_train, X_valid, y_train, y_valid = train_test_split(train.drop(['label'], axis=1), train['label'], random_state = 0)
def plot_vector(vec):

    '''

    Takes in image vector, transforms and plots

    '''

    v_sq = vec.values.reshape((28,28))

    plt.imshow(v_sq, interpolation='nearest', cmap = 'gray')
fig = plt.figure(figsize=(8, 8))

for i in range(16):

    fig.add_subplot(4, 4, i + 1)

    plot_vector(X_train.iloc[i])

    plt.title(str(y_train.iloc[i]))

    plt.xticks([])

    plt.yticks([])
# knn = KNeighborsClassifier().fit(X_train, y_train)

# print('Mean Accuracy score (training): {:.3f}'.format(knn.score(X_train, y_train)))

# print('Mean Accuracy score (test): {:.3f}'.format(knn.score(X_valid, y_valid)))
nn = MLPClassifier(activation='logistic', learning_rate='invscaling').fit(X_train, y_train)

print('Mean Accuracy score (training): {:.3f}'.format(nn.score(X_train, y_train)))

print('Mean Accuracy score (validation): {:.3f}'.format(nn.score(X_valid, y_valid)))
nn_2 = MLPClassifier(hidden_layer_sizes=(300, 300), learning_rate='invscaling').fit(X_train, y_train)

print('Mean Accuracy score (training): {:.3f}'.format(nn_2.score(X_train, y_train)))

print('Mean Accuracy score (validation): {:.3f}'.format(nn_2.score(X_valid, y_valid)))
nn_5 = MLPClassifier(hidden_layer_sizes=(2500, 2000, 1500, 1000, 500), max_iter=15, learning_rate='invscaling', verbose=True).fit(X_train, y_train)

print('Mean Accuracy score (training): {:.3f}'.format(nn_5.score(X_train, y_train)))

print('Mean Accuracy score (test): {:.3f}'.format(nn_5.score(X_valid, y_valid)))
y_pred = nn_5.predict(X_valid)

cm = confusion_matrix(y_valid, y_pred)

num_classes = cm.shape[0]

count = np.unique(y_valid, return_counts=True)[1].reshape(num_classes, 1)



fig = plt.figure(figsize=(10,6))

ax = plt.subplot(111)

im = ax.imshow(cm/count, cmap='YlGnBu')

im.set_clim(0, 1)

cbar = ax.figure.colorbar(im, ax=ax)

ax.set_xticks(np.arange(num_classes))

ax.set_yticks(np.arange(num_classes))

plt.yticks(fontsize=13)

plt.xticks(fontsize=13)

for i in range(num_classes):

    for j in range(num_classes):

        text = ax.text(i, j, cm[j][i], ha="center", va="center", color="w" if (cm/count)[j, i] > 0.5 else "black", fontsize=13)

ax.set_ylabel('True Label', fontsize=16)

ax.set_xlabel('Predicted Label', fontsize=16)

ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')

plt.show()
num_classes=10

bi_y_test = label_binarize(y_valid, classes=range(num_classes))

y_pred_proba = nn_5.predict_proba(X_valid)

fpr = {}

tpr = {}

roc_auc = {}

for i in range(num_classes):

    fpr[i], tpr[i], _ = roc_curve(bi_y_test[:, i], y_pred_proba[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])

    

fpr['micro'], tpr['micro'], _ = roc_curve(bi_y_test.ravel(), y_pred_proba.ravel())

roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])



# Compute macro-average ROC curve and AUC

# Aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

# Interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(num_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Average and compute AUC

mean_tpr /= num_classes



fpr['macro'] = all_fpr

tpr['macro'] = mean_tpr

roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])



# Plot all ROC curves

plt.figure(figsize=(10, 10))

    

for i in range(num_classes):

        plt.plot(fpr[i], tpr[i], alpha=0.2,

                 label='ROC curve of class {0} (area = {1:0.4f})'

                 ''.format(i+1, roc_auc[i]))



plt.plot(fpr['micro'], tpr['micro'],

         label='micro-average ROC curve (area = {0:0.4f})'

         ''.format(roc_auc['micro']),

         color='orangered', linestyle=':', linewidth=3)



plt.plot(fpr['macro'], tpr['macro'],

         label='macro-average ROC curve (area = {0:0.4f})'

         ''.format(roc_auc['macro']),

         color='navy', linestyle=':', linewidth=3)



plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xticks(fontsize=13)

plt.xticks(fontsize=13)

plt.title('ROC Curves', fontsize=16)

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.legend(loc=4)

plt.show()
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission['Label'] = nn_5.predict(test)

sample_submission.head()

sample_submission.to_csv('submission.csv', index=False)