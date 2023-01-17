import numpy as np
from __future__ import print_function
import pandas as pd
from pandas import read_excel
from sklearn import decomposition, preprocessing, svm 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy import interp
from matplotlib import pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

from keras.utils.np_utils import to_categorical
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras import regularizers
from keras.callbacks import History 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from keras.utils import np_utils
history = History()

%matplotlib inline

pD = pd.read_csv("../input/data.csv",header = None, low_memory = False)
pData = pD.as_matrix()
# convert every '?' to a nan and then convert the array of strings to floats.
lab = pData[0,:];
pData = np.delete(pData, (0), axis=0)
pData[pData == '?'] = np.nan;
pData = pData.astype('float32')
# pData = pData[~np.isnan(pData).any(axis=1)]


# Grab all of the columns that aren't completely void, concatenate the target data and take out any row with remaining missing data.
d = pData[:,0:pData.shape[1]-4];
d = np.hstack((d,pData[:,pData.shape[1]-1].reshape(len(pData[:,0]),1)));
d = d[~np.isnan(d).any(axis=1)]
targets = d[:,d.shape[1]-1];
d = np.delete(d, (d.shape[1]-1), axis=1)

# My first attempt was to use LDA and min-max normalization.
dN = preprocessing.minmax_scale(d, feature_range=(-1, 1), axis=0, copy=True)
lda = LinearDiscriminantAnalysis(n_components=3)
X = lda.fit(dN, targets).transform(dN)  
targets = np.reshape(targets.astype(int),(len(targets),1))
# sort for easy plotting (out of habit)
x = np.hstack((X,targets))
x = x[x[:,1].argsort()]
# Determined how many patients belong to each group in order to plot accordingly.
type0 = sum(np.isin(x[:,1], 0));
type1 = sum(np.isin(x[:,1], 1));
q=0;
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.bar(np.linspace(1,type0,type0), x[0:type0,q], align='center', label='None')
ax1.bar(np.linspace(type0+1,type0+type1,type1), x[type0:type0+type1,q], color='red', align='center', label='Afflicted')
plt.xlabel('Patient',fontsize=18)
plt.ylabel('LDA Loading',fontsize=18)
plt.title('1D LDA Bar Plot',fontsize=18)
plt.legend(loc='upper left',prop={'size': 18});
plt.show()
clf = DecisionTreeClassifier(random_state=0)
# cross_val_score(clf, x[:,0].reshape(len(x[:,0]),1), x[:,1], cv=5)
fiveF = cross_val_score(clf, x[:,0].reshape(len(x[:,0]),1), x[:,1], cv=5)
print("All: ", fiveF, ". \nAverage: ", np.mean(fiveF) )
X = x[:,0].reshape(x[:,0].shape[0],1);
# X = x[:,0];
y = x[:,1];
n_samples, n_features = X.shape
cv = StratifiedKFold(n_splits=5)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Cross-Validation ROC of LDA',fontsize=18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()
# Normalize appropriately for PCA
dN = stats.zscore(d.astype(float), axis=0, ddof=1)
pca = decomposition.PCA(n_components=2)
pca.fit(dN)
X = pca.transform(dN)
x = np.hstack((X,targets))
x = x[x[:,2].argsort()]

type0 = sum(np.isin(x[:,2], 0));
type1 = sum(np.isin(x[:,2], 1));
# type2 = sum(np.isin(x[:,2], 2));
q=0;
r=1;
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)
ax1.scatter(x[0:type0,q],x[0:type0,r],s=25, c='blue', marker="s", label='None')
ax1.scatter(x[type0:type0+type1,q],x[type0:type0+type1,r],s=25, c='red', marker="o", label='Afflicted')
plt.xlabel('PC 1',fontsize=18)
plt.ylabel('PC 2',fontsize=18)
plt.title('2D PCA Scatter Plot',fontsize=18)
plt.legend(loc='upper left',prop={'size': 18});
clf = DecisionTreeClassifier(random_state=0)
# cross_val_score(clf, x[:,0].reshape(len(x[:,0]),1), x[:,1], cv=5)
fiveF = cross_val_score(clf, x[:,0:2], x[:,2], cv=5)
print("All: ", fiveF, ". \nAverage: ", np.mean(fiveF) )
X = x[:,0:2];
y = x[:,2];
n_samples, n_features = X.shape
cv = StratifiedKFold(n_splits=5)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=0)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=18)
plt.ylabel('True Positive Rate',fontsize=18)
plt.title('Cross-Validation ROC of PCA',fontsize=18)
plt.legend(loc="lower right", prop={'size': 15})
plt.show()
# One-hot encoding and back to min-max norm
y = np_utils.to_categorical(targets,num_classes=2)
dN = preprocessing.minmax_scale(d, feature_range=(0, 1), axis=0, copy=True)
# 50/50 train/test
train_X, test_X, train_y, test_y = train_test_split(dN, y, train_size=0.5, random_state=0)
# Model definition. Dropout did not seem to help much here
def create_baseline():
    model = Sequential()
    model.add(Dense(15, activation='relu',input_shape=(10,)))
    model.add(Dense(10, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))
#     model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid',kernel_regularizer=regularizers.l2(0.0001)))
    keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy',auc_roc])
#     my_callbacks = [EarlyStopping(monitor='auc_roc', patience=50, verbose=1, mode='max')]
    return model

# This is a function seen at 'https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras' 
# by 'https://stackoverflow.com/users/7093436/tom'
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value
# time to train
model = create_baseline();
history = model.fit(train_X, train_y,
          validation_data=(test_X, test_y),
          batch_size=32, epochs=500, verbose=1)
# AUROC curve and test accuracy for performance metric
y_pred = model.predict_proba(test_X);
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_y[:,0], y_pred[:,0]);
auc_keras = auc(fpr_keras, tpr_keras);
accuracy = np.mean(np.equal(test_y, np.round(y_pred)));
plt.figure(figsize=(10,10))
plt.plot(fpr_keras, tpr_keras, color='black', label='AUC = {:.3f}'.format(auc_keras));
plt.xlabel('False positive rate',fontsize=18);
plt.ylabel('True positive rate',fontsize=18);
plt.title('ROC curve: Max-Min Normalized - Test Accuracy = %0.2f' % (accuracy),fontsize=18);
plt.legend(loc='lower right',fontsize=18);
# print('Test Accuracy: ', np.mean(np.equal(test_y, np.round(y_pred))));

# Train and test loss for reference.
train_loss = history.history['loss']
val_loss   = history.history['val_loss']
# train_acc  = estimator.history['acc']
# val_acc    = estimator.history['val_acc']
xc         = range(500)

_=plt.figure(figsize=(10,10))
plt.plot(xc, train_loss,label='Training')
plt.plot(xc, val_loss, label='Validation')
plt.xlabel('Epochs',fontsize=18)
plt.ylabel('Loss',fontsize=18)
plt.title('Cost Curves',fontsize=18)
plt.legend(loc="upper right", prop={'size': 15})
model = ExtraTreesClassifier();
model.fit(dN, y);
importance = pd.DataFrame({ '1. Params' : lab[0:-4], '2. Importance' : model.feature_importances_});
importance
