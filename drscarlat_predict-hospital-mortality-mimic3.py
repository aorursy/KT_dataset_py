# IMPORT modules

# Turn GPU on



import pandas as pd

import numpy as np

import random as rnd

import pprint

from itertools import cycle, islice

import numpy as np



from scipy.stats import multivariate_normal



from sklearn.model_selection import train_test_split

from sklearn import model_selection, preprocessing

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV,KFold, cross_val_predict, StratifiedKFold, train_test_split, learning_curve, ShuffleSplit

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC 

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, ShuffleSplit

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_curve, average_precision_score, auc

from sklearn.utils.fixes import signature



from sklearn.decomposition import PCA



from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import tensorflow as tf



from keras import models, regularizers, layers, optimizers, losses, metrics

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils



import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))
# Load MIMIC2 data 



data = pd.read_csv('../input/mimic3c.csv')

print("With id", data.shape)



data_full = data.drop('hadm_id', 1)

print("No id",data_full.shape)

print(data_full.shape)

data_full.info()

data_full.describe()
data_full.head(10)
data_full.hist(bins=50, figsize=(20,15))

plt.show()
age_histogram = data_full.hist(column='age', bins=20, range=[0, 100])

for ax in age_histogram.flatten():

    ax.set_xlabel("Age")

    ax.set_ylabel("Num. of Patients")

plt.show()

data_full.groupby('ExpiredHospital').size().plot.bar()

plt.show()
# Label = ExpiredHospital

y = data_full['ExpiredHospital']

X = data_full.drop('ExpiredHospital', 1)



X = X.drop('LOSdays', 1)

X = X.drop('LOSgroupNum', 1)

X = X.drop('AdmitDiagnosis', 1)

X = X.drop('AdmitProcedure', 1)

X = X.drop('marital_status', 1)

X = X.drop('ethnicity', 1)

X = X.drop('religion', 1)

X = X.drop('insurance', 1)



print("y - Labels", y.shape)

print("X - No Label No id ", X.shape)

print(X.columns)
# Check that all X columns have no missing values

X.info()

X.describe()
data_full.groupby('ExpiredHospital').size().plot.bar()

plt.show()

data_full.groupby('admit_type').size().plot.bar()

plt.show()

data_full.groupby('admit_location').size().plot.bar()

plt.show()
# MAP Text to Numerical Data with one-hot-encoding to convert categorical features to numerical



print(X.shape)

categorical_columns = [

                    'gender',                     

                    'admit_type',

                    'admit_location'

                      ]

for col in categorical_columns:

    #if the original column is present replace it with a one-hot

    if col in X.columns:

        one_hot_encoded = pd.get_dummies(X[col])

        X = X.drop(col, axis=1)

        X = X.join(one_hot_encoded, lsuffix='_left', rsuffix='_right')

        

print(X.shape)
print(X.columns)

#print(X['VENTRICULOSTOMY          '])
print(data_full.shape)

print(X.shape)



XnotNorm = X.copy()

print('XnotNorm ', XnotNorm.shape)



#yFI = data_full.expired_icu

ynotNorm = y.copy()

print('ynotNorm ', ynotNorm.shape)
# Normalize X



x = XnotNorm.values #returns a numpy array

scaler = preprocessing.StandardScaler()

x_scaled = scaler.fit_transform(x)

XNorm = pd.DataFrame(x_scaled, columns=XnotNorm.columns)

print(XNorm)

# SPLIT into Train & Test



X_train, X_test, y_train, y_test = train_test_split(XNorm, y, test_size=0.1, random_state=42)

print ('X_train: ', X_train.shape)

print ('X_test: ', X_test.shape)

print ('y_train: ', y_train.shape)

print ('y_test: ', y_test.shape)
# Test Models and evaluation metric

seed = 7

scoring = 'accuracy' 



# Spot Check Algorithms

Mymodels = []

#Mymodels.append(('LogReg', LogisticRegression()))

Mymodels.append(('RandomForest', RandomForestClassifier()))

#Mymodels.append(('SGDclassifier', SGDClassifier()))

#Mymodels.append(('KNearestNeighbors', KNeighborsClassifier()))

#Mymodels.append(('DecisionTreeClassifier', DecisionTreeClassifier()))

#Mymodels.append(('GaussianNB', GaussianNB()))

#Mymodels.append(('SVM', SVC()))



# Evaluate each model in turn

results = []

names = []

for name, model in Mymodels:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg) 
# Set the model according to above results



model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,

            oob_score=False, random_state=None, verbose=0,

            warm_start=False)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Error")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = 1-np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = 1-np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
# LEARNING CURVES Train / Validation



title = "Learning Curves "

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

plot_learning_curve(model, title, XNorm, y, cv=cv, n_jobs=5)
# Model FINAL fit and evaluation on test



model.fit(X_train, y_train)

final_predictions = model.predict(X_test)



#final_acc = accuracy(y_test, final_predictions)

# Confusion matrix



conf_mx = confusion_matrix(y_test, final_predictions)



TN = conf_mx[0,0]

FP = conf_mx[0,1]

FN = conf_mx[1,0]

TP = conf_mx[1,1]



print ('TN: ', TN)

print ('FP: ', FP)

print ('FN: ', FN)

print ('TP: ', TP)



recall = TP/(TP+FN)

precision = TP/(TP+FP)



print (recall, precision)
def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,

                          normalize=False):

    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()

plot_confusion_matrix(conf_mx, 

                      normalize    = False,

                      target_names = ['lived', 'died'],

                      title        = "Confusion Matrix")
print ('precision ',round(precision_score(y_test, final_predictions),4))

print ('recall ',round(recall_score(y_test, final_predictions) ,4))

print ('accuracy ',round(accuracy_score(y_test, final_predictions),4))

print ('F1 score ',round(f1_score(y_test, final_predictions),4))
# FEATURE IMPORTANCE 



trainFinalFI = XNorm

yFinalFI = y

model.fit(trainFinalFI,yFinalFI)



FI_model = pd.DataFrame({"Feature Importance":model.feature_importances_,}, index=trainFinalFI.columns)

FI_model[FI_model["Feature Importance"] > 0.01].sort_values("Feature Importance").plot(kind="barh",figsize=(15,25))

plt.xticks(rotation=90)

plt.xticks(rotation=90)

plt.show()
# List of important features for model

FI_model = pd.DataFrame({"Feature Importance":model.feature_importances_,}, index=trainFinalFI.columns)

FI_model=FI_model.sort_values('Feature Importance', ascending = False)

print(FI_model[FI_model["Feature Importance"] > 0.0025])
# AUC/ROC curves should be used when there are roughly equal numbers of observations for each class

# Precision-Recall curves should be used when there is a moderate to large class imbalance



# calculate AUC

auc = roc_auc_score(y_test, final_predictions)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, final_predictions)

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')

plt.title('AUC for ROC')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
# Modify the raw final_predictions - prediction probs into 0 and 1



Preds = final_predictions.copy()

#print(len(Preds))

#print(Preds)

Preds[ np.where( Preds >= 0.5 ) ] = 1

Preds[ np.where( Preds < 0.5 ) ] = 0



# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, Preds)

# calculate F1 score

f1 = f1_score(y_test, Preds)

print('f1=%.3f' % (f1))

# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the roc curve for the model

plt.plot(recall, precision, marker='.')

# show the plot

plt.show()
# NN MODEL



# Use of DROPOUT

model = models.Sequential()

model.add(layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(30,)))

#model.add(layers.BatchNormalization())

model.add(layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.001)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())



# FIT / TRAIN model



NumEpochs = 100

BatchSize = 16



model.compile(optimizer=optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['binary_accuracy'])

history = model.fit(X_train, y_train, epochs=NumEpochs, batch_size=BatchSize, validation_data=(X_test, y_test))



results = model.evaluate(X_test, y_test)

print("_"*100)

print("Test Loss and Accuracy")

print("results ", results)

history_dict = history.history

history_dict.keys()
# VALIDATION LOSS curves



plt.clf()

history_dict = history.history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, (len(history_dict['loss']) + 1))

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



# VALIDATION ACCURACY curves



plt.clf()

acc_values = history_dict['binary_accuracy']

val_acc_values = history_dict['val_binary_accuracy']

epochs = range(1, (len(history_dict['binary_accuracy']) + 1))

plt.plot(epochs, acc_values, 'bo', label='Training acc')

plt.plot(epochs, val_acc_values, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()



# Final Fit / Predict



# NOTE final_predictions is a list of probabilities

#model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

#history = model.fit(X_train, y_train, epochs=NumEpochs, batch_size=BatchSize)



final_predictions = model.predict(X_test)
# Modify the raw final_predictions - prediction probs into 0 and 1



Preds = final_predictions.copy()

#print(len(Preds))

#print(Preds)

Preds[ np.where( Preds >= 0.5 ) ] = 1

Preds[ np.where( Preds < 0.5 ) ] = 0

#print(Preds)


# Confusion matrix



conf_mx = confusion_matrix(y_test, Preds)



TN = conf_mx[0,0]

FP = conf_mx[0,1]

FN = conf_mx[1,0]

TP = conf_mx[1,1]



print ('TN: ', TN)

print ('FP: ', FP)

print ('FN: ', FN)

print ('TP: ', TP)



recall = TP/(TP+FN)

precision = TP/(TP+FP)



print (recall, precision)
plot_confusion_matrix(conf_mx, 

                      normalize    = False,

                      target_names = ['lived', 'died'],

                      title        = "Confusion Matrix")
print ('precision ',precision_score(y_test, Preds))

print ('recall ',recall_score(y_test, Preds) )

print ('accuracy ',accuracy_score(y_test, Preds))

print ('F1 score ',f1_score(y_test, Preds))
# AUC/ROC curves should be used when there are roughly equal numbers of observations for each class

# Precision-Recall curves should be used when there is a moderate to large class imbalance



# calculate AUC

auc = roc_auc_score(y_test, Preds)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, Preds)

# plot no skill

plt.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

plt.plot(fpr, tpr, marker='.')

plt.title('ROC ')

# show the plot

plt.show()
# calculate precision-recall curve

precision, recall, thresholds = precision_recall_curve(y_test, Preds)

# calculate F1 score

f1 = f1_score(y_test, Preds)

# calculate precision-recall AUC

#auc = auc(recall, precision)

# calculate average precision score

ap = average_precision_score(y_test, Preds)

print('f1=%.3f ap=%.3f' % (f1, ap))

# plot no skill

plt.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the roc curve for the model

plt.plot(recall, precision, marker='.')

# show the plot

plt.show()