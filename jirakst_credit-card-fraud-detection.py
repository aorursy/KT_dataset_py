# Basic

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# Tools

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.manifold import TSNE

from imblearn.over_sampling import SMOTE

from scipy import stats

from collections import Counter



# Model

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score, cross_val_predict

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.under_sampling import NearMiss



# Algorithms

from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network

import xgboost as xgb

from xgboost.sklearn import XGBClassifier



# Evaluation

from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix, recall_score, precision_score, precision_recall_curve



# System

import os

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
print(os.listdir("../input"))
df = pd.read_csv("../input/creditcard.csv")
df.shape
df.columns
df.head()
sns.countplot('Class', data=df)

print('Frauds: ', round(df['Class'].value_counts()[1] / len(df) * 100, 2), '%')
df.describe()
df.info()
df.isnull().sum()
# Scale whole dataset

rs = RobustScaler()



df['Time'] = rs.fit_transform(df['Time'].values.reshape(-1,1))

df['Amount'] = rs.fit_transform(df['Amount'].values.reshape(-1,1))
# Check

df.head()
# TODO: Normlize dataset with BoxCox transformation.
#TODO: Random undersample with NearMiss algorithm. 
# Count Frauds

df['Class'].value_counts()[1]
# Create balanced sub-dataset

frauds = df.loc[df['Class'] == 1]

nonfrauds = df.loc[df['Class'] == 0][:492]



undersample = pd.concat([frauds, nonfrauds])
# Check

undersample.shape
X = df.drop(['Class'], axis=1)

y = df['Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Apply SMOTE to train set

sm = SMOTE(random_state=22)

X_oversample, y_oversample = sm.fit_sample(X_train, y_train)



#Check majority vs. minority class distribution in train set after resampling



print('Fraudulent share, train set (after SMOTE): {0:.2%}'.format(sum(y_resampled==1)/len(y_resampled)))
# Convert array into dataframe

X_oversample = pd.DataFrame(X_oversample)

y_oversample = pd.DataFrame(y_oversample)
X = undersample.drop(['Class'], axis=1)

y = undersample['Class']



tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
# Define plot

f, (ax1) = plt.subplots(1, 1, figsize=(16,8))

f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)



# t-SNE scatter plot

ax1.scatter(tsne[:,0], tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

ax1.scatter(tsne[:,0], tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)

ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)
X = undersample.drop('Class', axis=1)

y = undersample['Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Convert values into an array

X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
MLA = [

    ensemble.AdaBoostClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    gaussian_process.GaussianProcessClassifier(),

    linear_model.LogisticRegressionCV(),

    linear_model.RidgeClassifierCV(),

    linear_model.Perceptron(),

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    neighbors.KNeighborsClassifier(),

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    xgb.XGBClassifier()

    ]
col = []

algorithms = pd.DataFrame(columns = col)

idx = 0



#Train and score algorithms

for a in MLA:

    

    a.fit(X_oversample, y_oversample)

    pred = a.predict(original_Xtest)

    acc = accuracy_score(original_ytest, pred) #Other way: a.score(X_test, y_test)

    f1 = f1_score(original_ytest, pred)

    cv = cross_val_score(a, original_Xtest, original_ytest).mean()

    

    Alg = a.__class__.__name__

    

    algorithms.loc[idx, 'Algorithm'] = Alg

    algorithms.loc[idx, 'Accuracy'] = round(acc * 100, 2)

    algorithms.loc[idx, 'F1 Score'] = round(f1 * 100, 2)

    algorithms.loc[idx, 'CV Score'] = round(cv * 100, 2)



    idx+=1
#Compare invidual models

algorithms.sort_values(by = ['CV Score'], ascending = False, inplace = True)    

algorithms.head()
#Plot them

g = sns.barplot("CV Score", "Algorithm", data = algorithms)

g.set_xlabel("CV score")

g = g.set_title("Algorithm Scores")
# Train

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_train)

#cv_pred = cross_val_predict(xgb, X_train, y_train, cv=kfold)



# Score

print('XGBoost classifier ROC AUC score: ', roc_auc_score(y_train, y_pred))
# We will undersample during cross validating

undersample_X = df.drop('Class', axis=1)

undersample_y = df['Class']



sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for train_index, test_index in sss.split(undersample_X, undersample_y):

    print("Train:", train_index, "Test:", test_index)

    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]

    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]

    

undersample_Xtrain = undersample_Xtrain.values

undersample_Xtest = undersample_Xtest.values

undersample_ytrain = undersample_ytrain.values

undersample_ytest = undersample_ytest.values 



undersample_accuracy = []

undersample_precision = []

undersample_recall = []

undersample_f1 = []

undersample_auc = []



# Implementing NearMiss Technique 

# Distribution of NearMiss (Just to see how it distributes the labels we won't use these variables)

X_nearmiss, y_nearmiss = NearMiss().fit_sample(undersample_X.values, undersample_y.values)

print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))

# Cross Validating the right way



for train, test in sss.split(undersample_Xtrain, undersample_ytrain):

    undersample_pipeline = imbalanced_make_pipeline(NearMiss(sampling_strategy='majority'), xgb) # SMOTE happens during Cross Validation not before..

    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])

    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])

    

    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))

    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))

    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))

    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))

    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))
X = df.drop('Class', axis=1)

y = df['Class']



original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)
kfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
# Use pre-trained model on the full sample

#xgb = XGBClassifier()

#xgb.fit(X_train, y_train)

#cv_pred = cross_val_predict(xgb, X_train, y_train, cv=kfold)

cv_pred = cross_val_predict(xgb, original_Xtrain, original_ytrain, cv=kfold)



# Score

print('XGBoost classifier ROC AUC score: ', roc_auc_score(original_ytrain, cv_pred))
clf_xgb = XGBClassifier(objective = 'binary:logistic')
param_dist = {'n_estimators': stats.randint(150, 1000),

              'learning_rate': stats.uniform(0.01, 0.6),

              'subsample': stats.uniform(0.3, 0.9),

              'max_depth': [3, 4, 5, 6, 7, 8, 9],

              'colsample_bytree': stats.uniform(0.5, 0.9),

              'min_child_weight': [1, 2, 3, 4]

             }
#numFolds = 5

#kfold_5 = cross_validation.KFold(n = len(X), shuffle = True, n_folds = numFolds)



kfold = StratifiedKFold(n_splits=5)



clf = RandomizedSearchCV(clf_xgb, 

                         param_distributions = param_dist,

                         cv = kfold,  

                         n_iter = 5,

                         scoring = 'roc_auc', 

                         error_score = 0, 

                         verbose = 3, 

                         n_jobs = -1)
# Train

clf = XGBClassifier()

clf.fit(X_train, y_train)

cv_pred = cross_val_predict(clf, original_Xtest, original_ytest, cv=kfold)



# Score

print('XGBoost classifier ROC AUC score: ', roc_auc_score(original_ytrain, cv_pred))
clf = clf.fit(X_train, y_train)

pred = clf.predict(original_Xtest)

acc = accuracy_score(original_ytest, pred) #Other way: vc.score(X_test, y_test)

f1 = f1_score(original_ytest, pred)

cv = cross_val_score(clf, original_Xtest, original_ytest).mean()
# Convert values into an array

'''original_Xtrain = original_Xtrain.values

original_Xtest = original_Xtest.values

original_ytrain = original_ytrain.values

original_ytest = original_ytest.values'''
# List to append the score and then find the average

'''accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []'''
# Make SMOTE happen during Cross Validation

'''for train, test in kfold.split(original_Xtrain, original_ytrain):

    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), clf)

    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])

    best_est = clf.best_estimator_

    prediction = clf.predict(original_Xtrain[test])

    

    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))

    precision_lst.append(precision_score(original_ytrain[test], prediction))

    recall_lst.append(recall_score(original_ytrain[test], prediction))

    f1_lst.append(f1_score(original_ytrain[test], prediction))

    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))'''
import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy



n_inputs = X_train.shape[1]



undersample_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(32, activation='relu'),

    Dense(2, activation='softmax')

])
undersample_model.summary()
undersample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
undersample_model.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=2)
undersample_predictions = undersample_model.predict(original_Xtest, batch_size=200, verbose=0)
undersample_fraud_predictions = undersample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)
import itertools



# Create a confusion matrix

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=14)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
undersample_cm = confusion_matrix(original_ytest, undersample_fraud_predictions)

labels = ['No Fraud', 'Fraud']



plot_confusion_matrix(undersample_cm, labels, title="UnderSample \n Confusion Matrix", cmap=plt.cm.Reds)
fpr, tpr, thresold = roc_curve(y_train, cv_pred)
def logistic_roc_curve(fpr, tpr):

    plt.figure(figsize=(12,8))

    plt.title('ROC AUC Curve', fontsize=16)

    plt.plot(fpr, tpr, 'b-', linewidth=2)

    plt.plot([0, 1], [0, 1], 'r--')

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.axis([-0.01,1,0,1])

    

    

logistic_roc_curve(fpr, tpr)

plt.show()
print('ROC AUC score: ', roc_auc_score(y_train, cv_pred))
confusion_matrix(y_train, cv_pred)
precision, recall, threshold = precision_recall_curve(y_train, cv_pred)



def plot_precision_and_recall(precision, recall, threshold):

    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)

    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)

    plt.xlabel("threshold", fontsize=19)

    plt.legend(loc="upper right", fontsize=19)

    plt.ylim([0, 1])



plt.figure(figsize=(14, 7))

plot_precision_and_recall(precision, recall, threshold)

plt.show()
print("Precision:", precision_score(y_train, cv_pred))

print("Recall:",recall_score(y_train, cv_pred))