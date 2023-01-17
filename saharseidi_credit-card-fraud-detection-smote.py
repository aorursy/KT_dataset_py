# Imported Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import collections

import imblearn

from sklearn.ensemble import IsolationForest

from sklearn.metrics import make_scorer, f1_score

from sklearn import model_selection

from sklearn.datasets import make_classification

from sklearn.neighbors import LocalOutlierFactor

from pylab import rcParams

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

LABELS = ["Normal", "Fraud"]



# Other Libraries

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import warnings

warnings.filterwarnings("ignore")
import pandas as pd

data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

data.head()

data.isnull().sum().max()

print('Legitimate transactions are', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')

print('Fraud transactions are', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')
#colors = ["#DF0101", "#0101DF"]

colors = ['red', 'gold']

explode = (0.1, 0)  # explode 1st slice



#sns.countplot('Class', data=data, palette=colors)

labels = ['Fraud (0.17)', 'Legit (99.83)']

sizes = [492,284315]

plt.pie(sizes,explode = explode, colors=colors, labels = labels,shadow=True, startangle=90)

#plt.title('Distribution of 2 classes \n (0: Legit || 1: Fraud)', fontsize=10)





# Scaling the Time and Amount features

from sklearn.preprocessing import StandardScaler, RobustScaler



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



data['scaled_amount'] = std_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

data['scaled_time'] = std_scaler.fit_transform(data['Time'].values.reshape(-1,1))



data.drop(['Time','Amount'], axis=1, inplace=True)

scaled_amount = data['scaled_amount']

scaled_time = data['scaled_time']



data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

data.insert(0, 'scaled_amount', scaled_amount)

data.insert(1, 'scaled_time', scaled_time)



# Amount and Time are Scaled!



data.head()
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit



print('No Frauds', round(data['Class'].value_counts()[0]/len(data) * 100,2), '% of the dataset')

print('Frauds', round(data['Class'].value_counts()[1]/len(data) * 100,2), '% of the dataset')



X = data.drop('Class', axis=1)

y = data['Class']



sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]



# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)



# Check the Distribution of the labels





# Turn into an array

original_Xtrain = original_Xtrain.values

original_Xtest = original_Xtest.values

original_ytrain = original_ytrain.values

original_ytest = original_ytest.values



# See if both the train and test label distribution are similarly distributed

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)

test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print('-' * 100)



print('Label Distributions: \n')

print(train_counts_label/ len(original_ytrain))

print(test_counts_label/ len(original_ytest))
# To improve the accuracy of the model, we can remove those features that are highly

# correlated with the class and are extreme outliers. We can change the threshold 

# to detect the outliers

from scipy.stats import norm



f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))



v14_fraud_dist = data['V14'].loc[data['Class'] == 1].values

sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')

ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)



v12_fraud_dist = data['V12'].loc[data['Class'] == 1].values

sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB')

ax2.set_title('V12 Distribution \n (Fraud Transactions)', fontsize=14)





v17_fraud_dist = data['V17'].loc[data['Class'] == 1].values

sns.distplot(v17_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9')

ax3.set_title('V17 Distribution \n (Fraud Transactions)', fontsize=14)



plt.show()
# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)

v14_fraud = data['V14'].loc[data['Class'] == 1].values

q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)

print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))

v14_iqr = q75 - q25

print('iqr: {}'.format(v14_iqr))



v14_cut_off = v14_iqr * 1.5

v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off

print('Cut Off: {}'.format(v14_cut_off))

print('V14 Lower: {}'.format(v14_lower))

print('V14 Upper: {}'.format(v14_upper))



outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]

print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))

print('V10 outliers:{}'.format(outliers))



data = data.drop(data[(data['V14'] > v14_upper) | (data['V14'] < v14_lower)].index)

print('----' * 44)



# -----> V12 removing outliers from fraud transactions

v12_fraud = data['V12'].loc[data['Class'] == 1].values

q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)

v12_iqr = q75 - q25



v12_cut_off = v12_iqr * 1.5

v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off

print('V12 Lower: {}'.format(v12_lower))

print('V12 Upper: {}'.format(v12_upper))

outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]

print('V12 outliers: {}'.format(outliers))

print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))

data = data.drop(data[(data['V12'] > v12_upper) | (data['V12'] < v12_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(data)))

print('----' * 44)





# Removing outliers V17 Feature

v17_fraud = data['V17'].loc[data['Class'] == 1].values

q25, q75 = np.percentile(v17_fraud, 25), np.percentile(v17_fraud, 75)

v17_iqr = q75 - q25



v17_cut_off = v17_iqr * 1.5

v17_lower, v17_upper = q25 - v17_cut_off, q75 + v17_cut_off

print('V17 Lower: {}'.format(v17_lower))

print('V17 Upper: {}'.format(v17_upper))

outliers = [x for x in v17_fraud if x < v17_lower or x > v17_upper]

print('V17 outliers: {}'.format(outliers))

print('Feature V17 Outliers for Fraud Cases: {}'.format(len(outliers)))

data = data.drop(data[(data['V17'] > v17_upper) | (data['V17'] < v17_lower)].index)

print('Number of Instances after outliers removal: {}'.format(len(data)))



# Boxplots with outliers removed



f,(ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,6))



colors = ['#B3F9C5', '#f9c5b3']

# Feature V14

sns.boxplot(x="Class", y="V14", data=data,ax=ax1, palette=colors)

ax1.set_title("V14 Feature \n Reduction of outliers", fontsize=14)

ax1.annotate('Fewer extreme \n outliers', xy=(0.98, -17.5), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)



# Feature 12

sns.boxplot(x="Class", y="V12", data=data, ax=ax2, palette=colors)

ax2.set_title("V12 Feature \n Reduction of outliers", fontsize=14)

ax2.annotate('Fewer extreme \n outliers', xy=(0.98, -17.3), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)



# Feature V17

sns.boxplot(x="Class", y="V17", data=data, ax=ax3, palette=colors)

ax3.set_title("V17 Feature \n Reduction of outliers", fontsize=14)

ax3.annotate('Fewer extreme \n outliers', xy=(0.95, -16.5), xytext=(0, -12),

            arrowprops=dict(facecolor='black'),

            fontsize=14)





plt.show()
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
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.metrics import confusion_matrix







print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))

print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))



# List to append the score and then find the average

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []



# Logistic Regression



log_reg_sm = LogisticRegression()

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)

# Implementing SMOTE Technique for Log reg

# Cross Validating the right way

# Parameters

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

for train, test in sss.split(original_Xtrain, original_ytrain):

    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..

    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])

    best_est = rand_log_reg.best_estimator_

    prediction = best_est.predict(original_Xtrain[test])

    

    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))

    precision_lst.append(precision_score(original_ytrain[test], prediction))

    recall_lst.append(recall_score(original_ytrain[test], prediction))

    f1_lst.append(f1_score(original_ytrain[test], prediction))

    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))

    

print('---' * 45)

print('Logistic Regression')

print("accuracy: {}".format(np.mean(accuracy_lst)))

print("precision: {}".format(np.mean(precision_lst)))

print("recall: {}".format(np.mean(recall_lst)))

print("f1: {}".format(np.mean(f1_lst)))

print('---' * 45)



# Prediction for Logistic Regression with SMOTE

labels = ['No Fraud', 'Fraud']

smote_prediction = best_est.predict(original_Xtest)

print(classification_report(original_ytest, smote_prediction, target_names=labels))







from sklearn.metrics import confusion_matrix



log_reg_cf = confusion_matrix(original_ytest, smote_prediction)

plot_confusion_matrix(log_reg_cf, labels, title="Logistic Regression \n Confusion Matrix")
# I skip cross validation bcs of long run time



sm = SMOTE(ratio='minority', random_state=42)

Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)





Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)



# KNN

KNN_sm = KNeighborsClassifier()



KNN_sm.fit(Xsm_train, ysm_train)

# Prediction for KNN with SMOTE

labels = ['No Fraud', 'Fraud']

smote_prediction_KNN = KNN_sm.predict(original_Xtest)

print(classification_report(original_ytest, smote_prediction_KNN, target_names=labels))
from sklearn.metrics import confusion_matrix



KNN_cf = confusion_matrix(original_ytest, smote_prediction_KNN)

plot_confusion_matrix(KNN_cf, labels, title="KNN \n Confusion Matrix")
# I skip cross validation bcs of long run time



sm = SMOTE(ratio='minority', random_state=42)

Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)





Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)

# We Improve the score by 2% points approximately 

# Implement GridSearchCV and the other models.



# RF

RF_sm = RandomForestClassifier()



RF_sm.fit(Xsm_train, ysm_train)

# Prediction for RF with SMOTE

labels = ['No Fraud', 'Fraud']

smote_prediction_RF = RF_sm.predict(original_Xtest)

print(classification_report(original_ytest, smote_prediction_RF, target_names=labels))
from sklearn.metrics import confusion_matrix



RF_cf = confusion_matrix(original_ytest, smote_prediction_RF)

plot_confusion_matrix(RF_cf, labels, title="RF \n Confusion Matrix")
# This algorithm runs faster than the othe ones

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, RandomizedSearchCV

#XGBoost

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []

xgb_sm = XGBClassifier()   

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['binary:logistic'],

              'learning_rate': [0.05], #so called `eta` value

              'max_depth': [6],

              'min_child_weight': [11],

              'silent': [1],

              'subsample': [0.8],

              'colsample_bytree': [0.7],

              'n_estimators': [10], #number of trees, change it to 1000 for better results

              'missing':[-999],

              'seed': [1337]}

rand_xgb = RandomizedSearchCV(XGBClassifier(), parameters, n_iter=4)

# Implementing SMOTE Technique for xgb

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['binary:logistic'],

              'learning_rate': [0.05], #so called `eta` value

              'max_depth': [6],

              'min_child_weight': [11],

              'silent': [1],

              'subsample': [0.8],

              'colsample_bytree': [0.7],

              'n_estimators': [10], #number of trees, change it to 1000 for better results

              'missing':[-999],

              'seed': [1337]}

for train, test in sss.split(original_Xtrain, original_ytrain):

    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_xgb) # SMOTE happens during Cross Validation not before..

    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])

    best_est = rand_xgb.best_estimator_

    prediction = best_est.predict(original_Xtrain[test])

    

    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))

    precision_lst.append(precision_score(original_ytrain[test], prediction))

    recall_lst.append(recall_score(original_ytrain[test], prediction))

    f1_lst.append(f1_score(original_ytrain[test], prediction))

    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))

    

print('---' * 45)

print('XGBoost')

print("accuracy: {}".format(np.mean(accuracy_lst)))

print("precision: {}".format(np.mean(precision_lst)))

print("recall: {}".format(np.mean(recall_lst)))

print("f1: {}".format(np.mean(f1_lst)))

print('---' * 45)



# Prediction for XGBoost with SMOTE

labels = ['No Fraud', 'Fraud']

smote_prediction_xgb = best_est.predict(original_Xtest)

print(classification_report(original_ytest, smote_prediction_xgb, target_names=labels))



from sklearn.metrics import confusion_matrix



xgb_cf = confusion_matrix(original_ytest, smote_prediction_xgb)

plot_confusion_matrix(xgb_cf, labels, title="XGB \n Confusion Matrix")
#%% Neural Networks with SMOTE

import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy

from keras.layers import Dropout



n_inputs = original_Xtrain.shape[1]





# Oversampling

n_inputs = Xsm_train.shape[1]



oversample_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(32, activation='relu'),

    Dense(2, activation='softmax')

])

oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)

oversample_predictions = oversample_model.predict(original_Xtest, batch_size=200, verbose=0)

oversample_fraud_predictions = oversample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)



oversample_smote = confusion_matrix(original_ytest, oversample_fraud_predictions)

actual_cm = confusion_matrix(original_ytest, original_ytest)

labels = ['No Fraud', 'Fraud']



fig = plt.figure(figsize=(16,8))



fig.add_subplot(221)

plot_confusion_matrix(oversample_smote, labels, title="OverSampling (SMOTE) \n Confusion Matrix", cmap=plt.cm.Oranges)



fig.add_subplot(222)

plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)    



print('Neural Networks with SMOTE:')

print(classification_report(original_ytest, oversample_fraud_predictions))