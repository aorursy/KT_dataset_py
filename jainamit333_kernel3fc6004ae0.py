##Data Understanding

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import time

from IPython.display import Image



##Data Normalization

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



## Modeling

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import tensorflow as tf



## Model Evaludation

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score
print(tf.__version__)


data = pd.read_csv("../input/pulsar_stars.csv")
print(data.shape)
data.head()
def panda_null_col_evaluation(data):

    return data.columns[data.isna().any()].tolist()

print(panda_null_col_evaluation(data))
data.corr()
data.describe()
# distribution of taeget class

print('Target Class Count', data.target_class.value_counts())



ration = ((int)(data.target_class.value_counts()[0])) / data.shape[0]

print('Negative Class Ration ', ration)

print('Positive Class Ration ', ((int)(data.target_class.value_counts()[1])) / data.shape[0])
plt.hist(data.target_class, bins=2, rwidth=0.8)
data.loc[:, data.columns !=  'target_class'].hist(figsize= [20,20], layout=[4,2])
sns.pairplot(data=data,

             palette="husl",

             hue="target_class",

             vars=[" Mean of the integrated profile",

                   " Excess kurtosis of the integrated profile",

                   " Skewness of the integrated profile",

                   " Mean of the DM-SNR curve",

                   " Excess kurtosis of the DM-SNR curve",

                   " Skewness of the DM-SNR curve"])

plt.suptitle("PairPlot of Data Without Std. Dev. Fields",fontsize=20)



plt.tight_layout()

plt.show() 
plt.figure(figsize=(18,20))



plt.subplot(4,2,1)

sns.violinplot(data=data,y=" Mean of the integrated profile",x="target_class")



plt.subplot(4,2,2)

sns.violinplot(data=data,y=" Mean of the DM-SNR curve",x="target_class")



plt.subplot(4,2,3)

sns.violinplot(data=data,y=" Standard deviation of the integrated profile",x="target_class")



plt.subplot(4,2,4)

sns.violinplot(data=data,y=" Standard deviation of the DM-SNR curve",x="target_class")



plt.subplot(4,2,5)

sns.violinplot(data=data,y=" Excess kurtosis of the integrated profile",x="target_class")



plt.subplot(4,2,6)

sns.violinplot(data=data,y=" Skewness of the integrated profile",x="target_class")



plt.subplot(4,2,7)

sns.violinplot(data=data,y=" Excess kurtosis of the DM-SNR curve",x="target_class")



plt.subplot(4,2,8)

sns.violinplot(data=data,y=" Skewness of the DM-SNR curve",x="target_class")





plt.suptitle("ViolinPlot",fontsize=30)



plt.show()
data_negative = data.loc[data['target_class'] == 0]

data_positive = data.loc[data['target_class'] == 1]

print('Data Negative shape ', data_negative.shape)

print('Data Negative shape ', data_positive.shape)
##Shuffle Data Before Splitting

def shuffle(data_frame):

     return data_frame.reindex(np.random.permutation(data_frame.index))



data_negative = shuffle(data_negative)

data_positive = shuffle(data_positive)
## Split Data

def split_training_and_test(data_frame, training_percentage):

    training_number = data_frame.shape[0] * training_percentage / 100

    test_number = data_frame.shape[0] - training_number

    return data_frame.head(int(training_number)), data_frame.tail(int(test_number))





data_negative_train, data_negative_val = split_training_and_test(data_negative, 80)

data_positive_train, data_positive_val = split_training_and_test(data_positive, 80)
print('Data Positive train', data_positive_train.shape)

print('Data Positive val', data_positive_val.shape)



print('Data Negative train', data_negative_train.shape)

print('Data Negative val', data_negative_val.shape)
data_train = shuffle(pd.concat([data_positive_train, data_negative_train]))

data_val = shuffle(pd.concat([data_positive_val, data_negative_val]))



print('Training Set Shape', data_train.shape)

print('Validation Set Shape', data_val.shape)
def seperate_feature_and_target(data, feature_name):

    return data_train.loc[:, data.columns !=  feature_name], data_train[feature_name] 
X_train, y_train = seperate_feature_and_target(data_train, 'target_class')

X_val, y_val = seperate_feature_and_target(data_val, 'target_class')
min_max_scaler = MinMaxScaler()

standard_scaler = StandardScaler()


X_train_min_max = min_max_scaler.fit_transform(X_train)

X_val_min_max = min_max_scaler.transform(X_val)



X_train_std = standard_scaler.fit_transform(X_train)

X_val_std = standard_scaler.transform(X_val)
def evaluate(model, params, X_train, y_train, X_test, y_test):

    grid_search = GridSearchCV(model, params, n_jobs=-1, cv=3)

    grid_search.fit(X_train, y_train)

    print('Best Params', grid_search.best_params_)

    print("Best Score", grid_search.best_score_)

    prediction = grid_search.best_estimator_.predict(X_test)

    print('Accuracy Score', accuracy_score(y_test, prediction))

    print('Classification  Report', classification_report(y_test, prediction))    
%%time

parameters = {'max_depth':[11,12,13,None], 'min_samples_split':[20,22,25,30], 'criterion':['gini'],

             'n_estimators':[8,9,10,11]}

model = RandomForestClassifier(random_state=42)

evaluate(model, parameters, X_train, y_train, X_val, y_val)
%%time

parameters = {'max_depth':[11,12,13,None], 'min_samples_split':[20,22,25,30], 'criterion':['gini'],

             'n_estimators':[8,9,10,11]}

model = RandomForestClassifier(random_state=42)

evaluate(model, parameters, X_train_min_max, y_train, X_val_min_max, y_val)
%%time

parameters = {'max_depth':[11,12,13,None], 'min_samples_split':[20,22,25,30], 'criterion':['gini'],

             'n_estimators':[8,9,10,11]}

model = RandomForestClassifier(random_state=42)

evaluate(model, parameters, X_train_std, y_train, X_val_std, y_val)
%%time

parameters = {

    "kernel":[ 'linear','rbf']

    }

model = SVC(random_state=42)

evaluate(model, parameters, X_train, y_train, X_val, y_val)
%%time

parameters = {

    "kernel":[ 'linear','rbf']

    }

model = SVC(random_state=42)

evaluate(model, parameters, X_train_min_max, y_train, X_val_min_max, y_val)
%%time

parameters = {

    "kernel":[ 'linear','rbf'],

    "degree":[2]

    }

model = SVC(random_state=42)

evaluate(model, parameters, X_train_std, y_train, X_val_std, y_val)
hidden_units = [3]

batch_size = 10

steps = 2000

model_dir = "/Users/amitjain/personalProjects/Machine-Learning/Complete_Guide_Self/Simple Classification/Complete Practise Of Classification/tf_models_pred"
t0 = time.time()

run = "/without_normalized_1"

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units = hidden_units, n_classes=2, feature_columns= feature_cols,

                                        model_dir=model_dir + run)

dnn_clf.fit(X_train, y_train, batch_size = batch_size, steps = steps)

t1 = time.time()

print("Training Completed in ", t1 - t0)
y_pred = dnn_clf.predict(X_val)

predictions = list(y_pred)
print('Without normalization : time taken ', t1-t0)

print('Accuracy Score', accuracy_score(y_val, predictions))

print('Classification  Report', classification_report(y_val, predictions))  
hidden_units = [3]

batch_size = 10

steps = 2000

model_dir = "/Users/amitjain/personalProjects/Machine-Learning/Complete_Guide_Self/Simple Classification/Complete Practise Of Classification/tf_models_pred"
t0 = time.time()

run = "/with_normalized"

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units = hidden_units, n_classes=2, feature_columns= feature_cols,

                                        model_dir=model_dir + run)

dnn_clf.fit(X_train_std, y_train, batch_size = batch_size, steps = steps)

t1 = time.time()

print("Training Completed in ", t1 - t0)
y_pred = dnn_clf.predict(X_val_std)

predictions = list(y_pred)
print('Without normalization : time taken ', t1-t0)

print('Accuracy Score', accuracy_score(y_val, predictions))

print('Classification  Report', classification_report(y_val, predictions))  
hidden_units = [3]

batch_size = 120

steps = 4000

model_dir = "/Users/amitjain/personalProjects/Machine-Learning/Complete_Guide_Self/Simple Classification/Complete Practise Of Classification/tf_models_pred"

optimizer = tf.train.AdamOptimizer( learning_rate=0.0009,

    beta1=0.9,

    beta2=0.999,

    epsilon=1e-08)
t0 = time.time()

run = "/final"

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units = hidden_units, n_classes=2, feature_columns= feature_cols,

                                        model_dir=model_dir + run,

                                        optimizer=optimizer)

dnn_clf.fit(X_train_std, y_train, batch_size = batch_size, steps = steps)

t1 = time.time()

print("Training Completed in ", t1 - t0)
y_pred = dnn_clf.predict(X_val_std)

predictions = list(y_pred)
print('Without normalization : time taken ', t1-t0)

print('Accuracy Score', accuracy_score(y_val, predictions))

print('Classification  Report', classification_report(y_val, predictions))  