#loading libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, accuracy_score

import seaborn as sb

from imblearn.over_sampling import SMOTE, SMOTENC

import tensorflow as tf

from tensorflow import keras

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential, Model

from keras.layers import Dense, BatchNormalization, GaussianNoise

from keras.callbacks import EarlyStopping

from keras import regularizers, Input

from keras import backend
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#loading PIMA dataset

pima_df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv", delimiter = ',', header = 0)
#preview of data

pima_df.head(10)
#exploring datatypes and non-null value count

pima_df.info()
# Computing descriptive statistics

pima_df.describe()
# bivariate scatterplots of each combination of features

sb.pairplot(pima_df, hue = 'Outcome', diag_kind = 'hist')
# calculating column medians and storing as a dictionary

column_medians = pima_df.median().to_dict()
# # replacing zero values with column medians from column_medians dictionary

pima_df = pima_df.replace({

    'Glucose': {0: column_medians['Glucose']},

    'BloodPressure': {0: column_medians['BloodPressure']},

    'BMI': {0: column_medians['BMI']},

    'SkinThickness': {0: column_medians['SkinThickness']}

                           })
#calculating correlation coefficients for the features

pima_corr = pima_df.corr()



#creating mask to hide upper triangular part of matrix (mirror of bottom triangle)

mask = np.zeros_like(pima_corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



#plotting matrix using seaborn

sb.heatmap(pima_corr, annot = True, mask = mask)
#exploring the target feature

pima_df.iloc[:, 8].value_counts(normalize = True)
#defining predictor features dataframe

X = pima_df.drop(columns = 'Outcome')
#Obtaining indicator of categorical columns

categorical_columns = np.where(((X.columns=='Pregnancies') | (X.columns == 'Age')), True, False)
#defining target feature

y = pima_df['Outcome']
#converting predictor and target feature to numpy array to feed into model.

X = X.values

y = y.values
#splitting data to train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify = y, random_state = 0)
def f1_threshold(threshold=0.5):

    """ 

    Given a threshold value, this function will take the class probabilities output by a model, apply the threshold as a decision boundary to classify the result as positive or

    negative class, then compute the f1-score between the predicted classes and the true classes. The default boundary is 0.5. Changing the threshold will make 

    class probabilities above the threshold belong to the positive class and below the threshold belong to the negative class.

    

    """

    def f1(y_true, y_pred):

        threshold_value = threshold

        

        y_true = backend.cast(backend.greater_equal(backend.clip(y_true, 0, 1), threshold_value), 'int32')

        y_pred = backend.cast(backend.greater_equal(backend.clip(y_pred, 0, 1), threshold_value), 'int32')



        true_positives = backend.cast(backend.sum(y_true*y_pred), 'float32') + backend.epsilon()

        false_positives = backend.cast(backend.shape(y_true[(y_true==0) & (y_pred==1)])[0], 'float32') + backend.epsilon()

        

        false_negatives = backend.cast(backend.shape(y_true[(y_true==1) & (y_pred==0)])[0], 'float32') + backend.epsilon()



        precision = true_positives/(true_positives + false_positives)

        recall = true_positives/(true_positives + false_negatives)



        f = 2*precision*recall/(precision + recall)

        return f

    return f1
def precision_threshold(threshold=0.5):

    """ 

    Given a threshold value, this function will take the class probabilities output by a model, apply the threshold as a decision boundary to classify the result as positive or

    negative class, then compute the precision score between the predicted classes and the true classes. The default boundary is 0.5. Changing the threshold will make 

    class probabilities above the threshold belong to the positive class and below the threshold belong to the negative class.

    

    """

    def precision(y_true, y_pred):

        threshold_value = threshold

        

        y_true = backend.cast(backend.greater_equal(backend.clip(y_true, 0, 1), threshold_value), 'int32')

        y_pred = backend.cast(backend.greater_equal(backend.clip(y_pred, 0, 1), threshold_value), 'int32')

        

        true_positives = backend.cast(backend.sum(y_true*y_pred), 'float32') + backend.epsilon()

        false_positives = backend.cast(backend.shape(y_true[(y_true==0) & (y_pred==1)])[0], 'float32') + backend.epsilon()



        precision = true_positives/(true_positives + false_positives)



        return precision

    

    return precision
def recall_threshold(threshold=0.5):

    """ 

    Given a threshold value, this function will take the class probabilities output by a model, apply the threshold as a decision boundary to classify the result as positive or

    negative class, then compute the recall score between the predicted classes and the true classes. The default boundary is 0.5. Changing the threshold will make 

    class probabilities above the threshold belong to the positive class and below the threshold belong to the negative class.

    

    """

    def recall(y_true, y_pred):

        threshold_value = threshold

        

        y_true = backend.cast(backend.greater_equal(backend.clip(y_true, 0, 1), threshold_value), 'int32')

        y_pred = backend.cast(backend.greater_equal(backend.clip(y_pred, 0, 1), threshold_value), 'int32')

        

        true_positives = backend.cast(backend.sum(y_true*y_pred), 'float32') + backend.epsilon()

        false_negatives = backend.cast(backend.shape(y_true[(y_true==1) & (y_pred==0)])[0], 'float32') + backend.epsilon()



        recall = true_positives/(true_positives + false_negatives)

    

        return recall

    return recall
def fn_threshold(threshold=0.5):

    """ 

    Given a threshold value, this function will take the class probabilities output by a model, apply the threshold as a decision boundary to classify the result as positive or

    negative class, then compute (1 - false-negative fraction) between the predicted classes and the true classes. The default boundary is 0.5. Changing the threshold will make 

    class probabilities above the threshold belong to the positive class and below the threshold belong to the negative class.

    

    """

    def fn(y_true, y_pred):



        threshold_value = threshold

        

        y_true = backend.cast(backend.greater_equal(backend.clip(y_true, 0, 1), threshold_value), 'int32')

        y_pred = backend.cast(backend.greater_equal(backend.clip(y_pred, 0, 1), threshold_value), 'int32')



        false_negatives = backend.cast(backend.shape(y_true[(y_true==1) & (y_pred==0)])[0], 'float32') + backend.epsilon()

        total = backend.cast(backend.shape(y_true)[0], 'float32') + backend.epsilon()



        false_negative_fraction = false_negatives/total

        

        return (1-false_negative_fraction)

    

    return fn
#initializing MLN Network

def create_network(THRESH = 0.5, weighted = False):

    """

    This function creates a pre-defined Keras multi-layer neural network and compiles it. The THRESH input to this function enables the network to compute metrics based on different

    decision boundaries for the sigmoid function in the output layer. The default boundary is 0.5. Changing the threshold will make class probabilities above the threshold belong to the

    positive class and below the threshold belong to the negative class.

    """

    model = Sequential([

        BatchNormalization(),

        GaussianNoise(0.01),

        Dense(10, activation = 'elu', input_dim = 8),

        BatchNormalization(),

        Dense(4, activation = 'elu'),

        BatchNormalization(),

        Dense(1, activation = 'sigmoid')])

    #compiling model with scoring parameters and optimizer

    metrics_list = ['accuracy', precision_threshold(THRESH), recall_threshold(THRESH), f1_threshold(THRESH), fn_threshold(THRESH)]

    

    if weighted == True:

        weighted_metrics_list = metrics_list

    else:

        weighted_metrics_list = None



    model.compile(

                  loss = 'binary_crossentropy', 

                  optimizer = keras.optimizers.Adam(1e-4),

                  metrics = metrics_list,

                  weighted_metrics = weighted_metrics_list

                 )

    return model
#defining simple model

y_pred = np.zeros(X_test.shape[0])
print('Accuracy of simple model: {:.2f}'.format(accuracy_score(y_pred, y_test)))
def cross_validate(x, y, model, k = 10, oversample = True, weight_ratio = None):

    """

    This function splits the data set aside for training into 'k' stratified folds (10 default) for training and validation. It will train a neural network model passed to it

    on the training folds and evaluate on the testing fold for all k splits. Early stopping is implemented to avoid long periods of no improvement in validation loss. The function outputs

    a dictionary of folds. For each fold a dictionary of metrics and the corresponding score is output.

    """

    kv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

    

    scores_train = {}

    scores_val = {}



    

    fold = 0

    for train, val in kv.split(x, y):

        fold += 1

        

        print('Working on fold:{}'.format(fold))

        

        x_train, y_train = x[train], y[train]

        x_val, y_val = x[val], y[val]

        

        if oversample == True:

            smt = SMOTENC(categorical_features = categorical_columns)

            x_train, y_train = smt.fit_sample(x_train, y_train)

        

        es = EarlyStopping(monitor='val_loss', mode='min', patience = 20)

        

        if weight_ratio == None:

            class_weight = None

        else:    

            class_weight = {0: 1*weight_ratio, 1: 1}

            

        model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 50, 

                  batch_size = 32, class_weight = class_weight, callbacks = [es], verbose = 0)

        

        scores_train['fold' + str(fold)] = dict(zip(model.metrics_names, model.evaluate(x_train, y_train)))

        scores_val['fold' + str(fold)] = dict(zip(model.metrics_names, model.evaluate(x_val, y_val)))

        if fold == k:

            break

    

    return (scores_train, scores_val)

        
#creating a mode instance

network = create_network()
#applying cross-validation

cvs = cross_validate(x = X_train, y = y_train, model = network)
#converting results into dataframe for easy-read

cvs_train_df = pd.DataFrame(cvs[0])

cvs_val_df = pd.DataFrame(cvs[1])
#computing means across training folds

cvs_train_df.mean(axis = 1).round(2)
#computing standard error for mean calculations across training folds

cvs_train_df.sem(axis= 1).round(2)
#computing means calculations across validation folds

cvs_val_df.mean(axis = 1).round(2)
#computing standard error for mean calculations across validation folds

cvs_val_df.sem(axis = 1).round(2)
#predicting on the test set

y_pred = network.predict_classes(X_test)
#calculating confusion matrix

cm = confusion_matrix(y_test, y_pred)

cm
#computing the accuracy between the prediction and the actual values

print('Accuracy score for test set: {:.2f}'.format(accuracy_score(y_pred, y_test)))

#computing the recall score between the prediction and actual values

print('Recall score for test set: {:.2f}'.format(recall_score(y_pred, y_test)))

#computing the precision score between the prediction and the actual values

print('Precision score for test set: {:.2f}'.format(precision_score(y_pred, y_test)))

#computing the f1-score score between the prediction and the actual values

print('F1-score for test set: {:.2f}'.format(f1_score(y_pred, y_test)))

#computing 1 minus the false_negative_fraction between the prediction and the actual values

print('1-FalseNegativeFraction for test set: {:.2f}'.format(1-cm[1, 0]/np.sum(cm)))
#heatmap of confusion matrix

sb.heatmap(cm, annot = True)
#defining threshold search space

threshold_list = np.linspace(0, 1, 101)
#evaluating model at various thresholds

y_prob = network.predict(X_test)

metrics = {}

for t in threshold_list:

    t = np.round(t, 3)

    

    y_thresh = np.array([1 if i>=t else 0 for i in y_prob])

    

    cm_thresh = confusion_matrix(y_test, y_thresh)

    

    metrics[str(t)] = {}

#   computing the accuracy between the prediction and the actual values

    metrics[str(t)]['accuracy'] = accuracy_score(y_thresh, y_test)

    #computing the recall score between the prediction and actual values

    metrics[str(t)]['recall'] = recall_score(y_thresh, y_test)

    #computing the precision score between the prediction and the actual values

    metrics[str(t)]['precision'] = precision_score(y_thresh, y_test)

    #computing the f1-score score between the prediction and the actual values

    metrics[str(t)]['f1-score'] = f1_score(y_thresh, y_test)

    #computing 1 minus the false_negative_fraction between the prediction and the actual values

    metrics[str(t)]['fn'] = 1-(cm_thresh[1, 0]/np.sum(cm_thresh))
#plotting metrics for various thresholds

pd.DataFrame(metrics).T.plot()

plt.xlabel('Thresholds')

plt.ylabel('Metric Score')
#defining new model

network_weighted = create_network(weighted = True)

#testing a weight ratio of 1:2 (equivalent to 0.5: 1)

cv_weights = cross_validate(X_train, y_train, network_weighted, weight_ratio = 0.50, oversample = True)
#results for training set

pd.DataFrame(cv_weights[0]).mean(axis = 1)
#result for validation set

pd.DataFrame(cv_weights[1]).mean(axis = 1)
#predicted classes with weights

y_weighted = network_weighted.predict_classes(X_test)
#calculating confusion matrix

cm_weighted = confusion_matrix(y_test, y_weighted)

cm_weighted
sb.heatmap(cm_weighted, annot = True)