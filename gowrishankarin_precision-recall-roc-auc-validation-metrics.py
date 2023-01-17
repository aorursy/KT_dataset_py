import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split





import os

print(os.listdir("../input"))



train_df = pd.read_csv('../input/train.csv')



y_train = train_df.label.values

X_train = train_df.drop(columns=["label"]).values

X_train = X_train / 255.0



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.4, random_state=42)

print(X_train.shape, X_test.shape)

print(train_df.shape)
import tensorflow as tf

print(tf.__version__)



from sklearn.metrics import accuracy_score
def incomplete_model(X, y, X_test, epochs=1):

    model = tf.keras.models.Sequential([

        tf.keras.layers.Flatten(input_shape=(784,)),

        #tf.keras.layers.Dense(256, activation=tf.nn.relu),

        #tf.keras.layers.Dense(128, activation=tf.nn.relu),

        tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(X, y, epochs=epochs, verbose=0)

    predictions = model.predict(X_test)

    return np.argmax(predictions, axis=1)



classifications = incomplete_model(X_train, y_train, X_test, 2) 
binary_classifications = np.absolute(y_test - classifications)

binary_classifications = np.clip(binary_classifications, 0, 1)

accuracy = 1 - (np.sum(binary_classifications) / y_test.shape[0])

print(accuracy)

NUM_TO_ANALYSE = 9

SAMPLE_COUNT = 3000



sample_indices = np.random.randint(low=0, high=y_test.shape[0], size=SAMPLE_COUNT)



actual_sample = y_test[sample_indices]

predicted_sample = classifications[sample_indices]
def confusion_matrix(predicted, actual, klass=NUM_TO_ANALYSE):

    

    actual_indices = np.where(actual == klass)[0]   

    predicted_indices = np.where(predicted == klass)[0]    

    not_in_predicted_indices = np.where(predicted != klass)[0]    

    not_in_actual_indices = np.where(actual != klass)[0]

    

    # True Positives: TPs are count of rightly predicted 

    true_positive_indices = np.where(actual[predicted_indices] == klass)[0]

    TRUE_POSITIVES = len(true_positive_indices)

    

    # False Positives: Failed to predict correctly

    false_positive_indices = np.where(actual[predicted_indices] != klass)[0]

    FALSE_POSITIVES = len(false_positive_indices)

    

    # True Negatives: Predicted as not part of the class and they are true in the actuals

    true_negative_indices = predicted[not_in_actual_indices]

    TRUE_NEGATIVES = len(np.where(true_negative_indices != klass)[0])

    

    # False Negatives: False negatives are not predicted as the class of interest but they are actually belongs to the class

    false_negative_indices = actual[not_in_predicted_indices]

    FALSE_NEGATIVES = len(np.where(false_negative_indices == klass)[0])

    

    return {'TP': TRUE_POSITIVES, 'FP': FALSE_POSITIVES, 'TN': TRUE_NEGATIVES, 'FN': FALSE_NEGATIVES}





metrices = confusion_matrix(predicted_sample, actual_sample, NUM_TO_ANALYSE)



print(metrices['TP'], metrices['FP'], metrices['TN'], metrices['FN'])

print(sum([metrices['TP'], metrices['FP'], metrices['TN'], metrices['FN']]))
class Metrics:

    metrics = None

    precision = None

    recall = None

    f1_score = None

    tpr = None

    fpr = None

    

    def __init__(self, metrics):

        self.metrics = metrics

        

    def calculate(self):

        self.precision = self.metrics['TP'] / (self.metrics['TP'] + self.metrics['FP'])

        self.recall = self.metrics['TP'] / (self.metrics['TP'] + self.metrics['FN'])

        self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)

        self.tpr = self.metrics['TP'] / (self.metrics['TP'] + self.metrics['FN'])

        self.fpr = self.metrics['FP'] / (self.metrics['FP'] + self.metrics['TN'])

        

metrics = Metrics(metrices)

metrics.calculate()

print(metrics.precision)

def precision(metrics):

    return metrics['TP'] / (metrics['TP'] + metrics['FP'])



metrices['precision'] = precision(metrices)

metrices['precision']
def recall(metrices):

    return metrices['TP'] / (metrices['TP'] + metrices['FN'])



metrices['recall'] = recall(metrices)

metrices['recall']
def f1_score(metrices):

    return 2 * metrices['precision'] * metrices['recall'] / (metrices['precision'] + metrices['recall'])



metrices['f1_score'] = f1_score(metrices)

metrices['f1_score']
def tpr(metrics):

    return metrics['TP'] / (metrics['TP'] + metrics['FN'])



metrices['TPR'] = tpr(metrices)

metrices['TPR']
def fpr(metrics):

    return metrics['FP'] / (metrics['FP'] + metrics['TN'])



metrices['FPR'] = fpr(metrices)

metrices['FPR']
def run(iterations=5):    

    outcomes = {'precision': [], 'recall': [], 'f1_score':[], 'tpr': [], 'fpr': []}

    

        

    for index in np.arange(1,iterations):

        print("Iternation {0}".format(index))

        classification = incomplete_model(X_train, y_train, X_test, index)

        sample_indices = np.random.randint(low=0, high=y_test.shape[0], size=index * 50)



        actual_sample = y_test[sample_indices]

        predicted_sample = classifications[sample_indices]



        metrices = confusion_matrix(predicted_sample, actual_sample)

        metrics = Metrics(metrices)

        metrics.calculate()

        outcomes['precision'].append(metrics.precision)

        outcomes['recall'].append(metrics.recall)

        outcomes['f1_score'].append(metrics.f1_score)

        outcomes['tpr'].append(metrics.tpr)

        outcomes['fpr'].append(metrics.fpr)

    return outcomes

    

outcomes = run(15)
df = pd.DataFrame(data=outcomes)

df
import matplotlib.pyplot as plt

from PIL import Image



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 9999

pd.options.display.float_format = '{:20, .2f}'.format
df = df.sort_values(by ='recall')



trace = go.Scatter(x = df.recall, y = df.precision)

data = [trace]

py.iplot(data, filename='AUC')
df = df.sort_values(by ='fpr')

 

trace = go.Scatter(x = df.fpr, y = df.tpr)

data = [trace]

py.iplot(data, filename='ROC')