# import packages

import pandas as pd

import numpy as np

from scipy import stats

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, precision_recall_curve

from sklearn.metrics import recall_score, classification_report, auc, roc_curve, classification_report, confusion_matrix

from sklearn.metrics import precision_recall_fscore_support, f1_score

from sklearn.preprocessing import StandardScaler, scale, MinMaxScaler

from tensorflow import keras

from keras.models import Model, load_model, Sequential

from keras.layers import Input, Dense, Dropout

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

from keras import regularizers, backend

sb.set(style='white', font_scale=1.75)
#loading data from file

credit_df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
#exploring data types and non-null values

credit_df.info()
#computing descriptive statistics

credit_df.describe()
#preview of 10 rows of data

credit_df.head(10)
#exploring target variable distribution

credit_df['Class'].value_counts(normalize = True)
#checking for duplicates

credit_df.duplicated().any()
#finding out how many duplicates

credit_df.duplicated().value_counts()
#dropping duplicate rows

credit_df = credit_df.drop_duplicates()
#converting time to hours

credit_df['Time'] = credit_df['Time']/3600
#making a copy before scaling

X = credit_df.copy()
#scaling

X['Time'] = scale(X['Time'])

X['Amount'] = scale(X['Amount'])
#splitting training and testing

X_train, X_test = train_test_split(X, test_size = 0.5, stratify = X['Class'], random_state = 123)
#splitting testing and validation

X_test, X_val = train_test_split(X_test, test_size = 0.6, stratify = X_test['Class'], random_state = 123)
#isolating the 'normal' class examples from the training set

X_train = X_train[X_train['Class']==0]

#dropping the label column

X_train = X_train.drop(columns = 'Class')
#isolating target variable and predictor features for validation

y_val = X_val['Class']

X_val = X_val.drop(columns = 'Class')

#repeating for testing

y_test = X_test['Class']

X_test = X_test.drop(columns = 'Class')
#defining model parameters

INPUT_DIM = X_train.shape[1]

ENCODING_DIM = 15

HIDDEN_DIM = 7

SPARSITY = 1e-7
#constructing model

input_layer = Input(shape=(INPUT_DIM, )) 

encoder = Dense(ENCODING_DIM, activation = "elu", kernel_regularizer = regularizers.l1(SPARSITY))(input_layer)

encoder = Dense(HIDDEN_DIM, activation = "tanh")(encoder)

decoder = Dense(ENCODING_DIM, activation = "elu")(encoder)

output_layer = Dense(INPUT_DIM, activation = 'elu')(decoder)

autoencoder = Model(inputs=input_layer, outputs = output_layer)
#compiling model

autoencoder.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
#defining early stopping condition

es = EarlyStopping(monitor = 'val_loss', patience = 10)
#fitting model to training data (for input and output!)

BATCH_SIZE = 128

EPOCHS = 100



history = autoencoder.fit(X_train, X_train, validation_data = (X_val, X_val), epochs = EPOCHS,

                         batch_size = BATCH_SIZE, shuffle = True, callbacks = [es])
#plotting model history

pd.DataFrame(history.history).plot(figsize=(8, 5))

plt.grid(True)

plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]

plt.xlabel('Epoch')

plt.ylabel('Metrics')

plt.title('Metrics vs Training Epochs')
#reconstructing validation data

pred = autoencoder.predict(X_val)
# pd.DataFrame(normal_error).median()

error[y_val==0].describe()
# pd.DataFrame(fraud_error).median()

error[y_val==1].describe()
#plotting the reconstruction error distribution for the validation set (by class)

plt.figure(figsize = (20, 10));

plt.hist(error[y_val==0].values, bins = 200, range = (0, 150), density = True, alpha = 1, label = 'normal');

plt.hist(error[y_val==1].values, bins = 200, density = True, alpha = 0.5, label = 'fraud');

plt.legend();

plt.xlim((0, 145));

plt.xticks(np.arange(0, 150, 5), rotation = 45);

plt.grid()

plt.xlabel('Mean Squared Error');

plt.ylabel('Frequency');

plt.title('Reconstruction Error Distribution of Validation Set Across Classes');
#setting threshold

threshold = 4
#reconstructing the test data with the autoencoder

X_pred = autoencoder.predict(X_test)
#calculating reconstruction error

error_X = np.mean(np.power(X_test - X_pred, 2), axis = 1)
#making prediction by comparing reconstruction error to threshold

y_pred = [1 if x>=threshold else 0 for x in error_X]
print(classification_report(y_test, y_pred))
labels = ['Normal', 'Fraud']

plt.figure(figsize = (20, 10))

cm = confusion_matrix(y_test, y_pred)

sb.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt = 'g', annot_kws={"size": 20})

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()
#plotting Receiver Operating Characteristic Curve

false_pos_rate, true_pos_rate, thresholds = roc_curve(y_test, error_X)

roc_auc = auc(false_pos_rate, true_pos_rate,)



plt.figure(figsize = (20, 10))

plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)

plt.plot([0,1],[0,1], linewidth=5)

plt.xlim([-0.01, 1])

plt.ylim([0, 1.01])

plt.legend(loc='lower right')

plt.title('Receiver Operating characteristic curve (ROC)')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
#Plotting recall vs precision

plt.figure(figsize = (20, 10))

precision_rt, recall_rt, threshold_rt = precision_recall_curve(y_test, error_X)

plt.plot(recall_rt, precision_rt, linewidth=5, label='Precision-Recall curve')

plt.title('Recall vs Precision')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.show()
#plotting Precision-Recall vs Threshold

plt.figure(figsize = (20, 10))

plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)

plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)

plt.title('Precision and Recall versus Threshold values')

plt.xlabel('Threshold')

plt.ylabel('Precision/Recall')

plt.legend()

plt.show()