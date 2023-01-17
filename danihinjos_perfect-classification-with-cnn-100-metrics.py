import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import savgol_filter
from  scipy import ndimage
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, normalize 
from scipy.ndimage.filters import uniform_filter1d
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam, SGD
from keras import metrics, regularizers
from keras.callbacks import EarlyStopping
from keras import backend as K
import keras
import seaborn as sn
import math
import timeit
import tensorflow as tf
rutatrain = '../input/kepler-labelled-time-series-data/exoTrain.csv'
rutatest = '../input/kepler-labelled-time-series-data/exoTest.csv'
extrain_df = pd.read_csv(rutatrain)
extest_df = pd.read_csv(rutatest)

# Separate the label from the rest of attributes
Y_train = extrain_df.LABEL
X_train = extrain_df.drop('LABEL',axis=1)
Y_test = extest_df.LABEL
X_test = extest_df.drop('LABEL',axis=1)
# Applying Fast Fourier Transform (FFT)
def fourier_transform(df):
    df_fft = np.abs(np.fft.fft(df, axis=1))
    return df_fft

# Handling upper outliers: personalized function
def reduce_upper_outliers(df,reduce = 0.01, half_width=4):
    length = len(df.iloc[0,:])
    remove = int(length*reduce)
    for i in df.index.values: #Para cada muestra
        values = df.loc[i,:]
        sorted_values = values.sort_values(ascending = False)
        for j in range(remove): 
            idx = sorted_values.index[j]
            new_val = 0
            count = 0
            idx_num = int(idx[5:])
            for k in range(2*half_width+1):
                idx2 = idx_num + k - half_width
                if idx2 <1 or idx2 >= length or idx_num == idx2:
                    continue
                new_val += values['FLUX.'+str(idx2)]

                count += 1
            new_val /= count # count will always be positive here
            if new_val < values[idx]: # just in case there's a few persistently high adjacent values
                df.at[i,idx] = new_val
    return df

def iterar_ruo(df, n=2):
    for i in range(n): 
        df2 = reduce_upper_outliers(df)
    return df2

# Handling outliers: Smoothing filters
def apply_filter(df,filternumber):
    #UNIFORM FILTER == 0
    if filternumber == 0:
        filt = uniform_filter1d(df, axis=1, size=50)
    #GAUSSIAN FILTER == 1
    elif filternumber == 1:
        filt = ndimage.filters.gaussian_filter(df, sigma=10)
    #Savitzky-Golay FILTER == 2
    elif filternumber == 2:
        filt = savgol_filter(df,21,4,deriv=0)

    return filt

# Normalizing data
def apply_normalization(df_train, df_test, nnumber):
    #MinMax Scaler
    if nnumber == 0:
        scaler = MinMaxScaler()
        norm_train = scaler.fit_transform(df_train)
        norm_test = scaler.transform(df_test)
    #Normalize
    elif nnumber == 1:
        norm_train = normalize(df_train)
        norm_test = normalize(df_test)
    #Robust Scaler
    elif nnumber == 2:
        scaler = RobustScaler()
        norm_train = scaler.fit_transform(df_train)
        norm_test = scaler.transform(df_test)
        
    
    norm_train = pd.DataFrame(norm_train)
    norm_test = pd.DataFrame(norm_test)
    return norm_train, norm_test

# Standardizing data
def apply_standarization(df_train, df_test):
    scaler = StandardScaler()
    norm_train = scaler.fit_transform(df_train)
    norm_test = scaler.transform(df_test)
    
    norm_train = pd.DataFrame(norm_train)
    norm_test = pd.DataFrame(norm_test)
    return norm_train, norm_test

class dataProcessor:

    def __init__(self, outlier=False, smoothing=False, fourier=False, normalize=False, standardize=False):
        self.outlier = outlier
        self.smoothing = smoothing
        self.normalize = normalize
        self.standardize = standardize
        self.fourier = fourier
    
    def process(self, df_train_x, df_test_x):
        
        # Handling outliers
        if self.outlier:
            print("Removing upper outliers...")
            df_train_x = iterar_ruo(df_train_x, P_OUTLIERS)
            df_test_x = iterar_ruo(df_test_x, P_OUTLIERS)
            
        # Apply fourier transform
        if self.fourier:
            print("Applying Fourier...")
            df_train_x = fourier_transform(df_train_x)
            df_test_x = fourier_transform(df_test_x)
        
        # Applying smoothing filters
        if self.smoothing:
            print("Applying smoothing filter...")
            df_train_x = pd.DataFrame(apply_filter(df_train_x, FILTER_NUMBER))
            df_test_x = pd.DataFrame(apply_filter(df_test_x, FILTER_NUMBER))
            
        # Normalization
        if self.normalize:
            print("Normalizing...")
            df_train_x, df_test_x = apply_normalization(df_train_x, df_test_x, TIPO_NORM)
            
        # Normalization
        if self.standardize:
            print("Standardizing...")
            df_train_x, df_test_x = apply_standarization(df_train_x, df_test_x)
        

        print("Finished Processing!")
        return df_train_x, df_test_x
# Change the value of P_OUTLIERS to apply reduce_upper_outliers over a specific percentage of the values of each sample
# Change the value of FILTER_NUMBER to apply a specific smoothing filter
    # FILTER_NUMBER == 0 -> UNIFORM FILTER
    # FILTER_NUMBER == 1 -> GAUSSIAN FILTER
    # FILTER_NUMBER == 2 -> SAVITZKY-GOLAY FILTER
# Change the value of TIPO_NORM to apply a specific normalization method
    # TIPO_NORM == 0 -> MaxMinScaler
    # TIPO_NORM == 1 -> normalize
    # TIPO_NORM == 2 -> RobustScaler
    
P_OUTLIERS = 2
FILTER_NUMBER = 1
TIPO_NORM = 1

Processor = dataProcessor(
    outlier = True,
    fourier = True,
    smoothing = True,
    normalize= True,
    standardize= True
)

df_train_x = X_train.copy()
df_test_x = X_test.copy()
df_train_x, df_test_x = Processor.process(df_train_x, df_test_x)
from sklearn.utils import resample
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


# UPSAMPLING: RandomOverSampler
X_total = df_train_x.copy()
os =  RandomOverSampler(sampling_strategy='minority')
upsampling, upsampling_Y = os.fit_sample(X_total, Y_train)

print ("Distribution before resampling {}".format(Counter(Y_train)))
print ("Distribution labels after resampling {}".format(Counter(upsampling_Y)))

upsampling_Y.value_counts().plot(kind='bar', title='Count (target)');
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def specificity_m(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    specificity = true_negatives / (possible_negatives + K.epsilon())
    return specificity

def imb_accuracy(y_true, y_pred):
    recall = recall_m(y_true, y_pred)
    specificity = specificity_m(y_true, y_pred)
    score = (0.5 * recall) + (0.5 * specificity)
    return score

# Transformation of train labels to OneHotEncoder format
def transform_Y(y_samp, y_sampTest):
    y_samp[y_samp < 2] = 0
    y_samp[y_samp > 1] = 1
    y_sampTest[y_sampTest < 2] = 0
    y_sampTest[y_sampTest > 1] = 1
    Y_train_ohe = np.zeros((y_samp.shape[0], 2))

    for i in range(y_samp.shape[0]):
        Y_train_ohe[i, int(y_samp[i])] = 1
        
    return Y_train_ohe, y_sampTest
# Crear y configurar CNN
def create_model(X_samp, activation='relu', learn_rate=0.01):
    model = Sequential()
    model.add(Conv1D(filters = 16, input_shape = (X_samp.shape[1],1), kernel_size=(3), activation = activation, kernel_regularizer='l2', padding='same'))
    model.add(MaxPooling1D(pool_size = 2, strides = 2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(32, activation = activation, kernel_regularizer='l2'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = "sigmoid", kernel_regularizer='l2'))
    
    optimizer = Adam(lr=learn_rate)
    #Adam is a optimized version of a SGD (Stochastic Gradient Descendant) optimizer.
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[imb_accuracy])
    #binary_crossentropy is the go-to loss function for classification tasks, either balanced or imbalanced. 
    #It is the first choice when no preference is built from domain knowledge yet.
    return model
    
EPOCHS = 50
VAL_SPLIT = 0.2
BATCH_SIZE = 75
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
Test_Y = Y_test.copy()
X_sampTest = df_test_x.copy()
X_sampTest = np.expand_dims(X_sampTest, axis=2)
upsamplingg_Y = upsampling_Y.copy()
X_sampUp = upsampling.copy()
X_sampUp = np.expand_dims(X_sampUp, axis=2)

# Creation and training of the NN
modelUp = create_model(X_sampUp)
Y_train_ohe, Test_Y = transform_Y(upsamplingg_Y, Test_Y)
start_time_train = timeit.default_timer()
baseline_historyUp = modelUp.fit(X_sampUp, Y_train_ohe, validation_split = VAL_SPLIT, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, callbacks=[early_stop])
upsampling_elapsed = timeit.default_timer() - start_time_train
print('Training time: ' + str(upsampling_elapsed))


# Representation of the NN arquitecture
print(modelUp.summary())

# Prediction
y_test_pred = modelUp.predict_classes(np.array(X_sampTest))
y_scores = modelUp.predict_proba(X_sampTest)[:,1]

# Confussion matrix and classification_report
print(classification_report(Test_Y, y_test_pred))

print('Confussion matrix') 
matrix = confusion_matrix(Test_Y, y_test_pred)
df_cm = pd.DataFrame(matrix, columns=np.unique(Test_Y), index = np.unique(Test_Y))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4) 
sn.heatmap(df_cm, cmap="PuRd", annot=True,annot_kws={"size": 16})
TN = matrix[0][0]
FP = matrix[0][1]
FN = matrix[1][0]
TP = matrix[1][1]

# Prediction metrics
upsampling_accuracy = accuracy_score(Test_Y, y_test_pred)
upsampling_imbaccuracy = balanced_accuracy_score(Test_Y, y_test_pred)
upsampling_precision = precision_score(Test_Y, y_test_pred)
upsampling_recall = recall_score(Test_Y, y_test_pred)
upsampling_f1 = f1_score(Test_Y, y_test_pred)
upsampling_auc = roc_auc_score(Test_Y, y_scores)
upsampling_especifidad = TN / (TN + FP)

print('\t\t Upsampling\n')
print("Accuracy:\t", "{:0.10f}".format(upsampling_accuracy))
print("Precision:\t", "{:0.10f}".format(upsampling_precision))
print("Recall:\t\t", "{:0.10f}".format(upsampling_recall))
print("Specificity:\t", "{:0.10f}".format(upsampling_especifidad))
print("\nF1 Score:\t", "{:0.10f}".format(upsampling_f1))
print("ROC AUC:\t", "{:0.10f}".format(upsampling_auc))
print("Balanced\nAccuracy:\t", "{:0.10f}".format(upsampling_imbaccuracy))
print("Training time:\t", "{:0.2f}s".format(upsampling_elapsed))
