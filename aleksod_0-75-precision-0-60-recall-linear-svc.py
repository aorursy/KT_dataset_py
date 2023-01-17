import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import signal

from  scipy import ndimage

import matplotlib

from scipy import stats

import warnings

warnings.filterwarnings('ignore')
extrain = pd.read_csv('../input/exoTrain.csv')

extest = pd.read_csv('../input/exoTest.csv')
extrain.head()
# Obtaining flux for several stars without exoplanets from the train data:

for i in [0, 999, 1999, 2999, 3999, 4999]:

    flux = extrain[extrain.LABEL == 1].drop('LABEL', axis=1).iloc[i,:]

    time = np.arange(len(flux)) * (36.0/60.0) # time in units of hours

    plt.figure(figsize=(15,5))

    plt.title('Flux of star {} with no confirmed exoplanets'.format(i+1))

    plt.ylabel('Flux, e-/s')

    plt.xlabel('Time, hours')

    plt.plot(time, flux)
# Obtaining flux for several stars without exoplanets from the train data:

for i in [0, 9, 14, 19, 24, 29]:

    flux = extrain[extrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]

    time = np.arange(len(flux)) * (36.0/60.0) # time in units of hours

    plt.figure(figsize=(15,5))

    plt.title('Flux of star {} with confirmed exoplanets'.format(i+1))

    plt.ylabel('Flux, e-/s')

    plt.xlabel('Time, hours')

    plt.plot(time, flux)
i = 13

flux1 = extrain[extrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]

time = np.arange(len(flux1)) * (36.0/60.0) # time in units of hours

plt.figure(figsize=(15,5))

plt.title('Original flux of star {} with confirmed exoplanets'.format(i+1))

plt.ylabel('Flux, e-/s')

plt.xlabel('Time, hours')

plt.plot(time, flux1)
i = 13

flux2 = ndimage.filters.gaussian_filter(flux1, sigma=10)

time = np.arange(len(flux2)) * (36.0/60.0) # time in units of hours

plt.figure(figsize=(15,5))

plt.title('Smoothed flux of star {} with confirmed exoplanets'.format(i+1))

plt.ylabel('Flux, e-/s')

plt.xlabel('Time, hours')

plt.plot(time, flux2)
i = 13

flux3 = flux1 - flux2

time = np.arange(len(flux3)) * (36.0/60.0) # time in units of hours

plt.figure(figsize=(15,5))

plt.title('Detrended flux of star {} with confirmed exoplanets'.format(i+1))

plt.ylabel('Flux, e-/s')

plt.xlabel('Time, hours')

plt.plot(time, flux3)
i = 13

flux3normalized = (flux3-np.mean(flux3))/(np.max(flux3)-np.min(flux3))

time = np.arange(len(flux3normalized)) * (36.0/60.0) # time in units of hours

plt.figure(figsize=(15,5))

plt.title('Normalized detrended flux of star {} with confirmed exoplanets'.format(i+1))

plt.ylabel('Normalized flux')

plt.xlabel('Time, hours')

plt.plot(time, flux3normalized)
def detrender_normalizer(X):

    flux1 = X

    flux2 = ndimage.filters.gaussian_filter(flux1, sigma=10)

    flux3 = flux1 - flux2

    flux3normalized = (flux3-np.mean(flux3)) / (np.max(flux3)-np.min(flux3))

    return flux3normalized
extrain.iloc[:,1:] = extrain.iloc[:,1:].apply(detrender_normalizer,axis=1)

extest.iloc[:,1:] = extest.iloc[:,1:].apply(detrender_normalizer,axis=1)
extrain.head()
i = 13

flux1 = extrain[extrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]

flux1 = flux1.reset_index(drop=True)

time = np.arange(len(flux1)) * (36.0/60.0) # time in units of hours

plt.figure(figsize=(15,5))

plt.title('Processed flux of star {} with confirmed exoplanets'.format(i+1))

plt.ylabel('Flux, e-/s')

plt.xlabel('Time, hours')

plt.plot(time, flux1)
def reduce_upper_outliers(df,reduce = 0.01, half_width=4):

    '''

    Since we are looking at dips in the data, we should remove upper outliers.

    The function is taken from here:

    https://www.kaggle.com/muonneutrino/exoplanet-data-visualization-and-exploration

    '''

    length = len(df.iloc[0,:])

    remove = int(length*reduce)

    for i in df.index.values:

        values = df.loc[i,:]

        sorted_values = values.sort_values(ascending = False)

       # print(sorted_values[:30])

        for j in range(remove):

            idx = sorted_values.index[j]

            #print(idx)

            new_val = 0

            count = 0

            idx_num = int(idx[5:])

            #print(idx,idx_num)

            for k in range(2*half_width+1):

                idx2 = idx_num + k - half_width

                if idx2 <1 or idx2 >= length or idx_num == idx2:

                    continue

                new_val += values['FLUX.'+str(idx2)] # corrected from 'FLUX-' to 'FLUX.'

                

                count += 1

            new_val /= count # count will always be positive here

            #print(new_val)

            if new_val < values[idx]: # just in case there's a few persistently high adjacent values

                df.set_value(i,idx,new_val)

        

            

    return df
extrain.iloc[:,1:] = reduce_upper_outliers(extrain.iloc[:,1:])

extest.iloc[:,1:] = reduce_upper_outliers(extest.iloc[:,1:])
i = 13

flux1 = extrain[extrain.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]

flux1 = flux1.reset_index(drop=True)

time = np.arange(len(flux1)) * (36.0/60.0) # time in units of hours

plt.figure(figsize=(15,5))

plt.title('Processed flux of star {} with confirmed exoplanets (removed upper outliers)'.format(i+1))

plt.ylabel('Normalized flux')

plt.xlabel('Time, hours')

plt.plot(time, flux1)
import warnings

warnings.filterwarnings('ignore')



from sklearn.svm import LinearSVC

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import StratifiedKFold
extrain.LABEL.value_counts()
extest.LABEL.value_counts()
from imblearn.over_sampling import SMOTE
def model_evaluator(X, y, model, n_splits=10):

    skf = StratifiedKFold(n_splits=n_splits)

    

    bootstrapped_accuracies = list()

    bootstrapped_precisions = list()

    bootstrapped_recalls    = list()

    bootstrapped_f1s        = list()

    

    SMOTE_accuracies = list()

    SMOTE_precisions = list()

    SMOTE_recalls    = list()

    SMOTE_f1s        = list()

    

    for train_index, test_index in skf.split(X, y):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = y[train_index], y[test_index]

                

        df_train    = X_train.join(y_train)

        df_planet   = df_train[df_train.LABEL == 2].reset_index(drop=True)

        df_noplanet = df_train[df_train.LABEL == 1].reset_index(drop=True)

        df_boot     = df_noplanet

                        

        index = np.arange(0, df_planet.shape[0])

        temp_index = np.random.choice(index, size=df_noplanet.shape[0])

        df_boot = df_boot.append(df_planet.iloc[temp_index])

        

        df_boot = df_boot.reset_index(drop=True)

        X_train_boot = df_boot.drop('LABEL', axis=1)

        y_train_boot = df_boot.LABEL

                    

        est_boot = model.fit(X_train_boot, y_train_boot)

        y_test_pred = est_boot.predict(X_test)

        

        bootstrapped_accuracies.append(accuracy_score(y_test, y_test_pred))

        bootstrapped_precisions.append(precision_score(y_test, y_test_pred, pos_label=2))

        bootstrapped_recalls.append(recall_score(y_test, y_test_pred, pos_label=2))

        bootstrapped_f1s.append(f1_score(y_test, y_test_pred, pos_label=2))

    

        sm = SMOTE(ratio = 1.0)

        X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)

                    

        est_sm = model.fit(X_train_sm, y_train_sm)

        y_test_pred = est_sm.predict(X_test)

        

        SMOTE_accuracies.append(accuracy_score(y_test, y_test_pred))

        SMOTE_precisions.append(precision_score(y_test, y_test_pred, pos_label=2))

        SMOTE_recalls.append(recall_score(y_test, y_test_pred, pos_label=2))

        SMOTE_f1s.append(f1_score(y_test, y_test_pred, pos_label=2))

        

    print('\t\t\t Bootstrapped \t SMOTE')

    print("Average Accuracy:\t", "{:0.10f}".format(np.mean(bootstrapped_accuracies)),

          '\t', "{:0.10f}".format(np.mean(SMOTE_accuracies)))

    print("Average Precision:\t", "{:0.10f}".format(np.mean(bootstrapped_precisions)),

          '\t', "{:0.10f}".format(np.mean(SMOTE_precisions)))

    print("Average Recall:\t\t", "{:0.10f}".format(np.mean(bootstrapped_recalls)),

          '\t', "{:0.10f}".format(np.mean(SMOTE_recalls)))

    print("Average F1:\t\t", "{:0.10f}".format(np.mean(bootstrapped_f1s)),

          '\t', "{:0.10f}".format(np.mean(SMOTE_f1s)))
extrain_raw = pd.read_csv('../input/exoTrain.csv')
X_raw = extrain_raw.drop('LABEL', axis=1)

y_raw = extrain_raw.LABEL
model_evaluator(X_raw, y_raw, LinearSVC())
X = extrain.drop('LABEL', axis=1)

y = extrain.LABEL
model_evaluator(X, y, LinearSVC())
import scipy
def spectrum_getter(X):

    Spectrum = scipy.fft(X, n=X.size)

    return np.abs(Spectrum)
X_train = extrain.drop('LABEL', axis=1)

y_train = extrain.LABEL



X_test = extest.drop('LABEL', axis=1)

y_test = extest.LABEL
new_X_train = X_train.apply(spectrum_getter,axis=1)

new_X_test = X_test.apply(spectrum_getter,axis=1)
new_X_train.head()
# Segregate data for desigining the model and for the final test

y = y_train

X = new_X_train



y_final_test = y_test

X_final_test = new_X_test
df = X.join(y)

i = 13

spec1 = df[df.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]

freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz

plt.figure(figsize=(15,5))

plt.title('Frequency spectrum of processed flux of star {} with confirmed exoplanets (removed upper outliers)'

          .format(i+1))

plt.ylabel('Unitless flux')

plt.xlabel('Frequency, Hz')

plt.plot(freq, spec1)
X = X.iloc[:,:(X.shape[1]//2)]

X_final_test = X_final_test.iloc[:,:(X_final_test.shape[1]//2)]
# Obtaining flux frequency spectra for several stars without exoplanets from the train data:

df = X.join(y)

for i in [0, 999, 1999, 2999, 3999, 4999]:

    spec1 = df[df.LABEL == 1].drop('LABEL', axis=1).iloc[i,:]

    freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz

    plt.figure(figsize=(15,5))

    plt.title('Frequency spectrum of processed flux of star {} with NO confirmed exoplanets (removed upper outliers)'

              .format(i+1))

    plt.ylabel('Unitless flux')

    plt.xlabel('Frequency, Hz')

    plt.plot(freq, spec1)
# Obtaining flux frequency spectra for several stars with exoplanets from the train data:

df = X.join(y)

for i in [0, 9, 14, 19, 24, 29]:

    spec1 = df[df.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]

    freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz

    plt.figure(figsize=(15,5))

    plt.title('Frequency spectrum of processed flux of star {} WITH confirmed exoplanets (removed upper outliers)'

              .format(i+1))

    plt.ylabel('Unitless flux')

    plt.xlabel('Frequency, Hz')

    plt.plot(freq, spec1)
X.columns
X_columns = np.arange(len(X.columns))

X_columns = X_columns * (1.0/(36.0*60.0)) # sampling frequency of our data

X.columns = X_columns

X_final_test.columns = X_columns
X.columns
model_evaluator(X, y, LinearSVC())
from sklearn.preprocessing import normalize
X = pd.DataFrame(normalize(X))

X_final_test = pd.DataFrame(normalize(X_final_test))
# Obtaining flux frequency spectra for several stars with exoplanets from the train data:

df = X.join(y)

for i in [0, 9, 14, 19, 24, 29]:

    spec1 = df[df.LABEL == 2].drop('LABEL', axis=1).iloc[i,:]

    freq = np.arange(len(spec1)) * (1/(36.0*60.0)) # Sampling frequency is 1 frame per ~36 minutes, or about 0.00046 Hz

    plt.figure(figsize=(15,5))

    plt.title('Frequency spectrum of processed flux of star {} WITH confirmed exoplanets (removed upper outliers)'

              .format(i+1))

    plt.ylabel('Unitless flux')

    plt.xlabel('Frequency, Hz')

    plt.plot(freq, spec1)
model_evaluator(X, y, LinearSVC())
def SMOTE_synthesizer(X, y):

        sm = SMOTE(ratio = 1.0)

        X, y = sm.fit_sample(X, y)

        return X, y
X_sm, y_sm = SMOTE_synthesizer(X, y)
final_model = LinearSVC()

final_model.fit(X_sm, y_sm)
y_pred = final_model.predict(X_final_test)
from sklearn.metrics import classification_report

print(classification_report(y_final_test, y_pred))