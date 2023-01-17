from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
from sklearn.preprocessing import scale
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, balanced_accuracy_score, make_scorer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
# Function for plotting the confusion matrix. It makes use of the Sklearn class ConfusionMatrixDisplay
def plot_confusion_matrix(cm, labels):
    display_labels = labels
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                     display_labels=display_labels)
    return display.plot(include_values=True,
                        cmap='viridis', ax=None, xticks_rotation='horizontal',
                        values_format=None)
rawTrain = pd.read_csv('../input/kepler-labelled-time-series-data/exoTrain.csv')
rawTest = pd.read_csv('../input/kepler-labelled-time-series-data/exoTest.csv')
print(rawTrain.head())
print(rawTrain.dtypes)
train_orig = rawTrain.iloc[:, 1:].to_numpy()
test_orig = rawTest.iloc[:, 1:].to_numpy()
train_y = rawTrain.iloc[:, 0].to_numpy() == 2
test_y = rawTest.iloc[:, 0].to_numpy() == 2
x_t = np.array(list(i*1765.5 for i in range(train_orig.shape[1]))) / 60 / 60 / 24

fig, axs = plt.subplots(2, 2, sharex=True)
fig.suptitle("Stars comparison")
np.random.seed(4)
axs[0,0].plot(x_t, train_orig[np.random.choice(np.where(train_y)[0]), :])
axs[1,0].plot(x_t, train_orig[np.random.choice(np.where(train_y)[0]), :])
axs[0,0].set_title("Confirmed exoplanet")
axs[1,0].set_xlabel("Time (day)")
axs[0,0].set_ylabel("Star flux")
axs[1,0].set_ylabel("Star flux")
axs[0,0].grid()
axs[1,0].grid()
np.random.seed(3)
axs[0,1].plot(x_t, train_orig[np.random.choice(np.where(~train_y)[0]), :])
axs[1,1].plot(x_t, train_orig[np.random.choice(np.where(~train_y)[0]), :])
axs[0,1].set_title("No exoplanet")
axs[1,1].set_xlabel('Time (day)')
axs[0,1].grid()
axs[1,1].grid()
plt.show()
print("Proportion of confirmed exoplanet in training set: %0.3f%%" %(sum(train_y)/len(train_y)*100))
print("Proportion of confirmed exoplanet in test set: %0.3f%%" %(sum(test_y)/len(test_y)*100))
print("Cconfirmed exoplanet in training set: %d" %sum(train_y))
print("Confirmed exoplanet in test set: %d" %sum(test_y))
train_time = scale(ss.detrend(train_orig, bp=np.linspace(0, train_orig.shape[1], 50, dtype=int)), axis=1)
test_time = scale(ss.detrend(test_orig, bp=np.linspace(0, test_orig.shape[1], 50, dtype=int)), axis=1)

fig, axs = plt.subplots(2, 2, sharex=True)
fig.suptitle("Before (top) and after (bottom) scaling and detrending")
np.random.seed(4)
axs[0, 0].plot(x_t, train_orig[np.random.choice(np.where(train_y)[0]), :])
axs[0, 1].plot(x_t, train_orig[np.random.choice(np.where(train_y)[0]), :])
axs[0, 0].set_ylabel('Star flux')
axs[0, 0].grid()
axs[0, 1].grid()
np.random.seed(4)
axs[1, 0].plot(x_t, train_time[np.random.choice(np.where(train_y)[0]), :])
axs[1, 1].plot(x_t, train_time[np.random.choice(np.where(train_y)[0]), :])
axs[1, 0].set_ylabel('Normalised star flux')
axs[1, 0].set_xlabel('Time (day)')
axs[1, 1].set_xlabel('Time (day)')
axs[1, 0].grid()
axs[1, 1].grid()
plt.show()
x_f, train_freq = ss.periodogram(train_time, fs=1 / 1765.5, window=('kaiser', 4.0), axis=1)
test_freq = ss.periodogram(test_time, fs=1 / 1765.5, window=('kaiser', 4.0), axis=1)[1]

fig, axs = plt.subplots(2, 2)
fig.suptitle("Power spectrum density of two stars")
np.random.seed(4)
_next = np.random.choice(np.where(train_y)[0])
axs[0, 0].plot(x_t, train_time[_next, :])
axs[1, 0].plot(x_f, train_freq[_next, :])
axs[0, 0].set_title('With exoplanet')
axs[0, 0].set_xlabel('Time')
axs[1, 0].set_xlabel('Frequency')
axs[0, 0].set_ylabel('Star flux')
axs[1, 0].set_ylabel('Power per frequency')
axs[0, 0].grid()
axs[0, 1].grid()
_next = np.random.choice(~np.where(train_y)[0])
axs[0, 1].plot(x_t, train_time[_next, :])
axs[1, 1].plot(x_f, train_freq[_next, :])
axs[0, 1].set_title('No exoplanet')
axs[0, 1].set_xlabel('Time')
axs[1, 1].set_xlabel('Frequency')
axs[1, 0].grid()
axs[1, 1].grid()
plt.show()
b, a = ss.butter(N=4, Wn=(0.00001, 0.00013), btype='bandpass', fs=1 / 1765.5)
train_time_filt = ss.filtfilt(b, a, train_time, axis=1)
test_time_filt = ss.filtfilt(b, a, test_time, axis=1)
train_freq_filt = ss.periodogram(train_time_filt, fs=1 / 1765.5, window=('kaiser', 4.0), axis=1)[1]
test_freq_filt = ss.periodogram(test_time_filt, fs=1 / 1765.5, window=('kaiser', 4.0), axis=1)[1]

fig, axs = plt.subplots(2, 2)
fig.suptitle("Time and spectrum comparison before and after filtering")
np.random.seed(4)
_next = np.random.choice(np.where(train_y)[0])
axs[0, 0].plot(x_t, train_time[_next, :])
axs[1, 0].plot(x_f, train_freq[_next, :])
axs[0, 0].set_title('Original')
axs[0, 0].set_xlabel('Time')
axs[1, 0].set_xlabel('Frequency')
axs[0, 0].set_ylabel('Star flux')
axs[1, 0].set_ylabel('Power per frequency')
axs[0, 0].grid()
axs[0, 1].grid()
axs[0, 1].plot(x_t, train_time_filt[_next, :])
axs[1, 1].plot(x_f, train_freq_filt[_next, :])
axs[0, 1].set_title('Filtered')
axs[0, 1].set_xlabel('Time')
axs[1, 1].set_xlabel('Frequency')
axs[1, 0].grid()
axs[1, 1].grid()
plt.show()
svc_estimator = SVC(tol=1e-4, class_weight='balanced', max_iter=1e+5, random_state=1)

svc_tuning_grid = {"C": (0.12,)}
svc_cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=1)
svc_time_gridcv = GridSearchCV(estimator=svc_estimator, param_grid=svc_tuning_grid,
                               scoring='f1', cv=svc_cv, return_train_score=True)

svc_time_filt_gridcv = deepcopy(svc_time_gridcv)
svc_freq_gridcv = deepcopy(svc_time_gridcv)
svc_freq_filt_gridcv = deepcopy(svc_time_gridcv)
svc_time_model = svc_time_gridcv.fit(train_time, train_y)
svc_time_filt_model = svc_time_gridcv.fit(train_time_filt, train_y)
svc_freq_model = svc_freq_gridcv.fit(train_freq, train_y)
svc_freq_filt_model = svc_time_gridcv.fit(train_freq_filt, train_y)
print("Time signal best parameters: %s" %svc_time_model.best_params_)
print("Filtered time signal best parameters: %s" %svc_time_filt_model.best_params_)
print("Frequency signal best parameters: %s" %svc_freq_model.best_params_)
print("Filtered frequency signal best parameters: %s" %svc_freq_filt_model.best_params_)
print("Time model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(svc_time_model.cv_results_['mean_test_score'][0],
                                                svc_time_model.cv_results_['std_test_score'][0]))

print("-----\nFiltered time model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(svc_time_filt_model.cv_results_['mean_test_score'][0],
                                                svc_time_filt_model.cv_results_['std_test_score'][0]))

print("-----\nFrequency model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(svc_freq_model.cv_results_['mean_test_score'][0],
                                                svc_freq_model.cv_results_['std_test_score'][0]))

print("-----\nFiltered frequency model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(svc_freq_filt_model.cv_results_['mean_test_score'][0],
                                                svc_freq_filt_model.cv_results_['std_test_score'][0]))
pca_time = PCA(svd_solver='full').fit(train_time)
pca_freq = PCA(svd_solver='full').fit(train_freq)
plt.plot(np.cumsum(pca_time.explained_variance_ratio_), label='Time PCA')
plt.plot(np.cumsum(pca_freq.explained_variance_ratio_), label='Freq. PCA')
plt.xlabel("Number of components")
plt.ylabel("% of variance explained")
plt.legend()
plt.show()
plt.plot(range(150,600), np.cumsum(pca_freq.explained_variance_ratio_)[150:600])
plt.xlabel("Number of components")
plt.ylabel("% of variance explained")
plt.title("Frequency PCA knee")
plt.show()
train_pca = PCA(n_components=0.95).fit_transform(train_freq)
print("Number of PCA components to get 95%%: %d" %train_pca.shape[1])
pca_cv = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=1)
pca_gridcv = GridSearchCV(estimator=svc_estimator, param_grid=svc_tuning_grid, scoring='f1',
                          cv=pca_cv, return_train_score=True)
pca_model = pca_gridcv.fit(train_pca, train_y)
print("\nMean score: %.3f\tSt. dev.: %.3f" %(pca_model.cv_results_['mean_test_score'][0],
                                             pca_model.cv_results_['std_test_score'][0]))
def train_dl_model(x, y, network, tuning_grid=dict()):
    splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=1)
    grid_searcher = GridSearchCV(estimator=network,
                                 param_grid=tuning_grid,
                                 cv=splitter,
                                 scoring='f1',
                                 return_train_score=True)
    np.random.seed(1)
    model = grid_searcher.fit(x, y,
                              batch_size=32,
                              epochs=50,
                              verbose=0)
    return model
# Source: https://stackoverflow.com/a/45305384/4180457

from keras import backend as K
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def build_mlp(n_units):
    model = Sequential()
    model.add(Dropout(0.1))
    model.add(Dense(n_units, input_shape=nn_shape))
    model.add(Dropout(0.2))
    model.add(Dense(n_units))
    model.add(Dropout(0.2))
    model.add(Dense(n_units))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=[f1])
    return model
nn_shape = train_freq[1:]

mlp_tuning_grid = {"n_units": (300,)}

mlp_freq_model = KerasClassifier(build_fn=build_mlp)
mlp_freq_model = train_dl_model(train_freq, train_y, mlp_freq_model, mlp_tuning_grid)
nn_shape = train_pca[1:]

mlp_pca_model = KerasClassifier(build_fn=build_mlp)
mlp_pca_model = train_dl_model(train_pca, train_y, mlp_pca_model, mlp_tuning_grid)
print("Frequency domain best parameters: %s" %mlp_freq_model.best_params_)
print("PCA doman best parameters: %s" %mlp_pca_model.best_params_)
print("Frequency domain model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(mlp_freq_model.cv_results_['mean_test_score'][0],
                                                mlp_freq_model.cv_results_['std_test_score'][0]))

print("-----\nPCA domain model")
print("\nMean CV score: %.3f\tSt. dev.: %.3f" %(mlp_pca_model.cv_results_['mean_test_score'][0],
                                                mlp_pca_model.cv_results_['std_test_score'][0]))
final_model = SVC(C=0.12, tol=1e-4, class_weight='balanced', max_iter=1e+5,
                  random_state=1).fit(train_freq_filt, train_y)
predictions = final_model.predict(test_freq_filt)
false_negatives = [i for i, v in enumerate(test_y) if v and test_y[i] != predictions[i]]
false_positives = [i for i, v in enumerate(test_y) if not v and test_y[i] != predictions[i]]
print("False negatives: %d" %len(false_negatives))
print("False positives: %d" %len(false_positives))
print("f1-score:  %.3f" %f1_score(test_y, predictions))
print("Balanced accuracy:  %.3f" %balanced_accuracy_score(test_y, predictions))

plot_confusion_matrix(confusion_matrix(test_y, predictions, labels=[True, False]), labels=[True, False])
fig, axs = plt.subplots(1, len(false_negatives), sharey=True)
fig.suptitle("Time series false negatives")
for i in range(len(false_negatives)):
    axs[i].plot(x_t, test_time[false_negatives[i],:])
axs[0].set_xlabel('Time')
axs[1].set_xlabel('Time')
axs[0].set_ylabel('Star flux')
axs[0].grid()
axs[1].grid()
plt.show()
fig, axs = plt.subplots(1, len(false_negatives), sharey=True)
fig.suptitle("Frequency domain false negatives")
for i in range(len(false_negatives)):
    axs[i].plot(x_f, test_freq_filt[false_negatives[i],:])
axs[0].set_xlabel('Frequency')
axs[1].set_xlabel('Frequency')
axs[0].set_ylabel('Power per frequency')
axs[0].grid()
axs[1].grid()
plt.show()