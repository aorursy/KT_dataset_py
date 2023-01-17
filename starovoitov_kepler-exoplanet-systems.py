import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns; sns.set()

from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, classification_report, precision_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import normalize, StandardScaler
from sklearn import linear_model
from imblearn.over_sampling import SMOTE
from scipy import ndimage, fft

%matplotlib inline

#see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
train_df = pd.read_csv('../input/exoTrain.csv')
test_df = pd.read_csv('../input/exoTest.csv')
train_df.head()
print(train_df.shape)
train_df.LABEL.value_counts(normalize=True)
test_df.LABEL.value_counts(normalize=True)
X_train = train_df.loc[:, train_df.columns != 'LABEL'].values
y_train = train_df.LABEL.values

X_test = test_df.loc[:, train_df.columns != 'LABEL'].values
y_test = test_df.LABEL.values
def get_distribution_params(intensity_vals, window_size=10):
    """Returns rolling meand and rolling standard deviation by given time series and window size"""
    return intensity_vals.rolling(window_size).mean(), intensity_vals.rolling(window_size).std()

def plot_light_intensity(X, figsize=(16,8), resolution=1.0):
    """Plots given light intensity time series"""
    
    if resolution < 1.0:
        resolution = int(1.0//resolution)
        X = X[::resolution]
    
    intensity_vals = pd.DataFrame(X)
    measurements = [i for i in range(1, len(X) + 1, 1)]
    
    rolling_mean_variety, rolling_std_variety = get_distribution_params(intensity_vals)
    
    plt.figure(figsize=(figsize[0],figsize[1]))
    plt.title = "Start light intensity variation"
     
    plt.plot(measurements, intensity_vals.values,color='b')
    plt.plot(measurements, rolling_mean_variety.values,color='r')
    plt.plot(measurements, rolling_std_variety.values,color='g')

    blue_line = mlines.Line2D([], [], color='blue', label='Light intensity meterages')
    red_line = mlines.Line2D([], [], color='red', label='Mean light intensity')
    green_line = mlines.Line2D([], [], color='green', label='Standard deviation of light intensity')
    plt.legend(handles=[blue_line, red_line, green_line])

    plt.xlabel('Meterages', fontsize=18)
    
    plt.xticks(rotation=90)
    plt.ylabel('Light intensity', fontsize=18)

    plt.show()
series_number = list(y_train).index(2)
print("Number of plotted series: ", series_number)

plot_light_intensity(X_train[series_number], resolution=1.0)
def seasonal_decompose_fft(X, freq):
    """Returns decomposed time series into seasonal, trend, residual, observed
    and fourier transform components"""
    decomposition = seasonal_decompose(X, freq=freq)
    
    return DecomposeResult(seasonal=decomposition.seasonal, trend=decomposition.trend,
                           resid=decomposition.resid, observed=decomposition.observed, 
                           fft=np.fft.fft(decomposition.seasonal))

def plot_decomposed_seasonal(decomposition):
    """Plots decomposition of seasonal_decompose_fft function"""
    plt.figure(figsize=(16,8))
    plt.subplot(511)
    plt.plot(decomposition.observed, label='Original')
    plt.legend(loc='best')
    plt.subplot(512)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(513)
    plt.plot(decomposition.seasonal,label='Seasonality')
    plt.legend(loc='best')
    plt.subplot(514)
    plt.plot(decomposition.resid, label='Residuals')
    plt.legend(loc='best')
    if hasattr(decomposition, 'fft'):
        plt.subplot(515)
        plt.plot(decomposition.fft, label='Fourier decomposition')
        plt.legend(loc='best')
    plt.tight_layout()
decomposition = seasonal_decompose_fft(X_train[series_number], freq=900)
plot_decomposed_seasonal(decomposition)
X_dec = [seasonal_decompose_fft(X_train[i], freq=900) for i in range(0, len(X_train), 1)]
X_test_dec = [seasonal_decompose_fft(X_test[i], freq=900) for i in range(0, len(X_test), 1)]
X_fft = absolute = [np.abs(X.fft[:(len(X.fft)//2)]) for X in X_dec]
X_test_fft = [np.abs(X.fft[:(len(X.fft)//2)]) for X in X_test_dec]
X_fft = normalized = normalize(X_fft)
X_test_fft = normalize(X_test_fft)
X_fft = filtered = ndimage.filters.gaussian_filter(X_fft, sigma=10)
X_test_fft = ndimage.filters.gaussian_filter(X_test_fft, sigma=10)
std_scaler = StandardScaler()
X_fft = scaled = std_scaler.fit_transform(X_fft)
X_test_fft = std_scaler.fit_transform(X_test_fft)
plt.figure(figsize=(16,8))
plt.subplot(221)
plt.plot(absolute[series_number], label='Original')
plt.legend(loc='best')
plt.subplot(222)
plt.plot(normalized[series_number], label='Normalized')
plt.legend(loc='best')
plt.subplot(223)
plt.plot(filtered[series_number],label='Filtered')
plt.legend(loc='best')
plt.subplot(224)
plt.plot(scaled[series_number],label='Scaled')
plt.legend(loc='best')
plt.tight_layout()
sm = SMOTE(ratio = 1.0)
X_fft_sm, y_train_sm = sm.fit_sample(X_fft, y_train)

print(len(X_fft_sm))
model = linear_model.SGDClassifier(max_iter=1000, loss="perceptron", penalty="l2", alpha=1e-2, eta0=1.0, learning_rate="invscaling")
model.fit(X_fft_sm, y_train_sm)
Y_train_predicted = model.predict(X_fft_sm)
Y_test_predicted =  model.predict(X_test_fft)

print("Train accuracy = %.4f" % accuracy_score(y_train_sm, Y_train_predicted))
print("Test accuracy = %.4f" % accuracy_score(y_test, Y_test_predicted))
print(y_test)
print(Y_test_predicted)
confusion_matrix_train = confusion_matrix(y_train_sm, Y_train_predicted)
confusion_matrix_test = confusion_matrix(y_test, Y_test_predicted)
classification_report_train = classification_report(y_train_sm, Y_train_predicted)
classification_report_test = classification_report(y_test, Y_test_predicted)

print("Confusion Matrix (train sample):\n", confusion_matrix_train)
print("Confusion Matrix (test sample):\n", confusion_matrix_test)
print("\n")
print("Classification_report (train sample):\n", classification_report_train)
print("Classification_report (test sample):\n", classification_report_test)