import os

path = os.getcwd()
print(os.listdir('../input'))
# Pandas and numpy for data manipulation

import pandas as pd

import numpy as np



# No warnings about setting value on copy of slice

pd.options.mode.chained_assignment = None



# pandas doesn't show us all the decimals

pd.options.display.precision = 15



# Matplotlib visualization

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

%matplotlib inline



# Scipy

from scipy import stats

from scipy.signal import hann

from scipy.signal import hilbert

from scipy.signal import convolve



# Scaling values

from sklearn.preprocessing import StandardScaler



# Splitting data into training and testing

from sklearn.model_selection import train_test_split



# Metric

from sklearn.metrics import mean_absolute_error



# Machine Learning

from xgboost.sklearn import XGBRegressor



# Hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



# Others

from sklearn.linear_model import LinearRegression

from tqdm import tqdm_notebook



import warnings

warnings.filterwarnings("ignore")
# Read in data into a dataframe 

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
# Show dataframe columns

print(train.columns)
# Display top of dataframe

train.head()
# Display bottom of dataframe

train.tail()
# Display the shape of dataframe

train.shape
# See the column data types and non-missing values

train.info()
# https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction

train_ad_sample_df = train['acoustic_data'].values[::100]

train_ttf_sample_df = train['time_to_failure'].values[::100]



def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):

    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.title(title)

    plt.plot(train_ad_sample_df, color='r')

    ax1.set_ylabel('acoustic data', color='r')

    plt.legend(['acoustic data'], loc=(0.01, 0.95))

    ax2 = ax1.twinx()

    plt.plot(train_ttf_sample_df, color='b')

    ax2.set_ylabel('time to failure', color='b')

    plt.legend(['time to failure'], loc=(0.01, 0.9))

    plt.grid(True)



plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)

del train_ad_sample_df

del train_ttf_sample_df
# https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction

train_ad_sample_df = train['acoustic_data'].values[:6291455]

train_ttf_sample_df = train['time_to_failure'].values[:6291455]

plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% of data")

del train_ad_sample_df

del train_ttf_sample_df
# https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction

rows = 150000

segments = int(np.floor(train.shape[0] / rows))

print("Number of segments: ", segments)
# https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction

def add_trend_feature(arr, abs_values=False):

    idx = np.array(range(len(arr)))

    if abs_values:

        arr = np.abs(arr)

    lr = LinearRegression()

    lr.fit(idx.reshape(-1, 1), arr)

    return lr.coef_[0]



def classic_sta_lta(x, length_sta, length_lta):

    sta = np.cumsum(x ** 2)

    # Convert to float

    sta = np.require(sta, dtype=np.float)

    # Copy for LTA

    lta = sta.copy()

    # Compute the STA and the LTA

    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]

    sta /= length_sta

    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]

    lta /= length_lta

    # Pad zeros

    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float

    dtiny = np.finfo(0.0).tiny

    idx = lta < dtiny

    lta[idx] = dtiny

    return sta / lta
# https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction

# Create a training file with simple derived features

X_train = pd.DataFrame(index=range(segments), dtype=np.float64)

y_train = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

total_mean = train['acoustic_data'].mean()

total_std = train['acoustic_data'].std()

total_max = train['acoustic_data'].max()

total_min = train['acoustic_data'].min()

total_sum = train['acoustic_data'].sum()

total_abs_sum = np.abs(train['acoustic_data']).sum()
# https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction

def create_features(seg_id, seg, X):

    xc = pd.Series(seg['acoustic_data'].values)

    zc = np.fft.fft(xc)

    

    X.loc[seg_id, 'mean'] = xc.mean()

    X.loc[seg_id, 'std'] = xc.std()

    X.loc[seg_id, 'max'] = xc.max()

    X.loc[seg_id, 'min'] = xc.min()

    

    #FFT transform values

    realFFT = np.real(zc)

    imagFFT = np.imag(zc)

    X.loc[seg_id, 'Rmean'] = realFFT.mean()

    X.loc[seg_id, 'Rstd'] = realFFT.std()

    X.loc[seg_id, 'Rmax'] = realFFT.max()

    X.loc[seg_id, 'Rmin'] = realFFT.min()

    X.loc[seg_id, 'Imean'] = imagFFT.mean()

    X.loc[seg_id, 'Istd'] = imagFFT.std()

    X.loc[seg_id, 'Imax'] = imagFFT.max()

    X.loc[seg_id, 'Imin'] = imagFFT.min()

    X.loc[seg_id, 'Rmean_last_5000'] = realFFT[-5000:].mean()

    X.loc[seg_id, 'Rstd__last_5000'] = realFFT[-5000:].std()

    X.loc[seg_id, 'Rmax_last_5000'] = realFFT[-5000:].max()

    X.loc[seg_id, 'Rmin_last_5000'] = realFFT[-5000:].min()

    X.loc[seg_id, 'Rmean_last_15000'] = realFFT[-15000:].mean()

    X.loc[seg_id, 'Rstd_last_15000'] = realFFT[-15000:].std()

    X.loc[seg_id, 'Rmax_last_15000'] = realFFT[-15000:].max()

    X.loc[seg_id, 'Rmin_last_15000'] = realFFT[-15000:].min()

    

    X.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(xc))

    X.loc[seg_id, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(xc) / xc[:-1]))[0])

    X.loc[seg_id, 'abs_max'] = np.abs(xc).max()

    X.loc[seg_id, 'abs_min'] = np.abs(xc).min()

    

    X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()

    X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()

    X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()

    X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()

    

    X.loc[seg_id, 'avg_first_50000'] = xc[:50000].mean()

    X.loc[seg_id, 'avg_last_50000'] = xc[-50000:].mean()

    X.loc[seg_id, 'avg_first_10000'] = xc[:10000].mean()

    X.loc[seg_id, 'avg_last_10000'] = xc[-10000:].mean()

    

    X.loc[seg_id, 'min_first_50000'] = xc[:50000].min()

    X.loc[seg_id, 'min_last_50000'] = xc[-50000:].min()

    X.loc[seg_id, 'min_first_10000'] = xc[:10000].min()

    X.loc[seg_id, 'min_last_10000'] = xc[-10000:].min()

    

    X.loc[seg_id, 'max_first_50000'] = xc[:50000].max()

    X.loc[seg_id, 'max_last_50000'] = xc[-50000:].max()

    X.loc[seg_id, 'max_first_10000'] = xc[:10000].max()

    X.loc[seg_id, 'max_last_10000'] = xc[-10000:].max()

    

    X.loc[seg_id, 'max_to_min'] = xc.max() / np.abs(xc.min())

    X.loc[seg_id, 'max_to_min_diff'] = xc.max() - np.abs(xc.min())

    X.loc[seg_id, 'count_big'] = len(xc[np.abs(xc) > 500])

    X.loc[seg_id, 'sum'] = xc.sum()

    

    X.loc[seg_id, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(xc[:50000]) / xc[:50000][:-1]))[0])

    X.loc[seg_id, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(xc[-50000:]) / xc[-50000:][:-1]))[0])

    X.loc[seg_id, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(xc[:10000]) / xc[:10000][:-1]))[0])

    X.loc[seg_id, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(xc[-10000:]) / xc[-10000:][:-1]))[0])

    

    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)

    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)

    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)

    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)

    

    X.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(xc), 0.95)

    X.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(xc), 0.99)

    X.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(xc), 0.05)

    X.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(xc), 0.01)

    

    X.loc[seg_id, 'trend'] = add_trend_feature(xc)

    X.loc[seg_id, 'abs_trend'] = add_trend_feature(xc, abs_values=True)

    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()

    X.loc[seg_id, 'abs_std'] = np.abs(xc).std()

    

    X.loc[seg_id, 'mad'] = xc.mad()

    X.loc[seg_id, 'kurt'] = xc.kurtosis()

    X.loc[seg_id, 'skew'] = xc.skew()

    X.loc[seg_id, 'med'] = xc.median()

    

    X.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(xc)).mean()

    X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()

    X.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(xc, 500, 10000).mean()

    X.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(xc, 5000, 100000).mean()

    X.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(xc, 3333, 6666).mean()

    X.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(xc, 10000, 25000).mean()

    X.loc[seg_id, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_1500_mean'] = xc.rolling(window=1500).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_3000_mean'] = xc.rolling(window=3000).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_6000_mean'] = xc.rolling(window=6000).mean().mean(skipna=True)

    ewma = pd.Series.ewm

    X.loc[seg_id, 'exp_Moving_average_300_mean'] = (ewma(xc, span=300).mean()).mean(skipna=True)

    X.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)

    X.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(xc, span=6000).mean().mean(skipna=True)

    no_of_std = 2

    X.loc[seg_id, 'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()

    X.loc[seg_id,'MA_700MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X.loc[seg_id,'MA_700MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X.loc[seg_id, 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()

    X.loc[seg_id,'MA_400MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X.loc[seg_id,'MA_400MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X.loc[seg_id, 'MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()

    

    X.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(xc, [75, 25]))

    X.loc[seg_id, 'q999'] = np.quantile(xc,0.999)

    X.loc[seg_id, 'q001'] = np.quantile(xc,0.001)

    X.loc[seg_id, 'ave10'] = stats.trim_mean(xc, 0.1)

    

    for windows in [10, 100, 1000]:

        x_roll_std = xc.rolling(windows).std().dropna().values

        x_roll_mean = xc.rolling(windows).mean().dropna().values

        

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()

        X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()

        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()

        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)

        X.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)

        X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))

        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        

        X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()

        X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()

        X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()

        X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()

        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
# https://www.kaggle.com/gpreda/lanl-earthquake-eda-and-prediction

# iterate over all segments

for seg_id in tqdm_notebook(range(segments)):

    seg = train.iloc[seg_id*rows:seg_id*rows+rows]

    create_features(seg_id, seg, X_train)

    y_train.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
X_train.shape
X_train.head()
y_train.shape
y_train.head()
# Read in data into a dataframe 

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)
submission.shape
X_test.shape
for seg_id in tqdm_notebook(X_test.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    create_features(seg_id, seg, X_test)
X_train.to_csv('training_features.csv', index = False)

X_test.to_csv('testing_features.csv', index = False)

y_train.to_csv('training_labels.csv', index = False)
# Read in data into dataframes 

train_features = pd.read_csv('training_features.csv')

train_labels = pd.read_csv('training_labels.csv')

test_features = pd.read_csv('testing_features.csv')
# Split into training and testing set

X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size = 0.2, random_state = 42)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Create the scaler object with a range of 0-1

scaler = StandardScaler()



# Fit on the training data

scaler.fit(X_train)



# Transform both the training and testing data

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
# Function to calculate mean absolute error

def mae(y_true, y_pred):

    return mean_absolute_error(y_true.values.flatten(), y_pred)
# Takes in a model, trains the model, and evaluates the model on the test set

def fit_and_evaluate(model):

    

    # Train the model

    model.fit(X_train, y_train.values.flatten())

    

    # Make predictions and evaluate

    model_pred = model.predict(X_test)

    model_mae = mae(y_test, model_pred)

        

    # Return the performance metric

    return model_mae
xgboost = XGBRegressor()

xgboost_mae = fit_and_evaluate(xgboost)



print('XGBoost Regression Performance: MAE = %0.3f' % xgboost_mae)
# Define the grid of hyperparameters to search

one_to_left = stats.beta(10, 1)  

from_zero_positive = stats.expon(0, 50)



hyperparameter_grid = {'n_estimators': stats.randint(3, 40),

                       'max_depth': stats.randint(3, 40),

                       'learning_rate': stats.uniform(0.05, 0.4),

                       'colsample_bytree': one_to_left,

                       'subsample': one_to_left,

                       'gamma': stats.uniform(0, 10),

                       'reg_alpha': from_zero_positive,

                       'min_child_weight': from_zero_positive}
# Create the model to use for hyperparameter tuning

model = XGBRegressor(nthreads=-1, random_state = 42)



# Set up the random search with 10-fold cross validation

random_cv = RandomizedSearchCV(estimator=model,

                               param_distributions=hyperparameter_grid,

                               cv=10, n_iter=100, 

                               scoring = 'neg_mean_absolute_error',

                               n_jobs = -1, verbose = 1, 

                               return_train_score = True,

                               random_state=42)
# Fit on the training data

random_cv.fit(X_train, y_train)
# Get all of the cv results and sort by the test performance

random_results = pd.DataFrame(random_cv.cv_results_).sort_values('mean_test_score', ascending = False)



random_results.head(10)
random_cv.best_estimator_
# Default model

default_model = XGBRegressor(nthreads=-1, random_state = 42)



# Select the best model

final_model = random_cv.best_estimator_



final_model
default_model.fit(X_train, y_train)
final_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)

final_pred = final_model.predict(X_test)



print('Default model performance on the test set: MAE = %0.3f.' % mae(y_test, default_pred))

print('Final model performance on the test set:   MAE = %0.3f.' % mae(y_test, final_pred))
final_pred
# Read in data into a dataframe 

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
# Collecting submission dataset

X_submission = test_features
X_submission.info()
X_submission.head()
# Create the scaler object with a range of 0-1

scaler = StandardScaler()



# Fit on the training data

scaler.fit(X_submission)



# Transform the test data

X_submission = scaler.transform(X_submission)
# Make predictions on the submission set

model_pred = final_model.predict(X_submission)

model_pred
# Final dataset with predictions

submission['time_to_failure'] = model_pred



submission.to_csv('submission.csv', index=True)
submission.head(20)