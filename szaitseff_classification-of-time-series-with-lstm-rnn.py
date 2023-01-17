# Import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt   # plotting
import seaborn as sns   # plotting heatmap

%matplotlib inline
# Import data, convert string dates to 'datetime64' and set the date column as index:
df = pd.read_csv('../input/test_task_data.csv',
                 parse_dates=['date'],
                 infer_datetime_format=True,
                 index_col='date',
                 thousands=',',
                 decimal='.'
                )
#  Review the general info on data, paying attention to missing values and dtypes
df.info()
# Let's remove the empty column and look at some examples of data:
df = df.drop(columns='Unnamed: 17')
print(f'data shape = {df.shape}')
df.head()
# It appears that 'feature_5' has missing values up to 2012-10-18
# let's fill them backwards
df.feature_5 = df.feature_5.fillna(method='bfill')
# Basic statistics of the data:
df.describe()
# Plot the time series
plt.style.use('fivethirtyeight')
df.plot(subplots=True,
        layout=(6, 3),
        figsize=(22,22),
        fontsize=10, 
        linewidth=2,
        sharex=False,
        title='Visualization of the original Time Series')
plt.show()
# Let's also draw a heatmap visualization of the correlation matrix
corr_matrix = df.corr(method='spearman')
f, ax = plt.subplots(figsize=(16,8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidth=0.4,
            annot_kws={"size": 10}, cmap='coolwarm', ax=ax)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
# We want to keep original time series for the EDA in APPENDIX
# So we apply the 'pc_change()' transformation to a copy of the original time series
df_transform = df.loc[:,:'target_value'].copy().pct_change(1)
df_transform.head()
# fill the 1st row with NA data
df_transform.fillna(method='bfill', inplace=True)
df_transform.head()
# Plot the transformed time series
plt.style.use('fivethirtyeight')
df_transform.plot(subplots=True,
                  layout=(6, 3),
                  figsize=(24,24),
                  fontsize=10, 
                  linewidth=2, 
                  title='Visualization of the transformed Features')
plt.show()
# Split train and test data
train_features = df_transform.loc['2012-01-02':'2016-12-31']
train_labels = df.loc['2012-01-02':'2016-12-31', 'target_class']

test_features = df_transform.loc['2017-01-02':'2018-06-19']
test_labels = df.loc['2017-01-02':'2018-06-19', 'target_class']

# I want to use a T-days window of input data for predicting target_class
# It means I need to prepend (T-1) last train records to the 1st test window
T = 45  # my choice of the timesteps window

prepend_features = train_features.iloc[-(T-1):]
test_features = pd.concat([prepend_features, test_features], axis=0)

train_features.shape, train_labels.shape, test_features.shape, test_labels.shape
# Rescale the features
from sklearn.preprocessing import StandardScaler  # MinMaxScaler

scaler = StandardScaler()  # MinMaxScaler(feature_range=(-1,1))
scaled_train_features = pd.DataFrame(scaler.fit_transform(train_features.values),
                                     index=train_features.index,
                                     columns=train_features.columns)
# The Scaler is fit on the training set and then applied to the test set
scaled_test_features = pd.DataFrame(scaler.transform(test_features.values),
                                    index=test_features.index,
                                    columns=test_features.columns)

scaled_train_features.shape, scaled_test_features.shape
# Plot the rescaled_train_features
plt.style.use('fivethirtyeight')
scaled_train_features.plot(subplots=True,
                           layout=(6, 3),
                           figsize=(24,24),
                           fontsize=10, 
                           linewidth=2, 
                           title='Visualization of the scaled Train Features')
plt.show()
# Create sequences of T timesteps
X_train, y_train = [], []
for i in range(train_labels.shape[0] - (T-1)):
    X_train.append(scaled_train_features.iloc[i:i+T].values)
    y_train.append(train_labels.iloc[i + (T-1)])
X_train, y_train = np.array(X_train), np.array(y_train).reshape(-1,1)
print(f'Train data dimensions: {X_train.shape}, {y_train.shape}')

X_test, y_test = [], []
for i in range(test_labels.shape[0]):
    X_test.append(scaled_test_features.iloc[i:i+T].values)
    y_test.append(test_labels.iloc[i])
X_test, y_test = np.array(X_test), np.array(y_test).reshape(-1,1)  

print(f'Test data dimensions: {X_test.shape}, {y_test.shape}')
# Import Keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from time import time
# Let's make a list of CONSTANTS for modelling:
LAYERS = [8, 8, 8, 1]                # number of units in hidden and output layers
M_TRAIN = X_train.shape[0]           # number of training examples (2D)
M_TEST = X_test.shape[0]             # number of test examples (2D),full=X_test.shape[0]
N = X_train.shape[2]                 # number of features
BATCH = M_TRAIN                          # batch size
EPOCH = 50                           # number of epochs
LR = 5e-2                            # learning rate of the gradient descent
LAMBD = 3e-2                         # lambda in L2 regularizaion
DP = 0.0                             # dropout rate
RDP = 0.0                            # recurrent dropout rate
print(f'layers={LAYERS}, train_examples={M_TRAIN}, test_examples={M_TEST}')
print(f'batch = {BATCH}, timesteps = {T}, features = {N}, epochs = {EPOCH}')
print(f'lr = {LR}, lambda = {LAMBD}, dropout = {DP}, recurr_dropout = {RDP}')

# Build the Model
model = Sequential()
model.add(LSTM(input_shape=(T, N), units=LAYERS[0],
               activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False
              ))
model.add(BatchNormalization())
model.add(LSTM(units=LAYERS[1],
               activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=True, return_state=False,
               stateful=False, unroll=False
              ))
model.add(BatchNormalization())
model.add(LSTM(units=LAYERS[2],
               activation='tanh', recurrent_activation='hard_sigmoid',
               kernel_regularizer=l2(LAMBD), recurrent_regularizer=l2(LAMBD),
               dropout=DP, recurrent_dropout=RDP,
               return_sequences=False, return_state=False,
               stateful=False, unroll=False
              ))
model.add(BatchNormalization())
model.add(Dense(units=LAYERS[3], activation='sigmoid'))

# Compile the model with Adam optimizer
model.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=Adam(lr=LR))
print(model.summary())

# Define a learning rate decay method:
lr_decay = ReduceLROnPlateau(monitor='loss', 
                             patience=1, verbose=0, 
                             factor=0.5, min_lr=1e-8)
# Define Early Stopping:
early_stop = EarlyStopping(monitor='val_acc', min_delta=0, 
                           patience=30, verbose=1, mode='auto',
                           baseline=0, restore_best_weights=True)
# Train the model. 
# The dataset is small for NN - let's use test_data for validation
start = time()
History = model.fit(X_train, y_train,
                    epochs=EPOCH,
                    batch_size=BATCH,
                    validation_split=0.0,
                    validation_data=(X_test[:M_TEST], y_test[:M_TEST]),
                    shuffle=True,verbose=0,
                    callbacks=[lr_decay, early_stop])
print('-'*65)
print(f'Training was completed in {time() - start:.2f} secs')
print('-'*65)
# Evaluate the model:
train_loss, train_acc = model.evaluate(X_train, y_train,
                                       batch_size=M_TRAIN, verbose=0)
test_loss, test_acc = model.evaluate(X_test[:M_TEST], y_test[:M_TEST],
                                     batch_size=M_TEST, verbose=0)
print('-'*65)
print(f'train accuracy = {round(train_acc * 100, 4)}%')
print(f'test accuracy = {round(test_acc * 100, 4)}%')
print(f'test error = {round((1 - test_acc) * M_TEST)} out of {M_TEST} examples')

# Plot the loss and accuracy curves over epochs:
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18,6))
axs[0].plot(History.history['loss'], color='b', label='Training loss')
axs[0].plot(History.history['val_loss'], color='r', label='Validation loss')
axs[0].set_title("Loss curves")
axs[0].legend(loc='best', shadow=True)
axs[1].plot(History.history['acc'], color='b', label='Training accuracy')
axs[1].plot(History.history['val_acc'], color='r', label='Validation accuracy')
axs[1].set_title("Accuracy curves")
axs[1].legend(loc='best', shadow=True)
plt.show()
y_hat = model.predict_classes(X_test, batch_size=M_TEST, verbose=1)
#score = sum(y_hat == y_test) / len(y_test)
#print(f'Prediction accuracy = {score*100}%')
index = pd.date_range(start='2017-01-02', end='2018-06-19', freq='B')
for i in range(y_hat.shape[0]):
    print(index[i], y_hat[i], y_test[i])
# Let's define a EDA function for repeated calls on individual time series:

import statsmodels.api as sm  # seasonal trend decomposition
from statsmodels.graphics import tsaplots   # autocorrelation

def eda(df_name, ts_name, decomp_model='additive'):
    """ 
    Inputs: df_name - name of the dataframe
            ts_name - name of the time series in the dataframe
            decomp_model - 'additive'/'multiplicative'
    Outputs: EDA statistics and plots for individual time series in df_name
    """
    # Statistics
    print(f'Statistic of {ts_name} time series')
    print(df_name[ts_name].describe())
    
    # Plotting
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(24,24))
    fig.suptitle(f'Visualization of the "{ts_name}" time series', fontsize=24)
        
    # Observed values of the time series against target_class values
    df_name[ts_name].plot(ylim=[df_name[ts_name].min(), df_name[ts_name].max()],
                          linewidth=2, ax=axs[0,0])
    axs[0,0].set_title('Observed values (red/green where target_class=0/1)')
    axs[0,0].set_xlabel('')
    axs[0,0].fill_between(df_name.index, df_name[ts_name], 
                          where=(df_name.target_class==0),
                          facecolor='red', alpha=0.5)
    axs[0,0].fill_between(df_name.index, df_name[ts_name], 
                          where=(df_name.target_class==1),
                          facecolor='green', alpha=0.5)
    axs[0,0].axvline('2017-01-01', color='red', linestyle='dashed')
    
    # Seasonality, trend and noise in time series data
    decomp = sm.tsa.seasonal_decompose(df_name[ts_name],
                                       model=decomp_model)
    decomp.trend.plot(linewidth=2, ax=axs[0,1])
    axs[0,1].set_title('Trend values')
    axs[0,1].set_xlabel('')
    decomp.seasonal.plot(linewidth=2, ax=axs[1,0])
    axs[1,0].set_title('Seasonal values')
    axs[1,0].set_xlabel('')
    decomp.resid.plot(linewidth=2, ax=axs[1,1])
    axs[1,1].set_title('Residual values')
    axs[1,1].set_xlabel('')
    
    # Distribution of values of time series
    df_name[ts_name].plot.hist(bins=30, ax=axs[2,0])
    axs[2, 0].set_title('Histogram')
    df_name[[ts_name]].boxplot(ax=axs[2,1])
    axs[2, 1].set_title('Boxplot')
        
    # Autocorrelation of time series
    tsaplots.plot_acf(df_name[ts_name], lags=40, ax=axs[3,0])
    tsaplots.plot_pacf(df_name[ts_name], lags=40, ax=axs[3,1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
# Call EDA function to explore the time series
eda(df, 'target_value', 'multiplicative')
# Call EDA function to explore the time series
eda(df, 'feature_1', 'multiplicative')
# Call EDA function to explore the time series
eda(df, 'feature_2', 'multiplicative')
# Call EDA function to explore the time series
eda(df, 'feature_3', 'multiplicative')
# Call EDA function to explore the time series
eda(df, 'feature_4', 'multiplicative')
# Call EDA function to explore the time series
eda(df, 'feature_5')
# Call EDA function to explore the time series
eda(df, 'feature_6')
# Call EDA function to explore the time series
eda(df, 'feature_7')
# Call EDA function to explore the time series
eda(df, 'feature_8')
# Call EDA function to explore the time series
eda(df, 'feature_9')
# Call EDA function to explore the time series
eda(df, 'feature_10')
# Call EDA function to explore the time series
eda(df, 'feature_11')
# Call EDA function to explore the time series
eda(df, 'feature_12')
# Call EDA function to explore the time series
eda(df, 'feature_13')
# Call EDA function to explore the time series
eda(df, 'feature_14')
# Call EDA function to explore the time series
eda(df, 'feature_15')
# Call EDA function to explore the time series
eda(df, 'feature_16')
