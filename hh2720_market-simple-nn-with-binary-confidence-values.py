import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import CountVectorizer
from keras import optimizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # graphing
import seaborn as sb # visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
print('data loaded')
print('There are ', len(market_train_df['returnsOpenNextMktres10']), 'data before data selection')
market_train_df = market_train_df.loc[market_train_df['time'] >= '2010-01-01 22:00:00+0000']
print('There are ', len(market_train_df['returnsOpenNextMktres10']), 'data after data selection')
# print out the number of NaNs in each column
market_train_df.isna().sum()
print('There are ', len(market_train_df['returnsOpenNextMktres10']), 'data before dropping NaN')
market_train_df = market_train_df.dropna()
print('There are ', len(market_train_df['returnsOpenNextMktres10']), 'data after dropping NaN')
# print out the number of NaNs in each column again -> should be 0
market_train_df.isna().sum()
# Heat map before removing outliers
Corr_matrix = market_train_df.corr()
fig = plt.figure(figsize=(15,15))
sb.heatmap(Corr_matrix,vmax=0.5,square=True,annot=True)
plt.show()
def remove_outlier_by_percentile(df,col_list,lower_p,upper_p):
    """
    this function removes outliers given percentile boundaries
    lower_p = lower percentile
    upper_p = upper percentile
    """
    for i in range(len(col_list)):
        df = (df[(df[col_list[i]]<np.percentile(df[col_list[i]],upper_p)) & (df[col_list[i]]>np.percentile(df[col_list[i]],lower_p))])
    return df

outlier_removal_list = [ 'returnsClosePrevRaw1',
                         'returnsOpenPrevRaw1',
                         'returnsClosePrevRaw10',
                         'returnsOpenPrevRaw10',
                         'returnsOpenNextMktres10']

print('There are ', len(market_train_df['returnsOpenNextMktres10']), 'data before removing outliers')
market_train_df = remove_outlier_by_percentile(market_train_df,outlier_removal_list,2,98)
print('There are ', len(market_train_df['returnsOpenNextMktres10']), 'data after removing outliers')
# create a new column based on the close-open ratio
'''
market_train_df['close_open_ratio'] = np.abs(market_train_df['close']/market_train_df['open'])
'''

# print out the number of outliers
'''
threshold = 0.5
print('In %i lines price increases by 50%% or more in a day' %(market_train_df['close_open_ratio']>=(1+threshold)).sum())
print('In %i lines price decreases by 50%% or more in a day' %(market_train_df['close_open_ratio']<=(1-threshold)).sum())
'''
# clip away outliers
'''
market_train_df = market_train_df.loc[market_train_df['close_open_ratio'] < 1.5]
market_train_df = market_train_df.loc[market_train_df['close_open_ratio'] > 0.5]
'''
# print out the numeber of outliers again to check -> should be 0
'''
print('In %i lines price increases by 50%% or more in a day' %(market_train_df['close_open_ratio']>=1.5).sum())
print('In %i lines price decreases by 50%% or more in a day' %(market_train_df['close_open_ratio']<=0.5).sum())
market_train_df = market_train_df.drop(columns=['close_open_ratio'])
'''
# Heat map after removing outliers
Corr_matrix = market_train_df.corr()
fig = plt.figure(figsize=(15,15))
sb.heatmap(Corr_matrix,vmax=0.5,square=True,annot=True)
plt.show()
# Use vectorizer to transform the column from a Series of texts to sparse matrix
'''
assetCode_column = market_train_df['assetCode']
vectorizer = CountVectorizer()
vectorizer.fit(assetCode_column)
vectorized_texts = vectorizer.transform(assetCode_column)
market_train_df['assetCode'] = vectorized_texts
market_train_df.head()
'''
from sklearn.preprocessing import StandardScaler
def scale_data(df,features):
    scaler = StandardScaler()
    df[features]=scaler.fit_transform(df[features])
    return df
features = ['volume','close','open','returnsClosePrevRaw1','returnsOpenPrevRaw1',
            'returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevRaw10',
            'returnsOpenPrevRaw10','returnsClosePrevMktres10','returnsOpenPrevMktres10', 'returnsOpenNextMktres10'] 
market_train_df = scale_data(market_train_df,features)
# visualize standardized data
market_train_df[features].head()
# specify the ratio of validation data set
valid_ratio = 0.20

# split market data into features and outputs
features = ['volume','close','open','returnsClosePrevRaw1','returnsOpenPrevRaw1',
            'returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevRaw10',
            'returnsOpenPrevRaw10','returnsClosePrevMktres10','returnsOpenPrevMktres10']
X = market_train_df[features]
Y = market_train_df['returnsOpenNextMktres10']
X_train, X_valid, Y_train, Y_valid = train_test_split(X,Y,test_size=valid_ratio)

# check training and validation set sizes
len(X_train),len(Y_train), len(X_valid),len(Y_valid)
# plot history function
def plot_history(history):
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))
    #plt.subplot(1, 2, 1)
    # plt.plot(x, acc, 'b', label='Training acc')
    # plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation loss')
    # plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.legend()
    plt.show()
    plt.close()
    
from keras.callbacks import ModelCheckpoint,EarlyStopping

# early stopping is used to prevent overtraining -> we will stop the training "early" if it has reached maximum accuracy
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto',restore_best_weights=True)
callbacks_list = [early_stopping]
print('functions loaded')
input_dim = X_train.shape[1]
input_dim
# specify the input and output dimensions of the neural network
input_dim = X_train.shape[1]
output_dim = 1

# specify dropout rate -> to prevent overfitting
dropout_rate = 0.2

# specify optimizer with gradient clipping and learning rate
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
# adam = optimizers.Adam(lr=0.001)

model = Sequential()
# input and dropout layer
model.add(Dense(11, input_dim=input_dim, activation = 'relu'))
# model.add(Dropout(rate=dropout_rate))
# hidden and dropout layer
model.add(Dense(6, activation = 'relu'))
model.add(Dropout(rate=dropout_rate))
# output layer
model.add(Dense(output_dim))
# Compile the architecture and view summary
model.compile(loss='mean_squared_error',optimizer=sgd)
history = model.fit(X_train, Y_train,
                    epochs=20,
                    verbose=1,
                    validation_data = (X_valid, Y_valid),
                    callbacks=callbacks_list,
                    shuffle=True,
                    batch_size=128)
print(model.summary())

# print the losses and accuracies of the model on training and validation set  
loss = model.evaluate(X_train, Y_train, verbose=False)
print("Training Loss: {:.4f}".format(loss))
loss = model.evaluate(X_valid, Y_valid, verbose=False)
print("Validation Loss: {:.4f}".format(loss))
# plot history
plot_history(history)
data = {'y_real':Y_valid[0:15],'y_pred':(model.predict(X_valid.values[0:15])).reshape(1,-1)[0]}
pd.DataFrame(data)
data = {'y_train':Y_train[0:15],'y_pred':(model.predict(X_train.values[0:15])).reshape(1,-1)[0]}
pd.DataFrame(data)
_df = pd.DataFrame((model.predict(X_train.values[0:15])))
_df[0]
all_Y = market_train_df[['returnsOpenNextMktres10']]
all_Y.head()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(all_Y)
print('scaler fitting done')
def make_prediction(df):
    pred = pd.DataFrame(model.predict(df))
    scaled_pred = pd.DataFrame(scaler.transform(pred[[0]]))
    scaled_pred[scaled_pred>0]=1
    scaled_pred[scaled_pred<0]=-1
    pred_arr = np.array(scaled_pred).reshape(1,-1)[0]
    return pred_arr

print('function loaded')
my_pred = make_prediction(X_train.head())
my_pred
temp_df = X_train.head().copy()
temp_df['confidenceValue'] = my_pred

temp_df
days = env.get_prediction_days()
print('days loaded')
n_days = 0
for (market_obs_df, _, predictions_template_df) in days: 
    n_days += 1
    print(n_days)
    features = ['volume','close','open','returnsClosePrevRaw1','returnsOpenPrevRaw1',
            'returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevRaw10',
            'returnsOpenPrevRaw10','returnsClosePrevMktres10','returnsOpenPrevMktres10']
    returns_features = ['returnsClosePrevRaw10','returnsClosePrevMktres1','returnsOpenPrevMktres1',
                         'returnsOpenPrevRaw10','returnsClosePrevMktres10','returnsOpenPrevMktres10']
    market_obs_df_scaled = scale_data(market_obs_df,features)    
    x_submission = market_obs_df_scaled[features].copy()
    # fill in NaN values with mean of rest of the values
    for i in range(len(returns_features)):
         x_submission[returns_features[i]]= x_submission[returns_features[i]].fillna(x_submission[returns_features[i]].mean())
    pred = make_prediction(x_submission)
    print(pred)
    predictions_template_df['confidenceValue'] = pred
    env.predict(predictions_template_df)
    del x_submission
print('Done!')
# Write submission file    
env.write_submission_file()