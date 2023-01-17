# Loading libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline  



df_train = pd.read_csv("../input/PM_train.csv")
df_train.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
df_train.rename({'pos ext temp' : 'pos X ext temp'})
df_train.head()
print('Shape of Train dataset: ',df_train.shape)
display(df_train)



df_test = pd.read_csv("../input/PM_train.csv")
df_test = df_test.dropna(axis=0, how='any', inplace=False)
df_test.rename({'pos ext temp' : 'pos X ext temp'})
df_test.head()
print('Shape of Test dataset: ',df_test.shape)
display(df_test)



df_truth = pd.read_csv("../input/PM_truth.csv")
df_truth.loc[-1] = ['112']  # adding a row
df_truth.index = df_truth.index + 1  # shifting index
df_truth.sort_index(inplace=True) 
df_truth.columns=['Passes till Anomaly']
df_truth['id'] = df_truth.index+1
df_truth.head()
# generate column max for test data
rul = pd.DataFrame(df_test.groupby('ID')['Passes'].max()).reset_index()
rul.columns = ['id', 'max']
rul.head()
# Converting column to a float
df_truth['Passes till Anomaly'] = df_truth['Passes till Anomaly'].astype(float)
display(df_truth['Passes till Anomaly'])
# run to failure
df_truth['rtf']=df_truth['Passes till Anomaly']+rul['max']
df_truth.head()
df_truth['id'] = df_truth['id'].astype(float)
display(df_truth['id'])
#df_test.dtypes
#df_truth.dtypes
df_truth.drop('Passes till Anomaly', axis=1, inplace=True)
#df_test=df_test.merge(df_truth,on=['id'],how='left')
df_test['ttf']=df_truth['rtf'] - df_test['Passes']
#dataset_test.drop('rtf', axis=1, inplace=True)
df_test.head()
df_train.head()
df_train['ttf'] = df_train.groupby(['ID'])['Passes'].transform(max)-df_train['Passes']
df_train.head()
dataset_train=df_train.copy()
dataset_test=df_test.copy()
period=30
dataset_train['label_bc'] = dataset_train['ttf'].apply(lambda x: 1 if x <= period else 0)
dataset_test['label_bc'] = dataset_test['ttf'].apply(lambda x: 1 if x <= period else 0)
dataset_train.head()
features_col_name=list(dataset_train)
features_col_name.remove('label_bc')
features_col_name.remove('ttf')
target_col_name='label_bc'
display(features_col_name)
sc=MinMaxScaler()
df_train[features_col_name]=sc.fit_transform(df_train[features_col_name])
df_test[features_col_name]=sc.transform(df_test[features_col_name])
#function to reshape dataset as required by LSTM
def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)

# Function to generate labels
def gen_label(id_df, seq_length, seq_cols,label):
    df_zeros=pd.DataFrame(np.zeros((seq_length-1,id_df.shape[1])),columns=id_df.columns)
    id_df=df_zeros.append(id_df,ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label=[]
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label][stop])
    return np.array(y_label)

# timestamp or window size
seq_length=50
seq_cols=features_col_name
# generate X_train
X_train=np.concatenate(list(list(gen_sequence(dataset_train[dataset_train['ID']==id], seq_length, seq_cols)) for id in dataset_train['ID'].unique()))
print(X_train.shape)
# generate y_train
y_train=np.concatenate(list(list(gen_label(dataset_train[dataset_train['ID']==id], 50, seq_cols,'label_bc')) for id in dataset_train['ID'].unique()))
print(y_train.shape)
# generate X_test
X_test=np.concatenate(list(list(gen_sequence(dataset_test[dataset_test['ID']==id], seq_length, seq_cols)) for id in dataset_test['ID'].unique()))
print(X_test.shape)
# generate y_test
y_test=np.concatenate(list(list(gen_label(dataset_test[dataset_test['ID']==id], 50, seq_cols,'label_bc')) for id in dataset_test['ID'].unique()))
print(y_test.shape)
nb_features =X_train.shape[2]
timestamp=seq_length

model = Sequential()

model.add(LSTM(
         input_shape=(timestamp, nb_features),
         units=100,
         return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
          units=50,
          return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
# fit the network
model.fit(X_train, y_train, epochs=10, batch_size=200, validation_split=0.05, verbose=1,
          callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])

# training metrics
scores = model.evaluate(X_train, y_train, verbose=1, batch_size=200)
print('Accuracy: {}'.format(scores[1]))
y_pred=model.predict_classes(X_test)
print('Accuracy of model on test data: ',accuracy_score(y_test,y_pred))
print('Confusion Matrix: \n',confusion_matrix(y_test,y_pred))
# Probability of satellite error
def prob_failure(satellite_id):
    satellite_df=dataset_test[dataset_test['ID']==satellite_id]
    satellite_test=gen_sequence(satellite_df,seq_length,seq_cols)
    m_pred=model.predict(satellite_test)
    failure_prob=list(m_pred[-1]*100)[0]
    return failure_prob
satellite_id=10
print('Probability that satellite will receive an error within 30 days: ',prob_failure(satellite_id))
