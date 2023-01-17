import numpy as np

import pandas as pd
df = pd.read_csv('../input/pump-sensor-data/sensor.csv')
df.shape

df.columns

100*df.isna().sum()/len(df)
#Remove sensor_15 and sensor_50, too many missing values
df = df.drop(['sensor_15','sensor_50','Unnamed: 0'],axis=1)
100*df['machine_status'].value_counts()/len(df)
columns = [col for col in df.columns if not col.find('sensor')]

for col in columns:

    df[col] = df[col].fillna(df[col].mean())
df.isna().sum()
corr = df.corr()
import seaborn as sns

from matplotlib import pyplot as plt



corr80 = corr[abs(corr)> 0.8]

sns.heatmap(corr80)
from sklearn.decomposition import PCA



data = df[[sensor for sensor in df.columns.tolist() if not sensor.find('sensor')]]



pca = PCA(n_components=2)

data_reduce = pca.fit_transform(data)
print(pca.explained_variance_ratio_)
import matplotlib.pyplot as plt

plt.plot(data_reduce)
data_reduce
from statsmodels.tsa.api import VAR
model = VAR(pd.DataFrame(data_reduce))
results = model.fit(2)
results.summary()
results.plot()
results.plot_acorr()
lag_order = results.k_ar

print(lag_order)

results.plot()
plt.plot(results.forecast(pd.DataFrame(data_reduce).values[-lag_order:], 5))
results.plot_forecast(200000)
!pip install pyjanitor
import datetime



format_str = '%Y-%m-%d %H:%M:%S'  # The format



df['timestamp'] = pd.DataFrame([datetime.datetime.strptime(date, format_str) for date in df['timestamp']])

#y = df.groupby(df.timestamp.dt.year).count()['Date'].values

#x = df.groupby(df.timestamp.dt.year).count()['Date'].index
import janitor



def calculate_rul(df):

    

    df_broken = df[df['machine_status'] == 'BROKEN']

    L=[]

    d = df['timestamp'].iloc[0]

    for date_broken in df_broken['timestamp']:

        

        df_filtered = df.filter_date('timestamp',d, date_broken)

        df_filtered['rul'] = date_broken - df_filtered['timestamp']

        

        L.append(df_filtered)

        d = date_broken

    

    #L.append(df.filter_date('timestamp',df_broken['timestamp'].iloc[-1], df['timestamp'].iloc[-1]))

    result = pd.concat(L)

    



    return result.loc[~result.index.duplicated(keep='first')] #remove duplicates indices



df_rul = calculate_rul(df)
df_rul['rul']



delta_t = np.timedelta64(4, 'D')

delta_t_2 = np.timedelta64(2, 'D')
# Label1 indicates a failure will occur within the next 30 cycles.

# 1 indicates failure, 0 indicates healthy 

df_rul['label1'] = np.where(df_rul['rul'] <= delta_t, 1, 0 )
from sklearn.preprocessing import MinMaxScaler



cols_normalize = df_rul.columns.difference(['id','timestamp','rul','label1','label2','machine_status'])



scaler = MinMaxScaler()



df_rul[cols_normalize] = scaler.fit_transform(df_rul[cols_normalize])
df_rul["rul"] = df_rul["rul"].apply(lambda x: x.total_seconds()/3600) #converting to hours
#separate in train and test dataset

df_rul[df_rul['machine_status'] =="BROKEN"]



df_rul['time_norm']= df_rul['timestamp'].values.tolist()

df_rul["time_norm"] = (df_rul["time_norm"]-df_rul["time_norm"].min())/(df_rul["time_norm"].max()-df_rul["time_norm"].min())



train_df = df_rul.iloc[:77790,:].drop(['machine_status',"timestamp"],axis=1)

train_df["time_norm"] = (train_df["time_norm"]-train_df["time_norm"].min())/(train_df["time_norm"].max()-train_df["time_norm"].min())





test_df = df_rul.iloc[77791:,:].drop(['machine_status',"timestamp"],axis=1)

test_df["time_norm"] = (test_df["time_norm"]-test_df["time_norm"].min())/(test_df["time_norm"].max()-test_df["time_norm"].min())



train_df['rul'] = train_df['rul'].values.tolist()
test_df['rul'].describe()
features_col_name=[column for column in df.columns if not column.find('sensor')]

target_col_name='label1'



features_col_name
# window size

seq_length=100

seq_cols=features_col_name
def gen_sequence(id_df, seq_length, seq_cols):

    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing

    we need to drop those which are below the window-length. An alternative would be to pad sequences so that

    we can use shorter ones """

    # for one id I put all the rows in a single matrix

    data_matrix = id_df[seq_cols].values

    num_elements = data_matrix.shape[0]

    # Iterate over two lists in parallel.

    # For example id1 have 192 rows and sequence_length is equal to 50

    # so zip iterate over two following list of numbers (0,112),(50,192)

    # 0 50 -> from row 0 to row 50

    # 1 51 -> from row 1 to row 51

    # 2 52 -> from row 2 to row 52

    # ...

    # 111 191 -> from row 111 to 191

    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):

        yield data_matrix[start:stop, :]



# function to generate labels

def gen_labels(id_df, seq_length, label):

    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing

    we need to drop those which are below the window-length. An alternative would be to pad sequences so that

    we can use shorter ones """

    # For one id I put all the labels in a single matrix.

    # For example:

    # [[1]

    # [4]

    # [1]

    # [5]

    # [9]

    # ...

    # [200]] 

    data_matrix = id_df[label].values

    num_elements = data_matrix.shape[0]

    # I have to remove the first seq_length labels

    # because for one id the first sequence of seq_length size have as target

    # the last label (the previus ones are discarded).

    # All the next id's sequences will have associated step by step one label as target.

    return data_matrix[seq_length:num_elements, :]
# generate X_train for LSTM 3D dimension array

seq_gen = [list(gen_sequence(train_df, seq_length, seq_cols))]



seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

print(seq_array.shape)
label_gen = [gen_labels(train_df, seq_length, ['label1'])]



label_array = np.concatenate(label_gen).astype(np.float32)

label_array.shape
from sklearn.metrics import confusion_matrix, recall_score, precision_score

from keras.models import Sequential

from sklearn import datasets

from keras.layers import Dense, Dropout, LSTM, Activation
from sklearn.metrics import fbeta_score

from keras import backend as K





def fbeta(y_true, y_pred, threshold_shift=0):

    beta = 10



    # just in case of hipster activation at the final layer

    y_pred = K.clip(y_pred, 0, 1)



    # shifting the prediction threshold from .5 if needed

    y_pred_bin = K.round(y_pred + threshold_shift)



    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()

    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))

    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))



    precision = tp / (tp + fp)

    recall = tp / (tp + fn)



    beta_squared = beta ** 2

    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())
from sklearn.metrics import fbeta_score



# build the LSTM network

# Feature weights

nb_features = seq_array.shape[2]

nb_out = label_array.shape[1]



# LSTM model

model = Sequential()



# The first layer

model.add(LSTM(

         input_shape=(seq_length, nb_features),

         units=100,

         return_sequences=True))



# Plus a 20% dropout rate

model.add(Dropout(0.2))



# The second layer

model.add(LSTM(

          units=50,

          return_sequences=False))



# Plus a 20% dropout rate

model.add(Dropout(0.2))



# Dense sigmoid layer

model.add(Dense(units=nb_out, activation='sigmoid'))



# With adam optimizer and a binary crossentropy loss. We will opimize for model accuracy.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[fbeta])



# Verify the architecture 

print(model.summary())
import keras

import time





t0 = time.time()

# fit the network

model.fit(seq_array, # Training features

          label_array, # Training labels

          epochs=10,   # We'll stop after 10 epochs

          batch_size=200, # 

          validation_split=0.10, # Use 10% of data to evaluate the loss. (val_loss)

          verbose=1, #

          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss

                                                     min_delta=0,    # until it doesn't change (or gets worse)

                                                     patience=5,  # patience > 1 so it continutes if it is not consistently improving

                                                     verbose=0, 

                                                     mode='auto')])



print("Training took "+str(time.time() - t0)+" seconds")
# training metrics

scores = model.evaluate(seq_array, label_array, verbose=1, batch_size=200)

print('Training Accurracy: {}'.format(scores[1]))
# make predictions and compute confusion matrix

y_pred = model.predict_classes(seq_array,verbose=1, batch_size=200)

y_true = label_array

print('Training Confusion matrix\n- x-axis is true labels.\n- y-axis is predicted labels')

cm = confusion_matrix(y_true, y_pred)

cm
from sklearn.metrics import fbeta_score



# compute precision and recall

precision = precision_score(y_true, y_pred)

recall = recall_score(y_true, y_pred)

f1 = 2 * (precision * recall) / (precision + recall)

print( 'Training Precision: ', precision, '\n', 'Training Recall: ', recall, '\n', 'Training F1 Score:', f1)

print(fbeta_score(y_true, y_pred,beta=0.5))