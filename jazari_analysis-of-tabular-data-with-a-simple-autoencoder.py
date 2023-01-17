
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
rawdata = pd.read_csv("../input/churn_data.csv")
rawdata.dtypes
rawdata.pop('customerID');
rawdata = rawdata.replace(r'^\s+$', np.nan, regex=True)
rawdata.isnull().sum()
rawdata[rawdata.isnull().any(axis=1)]
rawdata.TotalCharges = rawdata.TotalCharges.replace(np.nan, 0.0)
rawdata.TotalCharges = pd.to_numeric(rawdata.TotalCharges)
rawdata[rawdata.select_dtypes('object').columns] = rawdata.select_dtypes('object').apply(lambda x: x.astype('category'))
rawdata.dtypes
rawdata.SeniorCitizen.unique()
rawdata.SeniorCitizen = rawdata.SeniorCitizen.astype('category')
rawdata.SeniorCitizen = rawdata.SeniorCitizen.replace(0, 'No')
rawdata.SeniorCitizen = rawdata.SeniorCitizen.replace(1, 'Yes')
count_churned = pd.value_counts(rawdata['Churn'])
count_churned.plot(kind='bar', rot=0)
plt.title('Churn class distribution')
plt.xticks(range(2), ['Active', 'Churned'])
plt.xlabel("Class")
plt.ylabel("Frequency");
churned = rawdata[rawdata.Churn == 'Yes']
active = rawdata[rawdata.Churn == 'No']
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Tenure per Account by Class')

bins = 24

ax1.hist(churned.tenure, bins = bins)
ax1.set_title('Churned')

ax2.hist(active.tenure, bins = bins)
ax2.set_title('Active')

plt.xlabel('Tenure (months)')
plt.ylabel('Number of Accounts')
plt.xlim((0, rawdata.tenure.max()+10))
plt.show();
f, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
f.suptitle('Monthly Charges per Account by Class')

bins = 50

ax1.hist(churned.MonthlyCharges, bins = bins)
ax1.set_title('Churned')

ax2.hist(active.MonthlyCharges, bins = bins)
ax2.set_title('Active')

plt.xlabel('Monthly Charge ($)')
plt.ylabel('Number of Accounts')
plt.xlim((rawdata.MonthlyCharges.min()-10, rawdata.MonthlyCharges.max()+10))
plt.yscale('log')
plt.show();
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Total Charged per Account by Class')

bins = 80

ax1.hist(churned.TotalCharges, bins = bins)
ax1.set_title('Churned')

ax2.hist(active.TotalCharges, bins = bins)
ax2.set_title('Active')

plt.xlabel('Total Charged ($)')
plt.ylabel('Number of Accounts')
plt.xlim((0, rawdata.TotalCharges.max()+100))
plt.yscale('log')
plt.show();
from sklearn.preprocessing import StandardScaler
data = rawdata.copy(deep = True)
data['tenure'] = StandardScaler().fit_transform(data['tenure'].values.reshape(-1, 1))
data['MonthlyCharges'] = StandardScaler().fit_transform(data['MonthlyCharges'].values.reshape(-1, 1))
data['TotalCharges'] = StandardScaler().fit_transform(data['TotalCharges'].values.reshape(-1, 1))
from sklearn.model_selection import train_test_split
RANDOM_SEED = 47
X_train, X_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
X_train = X_train[X_train.Churn == 'No']
X_train = X_train.drop(['Churn'], axis=1)

X_train_noisy = X_train.copy(deep = True)
columns = X_train_noisy.select_dtypes(['object', 'category']).columns
for col in columns:
    X_train_noisy[col] = X_train_noisy[[col]].apply(lambda x: np.random.choice(X_train[col].unique()), axis = 1)
    
columns = X_train_noisy.select_dtypes(['int64', 'float64']).columns
for col in columns:
    X_train_noisy[col] = X_train_noisy[[col]].apply(lambda x: np.random.normal(X_train[col].mean(), X_train[col].std()), axis = 1)
    
X_train = pd.get_dummies(X_train, columns=X_train.select_dtypes(['object', 'category']).columns)
X_train_noisy = pd.get_dummies(X_train_noisy, columns=X_train_noisy.select_dtypes(['object', 'category']).columns)

y_test = X_test['Churn']
X_test = X_test.drop(['Churn'], axis=1)
X_test= pd.get_dummies(X_test, columns=X_test.select_dtypes(['object', 'category']).columns)
X_train = X_train.values
X_test = X_test.values
X_train.shape
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
nb_epoch = 200
batch_size = 32
learning_rate = 1e-3
input_dim = X_train.shape[1]
encoding_dim = 10

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", 
                activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = Dense(encoding_dim, activation='relu')(decoder)
output_layer = Dense(input_dim, activation='tanh')(decoder)
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.z1",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
error_df.describe()
fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 'No') & (error_df['reconstruction_error'] < 10)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 'Yes')]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.z1",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history_noisy = autoencoder.fit(X_train_noisy, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
plt.plot(history_noisy['loss'])
plt.plot(history_noisy['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');
predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - predictions, 2), axis=1)
error_df = pd.DataFrame({'reconstruction_error': mse,
                        'true_class': y_test})
error_df.describe()
fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 'No') & (error_df['reconstruction_error'] < 10)]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)
fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = error_df[(error_df['true_class']== 'Yes')]
_ = ax.hist(normal_error_df.reconstruction_error.values, bins=10)