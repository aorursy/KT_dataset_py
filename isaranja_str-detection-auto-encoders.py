import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers



import matplotlib.pyplot as plt

import seaborn as sns

import pickle



from sklearn.metrics import confusion_matrix, precision_recall_curve



from keras import backend as K

K.tensorflow_backend._get_available_gpus()
# loading the log file

df = pd.read_csv('/kaggle/input/paysim1/PS_20174392719_1491204439457_log.csv')

df_bkp = df.copy()

df.head()
# statistical analysis

df[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']].describe()



#df.shape

#df.nameDest.nunique()



#total    = 6362620

#step     = 743

#type     = 5

#nameOrig = 6353307

#nameDest = 2722362
# Fraud destribution

df.isFraud.value_counts(normalize=True)*100
df = df_bkp.copy()

# numerical columns

f1 = ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest']



# log transformation

df.loc[:,f1] =  np.log(df.loc[:,f1]+1)



# scaling

from sklearn.preprocessing import MinMaxScaler # normalization

sc = MinMaxScaler()



df.loc[:,f1] = sc.fit_transform(df.loc[:,f1])



# one-hot encoding type variable

df = pd.get_dummies(df,columns=['type'],drop_first=True)



df.head()
# train/val/test split

train_x = df[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','type_CASH_OUT','type_DEBIT','type_PAYMENT','type_TRANSFER']].values

train_y = df.isFraud.values
# Autoencoder



input_dim = 9

hidden_dim = 16

code_dim = 8



input_layer = Input(shape=(input_dim, ))

encoder = Dense(hidden_dim, activation="relu",activity_regularizer=regularizers.l1(1e-6))(input_layer)

code = Dense(code_dim, activation="relu")(encoder)

decoder = Dense(hidden_dim, activation='relu')(code)

output_layer = Dense(input_dim, activation='sigmoid')(decoder)



autoencoder = Model(inputs=input_layer, outputs=output_layer)



autoencoder.summary()
autoencoder.compile(metrics=['accuracy'],

                    loss='binary_crossentropy',

                    optimizer='adam')



cp = ModelCheckpoint(filepath="autoencoder_fraud.h5",

                               save_best_only=True,

                               verbose=0)



tb = TensorBoard(log_dir='./logs',

                histogram_freq=0,

                write_graph=True,

                write_images=True)
history = autoencoder.fit(train_x, train_x,

                    epochs=10,

                    batch_size=128,

                    shuffle=True,

                    validation_split = 0.2,

                    verbose=1,

                    callbacks=[cp, tb]).history
# model loss

plt.plot(history['loss'], linewidth=2, label='Train')

plt.plot(history['val_loss'], linewidth=2, label='Test')

plt.legend(loc='upper right')

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

#plt.ylim(ymin=0.70,ymax=1)

plt.show()
# Error df prediction

train_x_predictions = autoencoder.predict(train_x)

mse = np.mean(np.power(train_x - train_x_predictions, 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,

                        'True_class': train_y})

error_df.describe()
# Recall vs Precision

precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)

plt.plot(recall_rt, precision_rt, linewidth=2, label='Precision-Recall curve')

plt.title('Recall vs Precision')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.show()
# precision recall with threshold

plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)

plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)

plt.title('Precision and recall for different threshold values')

plt.xlabel('Threshold')

plt.ylabel('Precision/Recall')

plt.legend()

plt.show()
# confusion matrix

threshold_fixed = 0.05

LABELS = ["Normal","Fraud"]



pred_y = [1 if e > threshold_fixed else 0 for e in error_df.Reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.True_class, pred_y)



plt.figure(figsize=(5, 5))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Confusion matrix")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()