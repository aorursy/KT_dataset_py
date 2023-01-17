import pandas as pd

import numpy as np

import os



# Plotting libraries

import matplotlib.pyplot as plt

import seaborn as sns



# SKLearn related libraries

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import RobustScaler



# Classifiers

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC



# Metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# Keras NN related libraries

from keras import layers

from keras.layers import Input, Dense

from keras.models import Model, Sequential 

from keras import regularizers
data_path = '/kaggle/input/creditcardfraud/creditcard.csv'



# print(os.path.exists(data_path))



# Load the data

card_df = pd.read_csv(data_path, header=0)
card_df.info()

print("===="*30)

card_df.head()
card_df.describe().T
# Unique class labels

print(f"Unique classes in the dataset are : {np.unique(card_df['Class'])}" )
card_df.groupby('Class')['Class'].count().plot.bar(logy=True)
# Change the time attribute in day

card_df['Time'] = card_df['Time'].apply(lambda t: (t/3600) % 24 )
# Sampling of data

normal_trans = card_df[card_df['Class'] == 0].sample(4000)

fraud_trans = card_df[card_df['Class'] == 1]
reduced_set = normal_trans.append(fraud_trans).reset_index(drop=True)
print(f"Cleansed dataset shape : {reduced_set.shape}")
# Splitting the dataset into X and y features

y = reduced_set['Class']

X = reduced_set.drop('Class', axis=1)

print(f"Shape of Features : {X.shape} and Target: {y.shape}")
def dimensionality_plot(X, y):

    sns.set(style='whitegrid', palette='muted')

    # Initializing TSNE object with 2 principal components

    tsne = TSNE(n_components=2, random_state = 42)

    

    # Fitting the data

    X_trans = tsne.fit_transform(X)

    

    plt.figure(figsize=(12,8))

    

    plt.scatter(X_trans[np.where(y == 0), 0], X_trans[np.where(y==0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Normal')

    plt.scatter(X_trans[np.where(y == 1), 0], X_trans[np.where(y==1), 1], marker='o', color='k', linewidth='1', alpha=0.8, label='Fraud')

    

    plt.legend(loc = 'best')

    

    plt.show()



# Invoking the method dimensionality_plot

dimensionality_plot(X, y)
scaler = RobustScaler().fit_transform(X)



# Scaled data

X_scaled_normal = scaler[y == 0]

X_scaled_fraud = scaler[y == 1]
print(f"Shape of the input data : {X.shape[1]}")
# Input layer with a shape of features/columns of the dataset

input_layer = Input(shape = (X.shape[1], ))



# Construct encoder network

encoded = Dense(100, activation= 'tanh', activity_regularizer=regularizers.l1(10e-5))(input_layer)

encoded = Dense(50, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(encoded)

encoded = Dense(25, activation='tanh', activity_regularizer=regularizers.l1(10e-5))(encoded)

encoded = Dense(12, activation = 'tanh', activity_regularizer=regularizers.l1(10e-5))(encoded)

encoded = Dense(6, activation='relu')(encoded)



# Decoder network

decoded = Dense(12, activation='tanh')(encoded)

decoded = Dense(25, activation='tanh')(decoded)

decoded = Dense(50, activation='tanh')(decoded)

decoded = Dense(100, activation='tanh')(decoded)



output_layer = Dense(X.shape[1], activation='relu')(decoded)



# Building a model

auto_encoder = Model(input_layer, output_layer)
# Compile the auto encoder model

auto_encoder.compile(optimizer='adadelta', loss='mse')



# Training the auto encoder model

auto_encoder.fit(X_scaled_normal, X_scaled_normal, batch_size=32, epochs=20, shuffle=True, validation_split=0.20)
latent_model = Sequential()

latent_model.add(auto_encoder.layers[0])

latent_model.add(auto_encoder.layers[1])

latent_model.add(auto_encoder.layers[2])

latent_model.add(auto_encoder.layers[3])

latent_model.add(auto_encoder.layers[4])
normal_tran_points = latent_model.predict(X_scaled_normal)

fraud_tran_points = latent_model.predict(X_scaled_fraud)

# Making as a one collection

encoded_X = np.append(normal_tran_points, fraud_tran_points, axis=0)

y_normal = np.zeros(normal_tran_points.shape[0])

y_fraud = np.ones(fraud_tran_points.shape[0])

encoded_y = np.append(y_normal, y_fraud, axis=0)

# Calling TSNE plot function

dimensionality_plot(encoded_X, encoded_y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_enc_train, X_enc_test, y_enc_train, y_enc_test = train_test_split(encoded_X, encoded_y, test_size=0.3)
print(f"Encoded train data X: {X_enc_train.shape}, Y: {y_enc_train.shape}, X_test :{X_enc_test.shape}, Y_test: {y_enc_test.shape}")

print(f"Actual train & test data X: {X_train.shape}, Y: {X_train.shape}, X_test :{X_test.shape}, Y_test: {y_test.shape}")
# Instance of SVM

svc_clf = SVC()



svc_clf.fit(X_train, y_train)



svc_predictions = svc_clf.predict(X_test)
print("Classification report \n {0}".format(classification_report(y_test, svc_predictions)))
print("Accuracy of SVC \n {:.2f}".format(accuracy_score(y_test, svc_predictions)))
lr_clf = LogisticRegression()



lr_clf.fit(X_enc_train, y_enc_train)



# Predict the Test data

predictions = lr_clf.predict(X_enc_test)
print("Classification report \n {0}".format(classification_report(y_enc_test, predictions)))

print("Accuracy score is : {:.2f}".format(accuracy_score(y_enc_test, predictions)))