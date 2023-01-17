import json

import zipfile

from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

pd.set_option('precision', 2)



from imblearn.combine import SMOTETomek

from imblearn.over_sampling import SMOTE

from imblearn.pipeline import Pipeline as imbPipeline

from imblearn.under_sampling import RandomUnderSampler



from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from sklearn.metrics import (confusion_matrix, plot_confusion_matrix,

                             plot_precision_recall_curve)

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler



import tensorflow as tf

from tensorflow.keras import regularizers

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model, Sequential

from tqdm.keras import TqdmCallback

tf.random.set_seed(42)

print(tf.__version__)
seed = 1

accuracy = 'average_precision' # equals to area under recall precision curve

lr_max_iterations = 10000 # increased max_iter to allow lbfgs-solver converging

results = []
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")



assert(df.shape[0] == 284807) # make sure the data is loaded as expected

assert(df.shape[1] == 31)
X_orig = df.drop('Class', axis=1)

y_orig = df.Class



X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2,

                                                    random_state=seed, stratify=y_orig)

test_data = [X_test, y_test]



print(f"Training data class counts:\n{y_train.value_counts()}")

print('')

print(f"Test data class counts:\n{y_test.value_counts()}")

print('')

assert(y_test.shape[0]/y_orig.shape[0] > 0.19)
scaler = StandardScaler() 

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



test_data_scaled = [X_test_scaled, y_test]
def evaluate_model(test_data, estimator, name='Not specified'):

    

    X_test = test_data[0]

    y_test = test_data[1]

    

    num_fraud_cases_in_test = len(y_test[y_test==1])

    num_normal_cases_in_test = len(y_test[y_test==0])

    

    predictions = estimator.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    

    # Plot normalized confusion matrix and precision recall curve

    fig, axes = plt.subplots(1,2, figsize=(14,6))

    

    plot_confusion_matrix(estimator, X_test, y_test, normalize='true',

                          display_labels=['Normal', 'Fraud'], cmap='Greens', values_format=".4f", ax=axes[0])

    axes[0].set_title('Confusion Matrix (normalized)')

    

    prc = plot_precision_recall_curve(estimator, X_test, y_test, name=name, ax=axes[1])

    axes[1].set_title('Precision Recall Curve')

    plt.tight_layout()

    

    # Print summary    

    print(f"Classified \t{cm[1,1]} out of {num_fraud_cases_in_test} \tfraud cases correctly")

    print(f"Misclassified \t{cm[0,1]} out of {num_normal_cases_in_test} normal cases")

    print(f"Average preicision score is {prc.average_precision:.4f}")

    

    return [name, cm[1,1], cm[1,1]/num_fraud_cases_in_test*100, cm[0,1], prc.average_precision]
simple_LR = LogisticRegression(max_iter=lr_max_iterations, random_state=seed) 

simple_LR.fit(X_train_scaled, y_train)

res = evaluate_model(test_data_scaled, simple_LR, name='Logistic Regression Baseline')

results.append(res)
X_train_0 = X_train[y_train == 0]

X_train_AE = X_train_0.sample(frac=0.5, random_state=seed)

X_train_AE_scaled = StandardScaler().fit_transform(X_train_AE)



X_train_est = X_train.drop(X_train_AE.index)

y_train_est = y_train.drop(X_train_AE.index)



latent_scaler = StandardScaler()

X_train_est_scaled = latent_scaler.fit_transform(X_train_est)

X_test_est_scaled = latent_scaler.transform(X_test)
def create_autoencoder(input_dim=30, latent_dim=50):

    """ Creates an Autoencoder Model where input_dim is the number of features.

    The encoding part uses L1-regularization as sparsity constraint """

    

    input_layer = Input(shape=(input_dim,), name='Input')

    encoded = Dense(100, activation='relu', activity_regularizer=regularizers.l1(10e-5), name='Encoding')(input_layer)

    latent =  Dense(latent_dim, activation='relu', name='Latent')(encoded)

    decoded = Dense(100, activation='relu', name='Decoding')(latent)

    output_layer = Dense(input_dim, activation='linear', name='Output')(decoded)

    

    autoencoder = Model(input_layer, output_layer)

    return autoencoder
autoencoder = create_autoencoder()

autoencoder.compile(optimizer="adadelta", loss="mse")

autoencoder.summary()

    

history = autoencoder.fit(X_train_AE_scaled, X_train_AE_scaled,

                          batch_size=64, epochs=500, verbose=0, validation_split=0.15,

                          callbacks=[TqdmCallback(), EarlyStopping(patience=3)])
plt.figure(figsize=(12,8))

plt.plot(history.history['loss'], label='loss')

plt.plot(history.history['val_loss'], label='val loss')

plt.ylabel('MSE')

plt.xlabel('No. epoch')

plt.legend(loc="upper left")

plt.title("Autoencoder Training History")

plt.show()
def create_encoder(autoencoder):

    encoder = Sequential([autoencoder.layers[0],

                          autoencoder.layers[1],

                          autoencoder.layers[2]])

    return encoder
encoder = create_encoder(autoencoder)

encoder.summary()
X_train_latent = encoder.predict(X_train_est_scaled)

X_test_latent = encoder.predict(X_test_est_scaled)

test_data_latent = [X_test_latent, y_test]
log_reg_cv = LogisticRegressionCV(Cs=5, scoring=accuracy, max_iter=lr_max_iterations, n_jobs=-1, random_state=seed)

log_reg_cv.fit(X_train_latent, y_train_est)

res = evaluate_model(test_data_latent, log_reg_cv, name='Autoencoder')

results.append(res)
oversampler = SMOTE(sampling_strategy='minority', n_jobs=-1, random_state=seed)

log_reg_os  = LogisticRegressionCV(Cs=5, scoring=accuracy, max_iter=lr_max_iterations, n_jobs=-1, random_state=seed)



pipeline = imbPipeline([

    ('sampler', oversampler),

    ('transformer', StandardScaler()),

    ('classification', log_reg_os)])



pipeline.fit(X_train, y_train)



res = evaluate_model(test_data, pipeline, name='Oversampling')

results.append(res)
undersampler = RandomUnderSampler(sampling_strategy='majority', random_state=seed)

log_reg_us   = LogisticRegressionCV(Cs=5, scoring=accuracy, max_iter=lr_max_iterations, n_jobs=-1, random_state=seed)



pipeline = imbPipeline([

        ('sampler', undersampler),

        ('transformer', StandardScaler()),

        ('estimator', log_reg_us)])



pipeline.fit(X_train, y_train)



res = evaluate_model(test_data, pipeline, 'Undersampling')

results.append(res)
combined_sampler = SMOTETomek(n_jobs=-1, random_state=seed)

log_reg_comb   = LogisticRegressionCV(Cs=5, scoring=accuracy, max_iter=lr_max_iterations, n_jobs=-1, random_state=seed)



pipeline = imbPipeline([

        ('sampler', combined_sampler),

        ('transformer', StandardScaler()),

        ('estimator', log_reg_comb)])



pipeline.fit(X_train, y_train)



res = evaluate_model(test_data, pipeline, 'Combined Sampling')

results.append(res)
pd.DataFrame(results,

             columns=["Model", "True Negatives (TN)", "TN in %", "False Positives (FP)", "AUPRC"]

            ).set_index("Model")