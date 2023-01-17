# IMPORT MODULES



import numpy as np

from numpy import ma

import pandas as pd

import math

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.colors as colors

%matplotlib inline

from matplotlib import ticker, cm

from matplotlib.pyplot import figure

import seaborn as sns



from scipy.stats import multivariate_normal

from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_fscore_support

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.neighbors import LocalOutlierFactor

from sklearn.ensemble import IsolationForest

from sklearn.covariance import EllipticEnvelope

from sklearn.svm import OneClassSVM



from keras.models import Model, load_model

from keras.layers import Input, Dense

from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import regularizers



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# LOAD DATA



dfRaw = pd.read_csv('../input/creditcard.csv')

print(dfRaw.shape)

print(dfRaw.columns)
data = dfRaw.copy()

normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



print("data ", data.shape)

print("normal_data ", normal_data.shape)

print("fraud_data ", fraud_data.shape)

print("Percent fraud ", round(100*492/284807, 3),"%")

print("_"*100)

print(data.head())
# PLOT AMOUNT - Norm vs Fraud



normal_data["Amount"].loc[normal_data["Amount"] < 500].hist(bins=100);

plt.figure()

fraud_data["Amount"].loc[fraud_data["Amount"] < 500].hist(bins=100);

plt.figure()

print("Mean", normal_data["Amount"].mean(), fraud_data["Amount"].mean())

print("Median", normal_data["Amount"].median(), fraud_data["Amount"].median())
# PLOT TIME - Norm vs Fraud



normal_data["Time"].hist(bins=100);

plt.figure()

fraud_data["Time"].hist(bins=100);

plt.figure()
#  SCALER / Normalization



data = dfRaw.copy()

scl = StandardScaler()

all_cols = list(data)[:] 

pca_columns = list(data)[:-1] # all cols without Class

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

Xscaled = scl.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((Xscaled, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = XcopyALL.columns)



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



print("data ", data.shape)

print("normal_data ", normal_data.shape)

print("fraud_data ", fraud_data.shape)

print("Percent fraud ", round(100*492/284807, 4),"%")
# Check values are centered around 0 after normalization



print("data['Time'].mean()  ", data['Time'].mean())

print("data['Amount'].mean()  ", data['Amount'].mean())
# CREATE the TRAIN and TEST sets

# Fraud data is ONLY in TEST - not in TRAIN



normal_pca_data = normal_data[pca_columns]

fraud_pca_data = fraud_data[pca_columns]



num_test = 75000

shuffled_data = normal_pca_data.sample(frac=1, random_state=1960)[:-num_test].values

X_train = shuffled_data



#X_valid = np.concatenate([shuffled_data[-2*num_test:-num_test], fraud_pca_data[:246]])

#y_valid = np.concatenate([np.zeros(num_test), np.ones(246)])



X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[:]])

y_test = np.concatenate([np.zeros(num_test), np.ones(492)])



print("normal_pca_data ", normal_pca_data.shape)

print("fraud_pca_data", fraud_pca_data.shape)

print("Fraud data only in Test with NONE in the training")

print("X_train ", X_train.shape)

#print("X_valid ", X_valid.shape)

#print("y_valid ", y_valid.shape)

print("X_test ", X_test.shape)

print("y_test ", y_test.shape)
# Calculate the  prob on train vs test vs fraud data only (no normals at all)



p = multivariate_normal(mean=np.mean(X_train,axis=0), cov=np.cov(X_train.T))



x = p.pdf(X_train)

print("max prob of x on X_train", max(x))

print("mean prob of x on X_train", np.mean(x))

print('-' * 60)

MyTrain = np.mean(x)



x = p.pdf(X_test)

print("max prob of x on X_test", max(x))

print("mean prob of x on X_test", np.mean(x))

print('-' * 60)

MyTest = np.mean(x)



x = p.pdf(fraud_pca_data)

print("max prob of x on fraud_pca_data", max(x))

print("mean prob of x on fraud_pca_data", np.mean(x))

print('-' * 60)



print('Difference between mean prob of Train vs Test ', MyTrain - MyTest)

# Precision = percentage of Fraud caught

# Recall = percentage of those caught that are actually Fraud

# F1 score = Harmonic mean of P & R

# Need to optimize on the hyperparamter of EPSILON
# Find best epsilon re F1 score



x = p.pdf(X_test)



EpsF1 = []



epsilons = [1e-10, 1e-20, 1e-30, 1e-40, 1e-50, 1e-60, 1e-70, 1e-80, 1e-90, 1e-100, 1e-110, 1e-120,

           1e-130, 1e-140, 1e-150, 1e-160, 1e-170, 1e-180, 1e-190, 1e-200, 1e-210, 1e-220, 1e-230, 1e-240]



for e in range(len(epsilons)):

    eps = epsilons[e]

    pred = (x <= eps)

    f = f1_score(y_test, pred, average='binary')

    #print("F1 score on test", round(f,4), " with epsilon ", eps)

    EpsF1.append([eps, round(f,4)])

    

EpsF1df = pd.DataFrame(EpsF1, columns = ['epsilon', 'F1'])

EpsF1df.head()
# Best Epsilon ... Max F1 on test



EpsF1df.loc[EpsF1df['F1'].idxmax()]

EpsF1df.plot.line("epsilon","F1")

plt.xscale('log')

plt.xlim(1e-10, 1e-240)

plt.title("F1 vs decreasing log Epsilon")

plt.show()
# CONFUSION MATRIX and F1 SCORE on Test w best epsilon



eps = EpsF1df.loc[EpsF1df['F1'].idxmax()]['epsilon']



print("epsilon ", eps)

print("_"*50)

pred = (x<=eps)

CM = confusion_matrix(y_test,pred)

tn, fp, fn, tp = confusion_matrix(y_test,pred).ravel()



print(CM)

print("_"*50)

print("TP ", tp)

print("FP ", fp)

print("TN ", tn)

print("FN ", fn)
def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,

                          normalize=False):

    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
plot_confusion_matrix(CM, 

                      normalize    = False,

                      target_names = ['Normal', 'Fraud'],

                      title        = "Confusion Matrix")
# F1 Score

#print("F1 score", round(f1_score(y_valid,pred, average='binary'), 4))

precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_test,pred, average='binary')

print("precision ", round((precision), 3))

print("recall ", round((recall), 3))

print("F1 score on Test", round((fbeta_score), 3))
#  PCA 



data = dfRaw.copy()

print("Before PCA")

print(data.shape)

print("AFTER PCA")



pca = PCA(n_components = 0.999999) # This way of setting the components = knowledge of the VARIANCE loss during PCA



all_cols = list(data)[:] 

pca_columns = list(data)[:-1] 

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

dataPostPCA = pca.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((dataPostPCA, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = [0,1,'Class'])



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



print(data.shape)

print("data ", data.shape)

print("normal_data ", normal_data.shape)

print("fraud_data ", fraud_data.shape)

print("Percent fraud ", round(100*492/284807, 4),"%")



#print(data.head)
#postPCA = pd.DataFrame(normal_data.sample(10000))

postPCA = pd.DataFrame(data)



postPCA.plot.scatter(0,1)

plt.xlim(-100000, 100000)

plt.ylim(0, 30000)

plt.title("Normal post PCA")

plt.show()



postPCA = pd.DataFrame(fraud_data)



postPCA.plot.scatter(0,1, c='r')

plt.xlim(-100000, 100000)

plt.ylim(0, 30000)

plt.title("Fraud post PCA")

plt.show()


normal_data = normal_data.drop('Class', axis=1)

fraud_data = fraud_data.drop('Class', axis=1)
# View the FRAUD on a 2 dims (Post PCA) Guassian distribution of the normal data

# Reducing from 30 dims to 2 - helps with the visualization but surely doesn't help with separating the Fraud from the Normal



x, y = np.mgrid[-100000:100000:100, -1000:3000:100] 

pos = np.empty(x.shape + (2,))

pos[:, :, 0] = x; pos[:, :, 1] = y

rv = multivariate_normal(mean=np.mean(normal_data,axis=0), cov=np.cov(normal_data.T)) # mean and covariance matrix for 2 dims dataset



fig, ax = plt.subplots()

cs = ax.contourf(x, y, rv.pdf(pos))

cbar = fig.colorbar(cs)

plt.title("Fraud projected on normal distribution")

plt.scatter(fraud_data[0],fraud_data[1], edgecolor="r") # Location on chart of the anomaly points

fig = matplotlib.pyplot.gcf()

fig.set_size_inches(18.5, 10.5)

#plt.show()
# View some NORMAL on a 2 dims (Post PCA) Guassian distribution of the normal data

# Reducing from 30 dims to 2 - helps with the visualization but surely doesn't help with separating the Fraud from the Normal



SampleNormal = normal_data.sample(500)



x, y = np.mgrid[-100000:100000:100, -1000:3000:100] 

pos = np.empty(x.shape + (2,))

pos[:, :, 0] = x; pos[:, :, 1] = y

rv = multivariate_normal(mean=np.mean(normal_data,axis=0), cov=np.cov(normal_data.T)) # mean and covariance matrix for 2 dims dataset



fig, ax = plt.subplots()

cs = ax.contourf(x, y, rv.pdf(pos))

cbar = fig.colorbar(cs)

plt.title("Normal sample of 500 projected on normal distribution")

plt.scatter(SampleNormal[0],SampleNormal[1], edgecolor="b") # Location on chart of the anomaly points

fig = matplotlib.pyplot.gcf()

fig.set_size_inches(18.5, 10.5)

#plt.show()
data = dfRaw.copy()



print("Before PCA")

print(data.shape)

print("AFTER PCA")



pca = PCA(n_components = 0.999999) 



all_cols = list(data)[:] 

pca_columns = list(data)[:-1] 

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

dataPostPCA = pca.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((dataPostPCA, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = [0,1,'Class'])



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



print(data.shape)

print("data ", data.shape)

print("normal_data ", normal_data.shape)

print("fraud_data ", fraud_data.shape)

print("Percent fraud ", round(100*492/284807, 4),"%")
# CREATE the TRAIN and TEST sets

# Fraud data is ONLY in TEST - not in TRAIN



normal_pca_data = normal_data.drop('Class',1)

fraud_pca_data = fraud_data.drop('Class',1)



num_test = 75000

shuffled_data = normal_pca_data.sample(frac=1, random_state=1960)[:-num_test].values

X_train = shuffled_data



#X_valid = np.concatenate([shuffled_data[-2*num_test:-num_test], fraud_pca_data[:246]])

#y_valid = np.concatenate([np.zeros(num_test), np.ones(246)])



X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[:]])

y_test = np.concatenate([np.zeros(num_test), np.ones(492)])



print("normal_pca_data ", normal_pca_data.shape)

print("fraud_pca_data", fraud_pca_data.shape)

print("Fraud data only in Test with NONE in the training")

print("X_train ", X_train.shape)

#print("X_valid ", X_valid.shape)

#print("y_valid ", y_valid.shape)

print("X_test ", X_test.shape)

print("y_test ", y_test.shape)
# Calculate the  prob on train vs test vs fraud data only (no normals at all)



p = multivariate_normal(mean=np.mean(X_train,axis=0), cov=np.cov(X_train.T))



x = p.pdf(X_train)

print("max prob of x on X_train", max(x))

print("mean prob of x on X_train", np.mean(x))

print('-' * 60)

MyTrain = np.mean(x)



x = p.pdf(X_test)

print("max prob of x on X_test", max(x))

print("mean prob of x on X_test", np.mean(x))

print('-' * 60)

MyTest = np.mean(x)



x = p.pdf(fraud_pca_data)

print("max prob of x on fraud_pca_data", max(x))

print("mean prob of x on fraud_pca_data", np.mean(x))

print('-' * 60)



print('Difference between mean prob of Train vs Test ', MyTrain - MyTest)
# Find best epsilon re F1 score



x = p.pdf(X_test)



EpsF1 = []



epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]



for e in range(len(epsilons)):

    eps = epsilons[e]

    pred = (x <= eps)

    f = f1_score(y_test, pred, average='binary')

    #print("F1 score on test", round(f,4), " with epsilon ", eps)

    EpsF1.append([eps, round(f,4)])

    

EpsF1df = pd.DataFrame(EpsF1, columns = ['epsilon', 'F1'])

EpsF1df.head()
# Best Epsilon ... Max F1 on test



EpsF1df.loc[EpsF1df['F1'].idxmax()]
EpsF1df.plot.line("epsilon","F1")

plt.xscale('log')

plt.xlim(1e-1, 1e-10)

plt.title("F1 vs decreasing log Epsilon")

plt.show()
#  SCALER / Normalization



data = dfRaw.copy()

scl = StandardScaler()

all_cols = list(data)[:] 

pca_columns = list(data)[:-1] # all cols without Class

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

Xscaled = scl.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((Xscaled, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = XcopyALL.columns)



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



print("data ", data.shape)

print("normal_data ", normal_data.shape)

print("fraud_data ", fraud_data.shape)

print("Percent fraud ", round(100*492/284807, 4),"%")
# Check values are centered around 0 after normalization



print("data['Time'].mean()  ", data['Time'].mean())

print("data['Amount'].mean()  ", data['Amount'].mean())
# CREATE the TRAIN and TEST sets

# Fraud data is ONLY in TEST - not in TRAIN



normal_pca_data = normal_data[pca_columns]

fraud_pca_data = fraud_data[pca_columns]



num_test = 75000

shuffled_data = normal_pca_data.sample(frac=1, random_state=1960)[:-num_test].values

X_train = shuffled_data



X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[:]])

y_test = np.concatenate([np.zeros(num_test), np.ones(492)])



print("normal_pca_data ", normal_pca_data.shape)

print("fraud_pca_data", fraud_pca_data.shape)

print("Fraud data only in Test with NONE in the training")

print("X_train ", X_train.shape)

#print("X_valid ", X_valid.shape)

#print("y_valid ", y_valid.shape)

print("X_test ", X_test.shape)

print("y_test ", y_test.shape)
input_dim = X_train.shape[1]

encoding_dim = 14
# Keras Auto Encoder model



input_layer = Input(shape=(input_dim, ))



encoder = Dense(encoding_dim, activation="tanh", 

                activity_regularizer=regularizers.l1(10e-5))(input_layer)

encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)



decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)

decoder = Dense(input_dim, activation='relu')(decoder)



autoencoder = Model(inputs=input_layer, outputs=decoder)



autoencoder.summary()
nb_epoch = 10

batch_size = 32



autoencoder.compile(optimizer='adam', 

                    loss='mean_squared_error', 

                    metrics=['accuracy'])



checkpointer = ModelCheckpoint(filepath="model.h5",

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
# VALIDATION LOSS curves



plt.clf()

history_dict = history

loss_values = history_dict['loss']

val_loss_values = history_dict['val_loss']

epochs = range(1, (len(history_dict['loss']) + 1))

plt.plot(epochs, loss_values, 'bo', label='Training loss')

plt.plot(epochs, val_loss_values, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()



# VALIDATION ACCURACY curves



plt.clf()

acc_values = history_dict['acc']

val_acc_values = history_dict['val_acc']

epochs = range(1, (len(history_dict['acc']) + 1))

plt.plot(epochs, acc_values, 'bo', label='Training acc')

plt.plot(epochs, val_acc_values, 'r', label='Validation acc')

plt.title('Training and validation accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
# Load the best model saved above during training



autoencoder = load_model('model.h5')
# Reconstruction error on Train



# As Train has no Fraud

y_train = np.zeros(X_train.shape[0])



predictions = autoencoder.predict(X_train)

predictions.shape



mse = np.mean(np.power(X_train - predictions, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse,

                        'true_class': y_train})

print(error_df.shape[0], ' rows')

print('mean error of recon on TRAIN', round(error_df.reconstruction_error.mean(),2))

print('std error of recon on TRAIN', round(error_df.reconstruction_error.std(),2))
# Reconstruction error on Test



predictions = autoencoder.predict(X_test)

predictions.shape



mse = np.mean(np.power(X_test - predictions, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse,

                        'true_class': y_test})

print(error_df.shape[0], ' rows')

print('mean error of recon on TEST', round(error_df.reconstruction_error.mean(),2))

print('std error of recon on TEST', round(error_df.reconstruction_error.std(),2))
# Reconstruction error on Fraud



# As Fraud is all Fraud

y_Fraud = np.ones(fraud_pca_data.shape[0])

y_Fraud.shape



predictions = autoencoder.predict(fraud_pca_data)

predictions.shape



mse = np.mean(np.power(fraud_pca_data - predictions, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse,

                        'true_class': y_Fraud})

print(error_df.shape[0], ' rows')

print('mean error of recon on FRAUD', round(error_df.reconstruction_error.mean(),2))

print('std error of recon on FRAUD', round(error_df.reconstruction_error.std(),2))
# Predictions on Normal vs Fraud on Test ... using the reconstruction error as the parameter to tweak for best F1



# Reconstruction error on Test



predictions = autoencoder.predict(X_test)

predictions.shape



mse = np.mean(np.power(X_test - predictions, 2), axis=1)

error_df = pd.DataFrame({'reconstruction_error': mse,

                        'true_class': y_test})

print(error_df.shape[0], ' rows')

print('mean error of recon on TEST', round(error_df.reconstruction_error.mean(),2))

print('std error of recon on TEST', round(error_df.reconstruction_error.std(),2))



ReconError = 4.0



pred = [1 if e > ReconError else 0 for e in error_df.reconstruction_error.values]

len(pred)

#pred = (x <= eps)

f = f1_score(y_test, pred, average='binary')

print("F1 score on test", round(f,4), " with reconstruction error  ", ReconError)
minRE = 1

maxRE = 50

    

EpsF1 = []



for TryRE in range(minRE,maxRE):

    pred = [1 if e > TryRE else 0 for e in error_df.reconstruction_error.values]

    f = f1_score(y_test, pred, average='binary')

    #print("F1 score on test", round(f,4), " with epsilon ", eps)

    EpsF1.append([TryRE, round(f,4)])

    

EpsF1df = pd.DataFrame(EpsF1, columns = ['ReconError', 'F1'])

EpsF1df.head()
# Best Recon Error ... Max F1 on test



EpsF1df.loc[EpsF1df['F1'].idxmax()]
EpsF1df.plot.line("ReconError","F1")

plt.xlim(1, 50)

plt.title("F1 vs ReconError")

plt.show()
#  SCALER / Normalization



data = dfRaw.copy()

scl = StandardScaler()

all_cols = list(data)[:] 

pca_columns = list(data)[:-1] # all cols without Class

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

Xscaled = scl.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((Xscaled, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = XcopyALL.columns)



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



print("data ", data.shape)

print("normal_data ", normal_data.shape)

print("fraud_data ", fraud_data.shape)

print("Percent fraud ", round(100*492/284807, 4),"%")
# CREATE the TRAIN and TEST sets

# Fraud data is ONLY in TEST - not in TRAIN



normal_pca_data = normal_data[pca_columns]

fraud_pca_data = fraud_data[pca_columns]



num_test = 20000

shuffled_data = normal_pca_data.sample(frac=1, random_state=1960)[:-num_test].values

X_train = shuffled_data



X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[:]])

y_test = np.concatenate([np.zeros(num_test), np.ones(492)])



print("normal_pca_data ", normal_pca_data.shape)

print("fraud_pca_data", fraud_pca_data.shape)

print("Fraud data only in Test with NONE in the training")

print("X_train ", X_train.shape)

#print("X_valid ", X_valid.shape)

#print("y_valid ", y_valid.shape)

print("X_test ", X_test.shape)

print("y_test ", y_test.shape)
X_inliers = shuffled_data[-num_test:]

X_outliers = fraud_pca_data[:]

X = np.r_[X_inliers, X_outliers]



n_outliers = len(X_outliers)

ground_truth = np.ones(len(X), dtype=int)

ground_truth[-n_outliers:] = -1



PercFraud = n_outliers / X_test.shape[0]

PercFraud



print('X_inliers ', X_inliers.shape)

print('X_outliers ', X_outliers.shape)

print('X ', X.shape)

print('n_outliers ', n_outliers)

print('percent fraud in test: ', PercFraud)
%%time



# fit the model for outlier detection (default)

clf = LocalOutlierFactor(n_neighbors=20, contamination = PercFraud)

# use fit_predict to compute the predicted labels of the training samples

# (when LOF is used for outlier detection, the estimator has no predict,

# decision_function and score_samples methods).

y_pred = clf.fit_predict(X)

n_errors = (y_pred != ground_truth).sum()

X_scores = clf.negative_outlier_factor_

n_errors
print('accuracy ' , round(1 - (n_errors / X.shape[0]),4))
# Note that the ground truth and the y_pred for LOF is different than the original ... inliers = normal = 1 and outliers = fraud = -1

# We have to modify the y_pred for the F1 score calculation to be similar to the above 



y_predLOF = y_pred.copy()

y_predDF = pd.DataFrame(y_predLOF)

print(y_predDF[y_predDF[0] == -1].count())



y_predDF[y_predDF[0] == 1] = 0

y_predDF[y_predDF[0] == -1] = 1

print(y_predDF[y_predDF[0] == 1].count())



y_predLOF = y_predDF.values

y_predLOF = np.ravel(y_predLOF)

# F1 Score

#print("F1 score", round(f1_score(y_valid,pred, average='binary'), 4))

precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_test, y_predLOF, average='binary')

print("precision ", round((precision), 4))

print("recall ", round((recall), 4))

print("F1 score on Test", round((fbeta_score), 4))
# Optimize num of neighbors hyper paramter for best F1



minRE = 500

maxRE = 1100

    

EpsF1 = []



for TryRE in range(minRE,maxRE,100):

    clf = LocalOutlierFactor(n_neighbors=TryRE, contamination = PercFraud)

    y_pred = clf.fit_predict(X)

    n_errors = (y_pred != ground_truth).sum()

    X_scores = clf.negative_outlier_factor_

    

    y_predLOF = y_pred.copy()

    y_predDF = pd.DataFrame(y_predLOF)

    

    y_predDF[y_predDF[0] == 1] = 0

    y_predDF[y_predDF[0] == -1] = 1

    

    y_predLOF = y_predDF.values

    y_predLOF = np.ravel(y_predLOF)

    

    precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_test, y_predLOF, average='binary')

    

    print("F1 score on test", round(fbeta_score,4), " with num neighbors ", TryRE)

    EpsF1.append([TryRE, round(fbeta_score,4)])

    

EpsF1df = pd.DataFrame(EpsF1, columns = ['NumNeighb', 'F1'])

EpsF1df.head()
EpsF1df.plot.line("NumNeighb","F1")

plt.xlim(500, 1000)

plt.title("F1 vs NumNeighb")

plt.show()
data = dfRaw.copy()

scl = StandardScaler()

all_cols = list(data)[:] 

pca_columns = list(data)[:-1] # all cols without Class

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

Xscaled = scl.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((Xscaled, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = XcopyALL.columns)



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



print("data ", data.shape)

print("normal_data ", normal_data.shape)

print("fraud_data ", fraud_data.shape)

print("Percent fraud ", round(100*492/284807, 4),"%")



# CREATE the TRAIN and TEST sets

# Fraud data is ONLY in TEST - not in TRAIN



normal_pca_data = normal_data[pca_columns]

fraud_pca_data = fraud_data[pca_columns]



num_test = 75000

shuffled_data = normal_pca_data.sample(frac=1, random_state=1960)[:-num_test].values

X_train = shuffled_data



X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[:]])

y_test = np.concatenate([np.zeros(num_test), np.ones(492)])



print("normal_pca_data ", normal_pca_data.shape)

print("fraud_pca_data", fraud_pca_data.shape)

print("Fraud data only in Test with NONE in the training")

print("X_train ", X_train.shape)

#print("X_valid ", X_valid.shape)

#print("y_valid ", y_valid.shape)

print("X_test ", X_test.shape)

print("y_test ", y_test.shape)



X_inliers = shuffled_data[-num_test:]

X_outliers = fraud_pca_data[:]

X = np.r_[X_inliers, X_outliers]



n_outliers = len(X_outliers)

ground_truth = np.ones(len(X), dtype=int)

ground_truth[-n_outliers:] = -1



PercFraud = n_outliers / X_test.shape[0]

PercFraud



print('X_inliers ', X_inliers.shape)

print('X_outliers ', X_outliers.shape)

print('X ', X.shape)

print('n_outliers ', n_outliers)

print('percent fraud in test: ', PercFraud)
clf = LocalOutlierFactor(n_neighbors=900, contamination = PercFraud)



y_pred = clf.fit_predict(X)

n_errors = (y_pred != ground_truth).sum()

X_scores = clf.negative_outlier_factor_

#print('accuracy ' , round(1 - (n_errors / X.shape[0]),4))

n_errors
y_predLOF = y_pred.copy()

y_predDF = pd.DataFrame(y_predLOF)



y_predDF[y_predDF[0] == 1] = 0

y_predDF[y_predDF[0] == -1] = 1



y_predLOF = y_predDF.values

y_predLOF = np.ravel(y_predLOF)



# F1 Score

#print("F1 score", round(f1_score(y_valid,pred, average='binary'), 4))

precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_test, y_predLOF, average='binary')

print("precision ", round((precision), 4))

print("recall ", round((recall), 4))

print("F1 score on Test", round((fbeta_score), 4))
#  SCALER / Normalization



data = dfRaw.copy()

scl = StandardScaler()

all_cols = list(data)[:] 

pca_columns = list(data)[:-1] # all cols without Class

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

Xscaled = scl.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((Xscaled, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = XcopyALL.columns)



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



data = dfRaw.copy()

scl = StandardScaler()

all_cols = list(data)[:] 

pca_columns = list(data)[:-1] # all cols without Class

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

Xscaled = scl.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((Xscaled, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = XcopyALL.columns)



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



# CREATE the TRAIN and TEST sets

# Fraud data is ONLY in TEST - not in TRAIN



normal_pca_data = normal_data[pca_columns]

fraud_pca_data = fraud_data[pca_columns]



num_test = 75000

shuffled_data = normal_pca_data.sample(frac=1, random_state=1960)[:-num_test].values

X_train = shuffled_data



X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[:]])

y_test = np.concatenate([np.zeros(num_test), np.ones(492)])





X_inliers = shuffled_data[-num_test:]

X_outliers = fraud_pca_data[:]

X = np.r_[X_inliers, X_outliers]



n_outliers = len(X_outliers)

ground_truth = np.ones(len(X), dtype=int)

ground_truth[-n_outliers:] = -1



PercFraud = n_outliers / X_test.shape[0]

PercFraud



print('X_inliers ', X_inliers.shape)

print('X_outliers ', X_outliers.shape)

print('X ', X.shape)

print('n_outliers ', n_outliers)

print('percent fraud in test: ', PercFraud)

cov = EllipticEnvelope(support_fraction = 0.994, contamination = PercFraud)



y_pred = cov.fit_predict(X)

n_errors = (y_pred != ground_truth).sum()

n_errors
y_predLOF = y_pred.copy()

y_predDF = pd.DataFrame(y_predLOF)



y_predDF[y_predDF[0] == 1] = 0

y_predDF[y_predDF[0] == -1] = 1



y_predLOF = y_predDF.values

y_predLOF = np.ravel(y_predLOF)



# F1 Score

#print("F1 score", round(f1_score(y_valid,pred, average='binary'), 4))

precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_test, y_predLOF, average='binary')

print("precision ", round((precision), 4))

print("recall ", round((recall), 4))

print("F1 score on Test", round((fbeta_score), 4))
CM = confusion_matrix(y_test, y_predLOF)

tn, fp, fn, tp = confusion_matrix(y_test, y_predLOF).ravel()



print(CM)

print("_"*50)

print("TP ", tp)

print("FP ", fp)

print("TN ", tn)

print("FN ", fn)
plot_confusion_matrix(CM, 

                      normalize    = False,

                      target_names = ['Normal', 'Fraud'],

                      title        = "Confusion Matrix")
# Optimize support_fraction hyper paramter for best F1



minRE = 0.95

maxRE = 0.99

    

EpsF1 = []



for TryRE in np.arange(minRE, maxRE, 0.01):

    cov = EllipticEnvelope(support_fraction = TryRE, contamination = PercFraud)

    y_pred = cov.fit_predict(X)

    n_errors = (y_pred != ground_truth).sum()

    

    y_predLOF = y_pred.copy()

    y_predDF = pd.DataFrame(y_predLOF)

    

    y_predDF[y_predDF[0] == 1] = 0

    y_predDF[y_predDF[0] == -1] = 1

    

    y_predLOF = y_predDF.values

    y_predLOF = np.ravel(y_predLOF)

    

    precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_test, y_predLOF, average='binary')

    

    print("F1 score on test", round(fbeta_score,4), " with support_fraction ", TryRE)

    EpsF1.append([TryRE, round(fbeta_score,4)])

    

EpsF1df = pd.DataFrame(EpsF1, columns = ['SupFrac', 'F1'])

EpsF1df.head()
EpsF1df.plot.line("SupFrac","F1")

plt.xlim(minRE, maxRE)

plt.title("F1 vs SupFrac")

plt.show()
#  SCALER / Normalization



data = dfRaw.copy()

scl = StandardScaler()

all_cols = list(data)[:] 

pca_columns = list(data)[:-1] # all cols without Class

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

Xscaled = scl.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((Xscaled, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = XcopyALL.columns)



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



data = dfRaw.copy()

scl = StandardScaler()

all_cols = list(data)[:] 

pca_columns = list(data)[:-1] # all cols without Class

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

Xscaled = scl.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((Xscaled, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = XcopyALL.columns)



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



# CREATE the TRAIN and TEST sets

# Fraud data is ONLY in TEST - not in TRAIN



normal_pca_data = normal_data[pca_columns]

fraud_pca_data = fraud_data[pca_columns]



num_test = 75000

shuffled_data = normal_pca_data.sample(frac=1, random_state=1960)[:-num_test].values

X_train = shuffled_data



X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[:]])

y_test = np.concatenate([np.zeros(num_test), np.ones(492)])





X_inliers = shuffled_data[-num_test:]

X_outliers = fraud_pca_data[:]

X = np.r_[X_inliers, X_outliers]



n_outliers = len(X_outliers)

ground_truth = np.ones(len(X), dtype=int)

ground_truth[-n_outliers:] = -1



PercFraud = n_outliers / X_test.shape[0]

PercFraud



print('X_inliers ', X_inliers.shape)

print('X_outliers ', X_outliers.shape)

print('X ', X.shape)

print('n_outliers ', n_outliers)

print('percent fraud in test: ', PercFraud)

%%time



isofo = IsolationForest(n_estimators = 1050, max_features = 1.0, max_samples=1.0, 

                        behaviour="new", bootstrap=False, random_state=22,

                        contamination = PercFraud)



y_pred = isofo.fit_predict(X)

n_errors = (y_pred != ground_truth).sum()

print(n_errors)
y_predLOF = y_pred.copy()

y_predDF = pd.DataFrame(y_predLOF)



y_predDF[y_predDF[0] == 1] = 0

y_predDF[y_predDF[0] == -1] = 1



y_predLOF = y_predDF.values

y_predLOF = np.ravel(y_predLOF)



# F1 Score

#print("F1 score", round(f1_score(y_valid,pred, average='binary'), 4))

precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_test, y_predLOF, average='binary')

print("precision ", round((precision), 4))

print("recall ", round((recall), 4))

print("F1 score on Test", round((fbeta_score), 4))
# Optimize Num Estimators hyper paramter for best F1



minRE = 900

maxRE = 1150

    

EpsF1 = []



for TryRE in np.arange(minRE, maxRE, 50):

    isofo = IsolationForest(n_estimators = TryRE, max_features = 1.0, max_samples=1.0, 

                        behaviour="new", bootstrap=False, random_state=22,

                        contamination = PercFraud)



    y_pred = isofo.fit_predict(X)

    n_errors = (y_pred != ground_truth).sum()

    

    y_predLOF = y_pred.copy()

    y_predDF = pd.DataFrame(y_predLOF)

    

    y_predDF[y_predDF[0] == 1] = 0

    y_predDF[y_predDF[0] == -1] = 1

    

    y_predLOF = y_predDF.values

    y_predLOF = np.ravel(y_predLOF)

    

    precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_test, y_predLOF, average='binary')

    

    print("F1 score on test", round(fbeta_score,4), " with num_estimators ", TryRE)

    EpsF1.append([TryRE, round(fbeta_score,4)])

    

EpsF1df = pd.DataFrame(EpsF1, columns = ['NumEstim', 'F1'])

EpsF1df.head()
EpsF1df.plot.line("NumEstim","F1")

plt.xlim(minRE, maxRE)

plt.title("F1 vs NumEstim")

plt.show()
#  SCALER / Normalization



data = dfRaw.copy()

scl = StandardScaler()

all_cols = list(data)[:] 

pca_columns = list(data)[:-1] # all cols without Class

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

Xscaled = scl.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((Xscaled, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = XcopyALL.columns)



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



data = dfRaw.copy()

scl = StandardScaler()

all_cols = list(data)[:] 

pca_columns = list(data)[:-1] # all cols without Class

Xcopy = data[pca_columns]

XcopyALL = data[all_cols]

Xscaled = scl.fit_transform(Xcopy)

OnlyClass = data['Class'].values.reshape(-1,1)

data = np.concatenate((Xscaled, OnlyClass), axis=1)

data = pd.DataFrame(data, columns = XcopyALL.columns)



normal_data = data.loc[data["Class"] == 0]

fraud_data = data.loc[data["Class"] == 1]



# CREATE the TRAIN and TEST sets

# Fraud data is ONLY in TEST - not in TRAIN



normal_pca_data = normal_data[pca_columns]

fraud_pca_data = fraud_data[pca_columns]



num_test = 75000

shuffled_data = normal_pca_data.sample(frac=1, random_state=1960)[:-num_test].values

X_train = shuffled_data



X_test = np.concatenate([shuffled_data[-num_test:], fraud_pca_data[:]])

y_test = np.concatenate([np.zeros(num_test), np.ones(492)])





X_inliers = shuffled_data[-num_test:]

X_outliers = fraud_pca_data[:]

X = np.r_[X_inliers, X_outliers]



n_outliers = len(X_outliers)

ground_truth = np.ones(len(X), dtype=int)

ground_truth[-n_outliers:] = -1



PercFraud = n_outliers / X_test.shape[0]

PercFraud



print('X_inliers ', X_inliers.shape)

print('X_outliers ', X_outliers.shape)

print('X ', X.shape)

print('n_outliers ', n_outliers)

print('percent fraud in test: ', PercFraud)

%%time



OneSVM = OneClassSVM(nu = PercFraud)



y_pred = OneSVM.fit_predict(X)

n_errors = (y_pred != ground_truth).sum()

print(n_errors)
y_predLOF = y_pred.copy()

y_predDF = pd.DataFrame(y_predLOF)



y_predDF[y_predDF[0] == 1] = 0

y_predDF[y_predDF[0] == -1] = 1



y_predLOF = y_predDF.values

y_predLOF = np.ravel(y_predLOF)



# F1 Score

#print("F1 score", round(f1_score(y_valid,pred, average='binary'), 4))

precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_test, y_predLOF, average='binary')

print("precision ", round((precision), 4))

print("recall ", round((recall), 4))

print("F1 score on Test", round((fbeta_score), 4))
# Optimize nu hyper paramter for best F1



minRE = 0.017

maxRE = 0.022

    

EpsF1 = []



for TryRE in np.arange(minRE, maxRE, 0.001):

    OneSVM = OneClassSVM(nu = TryRE)

    y_pred = OneSVM.fit_predict(X)

    n_errors = (y_pred != ground_truth).sum()

    

    y_predLOF = y_pred.copy()

    y_predDF = pd.DataFrame(y_predLOF)

    

    y_predDF[y_predDF[0] == 1] = 0

    y_predDF[y_predDF[0] == -1] = 1

    

    y_predLOF = y_predDF.values

    y_predLOF = np.ravel(y_predLOF)

    

    precision,recall,fbeta_score, support  = precision_recall_fscore_support(y_test, y_predLOF, average='binary')

    

    print("F1 score on test", round(fbeta_score,4), " with nu ", TryRE)

    EpsF1.append([TryRE, round(fbeta_score,4)])

    

EpsF1df = pd.DataFrame(EpsF1, columns = ['nu', 'F1'])

EpsF1df.head()
EpsF1df.plot.line("nu","F1")

plt.xlim(minRE, maxRE)

plt.title("F1 vs nu")

plt.show()