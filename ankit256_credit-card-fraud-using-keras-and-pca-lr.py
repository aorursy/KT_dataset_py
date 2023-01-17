import pandas as pd 

from sklearn.feature_selection import RFECV

import numpy as np

import seaborn as sns

from sklearn.model_selection import cross_val_score

from sklearn import metrics

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import minmax_scale

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf



import warnings

import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

from scipy import stats

from scipy.stats import shapiro

from scipy.stats import anderson

from sklearn.model_selection import train_test_split

from scipy.stats import mannwhitneyu

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler

from sklearn.metrics import roc_curve, auc

from sklearn.decomposition import PCA

from imblearn.under_sampling import TomekLinks

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import ClusterCentroids

from imblearn.pipeline import make_pipeline

from sklearn import preprocessing

from sklearn.manifold import TSNE

from mpl_toolkits.mplot3d import Axes3D

import keras

from keras.datasets import imdb

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Dropout, Embedding, SpatialDropout1D, LSTM, BatchNormalization

from keras.layers.wrappers import Bidirectional # new! 

from keras.callbacks import ModelCheckpoint 



import os

from sklearn.metrics import roc_auc_score 

import matplotlib.pyplot as plt 

%matplotlib inline
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print(data.head())

print(data['Class'].value_counts())

print(data.columns)
X = data[['V27', 'V4', 'V10', 'V14', 'V28', 'V20', 'V21', 'V16', 'V13', 'V24']]

y = data['Class']
X.head()
X= data[['V27', 'V4', 'V10', 'V14', 'V28', 'V20', 'V21', 'V16', 'V13', 'V24']]

X = X.loc[:,:].apply(lambda x: round(x,2))

mm_scaler = preprocessing.StandardScaler()

X.columns

X[['V27', 'V4', 'V10', 'V14', 'V28', 'V20', 'V21', 'V16', 'V13', 'V24']] = mm_scaler.fit_transform(X[['V27', 'V4', 'V10', 'V14', 'V28', 'V20', 'V21', 'V16', 'V13', 'V24']])

X.head()
def logist_regression(X_sampling, y_sampling, sampling_type,XR_test, yR_test):

   

   #### logistic regression part and grid search 

   lr7 = LogisticRegression()

   penalty = ['l1', 'l2']

   C = [0.001,0.01,0.1,1,10,100]

   hyperparameters = dict(C=C, penalty=penalty)

   gridsearch = GridSearchCV(lr7, hyperparameters, cv=3, verbose=1)

   best_model_gs = gridsearch.fit(X_sampling, y_sampling)

   predictions7 = best_model_gs.predict(XR_test)



   #### printing the right metrics

   print(metrics.classification_report(yR_test,predictions7))

   print(metrics.confusion_matrix(yR_test,predictions7))

    

    

   #### plotting the sampling distribution

   datafra = pd.DataFrame(data=y_sampling, index=range(len(y_sampling)), columns=['Class'])

   datafra.Class.value_counts().index 

   #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=False, sharey=False,figsize=[12,12])

   fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex=False, sharey=False,figsize=[12,4])

   sns.barplot(x=datafra.Class.value_counts().index, y=datafra.Class.value_counts(), data=datafra, ax = ax1)

   

   ####plotting the heatmap for confusion matrix

   metric = metrics.confusion_matrix(yR_test,predictions7)

   confusion_dataframe = pd.DataFrame(data=metric, index=['Actual_Negative', 'Actual_Positive'], columns=['Guessed_Negative', 'Guessed_Positive',])

   #confusion_dataframe

   sns.heatmap(confusion_dataframe, annot=True, fmt="d", cmap="YlGnBu", ax = ax2)

   

    

   #### plotting the ROC and AUC curve

   fpr, tpr, _ = (metrics.roc_curve(yR_test,predictions7))

   ax3.plot(fpr, tpr)

   roc_auc = auc(fpr, tpr)

   ax3.plot([0, 1], [0, 1], 'k--', label='AUC = %0.3f'% roc_auc)

   ax3.legend(loc='lower right')

   ax3.plot([0,1],[0,1],'r--')

   ax3.set_xlim([-0.1,1.0])

   ax3.set_ylim([-0.1,1.01])

   ax3.set_ylabel('True Positive Rate')

   ax3.set_xlabel('False Positive Rate')

   plt.tight_layout()

   plt.show()

   print(f'Area Under the Curve: {round(roc_auc,2)}')
XN_train, XN_test, yN_train, yN_test = train_test_split(X, y, test_size=0.33, random_state=42)
rus = RandomUnderSampler(random_state=0) 

rus.fit(XN_train, yN_train) 

X_smn, y_smn = rus.fit_resample(XN_train, yN_train)

logist_regression(X_smn, y_smn, "RANDOM UNDER-sampling",XN_test, yN_test)
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal_component_1', 'principal_component_2'])

print(pca.get_params())

print(principalComponents.shape)
finalDf = pd.concat([principalDf, data['Class']], axis = 1)

print(finalDf.head())

print(pca.explained_variance_ratio_)

features = finalDf.columns

features = finalDf.columns
X_pca= finalDf[features].drop('Class',axis=1)

y_pca = finalDf['Class']

XP_train, XP_test, yP_train, yP_test = train_test_split(X_pca, y_pca, test_size=0.33, random_state=42)

rus = RandomUnderSampler(random_state=0)

rus.fit(XP_train, yP_train)

X_pca, y_pca = rus.fit_resample(XP_train, yP_train)

print(len(X_pca))

logist_regression(X_pca, y_pca, "RANDOM UNDER-sampling",XP_test, yP_test)
XN_train, XN_test, yN_train, yN_test = train_test_split(X_smn, y_smn, test_size=0.20, random_state=42)
input_train = XN_train

target_train = yN_train

yN_test= yN_test

XN_test= XN_test

target_train

print(f'Shape of Input Training Data {input_train.shape} ')

print(f'Shape of Target Training Data {target_train.shape} ')
from tensorflow.keras.callbacks import EarlyStopping
def ANN(x_train, x_test, y_train, y_test):

  model = Sequential()

  model.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform', input_shape=(10,)))

  model.add(Dropout(0.2))

  model.add(Dense(128,kernel_initializer='glorot_uniform', activation='relu'))

  model.add(Dropout(0.2))

  

  model.add(Dense(128,kernel_initializer='glorot_normal', activation='relu'))

  model.add(Dense(64,kernel_initializer='glorot_uniform', activation='relu'))

  model.add(Dense(32,kernel_initializer='glorot_uniform', activation='relu'))

  model.add(BatchNormalization())

  model.add(Dense(1, activation='sigmoid'))

  

  rms = keras.optimizers.RMSprop(lr=0.0005, rho=0.9)

  model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])



  model.summary()

  #checkpoint_name = '../input/Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 

  #checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=True,  save_best_only = True, mode ='auto')

  #callbacks_list = [checkpoint]





  early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

  model.fit(x_train, y_train, 

          batch_size=64, epochs=100, verbose=1, 

          validation_data=(x_test, y_test), callbacks=[early_stop])

  

  return model

Model1 = ANN(input_train, XN_test, target_train, yN_test)
np.bincount(yN_train)
from sklearn.metrics import classification_report,confusion_matrix

predictions = Model1.predict_classes(XN_test)
predictions= predictions.reshape(-1,)

predictions
def model_performance(target, predictions):

  fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex=False, sharey=False,figsize=[16,4])

  ax1.bar(x=target.value_counts().index.astype(str), height=target.value_counts().to_list(), data=target, color=['b','r'])

  metric = metrics.confusion_matrix(target,predictions)

  confusion_dataframe = pd.DataFrame(data=metric, index=['Actual_Negative', 'Actual_Positive'], columns=['Guessed_Negative', 'Guessed_Positive'])

  sns.heatmap(confusion_dataframe, annot=True, fmt="d", cmap="YlGnBu", ax = ax2)

  fpr, tpr, _ = (metrics.roc_curve(target,predictions))

  plt.plot(fpr, tpr)

  fpr, tpr, _ = (metrics.roc_curve(target,predictions))

  ax3.plot(fpr, tpr)

  roc_auc = auc(fpr, tpr)

  ax3.plot([0, 1], [0, 1], 'k--', label='AUC = %0.3f'% roc_auc)

  ax3.legend(loc='lower right')

  ax3.plot([0,1],[0,1],'r--')

  ax3.set_xlim([-0.1,1.0])

  ax3.set_ylim([-0.1,1.01])

  ax3.set_ylabel('True Positive Rate')

  ax3.set_xlabel('False Positive Rate')

  plt.tight_layout()

  print(f'Area Under the Curve: {round(roc_auc,2)}')

  plt.show()
model_performance(pd.Series(yN_test), pd.Series(predictions))

print(metrics.classification_report(yN_test,predictions))
new_data = pd.concat([data[X.columns], data['Class']], axis=1)

new_data= new_data[new_data['Class']==1]

x_new_data = new_data.iloc[:,:-1].values

y_new_data = new_data.iloc[:,-1].values

predictions = Model1.predict_classes(x_new_data)

print(predictions.shape)

predictions = predictions.reshape(-1,)

predictions.shape
np.bincount(predictions)

predictions
model_performance(pd.Series(y_new_data), pd.Series(predictions))

print(metrics.classification_report(y_new_data,predictions))