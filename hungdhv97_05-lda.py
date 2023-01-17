import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.decomposition import PCA

from sklearn.feature_selection import VarianceThreshold

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.models import Sequential

from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns
# Load data

data_file_path = "../input/123456/05-CYP2C9_Inhibition.csv"

data = pd.read_csv(data_file_path, index_col = "No.")
# Split train data X and label y

features = [col for col in data.columns if col not in ['NAME', 'CLASS', 'nHBonds']]

X = data[features]

y = data.CLASS
# Convert label "Active Inactive" to "0 1"

le = LabelEncoder()

y = le.fit_transform(y)
# split into a training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Fill NaN item by most frequence value of columns

si = SimpleImputer(strategy = 'most_frequent')

X_train_impute = si.fit_transform(X_train)

X_test_impute = si.transform(X_test)
# Plot train data after impute (only column 0 and column 1)

plt.scatter(X_train_impute[:500,0], X_train_impute[:500, 1], c = y_train[:500], cmap = 'rainbow', alpha = 0.7, edgecolor = 'b')
# Scaling data using normal distribution

ss = StandardScaler()

X_train_scale = ss.fit_transform(X_train_impute)

X_test_scale = ss.transform(X_test_impute)
# Plot data after scaling

plt.scatter(X_train_scale[:500,0], X_train_scale[:500, 1], c = y_train[:500], cmap = 'rainbow', alpha = 0.7, edgecolor = 'b')
# Reduction low variance (= 0) feature of data 

vt = VarianceThreshold()

X_train_vt = vt.fit_transform(X_train_scale)

X_test_vt = vt.transform(X_test_scale)

X_train_vt.shape, X_test_vt.shape
# Plot tran data after dimension reduction

plt.scatter(X_train_vt[:500,0], X_train_vt[:500, 1], c = y_train[:500], cmap = 'rainbow', alpha = 0.7, edgecolor = 'b')
# Using lda for reduction data for classification. Data after transform have only one dimension (= min(n_classes - 1, n_features) = min(2 - 1, 2889) = 1)

lda = LinearDiscriminantAnalysis()

X_train_lda = lda.fit_transform(X_train_vt, y_train)

X_test_lda = lda.transform(X_test_vt)
# Plot 500 sample of training data after lda 

plt.scatter(np.arange(500), X_train_lda[:500], c = y_train[:500], cmap = 'rainbow', alpha = 0.7, edgecolor = 'b')
# Plot 500 sample of testing data after using lda training data parameter. 

plt.scatter(np.arange(300), X_test_lda[:300], c = y_test[:300], cmap = 'rainbow', alpha = 0.7, edgecolor = 'b')
X_train_lda.shape, np.bincount(y_train), np.bincount(y_test)
# Build MLP model

model = Sequential()

model.add(Dense(32, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))

model.add(Dense(16, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))
num_epochs = 10

batch_size = 16

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

H = model.fit(X_train_lda, y_train, validation_data = (X_test_lda, y_test), epochs = num_epochs, batch_size = batch_size)

prediction = model.predict_classes(X_test_lda)

score = accuracy_score(y_test, prediction)

cm = confusion_matrix(y_test, prediction)

cm, score