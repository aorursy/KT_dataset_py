import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from sklearn.impute import SimpleImputer

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif,f_classif

from sklearn.feature_selection import VarianceThreshold

from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, MaxPool2D, Conv1D, MaxPooling1D

from keras.models import Sequential

from keras.optimizers import SGD

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from imblearn.over_sampling import SMOTE

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.model_selection import StratifiedKFold, KFold

import matplotlib.pyplot as plt
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
# Dataset size

n_samples, n_features = data.shape

n_classes = len(np.unique(y))



print("Total dataset size:")

print("n_samples: %d" % n_samples)

print("n_features: %d" % n_features)

print("n_classes: %d" % n_classes)
# split into a training and testing set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Fill NaN item by most frequence value of columns

si = SimpleImputer(strategy = 'most_frequent')

X_train_impute = si.fit_transform(X_train)

X_test_impute = si.transform(X_test)
# Scaling data using normal distribution

ss = StandardScaler()

X_train_scale = ss.fit_transform(X_train_impute)

X_test_scale = ss.transform(X_test_impute)
# Reduction low variance (= 0) feature of data 

vt = VarianceThreshold()

X_train_vt = vt.fit_transform(X_train_scale)

X_test_vt = vt.transform(X_test_scale)

X_train_vt.shape, X_test_vt.shape
# reduction dimension using pca

n_components = 256

pca = PCA(n_components = n_components)

X_train_pca = pca.fit_transform(X_train_vt)

X_test_pca = pca.transform(X_test_vt)
model = Sequential()

model.add(Dense(64, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(128, activation = 'relu'))

model.add(Dense(64, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))

model.add(Dense(16, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))
num_epochs = 10

batch_size = 32

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

H = model.fit(X_train_scale, y_train, validation_data = (X_test_scale, y_test), epochs = num_epochs, batch_size = batch_size)

prediction = model.predict_classes(X_test_scale)

score = accuracy_score(y_test, prediction)

cm = confusion_matrix(y_test, prediction)

cm, score
# 8. Plot loss graph, accuracy traning set and test set

fig = plt.figure()

plt.plot(np.arange(0, num_epochs), H.history['loss'], label='training loss')

plt.plot(np.arange(0, num_epochs), H.history['val_loss'], label='val loss')

plt.plot(np.arange(0, num_epochs), H.history['accuracy'], label='accuracy')

plt.plot(np.arange(0, num_epochs), H.history['val_accuracy'], label='val accuracy')

plt.title('Accuracy and Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss|Accuracy')

plt.legend()