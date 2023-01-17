import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

import time
digit = pd.read_csv('../input/digit-recognizer/train.csv')
# check the five rows
digit.head()
# printing the shape of the database
digit.shape
# print some imformation about database
digit.info()
# print descriptive statistic on numerical columns
digit.describe()
# Target variable 
digit.label.value_counts()
y = digit["label"]
x = digit.loc[:, digit.columns != "label"]
X_std = StandardScaler().fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.25, random_state = 42, stratify = y)
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
log  = LogisticRegression(random_state = 42, multi_class="multinomial", solver="saga", max_iter=200)

start_time = time.time()
log.fit(X_train, y_train)
end_time = time.time()

time1 = end_time-start_time
print("Time elapsed: ",time1)

y_pred = log.predict(X_test)

# Accuracy Estimation
print('Accuracy Score (Train Data):', np.round(log.score(X_train, y_train), decimals = 3))
print('Accuracy Score (Test Data):', np.round(log.score(X_test, y_test), decimals = 3))

# Classification Report
logistic_report = classification_report(y_test, y_pred)
print(logistic_report)
# Here we will aim to explain 98% of the variance with PCA. We could reduce or increase it as per the needs of our project
pca = PCA(.96)
lower_dimensional_data = pca.fit_transform(X_train)

approximation = pca.inverse_transform(lower_dimensional_data)
pca.n_components_
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)* 100)
plt.ylabel('% Variance Explained')
plt.xlabel('Number of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var)
plt.show()
plt.figure(figsize=(8,4));

# Original Image
plt.subplot(1, 2, 1);
plt.imshow(x.values[1].reshape(28,28),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255));
plt.xlabel('784 components', fontsize = 14)
plt.title('Original Image', fontsize = 20);

# 154 principal components
plt.subplot(1, 2, 2);
plt.imshow(approximation[1].reshape(28, 28),
              cmap = plt.cm.gray, interpolation='nearest',
              clim=(0, 255));
plt.xlabel('270 components', fontsize = 14)
plt.title('95% of Explained Variance', fontsize = 20);
# fit and transform  the data
pca = PCA(n_components=443, random_state = 0)
X_pca_t = pca.fit_transform(X_train)
print(X_pca_t.shape)
# transform  the data
X_std_t = pca.transform(X_std)
print(X_std_t.shape)
X_train, X_test, y_train, y_test = train_test_split(X_std_t, y, test_size=0.25, random_state=42, stratify = y)
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape)
log  = LogisticRegression(random_state = 42, multi_class="multinomial", solver="saga", max_iter=200)

start_time = time.time()

log.fit(X_train, y_train)

end_time = time.time()
time1 = end_time-start_time
print("Time elapsed: ",time1)

y_pred = log.predict(X_test)

# Accuracy Estimation
print('Accuracy Score (Train Data):', np.round(log.score(X_train, y_train), decimals = 3))
print('Accuracy Score (Test Data):', np.round(log.score(X_test, y_test), decimals = 3))

# Classification Report
logistic_report = classification_report(y_test, y_pred)
print(logistic_report)
# https://github.com/mGalarnyk/Python_Tutorials/blob/master/Sklearn/PCA/PCA_Image_Reconstruction_and_such.ipynb

