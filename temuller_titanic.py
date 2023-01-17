import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_palette('husl')

from sklearn.pipeline import Pipeline

# Pre-processing
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, Normalizer, 
                                      MaxAbsScaler, RobustScaler, PowerTransformer)
from sklearn.preprocessing import PolynomialFeatures  # , Imputer 
from sklearn.experimental import enable_iterative_imputer  # to import IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.utils import resample

# Dimensionallity Reduction and Feature Extraction
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, NMF, SparsePCA, FastICA, FactorAnalysis
from sklearn.manifold import TSNE, Isomap
from hdbscan import HDBSCAN
from umap import UMAP

# Clustering
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

# Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier

# Post-analysis
from sklearn.metrics import (classification_report, confusion_matrix, plot_confusion_matrix,
                                accuracy_score, roc_curve, roc_auc_score, classification_report)
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
predict_titanic_df = pd.read_csv('test.csv')
titanic_df = pd.read_csv('train.csv')
titanic_df.head()
# Separate majority and minority classes
df_majority = titanic_df[titanic_df.Survived==0].copy()
df_minority = titanic_df[titanic_df.Survived==1].copy()
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=df_majority.shape[0],    # to match majority class
                                 random_state=42) # reproducible results
 
# Combine majority class with upsampled minority class
resampled_titanic_df = pd.concat([df_majority, df_minority_upsampled])


data_columns = resampled_titanic_df.columns.drop(['Survived', 'PassengerId'])
X_df = resampled_titanic_df[data_columns].copy()
predict_X_df = predict_titanic_df[data_columns].copy()
y_df = resampled_titanic_df['Survived'].copy()
id_df = resampled_titanic_df['PassengerId'].copy()
y = y_df.values

X_df.drop(['Name', 'Ticket', 'Cabin'], axis='columns', inplace=True)
X_df = pd.get_dummies(X_df)
X = X_df.values
predict_X_df.drop(['Name', 'Ticket', 'Cabin'], axis='columns', inplace=True)
predict_X_df = pd.get_dummies(predict_X_df)
predict_X = predict_X_df.values


#imputer = SimpleImputer()
imputer = IterativeImputer()
X = imputer.fit_transform(X)
predict_X = imputer.fit_transform(predict_X)

scaler = StandardScaler()
#scaler = RobustScaler()
#scaler = Normalizer()
#scaler = MinMaxScaler()
#scaler = PowerTransformer()
X = scaler.fit_transform(X)
predict_X = scaler.fit_transform(predict_X)

X_df = pd.DataFrame(data=X, columns=X_df.columns)
predict_X_df = pd.DataFrame(data=predict_X, columns=predict_X_df.columns)
X_df.head()
X_df.info(), predict_X_df.info()
model = RandomForestRegressor()
model.fit(X, y)

num_features = 10
features = X_df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-num_features:]  # top "num_features" number of features
extracted_features = [features[i] for i in indices]

fig, ax = plt.subplots(figsize=(8, 6))
plt.title(f'Feature Importances', fontsize=18)
plt.bar(range(len(indices)), importances[indices], color='b', align='center')
plt.xticks(range(len(indices)), extracted_features)
plt.ylabel('Relative Importance', fontsize=16)
plt.show()
reducer = PCA(n_components=4)
X_reduced = reducer.fit_transform(X)
labels = y
reducer_name = str(reducer).split('(')[0]

exp_var = np.var(X_reduced, axis=0)/np.sum(np.var(X_reduced, axis=0))
print(f'Explained Variance Fraction: {exp_var}')

fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap=plt.cm.Set1, edgecolor='k')
ax.set_title(f"Dimensionality Reduction {reducer_name}", fontsize=18)
ax.set_xlabel(f"Eigen-vector 1", fontsize=16)
ax.set_ylabel(f"Eigen-vector 2", fontsize=16)
legend = ax.legend(*scatter.legend_elements(), ncol=2)
ax.add_artist(legend)
plt.show()
X_vals = X#_reduced

X_train, X_test, y_train, y_test = train_test_split(X_vals, y, test_size=0.2)
clf = RandomForestClassifier()
#clf = XGBClassifier()
#clf = DecisionTreeClassifier()
#clf = GaussianNB()

clf.fit(X_train, y_train)

labels = y
clf_name = str(clf).split('(')[0]

scores = cross_val_score(clf, X_vals, y, cv=10)
score = np.round(scores.mean(), 3)
std = np.round(scores.std(), 3)
print(f'Score: {score} +/- {std}')
        
fig, ax = plt.subplots(figsize=(8, 8))
plot_confusion_matrix(clf, X_test, y_test, display_labels=labels, 
                          cmap=plt.cm.Blues, normalize='true', ax=ax)
plt.title(f'Classifier {clf_name}', fontsize=18)
plt.xticks(rotation='vertical')
ax.xaxis.label.set_size(16)
ax.yaxis.label.set_size(16)
plt.margins(0.2)
plt.subplots_adjust(bottom=0.15)
plt.show()
predict_y = clf.predict(predict_X)
submission_df = predict_titanic_df.copy()
submission_df['Survived'] = predict_y
submission_df = submission_df[['PassengerId', 'Survived']]
submission_df.to_csv('submission.csv', index=False)
submission_df.head()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Lambda, Flatten, Activation, Reshape
from keras.optimizers import Adam, RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import  backend as K

# CNN
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D

from keras.datasets import mnist

#from keras.layers.core import Dense

import theano 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
np.random.seed(42) 

model = Sequential()
model.add(Dense(60, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, 
                      batch_size=32, verbose=1, 
                      validation_split=0.2,
                      workers=4, use_multiprocessing=True)
loss_train, acc_train  = model.evaluate(X_train, y_train, verbose=False)
loss_test, acc_test  = model.evaluate(X_test, y_test, verbose=False)
print(f'Train acc/loss: {acc_train:.3}, {loss_train:.3}')
print(f'Test acc/loss: {acc_test:.3}, {loss_test:.3}')
y_pred_train = model.predict(X_train, verbose=True)
y_pred_test = model.predict(X_test,verbose=True)
# set up figure
f = plt.figure(figsize=(12,5))
f.add_subplot(1, 2, 1)

# plot accuracy as a function of epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')

# plot loss as a function of epoch
f.add_subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show(block=True)
predict_y = model.predict(predict_X)
predict_y = np.round(predict_y.flatten(), 0).astype(int)
submission_df = predict_titanic_df.copy()
submission_df['Survived'] = predict_y
submission_df = submission_df[['PassengerId', 'Survived']]
submission_df.to_csv('submission_nn.csv', index=False)
submission_df.head()
