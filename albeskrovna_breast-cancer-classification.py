# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# importing other libraries I will need
import matplotlib.pyplot as plt
import seaborn as sns
# Other:
import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
# importing the dataset
data = pd.read_csv("../input/data.csv")
data.head()
# Drop useless variables
data = data.drop(['Unnamed: 32','id'],axis = 1)
data.shape
# checking for missing values
data.isna().any()
f,ax = plt.subplots(figsize=(10,5))
sns.countplot(y = data['diagnosis'], palette = "husl", ax=ax)
features = data.iloc[:, 1:]
from sklearn.preprocessing import MinMaxScaler
x = features.values #returns a numpy array
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
feat = pd.DataFrame(x_scaled, index = features.index, columns = features.columns)
feat.shape
diag = data.iloc[:,0]
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(feat.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
# Create correlation matrix
corr_matrix = feat.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.9
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
to_drop
# Drop features 
feat = feat.drop(to_drop, axis=1)
feat.shape
df = pd.concat([diag, feat], axis=1, sort=False)
df.head()
# rewriting the categorical values in the target column as numerical
y = data['diagnosis'].apply(lambda x: 1 if 'M' in x else 0)
y.head()

# putting the dataframe back together
df_encoded = pd.concat([y, feat], axis=1, sort=False)
df_encoded.head()
import math

vars = df_encoded.drop('diagnosis', axis = 1).keys()
plot_cols = 5
plot_rows = math.ceil(len(vars)/plot_cols)

plt.figure(figsize = (5*plot_cols,5*plot_rows))

for idx, var in enumerate(vars):
    plt.subplot(plot_rows, plot_cols, idx+1)
    sns.boxplot(x = 'diagnosis', y = var, data = df_encoded)
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df_encoded.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
df_select = df.iloc[:,0:8]
sns.pairplot(df_select, hue = 'diagnosis')
df_select = df.iloc[:,[0,8,9,10,11,12,13,14,15]]
sns.pairplot(df_select, hue = 'diagnosis')
df_select = df.iloc[:,[0,16,17,18,19,20]]
sns.pairplot(df_select, hue = 'diagnosis')
df_encoded = df_encoded.drop('concavity_mean', axis = 1)
df_encoded = df_encoded.drop('concavity_se', axis = 1)
df_encoded = df_encoded.drop('concavity_worst', axis = 1)
df_encoded.head()
df_encoded.shape
X = df_encoded.iloc[:,1:].values
y = df_encoded.iloc[:,0].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
# checking the accuracy score
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
# Applying 10-fold cross-validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
# Applying grid search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
# specifying the parameters I want to find optimal values for
parameters = [{'n_neighbors': [3,5,8,10], 'weights':['uniform']}, 
              {'n_neighbors': [3,5,8,10], 'weights':['distance']}
             ]
grid_search = GridSearchCV(estimator = classifier,
                          param_grid = parameters,
                          scoring = 'accuracy',
                          cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_accuracy
best_parameters = grid_search.best_params_
best_parameters