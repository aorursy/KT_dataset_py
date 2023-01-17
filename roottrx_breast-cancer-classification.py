# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
from matplotlib.colors import ListedColormap
import seaborn as sns # data visualization

# Scikit-Learn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.decomposition import PCA

# Warning Library
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read data
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv', sep=',')
# First 5 rows 
data.head()
# Drop id and Unnamed: 32 from data. 
data.drop(['id', 'Unnamed: 32'], inplace=True, axis=1) #axis = 1: do it by column
# Change diagnosis name to target
data.rename(columns={'diagnosis': 'target'}, inplace=True)
data.head()
# Print target column's values
# Visualize number of target (M and B's)
sns.countplot(data['target'])
print(data.target.value_counts())
# Replace M to 1 and B to 0
data['target'] = [1 if i.strip() == 'M' else 0 for i in data.target]
print("length of the data:", len(data))
print("Shape of the data:", data.shape)
# Information of data
data.info()
data.describe().T
# Correlation
corr_matrix = data.corr()
ax = sns.heatmap(
    corr_matrix, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
sns.clustermap(corr_matrix, annot=True, fmt=".2f")
plt.title('Correlation Between Features');
threshold = 0.75
filter_ = np.abs(corr_matrix['target']) > threshold
corr_features = corr_matrix.columns[filter_].tolist()
sns.clustermap(data[corr_features].corr(), annot=True, fmt=".2f")
plt.title("Correlation Between Features with Correlation Threshold 0.75");
# Box plot
data_melted = pd.melt(data,
                      id_vars='target',
                      var_name='features',
                      value_name='value')
plt.figure()
sns.boxplot(x='features', y="value", hue='target', data=data_melted)
plt.xticks(rotation=90)
plt.show()
# Pair plot
sns.pairplot(data[corr_features], diag_kind='kde', markers='+', hue='target')
plt.show();
y = data.target
X = data.drop('target', axis=1)
columns = X.columns.tolist()
clf = LocalOutlierFactor()
y_pred = clf.fit_predict(X)
X_score = clf.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score['score'] = X_score
outlier_score
plt.figure()
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], color='k', s=3, label='Data Points'); 
radius = (X_score.max() - X_score) / (X_score.max() - X_score.min())
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=1000*radius, edgecolors='r', facecolors='none', label='Outlier Scores')
plt.legend()
plt.show()
plt.figure()
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], color='k', s=3, label='Data Points')
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=1000*radius, edgecolors='r', facecolors='none', label='Outlier Scores')
plt.legend()
plt.show();
threshold = -2.5
filter_ = outlier_score['score'] < threshold
outlier_index = outlier_score[filter_].index.tolist()
plt.figure()
plt.scatter(X.iloc[outlier_index, 0], X.iloc[outlier_index, 1], color='blue', s=50, label='Oultlier Points')
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], color='k', s=3, label='Data Points')

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], s=1000*radius, edgecolors='r', facecolors='none', label='Outlier Scores')
plt.legend()
plt.show();
# Drop outliers 
X = X.drop(outlier_index)
y = y.drop(outlier_index).values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train_df = pd.DataFrame(X_train, columns=columns)
X_train_df_desc = X_train_df.describe().T
X_train_df['target'] = y_train
# Box plot
data_melted = pd.melt(X_train_df, id_vars='target',
                     var_name='features', value_name='value')
plt.figure()
sns.boxplot(x='features', y='value', hue='target', data=data_melted)
plt.xticks(rotation=90)
plt.show()
# KNN
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
score = knn.score(X_test, y_test)
print("Score:", score)
print("confusion matrix:", cm)
print("Accuracy Score:", acc)
# Choose best parameters

def KNN_Best_Params(X_train, X_test, y_train, y_test):
    k_range = list(range(1, 31))
    weight_options = ['uniform', 'distance']
    
    param_grid = dict(n_neighbors=k_range, weights=weight_options)
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
    grid.fit(X_train, y_train)
    
    print(f'Best training score: {grid.best_score_} with oarameters: {grid.best_params_}')
    print()
    
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(X_train, y_train)
    
    y_pred_test = knn.predict(X_test)
    y_pred_train = knn.predict(X_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    print(f"Test Score : {acc_test}, Train Score : {acc_train}")
    print()
    print("Confusion Matrix Test: ", cm_test)
    print("Confusion Matrix Train : ", cm_train)
    
    return grid
grid = KNN_Best_Params(X_train, X_test, y_train, y_test)