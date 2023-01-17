# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load dataset
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.dtypes
df.isna().sum()
df.describe().T
# 'Unnamed: 32' contains only NaN values and 'id' dont have any relevance. So, dropping both
df.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)
#Get the count of cancer types
df['diagnosis'].value_counts()
sns.countplot(df['diagnosis'])
df['diagnosis'].replace(['B','M'],[0,1],inplace=True)
df.columns
# Grouping columns based on features
feature_mean = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean','diagnosis']
feature_se = ['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'diagnosis']
feature_worst = ['radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst', 'diagnosis']
# Need to normalize the columns before plotting. This is because differences between values of features 
# are very high to observe on plot 
df_X = df.copy()
df_X.drop(['diagnosis'], axis = 1, inplace = True)

df_normzld = (df_X - df_X.mean()) / (df_X.std()) 

df_normzld['diagnosis'] = df['diagnosis']
# Viloin plots of feature means
df_normzld_mean = df_normzld[feature_mean]
df_normzld_mean = pd.melt(df_normzld_mean, id_vars="diagnosis", var_name="features", value_name='value')
plt.figure(figsize=(10,8))
sns.violinplot(x="features", y="value", hue="diagnosis", data=df_normzld_mean, split=True, inner="quart")
plt.xticks(rotation=90)
plt.title("Violin plot | Feature means (Normalized)")
# Viloin plots of feature se
df_normzld_se = df_normzld[feature_se]
df_normzld_se = pd.melt(df_normzld_se, id_vars="diagnosis", var_name="features", value_name='value')
plt.figure(figsize=(10,8))
sns.violinplot(x="features", y="value", hue="diagnosis", data=df_normzld_se, split=True, inner="quart")
plt.xticks(rotation=90)
plt.title("Violin plot | Feature std error (Normalized)")
# Viloin plots of feature worst
df_normzld_worst = df_normzld[feature_worst]
df_normzld_worst = pd.melt(df_normzld_worst, id_vars="diagnosis", var_name="features", value_name='value')
plt.figure(figsize=(10,8))
sns.violinplot(x="features", y="value", hue="diagnosis", data=df_normzld_worst, split=True, inner="quart")
plt.xticks(rotation=90)
plt.title("Violin plot | Feature worst (Normalized)")
# Box plots of feature means
plt.figure(figsize=(14,10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=df_normzld_mean)
plt.xticks(rotation=90)
#Box plots of feature se
plt.figure(figsize=(14,10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=df_normzld_se)
plt.xticks(rotation=90)
#Box plots of feature worst
plt.figure(figsize=(14,10))
sns.boxplot(x="features", y="value", hue="diagnosis", data=df_normzld_worst)
plt.xticks(rotation=90)
# swarm plots of feature means
plt.figure(figsize=(14,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=df_normzld_mean)
plt.xticks(rotation=90)
#Swarm plots of feature se
plt.figure(figsize=(14,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=df_normzld_se)
plt.xticks(rotation=90)
#Swarm plots of feature worst
plt.figure(figsize=(14,10))
sns.swarmplot(x="features", y="value", hue="diagnosis", data=df_normzld_worst)
plt.xticks(rotation=90)
selected_features =  ['radius_mean',  'texture_mean','perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean', 
                      'radius_se', 'perimeter_se', 'area_se',  'concavity_se',
                      'radius_worst', 'perimeter_worst', 'area_worst', 'concavity_worst', 'concave points_worst',
                     'diagnosis']
plt.figure(figsize=(12,8))
df_corr = df[selected_features].corr()
df_corr = df_corr.where(np.tril(np.ones(df_corr.shape)).astype(np.bool))
sns.heatmap(round(df_corr,2), annot=True, cmap="coolwarm",fmt='.2f',linewidths=.05)
final_features = ['area_mean', 'concavity_mean', 'area_se', 'texture_mean', 'concavity_se', 
                  'area_worst', 'concavity_worst', 'diagnosis']
X = df[final_features].drop('diagnosis', axis = 1)
y = df['diagnosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Performance evaluation
from sklearn import metrics
import itertools
rf_accuracy = metrics.accuracy_score(y_test, rf_pred)
print("Accuracy: %0.2f" % rf_accuracy)
rf_report = metrics.classification_report(y_test, rf_pred)
print("Clf Report:\n", rf_report)
rf_cm = metrics.confusion_matrix(y_test, rf_pred)
plot_confusion_matrix(rf_cm, classes = ['B','M'])