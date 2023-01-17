############## Necessary imports #################
import pandas as pd

########################
# Common 
########################
import sys
import os
import random
import gc
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import scipy

########################
# Plotting
########################
import matplotlib.pyplot as plt
import seaborn as sns

########################
# ML libs
########################
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense 
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau
import tensorflow
from sklearn.ensemble import RandomForestClassifier

########################
# Global variables and settings
########################
file_name='../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv'

# Set NumPy and TensorFlow random seed to make results reproducable
plt.style.use('seaborn')
np.random.seed(42)
tensorflow.set_random_seed(2)
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.

# Load the data
sky = pd.read_csv(file_name, dtype={'class':'category'})
#sky = sky.sample(1000)
sky.drop(['objid', 'specobjid'], axis=1,inplace=True)
sky.head()
# Look at data types and counts
sky.info()
sns.countplot('class', data = sky)
plt.title("Number of samples by class")
plt.ylabel('Count')
plt.show()
sky['class'].value_counts()
# Print distribution of features by labels: Galaxy, Star, Quazar
# 3 subplots columns for 3 labels: Galaxy, Star, Quazar
# subplots rows in number of features in dataset

# Rows and columns in subplots (not sky dataset rows and cols)
ncols = sky['class'].cat.categories.size
# -1 because we don't show histplot of class by class
nrows = sky.columns.size-1

f, ax = plt.subplots(nrows, ncols, figsize=(10,30))
axes = ax.flatten()
i = 0
# Go through sky dataset columns: dec, u, g, r etc.
for sky_col_name in sky.columns:
    if sky_col_name == 'class': continue
    # Go through classes: Galaxy, Star, Quazar
    for cat in sky['class'].cat.categories:
        data = sky[sky['class'] == cat][sky_col_name]
        # Draw the plot for current class
        axes[i].set_title('%s %s distribution' % (cat, sky_col_name))
        axes[i].set_ylabel('Count')
        sns.distplot(data, ax=axes[i])
        i +=1
        
plt.tight_layout()
plt.show()
sns.heatmap(sky.corr())
plt.title("Features correlations")
plt.show()
# Balancing not needed:
# Get number of smallest label and reduce features in other labels
#nsamples = sky['class'].value_counts().min()
#sky_balanced = sky.groupby('class', as_index=False).apply(lambda g:  g.sample(nsamples)).reset_index(drop=True)
# Fitting the model to get feature importances after training
model = RandomForestClassifier()
model.fit(sky.drop('class', axis=1) , sky['class'])

# Draw feature importances
imp = model.feature_importances_
f = sky.columns.drop('class')
# Sort by importance descending
f_sorted = f[np.argsort(imp)[::-1]]
sns.barplot(x=f,y = imp, order = f_sorted)
plt.title("Features importances")
plt.ylabel("Importance")
plt.show()
# Select features as input for classification
f_selected = f_sorted[:7].values
sky_features = sky.loc[:,f_selected]
sky_features.head()
# Store labels in this variable
sky_labels = sky['class']
# Train/test split
train_X, test_X, train_y, test_y = train_test_split(sky_features, sky_labels, random_state=42)

# The model
forest = RandomForestClassifier(random_state=42)

# Adjust tree's parameters with help of GridSearchCV
# best variant was n_estimators: 40 or 70, f1score: 99,97,1
tuned_parameters={'n_estimators': range(10,100,10)[1:]}
clf = GridSearchCV(forest, tuned_parameters,cv=5)

# Train and predict
train = clf.fit(train_X, train_y)
pred = clf.predict(test_X)  
print('Best params are: %s' % clf.best_params_)

# Print metrics
print(metrics.classification_report(pred, test_y))

# Draw a chart for f1 score metric
f1 = metrics.f1_score(pred, test_y, average=None)
sns.barplot(sky['class'].cat.categories, f1)
plt.title("F1 score by labels")
plt.show()
# sklearn.metrics.confusion_matrix result: y - true labels, x = predicted labels
cm = metrics.confusion_matrix(pred, test_y)
# Normalize
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
conf_matrix = pd.DataFrame(cm
                           ,index = sky['class'].cat.categories
                           ,columns = sky['class'].cat.categories)
# Visualize confusion matrix
sns.heatmap(conf_matrix, annot=True)
plt.title("Prediction confusion matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
print('Accuracy: %s' % metrics.accuracy_score(pred, test_y))