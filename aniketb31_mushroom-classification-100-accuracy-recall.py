# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import the essential libraries 

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from collections import defaultdict

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, SelectPercentile, f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve, classification_report, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
# Import the data

df = pd.read_csv('../input/mushroom-classification/mushrooms.csv')
df.head()
df.info()
# Check for any out of place values 

for col in df.columns:
    print(col, df[col].unique())
# Zoom into the 'stalk root' column

df['stalk-root'].value_counts()
# Drop the stalk root column

df.drop('stalk-root', axis=1, inplace=True)
# Rename the columns 

df.rename(columns={'cap-shape':'cap_shape', 'cap-surface':'cap_surface', 'cap-color':'cap_color', 
                   'gill-attachment':'gill_attachment', 'gill-spacing':'gill_spacing', 'gill-size':'gill_size', 
                   'gill-color':'gill_color', 'stalk-shape':'stalk_shape', 'stalk-surface-above-ring':'stalk_surface_above_ring', 
                   'stalk-surface-below-ring':'stalk_surface_below_ring', 
                   'stalk-color-above-ring':'stalk_color_above_ring', 'stalk-color-below-ring':'stalk_color_below_ring', 
                   'veil-type':'veil_type', 'veil-color':'veil_color', 'ring-number':'ring_number', 
                   'ring-type':'ring_type', 'spore-print-color':'spore_print_color'}, inplace=True)
# Create a train, test and validation set

# The test and validation size is set at 1000. There isn't any particular scientific reason for this choice. It just felt right.
# The shuffle is kept as True to ensure random distribution of instances in all sets. 

# Split out the test set from the original dataset
df_train_val, df_test = train_test_split(df, test_size=1000, random_state=42, shuffle=True)

# Split the remaining data into train and validation sets
df_train, df_validation = train_test_split(df_train_val, test_size=1000, random_state=42, shuffle=True)
# Create a separate train set for EDA

# A copy of train set is kept aside for conducting EDA since categories are to be encoded later.
# Conducting EDA on alphabetical categories, rather than encoded ones, is easier for understanding the data. 

df_train_eda = df_train.copy()
# Create functions to encode the data 

# Reason for creating elaborate functions to encode the data:
# 1. If LabelEncoder is applied directly on the entire data without splitting it, there will be information leakage into the test set. Hence, encoding had to be done after splitting.
# 2. However, applying LabelEncoder object with fit_transform method on train set and later applying the same object with transform method on test and validation set didn't work.
# 3. Hence, the need for elaborate functions which create encoded values out of the training data and later encode the train, test and validation set.
# 4. If anyone knows a more efficient approach, please let me know. 

# A function to create and return a dictionary of alphabetical categories mapped to their numerical codes for all train set columns

def create_list_of_encoded_values(df): # Input will be the dataset on which encoder object will be fit
    
    le = LabelEncoder() # Labelencoder object
    d_list = [] # An empty dictionary to store the alphabetical categories:codes mapping
    
    for col in df.columns: # For all columns, create the necessary mapping and add to the dictionary
        le.fit(df[col]) 
        d_list.append(dict(zip(le.classes_, le.transform(le.classes_))))
        
    return d_list # Return the dictionary 

# A function to encode other datasets (of the same family as that which acted as input in the above function) on the basis of mapping done by previous function.

def encode_datasets(d_list, df): # The dictionary output by above function and the dataset on which encoding is to be done are the inputs
    i=0
    for col in df.columns:
        df[col].replace(d_list[i], inplace=True)
        i+=1
        
    return (df) # Return the encoded dataset
# As planned, Create list of alphabetical categories:codes mapping from training data

list_encoded_values = create_list_of_encoded_values(df_train)

# Encode train, test and validation data

df_train = encode_datasets(list_encoded_values, df_train)
df_test = encode_datasets(list_encoded_values, df_test)
df_validation = encode_datasets(list_encoded_values, df_validation)
# Identify the most important features using correlation matrix 

df_train.corr()['class'].sort_values() # Check correlation of all features with 'class' 
# Gill size

fig, ax = plt.subplots(figsize=(10,7))
plt.style.use('ggplot')

# Plot the data
sns.countplot(x='gill_size', data=df_train_eda, hue='class')

# Manage the axes and title
ax.set_xlabel("Gill Size",fontsize=20)
ax.set_ylabel('No. of Mushrooms',fontsize=20)
ax.set_title('Mushroom Gill Size vis-a-vis Edibility',fontsize=22)
ax.set_xticklabels(('broad', 'narrow'), fontsize = 12)
ax.grid(False)

# Change the legend text
L = plt.legend()
L.get_texts()[0].set_text('Edible')
L.get_texts()[1].set_text('Poisonous')
# Gill color

fig, ax = plt.subplots(figsize=(10,7))
plt.style.use('ggplot')

sns.countplot(x='gill_color', data=df_train_eda, hue='class')

ax.set_xlabel("Gill Color",fontsize=20)
ax.set_ylabel('No. of Mushrooms',fontsize=20)
ax.set_title('Mushroom Gill Color vis-a-vis Edibility',fontsize=22)
ax.set_xticklabels(('purple', 'pink', 'red', 'brown', 'gray', 'buff', 'white', 'black', 'chocolate', 'yellow', 'orange', 'green'), fontsize = 12)
ax.grid(False)

L = plt.legend()
L.get_texts()[0].set_text('Edible')
L.get_texts()[1].set_text('Poisonous')
# Bruises

fig, ax = plt.subplots(figsize=(10,7))
plt.style.use('ggplot')

sns.countplot(x='bruises', data=df_train_eda, hue='class')

ax.set_xlabel("Bruises",fontsize=20)
ax.set_ylabel('No. of Mushrooms',fontsize=20)
ax.set_title('Mushroom Bruises vis-a-vis Edibility',fontsize=22)
ax.set_xticklabels(('yes', 'no'), fontsize = 12)
ax.grid(False)

#gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

L = plt.legend()
L.get_texts()[0].set_text('Edible')
L.get_texts()[1].set_text('Poisonous')
# Ring type

fig, ax = plt.subplots(figsize=(10,7))
plt.style.use('ggplot')

sns.countplot(x='ring_type', data=df_train_eda, hue='class')

ax.set_xlabel("Ring Type",fontsize=20)
ax.set_ylabel('No. of Mushrooms',fontsize=20)
ax.set_title('Mushroom Ring Types vis-a-vis Edibility',fontsize=22)
ax.set_xticklabels(('pendant', 'evanescent', 'large', 'none', 'flaring'), fontsize = 12)
ax.grid(False)

L = plt.legend()
L.get_texts()[0].set_text('Edible')
L.get_texts()[1].set_text('Poisonous')
# Use ExtraTreesClassifier to find out the most important features which will be used as input for the model

plt.figure(figsize=(20,15))
plt.style.use('fivethirtyeight')

et_clf = ExtraTreesClassifier(random_state=42)
et_clf.fit(df_train.drop('class', axis=1), df_train['class'])

pd.Series(et_clf.feature_importances_, index=df_train.drop('class', axis=1).columns).nlargest(22).plot(kind='barh')
plt.xlabel('Feature Importance')
plt.title('Features and their Importance')
# A function to create a train, test and validation set that contains only the top features 

def new_set(x, old_set): # x is the number of features to be shortlisted, old_set is the parent set
    
    nue_set = pd.DataFrame()
    
    for col in pd.Series(et_clf.feature_importances_, index=df_train.drop('class', axis=1).columns).nlargest(x).index:
        nue_set[col] = old_set[col]
    nue_set['class'] = old_set['class']
    
    return (nue_set) # The 'reconstructed' set will be returned 
# Create the new train and validation sets and split them into X and y. Test set will be dealt with later.

# X train and y train 

df_train_new = new_set(12, df_train)

X_train = df_train_new.drop('class', axis=1)
y_train = df_train_new['class']

# X validation and y validation 

df_validation_new = new_set(12, df_validation)

X_val = df_validation_new.drop('class', axis=1)
y_val = df_validation_new['class']
# Scale the train and validation data

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train) # Fit the scaler object on X_train and transform it

X_val = scaler.transform(X_val) # Use the object already fitted on X_train to transform X_val
# Lets check the bare bones classifiers first

sgd_clf = SGDClassifier(random_state=42) # Standard SGD Classifier with hinge loss; this is equivalent to linear SVM
log_clf = LogisticRegression(random_state=42) # Standard logistic regression
knn_clf = KNeighborsClassifier() # Standard KNN classifier 
svc_clf = SVC(random_state=42) # Standard SVC with RBF kernel
lsvc_clf = LinearSVC(random_state=42) # SVC with linear kernel
dt_clf = DecisionTreeClassifier() # Standard decision tree 
rf_clf = RandomForestClassifier() # Standard random forest

models = [sgd_clf, log_clf, knn_clf, svc_clf, lsvc_clf, dt_clf, rf_clf]
accuracy_scores = [] 
recall_scores = []

for clf in [sgd_clf, log_clf, knn_clf, svc_clf, lsvc_clf, dt_clf, rf_clf]:
    
    # Fit the classifier on training data
    clf.fit(X_train, y_train)
    
    # Make predictions for validation data
    y_pred = clf.predict(X_val)
    
    # Performance measures for validation data
    print('confusion matrix for {}:'.format(clf.__class__.__name__), '\n', confusion_matrix(y_val, y_pred))
    print('precision score for {}:'.format(clf.__class__.__name__), precision_score(y_val, y_pred))
    print('recall score for {}:'.format(clf.__class__.__name__), recall_score(y_val, y_pred))
    print('accuracy score for {}:'.format(clf.__class__.__name__), accuracy_score(y_val, y_pred))
    print('-'*100)
    
    accuracy_scores.append(accuracy_score(y_val, y_pred))
    recall_scores.append(recall_score(y_val, y_pred))
# Let us compare the Accuracy and Recall scores for all the models.
# As mentioned in the Introduction, Recall is more important than Precision in this study.

plt.figure(figsize=(15,6))
plt.style.use('fivethirtyeight')

mylist = ['SGD Classifier', 'Logistic Regression', 'KNN', 'SVC', 'Linear SVC', 'Decision Tree', 'Random Forest']

sns.lineplot(x=mylist, y=accuracy_scores, label='accuracy')
sns.lineplot(x=mylist, y=recall_scores, label='recall')

plt.title('Accuracy & Recall for Classifiers')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy/Recall score')

plt.legend(loc='center left')
# Hyperparameter tuning of SGDClassifier

# Tuning of 'penalty' and 'alpha' hyperparameters is intended to modify the regularization of the classifier so that it fits more snugly to the data.
# Tuning of 'max_iter' is done so that the optimization doesn't stop prematurely for lack of iterations allowed.

parameters = [{'penalty':['l1', 'l2'], 'alpha':np.arange(0.00005, 0.001, 0.00005), 'max_iter':range(1000, 2000, 100)}]

sgd_clf = SGDClassifier(random_state=42)

grid_search_sgd = GridSearchCV(sgd_clf, parameters, cv=3, scoring='accuracy') # cv is 5 by default, n_iter is 10 by default

grid_search_sgd.fit(X_train, y_train)
# Best parameters for SGD Classifier

grid_search_sgd.best_params_
# Let's fit the best estimator on validation data and see if there is any improvement vis-a-vis previous SGD classifier model

# Best estimator 
sgd_best = grid_search_sgd.best_estimator_

# Make predictions for validation data
y_pred = sgd_best.predict(X_val)
    
# Performance measures for validation data
print('confusion matrix:', '\n', confusion_matrix(y_val, y_pred))
print('precision score:', precision_score(y_val, y_pred))
print('recall score:', recall_score(y_val, y_pred))
print('accuracy score:', accuracy_score(y_val, y_pred))
# Hyperparameter tuning for Logistic Regression

# Tuning of 'penalty', 'C' and 'solver' hyperparameters is intended to modify the regularization of the classifier so that it fits more snugly to the data.

parameters = [{'penalty':['l1', 'l2'], 'C':np.arange(0.1, 2.0, 0.1), 'solver':['liblinear', 'saga']},
              {'penalty':['l2'], 'C':np.arange(0.1, 2.0, 0.1), 'solver':['newton-cg', 'lbfgs', 'sag']}]

log_clf = LogisticRegression(random_state=42)

grid_search_log = GridSearchCV(log_clf, parameters, cv=3, scoring='accuracy') # cv is 5 by default, n_iter is 10 by default

grid_search_log.fit(X_train, y_train)
# Best parameters for Logistic Regression

grid_search_log.best_params_
# Let's fit the best estimator on validation data and see if there is any improvement vis-a-vis previous SGD classifier model

# Best estimator
log_best = grid_search_log.best_estimator_

# Make predictions for validation data
y_pred = log_best.predict(X_val)
    
# Performance measures for validation data
print('confusion matrix:', '\n', confusion_matrix(y_val, y_pred))
print('precision score:', precision_score(y_val, y_pred))
print('recall score:', recall_score(y_val, y_pred))
print('accuracy score:', accuracy_score(y_val, y_pred))
# Hyperparameter tuning for linear SVC 

# Tuning of 'penalty' and 'C' hyperparameters is intended to modify the regularization of the classifier so that it fits more snugly to the data.
# Tuning of 'max_iter' is done so that the optimization doesn't stop prematurely for lack of iterations allowed.

parameters = [{'penalty':['l1', 'l2'], 'C':np.arange(0.1, 5.0, 0.2), 'dual':[False], 'max_iter':range(1000, 2000, 100)},
             {'penalty':['l2'], 'C':np.arange(0.1, 5.0, 0.2), 'loss':['squared_hinge'], 'max_iter':range(1000, 2000, 100)}]

lsvc_clf = LinearSVC(random_state=42)

grid_search_lsvc = GridSearchCV(lsvc_clf, parameters, cv=3, scoring='accuracy') # cv is 5 by default, n_iter is 10 by default

grid_search_lsvc.fit(X_train, y_train)
# Best parameters for Linear SVC

grid_search_lsvc.best_params_
# Let's fit the best estimator on validation data and see if there is any improvement vis-a-vis previous SGD classifier model

# Best estimator

lsvc_best = grid_search_lsvc.best_estimator_

# Make predictions for validation data
y_pred = lsvc_best.predict(X_val)
    
# Performance measures for validation data
print('confusion matrix:', '\n', confusion_matrix(y_val, y_pred))
print('precision score:', precision_score(y_val, y_pred))
print('recall score:', recall_score(y_val, y_pred))
print('accuracy score:', accuracy_score(y_val, y_pred))
# Voting Classifier 

voting_clf = VotingClassifier(estimators=[('lr', log_best), ('sg', sgd_best), ('lsvc', lsvc_best)], voting='hard')

voting_clf.fit(X_train, y_train)

# Make predictions for validation data
y_pred = voting_clf.predict(X_val)
    
# Performance measures for validation data
print('confusion matrix:', '\n', confusion_matrix(y_val, y_pred))
print('precision score:', precision_score(y_val, y_pred))
print('recall score:', recall_score(y_val, y_pred))
print('accuracy score:', accuracy_score(y_val, y_pred))
# AdaBoost Classifier 

for clf in [log_best, lsvc_best]: # Haven't considered SGD classifier here as an error is encountered during its execution which I am not able to understand. 
    ada_clf = AdaBoostClassifier(clf, n_estimators=200, algorithm="SAMME", learning_rate=0.5)
    ada_clf.fit(X_train, y_train)
    
    # Make predictions for validation data
    y_pred = ada_clf.predict(X_val)
    
    # Performance measures for validation data
    print('confusion matrix for {}:'.format(clf.__class__.__name__), '\n', confusion_matrix(y_val, y_pred))
    print('precision score for {}:'.format(clf.__class__.__name__), precision_score(y_val, y_pred))
    print('recall score for {}:'.format(clf.__class__.__name__), recall_score(y_val, y_pred))
    print('accuracy score for {}:'.format(clf.__class__.__name__), accuracy_score(y_val, y_pred))
    print('-'*100)
# Process the test set data 

# X test and y test 

df_test_new = new_set(12, df_test)

X_test = df_test_new.drop('class', axis=1)
y_test = df_test_new['class']

# Scale the X test values

X_test = scaler.transform(X_test)
# Evaluate the Classifiers on the Test data

for clf in [knn_clf, svc_clf, dt_clf, rf_clf]:
    
    # Make predictions for test data
    y_pred = clf.predict(X_test)
    
    # Performance measures for test data
    print('confusion matrix for {}:'.format(clf.__class__.__name__), '\n', confusion_matrix(y_test, y_pred))
    print('precision score for {}:'.format(clf.__class__.__name__), precision_score(y_test, y_pred))
    print('recall score for {}:'.format(clf.__class__.__name__), recall_score(y_test, y_pred))
    print('accuracy score for {}:'.format(clf.__class__.__name__), accuracy_score(y_test, y_pred))
    print('-'*100)