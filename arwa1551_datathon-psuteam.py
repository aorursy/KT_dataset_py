# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import seaborn as sns
import lightgbm
import matplotlib.pyplot as plt
from decimal import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score,cross_validate
from xgboost import XGBClassifier
from sklearn import ensemble
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, BaggingRegressor)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_absolute_error

train_data = pd.read_csv('/kaggle/input/widsdatathon2020/training_v2.csv')
test_data = pd.read_csv('/kaggle/input/widsdatathon2020/unlabeled.csv')
print(train_data.shape)
print(test_data.shape)
train_data.describe()
test_data.describe()
plt.figure(figsize=(15,9))
plt.xticks(rotation=90)
sns.countplot(x='age', hue='hospital_death', data= train_data);
sns.countplot(x='gender', hue='hospital_death', data=train_data);
sns.countplot(x='hospital_death', data=train_data);

np.round(train_data['hospital_death'].value_counts()/train_data.shape[0]*100,2)
train_data.head()
drop_columns = ['encounter_id','patient_id', 'hospital_id', 'icu_id']
train_data = train_data.drop(drop_columns, axis=1)
test_data = test_data.drop(drop_columns, axis=1)
# How much of the data is missing 
np.round(train_data.isna().sum()/train_data.shape[0]*100,2).sort_values(ascending=False)

np.round(test_data.isna().sum()/train_data.shape[0]*100,2)

null_values = train_data.isnull().sum()/len(train_data)*100
missing_features = null_values[null_values > 30].index

train_data.drop(missing_features, axis=1, inplace=True)
train_data.shape
test_data.drop(missing_features, axis=1, inplace=True)
test_data.shape
len(missing_features)
#list of features to dummy
todummy_list = []

# Check how many unique categories I have 
for col_name in train_data.columns:
    if train_data[col_name].dtype == 'object':
        unique_cat = len(train_data[col_name].unique())
        todummy_list.append(col_name)
        print('Feature {col_name} has {unique_cat} unique categories'.format(col_name=col_name,unique_cat=unique_cat ))
print(train_data['ethnicity'].value_counts(normalize=True, ascending=False)*100)
train_data['ethnicity'] = ['Caucasian' if value == 'Caucasian' else 'Others' for value in train_data['ethnicity']]
test_data['ethnicity'] = ['Caucasian' if value == 'Caucasian' else 'Others' for value in test_data['ethnicity']]

print(train_data['ethnicity'].value_counts(normalize=True, ascending=False)*100)
print(train_data['hospital_admit_source'].value_counts(normalize=True, ascending=False)*100)
train_data['hospital_admit_source'] = ['Emergency Department' if value == 'Emergency Department' else 'Others' for value in train_data['hospital_admit_source']]
test_data['hospital_admit_source'] = ['Emergency Department' if value == 'Emergency Department' else 'Others' for value in test_data['hospital_admit_source']]

print(train_data['hospital_admit_source'].value_counts(normalize=True, ascending=False)*100)
print(train_data['icu_admit_source'].value_counts(normalize=True, ascending=False)*100)
train_data['icu_admit_source'] = ['Accident & Emergency' if value == 'Accident & Emergency' else 'Others' for value in train_data['icu_admit_source']]
test_data['icu_admit_source'] = ['Accident & Emergency' if value == 'Accident & Emergency' else 'Others' for value in test_data['icu_admit_source']]

print(train_data['icu_admit_source'].value_counts(normalize=True, ascending=False)*100)
print(train_data['icu_stay_type'].value_counts(normalize=True, ascending=False)*100)
train_data['icu_stay_type'] = ['admit' if value == 'admit' else 'Others' for value in train_data['icu_stay_type']]
test_data['icu_stay_type'] = ['admit' if value == 'admit' else 'Others' for value in test_data['icu_stay_type']]

print(train_data['icu_stay_type'].value_counts(normalize=True, ascending=False)*100)
print(train_data['icu_type'].value_counts(normalize=True, ascending=False)*100)
train_data['icu_type'] = ['Med-Surg ICU' if value == 'Med-Surg ICU' else 'Others' for value in train_data['icu_type']]
test_data['icu_type'] = ['Med-Surg ICU' if value == 'Med-Surg ICU' else 'Others' for value in test_data['icu_type']]

print(train_data['icu_type'].value_counts(normalize=True, ascending=False)*100)
print(train_data['apache_3j_bodysystem'].value_counts(normalize=True, ascending=False)*100)
print(train_data['apache_2_bodysystem'].value_counts(normalize=True, ascending=False)*100)
#function to dummy all the categorical variables used for modeling
def dummy_df(df, todummy_list):
    for feature in todummy_list:
        dummies = pd.get_dummies(df[feature], prefix= feature, dummy_na = False)
        df = df.drop(feature, 1)
        df = pd.concat([df, dummies], axis =1)
    return df
train_data = dummy_df(train_data, todummy_list)
test_data = dummy_df(test_data, todummy_list)
print(train_data.shape)
print(test_data.shape)
#Assign X a DataFrame of features and y as a Series of outcome variable
X = train_data.drop('hospital_death', 1)
y = train_data.hospital_death
X_test = test_data.drop('hospital_death', 1)
y_test = test_data.hospital_death
pd.DataFrame(X).fillna(X.median(), inplace=True)
pd.DataFrame(X_test).fillna(X_test.median(), inplace=True)
X.isna().sum().sort_values(ascending=False)
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
print('XGBClassifier Model:')
XGB_CV = pd.DataFrame(cross_validate(XGBClassifier(), X_scaled, y, cv = 3, return_train_score=True, scoring = ['accuracy', 'precision', 'recall', 'f1']))
XGB_CV.mean()
import sklearn.feature_selection
select = sklearn.feature_selection.SelectKBest(k=50)
selected_features = select.fit(X,y)
indices_selected = selected_features.get_support(indices=True)
col_names_selected = [X.columns[i] for i in indices_selected]

X_selected = X[col_names_selected]

col_names_selected
scaler_2 = StandardScaler()
scaler.fit(X_scaled)
X_scaled_2 = scaler.transform(X_scaled)

print('LogisticRegression Model:')
log_CV = pd.DataFrame(cross_validate(LogisticRegression(), X_scaled_2, y, cv = 3, return_train_score=True, scoring = ['accuracy', 'precision', 'recall', 'f1']))
log_CV.mean()
print('XGBClassifier Model:')
XGB_CV = pd.DataFrame(cross_validate(XGBClassifier(), X_scaled_2, y, cv = 3, return_train_score=True, scoring = ['accuracy', 'precision', 'recall', 'f1']))
XGB_CV.mean()
print('AdaBoostClassifier Model:')
ada_model = pd.DataFrame(cross_validate(ensemble.AdaBoostClassifier(), X_selected, y, cv = 3, return_train_score=True, scoring = ['accuracy', 'precision', 'recall', 'f1']))
ada_model.mean()
from imblearn.over_sampling import SMOTE
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
sns.countplot(x=y, data=train_data);
np.round(y.value_counts()/len(y)*100,2)
scaler_3 = StandardScaler()
scaler.fit(X)
X_scaled_3= scaler.transform(X)
X_test_scaled = scaler.transform(X_test)
print('XGBClassifier Model:')
XGB_CV = pd.DataFrame(cross_validate(XGBClassifier(), X_scaled_3, y, cv = 3, return_train_score=True, scoring = ['accuracy', 'precision', 'recall', 'f1']))
XGB_CV.mean()
def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=18):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names, )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.xlim(0,len(class_names))
    plt.ylim(len(class_names),0)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
def model_test(model, X, y):
    # perform train/val split
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X, y,  eval_metric='auc')
    pred = model.predict(X_test)
    
    print('Mean Absolute Error:', mean_absolute_error(y_test,pred))
    print('Accuracy score:', accuracy_score(y_test, pred))
    conf = confusion_matrix(y_test, pred)
    print(classification_report(y_test, pred))
    print_confusion_matrix(conf, ['1', '0'])
model = XGBClassifier()
model_test(model, X_scaled_3, y)
#model.fit(X_scaled_3, y)
y_test = model.predict(X_test_scaled)
solution_template = pd.read_csv("/kaggle/input/widsdatathon2020/solution_template.csv")
print(y_test.shape)
print(solution_template.shape)
solution_template.hospital_death = y_test
solution_template.to_csv("Version_2.csv", index=0)
from keras.models import Sequential
from keras.layers import Dense, Dropout
# define the keras model
nn_model = Sequential()
nn_model.add(Dense(15, input_dim=X.shape[1], activation='relu'))
nn_model.add(Dense(10, activation='relu'))
nn_model.add(Dense(8, activation='relu'))
nn_model.add(Dropout(.2))
nn_model.add(Dense(1, activation='sigmoid'))
# compile the keras model
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
nn_model.fit(X, y, epochs=150, batch_size=10)
# save the model
nn_model.save('nn_model.h5')
# make class predictions with the model
predictions = nn_model.predict_classes(X_test)

solution_template.hospital_death = predictions
solution_template.to_csv("Version_3.csv", index=0)

