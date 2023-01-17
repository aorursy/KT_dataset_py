import numpy as np 
import pandas as pd 

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Graphics in SVG format are more sharp and legible
%config InlineBackend.figure_format = 'svg'


import os
print(os.listdir("../input/"))
DATA_PATH = "../input/WA_Fn-UseC_-HR-Employee-Attrition.csv"
df = pd.read_csv(DATA_PATH)
df.info()
df.head()
print(df['Attrition'].value_counts())
sns.countplot(df['Attrition']);
# Filter Data
numerical, categorical = [], []

for col in df.columns:
    if (df[col].dtype == 'O') or (len(df[col].value_counts()) < 10):
        categorical.append(col)
    else:
        numerical.append(col)
# Creating a correlation matrix
corr_matrix = df[numerical].corr()
# Plotting heatmap
sns.heatmap(corr_matrix);
for col in categorical:
    if len(df[col].value_counts()) < 2:
        print(col)
# Dropping the columns
df = df.drop(columns=['EmployeeCount', 'Over18','StandardHours'])
# Updating our numerical features to not include the dropped columns
categorical = list(set(categorical) - set(['EmployeeCount', 'Over18','StandardHours']))
categorical
_, axes = plt.subplots(nrows=1, ncols=(len(categorical)), figsize=(200, 4))
for i,col in enumerate(categorical):
    if col != 'Attrition':
        sns.countplot(x=col, hue='Attrition', data=df, ax=axes[i])
df_processed = df.drop(columns=['Attrition'])
objects, numbers = [], []

for col in df_processed.columns:
    if (df_processed[col].dtype == 'O'):
        objects.append(col)
    else:
        numbers.append(col)
df_object = df_processed[objects]
df_object = pd.get_dummies(df_object)
df_object.head()
df_final = pd.concat([df_processed[numbers], df_object], axis=1)
df_final.head()
target = df['Attrition'].map({'Yes': 1, 'No': 0})
# Importing train-test split method from sklearn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
train_X, test, train_y, val_y = train_test_split(df_final, target, train_size=0.9, random_state=0);
from imblearn.over_sampling import SMOTE
oversampler = SMOTE(random_state=0)
smote_X, smote_y = oversampler.fit_sample(train_X, train_y)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
params = {
    'n_jobs': -1,
    'n_estimators': 1000,
    'max_features': 0.3,
    'max_depth': 4,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'random_state' : 0,
    'verbose': 0
}
random_forest = RandomForestClassifier(**params)
random_forest.fit(smote_X, smote_y)
predictions = random_forest.predict(test)
print("Accuracy = {}".format(accuracy_score(val_y, predictions)))
