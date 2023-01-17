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
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.info()
cols_to_drop = ['id', 'Unnamed: 32']
df = df.drop(cols_to_drop, axis=1)
df.head()
df.describe().T
sns.countplot(df['diagnosis'])
correlation_coeffs = df.corr()
correlation_stack = correlation_coeffs.unstack()
correlation_stack_sorted = correlation_stack.sort_values(kind="quicksort", ascending=True)
correlation_stack_sorted[-50:]
y = df['diagnosis']
df.drop('diagnosis', axis=1, inplace=True)
df_std = (df - df.mean())/df.std()
df_std.head()
def plot_data(data, y, plot_type):
    data = pd.concat([y, data], axis=1)
    data = pd.melt(data,id_vars="diagnosis",
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10,10))
    if plot_type=='violin':
        sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")
    elif plot_type=='box':
        sns.boxplot(x="features", y="value", hue="diagnosis", data=data)
    elif plot_type=='swarm':
        sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)
    plt.xticks(rotation=90)
plot_data(df_std.iloc[:, 0:10], y, "violin")
plot_data(df_std.iloc[:, 11:20], y, "violin")
plot_data(df_std.iloc[:, 21:30], y, "violin")
plot_data(df_std.iloc[:, 0:10], y, 'box')
plot_data(df_std.iloc[:, 11:20], y, 'box')
plot_data(df_std.iloc[:, 0:10], y, 'swarm')
plot_data(df_std.iloc[:, 10:20], y, 'swarm')
plot_data(df_std.iloc[:, 21:30], y, 'swarm')
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), cmap='Blues', annot=True,linewidths=.5, fmt= '.1f')
plt.show()
df.columns
cols_to_drop = ['area_mean','radius_mean','compactness_mean','concavity_mean',
                'area_se','perimeter_se','perimeter_worst', 
                'compactness_worst','concave points_worst','compactness_se',
                'concave points_se','texture_worst','radius_worst']

data = df.drop(cols_to_drop, axis=1)
data.head()
plt.figure(figsize=(15,10))
sns.heatmap(data.corr(), cmap='Blues', annot=True,linewidths=.5, fmt= '.1f')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)

#random forest classifier with n_estimators=10 (default)
clf_rf = RandomForestClassifier(random_state=43)      
clr_rf = clf_rf.fit(x_train,y_train)

ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")
print(classification_report(y_test,clf_rf.predict(x_test)))
features = pd.DataFrame()
features['Feature'] = x_train.columns
features['Importance'] = clf_rf.feature_importances_
features.sort_values(by=['Importance'], ascending=False, inplace=True)
features.set_index('Feature', inplace=True)
features.plot(kind='bar', figsize=(20, 10))
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
rf_clf2 = RandomForestClassifier() 

rfecv = RFECV(estimator=rf_clf2, step=1, cv=3, scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(data, y)

print('Optimal number of features :', rfecv.n_features_)
print('Best features :', x_train.columns[rfecv.support_])
from numpy import sort
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
y = y.map({'B':0, 'M':1}).astype('int')
# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=7, stratify=y)
# fit model on all training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [np.round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
# How to get back feature_importances_ (gain based) from plot_importance fscore
# Calculate two types of feature importance:
# Weight = number of times a feature appears in tree
# Gain = average gain of splits which use the feature = average all the gain values of the feature if it appears multiple times
# Normalized gain = Proportion of average gain out of total average gain

k = model.get_booster().trees_to_dataframe()
group = k[k['Feature']!='Leaf'].groupby('Feature').agg(fscore = ('Gain', 'count'),
feature_importance_gain = ('Gain', 'mean'))

# Feature importance same as plot_importance(importance_type = ‘weight’), default value
group['fscore'].sort_values(ascending=False)
# Feature importance same as clf.feature_importance_ default = ‘gain’
group['feature_importance_gain_norm'] = group['feature_importance_gain']/group['feature_importance_gain'].sum()
group.sort_values(by='feature_importance_gain_norm', ascending=False)
print('3')
# Feature importance same as plot_importance(importance_type = ‘gain’)
group[['feature_importance_gain']].sort_values(by='feature_importance_gain', ascending=False)

