# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/heart-csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/heart-csv/heart.csv')
df.head()
df.shape
df.info()
df.describe().transpose()
df.isnull().sum()
df.columns
# Visualizing target variable Clicked on Ad
plt.figure(figsize = (14, 6)) 
plt.subplot(1,2,1)            
sns.countplot(x = 'target', data = df)
plt.subplot(1,2,2)
sns.distplot(df["sex"], bins = 20)
plt.show()
# Visualizing target variable Clicked on Ad
plt.figure(figsize = (20, 6)) 
plt.subplot(1,2,1)            
chart=sns.countplot(data = df,x='age')
plt.subplot(1,2,2)
sns.distplot(df["age"], bins = 20)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plt.show()
df[["target", "sex"]].groupby(['sex'], as_index=False).mean().sort_values(by='target', ascending=False)
g = sns.FacetGrid(df, col='sex')
g.map(plt.hist, 'age', bins=20)
df.columns
plt.figure(figsize = (20, 6)) 
plt.subplot(1,2,1)            
chart=sns.countplot(data = df,x='cp')
plt.subplot(1,2,2)
sns.distplot(df["cp"], bins = 20)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plt.show()
grid = sns.FacetGrid(df, col='target', row='cp', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();
plt.figure(figsize = (20, 6)) 
plt.subplot(1,2,1)            
chart=sns.countplot(data = df,x='trestbps')
plt.subplot(1,2,2)
sns.distplot(df["trestbps"], bins = 20)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plt.show()
# plt.figure(figsize = (20, 6)) 
# plt.subplot(1,2,1)            
# chart=sns.countplot(data = df,x='fbs')
# plt.subplot(1,2,2)
# sns.distplot(df["fbs"], bins = 20)
# chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
# plt.show()
plt.figure(figsize = (20, 6)) 
plt.subplot(1,2,1)            
chart=sns.countplot(data = df,x='restecg')
plt.subplot(1,2,2)
sns.distplot(df['restecg'], bins = 20)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plt.show()
plt.figure(figsize = (20, 6)) 
plt.subplot(1,2,1)            
chart=sns.countplot(data = df,x='exang')
plt.subplot(1,2,2)
sns.distplot(df['exang'], bins = 20)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plt.show()
df[["sex", "exang"]].groupby(['exang'], as_index=False).mean().sort_values(by='sex', ascending=False)
grid = sns.FacetGrid(df, col='exang', row='sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();
plt.figure(figsize = (20, 6)) 
plt.subplot(1,2,1)            
chart=sns.countplot(data = df,x='oldpeak')
plt.subplot(1,2,2)
sns.distplot(df['oldpeak'], bins = 20)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plt.show()
plt.figure(figsize = (20, 6)) 
plt.subplot(1,2,1)            
chart=sns.countplot(data = df,x='slope')
plt.subplot(1,2,2)
sns.distplot(df['slope'], bins = 20)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plt.show()
plt.figure(figsize = (20, 6)) 
plt.subplot(1,2,1)            
chart=sns.countplot(data = df,x='ca')
plt.subplot(1,2,2)
sns.distplot(df['ca'], bins = 20)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plt.show()
plt.figure(figsize = (20, 6)) 
plt.subplot(1,2,1)            
chart=sns.countplot(data = df,x='thal')
plt.subplot(1,2,2)
sns.distplot(df['thal'], bins = 20)
chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
plt.show()
grid = sns.FacetGrid(df, col='sex', row='thal', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();
fig,ax=plt.subplots(figsize=(15,15))
sns.heatmap(df.corr(), cmap='Blues', annot = True) 
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
sns.factorplot(x="age", y="thalach", col="target", data=df, kind="box",size=5, aspect=2.0)
sns.factorplot(x="age", y="cp", col="target", data=df, kind="box",size=5, aspect=2.0) 
# Importing train_test_split from sklearn.model_selection family
from sklearn.model_selection import train_test_split
# Import LogisticRegression from sklearn.linear_model family
from sklearn.linear_model import LogisticRegression
# Assigning Numerical columns to X & y only as model can only take numbers
X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal']]
y = df['target']
# Splitting the data into train & test sets 
# test_size is % of data that we want to allocate & random_state ensures a specific set of random splits on our data because 
#this train test split is going to occur randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 
# We dont have to use stratify method in train_tst_split to handle class distribution as its not imbalanced and does contain equal number of classes i.e 1's and 0's
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# Instantiate an instance of the linear regression model (Creating a linear regression object)
logreg = LogisticRegression()
# Fit the model on training data using a fit method
model = logreg.fit(X_train,y_train)
model
# The predict method just takes X_test as a parameter, which means it just takes the features to draw predictions
predictions = logreg.predict(X_test)
# Below are the results of predicted click on Ads
predictions[0:20]
# Importing classification_report from sklearn.metrics family
from sklearn.metrics import classification_report

# Printing classification_report to see the results
print(classification_report(y_test, predictions))
# Importing a pure confusion matrix from sklearn.metrics family
from sklearn.metrics import confusion_matrix

# Printing the confusion_matrix
print(confusion_matrix(y_test, predictions))
import  statsmodels.api  as sm
from scipy import stats

X2   = sm.add_constant(X_train)
est  = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())
#Creating K fold Cross-validation 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, # model
                         X_train, # Feature matrix
                         y_train, # Target vector
                         cv=kf, # Cross-validation technique
                         scoring="accuracy", # Loss function
                         n_jobs=-1) # Use all CPU scores
print('10 fold CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(criterion='gini', n_estimators=400,
                             min_samples_split=10,min_samples_leaf=1,
                             max_features='auto',oob_score=True,
                             random_state=42,n_jobs=-1)
rf.fit(X_train,y_train)
# Predict using model
rf_training_pred = rf.predict(X_train)
rf_training_prediction = accuracy_score(y_train, rf_training_pred)

print("Accuracy of Random Forest training set:",   round(rf_training_prediction,3))

from sklearn.model_selection import cross_val_predict
print('The cross validated score for Random Forest Classifier is:',round(scores.mean()*100,2))
y_pred = cross_val_predict(rf,X_train,y_train,cv=10)
sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="winter")
plt.title('Confusion_matrix', y=1.05, size=15)
new_df = df.copy() # just to keep the original dataframe unchanged
# Assigning Numerical columns to X & y only as model can only take numbers
X1 = df[[ 'sex', 'cp', 'chol', 'fbs', 'restecg', 'thalach', 'slope', 'thal']]
y1 = df['target']
# Splitting the data into train & test sets 
# test_size is % of data that we want to allocate & random_state ensures a specific set of random splits on our data because 
#this train test split is going to occur randomly
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.33, random_state=42) 
# We dont have to use stratify method in train_tst_split to handle class distribution as its not imbalanced and does contain equal number of classes i.e 1's and 0's
print(X_train1.shape, y_train1.shape)
print(X_test1.shape, y_test1.shape)
logreg = LogisticRegression()
# Fit the model on training data using a fit method
model1 = logreg.fit(X_train1,y_train1)
model1
# The predict method just takes X_test as a parameter, which means it just takes the features to draw predictions
predictions1 = logreg.predict(X_test1)
# Below are the results of predicted click on Ads
predictions1[0:20]
# Printing classification_report to see the results
print(classification_report(y_test1, predictions1))
# Importing a pure confusion matrix from sklearn.metrics family
from sklearn.metrics import confusion_matrix

# Printing the confusion_matrix
print(confusion_matrix(y_test1, predictions1))
import  statsmodels.api  as sm
from scipy import stats

X21   = sm.add_constant(X_train1)
est1 = sm.OLS(y_train1, X21)
est21 = est1.fit()
print(est21.summary())
print ("\n\n ---Logistic Regression Model---")
lr_auc = roc_auc_score(y_test, model.predict(X_test))

print ("Logistic Regression AUC = %2.2f" % lr_auc)
print(classification_report(y_test, model.predict(X_test)))

print ("\n\n ---Logistic Regression Model deleting some features---")
lr_auc1 = roc_auc_score(y_test1, model1.predict(X_test1))

print ("Logistic Regression AUC = %2.2f" % lr_auc)
print(classification_report(y_test1, model1.predict(X_test1)))

print ("\n\n ---Random Forest Model---")
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))

print ("Random Forest AUC = %2.2f" % rf_roc_auc)
print(classification_report(y_test, rf.predict(X_test)))
# Create ROC Graph
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])


plt.figure()

# Plot Logistic Regression ROC
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % lr_auc)

# Plot Random Forest ROC
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier (area = %0.2f)' % rf_roc_auc)


# Plot Base Rate ROC
plt.plot([0,1], [0,1],label='Base Rate')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()
columns = X.columns
train = pd.DataFrame(np.atleast_2d(X_train), columns=columns) # Converting numpy array list into dataframes
# Get Feature Importances
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)
feature_importances = feature_importances.reset_index()
feature_importances.head(10)
sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(13, 7))

# Plot the Feature Importance
sns.set_color_codes("pastel")
sns.barplot(x="importance", y='index', data=feature_importances[0:10],
            label="Total", color="b")