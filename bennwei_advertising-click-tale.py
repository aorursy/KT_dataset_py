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
df = pd.read_csv('../input/advertising.csv') # load the data
df.head()
# Featuretools is a framework to perform automated feature engineering. 
# It excels at transforming transactional and relational datasets into feature matrices for machine learning.
import featuretools as ft 
# Check if there is any missing values
df.isnull().sum()
# Explore a bit about the count, mean, standard deviation, minimum and maximum values and the quantiles of the data
df.describe()
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('classic')
%matplotlib inline
df.info()
len(df['City'].unique())
len(df['Country'].unique())
df['Clicked on Ad'].value_counts()
g0 = sns.countplot(x='Clicked on Ad', data = df, palette='husl')
df.groupby('Clicked on Ad').mean()
df.groupby('Country').mean()
df.groupby('Country')['Clicked on Ad'].mean().nlargest(30)
df_1 = df.groupby('Country').mean()
g1 = sns.pairplot(df_1, palette="husl", kind="reg")

g = sns.pairplot(df, hue="Clicked on Ad", palette="husl")
# Convert Timestamp column as a datetime object
# http://hamelg.blogspot.com/2015/11/python-for-data-analysis-part-17.html
df2 = df.copy()
df2['Timestamp'] = pd.to_datetime(df2["Timestamp"] )
df2.info()
df2["month"] = df2['Timestamp'].dt.month
df2["day"] = df2['Timestamp'].dt.day
df2["dayofweek"] = df2['Timestamp'].dt.dayofweek
df2["hour"] = df2['Timestamp'].dt.hour
df2.head()
g2 = sns.pairplot(df2[['Clicked on Ad', 'month', 'day', 'dayofweek', 'hour']], hue="Clicked on Ad", palette="husl")

df3 = df2.copy()
df3 = pd.concat([df3, pd.get_dummies(df3['Country'], prefix='Country')],axis=1)
df3 = pd.concat([df3, pd.get_dummies(df3['month'], prefix='Month')],axis=1)
df3 = pd.concat([df3, pd.get_dummies(df3['dayofweek'], prefix='Dayofweek')],axis=1)
# create a bucket for hours into [0-5, 6-11, 12-17, 18-23] hour
df3['Hour_bin'] = pd.cut(df3['hour'], [0, 5, 11, 17, 23], labels=['hour_0-5', 'hour_6-11', 'hour_12-17', 'hour_18-23'], include_lowest=True)
df3 = pd.concat([df3, pd.get_dummies(df3['Hour_bin'], prefix='Hour')],axis=1)
df3.drop(['Country', 'Ad Topic Line', 'City', 'Timestamp', 'day', 'month', 'dayofweek', 'hour', 'Hour_bin'],axis=1, inplace=True)
df3.head(10)
# Check our final column variable names
df3.columns.values
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# Select dependent variable and prediction outcome from the data
df3_final_vars = df3.columns.values.tolist()
y = df3['Clicked on Ad']
X_features = [i for i in df3_final_vars if i not in y]
X = df3[X_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=10)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
# Looking at different performance evaluation metrics in testing data set：confusion matrix, ROC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from time import time

def evaluation(estimator, X_test, y_test):

    start = time()
    y_pred = estimator.predict(X_test)
    print("Querying with the best model took %f seconds." % (time() - start))
    print(len(y_pred))
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, y=i,
                    s=confmat[i, j],
                    va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()

    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))
    print('ROC AUC: %.3f' % roc_auc_score(y_true=y_test, y_score=y_pred))
    print('Accuracy: %.3f' % accuracy_score(y_true=y_test, y_pred=y_pred))
    print('-----------------------------------------')
    print(metrics.classification_report(y_test, y_pred))
evaluation(logreg, X_test, y_test)
profit_of_campaign = 126 * 100 - 3 * 50
print('How much money is made: ${}'.format(profit_of_campaign))
print('How much money is made per customer: ${}'.format(profit_of_campaign/129))
# Using Sklearn's Pipeline function to combining features transformers and estimators; 
# It consisted of two intermediate steps, a StandardScaler and a PCA transformer, and a LogisticRegression classifier as a final estimator.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Check performance of LogisticRegression algorithm with feature thransformation of StandardScaler and a PCA transformer
pipe_1 = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)), ('lr', LogisticRegression(random_state=2))])
pipe_1.fit(X_train, y_train)
print('LogisticRegression (with scaler/PCA) Test Accuracy: %.3f' % pipe_1.score(X_test, y_test))
results_2 = model_selection.cross_val_score(pipe_1, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy of pipe_1: %.3f" % (results_2.mean()))
# Let's do an evaluation to check all performance metrics. 
evaluation(pipe_1, X_test, y_test)
profit_of_campaign_2 = 133 * 100 - 3 * 50
print('How much money is made: ${}'.format(profit_of_campaign_2))
print('How much money is made per customer: ${}'.format(profit_of_campaign_2/136))
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Select dependent variable and prediction outcome from the data
df3_final_vars = df3.columns.values.tolist()
y = ['Clicked on Ad']
X = [i for i in df3_final_vars if i not in y]

feature_names = df3.columns.values
# print(feature_names)
logreg = LogisticRegression()
rfe = RFE(logreg)
rfe = rfe.fit(df3[X], df3[y])
print(rfe.support_)
print("Features sorted by their rank:")
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), feature_names)))
mask = rfe.support_ #list of booleans
new_features = [] # The list of your K best features
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
print(new_features)
df3_final = df3[new_features]
#print(df3_final)
df3_final.head(10)