# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #graph
import seaborn as sns 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/bigml_59c28831336c6604c800002a.csv')
df.head()
df.dtypes
df["voice mail plan"].value_counts()
df["international plan"].value_counts()
cleanup_nums = {"voice mail plan":     {"no": 0, "yes": 1},
                "international plan": {"no": 0, "yes": 1 }
               }

obj_df = df.copy()
obj_df.replace(cleanup_nums, inplace=True)
obj_df.head()
obj_df = df.copy()
obj_df.replace(cleanup_nums, inplace=True)
obj_df.head()
print(obj_df.groupby('churn')['phone number'].count())
######################################################
############################
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

drp = obj_df[['state','area code','phone number','international plan','voice mail plan','churn']]
X= obj_df.drop(drp,1)
y= obj_df.churn
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn import metrics
print('Logistic regression score =',round(metrics.accuracy_score(y_test, y_pred),2))
conf = (metrics.confusion_matrix(y_test, y_pred))
cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
sns.heatmap(conf,cmap = cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True, fmt="d",)
plt.xlabel('Predicted')
plt.ylabel('Actual')
corr = obj_df.corr()
#print(corr)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
df_final = obj_df.drop([ "total intl calls", "voice mail plan", "number vmail messages", "state", "area code", "phone number", "total day minutes", "total eve minutes", "total night minutes", "total intl charge"], axis=1 )
df_final.head()

corr = df_final.corr()
#print(corr)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
df_final.boxplot()
df_final
label = df_final['churn']
features = df_final.drop('churn', axis=1)
features.dtypes
features.head()
# Encode the 'fraud_predicted' data to numerical values
#label_final = label.map({'False':0, 'True':1})
label_final = label.astype(int)
label_final.head()
#encode
encoded = list(features.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))
# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, label_final, test_size = 0.2, stratify=label_final, random_state = 10)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))
# Import the model we are using
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Instantiate model with 10 decision trees
model = LogisticRegression(random_state=None)
# Train the model on training data
model.fit(X_train, y_train)

# Use the forest's predict method on the test data
preds = model.predict(X_test)

from sklearn import metrics
print('Accuracy score: ', metrics.accuracy_score(y_test, preds))

print(metrics.confusion_matrix(y_test, preds))

print(metrics.classification_report(y_test, preds))
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 10 decision trees
model1 = RandomForestClassifier(n_estimators = 50, random_state = 10, n_jobs=-1)
# Train the model on training data
model1.fit(X_train, y_train)

# Use the forest's predict method on the test data
preds = model1.predict(X_test)

from sklearn import metrics
print('Accuracy score: ', metrics.accuracy_score(y_test, preds))

print(metrics.confusion_matrix(y_test, preds))

print(metrics.classification_report(y_test, preds))
