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
#!pip install kaggle

#!kaggle datasets download -d mlg-ulb/creditcardfraud
import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef

from sklearn.neural_network import MLPClassifier
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df.head()
df.info()
print("Mean amount of transactions for fraudulent ones: ", df[df.Class == 1]['Amount'].mean())

print("Mean amount of transactions for good ones: ", df[df.Class == 0]['Amount'].mean())
# Visualizing the distribution of data



for i, c in enumerate(df.columns):

    sns.distplot(df[df.columns[i]])

    plt.show()
# Splitting the data into testing and training sets

X = df.drop('Class', axis=1)

y = df['Class']



X_data = X.values

y_data = y.values



X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=2)



# Training the model

outlier_fraction = len(df[df.Class == 1])/len(df[df.Class == 0])



model_if = IsolationForest(max_samples=len(X_train), contamination=outlier_fraction, random_state=2)

model_if.fit(X_train)



scores_pred = model_if.decision_function(X_train)

y_pred = model_if.predict(X_test)
# Cleaning the predicted values for proper display of good and fraudulent transaction.



y_pred[y_pred == -1] = 1

y_pred[y_pred == 1] = 0



#number of errors by the model

print('Number of errors by the model: ', (y_pred != y_test).sum())
conf_matrix = confusion_matrix(y_test, y_pred)

Labels = ['Good', 'Fraud']

sns.heatmap(conf_matrix, xticklabels=Labels, yticklabels=Labels, annot=True, fmt='d')

plt.title('Confusion Matrix')

plt.xlabel('True Class')

plt.ylabel('Predicted Class')

plt.show()
# Evaluation of the model

print('Model Evaluation metrics - Isolation Forest')

print('Accuracy: ', accuracy_score(y_test, y_pred))

print('Precision: ', precision_score(y_test, y_pred))

print('F1: ', f1_score(y_test, y_pred))

print('Recall: ',recall_score(y_test, y_pred))

print('Matthews correlation coefficient', matthews_corrcoef(y_test, y_pred))
model_rf = RandomForestClassifier()

model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)



conf_matrix = confusion_matrix(y_test, y_pred)

Labels = ['Good', 'Fraud']

sns.heatmap(conf_matrix, xticklabels=Labels, yticklabels=Labels, annot=True, fmt='d')

plt.title('Confusion Matrix')

plt.xlabel('True Class')

plt.ylabel('Predicted Class')

plt.show()
# Evaluation of the Ranfom Forest model

print('Model Evaluation metrics')

print('Accuracy: ', accuracy_score(y_test, y_pred))

print('Precision: ', precision_score(y_test, y_pred))

print('F1: ', f1_score(y_test, y_pred))

print('Recall: ',recall_score(y_test, y_pred))

print('Matthews correlation coefficient', matthews_corrcoef(y_test, y_pred))
model_mlp = MLPClassifier()

model_mlp.fit(X_train, y_train)

y_pred = model_mlp.predict(X_test)



conf_matrix = confusion_matrix(y_test, y_pred)

Labels = ['Good', 'Fraud']

sns.heatmap(conf_matrix, xticklabels=Labels, yticklabels=Labels, annot=True, fmt='d')

plt.title('Confusion Matrix')

plt.xlabel('True Class')

plt.ylabel('Predicted Class')

plt.show()
# Evaluation of the Ranfom Forest model

print('Model Evaluation metrics')

print('Accuracy: ', accuracy_score(y_test, y_pred))

print('Precision: ', precision_score(y_test, y_pred))

print('F1: ', f1_score(y_test, y_pred))

print('Recall: ',recall_score(y_test, y_pred))

print('Matthews correlation coefficient', matthews_corrcoef(y_test, y_pred))
model_GB = GradientBoostingClassifier()

model_GB.fit(X_train, y_train)

y_pred = model_GB.predict(X_test)



conf_matrix = confusion_matrix(y_test, y_pred)

Labels = ['Good', 'Fraud']

sns.heatmap(conf_matrix, xticklabels=Labels, yticklabels=Labels, annot=True, fmt='d')

plt.title('Confusion Matrix')

plt.xlabel('True Class')

plt.ylabel('Predicted Class')

plt.show()
# Evaluation of the Ranfom Forest model

print('Model Evaluation metrics')

print('Accuracy: ', accuracy_score(y_test, y_pred))

print('Precision: ', precision_score(y_test, y_pred))

print('F1: ', f1_score(y_test, y_pred))

print('Recall: ',recall_score(y_test, y_pred))

print('Matthews correlation coefficient', matthews_corrcoef(y_test, y_pred))
import h2o

h2o.init()
df_cc = h2o.import_file('/kaggle/input/creditcardfraud/creditcard.csv')

df_cc.head()
train, test = df_cc.split_frame(ratios=[.8])
from h2o.automl import H2OAutoML



x = df_cc.col_names[:-1]



automl = H2OAutoML(project_name='cc_fraud_detect', 

                  max_models=5,

                  max_runtime_secs=500,

                  sort_metric='MAE',

                  exclude_algos=['StackedEnsemble'],

                  seed=111)



automl.train(training_frame=train, y='Class', x=x)
automl.leaderboard
automl.leader
automl.predict(test)