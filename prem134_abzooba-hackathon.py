# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/Training Data.csv')

test = pd.read_csv('../input/Test Data.csv')
gender_adherence = train.groupby(['Gender', 'Adherence'])['Adherence'].count().unstack()
gender_adherence = pd.DataFrame(gender_adherence)
gender_adherence
gender_adherence.plot(kind='bar', stacked=True)
train.isna().sum()
df = train

df_test = test
y = pd.get_dummies(df.Gender, prefix='Gender')

y1 = pd.get_dummies(df_test.Gender, prefix='Gender')
df = pd.merge(df, y, left_index=True, right_index=True)

df = df.drop(['Gender'],axis=1)



df_test = pd.merge(df_test, y1, left_index=True, right_index=True)

df_test = df_test.drop(['Gender'],axis=1)
from sklearn.preprocessing import LabelEncoder

df.Adherence = LabelEncoder().fit_transform(df.Adherence)
Adherence = df.Adherence

df = df.drop(['Adherence'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, Adherence, test_size=0.33)
from sklearn.ensemble import RandomForestClassifier



# Create the model with 100 trees

model = RandomForestClassifier(n_estimators=100)

# Fit on training data

model.fit(X_train, y_train)
# Actual class predictions

rf_predictions = model.predict(X_test)

# Probabilities for each class

rf_probs = model.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

# Calculate roc auc

auc = roc_auc_score(y_test, rf_probs)
print('AUC: %.2f' % auc)
def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)

plot_roc_curve(fpr, tpr)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

# confusion matrix

matrix = confusion_matrix(y_test,rf_predictions, labels=[1,0])

print('Confusion matrix : \n',matrix)



# outcome values order in sklearn

tp, fn, fp, tn = confusion_matrix(y_test,rf_predictions,labels=[1,0]).reshape(-1)

print('Outcome values : \n', tp, fn, fp, tn)



# classification report for precision, recall f1-score and accuracy

matrix = classification_report(y_test,rf_predictions,labels=[1,0])

print('Classification report : \n',matrix)
# Actual class predictions

rf_predictions = model.predict(df_test)

Adherence = ['No' if i==0 else 'Yes' for i in rf_predictions]

# Probabilities for each class

prob_0 = model.predict_proba(df_test)[:,0]

prob_1 = model.predict_proba(df_test)[:,1]
data = {'id':df_test.patient_id, 'Adherence':Adherence, 'probabilities_Yes':prob_1, 'probabilities_No':prob_0}
df = pd.DataFrame(data=data)

df.to_csv('output files.csv')