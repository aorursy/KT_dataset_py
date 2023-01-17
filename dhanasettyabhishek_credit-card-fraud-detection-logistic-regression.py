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
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict

from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import IsolationForest
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

print("Dimensions:",data.shape)

data.head()
data.describe()
def checkNull(data):

    return data.isnull().any()

def checkDatatype(data):

    return data.dtypes

print(checkNull(data))

print("=================")

print(checkDatatype(data))
# No null values and no object datatype.

def fraudDetection(data):

    fraud, not_fraud = 0, 0

    for i in data['Class']:

        if i == 0:

            fraud += 1

        else:

            not_fraud += 1

    return (fraud*100)/data.shape[0], fraud, (not_fraud*100)/data.shape[0], not_fraud

fraud_percentage, fraud, notfraud_detection, not_fraud = fraudDetection(data)

sns.countplot('Class', data=data)

plt.title("Non fraud transaction VS fraud transaction")

plt.xlabel("0->Not Fraud, 1-> Fraud")
# From the description, we can see that Time and Amount columns are not skewed properly.

# Plotting a distribution plot.

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

sns.distplot(data['Amount'].values, ax = ax[0])

ax[0].set_title("Distribution plot of Amounts")

sns.distplot(data['Time'].values, ax = ax[1])

ax[1].set_title("Distribution plot of Time")
# We need to normalize Amount and Time 



robust_scaler = RobustScaler()

data['normalized_amount'] = robust_scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

data['normalized_time'] = robust_scaler.fit_transform(data['Time'].values.reshape(-1, 1))

new_data = data.drop(['Time', 'Amount'], axis=1)

new_data.head()
# Train test split



y = data['Class'].values

new_data_no_class = new_data.drop(['Class'], axis=1)

X = new_data_no_class.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=9)
# Stratified split to equate fraud and non-fraud transactions



startified_k_fold = StratifiedKFold(n_splits=9, random_state=9)

for i, j in startified_k_fold.split(X, y):

    X_train_skf, X_test_skf = new_data.drop('Class', axis=1).iloc[i], new_data.drop('Class', axis=1).iloc[j]

    y_train_skf, y_test_skf = new_data['Class'].iloc[i], new_data['Class'].iloc[j]

X_train_skf
fraud_data = new_data.loc[new_data['Class']==1]

non_fraud_data = new_data.loc[new_data['Class']==0][:fraud_data.shape[0]]

data_distributed = pd.concat([fraud_data, non_fraud_data])

data_distributed = data_distributed.sample(frac = 1, random_state = 9)

sns.countplot('Class', data=data_distributed)

plt.title("Distribution")

data_distributed.describe()
data_X = data_distributed.drop('Class', axis=1)

data_y = data_distributed['Class']



X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.33, random_state=9)

X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.values, y_test.values
# Logistic Regression



model = LogisticRegression()

model.fit(X_train, y_train)

score = cross_val_score(model, X_train, y_train, cv=5)

cv_predict = cross_val_predict(model, X_train, y_train, cv=5, method="decision_function")



print("Training score:",round(score.mean()*100, 2),"%")

print("ROC score:",round(roc_auc_score(y_train, cv_predict)*100, 2),"%")



y_pred = model.predict(X_test)

accuracy = model.score(X_test, y_test)*100



f1_score_ = f1_score(y_test, y_pred)*100

print("Confusion Matrix of the dataset:")

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), ['Actual No','Actual Yes'],['Pred No','Pred Yes'])

sns.set(font_scale=1.4)

sns.heatmap(df_cm, annot=True, cmap="coolwarm", fmt='d')

plt.show()

print(f'Accuracy percetage of the the dataset: {round(accuracy, 2)}%')

print(f'F1-Score percetage of the the dataset: {round(f1_score_, 2)}%')
