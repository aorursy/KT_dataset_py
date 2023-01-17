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
import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix,auc,roc_auc_score, classification_report

%matplotlib inline
df= pd.read_csv("../input/creditcardfraud/creditcard.csv")

df.head()
df.describe()
df.info()
sns.set_style("darkgrid")

plt.figure(figsize=(8, 6))

sns.countplot(x= 'Class', data= df)

Labels= ('Genuine', 'Fraud')

plt.xticks(range(2), Labels)
df['Class'].value_counts()
#plot Time to see if there is any trend



print("Time in between the transactions: ")

df["Time_hr"] = df["Time"]/3600 # convert to hours

print(df["Time_hr"].head())

fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))

ax1.hist(df.Time_hr[df.Class==0],bins=50,color='g',alpha=0.7)

ax1.set_title('Genuine')

plt.xlabel('Time (hrs)')

plt.ylabel('# transactions')

ax2.hist(df.Time_hr[df.Class==1],bins=50,color='r',alpha=0.7)

ax2.set_title('Fraud')

plt.xlabel('Time (hrs)')

plt.ylabel('# transactions')
df = df.drop(['Time'],axis=1)

df.head()
cols = df.columns.tolist()

cols.insert(0, cols.pop(cols.index('Time_hr')))

df = df.reindex(columns= cols)

df.head()
fig, (ax3,ax4) = plt.subplots(2,1, figsize = (6,3), sharex = True)

ax3.hist(df.Amount[df.Class==0],bins=50,color='g',alpha=0.7)

ax3.set_yscale('log') # to see the tails

ax3.set_title('Genuine') # to see the tails

ax3.set_ylabel('No. of transactions')

ax4.hist(df.Amount[df.Class==1],bins=50,color='r',alpha=0.7)

ax4.set_yscale('log') # to see the tails

ax4.set_title('Fraud') # to see the tails

ax4.set_xlabel('Amount (USD)')

ax4.set_ylabel('No. of transactions')
from sklearn.preprocessing import StandardScaler

df['scaled_Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))

df['scaled_time'] = StandardScaler().fit_transform(df['Time_hr'].values.reshape(-1,1))

df = df.drop(['Amount', 'Time_hr'],axis=1)

df.head()
scaled_amount = df['scaled_Amount']

scaled_time = df['scaled_time']



df.drop(['scaled_Amount', 'scaled_time'], axis=1, inplace=True)

df.insert(0, 'scaled_Amount', scaled_amount)

df.insert(1, 'scaled_time', scaled_time)



df.head()
correlation= df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(correlation, cmap="magma", linecolor='white',linewidths=1)

def train_test_split(df, dropped_columns):

    df = df.drop(dropped_columns,axis=1)

    print(df.columns)

    

    from sklearn.model_selection import train_test_split

    

    y = df['Class']  #Labels

    X = df.drop(['Class'],axis= 1) #Variables

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state= 101)



    print("train set size: ", len(y_train), "\n test set size: ", len(y_test))

    print("fraud transactions in test set: ", sum(y_test))

    return X_train, X_test, y_train, y_test
def model_predictions(classifier, X_train, y_train, X_test):

    # create classifier

    classifier = classifier

    

    # fit it to training data

    classifier.fit(X_train,y_train)

    

    # predict using test data

    y_pred = classifier.predict(X_test)

    

    # Compute predicted probabilities: y_pred_prob

    y_pred_prob = classifier.predict_proba(X_test)

    

    return y_pred, y_pred_prob
def print_metrics(y_test,y_pred,y_pred_prob):

    print('test-set confusion matrix:\n', confusion_matrix(y_test,y_pred)) 

    print('Classification Report:\n', classification_report(y_test, y_pred))

    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_prob[:,1])))
from sklearn.naive_bayes import GaussianNB

dropped_columns = []

X_train, X_test, y_train, y_test = train_test_split(df, dropped_columns)

y_pred, y_pred_prob = model_predictions(GaussianNB(), X_train, y_train, X_test)

print_metrics(y_test,y_pred,y_pred_prob)
from xgboost import XGBClassifier

dropped_columns = []

X_train, X_test, y_train, y_test = train_test_split(df, dropped_columns)

y_pred, y_pred_prob = model_predictions(XGBClassifier(), X_train, y_train, X_test)

print_metrics(y_test,y_pred,y_pred_prob)
from sklearn.linear_model import LogisticRegression

dropped_columns = []

X_train, X_test, y_train, y_test = train_test_split(df, dropped_columns)

y_pred, y_pred_prob = model_predictions(LogisticRegression(C = 0.01), X_train, y_train, X_test)

print_metrics(y_test,y_pred,y_pred_prob)
dropped_columns = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V10','V9','V8']

X_train, X_test, y_train, y_test = train_test_split(df, dropped_columns)

y_pred, y_pred_prob = model_predictions(GaussianNB(), X_train, y_train, X_test)

print_metrics(y_test,y_pred,y_pred_prob)
dropped_columns = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V10','V9','V8']

X_train, X_test, y_train, y_test = train_test_split(df, dropped_columns)

y_pred, y_pred_prob = model_predictions(XGBClassifier(), X_train, y_train, X_test)

print_metrics(y_test,y_pred,y_pred_prob)
dropped_columns = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V10','V9','V8']

X_train, X_test, y_train, y_test = train_test_split(df, dropped_columns)

y_pred, y_pred_prob = model_predictions(LogisticRegression(C = 0.01), X_train, y_train, X_test)

print_metrics(y_test,y_pred,y_pred_prob)
fraud_class_len = len(df[df['Class'] == 1])

print(fraud_class_len)



genuine_indices = df[df['Class'] == 0].index

print(genuine_indices)



# taking random 492 samples from the genuine class

random_genuine_samples = np.random.choice(genuine_indices, fraud_class_len, replace=False)

print(len(random_genuine_samples))



fraud_indices = df[df['Class'] == 1].index

print(fraud_indices)

undersample_indices = np.concatenate([random_genuine_samples,fraud_indices])

undersample_df = df.loc[undersample_indices]

undersample_df.head()
plt.figure(figsize=(8, 6))

sns.countplot(x= 'Class', data= undersample_df)

Labels= ('Genuine', 'Fraud')

plt.xticks(range(2), Labels)
correlation_undersample_df= undersample_df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(correlation_undersample_df, cmap="magma", linecolor='white',linewidths=1)
dropped_columns = []

new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(undersample_df, dropped_columns)

new_y_pred, new_y_pred_prob = model_predictions(GaussianNB(), new_X_train, new_y_train, new_X_test)

print_metrics(new_y_test,new_y_pred,new_y_pred_prob)
dropped_columns = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V10','V9','V8']

new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(undersample_df, dropped_columns)

new_y_pred, new_y_pred_prob = model_predictions(GaussianNB(), new_X_train, new_y_train, new_X_test)

print_metrics(new_y_test,new_y_pred,new_y_pred_prob)
dropped_columns = []

new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(undersample_df, dropped_columns)

new_y_pred, new_y_pred_prob = model_predictions(XGBClassifier(), new_X_train, new_y_train, new_X_test)

print_metrics(new_y_test,new_y_pred,new_y_pred_prob)
dropped_columns = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V10','V9','V8']

new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(undersample_df, dropped_columns)

new_y_pred, new_y_pred_prob = model_predictions(XGBClassifier(), new_X_train, new_y_train, new_X_test)

print_metrics(new_y_test,new_y_pred,new_y_pred_prob)
dropped_columns = []

new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(undersample_df, dropped_columns)

new_y_pred, new_y_pred_prob = model_predictions(LogisticRegression(C = 0.01), new_X_train, new_y_train, new_X_test)

print_metrics(new_y_test,new_y_pred,new_y_pred_prob)
dropped_columns = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V10','V9','V8']

new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(undersample_df, dropped_columns)

new_y_pred, new_y_pred_prob = model_predictions(LogisticRegression(C = 0.01), new_X_train, new_y_train, new_X_test)

print_metrics(new_y_test,new_y_pred,new_y_pred_prob)