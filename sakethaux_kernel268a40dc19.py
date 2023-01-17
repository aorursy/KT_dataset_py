import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pickle as pkl

from sklearn.preprocessing import StandardScaler,MinMaxScaler
df = pd.read_csv("/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv")

df.columns = [x.strip().lower() for x in df.columns.values]

df.head()
df.columns.values
def preprocess(df):

    df['mean of the integrated profile'] = df['mean of the integrated profile'].apply(lambda x : 25 if x<=25 else x)

    df['standard deviation of the integrated profile'] = df['standard deviation of the integrated profile'].apply(lambda x : 75 if x>=75 else x)

    df['excess kurtosis of the integrated profile'] = df['excess kurtosis of the integrated profile'].apply(lambda x : 2 if x>=2 else x)

    df['mean of the dm-snr curve'] = df['mean of the dm-snr curve'].apply(lambda x : 12 if x>=12 else x)

    df['standard deviation of the dm-snr curve'] = df['standard deviation of the dm-snr curve'].apply(lambda x : 70 if x>=70 else x)

    df['excess kurtosis of the dm-snr curve'] = df['excess kurtosis of the dm-snr curve'].apply(lambda x : 25 if x>=25 else x)

    df['skewness of the dm-snr curve'] = df['skewness of the dm-snr curve'].apply(lambda x : 450 if x>=450 else x)

    return df



def scaling(df,scaler=None):

    if scaler==None:

        sc = StandardScaler()

        sc.fit(df)

        df = sc.transform(df)

        pkl.dump(sc,open("pulsar_scaler.pkl",'wb'))

    else:

        df = scaler.transform(df)

    return df
corr = df.corr()

corr
y = df['target_class']

X = df.drop(columns=['target_class'])
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
# finding outliers and handling them

X_train.boxplot(column=['mean of the integrated profile'])
X_train.boxplot(column=['standard deviation of the integrated profile'])
X_train.boxplot(column=['excess kurtosis of the integrated profile'])
X_train.boxplot(column=['mean of the dm-snr curve'])
X_train.boxplot(column=['standard deviation of the dm-snr curve'])
X_train.boxplot(column=['excess kurtosis of the dm-snr curve'])
X_train.boxplot(column=['skewness of the dm-snr curve'])
X_train = preprocess(X_train)
X_train = scaling(X_train)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)
X_test = preprocess(X_test)
X_test = scaling(X_test,pkl.load(open("pulsar_scaler.pkl",'rb')))
y_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix, f1_score

confusion_matrix(y_test,y_pred)
f1_score(y_test,y_pred)
import statsmodels.api as sm

logit_model=sm.Logit(y_train,X_train)

result=logit_model.fit()

print(result.summary2())
## ROC Curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, lr.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, lr.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()