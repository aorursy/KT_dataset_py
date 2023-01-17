import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.model_selection import train_test_split
%matplotlib inline

from sklearn import metrics
from sklearn.model_selection import cross_val_score

from xgboost.sklearn import XGBClassifier
from sklearn import preprocessing 
df=pd.read_csv('pre_processed_data_v2.csv',index_col=0)
df.head()
# based on correlation, drop columns with 90% with another column
df1 = df.drop(columns=['BILL_AMT4_rt', 
                       'BILL_AMT6_rt',
                       'BILL_AMT1_sq',
                       'BILL_AMT3_rt',
                       'BILL_AMT1_rt',
                       'BILL_AMT5_rt',
                       'BILL_AMT4_sq',
                       'BILL_AMT6_sq',
                       'BILL_AMT4_rt',
                       'BILL_AMT6_rt',
                       'BILL_AMT2_rt',
                       'BILL_AMT3_rt',
                       'BILL_AMT2',
                       'BILL_AMT3',
                       'BILL_AMT4',
                       'BILL_AMT5',
                       'BILL_AMT6',
                       'May_leftover_credit',
                       'LIMIT_BAL_sq',
                       'Aug_leftover_credit',
                       'LIMIT_BAL_rt',
                       'LIMIT_BAL',
                       'avg_bill',
                       'PAY_AMT2_sq',
                       'PAY_6',
                       'sum_pay',
                       'sum_exp'
                       ])
df1.head()
# Normalize some features to [0,1] before KNN

#scaled_values = scaler.fit_transform(df1) 
df1[['AGE', 'BILL_AMT1','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
     'PAY_AMT5','PAY_AMT6','BILL_AMT1_log','BILL_AMT2_sq','BILL_AMT2_log','BILL_AMT3_sq',
     'BILL_AMT3_log','BILL_AMT4_log','BILL_AMT5_sq','BILL_AMT5_log','BILL_AMT6_log',
     'PAY_AMT1_sq','PAY_AMT1_rt','PAY_AMT1_log','PAY_AMT2_rt','PAY_AMT2_log',
     'PAY_AMT3_sq','PAY_AMT3_rt','PAY_AMT3_log','PAY_AMT4_sq','PAY_AMT4_rt',
     'PAY_AMT4_log','PAY_AMT5_sq','PAY_AMT5_rt','PAY_AMT5_log','PAY_AMT6_sq',
     'PAY_AMT6_rt','PAY_AMT6_log','LIMIT_BAL_log','Sep_abs_expense',
     'Aug_abs_expense','Jul_abs_expense','Jun_abs_expense','May_abs_expense',
     'avg_abs_diff','avg_pay','Sep_pay_over_bill','Aug_pay_over_bill',
     'Jul_pay_over_bill','Jun_pay_over_bill','May_pay_over_bill',
     'Apr_pay_over_bill','pay_amount_var','PCAbill statements',
     'PCApay amount','PCAexpenses','PCAscaled expenses']]= MinMaxScaler().fit_transform(df1[['AGE', 'BILL_AMT1','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4',
     'PAY_AMT5','PAY_AMT6','BILL_AMT1_log','BILL_AMT2_sq','BILL_AMT2_log','BILL_AMT3_sq',
     'BILL_AMT3_log','BILL_AMT4_log','BILL_AMT5_sq','BILL_AMT5_log','BILL_AMT6_log',
     'PAY_AMT1_sq','PAY_AMT1_rt','PAY_AMT1_log','PAY_AMT2_rt','PAY_AMT2_log',
     'PAY_AMT3_sq','PAY_AMT3_rt','PAY_AMT3_log','PAY_AMT4_sq','PAY_AMT4_rt',
     'PAY_AMT4_log','PAY_AMT5_sq','PAY_AMT5_rt','PAY_AMT5_log','PAY_AMT6_sq',
     'PAY_AMT6_rt','PAY_AMT6_log','LIMIT_BAL_log','Sep_abs_expense',
     'Aug_abs_expense','Jul_abs_expense','Jun_abs_expense','May_abs_expense',
     'avg_abs_diff','avg_pay','Sep_pay_over_bill','Aug_pay_over_bill',
     'Jul_pay_over_bill','Jun_pay_over_bill','May_pay_over_bill',
     'Apr_pay_over_bill','pay_amount_var','PCAbill statements',
     'PCApay amount','PCAexpenses','PCAscaled expenses']])

# print the normalized dataframe
df1.head()
# Split the dataframe to X and Y
df2 = np.split(df1, [71], axis=1)
# X_df-->X features
X_df = df2[0] 

#Y_df -->Y feature(def_payment_nm)
Y_df = df2[1]
X_df.head()
Y_df.head()
# Fix Nan and Infinity values in X features, fill in NaN with mean
X_df[X_df==np.inf]=np.nan
X_df.fillna(X_df.mean(), inplace=True)
X_df.shape
# Get X and Y array
X=X_df.values
y=Y_df.values

#Create a train-test split in your data using the SKLearn Train-Test split library
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
kfold = StratifiedKFold(n_splits=5, random_state=2017)
num_rounds = 100
clf_XGB = XGBClassifier(n_estimators = num_rounds, objective= 'binary:logistic',seed=2017)
# Use early_stopping_rounds to stop the cv when there is no score imporovement
clf_XGB.fit(X_train,y_train, early_stopping_rounds=20, eval_set=[(X_test, y_test)], verbose=False)
results = cross_val_score(clf_XGB, X_train,y_train, cv=kfold)
print ("\nxgBoost - CV Train : %.2f" % results.mean())
print ("xgBoost - Train : %.2f" % metrics.accuracy_score(clf_XGB.predict(X_train), y_train))
print ("xgBoost - Test : %.2f" % metrics.accuracy_score(clf_XGB.predict(X_test), y_test))
# Get unnormalized confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

predict_test = clf_XGB.predict(X_test)
cm = confusion_matrix(y_test, predict_test)
print("Absolute confusion matrix is\n",cm)
from sklearn.metrics import classification_report
print(classification_report(y_test, predict_test))
# plot normalized confusion matrix
def plot_confusion_matrix(y_test, predict_test, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_test, predict_test)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="red" if cm[i, j] > thresh else "green")
    fig.tight_layout()
    return ax

plot_confusion_matrix(y_test, predict_test, classes=["0","1"], normalize=True)
