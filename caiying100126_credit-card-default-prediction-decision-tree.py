import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
%matplotlib inline
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
num_trees = 100
from sklearn.model_selection import cross_val_score
from sklearn import metrics
# Using Decision Tree with 5 fold cross validation
# Restrict max_depth to 1 to have more impure leaves
clf_DT = DecisionTreeClassifier(max_depth=1, random_state=2017).fit(X_train,y_train)
results = cross_val_score(clf_DT, X_train,y_train, cv=kfold)
print ("Decision Tree (stand alone) - CV Train : %.2f" % results.mean())
print ("Decision Tree (stand alone) - Train : %.2f" % metrics.accuracy_score(clf_DT.predict(X_train), y_train))
print ("Decision Tree (stand alone) - Test : %.2f" % metrics.accuracy_score(clf_DT.predict(X_test), y_test))
# Get unnormalized confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

y_pred_0 = clf_DT.predict(X_test)
cm = confusion_matrix(y_test, y_pred_0)
print("Absolute confusion matrix is\n",cm)
print(classification_report(y_test, y_pred_0))
# plot normalized confusion matrix
def plot_confusion_matrix(y_test, y_pred_0, classes,
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
    cm = confusion_matrix(y_test, y_pred_0)

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

plot_confusion_matrix(y_test, y_pred_0, classes=["0","1"], normalize=True)
# Using Adaptive Boosting of 100 iteration
clf_DT_Boost = AdaBoostClassifier(base_estimator=clf_DT, n_estimators=num_trees, learning_rate=0.1, random_state=2017).fit(X_train,y_train)
results = cross_val_score(clf_DT_Boost, X_train, y_train, cv=kfold)
print ("\nDecision Tree (AdaBoosting) - CV Train : %.2f" % results.mean())
print ("Decision Tree (AdaBoosting) - Train : %.2f" % metrics.accuracy_score(clf_DT_Boost.predict(X_train), y_train))
print ("Decision Tree (AdaBoosting) - Test : %.2f" % metrics.accuracy_score(clf_DT_Boost.predict(X_test), y_test))
# Get unnormalized confusion matrix
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

y_pred = clf_DT_Boost.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Absolute confusion matrix is\n",cm)
print(classification_report(y_test, y_pred))
# plot normalized confusion matrix
def plot_confusion_matrix(y_test, y_pred, classes,
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
    cm = confusion_matrix(y_test, y_pred)

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

plot_confusion_matrix(y_test, y_pred, classes=["0","1"], normalize=True)
