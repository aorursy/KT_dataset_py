import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from imblearn.under_sampling import (RandomUnderSampler)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score, KFold,cross_val_predict

from sklearn.metrics import auc,roc_auc_score,roc_curve,recall_score

from sklearn.model_selection import StratifiedKFold

from sklearn import metrics

from sklearn.model_selection import GridSearchCV
data = pd.read_csv("../input/creditcard.csv")

data.head()
data_fraud=data.loc[data["Class"] ==1]

data_Nonfraud=data.loc[data["Class"] ==0]

print("fraud dataset Shape: {} and non-fraud dataset Shape {}, ratio :{}".format(data_fraud.shape, data_Nonfraud.shape,(data_fraud.shape[0]/data.shape[0])*100))
count_classes =pd.value_counts(data['Class'])

count_classes.plot(kind="bar") # OR count_class.plot.bar()

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")

data["Class"].value_counts()
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))

data = data.drop(['Time','Amount'],axis=1)

data.head()
X = data.drop('Class', axis=1)

y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y,

                                                    stratify=y, 

                                                    test_size=0.20)
sampler = RandomUnderSampler(sampling_strategy=1)

X_train_balanced, y_train_balanced=sampler.fit_resample(X_train,y_train)
y_unRe_labels,y_unRe_counts_label = np.unique(y_train_balanced, return_counts=True)

y_unRe_labels,y_unRe_counts_label
#CONVERTING DATAFRAMES TO ARRAYS FOR TRAINING A CLASSIFICATION MODEL

X_train_balanced=X_train_balanced.values

X_test=X_test.values

y_test=y_test.values

y_train_balanced=y_train_balanced.values

y_train_balanced=y_train_balanced.ravel()

y_test=y_test.ravel()
lr=LogisticRegression()
c_param_range = [0.01,0.1,1,10,100]

dual=[True,False]

param_grid = dict(dual=dual, C=

                 c_param_range)
grid = GridSearchCV(estimator=lr, param_grid=param_grid, n_jobs=-1,scoring="roc_auc")
grid_result = grid.fit(X_train_balanced, y_train_balanced)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
lr_pred = cross_val_predict(lr.set_params(C=0.01), X_train_balanced, y_train_balanced, cv=5,

                             method="decision_function")
print('Logistic Regression Classifier: ', roc_auc_score(y_train_balanced, lr_pred))
lr_fpr, lr_tpr, lr_threshold = roc_curve(y_train_balanced, lr_pred)
def graph_roc_curve_multiple( lr_fpr, lr_tpr):

    plt.figure(figsize=(16,8))

    plt.title('ROC Curve \n Top Classifier', fontsize=18)

    plt.plot(lr_fpr, lr_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train_balanced, lr_pred)))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.01, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),

                arrowprops=dict(facecolor='#6E726D', shrink=0.05),

                )

    plt.legend()

    

graph_roc_curve_multiple(lr_fpr, lr_tpr)

plt.show()
final_Model=lr.set_params(C=0.01).fit(X_train_balanced,y_train_balanced)
final_Model.score(X_test,y_test)