import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from collections import Counter
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
import os

%matplotlib inline
sns.set_style('whitegrid')
import os
print('Data File ==>\t {}'.format(os.listdir("../input")[0]))
creditcard = pd.read_csv("../input/creditcard.csv")
creditcard.head(3)
# Missing values
print('No of Missing values :\t{}'.format(creditcard.isnull().sum().max()))
sns.countplot(data=creditcard,x = 'Class')
plt.title('Class Variables distribution', fontsize=14)
plt.show()

creditcard['Class'].value_counts() *100 /len(creditcard)
fig,ax = plt.subplots(nrows = 7, ncols=4, figsize=(12,21))
row = 0
col = 0
for i in range(len(creditcard.columns) -3):
    if col > 3:
        row += 1
        col = 0
    axes = ax[row,col]
    sns.boxplot(x = creditcard['Class'], y = creditcard[creditcard.columns[i +1]],ax = axes)
    col += 1
plt.tight_layout()
plt.show()
creditcard.drop(['Time','Amount'],axis = 1,inplace=True)
X = creditcard.iloc[:,range(0,28)].values
y = creditcard['Class'].values
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X = sel.fit_transform(X)
print('Remaining Features Count:\t{}'.format(X.shape[1]))
def print_result(title,actual, prediction,decision):
    print('****************************************************')
    print(title)
    print('****************************************************')
    print('Accuracy Score :\t\t{:.3}'.format(metrics.accuracy_score(actual, prediction)))
    print('Recall Score :\t\t\t{:.3}'.format(metrics.recall_score(actual, prediction)))
    print('Average Precision Score :\t{:.3}'.format(metrics.average_precision_score(actual, decision)))
    print('ROC AUC Score :\t\t\t{:.3}'.format(metrics.roc_auc_score(actual, decision)))
    print()
# split data in training set and test set
X_train, X_test, y_train, y_test = train_test_split(\
    X,y,test_size=0.3, random_state = 0)
# Data Distribution 
print('Normal Data Distribution {}'.format(Counter(creditcard['Class'])))
C_VALUES = [0.001,0.01,0.1,1]
# build normal Model
for c_value in C_VALUES:
    pipeline = make_pipeline(LogisticRegression(random_state=42, C = c_value))
    model = pipeline.fit(X_train,y_train)
    prediction = model.predict(X_test)
    decision = model.decision_function(X_test)
    # print(metrics.confusion_matrix(y_test,prediction))
    print_result('Normal Data Logistic -> C ={}'.\
                 format(c_value), y_test,prediction,decision)
X_SMOTE,y_SMOTE = SMOTE().fit_sample(X,y)
print('SMOTE Data Distribution {}'.format(Counter(y_SMOTE)))

C_VALUES = [0.001,0.01,0.1,1]
# build normal Model
for c_value in C_VALUES:
    smote_pipeline = make_pipeline_imb(SMOTE(random_state=42),\
                                       LogisticRegression(random_state=42, C = c_value))
    smote_model = smote_pipeline.fit(X_train,y_train)
    smote_prediction = smote_model.predict(X_test)
    smote_decision = smote_model.decision_function(X_test)
    # print(metrics.confusion_matrix(y_test,smote_prediction))
    print_result('SMOTE - Oversampling data(Logistic) -> C ={}'.\
                 format(c_value), y_test,smote_prediction,smote_decision)
X_NearMiss,y_NearMiss = NearMiss().fit_sample(X,y)
print('NearMiss Data Distribution {}'.format(Counter(y_NearMiss)))
# build moodel with  - undersampling
C_VALUES = [0.001,0.01,0.1,1]
# build normal Model
for c_value in C_VALUES:
    nearmiss =  LogisticRegression(random_state=42,C = c_value)
    nearmiss_model = nearmiss.fit(X_NearMiss,y_NearMiss)
    nearmiss_decision = nearmiss_model.decision_function(X_test)
    nearmiss_prediction = nearmiss_model.predict(X_test)
    print_result('NearMiss - Undersampling data(Logistic) -> C ={}'.\
                 format(c_value), y_test,nearmiss_prediction,nearmiss_decision)