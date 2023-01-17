from sklearn import linear_model as lm

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import scipy.stats as sts

import pandas as pd

import scipy as scp

import numpy as np

import sklearn.preprocessing as preproc

from sklearn.model_selection import train_test_split  ### for train and test split package

from sklearn import metrics  ## For calculation of MSE & RMSE

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, f1_score
bank = pd.read_csv("../input/QualitativeBankruptcy.csv")
bank.head()
bank.shape   ## It display the no of rows and column
## It gives the unique label of each column



for col in bank:

    print (col)

    print (bank[col].unique())
bank.describe()  ##It gives the summary of the DataFrame
var = ['indRisk','mgtRisk','finFlexibility','Credibility','Competitiveness', 'OperatingRisk', 'bclass']

var
def func_labelEncoder(var,features):

    encode= LabelEncoder()

    features[var] = encode.fit_transform(features[var].astype(str))

    

for i in var:

    func_labelEncoder(i,bank)
bank.head()
bank.describe()
bank.isnull().any()
sns.pairplot(bank)
bank.corr()
xVal = bank.drop(['bclass'], axis=1)
xVal.head()
yVal = bank.bclass.values.reshape(-1,1)
yVal.shape
X_train, X_test, Y_train, Y_test = train_test_split(xVal,yVal, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)

print("X_test shape:", X_test.shape)

print("Y_train shape:", Y_train.shape)

print("X_test shape:", Y_test.shape)
### 1. Logistic Regression Model



lmod = lm.LogisticRegression(penalty='l2',fit_intercept=True,max_iter=500,solver='lbfgs',tol=0.0001,multi_class='ovr')

lrmod = lmod.fit(X_train,Y_train.ravel())
lrmod.intercept_  ### Intercapt (B0)
lrmod.coef_   ### Coefficients (B1, B2...)
predicted_data = lrmod.predict(X_test)  ### Predicting the  model for independent test data
predicted_data
confusion_matrix(Y_test, predicted_data)
from sklearn import metrics as accuracyMatrics
accuracyMatrics.accuracy_score(Y_test, predicted_data)  ## Predicting accuracy score
prec = accuracyMatrics.precision_score(Y_test, predicted_data)  ## Precision score

prec
recall = accuracyMatrics.recall_score(Y_test, predicted_data)  ## Recall score

recall
probPred = lrmod.predict_proba(X_test)

predictProbAdmit = probPred[:,1]
### ROC curve calculation



fpr, tpr, threshold = accuracyMatrics.roc_curve(Y_test,predictProbAdmit)
auc_val = accuracyMatrics.auc(fpr,tpr)

auc_val   ### AUC Value
threshold
## Plotting ROC Curve



plt.plot(fpr,tpr,linewidth=2, color='g',label='ROC curve (area = %0.2f)' % auc_val)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

## F1 score 



F1_score = f1_score(Y_test, predicted_data)

F1_score