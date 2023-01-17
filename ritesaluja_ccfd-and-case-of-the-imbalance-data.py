# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score,  auc, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings("ignore")

# Any results you write to the current directory are saved as output.
#Gotta Detect'em All

df = pd.read_csv("../input/creditcard.csv")

df.head()
fig = plt.figure(figsize=(12,12))

plt.subplot(311)

plt.title("All (Amount)")

plt.plot(df['Amount']);

plt.subplot(312)

plt.title("Non Fraudulent")

plt.plot(df[df['Class']==0]['Amount']);

plt.subplot(313)

plt.title("Fraudulent")

plt.plot(df[df['Class']==1]['Amount']);
#Amount correlation with Fraud

df["Amount"].corr(df["Class"])
#cluster transactions ? (time?)

df.isna().sum() #no missing data

df.dtypes #clean

#How imbalanced is my Data?

sns.countplot("Class",data=df);
#split

x = np.array(df.iloc[:,1:28])

y= np.array(df["Class"])

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.33, random_state = 42)

lr = LogisticRegression(penalty = 'l1',C=0.01,class_weight= 'balanced' )



"""

#ran once

param_grid = { 

    'penalty': ['l1','l2'], 'C': [0.01,0.1,1,10,100], 'class_weight' : ['balanced']

}



CV_lr = GridSearchCV(estimator=lr, param_grid=param_grid, cv= 5)

CV_lr.fit(xTrain, yTrain)

print(CV_lr.best_params_)

"""
lr.fit(xTrain, yTrain)

yPred = lr.predict(xTest)

print("Accuracy of Prediction is: {} %".format(round(accuracy_score(yTest, yPred)*100,2)))
#Lets look at Recall - how many of the Actual Positives does our model capture

print("Recall: ",recall_score(yTest, yPred))

print("\nPrecision Score: ",precision_score(yTest, yPred))

print("\nF1 Score: ",f1_score(yTest, yPred))
#Using SMOTE

sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(xTrain, yTrain.ravel())
"""

param_grid = { 

    'penalty': ['l1','l2'], 'C': [0.01,0.1,1,10,100]

}



CV_lr = GridSearchCV(estimator=lr, param_grid=param_grid, cv= 5)

CV_lr.fit(X_train_res, y_train_res)

print(CV_lr.best_params_)

"""
lrs = LogisticRegression(penalty = 'l2',C=1)

lrs.fit(X_train_res, y_train_res)

yPred = lrs.predict(xTest)



#Evaluating

print("Accuracy of Prediction is: {} %".format(round(accuracy_score(yTest, yPred)*100,2)))

print("Recall: ",recall_score(yTest, yPred))

print("\nPrecision Score: ",precision_score(yTest, yPred))

print("\nF1 Score: ",f1_score(yTest, yPred))
y_pred_sample_score = lrs.decision_function(xTest)





fpr, tpr, thresholds = roc_curve(yTest, y_pred_sample_score)



roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()