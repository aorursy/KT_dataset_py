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



#Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns









df = pd.read_csv("/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv",sep=",")
df.head()
df.tail()
#Dataset Information 

df.info()
# Description of the Dataset

df.describe().T
#Checking for Null Values 

df.isnull().sum()
df.columns
#Checking the Balance in the diagnosis column 

df.diagnosis.value_counts()
#Representing this in the form of a countplot 

sns.countplot(df.diagnosis)

plt.show()
#Visualization :- PAIRPLOT

sns.pairplot(df)

plt.show()
#Importing the library

from statsmodels.tools import add_constant

import statsmodels.api as sm
df_logit = add_constant(df)
df_logit.head()
logitmodel=sm.Logit(endog=df_logit.diagnosis,exog=df_logit.drop('diagnosis',axis=1)).fit()
print(logitmodel.summary())
para=np.exp(logitmodel.params)

para=round(para,2)



pd.DataFrame([para, logitmodel.pvalues],columns=logitmodel.params.index,index=['Odds Ratio','pvalue']).T


from sklearn.model_selection import train_test_split
#Defining The X and Y values 

x=df.drop('diagnosis',axis=1)

y=df.diagnosis
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
#Fitting the Model 

lr.fit(xtrain,ytrain)
#Predicting the model

ypred = lr.predict(xtest)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print("Accuracy with Logisitic Regression model is ",accuracy_score(ytest,ypred)*100, "%")
print("Classification Report", classification_report(ytest,ypred))
print("Confusion Matrix ",confusion_matrix(ytest,ypred))


cm=sklearn.metrics.confusion_matrix(ytest,ypred)

plt.figure(figsize=(8,5))

sns.heatmap(cm,annot=True,fmt='.3g',xticklabels=['Predicted 0','Predicted 1'],yticklabels=['Actual 0','Actual 1'],cmap='Blues')

plt.title('Confusion Matrix')

plt.show()


True_Postive= cm[1,1]

True_Negative=cm[0,0]

False_Positive=cm[0,1] # Type 1 Error

False_Negative=cm[1,0] #Type 2 Error
Sensitivity= True_Postive/(True_Postive+False_Negative)

Specificity=True_Negative/(True_Negative+False_Positive)

print('Sensitivity is ',round(Sensitivity,2))

print('Specificity is',round(Specificity,2))
predicted_prob=lr.predict_proba(xtest)
predicted_prob=pd.DataFrame(predicted_prob,columns=['Prob of No cancer 0','Prob of Cancer 1'])


predicted_prob.head()


from sklearn.metrics import roc_curve,roc_auc_score


tpr,fpr,thresholds=roc_curve(ytest,predicted_prob['Prob of No cancer 0'])
plt.plot(fpr,tpr)

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.title('ROC Curve for Breast Cancer Prediction Classfier')

plt.show()

print('Area Under Curve is',roc_auc_score(ytest,predicted_prob['Prob of Cancer 1']))