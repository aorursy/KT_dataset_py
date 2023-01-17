import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
%matplotlib inline
data_in=pd.read_csv(r'../input/framingham-heart-study-dataset/framingham.csv')
data_in.isna().sum()
data_in=data_in.dropna()
sn.pairplot(data=data_in)
features=['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose','TenYearCHD']
correlation=data_in[features].corr()
correlation
features=correlation[abs(correlation.TenYearCHD)>0.07].index.tolist()
features=data_in[['male', 'age', 'BPMeds', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'glucose']]
independent=data_in[['TenYearCHD']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(features,independent,test_size=.20,random_state=5)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
pred=logreg.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix, annot=True,fmt='d')
TP=cm[1,1]
TN=cm[0,0]
FP=cm[0,1]
FN=cm[1,0]
print('Accuracy=',float(TP+TN)/float(TP+TN+FP+FN))
print('Sensitivity=',float(TP)/float(TP+FN))
print('Specificty=',float(TN)/float(TN+FP))
from sklearn.metrics import roc_curve
from sklearn.preprocessing import binarize
for x in range(1,10):
    y_pred_prob_yes=logreg.predict_proba(x_test)
    pred_new=binarize(y_pred_prob_yes,x*5/100)[:,1]
    cm2=confusion_matrix(y_test,pred_new)
    TP=cm2[1,1]
    TN=cm2[0,0]
    FP=cm2[0,1]
    FN=cm2[1,0]#Vital
    print('Threshold=',x*5/100)
    conf_matrix=pd.DataFrame(data=cm2,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
    plt.figure(figsize = (8,5))
    sn.heatmap(conf_matrix, annot=True,fmt='d')
    print('Wrongly Dianosed=',FN)
    print('Accuracy=',float(TP+TN)/float(TP+TN+FP+FN))
    print('Sensitivity=',float(TP)/float(TP+FN))
    print('Specificty=',float(TN)/float(TN+FP),'\n')
y_pred_prob_yes=logreg.predict_proba(x_test)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for Heart disease classifier')
plt.xlabel('Wrongly Diagnosed Rate (1-Specificity)')
plt.ylabel('Correctly Diagnosed Rate (Sensitivity)')
plt.grid(True)
from sklearn.metrics import auc
print('AUC',auc(fpr,tpr))