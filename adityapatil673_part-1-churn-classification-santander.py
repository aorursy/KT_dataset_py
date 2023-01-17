import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

classify = pd.read_csv('../input/classification_data.csv')
classify.head(5)
classify.shape
classify.target.value_counts()
classify = classify.drop('Unnamed: 0', 1)
classify['renta_sqrt'] =  classify['renta'].pow(1./2)
#transforming the renta column to reduce the variance
renta = classify.pop('renta')
train, test = train_test_split(classify, test_size=0.25)
n,m = train.shape
X = train.iloc[:,2:m]
Y = train.iloc[:,1]

X_test = test.iloc[:,2:m]
Y_test = test.iloc[:,1]
def get_metrics(pred, cutoff=0.5):
    fin_val = []
    fin_attr = ['True Positive rate','False Postive Rate',
                'Accuracy','Precision','Recall','Specificity',
                'TP','FN','FP','TN']
    pred.iloc[:,0] = np.where(pred.iloc[:,0]>cutoff,1,0)
    tot = pred.shape[0]
    TP = sum(pred.iloc[:,1]*pred.iloc[:,0])
    act_positive = sum(pred.iloc[:,1])
    TPR = float(TP)/float(act_positive)
    FP = sum(pred.iloc[:,0])-TP
    act_neg = tot-sum(pred.iloc[:,1])
    FPR = float(FP)/float(act_neg)
    acc = float(sum((pred.iloc[:,1]==pred.iloc[:,0])))/tot
    prec = float(TP)/sum(pred.iloc[:,0])
    recall = float(TP)/float(act_positive)
    specificity = 1 - FPR
    FN = act_positive - TP
    TN = act_neg - FP
    fin_val = [TPR,FPR,acc,prec,recall,specificity,TP,FN,FP,TN]
    fin = pd.DataFrame(list(zip(fin_attr,fin_val)))
    fin.columns = ['metric','value']
    return fin
logisticRegr = LogisticRegression()
lm = logisticRegr.fit(X, Y)

pred = lm.predict_proba(X_test)[:,1]
df = pd.DataFrame({'prediction': pred, 'true_output': Y_test})
logitic_metrics_half = get_metrics(df.iloc[:, 0:2], cutoff=0.5)
logitic_metrics_half
df_default = df.copy()
df_default['prediction'] = 1
logitic_metrics_default = get_metrics(df_default.iloc[:, 0:2], cutoff=0.5)
logitic_metrics_default
# Increasing the threshold to adjust for imbalanced data
logitic_metrics_default_88 = get_metrics(df_default.iloc[:, 0:2], cutoff=0.88)
logitic_metrics_88 = get_metrics(df.iloc[:, 0:2], cutoff=0.88)
logitic_metrics_default_88
logitic_metrics_88
clf=RandomForestClassifier(n_estimators=50)
rf = clf.fit(X,Y)
rf_y = clf.predict_proba(X_test)[:,1]
rf_df = pd.DataFrame({'prediction': rf_y, 'true_output': Y_test})
rf_metrics_half = get_metrics(rf_df.iloc[:, 0:2], cutoff=0.5)
rf_metrics_88 = get_metrics(rf_df.iloc[:, 0:2], cutoff=0.88)
rf_metrics_88
false_positive_rate_log, true_positive_rate_log, thresholds_log = roc_curve(Y_test, pred)
roc_auc_log = auc(false_positive_rate_log, true_positive_rate_log)

plt.title('Receiver Operating Characteristic \n Logistic Regression')
plt.plot(false_positive_rate_log, true_positive_rate_log, 'b',
label='AUC = %0.2f'% roc_auc_log)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

false_positive_rate_rf, true_positive_rate_rf, thresholds_rf = roc_curve(Y_test, rf_y)
roc_auc_rf = auc(false_positive_rate_rf, true_positive_rate_rf)

plt.title('Receiver Operating Characteristic \n Random Forest')
plt.plot(false_positive_rate_rf, true_positive_rate_rf, 'b',
label='AUC = %0.2f'% roc_auc_rf)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
def profit(pred,cost,save,cutoff):
    fin = pd.DataFrame()
    fin['predict'] = np.where(pred.iloc[:,0]>cutoff,1,0)
    fin['truth']=list(pred.iloc[:,1])
    tot = fin.shape[0]
    TP = sum(fin.iloc[:,1]*fin.iloc[:,0])
    FP = sum(fin.iloc[:,0])-TP
    act_neg = tot-sum(fin.iloc[:,1])
    TN = act_neg - FP
    pred_negative = sum(fin.iloc[:,0]==0)
    final = (save*TN) - (cost*pred_negative)
    return final
#Logistic regression 
cutoff = []
profit_values_lr = []
for i in range(65,95,1):
    cutoff.append(float(i)/100)
    profit_values_lr.append(profit(df,1,6,float(i)/100))
    print('cutoff: '+str(float(i)/100)+' Profit: '+str(profit(df,1,6,float(i)/100)))
#Random Forest
profit_values_rf = []
for i in range(65,95,1):
    profit_values_rf.append(profit(rf_df,1,6,float(i)/100))
    print('cutoff: '+str(float(i)/100)+' Profit: '+str(profit(rf_df,1,6,float(i)/100)))
plt.plot(cutoff, profit_values_lr, label='Logistic Regression')
plt.plot(cutoff, profit_values_rf, label='Random Forest')
plt.title('Profit for Logistic Regression and random Forest')
plt.xlabel("Cutoff")
plt.ylabel("Profit")
plt.legend()
plt.show()
