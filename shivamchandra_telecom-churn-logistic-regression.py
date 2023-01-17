%config IPCompleter.greedy=True
!pip uninstall statsmodels --yes

!pip install statsmodels==0.10.0rc2 --pre
!pip install --upgrade scipy==1.1.0

!pip install statsmodels==0.10.0rc2 --pre
import pandas as pd

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_excel('../input/Churn_data.xlsx')

df.head()
#checking columns datatypes

df.dtypes
#data summary, checking for outliers 

df.describe().transpose()
#checking for any null values

df.isnull().any().any()
#checking dependent variable classes

import numpy as np

class_freq=np.bincount(df.Churn)

pChurn=class_freq[1]/sum(class_freq) #will use it later

print("probabilities:")

print("No Churn: "+str(class_freq[0]/sum(class_freq)))

print("Churn: "+str(class_freq[1]/sum(class_freq)))
#Writing a function to calculate the VIF values

import statsmodels.formula.api as sm

def vif_cal(input_data, dependent_col):

    x_vars=input_data.drop([dependent_col], axis=1)

    xvar_names=x_vars.columns

    for i in range(0,xvar_names.shape[0]):

        y=x_vars[xvar_names[i]] 

        x=x_vars[xvar_names.drop(xvar_names[i])]

        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  

        vif=round(1/(1-rsq),2)

        print (xvar_names[i], " VIF = " , vif)
#correlation graph

sns.set(style='darkgrid',palette="muted")

#fig, ax=plt.subplots(figsize=dims)

df.corr()['Churn'][1:].sort_values(ascending = False).plot(kind='bar')

plt.xlabel("Dependent Variables")

plt.ylabel("Correlation to Churn")

plt.title("Correlation to Churn")
#Calculating VIF values using that function

vif_cal(input_data=df, dependent_col="Churn")
# Acceptable vif columns: AccountWeeks, Contract Renewal, CustServCalls, DayCalls, RoamMins

vif_cal(df.drop(columns=['MonthlyCharge','DataUsage']),dependent_col="Churn")

#Removing Monthly Charge and DataUsage leads to very good improvement in vif.
ax=sns.kdeplot(df.CustServCalls[(df['Churn']==1)],color="blue",shade=True)

ax=sns.kdeplot(df.CustServCalls[(df['Churn']==0)],color="red",shade=True)

ax.legend(["Not Churn","Churn"],loc='upper right')

ax.set_ylabel('Density')

ax.set_xlabel('Customer case calls')

ax.set_title('Distribution of customer service calls by churn')
ax=sns.kdeplot(df.AccountWeeks[(df['Churn']==1)],color="blue",shade=True)

ax=sns.kdeplot(df.AccountWeeks[(df['Churn']==0)],color="red",shade=True)

ax.legend(["Not Churn","Churn"],loc='upper right')

ax.set_ylabel('Density')

ax.set_xlabel('AccountWeeks')

ax.set_title('Distribution of AccountWeeks by churn')
ax=sns.countplot("ContractRenewal",data=df,palette="rainbow",hue="Churn")

ax.legend(["Not Churn","Churn"],loc='upper right')

ax.set_ylabel('Density')

ax.set_xlabel('ContractRenewal')

ax.set_title('Distribution of ContractRenewal by churn')
ax=sns.countplot("DataPlan",data=df,palette="rainbow",hue="Churn")

ax.legend(["Not Churn","Churn"],loc='upper right')

ax.set_ylabel('Density')

ax.set_xlabel('DataPlan')

ax.set_title('Distribution of DataPlan by churn')
ax=sns.kdeplot(df.DayMins[(df['Churn']==1)],color="blue",shade=True)

ax=sns.kdeplot(df.DayMins[(df['Churn']==0)],color="red",shade=True)

ax.legend(["Not Churn","Churn"],loc='upper right')

ax.set_ylabel('Density')

ax.set_xlabel('DayMins')

ax.set_title('Distribution of DayMins by churn')
ax=sns.kdeplot(df.DayCalls[(df['Churn']==1)],color="blue",shade=True)

ax=sns.kdeplot(df.DayCalls[(df['Churn']==0)],color="red",shade=True)

ax.legend(["Not Churn","Churn"],loc='upper right')

ax.set_ylabel('Density')

ax.set_xlabel('DayCalls')

ax.set_title('Distribution of DayCalls by churn')
ax=sns.kdeplot(df.OverageFee[(df['Churn']==1)],color="blue",shade=True)

ax=sns.kdeplot(df.OverageFee[(df['Churn']==0)],color="red",shade=True)

ax.legend(["Not Churn","Churn"],loc='upper right')

ax.set_ylabel('Density')

ax.set_xlabel('OverageFee')

ax.set_title('Distribution of OverageFee by churn')
ax=sns.kdeplot(df.RoamMins[(df['Churn']==1)],color="blue",shade=True)

ax=sns.kdeplot(df.RoamMins[(df['Churn']==0)],color="red",shade=True)

ax.legend(["Not Churn","Churn"],loc='upper right')

ax.set_ylabel('Density')

ax.set_xlabel('RoamMins')

ax.set_title('Distribution of RoamMins by churn')
#logistic regression

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
X = df.drop(columns = ['Churn','MonthlyCharge','DataUsage'])

y = df['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model = LogisticRegression()

result = model.fit(X_train, y_train)
prediction_test = model.predict(X_test)

# Print the prediction accuracy

metrics.accuracy_score(y_test, prediction_test)
model.coef_[0]
weights = pd.Series(model.coef_[0],

                 index=X.columns.values)

print (weights.sort_values(ascending = False)[:10].plot(kind='bar'))
arr=metrics.confusion_matrix(y_test,prediction_test)

df_cm = pd.DataFrame(arr, range(2),range(2))

#plt.figure(figsize = (10,7))

sns.set(font_scale=1)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 10},fmt="d")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix for threshold: .5")
metrics.precision_recall_fscore_support(y_test,prediction_test)
predicted_proba=model.predict_proba(X_test)

predicted_proba
#plot precicion, recall and thresholds

#predicted_proba[:,1]

def plotPrecisionRecallThreshold(y_test, pred_prob):

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_prob) 

   #retrieve probability of being 1(in second column of probs_y)

    pr_auc = metrics.auc(recall, precision)

    plt.title("Precision-Recall vs Threshold Chart")

    plt.plot(thresholds, precision[: -1], "b--", label="Precision")

    plt.plot(thresholds, recall[: -1], "r--", label="Recall")

    plt.ylabel("Precision, Recall")

    plt.xlabel("Threshold")

    plt.legend(loc="lower left")

    plt.ylim([0,1])

    

def plotROC(y_test,pred_prob):

    fpr, tpr, threshold=metrics.roc_curve(y_test,pred_prob)

    plt.title("ROC Curve")

    sns.lineplot(x=fpr,y=tpr,palette="muted")

    plt.ylabel("True Positive Rate")

    plt.xlabel("False Positive Rate")

    

def areaUnderROC(y_test, pred_prob):

    precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_prob) 

    return metrics.auc(recall, precision)
plotPrecisionRecallThreshold(y_test, predicted_proba[:,1])
plotROC(y_test, predicted_proba[:,1])
areaUnderROC(y_test, predicted_proba[:,1])
import numpy as np

import math

pred=np.empty(1000)

probsChurn= predicted_proba[:,1]

pred=np.empty(1000)

thresh=pChurn

for i in range(0, probsChurn.size):

    if probsChurn[i]>thresh:

        pred[i]=1

    else:

        pred[i]=0

        
metrics.precision_recall_fscore_support(y_test,pred)
arr=metrics.confusion_matrix(y_test,pred)

df_cm = pd.DataFrame(arr, range(2),

                  range(2))

#plt.figure(figsize = (10,7))

sns.set(font_scale=1.2)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 15},fmt="d")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix for threshold: .145")
metrics.accuracy_score(y_test, pred)