import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn import datasets,linear_model

from mpl_toolkits.mplot3d import axes3d

import seaborn as sns

from sklearn.preprocessing import scale

import sklearn.linear_model as skl_lm

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split, cross_val_score



import statsmodels.api as sm

print("Packages LOADED")
data = pd.read_csv('/kaggle/input/diabetes-type-2/diabetes2.csv')

data.info()
# Convert the DataFrame into array

array = data.values

array
X = array[:,0:8]

y = array[:,8]

test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size)
regr = skl_lm.LogisticRegression()

regr.fit(X_train,y_train)
pred = regr.predict(X_test)

pred
cm_df = pd.DataFrame(confusion_matrix(y_test,pred).T, index=regr.classes_, columns=regr.classes_)

cm_df.index.name = 'Predicted'

cm_df.columns.name = 'True'

print(cm_df)
# Classification Report

print(classification_report(y_test,pred))
# Accuracy

regr.score(X_test,y_test)
diabetes = data

array = data.values

X = array[:,0:8]

y = array[:,8]

model = skl_lm.LogisticRegression()
def buildAndPredict(X,y,model):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,stratify=y)

    

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    

    cm_df = pd.DataFrame(confusion_matrix(y_test,pred).T,index=regr.classes_,columns=model.classes_)

    cm_df.index.name = 'Predicted'

    cm_df.columns.name = 'True'

    print(cm_df)

    

    print(regr.score(X_test,y_test))

    

    accuracy = cross_val_score(regr,X,y,cv=10,scoring='accuracy').mean()

    print("Accuracy after CV{}".format(accuracy))

    

    # Store the predicted probabilities for Class 1 (diabetic)

    y_pred_prob = model.predict_proba(X_test)[:,1]

    

    # Plot the probability of becoming diabetic in a histogram

    plt.hist(y_pred_prob,bins=8,linewidth=1.2)

    plt.xlim(0,1)

    plt.title('Histogram of predicted probabilities')

    plt.xlabel('Predicted probability of diabetes')

    plt.ylabel('Frequency')

    

    return (y_test,pred,y_pred_prob)
y_test,pred,y_pred_prob = buildAndPredict(X,y,model)
from sklearn.preprocessing import binarize

from sklearn.metrics import recall_score
def setThreshold(thr):

    y_pred_class = binarize([y_pred_prob],thr)[0]

    y_pred_class

    

    # New confusion matrix (threshold of 0.3)

    confusion_new = confusion_matrix(y_test,y_pred_class)

    print(confusion_new)

    

    TP = confusion_new[1,1]

    TN = confusion_new[0,0]

    FP = confusion_new[0,1]

    FN = confusion_new[1,0]

    

    print("Sensitivity:", TP/float(TP+FN))

    print(recall_score(y_test,y_pred_class))

    

    # calculate the specificity for the new confusion matrix

    print("Specificity:", TN/float(TN+FP))
setThreshold(0.3)
# For example Threshold 0.4

setThreshold(0.4)
# For example Threshold 0.5

setThreshold(0.5)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score



def draw_roc(y_test,y_pred_prob):

    fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)

    plt.plot(fpr,tpr)

    plt.xlim([0.0,1.0])

    plt.ylim([0.0,1.0])

    plt.title('ROC curve for diabetes classifier')

    plt.xlabel('False Positive Rate (1-Specificity)')

    plt.ylabel('True Positive Rate (Sensitivity)')

    plt.grid(True)

    print(roc_auc_score(y_test,y_pred_prob))
draw_roc(y_test,y_pred_prob)
# AUC score using cross validation method

cross_val_score(model,X,y,cv=10,scoring='roc_auc').mean()
get_ipython().magic('matplotlib inline')



sns.boxplot(data.Outcome,data.Glucose)
sns.boxplot(data.Outcome,data.BloodPressure)
sns.boxplot(data.Outcome,data.SkinThickness)
sns.boxplot(data.Outcome,data.Insulin)
sns.boxplot(data.Outcome,data.BMI)
sns.boxplot(data.Outcome,data.DiabetesPedigreeFunction)
sns.boxplot(data.Outcome,data.Age)
data_n = data[['Glucose','Age','DiabetesPedigreeFunction','BMI','Insulin','SkinThickness','BloodPressure','Outcome']]

sns.pairplot(data_n)
corr = data.corr()

print(corr)

print('-'*30)

mask = np.zeros_like(corr,dtype=np.bool)

print(mask)

print('-'*30)

mask[np.triu_indices_from(mask)] = True

f,ax = plt.subplots(figsize=(11,9))

#Generate a custom diverging colormap

cmap = sns.diverging_palette(220,10,as_cmap=True)

sns.heatmap(corr,mask=mask,cmap=cmap,vmax=.3,square=True,linewidth=.5,cbar_kws={"shrink":.5},ax=ax)
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features',y=1.05,size=15)

sns.heatmap(data.corr(),linewidths=0.1,vmax=1.0,square=True,cmap=colormap,linecolor='white',annot=True)
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sb



sb.pairplot(data.dropna(),hue='Outcome',palette="husl")
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

cm = np.corrcoef(data[cols].values.T)

sns.set(font_scale=1.5)

hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols,xticklabels=cols)

plt.show()
data.describe()
truediabetes = data.loc[data['Outcome']==1]

truediabetes
len(truediabetes)
truediabetes.mean()
falsediabetes = data.loc[data['Outcome']==0]

len(falsediabetes)
falsediabetes.mean()
plt.figure(figsize=(20,20))

for column_index,column in enumerate(falsediabetes.columns):

    if column == 'Outcome':

        continue

    plt.subplot(4,4,column_index+1)

    sb.violinplot(x='Outcome',y=column,data=falsediabetes)
plt.figure(figsize=(20,20))

for column_index,column in enumerate(truediabetes.columns):

    if column=='Outcome':

        continue

    plt.subplot(4,4,column_index+1)

    sb.violinplot(x='Outcome',y=column,data=truediabetes)