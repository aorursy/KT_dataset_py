# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as ml

import matplotlib.pyplot as plt

%matplotlib inline

ml.style.use('ggplot')

import warnings

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff



warnings.filterwarnings('ignore')



# Models

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,recall_score,f1_score,confusion_matrix,roc_auc_score,roc_curve,auc



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

print(data.shape)

data.head(20)
data.describe()
data.isnull().sum()
data.drop(columns=['Unnamed: 32'],axis=1,inplace=True)

data.isnull().sum()
data['diagnosis_enc']=np.nan

rep_dict = {'M': int(1), 'B': int(0)}





for i in rep_dict.keys():

    data.loc[data['diagnosis']==i,'diagnosis_enc'] = int(rep_dict[i])

    

# Drop diagnosis

data.drop(columns=['diagnosis'],inplace=True)

data.head(20)
data.hist(figsize=(30,20), color='green', bins=25)

plt.show()
colors = ['lightslategray',] * 2

colors[1] = 'crimson'

fig = go.Figure([go.Bar(x=['malignant','benign'], y=data.diagnosis_enc.value_counts().values, marker_color=colors)])

fig.update_layout(title_text='Proportion Overview')

fig.show()
plt.figure(figsize=(30,20))

sns.heatmap(data.corr(),annot=True,linewidth=0.1,linecolor='white')

plt.show()
def max_corr():

    corr = pd.DataFrame(data.corr())

    corr_dict = {}

    for c in corr.columns:

        max_corr = sorted(corr[c].values)[-2]

        m_i = corr[corr[c] == max_corr].index[0]

        corr_dict[c] = m_i

    return corr_dict



# For this dataset

c_d = max_corr()

print("HIGHLY CORRELATED FEATURE FOR EACH FEATURE : \n")

for i in c_d.keys():

    print(i," ; ",c_d[i],"\n\n")
X = np.array(data.iloc[:,1:30].values)

y = np.array(data.iloc[:,-1].values)

# Scaling

scaler = StandardScaler()

X = scaler.fit_transform(X)

# Split

trainx,testx,trainy,testy = train_test_split(X,y,test_size=0.2,random_state=4)

print("For X : trainx = {} , testx = {}".format(trainx.shape,testx.shape))

print("For y : trainy = {} , testy = {}".format(trainy.shape,testy.shape))
accuracy,recall = [],[]

for i in range(1,31):

    knn = KNeighborsClassifier(n_neighbors=i,p=2)

    knn.fit(trainx,trainy)

    y_pred=knn.predict(testx)

    accuracy.append(accuracy_score(y_pred,testy))

    recall.append(recall_score(y_pred,testy))



# Plot and find best hyperparameter

plt.figure(figsize=(20,10))

plt.plot(range(1,31),accuracy,marker="o")

plt.plot(range(1,31),recall,marker="o")

plt.legend(["Accuracy","Recall_Score"])

plt.title("Accuracy vs Recall_score")

plt.show()

print("Best n_neighbors value(accuracy) : ",np.argmax(accuracy)+1)

print("Best n_neighbors value(recall) : ",np.argmax(recall)+1)
print("CHOOSING THE BEST HYPERPARAMETER : \n")

# Choosing n_neigbors = 17

print("n_neighbors = 17")

knn_n = KNeighborsClassifier(n_neighbors=17,p=2,algorithm='auto')

knn_n.fit(trainx,trainy)

y_pred_n=knn_n.predict(testx)

tn,fp,fn,tp = confusion_matrix(y_pred_n,testy).ravel()

print("Accuracy score = ",(tp+tn)/(tp+fn+fp+tn))

print(confusion_matrix(y_pred_n,testy))



# Choosing n_neighbors = 2

print("\nn_neighbors = 2")

knn_n = KNeighborsClassifier(n_neighbors=2,p=2,algorithm='auto')

knn_n.fit(trainx,trainy)

y_pred_n=knn_n.predict(testx)

tn,fp,fn,tp = confusion_matrix(y_pred_n,testy).ravel()

print("Accuracy score = ",(tp+tn)/(tp+fn+fp+tn))

print(confusion_matrix(y_pred_n,testy))

acc_knn = (tp+tn)/(tp+fn+fp+tn)
accuracy2,recall2 = [],[]

for i in range(1,11):

    rf = RandomForestClassifier(max_depth = i,criterion='gini',max_features='auto')

    rf.fit(trainx,trainy)

    y_pred=rf.predict(testx)

    accuracy2.append(accuracy_score(y_pred,testy))

    recall2.append(recall_score(y_pred,testy))



# Plot and find best hyperparameter

plt.figure(figsize=(20,10))

plt.plot(range(1,11),accuracy2,marker="o")

plt.plot(range(1,11),recall2,marker="o")

plt.legend(["Accuracy","Recall_Score"])

plt.title("Accuracy vs Recall_score")

plt.show()

print("Best max_depth value(accuracy) : ",np.argmax(accuracy2)+1)

print("Best max_depth value(recall) : ",np.argmax(recall2)+1)
accuracy3,recall3,choice = [],[],[]

for i in range(60,101,5):

    choice.append(i)

    rf = RandomForestClassifier(n_estimators = i,max_depth = 5,criterion='gini',max_features='auto')

    rf.fit(trainx,trainy)

    y_pred=rf.predict(testx)

    accuracy3.append(accuracy_score(y_pred,testy))

    recall3.append(recall_score(y_pred,testy))



# Plot and find best hyperparameter

plt.figure(figsize=(20,10))

plt.plot(range(60,101,5),accuracy3,marker="o")

plt.plot(range(60,101,5),recall3,marker="o")

plt.legend(["Accuracy","Recall_Score"])

plt.title("Accuracy vs Recall_score")

plt.show()

print("Best n_estimators value(accuracy) : ",choice[np.argmax(accuracy3)])

print("Best n_estimators value(recall) : ",choice[np.argmax(recall3)])
rf_n = RandomForestClassifier(n_estimators = choice[np.argmax(accuracy3)],max_depth = 5,criterion='gini',max_features='auto')

rf_n.fit(trainx,trainy)

y_pred_n=rf_n.predict(testx)

tn,fp,fn,tp = confusion_matrix(y_pred_n,testy).ravel()

print("Accuracy score = ",(tp+tn)/(tp+fn+fp+tn))

print("Recall = ",recall_score(y_pred_n,testy),"\n")

print(confusion_matrix(y_pred_n,testy))
# KNN

y_scores = knn_n.predict_proba(testx)

fptpk = y_scores[:,1]

knn_score = roc_auc_score(testy,fptpk)

print("FOR KNN :\nROC AUC score = ",knn_score)

knn_fp,knn_tp,tk = roc_curve(testy,fptpk)   # Returns FPR, TPR and thresholds.

auc_knn = auc(knn_fp,knn_tp)

plt.figure(figsize=(20,10))

plt.plot(knn_fp,knn_tp,marker='o',label="KNN ROC Curve")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("ROC Curve for KNN")

plt.show()



# Random Forest

y_scoresf = rf_n.predict_proba(testx)

fptp_rf = y_scoresf[:,1]

rf_score = roc_auc_score(testy,fptp_rf)

print("FOR RANDOM FOREST :\nROC AUC score = ",rf_score)

rf_fp,rf_tp,_ = roc_curve(testy,fptp_rf)   # Returns FPR, TPR and thresholds.

auc_rf = auc(rf_fp,rf_tp)

plt.figure(figsize=(20,10))

plt.plot(rf_fp,rf_tp,marker='o',label="Random Forest ROC Curve")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("ROC Curve for Random Forest")

plt.show()



# Comparing AUC

print("AUC for KNN = ",auc_knn)

print("AUC for Random Forest = ",auc_rf)
# Before optimization

print("BEFORE THRESHOLDING : \n")

print("Accuracy score = ",acc_knn)

print("F1 score = ",f1_score(knn_n.predict(testx),testy))

print("ROC AUC score = ",knn_score)

print("\nConfusion Matrix : ")

print(confusion_matrix(knn_n.predict(testx),testy))

print("\nROC curve :")

plt.plot(knn_fp,knn_tp,marker='o',label="KNN ROC Curve")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("ROC Curve for KNN before Threshold Optimization")

plt.show()
optimal = sorted(list(zip(np.abs(knn_tp - knn_fp),tk)),key = lambda i: i[0],reverse = True)[0][1]    # Sort on basis of thresholds in descending order

y_pred_optimal = [1 if i >= optimal else 0 for i in fptpk]



# After optimization

print("AFTER THRESHOLDING : \n")

print("Accuracy score = ",accuracy_score(testy,y_pred_optimal))

print("F1 score = ",f1_score(testy,y_pred_optimal))

print("\nConfusion Matrix : ")

print(confusion_matrix(testy,y_pred_optimal))

print("\nROC curve :")

knn_fp_n,knn_tp_n,_ = roc_curve(testy,y_pred_optimal)

plt.plot(knn_fp_n,knn_tp_n,marker='o',label="KNN ROC Curve")

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("ROC Curve for KNN after Threshold Optimization")

plt.show()