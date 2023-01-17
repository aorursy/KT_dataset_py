import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB 

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import auc

import warnings

warnings.filterwarnings('ignore')
plt.rc('axes', titlesize=18)

plt.rc('axes', labelsize=15)

plt.rc('xtick', labelsize=13)

plt.rc('ytick', labelsize=13)

plt.rc('figure', titlesize=35)

plt.rc('legend', fontsize=12)
df=pd.read_csv('../input/pima-indians-diabetes-dataset/diabetes.csv')
df.head()
X = df.drop('Outcome',axis=1)

y = df.Outcome
avg_accuracies={}

accuracies={}

roc_auc={}

pr_auc={}
def cal_score(name,model,folds):

    scores = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    avg_result=[]

    for sc in scores:

        scores = cross_val_score(model, X, y, cv = folds, scoring = sc)

        avg_result.append(np.average(scores))

    df_avg_score=pd.DataFrame(avg_result)

    df_avg_score= df_avg_score.rename(index={0: 'Accuracy', 1:'Precision', 2:'Recall',3:'F1 score',4:'Roc auc'},columns={0:'Average'})

    avg_accuracies[name]=np.round(df_avg_score.loc['Accuracy']*100,3)

    values=[np.round(df_avg_score.loc['Accuracy']*100,3),np.round(df_avg_score.loc['Precision']*100,3),np.round(df_avg_score.loc['Recall']*100,3),np.round(df_avg_score.loc['F1 score']*100,3),np.round(df_avg_score.loc['Roc auc']*100,3)]

    plt.figure(figsize=(15,8))

    sns.set_palette('mako')

    ax=sns.barplot(x=['Accuracy','Precision','Recall','F1 score','Roc auc'],y=values)

    plt.yticks(np.arange(0,100,10))

    plt.ylabel('Percentage %',labelpad=10)

    plt.xlabel('Scoring Parameters',labelpad=10)

    plt.title('Cross Validation '+str(folds)+'-Folds Average Scores',pad=20)

    for p in ax.patches:

        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),xytext=(p.get_x()+0.3,p.get_height()+1.02))

    plt.show()
def conf_matrix(ytest,pred):

    plt.figure(figsize=(9,6))

    global cm1

    cm1 = confusion_matrix(ytest, pred)

    ax=sns.heatmap(cm1, annot= True, cmap='Blues')

    plt.title('Confusion Matrix',pad=20)
def metrics_score(cm):

    total=sum(sum(cm))

    accuracy=(cm[0,0]+cm[1,1])/total

    precision = cm[1,1]/(cm[0,1]+cm[1,1])

    sensitivity = cm[1,1]/(cm[1,0]+cm[1,1])

    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    specificity = cm[0,0]/(cm[0,1]+cm[0,0])

    values=[np.round(accuracy*100,3),np.round(precision*100,3),np.round(sensitivity*100,3),np.round(f1*100,3),np.round(specificity*100,3)]

    plt.figure(figsize=(15,8))

    sns.set_palette('magma')

    ax=sns.barplot(x=['Accuracy','Precision','Recall','F1 score','Specificity'],y=values)

    plt.yticks(np.arange(0,100,10))

    plt.ylabel('Percentage %',labelpad=10)

    plt.xlabel('Scoring Parameter',labelpad=10)

    plt.title('Metrics Scores',pad=20)

    for p in ax.patches:

        ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),xytext=(p.get_x()+0.3,p.get_height()+1.02))

    plt.show()
def plot_roc_curve(fpr, tpr):

    plt.figure(figsize=(8,6))

    plt.plot(fpr, tpr, color='Orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='black', linestyle='--')

    plt.ylabel('True Positive Rate',labelpad=10)

    plt.xlabel('False Positive Rate',labelpad=10)

    plt.title('Receiver Operating Characteristic (ROC) Curve',pad=20)

    plt.legend()

    plt.show()
def plot_precision_recall_curve(recall, precision):

    plt.figure(figsize=(8,6))

    plt.plot(recall, precision, color='orange', label='PRC')

    plt.ylabel('Precision',labelpad=10)

    plt.xlabel('Recall',labelpad=10)

    plt.title('Precision Recall Curve',pad=20)

    plt.legend()

    plt.show()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 5)

rf = RandomForestClassifier(n_estimators = 108, random_state = 5)

rf.fit(X_train, y_train)

prediction1 = rf.predict(X_test)

accuracy1 = rf.score(X_test, y_test) 

print ('Model Accuracy:',accuracy1 * 100)
accuracies['Random Forest'] = np.round(accuracy1 * 100,3)
conf_matrix(y_test,prediction1)
metrics_score(cm1)
cal_score('Random Forest',rf,5)
probs = rf.predict_proba(X_test)

probs = probs[:, 1]

auc1 = roc_auc_score(y_test, probs)

roc_auc['Random Forest']=np.round(auc1,3)

print('Area under the ROC Curve (AUC): %.2f' % auc1)

fpr1, tpr1, _ = roc_curve(y_test, probs)

plot_roc_curve(fpr1, tpr1)
precision1, recall1, _ = precision_recall_curve(y_test, probs)

auc_score1 = auc(recall1, precision1)

pr_auc['Random Forest']=np.round(auc_score1,3)

print('Area under the PR Curve (AUCPR): %.2f' % auc_score1)

plot_precision_recall_curve(recall1, precision1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 4)

dtc = DecisionTreeClassifier(random_state = 5)

dtc.fit(X_train, y_train)

prediction2 = dtc.predict(X_test)

accuracy2 = dtc.score(X_test, y_test)

print ('Model Accuracy:',accuracy2 * 100)
accuracies['Decision Tree'] = np.round(accuracy2 * 100,3)
conf_matrix(y_test,prediction2)
metrics_score(cm1)
cal_score('Decision Tree',dtc,7)
probs = dtc.predict_proba(X_test)

probs = probs[:, 1]

auc2 = roc_auc_score(y_test, probs)

roc_auc['Decision Tree']=np.round(auc2,3)

print('Area under the ROC Curve (AUC): %.2f' % auc2)

fpr2, tpr2, _ = roc_curve(y_test, probs)

plot_roc_curve(fpr2, tpr2)
precision2, recall2, _ = precision_recall_curve(y_test, probs)

auc_score2 = auc(recall2, precision2)

pr_auc['Decision Tree']=np.round(auc_score2,3)

print('Area under the PR Curve (AUCPR): %.2f' % auc_score2)

plot_precision_recall_curve(recall2, precision2)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 5)

gb=GradientBoostingClassifier(n_estimators=134,learning_rate=0.2)

gb.fit(X_train, y_train)

prediction3 = gb.predict(X_test)

accuracy3 = gb.score(X_test, y_test)

print ('Model Accuracy:',accuracy3 * 100)
accuracies['Gradient Boosting'] = np.round(accuracy3 * 100,3)
conf_matrix(y_test,prediction3)
metrics_score(cm1)
cal_score('Gradient Boosting',gb,5)
probs = gb.predict_proba(X_test)

probs = probs[:, 1]

auc3 = roc_auc_score(y_test, probs)

roc_auc['Gradient Boosting']=np.round(auc3,3)

print('Area under the ROC Curve (AUC): %.2f' % auc3)

fpr3, tpr3, _ = roc_curve(y_test, probs)

plot_roc_curve(fpr3, tpr3)
precision3, recall3, _ = precision_recall_curve(y_test, probs)

auc_score3 = auc(recall3, precision3)

pr_auc['Gradient Boosting']=np.round(auc_score3,3)

print('Area under the PR Curve (AUCPR): %.2f' % auc_score3)

plot_precision_recall_curve(recall3, precision3)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 5)

knn = KNeighborsClassifier(n_neighbors =7 )

knn.fit(X_train, y_train)

prediction4 = knn.predict(X_test)

accuracy4 = knn.score(X_test, y_test) 

print ('Model Accuracy:',accuracy4 * 100)
accuracies['KNN'] = np.round(accuracy4 * 100,3)
conf_matrix(y_test,prediction4)
metrics_score(cm1)
cal_score('KNN',knn,5)
probs = knn.predict_proba(X_test)

probs = probs[:, 1]

auc4 = roc_auc_score(y_test, probs)

roc_auc['KNN']=np.round(auc4,3)

print('Area under the ROC Curve (AUC): %.2f' % auc4)

fpr4, tpr4, _ = roc_curve(y_test, probs)

plot_roc_curve(fpr4, tpr4)
precision4, recall4, _ = precision_recall_curve(y_test, probs)

auc_score4 = auc(recall4, precision4)

pr_auc['KNN']=np.round(auc_score4,3)

print('Area under the PR Curve (AUCPR): %.2f' % auc_score4)

plot_precision_recall_curve(recall4, precision4)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 2)

svm = SVC(probability=True)

svm.fit(X_train, y_train)

prediction5 = svm.predict(X_test)

accuracy5 = svm.score(X_test, y_test) 

print ('Model Accuracy:',accuracy5 * 100)
accuracies['SVM'] = np.round(accuracy5 * 100,3)
conf_matrix(y_test,prediction5)
metrics_score(cm1)
cal_score('SVM',svm,5)
probs = svm.predict_proba(X_test)

probs = probs[:, 1]

auc5 = roc_auc_score(y_test, probs)

roc_auc['SVM']=np.round(auc5,3)

print('Area under the ROC Curve (AUC): %.2f' % auc5)

fpr5, tpr5, _ = roc_curve(y_test, probs)

plot_roc_curve(fpr5, tpr5)
precision5, recall5, _ = precision_recall_curve(y_test, probs)

auc_score5 = auc(recall5, precision5)

pr_auc['SVM']=np.round(auc_score5,3)

print('Area under the PR Curve (AUCPR): %.2f' % auc_score5)

plot_precision_recall_curve(recall5, precision5)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 5)

gnb = GaussianNB()

gnb.fit(X_train, y_train) 

prediction6 = gnb.predict(X_test) 

accuracy6 = gnb.score(X_test, y_test) 

print ('Model Accuracy:',accuracy6 * 100)
accuracies['Naive Bayes'] = np.round(accuracy6 * 100,3)
conf_matrix(y_test,prediction6)
metrics_score(cm1)
cal_score('Naive Bayes',gnb,5)
probs = gnb.predict_proba(X_test)

probs = probs[:, 1]

auc6 = roc_auc_score(y_test, probs)

roc_auc['Naive Bayes']=np.round(auc6,3)

print('Area under the ROC Curve (AUC): %.2f' % auc6)

fpr6, tpr6, _ = roc_curve(y_test, probs)

plot_roc_curve(fpr6, tpr6)
precision6, recall6, _ = precision_recall_curve(y_test, probs)

auc_score6 = auc(recall6, precision6)

pr_auc['Naive Bayes']=np.round(auc_score6,3)

print('Area under the PR Curve (AUCPR): %.2f' % auc_score6)

plot_precision_recall_curve(recall6, precision6)
plt.figure(figsize=(15,8))

sns.set_palette('cividis')

ax=sns.barplot(x=list(accuracies.keys()),y=list(accuracies.values()))

plt.yticks(np.arange(0,100,10))

plt.ylabel('Percentage %',labelpad=10)

plt.xlabel('Algorithms',labelpad=10)

plt.title('Accuracy Scores Comparison',pad=20)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),xytext=(p.get_x()+0.3,p.get_height()+1.02))

plt.show()
plt.figure(figsize=(15,8))

sns.set_palette('viridis')

ax=sns.barplot(x=list(avg_accuracies.keys()),y=list(avg_accuracies.values()))

plt.yticks(np.arange(0,100,10))

plt.ylabel('Percentage %',labelpad=10)

plt.xlabel('Algorithms',labelpad=10)

plt.title('Average Accuracy Scores Comparison',pad=20)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),xytext=(p.get_x()+0.3,p.get_height()+1.02))

plt.show()
plt.figure(figsize=(8,6))

sns.set_palette('Set1')

plt.plot(fpr1, tpr1, label='Random Forest ROC')

plt.plot(fpr2, tpr2, label='Decision Tree ROC')

plt.plot(fpr3, tpr3, label='Gradient Boosting ROC')

plt.plot(fpr4, tpr4, label='KNN ROC')

plt.plot(fpr5, tpr5, label='SVM ROC')

plt.plot(fpr6, tpr6, label='Naive Bayes ROC')

plt.plot([0, 1], [0, 1], linestyle='--')

plt.ylabel('True Positive Rate',labelpad=10)

plt.xlabel('False Positive Rate',labelpad=10)

plt.title('Receiver Operating Characteristic (ROC) Curves',pad=20)

plt.legend()

plt.show()
plt.figure(figsize=(15,8))

sns.set_palette('magma')

ax=sns.barplot(x=list(roc_auc.keys()),y=list(roc_auc.values()))

#plt.yticks(np.arange(0,100,10))

plt.ylabel('Score',labelpad=10)

plt.xlabel('Algorithms',labelpad=10)

plt.title('Area under the ROC Curves (AUC)',pad=20)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),xytext=(p.get_x()+0.3,p.get_height()+0.01))

plt.show()
plt.figure(figsize=(8,6))

sns.set_palette('Set1')

plt.plot(recall1, precision1, label='Random Forest PRC')

plt.plot(recall2, precision2, label='Decision Tree PRC')

plt.plot(recall3, precision3, label='Gradient Boosting PRC')

plt.plot(recall4, precision4, label='KNN PRC')

plt.plot(recall5, precision5, label='SVM PRC')

plt.plot(recall6, precision6, label='Naive Bayes PRC')

plt.ylabel('Precision',labelpad=10)

plt.xlabel('Recall',labelpad=10)

plt.title('Precision Recall Curves',pad=20)

plt.legend()

plt.show()
plt.figure(figsize=(15,8))

sns.set_palette('mako')

ax=sns.barplot(x=list(pr_auc.keys()),y=list(pr_auc.values()))

plt.ylabel('Score',labelpad=10)

plt.xlabel('Algorithms',labelpad=10)

plt.title('Area under the PR Curves (AUCPR)',pad=20)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x(), p.get_height()),xytext=(p.get_x()+0.3,p.get_height()+0.01))

plt.show()