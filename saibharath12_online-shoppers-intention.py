import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

# dataframe display settings

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)



# filtering warnings

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv("/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv")

df.head()
df.dropna(inplace=True)
df_new=df.copy(deep=True)
#administration duration

q1=df['Administrative_Duration'].quantile(0.25)

q3=df['Administrative_Duration'].quantile(0.75)

iqr=q3-q1

u_l1=q3+1.5*iqr
df_new['Administrative_Duration']=df_new['Administrative_Duration'].map(lambda x:0 if x==0 else (1 if x>0 and x<u_l1 else 2))

df_new['Administrative_Duration'].value_counts()
df_new['Informational_Duration']=df_new['Informational_Duration'].map(lambda x:0 if x==0 else 1)

df_new['Informational_Duration'].value_counts()
#ProductRelated_Duration

q1=df['ProductRelated_Duration'].quantile(0.25)

q3=df['ProductRelated_Duration'].quantile(0.75)

iqr=q3-q1

u_l1=q3+1.5*iqr
df_new['ProductRelated_Duration']=df_new['ProductRelated_Duration'].map(lambda x:0 if x==0 else (1 if x>0 and x<u_l1 else 2))

df_new['ProductRelated_Duration'].value_counts()
df_new.shape
df_new['Revenue'].value_counts()
ct=pd.crosstab(df_new['Administrative_Duration'],df_new['Revenue'],values=df_new['BounceRates'],aggfunc='mean')

ct
ct.plot.bar()
ct=pd.crosstab(df_new['Administrative_Duration'],df_new['Revenue'],values=df_new['ExitRates'],aggfunc='mean')

ct
ct.plot.bar()
ct=pd.crosstab(df_new['Administrative_Duration'],df_new['Revenue'],values=df_new['PageValues'],aggfunc='mean')

ct
ct.plot.bar()
ct=pd.crosstab(df_new['Informational_Duration'],df_new['Revenue'],values=df_new['BounceRates'],aggfunc='mean')

ct
ct.plot.bar()
ct=pd.crosstab(df_new['Informational_Duration'],df_new['Revenue'],values=df_new['ExitRates'],aggfunc='mean')

ct
ct.plot.bar()
ct=pd.crosstab(df_new['Informational_Duration'],df_new['Revenue'],values=df_new['PageValues'],aggfunc='mean')

ct
ct.plot.bar()
ct=pd.crosstab(df_new['ProductRelated_Duration'],df_new['Revenue'],values=df_new['BounceRates'],aggfunc='mean')
ct.plot.bar()
ct=pd.crosstab(df_new['ProductRelated_Duration'],df_new['Revenue'],values=df_new['ExitRates'],aggfunc='mean')
ct.plot.bar()
ct=pd.crosstab(df_new['ProductRelated_Duration'],df_new['Revenue'],values=df_new['PageValues'],aggfunc='mean')
ct.plot.bar()
# count of sessions for each duration
sns.countplot(df_new['Administrative_Duration'],hue=df['Revenue'])
sns.countplot(df_new['Informational_Duration'],hue=df['Revenue'])
sns.countplot(df_new['ProductRelated_Duration'],hue=df['Revenue'])
ct=pd.crosstab(df_new['Region'],df_new['Administrative_Duration'],values=df_new['PageValues'],aggfunc='mean')

ct.plot.bar()
ct=pd.crosstab(df_new['Region'],df_new['Informational_Duration'],values=df_new['PageValues'],aggfunc='mean')

ct.plot.bar()
ct=pd.crosstab(df_new['Region'],df_new['ProductRelated_Duration'],values=df_new['PageValues'],aggfunc='mean')

ct.plot.bar()
ct=pd.crosstab(df_new['Region'],df_new['Administrative_Duration'],values=df_new['BounceRates'],aggfunc='mean')

ct.plot.bar()
ct=pd.crosstab(df_new['Region'],df_new['Informational_Duration'],values=df_new['BounceRates'],aggfunc='mean')

ct.plot.bar()
ct=pd.crosstab(df_new['Region'],df_new['ProductRelated_Duration'],values=df_new['BounceRates'],aggfunc='mean')

ct.plot.bar()
df_new.head()
df1=pd.get_dummies(data=df_new,columns=['Month','OperatingSystems','Browser','Region','TrafficType','VisitorType','Weekend','Revenue'],drop_first=True)

df1.head()
df1.rename(columns={'Revenue_True':'Revenue'},inplace=True)

df1.shape
df_p=df1.copy(deep=True)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report,confusion_matrix

X=df_p.drop('Revenue',axis=1)

y=df_p['Revenue']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=5)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
cm_reference = pd.DataFrame(np.array(["TP","FP","FN","TN"]).reshape(2,2), columns=['Predicted:0','Predicted:1'], index=['Actual:0','Actual:1'])

print(cm_reference)
TP=cm[0,0]

TN=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
print("True Negatives :",TN)

print("True Positives :",TP)

print("False Negative :",FN," (Type II error)")

print("False Positives :",FP," (Type I error)")

print("correctly predicted :",TP+TN)

print("miss-classified :",FN+FP)
print('The acuuracy of the model = TP+TN / (TP+TN+FP+FN) = ',(TP+TN)/float(TP+TN+FP+FN),'\n\n',



'The Miss-classification = 1-Accuracy = ',1-((TP+TN)/float(TP+TN+FP+FN)),'\n\n',



'Sensitivity or True Positive Rate = TP / (TP+FN) = ',TP/float(TP+FN),'\n\n',



'Specificity or True Negative Rate = TN / (TN+FP) = ',TN/float(TN+FP),'\n\n',



'Positive Predictive value = TP / (TP+FP) = ',TP/float(TP+FP),'\n\n',



'Negative predictive Value = TN / (TN+FN) = ',TN/float(TN+FN),'\n\n',



'Positive Likelihood Ratio = Sensitivity / (1-Specificity) = ',sensitivity/(1-specificity),'\n\n',



'Negative likelihood Ratio = (1-Sensitivity) / Specificity = ',(1-sensitivity)/specificity)
y_pred_prob=logreg.predict_proba(x_test)[:,:]

y_pred_prob_df=pd.DataFrame(data=y_pred_prob, columns=['Prob of no purchase (0)','Prob of purchase (1)'])

y_pred_prob_df.head()
roc_auc_score(y_test,y_pred)
# results matrix

df_results = pd.DataFrame(columns=['Description','Misclassifications','Type I errors','Type II errors','Precision','Recall','Accuracy','F1-score','ROC AUC'])
# itereation results

description = "Base logit model"

misclassifications = FP + FN

type1 = FP

type2 = FN

precision = round(precision_score(y_test,y_pred),2)

recall = round(recall_score(y_test,y_pred),2)

accuracy = round(accuracy_score(y_test,y_pred),2)

f1 = round(f1_score(y_test,y_pred),2)

auc = round(roc_auc_score(y_test,y_pred),2)



df_results = pd.concat([df_results,

                        pd.DataFrame(np.array([description,

                                     misclassifications,

                                     type1,

                                     type2,

                                     precision,

                                     recall,

                                     accuracy,

                                     f1,

                                     auc]).reshape(1,-1), columns=['Description','Misclassifications','Type I errors','Type II errors','Precision','Recall','Accuracy','F1-score','ROC AUC'])

                                  ], axis=0)



df_results
print(classification_report(y_test,y_pred))
from sklearn.preprocessing import binarize

for i in range(1,5):

    cm2=0

    y_pred_prob_yes=logreg.predict_proba(x_test)

    y_pred2=binarize(y_pred_prob_yes,i/10)[:,1]

    cm2=confusion_matrix(y_test,y_pred2)

    print ('With',i/10,'threshold the Confusion Matrix is ','\n',cm2,'\n',

            'with',cm2[0,0]+cm2[1,1],'correct predictions and',cm2[1,0],'Type II errors( False Negatives)','\n\n',

          'Sensitivity: ',cm2[1,1]/(float(cm2[1,1]+cm2[1,0])),'Specificity: ',cm2[0,0]/(float(cm2[0,0]+cm2[0,1])),'\n\n\n')

    
# ROC curve
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_yes[:,1])

plt.plot(fpr,tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for online shoppers classifier')

plt.xlabel('False positive rate (1-Specificity)')

plt.ylabel('True positive rate (Sensitivity)')

plt.grid(True)
## 1)Here,we can see that even decreasing threshold it is causing type 1 error.



## 2)we can use the default threshold 0.5 
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier

from sklearn.ensemble import VotingClassifier
DT=DecisionTreeClassifier()

RF=RandomForestClassifier(criterion='entropy',n_estimators=10)

Bagged=BaggingClassifier(n_estimators=100)

AB_RF=AdaBoostClassifier(base_estimator=RF,n_estimators=150)

GBoost=GradientBoostingClassifier(n_estimators=300)

KNN=KNeighborsClassifier(n_neighbors=9,weights='distance')
models = [] 

models.append(('DT',DT))

models.append(('RandomForest',RF))

models.append(('Bagged',Bagged))

models.append(('AdaBoostRF',AB_RF))

models.append(('GradientBoost',GBoost))

models.append(('KNN',KNN))
results = []

names = []

for name, model in models:

    kfold = model_selection.KFold(n_splits=3,shuffle=True,random_state=0)

    cv_results = model_selection.cross_val_score(model,X,y,cv=kfold,scoring='f1_weighted')

    results.append(np.sqrt(np.abs(cv_results)))

    names.append(name)

    print("%s: %f (%f)" % (name, 1-np.mean(cv_results),np.std(cv_results,ddof=1)))

rf=RandomForestClassifier().fit(x_train,y_train)

y_pred_rf=rf.predict(x_test)

print(classification_report(y_test,y_pred_rf))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_rf)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")



TP=cm[0,0]

TN=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
# itereation results

description = "Random Forest Classifier"  #change the name of models

misclassifications = FP + FN

type1 = FP

type2 = FN

precision = round(precision_score(y_test,y_pred_rf),2)

recall = round(recall_score(y_test,y_pred_rf),2)

accuracy = round(accuracy_score(y_test,y_pred_rf),2)

f1 = round(f1_score(y_test,y_pred_rf),2)

auc = round(roc_auc_score(y_test,y_pred_rf),2)



df_results = pd.concat([df_results,

                        pd.DataFrame(np.array([description,

                                     misclassifications,

                                     type1,

                                     type2,

                                     precision,

                                     recall,

                                     accuracy,

                                     f1,

                                     auc]).reshape(1,-1), columns=['Description','Misclassifications','Type I errors','Type II errors','Precision','Recall','Accuracy','F1-score','ROC AUC'])

                                  ], axis=0)



df_results
ada=AdaBoostClassifier().fit(x_train,y_train)

y_pred_ada=ada.predict(x_test)

print(classification_report(y_test,y_pred_ada))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_ada)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")



TP=cm[0,0]

TN=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
# itereation results

description = "Ada Boost Classifier"  #change the name of models

misclassifications = FP + FN

type1 = FP

type2 = FN

precision = round(precision_score(y_test,y_pred_ada),2)

recall = round(recall_score(y_test,y_pred_ada),2)

accuracy = round(accuracy_score(y_test,y_pred_ada),2)

f1 = round(f1_score(y_test,y_pred_ada),2)

auc = round(roc_auc_score(y_test,y_pred_ada),2)



df_results = pd.concat([df_results,

                        pd.DataFrame(np.array([description,

                                     misclassifications,

                                     type1,

                                     type2,

                                     precision,

                                     recall,

                                     accuracy,

                                     f1,

                                     auc]).reshape(1,-1), columns=['Description','Misclassifications','Type I errors','Type II errors','Precision','Recall','Accuracy','F1-score','ROC AUC'])

                                  ], axis=0)



df_results
gbc=GradientBoostingClassifier().fit(x_train,y_train)

y_pred_gbc=gbc.predict(x_test)

print(classification_report(y_test,y_pred_ada))
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred_gbc)

conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (8,5))

sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")



TP=cm[0,0]

TN=cm[1,1]

FN=cm[1,0]

FP=cm[0,1]

sensitivity=TP/float(TP+FN)

specificity=TN/float(TN+FP)
# itereation results

description = "Gradient Boost Classifier"  #change the name of models

misclassifications = FP + FN

type1 = FP

type2 = FN

precision = round(precision_score(y_test,y_pred_gbc),2)

recall = round(recall_score(y_test,y_pred_gbc),2)

accuracy = round(accuracy_score(y_test,y_pred_gbc),2)

f1 = round(f1_score(y_test,y_pred_gbc),2)

auc = round(roc_auc_score(y_test,y_pred_gbc),2)



df_results = pd.concat([df_results,

                        pd.DataFrame(np.array([description,

                                     misclassifications,

                                     type1,

                                     type2,

                                     precision,

                                     recall,

                                     accuracy,

                                     f1,

                                     auc]).reshape(1,-1), columns=['Description','Misclassifications','Type I errors','Type II errors','Precision','Recall','Accuracy','F1-score','ROC AUC'])

                                  ], axis=0)



df_results