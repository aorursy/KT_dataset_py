

#Imporing necessary Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import imblearn
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#reading source file

data=pd.read_csv("/kaggle/input/bank-full.csv")
#to check the head of the data-frame

data.head(10)
#checking the dtypes of the data

data.dtypes
#Checking the information of the data set

data.info()
#Checking the shape of the data-set and the target column

print(data.shape)

data['Target'].value_counts()
#To check if there are any null values present

nulllvalues=data.isnull().sum()

print(nulllvalues)

#To check if there are any NaN values present

NaNvalues=data.isna().sum()

print(NaNvalues)
#Changing Target to numerical representation to use in EDA

Target_dict={'yes':1,'no':0}



data['Target']=data.Target.map(Target_dict)



data.head()
#To describe the data- Five point summary

data.describe().T
#Distribution of continous data



plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,3,1)

plt.title('Age')

sns.distplot(data['age'],color='red')



#Subplot 2

plt.subplot(1,3,2)

plt.title('Balance')

sns.distplot(data['balance'],color='blue')



#Subplot 3

plt.subplot(1,3,3)

plt.title('Duration')

sns.distplot(data['duration'],color='green')







plt.figure(figsize=(30,6))



#Subplot 1- Boxplot

plt.subplot(1,3,1)

plt.title('Age')

sns.boxplot(data['age'],orient='horizondal',color='red')



#Subplot 2

plt.subplot(1,3,2)

plt.title('Balance')

sns.boxplot(data['balance'],orient='horizondal',color='blue')



#Subplot 3

plt.subplot(1,3,3)

plt.title('Duration')

sns.boxplot(data['duration'],orient='horizondal',color='green')

# Distribution of Categorical data



plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,3,1)

plt.title('Contact')

sns.countplot(data['contact'],color='cyan')



#Subplot 2

plt.subplot(1,3,2)

plt.title('Education')

sns.countplot(data['education'],color='violet')



#Subplot 3

plt.subplot(1,3,3)

plt.title('Marital')

sns.countplot(data['marital'],color='green')



plt.figure(figsize=(30,6))



#Subplot 4

plt.subplot(1,3,1)

plt.title('Default')

sns.countplot(data['default'],color='red')



#Subplot 5

plt.subplot(1,3,2)

plt.title('Housing')

sns.countplot(data['housing'],color='blue')



#Subplot 6

plt.subplot(1,3,3)

plt.title('Loan')

sns.countplot(data['loan'],color='orange')
# Distribution of Categorical data



plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,3,1)

plt.title('Day')

sns.countplot(data['day'],color='cyan')



#Subplot 2

plt.subplot(1,3,2)

plt.title('Month')

sns.countplot(data['month'],color='violet')



#Subplot 3

plt.subplot(1,3,3)

plt.title('Poutcome')

sns.countplot(data['poutcome'],color='green')
data['job'].value_counts().head(30).plot(kind='bar')
# Distribution of Target column

sns.countplot(data['Target'])
sns.catplot(x='Target',y='age', data=data)
sns.catplot(x='Target',y='balance', data=data)
sns.catplot(x='Target',y='duration', data=data)
sns.catplot(x='Target',y='campaign', data=data)
sns.catplot(x='Target',y='pdays', data=data)
sns.catplot(x='Target',y='previous', data=data)
sns.countplot(x='education',hue='Target', data=data)
sns.violinplot(x="Target", y="duration", data=data,palette='rainbow')

sns.catplot(x='marital',hue='Target',data=data,kind='count',height=4)
sns.pairplot(data, palette="Set2")
#To find the correlation between the continous variables

correlation=data.corr()

correlation.style.background_gradient(cmap='coolwarm')
sns.heatmap(correlation)
data.head()

data['Target']=data['Target'].astype('object')

data.head()
integers = data.columns[data.dtypes == 'int64']



for col in integers:

    col_z = col + '-z'

    data[col_z] = (data[col] - data[col].mean())/data[col].std(ddof=0) 



data.drop(['age','balance','day','duration','campaign','pdays','previous'],axis=1,inplace=True)
data.head()
#Checking the dtypes after obtaining z-score

data.dtypes
cleanup_nums = {

               "education":     {"primary": 1, "secondary": 2,"tertiary":3,"unknown":-1},

               "housing":     {"yes": 1, "no": 0},

               "loan":        {"yes": 1, "no": 0},

               "default":        {"yes": 1, "no": 0},

               "marital":     {"single": 1, "married": 2,"divorced":3},

               "poutcome":     {"success": 3, "other": 2,"unknown":-1,"failure":0},

               "contact":{"cellular": 1, "telephone": 2, "unknown": -1},

               "Target":{"1":1,"0":0}

                

                }

                

data.replace(cleanup_nums, inplace=True)



for categories in data.columns[data.columns=='object']:

    data[categories]=data[categories].astype("int32")



data.dtypes
data.head()
floats = data.columns[data.dtypes == 'float64']



for x in floats:

    indexNames_larger = data[ data[x]>3].index

    indexNames_lesser = data[ data[x]<-3].index

    # Delete these row indexes from dataFrame

    data.drop(indexNames_larger , inplace=True)

    data.drop(indexNames_lesser , inplace=True)

data.shape

data.head()
categoricals=['month','job']



for cols in categoricals:

    data=pd.concat([data,pd.get_dummies(data[cols],prefix=cols)],axis=1)

    data.drop(cols,axis=1,inplace=True)
data['Target']=data['Target'].astype('category')



data.dtypes
import imblearn

X=data.drop(['Target','duration-z'],axis=1)

Target_Variable=data['Target']

Y=Target_Variable

X.head()

Y=Y.astype("int32")
Y.head()

Y.value_counts()
#Importing necessary libraries

from sklearn.model_selection import train_test_split



Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.3,random_state=22)

print(Ytrain.value_counts())

print(Ytest.value_counts())
from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '1': {}".format(sum(Ytrain==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(Ytrain==0)))



sm = SMOTE(random_state=2)

Xtrain_res, Ytrain_res = sm.fit_sample(Xtrain, Ytrain)



print('After OverSampling, the shape of train_X: {}'.format(Xtrain_res.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(Ytrain_res.shape))



print("After OverSampling, counts of label '1': {}".format(sum(Ytrain_res==1)))

print("After OverSampling, counts of label '0': {}".format(sum(Ytrain_res==0)))

log_cols = ["Classifier", "Accuracy","Precision Score","Recall Score","F1-Score","roc-auc_Score"]

log = pd.DataFrame(columns=log_cols)
#importing necessary libraries

from sklearn.linear_model import LogisticRegression
model_log_regression=LogisticRegression(solver="liblinear")
model_log_regression.fit(Xtrain_res,Ytrain_res)

coef_df = pd.DataFrame(model_log_regression.coef_)

coef_df['intercept'] = model_log_regression.intercept_

print(coef_df)
#Checking the score for logistic regression

logistic_regression_Trainscore=model_log_regression.score(Xtrain_res,Ytrain_res)

print("The score for Logistic regression-Training Data is {0:.2f}%".format(logistic_regression_Trainscore*100))

logistic_regression_Testscore=model_log_regression.score(Xtest,Ytest)

print("The score for Logistic regression-Test Data is {0:.2f}%".format(logistic_regression_Testscore*100))
#Predicting the Y values

Ypred=model_log_regression.predict(Xtest)



#Misclassification error

LR_MSE=1-logistic_regression_Testscore

print("Misclassification error of Logistical Regression model is {0:.1f}%".format(LR_MSE*100))
#Confusion Matrix

from sklearn import metrics

cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True)

print(metrics.classification_report(Ytest, Ypred, digits=3))
accuracy_score=metrics.accuracy_score(Ytest,Ypred)

percision_score=metrics.precision_score(Ytest,Ypred)

recall_score=metrics.recall_score(Ytest,Ypred)

f1_score=metrics.f1_score(Ytest,Ypred)

print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))

print("The Percission of this model is {0:.2f}%".format(percision_score*100))

print("The Recall score of this model is {0:.2f}%".format(recall_score*100))

print("The F1 score of this model is {0:.2f}%".format(f1_score*100))
#AUC ROC curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



logit_roc_auc = roc_auc_score(Ytest, model_log_regression.predict(Xtest))

fpr, tpr, thresholds = roc_curve(Ytest, model_log_regression.predict_proba(Xtest)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
auc_score = metrics.roc_auc_score(Ytest, model_log_regression.predict_proba(Xtest)[:,1])

print("The AUC score is {0:.2f}".format(auc_score))
log_entry = pd.DataFrame([["Logistic Regression",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)

log = log.append(log_entry)

log
#Importing necessary libraries

from sklearn.tree import DecisionTreeClassifier

#Going with Decision Tree classifier with gini criteria, max_depth is kept at 10 to avoid overfitting of data

dtc=DecisionTreeClassifier(criterion='gini',random_state = 22,max_depth=10, min_samples_leaf=3,max_leaf_nodes=None)

#Fitting the data

dtc.fit(Xtrain_res,Ytrain_res)
#Predicting the data

Ypred=dtc.predict(Xtest)
from sklearn import metrics
#Checking the score for Decision Tree Classifier

Decision_Tree_Trainscore=dtc.score(Xtrain_res,Ytrain_res)

print("The score for Decision Tree-Training Data is {0:.2f}%".format(Decision_Tree_Trainscore*100))

Decision_Tree_Testscore=dtc.score(Xtest,Ytest)

print("The score for Decision Tree-Test Data is {0:.2f}%".format(Decision_Tree_Testscore*100))
#Misclassification error

DTC_MSE=1-Decision_Tree_Testscore

print("Misclassification error of Decision Tree Classification model is {0:.1f}%".format(DTC_MSE*100))
accuracy_score=metrics.accuracy_score(Ytest,Ypred)

percision_score=metrics.precision_score(Ytest,Ypred)

recall_score=metrics.recall_score(Ytest,Ypred)

f1_score=metrics.f1_score(Ytest,Ypred)

print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))

print("The Percission of this model is {0:.2f}%".format(percision_score*100))

print("The Recall score of this model is {0:.2f}%".format(recall_score*100))

print("The F1 score of this model is {0:.2f}%".format(f1_score*100))
#Confusion Matrix

cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True, cmap="YlGnBu")

print(metrics.classification_report(Ytest, Ypred, digits=3))
#AUC ROC curve



dtc_auc = roc_auc_score(Ytest, dtc.predict(Xtest))

fpr, tpr, thresholds = roc_curve(Ytest, dtc.predict_proba(Xtest)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Decision Tree Classifier (area = %0.2f)' % dtc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.savefig('dtc_ROC')

plt.show()
## Calculating feature importance

feat_importance = dtc.tree_.compute_feature_importances(normalize=False)



feat_imp_dict = dict(zip(X.columns, dtc.feature_importances_))

feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')

feat_imp.sort_values(by=0, ascending=False)
auc_score = metrics.roc_auc_score(Ytest, dtc.predict_proba(Xtest)[:,1])

print("The AUC score is {0:.2f}".format(auc_score))
log_entry = pd.DataFrame([["Decision Tree Classifier",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)

log = log.append(log_entry)

log
# Importing libraries

from sklearn.ensemble import RandomForestClassifier

from sklearn import model_selection



# Random Forest Classifier with gini critireon and max_depth of 150 to increase overfitting

kfold = model_selection.KFold(n_splits=10, random_state=22,shuffle=True)

rf = RandomForestClassifier(n_estimators = 100,criterion = 'gini', max_depth = 150, min_samples_leaf=1,class_weight='balanced')

rf = rf.fit(Xtrain_res, Ytrain_res)

results = model_selection.cross_val_score(rf, Xtrain_res, Ytrain_res, cv=kfold)

print(results)

Ypred = rf.predict(Xtest)

Random_Forest_Trainscore=rf.score(Xtrain_res,Ytrain_res)

print("The score for Random Forest-Training Data is {0:.2f}%".format(Random_Forest_Trainscore*100))

Random_Forest_Testscore=rf.score(Xtest,Ytest)

print("The score for Random Forest-Test Data is {0:.2f}%".format(Random_Forest_Testscore*100))
#Misclassification error

RF_MSE=1-Random_Forest_Testscore

print("Misclassification error of Random Forest Classification model is {0:.1f}%".format(RF_MSE*100))
accuracy_score=metrics.accuracy_score(Ytest,Ypred)

percision_score=metrics.precision_score(Ytest,Ypred)

recall_score=metrics.recall_score(Ytest,Ypred)

f1_score=metrics.f1_score(Ytest,Ypred)

print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))

print("The Percission of this model is {0:.2f}%".format(percision_score*100))

print("The Recall score of this model is {0:.2f}%".format(recall_score*100))

print("The F1 score of this model is {0:.2f}%".format(f1_score*100))

print(metrics.classification_report(Ytest,Ypred))
#Confusion Matrix

cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True, cmap="BuPu")
#AUC ROC curve



rf_auc = roc_auc_score(Ytest, rf.predict(Xtest))

fpr, tpr, thresholds = roc_curve(Ytest, rf.predict_proba(Xtest)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Random Forest Classifier (area = %0.2f)' % rf_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.savefig('rf_ROC')

plt.show()

auc_score = metrics.roc_auc_score(Ytest, rf.predict_proba(Xtest)[:,1])

print("The AUC score is {0:.2f}".format(auc_score))
log_entry = pd.DataFrame([["Random Forest Classifier",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)

log = log.append(log_entry)

log
# Importing libraries

from sklearn.ensemble import BaggingClassifier



bg = BaggingClassifier(n_estimators=100, max_samples= .9, bootstrap=True, oob_score=True, random_state=22)

bg = bg.fit(Xtrain_res, Ytrain_res)

Ypred = bg.predict(Xtest)

Bagging_Trainscore=bg.score(Xtrain_res, Ytrain_res)

print("The score for Bagging-Training Data is {0:.2f}%".format(Bagging_Trainscore*100))

Bagging_Testscore=bg.score(Xtest,Ytest)

print("The score for Bagging-Test Data is {0:.2f}%".format(Bagging_Testscore*100))
#Misclassification error

BG_MSE=1-Bagging_Testscore

print("Misclassification error of Bagging Classification model is {0:.1f}%".format(BG_MSE*100))
accuracy_score=metrics.accuracy_score(Ytest,Ypred)

percision_score=metrics.precision_score(Ytest,Ypred)

recall_score=metrics.recall_score(Ytest,Ypred)

f1_score=metrics.f1_score(Ytest,Ypred)

print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))

print("The Percission of this model is {0:.2f}%".format(percision_score*100))

print("The Recall score of this model is {0:.2f}%".format(recall_score*100))

print("The F1 score of this model is {0:.2f}%".format(f1_score*100))

print(metrics.classification_report(Ytest,Ypred))
#Confusion Matrix

cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True, cmap="Greens")
#AUC ROC curve



bg_auc = roc_auc_score(Ytest, bg.predict(Xtest))

fpr, tpr, thresholds = roc_curve(Ytest, bg.predict_proba(Xtest)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Bagging Classifier (area = %0.2f)' % bg_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.savefig('bg_ROC')

plt.show()

auc_score = metrics.roc_auc_score(Ytest, bg.predict_proba(Xtest)[:,1])

print("The AUC score is {0:.2f}".format(auc_score))
log_entry = pd.DataFrame([["Bagging Classifier",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)

log = log.append(log_entry)

log
#Importing necessary libraries

from sklearn.ensemble import AdaBoostClassifier

ab = AdaBoostClassifier(n_estimators= 100, learning_rate=0.5, random_state=22)

ab = ab.fit(Xtrain_res, Ytrain_res)
Ypred=ab.predict(Xtest)
Adaboosting_Trainscore=ab.score(Xtrain_res,Ytrain_res)

print("The score for Adaboosting-Training Data is {0:.2f}%".format(Adaboosting_Trainscore*100))

Adaboosting_Testscore=ab.score(Xtest,Ytest)

print("The score for Adaboosting-Test Data is {0:.2f}%".format(Adaboosting_Testscore*100))
#Misclassification error

AB_MSE=1-Adaboosting_Testscore

print("Misclassification error of Bagging Classification model is {0:.1f}%".format(AB_MSE*100))
accuracy_score=metrics.accuracy_score(Ytest,Ypred)

percision_score=metrics.precision_score(Ytest,Ypred)

recall_score=metrics.recall_score(Ytest,Ypred)

f1_score=metrics.f1_score(Ytest,Ypred)

print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))

print("The Percission of this model is {0:.2f}%".format(percision_score*100))

print("The Recall score of this model is {0:.2f}%".format(recall_score*100))

print("The F1 score of this model is {0:.2f}%".format(f1_score*100))

print(metrics.classification_report(Ytest,Ypred))
#Confusion Matrix

cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True, cmap="Reds")
#AUC ROC curve



ab_auc = roc_auc_score(Ytest, ab.predict(Xtest))

fpr, tpr, thresholds = roc_curve(Ytest, ab.predict_proba(Xtest)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Bagging Classifier (area = %0.2f)' % ab_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.savefig('ab_ROC')

plt.show()

auc_score = metrics.roc_auc_score(Ytest, ab.predict_proba(Xtest)[:,1])

print("The AUC score is {0:.2f}".format(auc_score))
log_entry = pd.DataFrame([["Adaptive Boosting Classifier",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)

log = log.append(log_entry)

log
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import GradientBoostingClassifier

num_estimators = [100,200]

learn_rates = [0.2,0.3]



scoreFunction = {"recall": "recall", "precision": "precision"}



param_grid = {'n_estimators': num_estimators,

              'learning_rate': learn_rates,

}



random_search =RandomizedSearchCV(GradientBoostingClassifier(loss='deviance'), param_grid, scoring = scoreFunction,               

                                       refit = "precision", random_state=1, n_iter=100, cv=5, verbose=0, n_jobs=-1)



random_search.fit(Xtrain_res, Ytrain_res)
random_search.best_params_
# Importing necessary libraries and fitting the data



from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators = 200, learning_rate = 0.3, random_state=22)

gb = gb.fit(Xtrain_res, Ytrain_res)

Ypred = gb.predict(Xtest)
Gradient_Booosting_Trainscore=gb.score(Xtrain_res,Ytrain_res)

print("The score for Gradient_Booosting-Training Data is {0:.2f}%".format(Gradient_Booosting_Trainscore*100))

Gradient_Booosting_Testscore=gb.score(Xtest,Ytest)

print("The score for Gradient_Booostinge-Test Data is {0:.2f}%".format(Gradient_Booosting_Testscore*100))
#Misclassification error

GB_MSE=1-Gradient_Booosting_Testscore

print("Misclassification error of Gradient Boosting Classification model is {0:.1f}%".format(GB_MSE*100))
accuracy_score=metrics.accuracy_score(Ytest,Ypred)

percision_score=metrics.precision_score(Ytest,Ypred)

recall_score=metrics.recall_score(Ytest,Ypred)

f1_score=metrics.f1_score(Ytest,Ypred)

print("The Accuracy of this model is {0:.2f}%".format(accuracy_score*100))

print("The Percission of this model is {0:.2f}%".format(percision_score*100))

print("The Recall score of this model is {0:.2f}%".format(recall_score*100))

print("The F1 score of this model is {0:.2f}%".format(f1_score*100))

print(metrics.classification_report(Ytest,Ypred))
#Confusion Matrix

cm=metrics.confusion_matrix(Ytest, Ypred, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True, cmap="Blues")

#AUC ROC curve



gb_auc = roc_auc_score(Ytest, gb.predict(Xtest))

fpr, tpr, thresholds = roc_curve(Ytest, gb.predict_proba(Xtest)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Gradient Boosting Classifier (area = %0.2f)' % gb_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.savefig('gb_ROC')

plt.show()

auc_score = metrics.roc_auc_score(Ytest, gb.predict_proba(Xtest)[:,1])

print("The AUC score is {0:.2f}".format(auc_score))
log_entry = pd.DataFrame([["Gradient Boosting Classifier",accuracy_score,percision_score,recall_score,f1_score,auc_score]], columns=log_cols)

log = log.append(log_entry)

log