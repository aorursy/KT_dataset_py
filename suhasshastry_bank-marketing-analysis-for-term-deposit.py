import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn.linear_model as lm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, KFold

from sklearn.preprocessing import StandardScaler, label_binarize

from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve, auc, f1_score, precision_score, recall_score

from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler, SMOTE

from imblearn.under_sampling import RandomUnderSampler
data = pd.read_csv("../input/bank-additional-full.csv",sep=';')

data.head()
data1 = data[data['y'] == 'yes']

data2 = data[data['y'] == 'no']
fig, ax = plt.subplots(2, 2, figsize=(12,10))



b1 = ax[0, 0].bar(data1['day_of_week'].unique(),height = data1['day_of_week'].value_counts(),color='#000000')

b2 = ax[0, 0].bar(data2['day_of_week'].unique(),height = data2['day_of_week'].value_counts(),bottom = data1['day_of_week'].value_counts(),color = '#DC4405') 

ax[0, 0].title.set_text('Day of week')

#ax[0, 0].legend((b1[0], b2[0]), ('Yes', 'No'))

ax[0, 1].bar(data1['month'].unique(),height = data1['month'].value_counts(),color='#000000')

ax[0, 1].bar(data2['month'].unique(),height = data2['month'].value_counts(),bottom = data1['month'].value_counts(),color = '#DC4405') 

ax[0, 1].title.set_text('Month')

ax[1, 0].bar(data1['job'].unique(),height = data1['job'].value_counts(),color='#000000')

ax[1, 0].bar(data1['job'].unique(),height = data2['job'].value_counts()[data1['job'].value_counts().index],bottom = data1['job'].value_counts(),color = '#DC4405') 

ax[1, 0].title.set_text('Type of Job')

ax[1, 0].tick_params(axis='x',rotation=90)

ax[1, 1].bar(data1['education'].unique(),height = data1['education'].value_counts(),color='#000000') #row=0, col=1

ax[1, 1].bar(data1['education'].unique(),height = data2['education'].value_counts()[data1['education'].value_counts().index],bottom = data1['education'].value_counts(),color = '#DC4405') 

ax[1, 1].title.set_text('Education')

ax[1, 1].tick_params(axis='x',rotation=90)

#ax[0, 1].xticks(rotation=90)

plt.figlegend((b1[0], b2[0]), ('Yes', 'No'),loc="right",title = "Term deposit")

plt.show()
fig, ax = plt.subplots(2, 3, figsize=(15,10))



b1 = ax[0, 0].bar(data1['marital'].unique(),height = data1['marital'].value_counts(),color='#000000')

b2 = ax[0, 0].bar(data1['marital'].unique(),height = data2['marital'].value_counts()[data1['marital'].value_counts().index],bottom = data1['marital'].value_counts(),color = '#DC4405') 

ax[0, 0].title.set_text('Marital Status')

#ax[0, 0].legend((b1[0], b2[0]), ('Yes', 'No'))

ax[0, 1].bar(data1['housing'].unique(),height = data1['housing'].value_counts(),color='#000000')

ax[0, 1].bar(data1['housing'].unique(),height = data2['housing'].value_counts()[data1['housing'].value_counts().index],bottom = data1['housing'].value_counts(),color = '#DC4405') 

ax[0, 1].title.set_text('Has housing loan')

ax[0, 2].bar(data1['loan'].unique(),height = data1['loan'].value_counts(),color='#000000')

ax[0, 2].bar(data1['loan'].unique(),height = data2['loan'].value_counts()[data1['loan'].value_counts().index],bottom = data1['loan'].value_counts(),color = '#DC4405') 

ax[0, 2].title.set_text('Has personal loan')

ax[1, 0].bar(data1['contact'].unique(),height = data1['contact'].value_counts(),color='#000000')

ax[1, 0].bar(data1['contact'].unique(),height = data2['contact'].value_counts()[data1['contact'].value_counts().index],bottom = data1['contact'].value_counts(),color = '#DC4405') 

ax[1, 0].title.set_text('Type of Contact')

ax[1, 1].bar(data1['default'].unique(),height = data1['default'].value_counts(),color='#000000')

ax[1, 1].bar(data1['default'].unique(),height = data2['default'].value_counts()[data1['default'].value_counts().index],bottom = data1['default'].value_counts(),color = '#DC4405') 

ax[1, 1].title.set_text('Has credit in default')

ax[1, 2].bar(data1['poutcome'].unique(),height = data1['poutcome'].value_counts(),color='#000000')

ax[1, 2].bar(data1['poutcome'].unique(),height = data2['poutcome'].value_counts()[data1['poutcome'].value_counts().index],bottom = data1['poutcome'].value_counts(),color = '#DC4405') 

ax[1, 2].title.set_text('Outcome of the previous marketing campaign')

plt.figlegend((b1[0], b2[0]), ('Yes', 'No'),loc="right",title = "Term deposit")

plt.show()
fig, ax = plt.subplots(2, 2, figsize=(12,10))



ax[0, 0].hist(data2['age'],color = '#DC4405',alpha=0.7,bins=20, edgecolor='white') 

ax[0, 0].hist(data1['age'],color='#000000',alpha=0.5,bins=20, edgecolor='white')

ax[0, 0].title.set_text('Age')

ax[0, 1].hist(data2['duration'],color = '#DC4405',alpha=0.7, edgecolor='white') 

ax[0, 1].hist(data1['duration'],color='#000000',alpha=0.5, edgecolor='white')

ax[0, 1].title.set_text('Contact duration')

ax[1, 0].hist(data2['campaign'],color = '#DC4405',alpha=0.7, edgecolor='white') 

ax[1, 0].hist(data1['campaign'],color='#000000',alpha=0.5, edgecolor='white')

ax[1, 0].title.set_text('Number of contacts performed')

ax[1, 1].hist(data2[data2['pdays'] != 999]['pdays'],color = '#DC4405',alpha=0.7, edgecolor='white') 

ax[1, 1].hist(data1[data1['pdays'] != 999]['pdays'],color='#000000',alpha=0.5, edgecolor='white')

ax[1, 1].title.set_text('Previous contact days')

plt.figlegend((b1[0], b2[0]), ('Yes', 'No'),loc="right",title = "Term deposit")

plt.show()
fig, ax = plt.subplots(2, 3, figsize=(15,10))

ax[0, 0].hist(data2['previous'],color = '#DC4405',alpha=0.7, edgecolor='white') 

ax[0, 0].hist(data1['previous'],color='#000000',alpha=0.5, edgecolor='white')

ax[0, 0].title.set_text('Number of contacts performed previously')

ax[0, 1].hist(data2['emp.var.rate'],color = '#DC4405',alpha=0.7, edgecolor='white') 

ax[0, 1].hist(data1['emp.var.rate'],color='#000000',alpha=0.5, edgecolor='white')

ax[0, 1].title.set_text('Employment variation rate')

ax[0, 2].hist(data2['cons.price.idx'],color = '#DC4405',alpha=0.7, edgecolor='white') 

ax[0, 2].hist(data1['cons.price.idx'],color='#000000',alpha=0.5, edgecolor='white')

ax[0, 2].title.set_text('Consumer price index')

ax[1, 0].hist(data2['cons.conf.idx'],color = '#DC4405',alpha=0.7, edgecolor='white') 

ax[1, 0].hist(data1['cons.conf.idx'],color='#000000',alpha=0.5, edgecolor='white')

ax[1, 0].title.set_text('Consumer confidence index')

ax[1, 1].hist(data2['euribor3m'],color = '#DC4405',alpha=0.7, edgecolor='white') 

ax[1, 1].hist(data1['euribor3m'],color='#000000',alpha=0.5, edgecolor='white')

ax[1, 1].title.set_text('Euribor 3 month rate')

ax[1, 2].hist(data2['nr.employed'],color = '#DC4405',alpha=0.7, edgecolor='white') 

ax[1, 2].hist(data1['nr.employed'],color='#000000',alpha=0.5, edgecolor='white')

ax[1, 2].title.set_text('Number of employees')

plt.figlegend((b1[0], b2[0]), ('Yes', 'No'),loc="right",title = "Term deposit")

plt.show()
predictors = data.iloc[:,0:20]

predictors = predictors.drop(['pdays'],axis=1)

y = data.iloc[:,20]

X = pd.get_dummies(predictors)
y.value_counts()
rus = RandomUnderSampler(random_state=0)

X_Usampled, y_Usampled = rus.fit_resample(X, y)

pd.Series(y_Usampled).value_counts()
ros = RandomOverSampler(random_state=0)

X_Osampled, y_Osampled = ros.fit_resample(X, y)

pd.Series(y_Osampled).value_counts()
sm = SMOTE(random_state=0)

X_SMOTE, y_SMOTE = sm.fit_resample(X, y)

pd.Series(y_SMOTE).value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)

perp_model = lm.Perceptron().fit(X_train_std,y_train)

y_pred = perp_model.predict(X_test_std)

print("Accuracy: ",round(accuracy_score(y_test, y_pred),2))
mat = confusion_matrix(y_test,y_pred,labels=['no','yes'])

print(mat)

y_test = label_binarize(y_test,classes=['no','yes'])

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sm = SMOTE(random_state=0)

X_SMOTE, y_SMOTE = sm.fit_resample(X_train, y_train)

sc = StandardScaler()

sc.fit(X_SMOTE)

X_train_std = sc.transform(X_SMOTE)

X_test_std = sc.transform(X_test)

perp_model = lm.Perceptron().fit(X_train_std,y_SMOTE)

y_pred = perp_model.predict(X_test_std)

print("Accuracy: ",round(accuracy_score(y_test, y_pred),2))

mat = confusion_matrix(y_test,y_pred)

print("Confusion Matrix: \n",mat)

y_test = label_binarize(y_test,classes=['no','yes'])

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

tree = DecisionTreeClassifier(criterion="entropy", max_depth=7)

model = tree.fit(X_train,y_train)

y_pred = model.predict(X_test)

y_test = label_binarize(y_test,classes=['no','yes'])

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

tree = DecisionTreeClassifier(criterion="entropy", max_depth=7)

X_SMOTE, y_SMOTE = sm.fit_resample(X_train, y_train)

model = tree.fit(X_SMOTE,y_SMOTE)

y_pred = model.predict(X_test)

y_test = label_binarize(y_test,classes=['no','yes'])

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

forest = RandomForestClassifier(n_estimators= 1000,criterion="gini", max_depth=5,min_samples_split = 0.4,min_samples_leaf=1, class_weight="balanced")

model = forest.fit(X_train,y_train)

y_pred = model.predict(X_test)

pd.Series(y_pred).value_counts()

y_test = label_binarize(y_test,classes=['no','yes'])

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

forest = RandomForestClassifier(n_estimators= 1000,criterion="gini", max_depth=5,min_samples_split = 0.4,min_samples_leaf=1, class_weight="balanced")

X_SMOTE, y_SMOTE = sm.fit_resample(X_train, y_train)

model = forest.fit(X_SMOTE,y_SMOTE)

y_pred = model.predict(X_test)

pd.Series(y_pred).value_counts()

y_test = label_binarize(y_test,classes=['no','yes'])

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = lm.LogisticRegression(random_state=0, solver='lbfgs',multi_class='auto',max_iter=1000).fit(X_train,y_train)

y_pred = model.predict_proba(X_test)

y_pred = y_pred[:,1]

y_test = label_binarize(y_test,classes=['no','yes'])

fpr_imb, tpr_imb, _ = roc_curve(y_test, y_pred)

roc_auc_imb = auc(fpr_imb, tpr_imb)

y_pred = model.predict(X_test)

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("Imbalanced -")

print("Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))

# Undersampled

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

rus = RandomUnderSampler(random_state=0)

X_Usampled, y_Usampled = rus.fit_resample(X_train, y_train)

model = lm.LogisticRegression(random_state=0, solver='lbfgs',multi_class='auto',max_iter=5000).fit(X_Usampled,y_Usampled)

y_pred = model.predict_proba(X_test)

y_pred = y_pred[:,1]

y_test = label_binarize(y_test,classes=['no','yes'])

fpr_us, tpr_us, _ = roc_curve(y_test, y_pred)

roc_auc_us = auc(fpr_us, tpr_us)

y_pred = model.predict(X_test)

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("Random undersampled -")

print("Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))

# Oversampled

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

ros = RandomOverSampler(random_state=0)

X_Osampled, y_Osampled = ros.fit_resample(X_train, y_train)

model = lm.LogisticRegression(random_state=0, solver='lbfgs',multi_class='auto',max_iter=5000).fit(X_Osampled, y_Osampled)

y_pred = model.predict_proba(X_test)

y_pred = y_pred[:,1]

y_test = label_binarize(y_test,classes=['no','yes'])

fpr_os, tpr_os, _ = roc_curve(y_test, y_pred)

roc_auc_os = auc(fpr_os, tpr_os)

y_pred = model.predict(X_test)

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("Random oversampled -")

print("Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))

# SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

sm = SMOTE(random_state=0)

X_SMOTE, y_SMOTE = sm.fit_resample(X_train, y_train)

model = lm.LogisticRegression(random_state=0, solver='lbfgs',multi_class='auto',max_iter=5000).fit(X_SMOTE,y_SMOTE)

y_pred = model.predict_proba(X_test)

y_pred = y_pred[:,1]

y_test = label_binarize(y_test,classes=['no','yes'])

fpr_smote, tpr_smote, _ = roc_curve(y_test, y_pred)

roc_auc_smote = auc(fpr_smote, tpr_smote)

y_pred = model.predict(X_test)

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("SMOTE -")

print("Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))
plt.figure()

lw = 2

plt.plot(fpr_imb, tpr_imb,

         label='Imbalanced data ROC curve (area = {0:0.4f})'

               ''.format(roc_auc_imb),

         color='deeppink', linestyle=':', linewidth=2)



plt.plot(fpr_us, tpr_us,

         label='Undersampled data ROC curve (area = {0:0.4f})'

               ''.format(roc_auc_us),

         color='blue', linestyle='--', linewidth=2)



plt.plot(fpr_os, tpr_os,

         label='Random Oversampled data ROC curve (area = {0:0.4f})'

               ''.format(roc_auc_os),

         color='darkred', linestyle='--', linewidth=2)



plt.plot(fpr_smote, tpr_smote,

         label='SMOTE data ROC curve (area = {0:0.4f})'

               ''.format(roc_auc_smote),

         color='darkgreen', linestyle='--', linewidth=2)



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.00])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
sm = SMOTE(random_state=0)

X_SMOTE, y_SMOTE = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_SMOTE, y_SMOTE, test_size=0.3)

svm = SVC(kernel='linear')

model = svm.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_test = label_binarize(y_test,classes=['no','yes'])

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("Linear kernel- ","Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))

fpr_linear, tpr_linear, _ = roc_curve(y_test, y_pred)

roc_auc_linear = auc(fpr_linear, tpr_linear)

sm = SMOTE(random_state=0)

X_SMOTE, y_SMOTE = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_SMOTE, y_SMOTE, test_size=0.3)

svm = SVC(kernel='rbf')

model = svm.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_test = label_binarize(y_test,classes=['no','yes'])

y_pred = label_binarize(y_pred,classes=['no','yes'])

print("Guassian kernel- ","Precision: ",round(precision_score(y_test,y_pred),2),"Recall: ",round(recall_score(y_test,y_pred),2))

fpr_rbf, tpr_rbf, _ = roc_curve(y_test, y_pred)

roc_auc_rbf = auc(fpr_rbf, tpr_rbf)
plt.figure()

lw = 2



plt.plot(fpr_linear, tpr_linear,

         label='Linear Kernel ROC curve (area = {0:0.4f})'

               ''.format(roc_auc_linear),

         color='darkred', linestyle='--', linewidth=2)



plt.plot(fpr_rbf, tpr_rbf,

         label='Gaussian Kernel ROC curve (area = {0:0.4f})'

               ''.format(roc_auc_rbf),

         color='darkgreen', linestyle='--', linewidth=2)



plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.00])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()