# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# reading the data
data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
data.head()
# checking for missing values
data.isnull().sum()
# checking the data types
data.dtypes
# descriptive statistics
data.describe()
# Checking number of missing values for these columns
columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for i in columns:
    print("missing values in",i," :",(data[i] == 0).sum())
# Checking outliers and distribution of these columns for imputation
for i in columns:
    fig,axes = plt.subplots(1,2,figsize=(10,5))
    sns.boxplot(x=data[i],orient = 'v',ax = axes[0])
    sns.distplot(data[i],ax = axes[1])
    fig.tight_layout()
    
# imputing missing values i.e. '0'
data['Glucose'].replace(0,data['Glucose'].mean(),inplace = True)
data['BloodPressure'].replace(0,data['BloodPressure'].mean(),inplace = True)
data['BMI'].replace(0,data['BMI'].mean(),inplace = True)
data['SkinThickness'].replace(0,data['SkinThickness'].median(),inplace = True)
data['Insulin'].replace(0,data['Insulin'].median(),inplace = True)
data.describe()
fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=data, width= 0.5,ax=ax,  fliersize=3)
for i in data.columns:
    plt.figure(figsize = (5,5))
    sns.distplot(data[i])
# removing outliers
data = data[data['SkinThickness']<80]
data = data[data['Insulin']<580]
data = data[data['BMI']<60]
data.shape
# Looking at the distribution again 
for i in data.columns:
    plt.figure(figsize = (5,5))
    sns.distplot(data[i])
# separating dependent and independent features
X = data.drop("Outcome",axis=1)
y = data['Outcome']
# heatmap for checking correlation
corr_matrix = data.corr()
sns.heatmap(corr_matrix)
# scaling the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled,columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
# splitting training and test data (80:20) ratio
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size = 0.20,random_state = 30)
from sklearn.svm import SVC
clf_svm1 = SVC()
clf_svm1.fit(X_train,y_train)
y_pred = clf_svm1.predict(X_test)
# accuracy of SVC
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
accuracy = accuracy_score(y_test,y_pred)
print("SVC Accuracy:",accuracy)
# plotting the confusion matrix 
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf_svm1,X_test,y_test,values_format = 'd',display_labels = ['diabetic','not diabetic'])
# selecting different parameters to use for improving the SVC Accuracy
param_grid = [
    {'C' : [0.5,1,10,100],
     'gamma' : ['scale','auto',1,0.1,0.01,0.001,0.0001],
    'kernel' : ['linear','poly','rbf']}
]
# Hyperparameter optimisation
from sklearn.model_selection import GridSearchCV

optimal_params = GridSearchCV(SVC(),param_grid,cv = 5,scoring = 'accuracy',verbose = 0)
optimal_params.fit(X_train,y_train)
print(optimal_params.best_params_)
clf_svm2 = SVC(C = 100, gamma = 0.0001,probability = True)
clf_svm2.fit(X_train,y_train)
svm_y_pred = clf_svm2.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
accuracy = accuracy_score(y_test,svm_y_pred)
print("SVC Accuracy score:",accuracy)
from sklearn.metrics import recall_score,precision_score,f1_score
recall = recall_score(y_test,svm_y_pred)
precision = precision_score(y_test,svm_y_pred)
f1 = f1_score(y_test,svm_y_pred)
print("SVC Recall:",recall)
print("SVC Precision:",precision)
print("SVC F1:",f1)
# printing the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,svm_y_pred))
# plotting confusion matrix again
plot_confusion_matrix(clf_svm2,X_test,y_test,values_format = 'd',display_labels = ['diabetic','not diabetic'])
from sklearn.linear_model import LogisticRegression
lr_clf1 = LogisticRegression()
lr_clf1.fit(X_train,y_train)
lr_y_pred = lr_clf1.predict(X_test)
accuracy_lr = accuracy_score(y_test,lr_y_pred)
print("Logistic Regression Accuracy:",accuracy_lr)
# Plotting confusion matrix
plot_confusion_matrix(lr_clf1,X_test,y_test,values_format = 'd',display_labels = ['diabetic','not diabetic'])
# # Hyperparameter optimisation
from sklearn.model_selection import GridSearchCV
grid_values = {'penalty': ['l2'], 'C': [0.001,0.01,0.1,1,10,100,1000]}
lr_optimal_params = GridSearchCV(LogisticRegression(),grid_values,cv =5,verbose = 0)
lr_optimal_params.fit(X_train,y_train)

print(lr_optimal_params.best_params_)
lr_clf2 = LogisticRegression(C = 1, penalty = 'l2')
lr_clf2.fit(X_train,y_train)
lr_y_pred = lr_clf2.predict(X_test)
accuracy = accuracy_score(y_test,lr_y_pred)
print("Accuracy score:",accuracy)
plot_confusion_matrix(lr_clf2,X_test,y_test,values_format = 'd',display_labels = ['diabetic','not diabetic'])
from sklearn.metrics import recall_score,precision_score,f1_score
recall = recall_score(y_test,lr_y_pred)
precision = precision_score(y_test,lr_y_pred)
f1 = f1_score(y_test,lr_y_pred)
print("LR Recall:",recall)
print("LR Precision:",precision)
print("LR F1:",f1)
# printing the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,lr_y_pred))
# applying random forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state = 32)
model = rf_clf.fit(X_train,y_train)
rf_y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,rf_y_pred)
print("Accuracy score:",accuracy)
params = [{
    'n_estimators' : [10,20,50,100,200,300,400],
    'criterion' : ['gini','entropy'],
    'max_leaf_nodes' : range(8,32)
}]
optimal_params = GridSearchCV(RandomForestClassifier(random_state = 32),params,cv =5,verbose = 0)
optimal_params.fit(X_train,y_train)
print(optimal_params.best_params_)
rf_clf = RandomForestClassifier(n_estimators=10,max_leaf_nodes = 29,criterion = 'entropy',random_state = 32 )
model = rf_clf.fit(X_train,y_train)
rf_y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,rf_y_pred)
print("Accuracy score:",accuracy)
plot_confusion_matrix(rf_clf,X_test,y_test,values_format = 'd',display_labels = ['diabetic','not diabetic'])
from sklearn.metrics import recall_score,precision_score,f1_score
recall = recall_score(y_test,rf_y_pred)
precision = precision_score(y_test,rf_y_pred)
f1 = f1_score(y_test,rf_y_pred)
print("RF Recall:",recall)
print("RF Precision:",precision)
print("RF F1:",f1)
# printing the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,rf_y_pred))
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train,y_train)
gb_y_pred = gb_clf.predict(X_test)
accuracy = accuracy_score(y_test,gb_y_pred)
print("Accuracy score:",accuracy)
gb_params = [{
    'learning_rate' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'criterion' : ['friedman_mse', 'mse', 'mae'],
    'min_samples_leaf' : range(1,6)
}]
gb_optimal_params = GridSearchCV(GradientBoostingClassifier(),gb_params,cv =5,verbose = 0)
gb_optimal_params.fit(X_train,y_train)

print(gb_optimal_params.best_params_)
gb_clf = GradientBoostingClassifier(criterion = 'mse',learning_rate = 0.1,min_samples_leaf = 4 )
gb_clf.fit(X_train,y_train)
gb_y_pred = gb_clf.predict(X_test)
accuracy = accuracy_score(y_test,gb_y_pred)
print("Accuracy score:",accuracy)
plot_confusion_matrix(gb_clf,X_test,y_test,values_format = 'd',display_labels = ['diabetic','not diabetic'])
from sklearn.metrics import recall_score,precision_score,f1_score
recall = recall_score(y_test,gb_y_pred)
precision = precision_score(y_test,gb_y_pred)
f1 = f1_score(y_test,gb_y_pred)
print("GB Recall:",recall)
print("GB Precision:",precision)
print("GB F1:",f1)
# printing the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,gb_y_pred))
# Applying Knn and finding the best value of 'k'
from sklearn.neighbors import KNeighborsClassifier

test_scores = []
train_scores = []
accuracy_max = 0
k_max = 0
for i in range(1,30):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    
    train_accuracy = knn.score(X_train,y_train)
    train_scores.append(train_accuracy)
    test_accuracy = accuracy_score(y_test,y_pred)
    test_scores.append(test_accuracy)
    if test_accuracy > accuracy_max:
        accuracy_max = test_accuracy
        k_max = i
print("maximum_test_accuracy:",accuracy_max, "is achieved at k:",k_max)
knn1 = KNeighborsClassifier(7)
knn1.fit(X_train,y_train)
knn_y_pred = knn1.predict(X_test)
plot_confusion_matrix(knn1,X_test,y_test,values_format = 'd',display_labels = ['diabetic','not diabetic'])
# Hyperparameter optimisation
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors':np.arange(1,50)}
knn2 = KNeighborsClassifier()
knn_cv= GridSearchCV(knn2,param_grid,cv =5,verbose = 0)
knn_cv.fit(X_train,y_train)

print("Best Parameters: " + str(knn_cv.best_params_))
knn2 = KNeighborsClassifier(n_neighbors = 17)
knn2.fit(X_train,y_train)
knn_y_pred = knn2.predict(X_test)
acc = accuracy_score(y_test,knn_y_pred)
acc
from sklearn.metrics import recall_score,precision_score,f1_score
recall = recall_score(y_test,knn_y_pred)
precision = precision_score(y_test,knn_y_pred)
f1 = f1_score(y_test,knn_y_pred)
print("KNN Recall:",recall)
print("KNN Precision:",precision)
print("KNN F1:",f1)
# plotting training and test scores of KNN model
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,30),train_scores,marker='*',label='Train Score')
p = sns.lineplot(range(1,30),test_scores,marker='o',label='Test Score')
svc_y_pred_proba = clf_svm2.predict_proba(X_test)[:,1]
lr_y_pred_proba = lr_clf2.predict_proba(X_test)[:,1]
rf_y_pred_proba = rf_clf.predict_proba(X_test)[:,1]
gb_y_pred_proba = gb_clf.predict_proba(X_test)[:,1]
knn_y_pred_proba = knn1.predict_proba(X_test)[:,1]
from sklearn.metrics import roc_curve
fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_test, svc_y_pred_proba)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, lr_y_pred_proba)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, rf_y_pred_proba)
fpr_gb, tpr_gb, thresholds_gb = roc_curve(y_test, gb_y_pred_proba)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, knn_y_pred_proba)
# Plotting the ROC Curve
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr_svc,tpr_svc, label='SVC')
plt.plot(fpr_knn,tpr_knn, label='KNN')
plt.plot(fpr_rf,tpr_rf, label='RF')
plt.plot(fpr_gb,tpr_gb, label='GB')
plt.plot(fpr_lr,tpr_lr, label='LR')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.title('ROC curve')
plt.show()
#Area under ROC curve
from sklearn.metrics import roc_auc_score
knn_auc = roc_auc_score(y_test,knn_y_pred_proba)
lr_auc = roc_auc_score(y_test,lr_y_pred_proba)
svc_auc = roc_auc_score(y_test,svc_y_pred_proba)
rf_auc = roc_auc_score(y_test,rf_y_pred_proba)
gb_auc = roc_auc_score(y_test,gb_y_pred_proba)
print("Area under KNN ROC curve:",knn_auc)
print("Area under LR ROC curve:",lr_auc)
print("Area under SVC ROC curve:",svc_auc)
print("Area under RF ROC curve:",rf_auc)
print("Area under GB ROC curve:",gb_auc)
