# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization
import seaborn as sns #for eda process

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn import metrics
from sklearn import preprocessing

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cancer =pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
cancer.head(100)
cancer.isnull().sum()

cancer = cancer.drop(['Unnamed: 32'],axis=1)

cancer = cancer.drop(['id'],axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cancer.diagnosis    = le.fit_transform(cancer.diagnosis)
cancer.diagnosis.value_counts()
plt.figure(figsize=(5,5))
plt.title('Class Distribution')
cancer['diagnosis'].value_counts().plot(kind = 'pie',autopct = '%1.0f%%')
plt.show()
sns.countplot(cancer['diagnosis'],label="Count")
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Frequency')
# Correlation plot
cancer_num = cancer.select_dtypes(include=['int64','float64' ])
correlation = cancer.corr()

plt.figure(figsize=(20,10))
sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=correlation.columns,annot=True, annot_kws={"size": 8})
plt.figure(figsize=(12,5))
plt.title('Distribution of Loan Amount')
sns.distplot(cancer['diagnosis'], color='b')
plt.show()

plt.figure(figsize=(12,5))
plt.title('Distribution of concave points_mean')
sns.distplot(cancer['concave points_mean'], color='r')
plt.show()

plt.figure(figsize=(12,5))
plt.title('Distribution of perimeter_worst')
sns.distplot(cancer['perimeter_worst'], color='k')
plt.show()

plt.figure(figsize=(12,5))
plt.title('Distribution of concave points_worst')
sns.distplot(cancer['concave points_worst'], color='k')
plt.show()


cancer.head()
cancer.shape
cancer_x = cancer.iloc[:,1:31] 
cancer_x.head()
cancer_y = cancer.iloc[:,0]
cancer_y.head()
import sklearn 
from sklearn.model_selection import train_test_split  #for spliiting the data


x_train, x_test, y_train, y_test = train_test_split(cancer_x, cancer_y, test_size=0.33, random_state=42)

print(x_train.shape, y_train.shape)

print(x_test.shape , y_test.shape)
from sklearn.linear_model import LogisticRegression #To build logistic regression
log = LogisticRegression()
log.fit(x_train,y_train)
#Prediction for model
pred = log.predict(x_test)
pred
confusionmatrix_log = pd.crosstab(pred, y_test, rownames=['predicted'], colnames= ['Actual'], margins=False)
confusionmatrix_log
tab_log = confusion_matrix(pred, y_test)
print(tab_log)
print('\n')
print(classification_report(pred, y_test))

print('\n')
accuracy_log = tab_log.diagonal().sum() / tab_log.sum()*100
accuracy_log
from sklearn.svm import LinearSVC
svm_model = LinearSVC()
svm_model.fit(x_train,y_train)
predict_svm = svm_model.predict(x_test)
predict_svm
tab_svm = confusion_matrix(predict_svm, y_test)
print(tab_svm)

print('\n')
    
print(classification_report(predict_svm, y_test))

accuracy_svm = tab_svm.diagonal().sum() / tab_svm.sum()*100
accuracy_svm

from sklearn.ensemble import RandomForestClassifier  #To build Random_Forest
 
rf = RandomForestClassifier(n_estimators=500)
rf.fit(x_train,y_train)

pred_rf = rf.predict(x_test)
pred_rf

tab_rf = confusion_matrix(pred_rf, y_test)
print(tab_rf)
print('\n')
    
print(classification_report(pred_rf, y_test))
accuracy_rf = tab_rf.diagonal().sum() / tab_rf.sum()*100
accuracy_rf
rf.feature_importances_
rf_feature_score = pd.DataFrame({'Importance':rf.feature_importances_ ,'Variables_Name':x_train.columns})
rf_feature_score

rf_feature_score.sort_values(['Importance'],ascending = False)

from sklearn.tree import DecisionTreeClassifier  #To build Decision_Tree
dtree = DecisionTreeClassifier()


dtree.fit(x_train,y_train)


pred_dtree = dtree.predict(x_test)
pred_dtree
tab_dtree = confusion_matrix(pred_dtree,y_test)

print(tab_dtree)


print(classification_report(pred_dtree,y_test))

accuracy_dtree = tab_dtree.diagonal().sum() / tab_dtree.sum()*100
accuracy_dtree
dtree_feature_score = pd.DataFrame({'Importance':dtree.feature_importances_ ,'Variables_Name':x_train.columns})
dtree_feature_score
dtree_feature_score.sort_values(['Importance'],ascending = False)
from sklearn.naive_bayes import MultinomialNB   #To build navie_bayes

naive_bay = MultinomialNB()

naive_bay.fit(x_train,y_train)

predictions_nb = naive_bay.predict(x_test)
predictions_nb
tab_nb = confusion_matrix(predictions_nb,y_test)

print(tab_nb)


print(classification_report(predictions_nb,y_test))

accuracy_nb = tab_nb.diagonal().sum() / tab_nb.sum()*100
accuracy_nb

accuracies = ({"Logistic Regression":accuracy_log,"Decision Tree":accuracy_dtree,"Random Forest":accuracy_rf,"Naive Bayes":accuracy_nb,'SVM':accuracy_svm})
print(accuracies)
colors = ["purple", "green", "orange", "magenta","#CFC60E"]
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()

plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(tab_log,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(tab_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(tab_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,4)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(tab_dtree,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,5)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(tab_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

