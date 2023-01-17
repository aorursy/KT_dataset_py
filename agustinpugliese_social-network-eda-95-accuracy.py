import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sns.set()

%matplotlib inline
dataset = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')
dataset.head()
dataset.describe().T
print(pd.isnull(dataset).sum())
dataset['Gender'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(6,6))
plt.title('Gender percentages', fontsize = 20)
plt.tight_layout()
plt.show()
sns.countplot(dataset['Gender'], palette = 'Set2')
plt.title ('Gender vs. quantity', fontsize = 20)
plt.show()
sns.distplot(dataset['Age'], bins = 5, color = 'orange', label = 'KDE')
plt.legend()
plt.gcf().set_size_inches(12, 5)
plt.figure(figsize = (22,10))
sns.countplot(x = 'Age',data = dataset , hue='Gender', palette = 'Set2')
plt.legend(loc='upper center')
plt.show()
tag1 = 'Male'
tag2 = 'Female'
Male = dataset[dataset["Gender"] == tag1][['Age','EstimatedSalary']]
Female = dataset[dataset["Gender"] == tag2][['Age','EstimatedSalary']]
f, (ax1, ax2) = plt.subplots(1, 2, sharey = True)

ax1.scatter(Male.Age, Male.EstimatedSalary, c = 'green', s = 15, alpha = 0.7)
ax1.set_title('Male age vs. Estimated Salary', c = 'green')
ax2.scatter(Female.Age, Female.EstimatedSalary, c='red', s = 15, alpha = 0.7)
ax2.set_title('Female age vs. Estimated Salary', c ='red')
plt.gcf().set_size_inches(15, 7)

plt.ylabel('Estimated Salary', fontsize = 20)

plt.show()
sns.catplot(x="Age", col = 'Purchased', data=dataset, kind = 'count', palette='pastel')
plt.gcf().set_size_inches(20, 10)
plt.show()
sns.catplot(x="Gender", col = 'Purchased', data=dataset, kind = 'count', palette='pastel')
plt.show()
dataset2 = dataset.copy()
dataset2 = dataset2.drop(['User ID'], axis = 1)
X = dataset2.iloc[:, 0:3]
y = dataset2.iloc[:, -1]
X = pd.get_dummies(X)
X = X[['Gender_Male','Gender_Female','Age','EstimatedSalary']]
X = X.drop(['Gender_Male'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# sc_X = MinMaxScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ['{0:0.0f}'.format(value) for value in
                cm.flatten()]
group_percentages = ['{0:.2%}'.format(value) for value in
                     cm.flatten()/np.sum(cm)]

labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm, annot = labels, fmt = '', cmap = 'Blues', cbar = False)
plt.gcf().set_size_inches(5, 5)
plt.title('Confusion Matrix Logistic Regression', fontsize = 20)
plt.show()
accuracy_LR = accuracy_score(y_test,y_pred) *100
print('The accuracy of the logistic regression is: ' +str(accuracy_LR) + ' %.')
parameters_LR = classifier.coef_
parameters_LR
def clf_model(model):
    clf = model
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test).round())
    return clf, accuracy
model_performance = pd.DataFrame(columns = ["Model", "Accuracy"])

models_to_evaluate = [RandomForestClassifier(n_estimators=1000), KNeighborsClassifier(n_neighbors = 7, metric = "minkowski", p = 2),
                      SVC(kernel = 'rbf'), GaussianNB(), GradientBoostingRegressor(n_estimators=300, learning_rate=0.01), 
                     AdaBoostClassifier(n_estimators=300, learning_rate=0.01), XGBClassifier(n_estimators=300, learning_rate=0.01)]

for model in models_to_evaluate:
    clf, accuracy = clf_model(model)
    model_performance = model_performance.append({"Model": model, "Accuracy": accuracy}, ignore_index=True)

model_performance
XGclassifier = XGBClassifier(n_estimators=300, learning_rate=0.01)
XGclassifier.fit(X_train, y_train)
y_pred_xg = XGclassifier.predict(X_test)
KNN = KNeighborsClassifier(n_neighbors = 7, metric = "minkowski", p = 2)
KNN.fit(X_train, y_train)
y_pred_knn = KNN.predict(X_test)
SVC_clf = SVC(kernel = 'rbf')
SVC_clf.fit(X_train, y_train)
y_pred_SVC = SVC_clf.predict(X_test)
# length of the test data 
total = len(y_test) 
  
# Counting '1' labels in test data 
one_count = np.sum(y_test) 
  
# counting '0' lables in test data  
zero_count = total - one_count 

plt.figure(figsize = (10, 6)) 
  
# x-axis ranges from 0 to total people on y_test  
# y-axis ranges from 0 to the total positive outcomes. 

# K-NN plot

K = [y for _, y in sorted(zip(y_pred_knn, y_test), reverse = True)] 

x = np.arange(0, total + 1) # Shape of Y_test
y = np.append([0], np.cumsum(K)) # Y values

plt.plot(x, y, c = 'green', label = 'K-NN', linewidth = 2)

# SVC Plot

S = [y for _, y in sorted(zip(y_pred_SVC, y_test), reverse = True)] 

x2 = np.arange(0, total + 1) # Shape of Y_test
y2 = np.append([0], np.cumsum(S)) # Y values

plt.plot(x2, y2, c = 'orange', label = 'SVC', linewidth = 2)


# XGClassifier plot 

XG = [y for _, y in sorted(zip(y_pred_xg, y_test), reverse = True)] 

x3 = np.arange(0, total + 1) # Shape of Y_test
y3 = np.append([0], np.cumsum(XG)) # Y values

plt.plot(x3, y3, c = 'red', label = 'XGClassifier', linewidth = 2)


# Random Model plot
  
plt.plot([0, total], [0, one_count], c = 'blue',  
         linestyle = '--', label = 'Random Model') 

# Perfect model plot

plt.plot([0, one_count, total], [0, one_count, one_count], 
         c = 'grey', linewidth = 2, label = 'Perfect Model') 

plt.title('Cumulative Accuracy Profile of different models', fontsize = 20)
plt.xlabel('Total y_test observations', fontsize = 15)
plt.ylabel('N° class 1 scores', fontsize = 15)
plt.legend() 
plt.show()
# Area under Random Model
a = auc([0, total], [0, one_count])

# Area between Perfect and Random Model
aP = auc([0, one_count, total], [0, one_count, one_count]) - a

# Area K-NN

aKNN = auc(x, y) - a
print("Accuracy Rate for K-NN: {}".format(aKNN / aP))

# Area SVC

aSVC = auc(x2, y2) - a
print("Accuracy Rate for Support Vector Classifier: {}".format(aSVC / aP))

# Area XGClassifier

aXG = auc(x3, y3) - a
print("Accuracy Rate for XGClassifier: {}".format(aXG / aP))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_knn)

fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred_SVC)

fpr3, tpr3, thresholds3 = roc_curve(y_test, y_pred_xg)

roc_auc = auc(fpr, tpr)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)

plt.figure(figsize = (10, 6)) 

plt.plot(fpr, tpr, c = 'green', linewidth = 2, label = 'K-NN AUC:' + ' {0:.2f}'.format(roc_auc))
plt.plot(fpr2, tpr2, c = 'orange', linewidth = 2, label = 'Support Vector Classifier AUC:' + ' {0:.2f}'.format(roc_auc2))
plt.plot(fpr3, tpr3, c = 'red', linewidth = 2, label = 'XGClassifier AUC:' + ' {0:.2f}'.format(roc_auc3))
plt.plot([0,1], [0,1], c = 'blue', linestyle = '--')

plt.xlabel('False Positive Rate', fontsize = 15)
plt.ylabel('True Positive Rate', fontsize = 15)
plt.title('ROC', fontsize = 20)
plt.legend(loc = 'lower right', fontsize = 13)
plt.show()