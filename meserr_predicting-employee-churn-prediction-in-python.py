# This is a review of the work at https://www.datacamp.com/community/tutorials/predicting-employee-churn-python. 
    
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("../input/HR_comma_sep.csv")
data.head(10)
data.tail()
data.info()
left = data.groupby('left')
left.mean()
data.describe()
left_count = data.groupby('left').count()
plt.bar(left_count.index.values, left_count['satisfaction_level'])
plt.xlabel('Employees Left Company')
plt.ylabel('Number of Employees')
plt.show()
data.left.value_counts()
num_projects = data.groupby('number_project').count()
plt.bar(num_projects.index.values, num_projects['satisfaction_level'])
plt.xlabel('Employees of Projects')
plt.ylabel('Number of Employees')
plt.show()
time_spent = data.groupby('time_spend_company').count()
plt.bar(time_spent.index.values, time_spent['average_montly_hours'])
plt.xlabel('Number of Years Spend Company')
plt.ylabel('Number of Employees')
plt.show()
features=['number_project','time_spend_company','Work_accident','left', 'promotion_last_5years','Departments ','salary']
fig=plt.subplots(figsize=(10,15))
for i,j in enumerate(features):
    plt.subplot(4,2, i+1)
    plt.subplots_adjust(hspace = 1.0)
    sns.countplot(x=j,data = data, hue='left')
    plt.xticks(rotation=90)
    plt.title("No. of employee")
from sklearn.cluster import KMeans
left_emp =  data[['satisfaction_level','last_evaluation']][data.left == 1]
kmeans = KMeans(n_clusters= 3, random_state = 0).fit(left_emp)
left_emp['label'] = kmeans.labels_
plt.scatter(left_emp['satisfaction_level'], left_emp['last_evaluation'], c=left_emp['label'], cmap = 'Accent')
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.title('3 Clusters of employees who left')
plt.show()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data['salary'] = le.fit_transform(data['salary'])
data['Departments '] = le.fit_transform(data['Departments '])

data.head()
X=data[['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'Departments ', 'salary']]
y=data['left']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, tree
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier


gb = GradientBoostingClassifier()
rf = RandomForestClassifier()
knn= KNeighborsClassifier()
tr = tree.DecisionTreeClassifier()
clf = Perceptron(fit_intercept=False, max_iter=10, tol=None,shuffle=False)
svm = SVC(gamma='auto')
nb = GaussianNB()
adb = AdaBoostClassifier(n_estimators=100, random_state=0)

clf.fit(X_train, y_train)
gb.fit(X_train, y_train)
rf.fit(X_train, y_train)
knn.fit(X_train, y_train)
tr.fit(X_train, y_train)
svm.fit(X_train, y_train)
nb.fit(X_train, y_train)
adb.fit(X_train, y_train)

y_predgb = gb.predict(X_test)
y_predrf = rf.predict(X_test)
y_predknn = knn.predict(X_test)
y_tree = tr.predict(X_test)
clf_pred = clf.predict(X_test)
svm_pred=svm.predict(X_test)
nb_pred = nb.predict(X_test)
adb_pred = adb.predict(X_test)
#adb_pred = adb.predict(X_test.values[1000].reshape(1, -1))
#adb_pred
from sklearn import metrics
print("Accuracy GradientBoostingClassifier: ", metrics.accuracy_score(y_test, y_predgb))
print("Accuracy RandomForestClassifier: ", metrics.accuracy_score(y_test, y_predrf))
print("Accuracy KNeighborsClassifier: ", metrics.accuracy_score(y_test, y_predknn))
print("Accuracy DecisionTreeClassifier: ", metrics.accuracy_score(y_test, y_tree))
print("Accuracy MLPClassifier: ", metrics.accuracy_score(y_test, clf_pred))
print("Accuracy SVM: ", metrics.accuracy_score(y_test, svm_pred))
print("Accuracy NB: ", metrics.accuracy_score(y_test, nb_pred))
print("Accuracy Ada: ", metrics.accuracy_score(y_test, adb_pred))


print("------------------------------------------------------")
print("Precision GradientBoostingClassifier: ", metrics.precision_score(y_test, y_predgb))
print("Precision RandomForestClassifier: ", metrics.precision_score(y_test, y_predrf))
print("Precision KNeighborsClassifier: ", metrics.precision_score(y_test, y_predknn))
print("------------------------------------------------------")
print("Recall GradientBoostingClassifier: ", metrics.recall_score(y_test, y_predgb))
print("Recall RandomForestClassifier: ", metrics.recall_score(y_test, y_predrf))
print("Recall KNeighborsClassifier: ", metrics.recall_score(y_test, y_predknn))

import seaborn as sns
from sklearn.metrics import confusion_matrix
Y_pred = tr.predict(X_test)
#Y_true = np.argmax(Y_pred,axis = 1) 
confusion_mtx = confusion_matrix(y_test, Y_pred) 
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
import pylab as pl
h = .02
Xnew = X.iloc[:, :2]
svm.fit(Xnew,y)
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
titles = ['Lineer SVM',
          'RBF SVM',
          'Polinom SVM',
         'SVM'
         ]
C = 1.0  # SVM regularization parameter

svc = SVC(kernel='linear', C=C).fit(Xnew, y)
rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(Xnew, y)
poly_svc = SVC(kernel='poly', degree=3, C=C).fit(Xnew, y)
svm_prediction = SVC(gamma='auto').fit(Xnew, y)
for i, clf in enumerate((svc, rbf_svc, poly_svc, svm_prediction)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    pl.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
    pl.axis('off')

    # Plot also the training points
    pl.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=pl.cm.Paired)

    pl.title(titles[i])

pl.show()
