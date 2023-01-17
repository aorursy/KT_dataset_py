import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

pd.options.display.max_columns = 999

data = pd.read_csv('../input/data.csv')
data.head()
data.shape
data.id.nunique()
data.columns
data['Unnamed: 32'].isnull().all()
data.drop(['Unnamed: 32'],axis = 1,inplace = True)
data.shape
data.info()
data.isnull().any()
(data['diagnosis'].value_counts())/data.shape[0]
labels='Benign','Malign'

plt.pie(data['diagnosis'].value_counts(), labels=labels,

        autopct='%1.1f%%', shadow=True, startangle=140)
map_cat = {'B':0,'M':1}
data['diagnosis'] = data['diagnosis'].map(map_cat)
data.drop(['id'],axis = 1,inplace = True)
X = data.iloc[:,1:]
y = data.iloc[:,0]
y.name
X.columns
corr = data.corr()
#sns.heatmap(corr)

plt.figure(figsize = (10,7))

top_corr_features = corr.index[abs(corr["diagnosis"])>0.7]

sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state = 0)
X_train.shape[0],X_test.shape[0]
from sklearn.preprocessing import StandardScaler

std = StandardScaler()

std.fit(X_train)

X_train = std.transform(X_train)
X_train
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train,y_train)
X_test = std.transform(X_test)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'

     .format(clf.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format(clf.score(X_test, y_test)))
from sklearn.model_selection import GridSearchCV

grid_values = {'C':[0.01, 0.1, 1, 10, 100],'penalty': ['l1', 'l2']}

grid_clf_acc = GridSearchCV(clf, param_grid = grid_values)

grid_clf_acc.fit(X_train,y_train)

print('Accuracy of Logistic regression classifier on training set: {:.2f}'

     .format(grid_clf_acc.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format(grid_clf_acc.score(X_test, y_test)))
grid_clf_acc.cv_results_
from sklearn.metrics import confusion_matrix

y_predicted = clf.predict(X_test)

confusion = confusion_matrix(y_test, y_predicted)

confusion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy = TP + TN / (TP + TN + FP + FN)

# Precision = TP / (TP + FP)

# Recall = TP / (TP + FN)  Also known as sensitivity, or True Positive Rate

# F1 = 2 * Precision * Recall / (Precision + Recall) 

print('Accuracy: {:.2f}'.format(accuracy_score(y_test, y_predicted)))

print('Precision: {:.2f}'.format(precision_score(y_test, y_predicted)))

print('Recall: {:.2f}'.format(recall_score(y_test, y_predicted)))

print('F1: {:.2f}'.format(f1_score(y_test, y_predicted)))
y_proba_lr = clf.predict_proba(X_test)
y_proba_lr
X_test.shape
from sklearn.metrics import roc_curve, auc

y_score_lr = clf.decision_function(X_test)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_score_lr)

roc_auc_lr = auc(fpr_lr, tpr_lr)



plt.figure()

plt.xlim([-0.01, 1.00])

plt.ylim([-0.01, 1.01])

plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC curve', fontsize=16)

plt.legend(loc='lower right', fontsize=13)

plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

plt.axes().set_aspect('equal')

plt.show()
from sklearn.tree import DecisionTreeClassifier



clf2 = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(clf2.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(clf2.score(X_test, y_test)))
print('Feature importances: {}'.format(clf2.feature_importances_))
print('Feature importances: {}'.format(X.columns.tolist()))
df = pd.DataFrame(data = clf2.feature_importances_,index = X.columns.tolist())



df = df[df.iloc[:,0] > 0].sort_values(by = 0,ascending = False)



sns.barplot(y = df.index, x= df[0])
from sklearn.ensemble import RandomForestClassifier



clf1 = RandomForestClassifier(max_features = 15, random_state = 0)

clf1.fit(X_train, y_train)



print('Breast cancer dataset')

print('Accuracy of RF classifier on training set: {:.2f}'

     .format(clf1.score(X_train, y_train)))

print('Accuracy of RF classifier on test set: {:.2f}'

     .format(clf1.score(X_test, y_test)))
df = pd.DataFrame(data = clf1.feature_importances_,index = X.columns.tolist())



df = df[df.iloc[:,0] > 0].sort_values(by = 0,ascending = False)



sns.barplot(y = df.index, x= df[0])
from sklearn.ensemble import GradientBoostingClassifier





clf2 = GradientBoostingClassifier(random_state = 0)

clf2.fit(X_train, y_train)



print('Breast cancer dataset (learning_rate=0.1, max_depth=3)')

print('Accuracy of GBDT classifier on training set: {:.2f}'

     .format(clf2.score(X_train, y_train)))

print('Accuracy of GBDT classifier on test set: {:.2f}\n'

     .format(clf2.score(X_test, y_test)))



clf2 = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 2, random_state = 0)

clf2.fit(X_train, y_train)



print('Breast cancer dataset (learning_rate=0.01, max_depth=2)')

print('Accuracy of GBDT classifier on training set: {:.2f}'

     .format(clf2.score(X_train, y_train)))

print('Accuracy of GBDT classifier on test set: {:.2f}'

     .format(clf2.score(X_test, y_test)))
df = pd.DataFrame(data = clf2.feature_importances_,index = X.columns.tolist())



df = df[df.iloc[:,0] > 0].sort_values(by = 0,ascending = False)



sns.barplot(y = df.index, x= df[0])