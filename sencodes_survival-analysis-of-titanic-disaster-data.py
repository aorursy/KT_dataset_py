# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Load training data.... 
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# df = train_df.append(test_df, ignore_index = True)
train_df.shape
# Check NaNs in training dataset....
train_df.isnull().sum()
# Let's check the percentage of NaNs 
print('The percentage of missing value in Age column is %.2f%%' % (100*train_df.isnull().sum()[5]/len(train_df)))
print('The percentage of missing value in Cabin column is %.2f%%' % (100*train_df.isnull().sum()[10]/len(train_df)))
# Process NaN value... 
print(len(train_df['Cabin'].unique()))
train_df.drop('Cabin', axis = 1, inplace = True)
test_df.drop('Cabin', axis = 1, inplace = True)
train_df
import seaborn as sns
sns.distplot(train_df['Age'].dropna(), hist = True, kde = True)
train_df['Age'].replace(np.nan, np.nanmedian(train_df['Age']), inplace = True)
test_df['Age'].replace(np.nan, np.nanmedian(test_df['Age']), inplace = True)

train_df
print(train_df['Embarked'].value_counts())
sns.countplot(x = 'Embarked', data = train_df)
train_df['Embarked'].replace(np.nan, 'S', inplace = True)
test_df['Embarked'].replace(np.nan, 'S', inplace = True)

train_df.isnull().sum() # Double check if there's any more missing value in the dataset... 
# test_df.isnull().sum()
training_df = train_df.copy()

training_df['Sex'].replace(['male', 'female'], [1,0], inplace = True)

test_df['Sex'].replace(['male', 'female'], [1,0], inplace = True)

training_df
training_df.drop(['Name', 'Ticket', 'PassengerId'], axis = 1, inplace = True)
test_df.drop(['Name', 'Ticket', 'PassengerId'], axis = 1, inplace = True)
training_df.head()
training_df['relatives'] = training_df['SibSp'] + training_df['Parch']
training_df.drop(['SibSp', 'Parch'], axis = 1, inplace = True)

test_df['relatives'] = test_df['SibSp'] + test_df['Parch']
test_df.drop(['SibSp', 'Parch'], axis = 1, inplace = True)

training_df.head()
# import matplotlib.pyplot as plt 
sns.barplot(x = 'Sex', y = 'Survived', data = train_df)
ex = sns.barplot(x = 'Pclass', y = 'Survived', data = training_df)
ex.set(xticklabels = ['1-Upper', '2-Middle', '3-Lower'])
import matplotlib.pyplot as plt
plt.figure(figsize = (12,6))
ax = sns.kdeplot(training_df['Fare'][training_df['Survived'] == 1], shade = True)
sns.kdeplot(training_df['Fare'][training_df['Survived'] == 0], shade = True)
plt.xlim(0.01, 200)
ax.set(xlabel = 'Fare')
training_df['Fare_Level'] = pd.cut(training_df['Fare'], 
                                   [0,30,80,max(training_df['Fare'])],
                                   labels = ['cheap', 'medium', 'expensive'])

test_df['Fare_Level'] = pd.cut(test_df['Fare'], 
                                   [0,30,80,max(test_df['Fare'])],
                                   labels = ['cheap', 'medium', 'expensive'])
training_df.head()
plt.figure(figsize = (12,6))
ax = sns.kdeplot(training_df['Age'][training_df['Survived'] == 1], shade = True)
sns.kdeplot(training_df['Age'][training_df['Survived'] == 0], shade = True)
plt.xlim(0.01, 120)
ax.set(xlabel = 'Age')
training_df['Age_Level'] = pd.cut(training_df['Age'], 
                                   [0,18,50,70,max(training_df['Age'])],
                                   labels = ['child','adult','old','70+'])
training_df.drop(['Age','Fare'], axis = 1, inplace = True)


test_df['Age_Level'] = pd.cut(test_df['Age'], 
                                   [0,18,50,70,max(test_df['Age'])],
                                   labels = ['child','adult','old','70+'])
test_df.drop(['Age','Fare'], axis = 1, inplace = True)


training_df.head(10)
training_df['Cherbourg'] = training_df['Embarked'].apply(lambda x: 1 if x == 'C' else 0)
training_df['Queenstown'] = training_df['Embarked'].apply(lambda x: 1 if x =='Q' else 0)
training_df['Southampton'] = training_df['Embarked'].apply(lambda x: 1 if x =='S' else 0)
training_df.drop('Embarked', axis = 1, inplace = True)


test_df['Cherbourg'] = test_df['Embarked'].apply(lambda x: 1 if x == 'C' else 0)
test_df['Queenstown'] = test_df['Embarked'].apply(lambda x: 1 if x =='Q' else 0)
test_df['Southampton'] = test_df['Embarked'].apply(lambda x: 1 if x =='S' else 0)
test_df.drop('Embarked', axis = 1, inplace = True)


training_df.head()
training_df['Fare_cheap'] = training_df['Fare_Level'].apply(lambda x: 1 if x =='cheap' else 0)
training_df['Fare_medium'] = training_df['Fare_Level'].apply(lambda x: 1 if x =='medium' else 0)
training_df['Fare_expensive'] = training_df['Fare_Level'].apply(lambda x: 1 if x =='expensive' else 0)
training_df.drop('Fare_Level', axis = 1, inplace = True)

test_df['Fare_cheap'] = test_df['Fare_Level'].apply(lambda x: 1 if x =='cheap' else 0)
test_df['Fare_medium'] = test_df['Fare_Level'].apply(lambda x: 1 if x =='medium' else 0)
test_df['Fare_expensive'] = test_df['Fare_Level'].apply(lambda x: 1 if x =='expensive' else 0)
test_df.drop('Fare_Level', axis = 1, inplace = True)

training_df.head()
training_df['Age_child'] = training_df['Age_Level'].apply(lambda x: 1 if x =='child' else 0)
training_df['Age_adult'] = training_df['Age_Level'].apply(lambda x: 1 if x =='adult' else 0)
training_df['Age_old'] = training_df['Age_Level'].apply(lambda x: 1 if x =='old' else 0)
training_df['Age_70+'] = training_df['Age_Level'].apply(lambda x: 1 if x =='70+' else 0)
training_df.drop('Age_Level', axis = 1, inplace = True)


test_df['Age_child'] = test_df['Age_Level'].apply(lambda x: 1 if x =='child' else 0)
test_df['Age_adult'] = test_df['Age_Level'].apply(lambda x: 1 if x =='adult' else 0)
test_df['Age_old'] = test_df['Age_Level'].apply(lambda x: 1 if x =='old' else 0)
test_df['Age_70+'] = test_df['Age_Level'].apply(lambda x: 1 if x =='70+' else 0)
test_df.drop('Age_Level', axis = 1, inplace = True)

training_df
X_train = training_df.iloc[:,1:]
y_train = training_df.iloc[:,0]
X_train.head()
# Let me leave the hold-out data first... 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics, svm

import matplotlib.pyplot as plt

X_training, X_testing, y_training, y_testing = train_test_split(X_train, y_train,
                                                                test_size = 0.2, random_state = 2)

knn_clf = KNeighborsClassifier(n_neighbors = 7)
knn_clf.fit(X_training, y_training)
print('K-Nearest Neighbors Classifier Accuracy for training data is: {:.3f}'.format(accuracy_score(y_training, knn_clf.predict(X_training))))
print('K-Nearest Neighbors Classifier Accuracy for testing data is: {:.3f}'.format(accuracy_score(y_testing, knn_clf.predict(X_testing))))

# ROC curve and AUC
y_score_knn = knn_clf.predict_proba(X_testing)
fpr_knn, tpr_knn, thresholds_knn = metrics.roc_curve(y_testing, y_score_knn[:,1])
print('K-Nearest Neighbors Classifier AUC is: {:.3f}'.format(metrics.roc_auc_score(y_testing, y_score_knn[:,1])))

log_reg = LogisticRegression()
log_reg.fit(X_training, y_training)
y_score_lr = log_reg.decision_function(X_testing)
print('Logistic Regression Accuracy for training data is: {:.3f}'.format(log_reg.score(X_training, y_training)))
print('Logistic Regression Accuracy for testing data is: {:.3f}'.format(log_reg.score(X_testing, y_testing)))
print('Logistic Regression AUC is: {:.3f}'.format(metrics.roc_auc_score(y_testing, y_score_lr)))

fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(y_testing, y_score_lr)
svm_clf = svm.SVC(C = 5.0, kernel = 'rbf')
svm_clf.fit(X_training, y_training)
print('SVM classifier Accuracy for training data is: {:.3f}'.format(accuracy_score(y_training, svm_clf.predict(X_training))))
print('SVM classifier Accuracy for testing data is: {:.3f}'.format(accuracy_score(y_testing, svm_clf.predict(X_testing))))

# ROC curve and AUC
y_score_svm = svm_clf.decision_function(X_testing)
fpr_svm, tpr_svm, thresholds_svm = metrics.roc_curve(y_testing, y_score_svm)
print('SVM classifier AUC is: {:.3f}'.format(metrics.roc_auc_score(y_testing, y_score_svm)))

dt_clf = DecisionTreeClassifier(min_samples_leaf = 1, random_state = 0)
dt_clf.fit(X_training, y_training)
print('Decision Tree Classifier Accuracy for training data is: {:.3f}'.format(accuracy_score(y_training, dt_clf.predict(X_training))))
print('Decision Tree Classifier Accuracy for testing data is: {:.3f}'.format(accuracy_score(y_testing, dt_clf.predict(X_testing))))

# ROC curve and AUC
y_score_dt = dt_clf.predict_proba(X_testing)
fpr_dt, tpr_dt, thresholds_dt = metrics.roc_curve(y_testing, y_score_dt[:,1])
print('Decision Tree Classifier AUC is: {:.3f}'.format(metrics.roc_auc_score(y_testing, y_score_dt[:,1])))

RandomForest_clf = RandomForestClassifier(max_depth = 6, max_features = 'sqrt', n_estimators= 300, min_samples_leaf = 2)
RandomForest_clf.fit(X_training, y_training)

print('RandomForest Classifier Accuracy for training data is: {:.3f}'.format(accuracy_score(y_training, RandomForest_clf.predict(X_training))))
print('RandomForest Classifier Accuracy for testing data is: {:.3f}'.format(accuracy_score(y_testing, RandomForest_clf.predict(X_testing))))

# ROC curve and AUC
y_score_rf = RandomForest_clf.predict_proba(X_testing)
fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(y_testing, y_score_rf[:,1])
print('RandomForest Classifier AUC is: {:.3f}'.format(metrics.roc_auc_score(y_testing, y_score_rf[:,1])))

gb_clf = GradientBoostingClassifier(n_estimators = 500)
gb_clf.fit(X_training, y_training)
print('Gradient Boosting Accuracy for training data is: {:.3f}'.format(accuracy_score(y_training, gb_clf.predict(X_training))))
print('Gradient Boosting Accuracy for testing data is: {:.3f}'.format(accuracy_score(y_testing, gb_clf.predict(X_testing))))

# ROC curve and AUC
y_score_gb = gb_clf.predict_proba(X_testing)
fpr_gb, tpr_gb, thresholds_gb = metrics.roc_curve(y_testing, y_score_gb[:,1])
print('Gradient Boosting Classifier AUC is: {:.3f}'.format(metrics.roc_auc_score(y_testing, y_score_gb[:,1])))

# plt.figure(figsize = (16,8))
# plt.plot(fpr_gb, tpr_gb, lw = 2)
# plt.show()
nn_clf = MLPClassifier(hidden_layer_sizes=(50), random_state=0, learning_rate_init=.05)
nn_clf.fit(X_train, y_train)
print('Neural Network classifier  Accuracy for training data is: {:.3f}'.format(accuracy_score(y_training, nn_clf.predict(X_training))))
print('Neural Network classifier  Accuracy for testing data is: {:.3f}'.format(accuracy_score(y_testing, nn_clf.predict(X_testing))))

# ROC curve and AUC
y_score_nn = nn_clf.predict_proba(X_testing)
fpr_nn, tpr_nn, thresholds_nn = metrics.roc_curve(y_testing, y_score_nn[:,1])
print('Neural Network Classifier AUC is: {:.3f}'.format(metrics.roc_auc_score(y_testing, y_score_nn[:,1])))

fig = plt.figure(figsize = (20,10))

ax = fig.add_subplot(111)

ax1 = ax.plot(fpr_nn, tpr_nn, c = 'b', lw = 2) # blue
ax2 = ax.plot(fpr_rf, tpr_rf, c = 'g', lw = 2)
ax3 = ax.plot(fpr_gb, tpr_gb, c = 'r', lw = 2)
ax4 = ax.plot(fpr_dt, tpr_dt, c = 'c', lw = 2)
ax5 = ax.plot(fpr_svm, tpr_svm, c = 'm', lw = 2)
ax6 = ax.plot(fpr_lr, tpr_lr, c = 'y', lw = 2)
ax7 = ax.plot(fpr_knn, tpr_knn, c = 'k', lw = 2) # black

ax.grid()
lns = ax1 + ax2 + ax3 + ax4 + ax5 + ax6 + ax7
# labs = [l.get_label() for l in lns]
ax.legend(lns, loc=0)
# plt.plot(x = fpr_rf, y = tpr_rf, ax = axis)
plt.show()
