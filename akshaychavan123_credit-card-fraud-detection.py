import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline





# Function to perform data standardization 

from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import train_test_split



# SMOTE technique

from imblearn.over_sampling import SMOTE



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, precision_score, recall_score 



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.tree import DecisionTreeClassifier



from mlxtend.plotting import plot_confusion_matrix





import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
# Last 5 rows

df.tail()
df.shape
df.columns
# Checknig the types of columns

df.dtypes
df.info()
# Checking Null Values!

df.isnull().sum().max()
# Checking class distribution of the classes

print("Number of No fraud transactions = ", df[df['Class'] == 0].shape[0])

print("Number of fraud transactions = ", df[df['Class'] == 1].shape[0])

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
plt.figure(figsize = (10, 8))

colors = ["yellow", "blue"]



sns.countplot('Class', data=df, palette=colors)

plt.xticks(np.arange(2), ['No Fraud', 'Fraud'])

plt.xlabel("Classes", fontsize=12)

plt.ylabel("Count of transaction", fontsize=12)

plt.title('Distribution of target variable.', fontsize=20)
plt.figure(figsize = (10, 8))



sns.distplot(df['Amount'].values, color='b')

plt.xlim([min(df['Amount']), max(df['Amount'])])

plt.title('Distribution of Transaction amount.', fontsize=20)
plt.figure(figsize = (10, 8))



sns.distplot(df['Time'].values, color='b')

plt.xlim([min(df['Time']), max(df['Time'])])

plt.title('Distribution of Transaction time.', fontsize=20)
a = StandardScaler()
df['Amount'] = a.fit_transform(df['Amount'].values.reshape(-1,1))

df['Time'] = a.fit_transform(df['Time'].values.reshape(-1,1))
df.head()
y = df['Class']

X = df.drop(['Class'], axis = 1)
print("Shape of X", X.shape)

print("Shape of y",y.shape)
# Split X and y into train and test sets: 80-30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
# Printing number of samples before oversampling

print("Before OverSampling the count of label 1 (Fraud): {}".format(sum(y_train==1)))

print("Before OverSampling the count of label 0 (Non Fraud): {}".format(sum(y_train==0)))
sm = SMOTE(random_state=2)

X_train, y_train = sm.fit_sample(X_train, y_train.ravel())
# Printing number of samples after oversampling

print("After OverSampling the count of label 1 (Fraud): {}".format(sum(y_train==1)))

print("After OverSampling the count of label 0 (Non Fraud): {}".format(sum(y_train==0)))
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
df.corr()
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(), cmap="Greens")
log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)

print("Accuracy for the logistic regression model is", acc, "%.")

print("Precision for the logistic regression model is {}".format(precision_score(y_test, y_pred)))

print("Recall for the logistic regression model is {}".format(recall_score(y_test, y_pred)))
confusion_matrix(y_test,y_pred).T
cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, figsize = (10, 5 ), cmap = 'tab20c_r')

plt.xticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16, rotation = 90)

plt.yticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16)

plt.show()
gb = GradientBoostingClassifier()

gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)

acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)

print("Accuracy for the gradient boosting classifier is", acc, "%.")

print("Precision for the gradient boosting classifier is {}".format(precision_score(y_test, y_pred)))

print("Recall for the gradient boosting classifier is {}".format(recall_score(y_test, y_pred)))
cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, figsize = (10, 5 ), cmap = 'tab20c_r')

plt.xticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16, rotation = 90)

plt.yticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16)

plt.show()
rf = RandomForestClassifier()

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)

print("Accuracy for the random forest classifier is", acc, "%.")

print("Precision for the random forest classifier is {}".format(precision_score(y_test, y_pred)))

print("Recall for the random forest classifier is {}".format(recall_score(y_test, y_pred)))
cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, figsize = (10, 5 ), cmap = 'tab20c_r')

plt.xticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16, rotation = 90)

plt.yticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16)

plt.show()
tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)

print("Accuracy for the decision tree is", acc, "%.")

print("Precision for the decision tree is {}".format(precision_score(y_test, y_pred)))

print("Recall for the decision tree is {}".format(recall_score(y_test, y_pred)))
cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, figsize = (10, 5 ), cmap = 'tab20c_r')

plt.xticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16, rotation = 90)

plt.yticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16)

plt.show()
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)

print("Accuracy for KNN is", acc, "%.")

print("Precision for KNN is {}".format(precision_score(y_test, y_pred)))

print("Recall for KNN is {}".format(recall_score(y_test, y_pred)))
cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, figsize = (10, 5 ), cmap = 'tab20c_r')

plt.xticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16, rotation = 90)

plt.yticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16)

plt.show()
xgb = XGBClassifier()

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

acc = accuracy_score(y_test, y_pred, normalize=True) * float(100)

print("Accuracy for XGBoost is", acc, "%.")

print("Precision for XGBoost is {}".format(precision_score(y_test, y_pred)))

print("Recall for XGBoost is {}".format(recall_score(y_test, y_pred)))
cm = confusion_matrix(y_test,y_pred)

plot_confusion_matrix(cm, figsize = (10, 5 ), cmap = 'tab20c_r')

plt.xticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16, rotation = 90)

plt.yticks(range(2), ['Non Fraud', 'Fraud'], fontsize=16)

plt.show()