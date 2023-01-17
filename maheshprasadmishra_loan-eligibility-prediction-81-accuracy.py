import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train_df = pd.read_csv('../input/loan-eligible-dataset/loan-train.csv')
test_df = pd.read_csv('../input/loan-eligible-dataset/loan-test.csv')
train_df.head(10)
test_df.head(10)
train_df.shape
train_df.info()
train_df.describe()
100*train_df.isnull().sum()/len(train_df)
train_df['Gender'].value_counts()
train_df['Married'].value_counts()
train_df['Dependents'].value_counts()
train_df['Self_Employed'].value_counts()
train_df['Married'] = train_df['Married'].replace(np.nan,'Yes')
train_df['Dependents'] = train_df['Dependents'].replace(np.nan,'0')
train_df['Self_Employed'] = train_df['Self_Employed'].replace(np.nan,'No')
train_df['Gender'].value_counts()
train_df['Gender'] = train_df['Gender'].replace(np.nan,'Male')
100*train_df.isnull().sum()/len(train_df)
train_df['LoanAmount'] = train_df['LoanAmount'].fillna(train_df['LoanAmount'].median())
train_df['Loan_Amount_Term'] = train_df['Loan_Amount_Term'].fillna(train_df['Loan_Amount_Term'].median())
train_df['Credit_History'] = train_df['Credit_History'].fillna(train_df['Credit_History'].median())
100*train_df.isnull().sum()/len(train_df)
train_df.info()
train_df.columns
def plot_count(var_list):
    plt.figure(figsize=(30,30))
    for var in var_list:
        plt.subplot(4,4,var_list.index(var)+1)
        ax=sns.countplot(train_df[var], data = train_df)   
    plt.show()
plot_count(['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Property_Area', 'Loan_Status'])
plt.figure(figsize = (10,10))
sns.heatmap(train_df.corr(), annot = True)
sns.pairplot(train_df)
sns.countplot('Gender', data = train_df, hue = 'Married')
def plot_count1(var_list):
    plt.figure(figsize=(30,30))
    for var in var_list:
        plt.subplot(3,2,var_list.index(var)+1)
        ax=sns.countplot(train_df[var], data = train_df, hue = 'Loan_Status')   
    plt.show()
plot_count1(['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Property_Area'])
train_df.Loan_Status.replace(['N', 'Y'], [0, 1], inplace =True)
train_df.head()
plt.figure(figsize = (10,10))
sns.heatmap(train_df.corr(), annot = True)
train_df.columns
df_cat = train_df[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Property_Area']]
df_cat.info()
df_cat_dummies = pd.get_dummies(df_cat)
df_cat_dummies
train_df = pd.concat([train_df, df_cat_dummies], axis = 1)
train_df.head()
train_df.drop(['Loan_ID','Gender', 'Married', 'Dependents', 'Education','Self_Employed','Property_Area'], axis = 1, inplace  =True)
train_df
plt.figure(figsize = (20,10))
sns.heatmap(train_df.corr(), annot = True)
X = train_df.drop(['Loan_Status'], axis =1)
y = train_df['Loan_Status']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 142)
from sklearn.preprocessing import StandardScaler, MinMaxScaler


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import  mean_squared_error, r2_score,plot_roc_curve
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

lr_pred = logreg.predict(X_test)
print("Accuracy {}".format(metrics.accuracy_score(y_test, lr_pred)))
print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, lr_pred)))
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
print("Accuracy {}".format(metrics.accuracy_score(y_test, rf_pred)))
print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, rf_pred)))
print(confusion_matrix(y_test,rf_pred))
print(classification_report(y_test, rf_pred))
from sklearn.svm import SVC

svm=SVC()
svm.fit(X_train,y_train)
svm_pred=svm.predict(X_test)

print("Accuracy {}".format(metrics.accuracy_score(y_test, svm_pred)))
print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, svm_pred)))
print(confusion_matrix(y_test,svm_pred))
print(classification_report(y_test, svm_pred))
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=30, criterion='entropy')
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)

print("Accuracy {}".format(metrics.accuracy_score(y_test, dt_pred)))
print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, dt_pred)))
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(X_train, y_train)
gbc_pred = gbc.predict(X_test)

print("Accuracy {}".format(metrics.accuracy_score(y_test, gbc_pred)))
print("Recall/Sensitivity {}".format(metrics.recall_score(y_test, gbc_pred)))
print(confusion_matrix(y_test, gbc_pred))
print(classification_report(y_test, gbc_pred))
pd.concat([pd.DataFrame(X.columns, columns = ['variable']),
           pd.DataFrame(rf_model.feature_importances_, columns = ['importance'])],
          axis = 1).sort_values(by = 'importance', ascending = False)
test_df.head()
100*test_df.isnull().sum()/len(test_df)
test_df['Gender'].value_counts()
test_df['Gender'] = test_df['Gender'].replace(np.nan,'Male')
test_df['Dependents'] = test_df['Dependents'].replace(np.nan,'0')
test_df['Self_Employed'] = test_df['Self_Employed'].replace(np.nan,'No')
test_df['LoanAmount'] = test_df['LoanAmount'].fillna(test_df['LoanAmount'].median())
test_df['Loan_Amount_Term'] = test_df['Loan_Amount_Term'].fillna(test_df['Loan_Amount_Term'].median())
test_df['Credit_History'] = test_df['Credit_History'].fillna(test_df['Credit_History'].median())
100*test_df.isnull().sum()/len(test_df)
test_df.columns
test_df.info()
test_cat = test_df[['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed','Property_Area']]
test_cat_dummies = pd.get_dummies(test_cat)
test_df = pd.concat([test_df, test_cat_dummies], axis = 1)
test_df.head()
test_df.drop(['Loan_ID','Gender', 'Married', 'Dependents', 'Education','Self_Employed','Property_Area'], axis = 1, inplace  =True)
test_lr_predict = logreg.predict(test_df)
test_lr_predict
test_rf_predict = rf_model.predict(test_df)
test_rf_predict
test_sv_predict = svm.predict(test_df)
test_sv_predict
test_dt_predict = dt_clf.predict(test_df)
test_dt_predict
test_gbc_predict = gbc.predict(test_df)
test_gbc_predict


