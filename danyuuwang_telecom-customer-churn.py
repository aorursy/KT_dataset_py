import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns#visualization

import matplotlib.pyplot as plt

import itertools

import warnings

warnings.filterwarnings("ignore")

import io

import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
file_path = '../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'

data = pd.read_csv(file_path)

data.head()
print(data.describe())

print('\nThe features are: \n',data.columns.tolist())
for feature in data.columns.tolist():

    print('The unique value in',feature, data[feature].unique())
for feature in data.columns.tolist():

    print('Total number of missing values in', feature, data[feature].isnull().sum())
# change the type of TotalCharges 

data['TotalCharges'] = data["TotalCharges"].replace(" ",np.nan)



#select data without missing values

data = data[data["TotalCharges"].notnull()]

data = data.reset_index()[data.columns]



data["TotalCharges"] = data["TotalCharges"].astype(float)





change_col = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

for col in change_col:

    data[col].replace('No internet service','No',inplace = True)



data['MultipleLines'].replace('No phone service','No', inplace = True)

#customer id col

Id_col     = ['customerID']

#Target column"

target_col = ["Churn"]

#categorical columns

cat_cols   = data.nunique()[data.nunique() < 6].keys().tolist()

cat_cols   = [x for x in cat_cols if x not in target_col]

#numerical columns

num_cols   = [x for x in data.columns if x not in cat_cols + target_col + Id_col]

#Binary columns with 2 values

bin_cols   = data.nunique()[data.nunique() == 2].keys().tolist()

#Columns more than 2 values

multi_cols = [i for i in cat_cols if i not in bin_cols]

data['TotalCharges'].isnull().sum()
data['Churn'].value_counts()
labels = ['not churn','churn']

plt.figure(figsize=(8,8))

plt.pie(data['Churn'].value_counts(),labels = labels, autopct = '%1.1f%%',shadow = True,explode = (0.05,0.05))

plt.legend()
sns.set(style="white", palette="muted", color_codes=True)

plt.figure(figsize=(10,6))

f, axes = plt.subplots(2, 2, figsize=(14, 14))

sns.countplot(x = 'gender',hue = 'Churn', data = data, ax=axes[0,0])

sns.countplot(x = 'SeniorCitizen',hue = 'Churn', data = data, ax=axes[0,1])

sns.countplot(x = 'Partner',hue = 'Churn', data = data, ax=axes[1,0])

sns.countplot(x = 'Dependents',hue = 'Churn', data = data, ax=axes[1,1])
f, axes = plt.subplots(5, 2, figsize=(15, 18))

plt.subplots_adjust(wspace =0.55, hspace =0.5)

sns.countplot(y = 'PhoneService',hue = 'Churn', data = data, ax=axes[0,0]) 

sns.countplot(y = 'MultipleLines',hue = 'Churn', data = data, ax=axes[0,1])

sns.countplot(y = 'InternetService',hue = 'Churn', data = data, ax=axes[1,0])

sns.countplot(y = 'OnlineSecurity',hue = 'Churn', data = data, ax=axes[1,1])

sns.countplot(y = 'OnlineBackup',hue = 'Churn', data = data, ax=axes[2,0])

sns.countplot(y = 'DeviceProtection',hue = 'Churn', data = data, ax=axes[2,1])

sns.countplot(y = 'TechSupport',hue = 'Churn', data = data, ax=axes[3,0])

sns.countplot(y = 'StreamingTV',hue = 'Churn', data = data, ax=axes[3,1])

sns.countplot(y = 'Contract',hue = 'Churn', data = data, ax=axes[4,0])  

sns.countplot(y = 'PaymentMethod',hue = 'Churn', data = data, ax=axes[4,1])
#Label encoding Binary columns

label_encoder = LabelEncoder()

for i in bin_cols :

    data[i] = label_encoder.fit_transform(data[i])

    

#Duplicating columns for multi value columns

data = pd.get_dummies(data = data,columns = multi_cols )

#Scaling Numerical columns

std = StandardScaler()

scaled = std.fit_transform(data[num_cols])

scaled = pd.DataFrame(scaled,columns=num_cols)
#dropping original values merging scaled values for numerical columns

data_copy = data.copy()

data = data.drop(columns = num_cols,axis = 1)

data = data.merge(scaled,left_index=True,right_index=True,how = "left")
plt.figure(figsize = (13,7))



#get the correlation coefficient of each pair of attributes

correlation = data.corr()

sns.heatmap(data = correlation)
from sklearn.model_selection import train_test_split



train,test = train_test_split(data,test_size = .30 ,random_state = 1)

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

cols    = [i for i in data.columns if i not in Id_col+target_col]

smote_X = data[cols]

smote_Y = data[target_col]

os = SMOTE(random_state=1)

#Split train and test data

smote_train_X,smote_test_X,smote_train_Y,smote_test_Y = train_test_split(smote_X,smote_Y,test_size = .30 ,random_state = 1)

os_smote_X,os_smote_Y = os.fit_sample(smote_train_X,smote_train_Y)

os_smote_X = pd.DataFrame(data = os_smote_X,columns=cols)

os_smote_Y = pd.DataFrame(data = os_smote_Y,columns=target_col)



print("length of oversampled data is ",len(os_smote_X))
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 20)

rfe = rfe.fit(os_smote_X, os_smote_Y.values.ravel())

print(rfe.support_)

print(rfe.ranking_)

idc_rfe = pd.DataFrame({"rfe_support" :rfe.support_,

                       "columns" : [i for i in data.columns if i not in Id_col + target_col],

                       "ranking" : rfe.ranking_,

                      })

cols = idc_rfe[idc_rfe["rfe_support"] == True]["columns"].tolist()



#separating train and test data

train_rf_X = os_smote_X[cols]

train_rf_Y = os_smote_Y

test_rf_X  = test[cols]

test_rf_Y  = test[target_col]

import statsmodels.api as sm

logit_model=sm.Logit(train_rf_Y,train_rf_X)

result=logit_model.fit()

print(result.summary2())
cols = idc_rfe[idc_rfe["rfe_support"] == True]["columns"].tolist()

drop_cols = ['PhoneService','MultipleLines','Contract_Two year','MonthlyCharges']

cols = [i for i in cols if i not in drop_cols]

train_rf_X1 = os_smote_X[cols]

train_rf_Y1 = os_smote_Y

test_rf_X1  = test[cols]

test_rf_Y1 = test[target_col]

logit_model=sm.Logit(train_rf_Y1,train_rf_X1)

result1=logit_model.fit()

print(result1.summary2())
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score,recall_score

from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(os_smote_X, os_smote_Y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
from sklearn.metrics import classification_report

logit_report = classification_report(y_test, y_pred)
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.show()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,auc



knn = KNeighborsClassifier(n_neighbors=5)



#Train the model using the training sets

knn.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = knn.predict(X_test)



knn_prob_y_predict = knn.predict_proba(X_test)

y_predict = knn_prob_y_predict[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_predict)

roc_auc = auc(fpr, tpr)

plt.title('ROC Validation')

plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

plt.legend(loc='lower right')

plt.plot([0, 1], [0, 1], 'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

knn_report = classification_report(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier

forest_model = RandomForestClassifier(n_estimators = 100, max_depth=4, random_state=1)

forest_model.fit(X_train, y_train)

melb_preds = forest_model.predict(X_test)

from sklearn.metrics import roc_curve, auc

clf = RandomForestClassifier(n_estimators = 100, max_depth=4,random_state=1)

clf.fit(X_train, y_train)

r = clf.score(X_test,y_test)

clf.estimators_



prob_y_predict = clf.predict_proba(X_test)#给出带有概率值的结果，每个点所有label的概率和为1

y_predict = prob_y_predict[:, 1]

y_pred = clf.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_predict)

roc_auc = auc(fpr, tpr)

plt.title('ROC Validation')

plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)

plt.legend(loc='lower right')

plt.plot([0, 1], [0, 1], 'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

rf_report = classification_report(y_test, y_pred)
forest_model.feature_importances_
clf.estimators_[99]

from sklearn.ensemble import RandomForestClassifier

import graphviz

from sklearn.tree import DecisionTreeClassifier, export_graphviz



data = export_graphviz(clf.estimators_[99],out_file=None,feature_names=X_train.columns,

                       class_names=["Not churn","Churn"], 

                       filled=True, rounded=True,  

                       max_depth=4,

                       special_characters=True)

graph = graphviz.Source(data)

graph
from sklearn import svm

#Create a svm Classifier

svm_clf = svm.SVC(kernel='linear') # Linear Kernel



#Train the model using the training sets

svm_clf.fit(X_train, y_train)



#Predict the response for test dataset

svm_pred = svm_clf.predict(X_test)
from sklearn import metrics



print("Accuracy:",metrics.accuracy_score(y_test, svm_pred))

print("Precision:",metrics.precision_score(y_test, svm_pred))

svm_report = classification_report(y_test, svm_pred)

print('logistic report:\n',logit_report)

print('========================================================')

print('KNN classifier report:\n',knn_report)

print('========================================================')

print('Random Forest report:\n',rf_report)

print('========================================================')

print('Support Vector Machine report:\n',svm_report)