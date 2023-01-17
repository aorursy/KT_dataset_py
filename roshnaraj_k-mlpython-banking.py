# importing pandas for reading the datasets

import pandas as pd
# reading the training dataset with a ';' delimiter

bdata=pd.read_csv('../input/mlworkshop/bank-full.csv',delimiter=';')
# displaying first 10 observations from the dataset

bdata.head(10)
# describing the pandas dataframe bdata

bdata.describe()
# getting details of no of attributes and observations

print('No of observations :',bdata.shape[0])

print('No of attributes :',bdata.shape[1])

print('No of numerical attributes :',bdata.describe().shape[1])

print('No of categorical attributes :',bdata.shape[1]-bdata.describe().shape[1])
# getting list of attributes

bdata.columns.tolist()
# importing matplotlib for plotting the graphs

import matplotlib.pyplot as plt
bdata['y'].value_counts().plot(kind='bar')

plt.title('Subscriptions')

plt.xlabel('Term Deposit')

plt.ylabel('No of Subscriptions')

plt.show()
pd.crosstab(bdata.job,bdata.y).plot(kind='bar')

plt.title('Subscriptions based on Job')

plt.xlabel('Job')

plt.ylabel('No of Subscriptions')

plt.show()
pd.crosstab(bdata.marital,bdata.y).plot(kind='bar')

plt.title('Subscriptions based on Marital Status')

plt.xlabel('Marital Status')

plt.ylabel('No of Subscriptions')

plt.show()
pd.crosstab(bdata.education,bdata.y).plot(kind='bar')

plt.title('Subscriptions based on Education')

plt.xlabel('Education')

plt.ylabel('No of Subscriptions')

plt.show()
pd.crosstab(bdata.housing,bdata.y).plot(kind='bar')

plt.title('Subscriptions based on Housing Credit')

plt.xlabel('Housing Credit')

plt.ylabel('No of Subscriptions')

plt.show()
pd.crosstab(bdata.loan,bdata.y).plot(kind='bar')

plt.title('Subscriptions based on Personal Loan')

plt.xlabel('Personal Loan')

plt.ylabel('No of Subscriptions')

plt.show()
pd.crosstab(bdata.poutcome,bdata.y).plot(kind='bar')

plt.title('Subscriptions based on Outcome of Previous Campaign')

plt.xlabel('Outcome of Previous Campaign')

plt.ylabel('No of Subscriptions')

plt.show()
pd.crosstab(bdata.month,bdata.y).plot(kind='bar')

plt.title('Monthly Subscriptions')

plt.xlabel('Month')

plt.ylabel('No of Subscriptions')

plt.show()
# creating dummy variables for categorical variables



# creating a list of categorical variables to be transformed into dummy variables

category=['job','marital','education','default','housing','loan','contact',

          'month','poutcome']



# creating a backup

bdata_new = bdata



# creating dummy variables and joining it to the training set

for c in category:

    new_column = pd.get_dummies(bdata_new[c], prefix=c)

    bdata_dummy=bdata_new.join(new_column)

    bdata_new=bdata_dummy
bdata_new.head(10)
# see the dummy setup of one categorical variable

bdata_new[[col for col in bdata_new if col.startswith('education')]].head(10)
# drop the initial categorical variable

bdata_final=bdata_new.drop(category,axis=1)
bdata_final.head(10)
# coding no as '0' and yes as '1'

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

labels = le.fit_transform(bdata_final['y'])

bdata_final['y'] = labels
bdata_final.y.value_counts()
bdata_final.head(10)
# feature selection to reduce dimensionality

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import MinMaxScaler



# creating dataframe of features

X=bdata_final.drop(['y'],axis=1)

# creating dataframe of output variable

y=bdata_final['y']



# standard scaling

X_norm = MinMaxScaler().fit_transform(X)



rfe_selector = RFE(estimator=LogisticRegression(solver='liblinear',max_iter=100,multi_class='ovr',n_jobs=1), n_features_to_select=30, step=10, verbose=5)

rfe_selector.fit(X_norm, y)

rfe_support = rfe_selector.get_support()

rfe_feature = X.loc[:,rfe_support].columns.tolist()

print(str(len(rfe_feature)), 'selected features')
rfe_feature
# dropping age and pdays

bdata_final=bdata_final.drop(['age','pdays'],axis=1)
bdata_final.head(10)
cat=[col for col in bdata_final if col.startswith('job')]

mar_cat=[col for col in bdata_final if col.startswith('marital')]

edu_cat=[col for col in bdata_final if col.startswith('education')]

loan_cat=[col for col in bdata_final if col.startswith('loan')]

cat.extend(mar_cat)

cat.extend(edu_cat)

cat.extend(loan_cat)
cat
# creating a dataframe with lesser dimension

bdata_dr=bdata_final.drop(cat,axis=1)
bdata_dr.head(10)
# importing sklearn for train test split

from sklearn.model_selection import train_test_split
# creating training set of features

X=bdata_final.drop(['y'],axis=1)

# creating training set of output variable

y=pd.DataFrame(bdata_final['y'])
# splitting the dataset into train and test for both input and output variable

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
X_train.head(10)
y_train.head(10)
X_test.head(10)
y_test.head(10)
# importing the Standard Scaler from sklearn

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

y_train = y_train.values.ravel()

y_test = y_test.values.ravel()
X_train
X_test
y_train
y_test
# importing imblearn for Synthetic Minority Over Sampling Technique

# NOTE : SMOTE technique needs the dataset to be numpy array



# from imblearn.over_sampling import SMOTE

# sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=0)

# X_res, y_res = sm.fit_resample(X_train, y_train)

# import numpy as np

# np.savetxt('xres.txt', X_res, fmt='%f')

# np.savetxt('yres.txt', y_res, fmt='%d')



# SMOTE applied dataset

import numpy as np

X_res = np.loadtxt('../input/smotedata/xres.txt', dtype=float)

y_res = np.loadtxt('../input/smotedata/yres.txt', dtype=int)
print('No 0f 0 case :',y_res[y_res==0].shape[0])

print('No of 1 case :',y_res[y_res==1].shape[0])
from sklearn.ensemble import RandomForestClassifier

# Create the model with 100 trees

modelrf = RandomForestClassifier(n_estimators=100, 

                               bootstrap = True,

                               max_features = 'sqrt')
# Fit on training data

modelrf.fit(X_res, y_res)
# predicting the testing set results

y_pred = modelrf.predict(X_test)

y_pred = (y_pred > 0.50)
# importing confusion matrix and roc_auc_score from sklearn

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score



# importing seaborn for plotting the heatmap

import seaborn as sn



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = ('no', 'yes'), columns = ('predicted no',

                                                           'predicted yes'))

plt.figure(figsize = (5,4))

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True, fmt='g')

plt.show()

print("Test Data Accuracy: %0.4f" % roc_auc_score(y_test, y_pred))
# importing roc curve and metrics from sklearn

from sklearn.metrics import roc_curve

import sklearn.metrics as metrics



fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc=roc_auc_score(y_test, y_pred)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from sklearn.svm import LinearSVC

modelsv = LinearSVC(max_iter=100,random_state=0)
modelsv.fit(X_res, y_res)
# predicting the testing set results

y_pred = modelsv.predict(X_test)

y_pred = (y_pred > 0.50)
# importing confusion matrix and roc_auc_score from sklearn

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score



# importing seaborn for plotting the heatmap

import seaborn as sn



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = ('no', 'yes'), columns = ('predicted no',

                                                           'predicted yes'))

plt.figure(figsize = (5,4))

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True, fmt='g')

plt.show()

print("Test Data Accuracy: %0.4f" % roc_auc_score(y_test, y_pred))
# importing roc curve and metrics from sklearn

from sklearn.metrics import roc_curve

import sklearn.metrics as metrics



fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc=roc_auc_score(y_test, y_pred)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from sklearn.neighbors import KNeighborsClassifier

modelkn = KNeighborsClassifier(n_neighbors=3)
modelkn.fit(X_res, y_res)
# predicting the testing set results

y_pred = modelkn.predict(X_test)

y_pred = (y_pred > 0.50)
# importing confusion matrix and roc_auc_score from sklearn

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score



# importing seaborn for plotting the heatmap

import seaborn as sn



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = ('no', 'yes'), columns = ('predicted no',

                                                           'predicted yes'))

plt.figure(figsize = (5,4))

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True, fmt='g')

plt.show()

print("Test Data Accuracy: %0.4f" % roc_auc_score(y_test, y_pred))
# importing roc curve and metrics from sklearn

from sklearn.metrics import roc_curve

import sklearn.metrics as metrics



fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc=roc_auc_score(y_test, y_pred)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from sklearn.linear_model import LogisticRegression

modellr = LogisticRegression()
modellr.fit(X_res, y_res)
# predicting the testing set results

y_pred = modellr.predict(X_test)

y_pred = (y_pred > 0.50)
# importing confusion matrix and roc_auc_score from sklearn

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score



# importing seaborn for plotting the heatmap

import seaborn as sn



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = ('no', 'yes'), columns = ('predicted no',

                                                           'predicted yes'))

plt.figure(figsize = (5,4))

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True, fmt='g')

plt.show()

print("Test Data Accuracy: %0.4f" % roc_auc_score(y_test, y_pred))
# importing roc curve and metrics from sklearn

from sklearn.metrics import roc_curve

import sklearn.metrics as metrics



fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc=roc_auc_score(y_test, y_pred)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
from sklearn.naive_bayes import GaussianNB

modelnb = GaussianNB()
modelnb.fit(X_res, y_res)
# predicting the testing set results

y_pred = modelnb.predict(X_test)

y_pred = (y_pred > 0.50)
# importing confusion matrix and roc_auc_score from sklearn

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score



# importing seaborn for plotting the heatmap

import seaborn as sn



cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = ('no', 'yes'), columns = ('predicted no',

                                                           'predicted yes'))

plt.figure(figsize = (5,4))

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True, fmt='g')

plt.show()

print("Test Data Accuracy: %0.4f" % roc_auc_score(y_test, y_pred))
# importing roc curve and metrics from sklearn

from sklearn.metrics import roc_curve

import sklearn.metrics as metrics



fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)

roc_auc=roc_auc_score(y_test, y_pred)



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()