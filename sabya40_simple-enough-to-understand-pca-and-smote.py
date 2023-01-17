# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, auc, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, GenericUnivariateSelect, f_regression
from sklearn.decomposition import PCA
from xgboost import XGBClassifier

import eli5
from eli5.sklearn import PermutationImportance

import pickle

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv(r"/kaggle/input/santander-customer-transaction-prediction/train.csv")
test = pd.read_csv(r"/kaggle/input/santander-customer-transaction-prediction/test.csv")
train.columns = train.columns.str.strip()

train.head(10)
print("Train Dataset ",train.shape)
print("Test Dataset ",test.shape)
test.head(10)
#Missing Percentage of all the features
def missing_percentage(df):     
    missing_total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([missing_total, percent], axis=1, keys=['Missing_Total','Percent'])
missing_percentage(train)
missing_percentage(test)
target = train['target']
#train = train.drop(["ID_code", "target"], axis=1)
sns.set_style('whitegrid')
sns.countplot(target)
train['target'].value_counts()
#Check for imbalanced
def check_balance(df,target):
    check=[]
    print('size of data is:',df.shape[0] )
    for i in [0,1]:
        print('for target  {} ='.format(i))
        print(df[target].value_counts()[i]/df.shape[0]*100,'%')
check_balance(train, 'target')
train.describe()
data_prep = [col for col in train.columns if col not in ['ID_code', 'target']]
print("Skewness: %f" % train['target'].skew())
print("Kurtosis: %f" % train['target'].kurt())
plt.figure(figsize=(16,6))
plt.title("Distribution of mean values per row in the train and test set")
sns.distplot(train[data_prep].mean(axis=1),color="green", kde=True,bins=120, label='train')
sns.distplot(test[data_prep].mean(axis=1),color="red", kde=True,bins=120, label='test')
plt.legend()
plt.show()
plt.figure(figsize=(16,6))
plt.title("Distribution of std values per rows in the train and test set")
sns.distplot(train[data_prep].std(axis=1),color="black",kde=True,bins=120, label='train')
sns.distplot(test[data_prep].std(axis=1),color="green", kde=True,bins=120, label='test')
plt.legend(); plt.show()
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of skew values per row in the train set")
sns.distplot(t0[data_prep].skew(axis=1),color="pink", kde=True,bins=120, label='target = 0')
sns.distplot(t1[data_prep].skew(axis=1),color="green", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
t0 = train.loc[target == 0]
t1 = train.loc[target == 1]
plt.figure(figsize=(16,6))
plt.title("Distribution of kurtosis values per row in the train set")
sns.distplot(t0[data_prep].kurtosis(axis=1),color="red", kde=True,bins=120, label='target = 0')
sns.distplot(t1[data_prep].kurtosis(axis=1),color="blue", kde=True,bins=120, label='target = 1')
plt.legend(); plt.show()
X = train.drop(['ID_code', 'target'], axis = 1)
y = train['target']
# Function to calculate mean absolute error
def cross_val(X_train, y_train, model):
    # Applying k-Fold Cross Validation    
    accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 5)
    return accuracies.mean()

# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate(model):
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_cross = cross_val(X_train, y_train, model)
    
    # Return the performance metric
    return model_cross

#This method is used for selecting the Most important feature which affect the target value using PCA.
def fit_and_evaluate_select_k(model):
    
    # Train the model
    model.fit(X_train_pca, y_train_pca)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test_pca)
    model_cross = cross_val(X_train_pca, y_train_pca, model)
    
    # Return the performance metric
    return model_cross




#As our dataset is imbalanced so we used SMOTE
# Takes in a model, trains the model, and evaluates the model on the test set
def fit_and_evaluate_smote(model):
    
    # Train the model
    model.fit(X_train_smote, Y_train_smote)
    
    # Make predictions and evalute
    model_pred = model.predict(X_test)
    model_cross = cross_val(X_train_smote, Y_train_smote, model)
    
    # Return the performance metric
    return model_cross



def performance(Y_test, logist_pred):
    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]
    print('Confusion Matrix:')
    print(confusion_matrix(Y_test, logist_pred_var)) 
   
    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)
    print('AUC:')
    print(auc(fpr, tpr))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_scaled = sc.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
#Due to  logistic regression and it depends on euclidean distance so we scaled the value 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=2)
lr = LogisticRegression()
logistic_Regression = fit_and_evaluate(lr)

print('LogisticRegression Performance on the test set: Cross Validation Score = %0.4f' % logistic_Regression)
#Lets predict
Y_predict = lr.predict(X_test)
#Let us seaborn in confusion matrix
cm = confusion_matrix(y_test, Y_predict)
annot_kws = {"ha": 'left',"va": 'top'}
sns.heatmap(cm/np.sum(cm), annot=True, annot_kws=annot_kws,
           fmt='.2%', cmap='Blues')

performance(y_test, Y_predict)

#This model gave out an AUC of 0.626 on validation set
# calculate AUC
auc = roc_auc_score(y_test, Y_predict)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, Y_predict)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()
perm = PermutationImportance(lr, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist(), top=150)
#classification report about the model 
print(classification_report(y_test, Y_predict))
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_smote, Y_train_smote = smote.fit_sample(X_train, y_train)
from collections import Counter
print("Before SMOTE ", Counter(y_train))
print("After SMOTE", Counter(Y_train_smote))
lr_smote = LogisticRegression()
logistic_Regression_smote = fit_and_evaluate_smote(lr_smote)

print('LogisticRegression Performance on the SMOTE test set: Cross Validation Score = %0.4f' % logistic_Regression_smote)
#Lets predict
Y_predict_smote = lr_smote.predict(X_test)
#After using SMOTE lets check the peformance
#Let us seaborn in confusion matrix
cm = confusion_matrix(y_test, Y_predict_smote)
annot_kws = {"ha": 'left',"va": 'top'}
sns.heatmap(cm/np.sum(cm), annot=True, annot_kws=annot_kws,
           fmt='.2%', cmap='Blues')
# calculate AUC
auc = roc_auc_score(y_test, Y_predict_smote)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, Y_predict_smote)
# plot no skill
plt.plot([0, 1], [0, 1])
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()
#classification report about the model 
print(classification_report(y_test, Y_predict_smote))
pca = PCA(n_components= 100)

X_pca = pca.fit_transform(X_train_smote)
X_pca
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, Y_train_smote, test_size=0.3, random_state=2)

lr_pca = LogisticRegression()
logistic_Regression_pca = fit_and_evaluate_select_k(lr_pca)

print('LogisticRegression Performance on the test set: Cross Validation Score = %0.4f' % logistic_Regression_pca)
#Lets predict
Y_predict_pca = lr_pca.predict(X_test_pca)
#Let us seaborn in confusion matrix
cm = confusion_matrix(y_test_pca, Y_predict_pca)
annot_kws = {"ha": 'left',"va": 'top'}
sns.heatmap(cm/np.sum(cm), annot=True, annot_kws=annot_kws,
           fmt='.2%', cmap='Blues')
# calculate AUC
auc = roc_auc_score(y_test_pca, Y_predict_pca)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test_pca, Y_predict_pca)
# plot no skill
plt.plot([0, 1], [0, 1])
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()

#classification report about the model 
print(classification_report(y_test_pca, Y_predict_pca))
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
random_cross = fit_and_evaluate_select_k(random)

print('Random Forest Performance on the test set: Cross Validation Score = %0.4f' % random_cross)
#Lets predict
Y_predict_rf = random.predict(X_test_pca)
#Let us seaborn in confusion matrix
cm = confusion_matrix(y_test_pca, Y_predict_rf)
annot_kws = {"ha": 'left',"va": 'top'}
sns.heatmap(cm/np.sum(cm), annot=True, annot_kws=annot_kws,
           fmt='.2%', cmap='Blues')
# calculate AUC
auc = roc_auc_score(y_test_pca, Y_predict_rf)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test_pca, Y_predict_rf)
# plot no skill
plt.plot([0, 1], [0, 1])
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.show()
#classification report about the model 
print(classification_report(y_test_pca, Y_predict_rf))
model=XGBClassifier(random_state=1,learning_rate=0.01)

xgBoost_Cross = fit_and_evaluate_select_k(model)
print('XGBClassifier Performance on the test set: Cross Validation Score = %0.4f' % xgBoost_Cross)
from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X_train_pca, y_train_pca) 

svc_fit = fit_and_evaluate_select_k(clf)

print('SVC Performance on the test set: Cross Validation Score = %0.4f' % svc_fit)
#Now it is time to test our algorithm using TEST set.
data_x_test = test.drop(columns = ['ID_code'])

#scaling dataset
sc = StandardScaler()
sc.fit(data_x_test)
X_scaled = sc.transform(data_x_test)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

#Reducing the dimension
pca = PCA(n_components= 100)
X_pca = pca.fit_transform(X_scaled) 


y_pred = random.predict(X_pca)
data_y = pd.DataFrame(y_pred)
df_submission = pd.merge(pd.DataFrame(test['ID_code']),data_y, left_index=True, right_index=True)

print(df_submission)
df_submission.to_csv("submission.csv", index=False)
