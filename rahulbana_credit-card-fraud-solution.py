import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(style="darkgrid")


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_auc_score, roc_curve, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from mlxtend.classifier import StackingCVClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

from imblearn.under_sampling import NearMiss
from imblearn.combine import SMOTETomek
data_df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
data_df.shape
data_df.columns
data_df.head()
total = data_df.isnull().sum().sort_values(ascending = False)
percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()
data_df.Class.value_counts(normalize=True)*100
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=False)
plt.show();
data_df.groupby(['Class'])['Amount'].agg({'min','max','mean','median','std'}).reset_index()
sns.scatterplot(x='Time', y='Amount', data=data_df[data_df['Class'] == 1])
plt.show()
sns.scatterplot(x='Time', y='Amount', data=data_df[data_df['Class'] == 0])
plt.show()
plt.figure(figsize=(20,8))

ax = sns.violinplot(x='Class', y='Amount', data=data_df)

plt.show()
plt.figure(figsize = (14,14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = data_df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()
var = data_df.columns.values

i = 0
t0 = data_df.loc[data_df['Class'] == 0]
t1 = data_df.loc[data_df['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0")
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1")
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();

target = 'Class'
predictors = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',\
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',\
       'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',\
       'Amount']
X, y = data_df[predictors], data_df[target]
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
def show_model_report(x,y,model):
    y_pred = model.predict(x)

    print('Classification report:')
    print(classification_report(y, y_pred))
    print("-----------------------------------\n")
    print('Confusion Metrix: ')
    print(confusion_matrix(y, y_pred))
    print("-----------------------------------\n")
    print('Accuracy Score')
    print(roc_auc_score(y, y_pred))
    print("-----------------------------------\n")
    
    try:
        y_pred = model.predict_proba(x)[:,1]
        FPR, TPR, threshold = roc_curve(y, y_pred)
        print("ROC AUC Score")
        print(roc_auc_score(y, y_pred))

        #Plot ROC curve
        plt.title('ROC Curve')
        plt.plot(FPR, TPR)
        plt.plot([0,1], ls ='--')
        plt.plot([0,0], [1,0], c='.7'), plt.plot([1,1], c='.7')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    except Exception as ex:
        pass
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)


print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)

print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
lr_model = GaussianNB()
lr_model.fit(X_train, y_train)

print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
svc_model = SVC()
svc_model.fit(X_train, y_train)

print('\n\n')
print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,svc_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,svc_model)
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

print('\n\n')
print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,dt_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,dt_model)
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)


print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,rf_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,rf_model)
et_model = ExtraTreeClassifier()
et_model.fit(X_train, y_train)

print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,et_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,et_model)
abc_model = AdaBoostClassifier()
abc_model.fit(X_train, y_train)

print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,abc_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,abc_model)
xb_model = XGBClassifier()
xb_model.fit(X_train, y_train)

print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,xb_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,xb_model)
cb_model = CatBoostClassifier()
cb_model.fit(X_train, y_train)

print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,cb_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,cb_model)
lgbc_model = LGBMClassifier()
lgbc_model.fit(X_train, y_train)

print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lgbc_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lgbc_model)

nm = NearMiss()
X_us, y_us= nm.fit_sample(X_train,y_train)
from collections import Counter

print('Original dataset shape {}'.format(Counter(y_train)))
print('Undersampled dataset shape {}' .format(Counter(y_us)))
lr_model = LogisticRegression()
lr_model.fit(X_us, y_us)

print("Undersampld data report")
print("---------------------------------------")
show_model_report(X_us,y_us,lr_model)

print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)

print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
lr_model = GaussianNB()
lr_model.fit(X_us, y_us)

print("Undersampld data report")
print("---------------------------------------")
show_model_report(X_us,y_us,lr_model)
print('\n\n')
print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
lr_model = SVC()
lr_model.fit(X_us, y_us)

print("Undersampld data report")
print("---------------------------------------")
show_model_report(X_us,y_us,lr_model)
print('\n\n')
print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
lr_model = DecisionTreeClassifier()
lr_model.fit(X_us, y_us)

print("Undersampld data report")
print("---------------------------------------")
show_model_report(X_us,y_us,lr_model)
print('\n\n')
print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
lr_model = RandomForestClassifier()
lr_model.fit(X_us, y_us)

print("Undersampld data report")
print("---------------------------------------")
show_model_report(X_us,y_us,lr_model)
print('\n\n')
print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
lr_model = ExtraTreeClassifier()
lr_model.fit(X_us, y_us)

print("Undersampld data report")
print("---------------------------------------")
show_model_report(X_us,y_us,lr_model)
print('\n\n')
print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
lr_model = AdaBoostClassifier()
lr_model.fit(X_us, y_us)

print("Undersampld data report")
print("---------------------------------------")
show_model_report(X_us,y_us,lr_model)
print('\n\n')
print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
lr_model = XGBClassifier()
lr_model.fit(X_us, y_us)

print("Undersampld data report")
print("---------------------------------------")
show_model_report(X_us,y_us,lr_model)
print('\n\n')
print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
lr_model = CatBoostClassifier()
lr_model.fit(X_us, y_us)

print("Undersampld data report")
print("---------------------------------------")
show_model_report(X_us,y_us,lr_model)
print('\n\n')
print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
lr_model = LGBMClassifier()
lr_model.fit(X_us, y_us)

print("Undersampld data report")
print("---------------------------------------")
show_model_report(X_us,y_us,lr_model)
print('\n\n')
print("Train data report")
print("---------------------------------------")
show_model_report(X_train,y_train,lr_model)
print('\n\n')
print("Test data report")
print("---------------------------------------")
show_model_report(X_test,y_test,lr_model)
smk = SMOTETomek(random_state = 42)
X_ov_smk, y_ov_smk = smk.fit_sample(X_train, y_train)
print('Original dataset shape {}'.format(Counter(y_train)))
print('Undersampled dataset shape {}' .format(Counter(y_ov_smk)))
lr_model = LogisticRegression()
lr_model.fit(X_ov_smk, y_ov_smk)

print("Oversampled data report")
print("---------------------------------------")
show_model_report(X_ov_smk, y_ov_smk, lr_model)

print("Train data report")
print("---------------------------------------")
show_model_report(X_train, y_train, lr_model)

print("Test data report")
print("---------------------------------------")
show_model_report(X_test, y_test, lr_model)