import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report

from sklearn.metrics import r2_score, roc_auc_score

from sklearn.metrics import average_precision_score, auc, roc_curve, precision_recall_curve



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier



from sklearn.preprocessing import StandardScaler

import seaborn as sns

from imblearn.over_sampling import SMOTE



import featuretools as ft

import gc



%matplotlib inline

sns.set(style='whitegrid', palette='inferno', font_scale=1.5)



import warnings

warnings.filterwarnings(action="ignore")
# load the data

data = pd.read_csv("../input/creditcard.csv")
# get column names

colNames = data.columns.values

colNames
# get dataframe dimensions

print ("Dimension of dataset:", data.shape)
# get attribute summaries

print(data.describe())
# get class distribution

print ("Normal transaction:", data['Class'][data['Class']==0].count()) #class = 0

print ("Fraudulent transaction:", data['Class'][data['Class']==1].count()) #class = 1
sns.countplot(data['Class'])
# separate classes into different datasets

normal_class = data.query('Class == 0')

fraudulent_class = data.query('Class == 1')



# randomize the datasets

normal_class = normal_class.sample(frac=1,random_state=1210)

fraudulent_class = fraudulent_class.sample(frac=1,random_state=1210)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,9))

f.suptitle('Time of transaction vs Amount by class')



ax1.scatter(fraudulent_class.Time, fraudulent_class.Amount)

ax1.set_title('Fraud')



ax2.scatter(normal_class.Time, normal_class.Amount)

ax2.set_title('Normal')



plt.xlabel('Time (in Seconds)')

plt.ylabel('Amount')

plt.show()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15,9))

f.suptitle('Amount per transaction by class')



bins = 50



ax1.hist(fraudulent_class.Amount, bins = bins)

ax1.set_title('Fraud')



ax2.hist(normal_class.Amount, bins = bins)

ax2.set_title('Normal')



plt.xlabel('Amount ($)')

plt.ylabel('Number of Transactions')

plt.xlim((0, 20000))

plt.yscale('log')

plt.show();
data = data.drop(['Time'], axis=1)

data['Amount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
# separate classes into different datasets

normal_class = data.query('Class == 0')

fraudulent_class = data.query('Class == 1')



# randomize the datasets

normal_class = normal_class.sample(frac=1,random_state=1210)

fraudulent_class = fraudulent_class.sample(frac=1,random_state=1210)
X = data.drop(['Class'], axis = 1)



y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1210)
X_train.head()
gc.collect()
def score(model, test = X_test, y_true = y_test):

    

    pred = model.predict(test)



    print('Average precision-recall score RF:\t', round(average_precision_score(y_true, pred),4)*100)

    print()

    print("Cohen's Kappa Score:\t",round(cohen_kappa_score(y_true,pred),4)*100)

    print()

    print("R-Squared Score:\t",round(r2_score(y_true,pred),4)*100)

    print()

    print("Area Under ROC Curve:\t",round(roc_auc_score(y_true,pred),4)*100)

    print()

    print(classification_report(y_true,pred))

    

    

    precision, recall, _ = precision_recall_curve(y_true, pred)



    plt.step(recall, precision, color='b', alpha=0.2, where='post')

    

    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')



    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.ylim([0.0, 1.05])

    plt.xlim([0.0, 1.0])

    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision_score(y_true, pred)))

    

    

    

    fpr_rf, tpr_rf, _ = roc_curve(y_true, pred)

    roc_auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(8,8))

    plt.xlim([-0.01, 1.00])

    plt.ylim([-0.01, 1.01])

    plt.step(fpr_rf, tpr_rf, lw=1, label='{} curve (AUC = {:0.2f})'.format('RF',roc_auc_rf))

    #plt.fill_between(fpr_rf, tpr_rf, step='post', alpha=0.2, color='b')





    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.title('ROC curve', fontsize=16)

    plt.legend(loc='lower right', fontsize=13)

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')

    plt.axes().set_aspect('equal')

    plt.show()

    
def plot_2d_space(X, y, label='Classes'):   

    colors = ['#1F77B4', '#FF7F0E']

    markers = ['o', 's']

    plt.figure(figsize=(12, 9), dpi=80)

    for l, c, m in zip(np.unique(y), colors, markers):

        plt.scatter(X[y==l, 0], X[y==l, 1], c=c, label=l, marker=m)

    plt.title(label)

    plt.legend(loc='upper right')

    plt.show()
#np.unique(y_train, return_counts= True)

sns.countplot(y_train)
plot_2d_space(np.array(X), np.array(y), 'Before SMOTE over-sampling')
smote = SMOTE(ratio='minority', random_state=1210)

X_sm, y_sm = smote.fit_sample(X_train, y_train)



#np.unique(y_sm, return_counts= True)

sns.countplot(y_sm)
plot_2d_space(X_sm, y_sm, 'After SMOTE over-sampling')
# See category counts for test data

category, records = np.unique(y_test, return_counts= True)

cat_counts = dict(zip(category,records))



print(cat_counts)

sns.countplot(y_test)
rf_model = RandomForestClassifier(n_estimators=500,verbose=1,n_jobs=8)
rf_model.fit(X_sm,y_sm)
score(rf_model)
xgb_model = XGBClassifier(n_estimators=500, n_jobs=8)



xgb_model.fit(X_sm,y_sm)
score(xgb_model, test= np.array(X_test))
lr_model = LogisticRegression(max_iter=1000)



lr_model.fit(X_sm,y_sm)
score(lr_model)
import lightgbm
lgbm = lightgbm.LGBMClassifier(n_estimators=1000, verbose=1)
lgbm.fit(X_sm, y_sm)
score(lgbm)
normal_class.head(3)
fraudulent_class.head(3)
resampled = normal_class.sample(n=int(len(fraudulent_class)*3), random_state=1210)
len(resampled)
data = pd.concat([fraudulent_class,resampled])
X_tr, X_te, y_tr, y_te = train_test_split(data.drop('Class',axis=1), data['Class'], test_size=0.2, random_state=1210)
sns.countplot(data['Class'])
score(RandomForestClassifier(n_estimators=500,random_state=1210).fit(X_tr,y_tr),test=X_te, y_true=y_te)
score(lightgbm.LGBMClassifier(n_estimators=5000, random_state=1210).fit(X_tr,y_tr),test=X_te, y_true=y_te)