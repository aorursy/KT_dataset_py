import numpy as np
import pandas as pd
import time


import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head()
#observe the different feature type present in the data
df.info()
classes=df['Class'].value_counts()
normal_share=classes[0]/df['Class'].count()*100
fraud_share=classes[1]/df['Class'].count()*100

print("normal_share=",normal_share,"            ","fraud_share=",fraud_share)

imbalance= (fraud_share/normal_share)*100
print('Imbalance Percentage = ' + str(imbalance))
# Create a bar plot for the number and percentage of fraudulent vs non-fraudulent transcations
fig, ax = plt.subplots(1, 2, figsize=(18,4))

classes.plot(kind='bar', rot=0, ax=ax[0])
ax[0].set_title('Number of Class Distributions \n (0: No Fraud || 1: Fraud)')

(classes/df['Class'].count()*100).plot(kind='bar', rot=0, ax=ax[1])
ax[1].set_title('Percentage of Distributions \n (0: No Fraud || 1: Fraud)')

plt.show()
# Create a scatter plot to observe the distribution of classes with time
df.plot.scatter(y='Class', x='Time',figsize=(18,4))
# Create a scatter plot to observe the distribution of classes with Amount
df.plot.scatter(y='Class', x='Amount',figsize=(18,4))
# Drop unnecessary columns
df = df.drop(['Time'],axis=1)
df.head()
y= df['Class']
X= df.loc[:, df.columns != 'Class']
from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, train_size = 0.7, test_size = 0.3, random_state = 42, stratify=y)
print(np.sum(y))
print(np.sum(y_train))
print(np.sum(y_test))
# plot the histogram of a variable from the dataset to see the skewness

k=0
fig, ax = plt.subplots(7, 4, figsize=(20,20))
for i in range(7):
    for j in range(4):
        k=k+1
        sns.distplot(X_train['V'+str(k)], ax=ax[i][j])
        ax[i][j].set_title('V'+str(k))
       
# - Apply : preprocessing.PowerTransformer(copy=False) to fit & transform the train & test data
pt= preprocessing.PowerTransformer(method='yeo-johnson', copy=True)
pt.fit(X_train)                       

X_train_pt = pt.transform(X_train)
X_test_pt = pt.transform(X_test)

y_train_pt = y_train
y_test_pt = y_test
print(X_train_pt.shape)
print(y_train_pt.shape)
# plot the histogram of a variable from the dataset again to see the result 
X_train_pt_df = pd.DataFrame(X_train_pt,columns=X_train.columns)
k=0
fig, ax = plt.subplots(7, 4, figsize=(20,20))
for i in range(7):
    for j in range(4):
        k=k+1
        sns.distplot(X_train_pt_df['V'+str(k)], ax=ax[i][j])
        ax[i][j].set_title('V'+str(k))
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from imblearn import over_sampling

# perfom cross validation on the X_train & y_train 
skf = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)

print("XGBOOST Classifier: --------------------------")
cv_score_mean=0
for n_estimators in [100,50]:
    for learning_rate in [0.2,0.6]:
        for subsample in [0.3, 0.6, 0.9]:
            print("n_estimators=",n_estimators,"learning_rate=",learning_rate, "subsample=",subsample)
            for train_index, test_index in skf.split(X_train_pt, y_train_pt):
                print("Train:", train_index, "Test:", test_index)
                X_train_cv, X_test_cv = X_train_pt[train_index], X_train_pt[test_index]
                y_train_cv, y_test_cv = y_train_pt.iloc[train_index], y_train_pt.iloc[test_index]

                ros = over_sampling.SMOTE(sampling_strategy='minority', random_state=42)
                X_ros_cv,y_ros_cv = ros.fit_resample(X_train_cv,y_train_cv)

                xgboost_classifier= XGBClassifier(n_estimators=n_estimators,
                                                learning_rate=learning_rate,
                                                subsample=subsample, n_jobs=-1)
                xgboost_classifier.fit(X_ros_cv,y_ros_cv)

                y_test_pred= xgboost_classifier.predict_proba(X_test_cv)
                cv_score= metrics.roc_auc_score(y_true=y_test_cv,y_score=y_test_pred[:,1])
                cv_score_mean=cv_score_mean+cv_score
            print("Cross Val ROC-AUC Score=", cv_score_mean/3)
  
clf = XGBClassifier(n_estimators=100,learning_rate=0.2,subsample=0.3, n_jobs=-1) 
ros = over_sampling.SMOTE(sampling_strategy='minority', random_state=42)
X_ros,y_ros = ros.fit_resample(X_train,y_train) 
clf.fit(X_ros.values,y_ros)
y_pred= clf.predict_proba(X_test.values)
score= metrics.roc_auc_score(y_true=y_test,y_score=y_pred[:,1])
print("XGBOOST Classifier Test ROC-AUC Score =", score)
var_imp = []

for i in clf.feature_importances_:
    var_imp.append(i)
print('Top var =', var_imp.index(np.sort(clf.feature_importances_)[-1])+1)
print('2nd Top var =', var_imp.index(np.sort(clf.feature_importances_)[-2])+1)
print('3rd Top var =', var_imp.index(np.sort(clf.feature_importances_)[-3])+1)

# Variable on Index-13 and Index-9 seems to be the top 2 variables
top_var_index = var_imp.index(np.sort(clf.feature_importances_)[-1])
second_top_var_index = var_imp.index(np.sort(clf.feature_importances_)[-2])

X_train_1 = X_train.to_numpy()[np.where(y_train==1.0)]
X_train_0 = X_train.to_numpy()[np.where(y_train==0.0)]

np.random.shuffle(X_train_0)

import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = [20, 20]

plt.scatter(X_train_1[:, top_var_index], X_train_1[:, second_top_var_index], label='Actual Class-1 Examples')
plt.scatter(X_train_0[:X_train_1.shape[0], top_var_index], X_train_0[:X_train_1.shape[0], second_top_var_index],
            label='Actual Class-0 Examples')
plt.legend()
print('Train auc =', metrics.roc_auc_score(y_test, y_pred[:,1]))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred[:,1])
threshold = thresholds[np.argmax(tpr-fpr)]
print(threshold)