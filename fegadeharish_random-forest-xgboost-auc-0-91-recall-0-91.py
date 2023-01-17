# import tensorflow library to use GPU into this code for faster processing
import tensorflow
# import the required libraries

import zipfile  #read the csv file from zip format without extracting it in drive, we save space
import numpy as np  #linear algebra computations and transformations
import pandas as pd  #read the dataframe and dataframe operations
import matplotlib.pyplot as plt  #visualization of data
import seaborn as sns  #visualization of data
import re  #support for regular expressions

# pd.set_option('display.max_columns', 500)  #set the default option to show all columns when we want to show data

import warnings
warnings.filterwarnings(action='ignore')

from scipy import stats
# read contents of zip file
zf = zipfile.ZipFile('creditcard.csv.zip')

# read the data from csv file into pandas dataframe
cc_fraud = pd.read_csv(zf.open('creditcard.csv'))
# create a copy of original dataframe so as to avoid reading from drive again
df = cc_fraud.copy()
print(df.shape)
df.head()
# We perform descriptive statistics to check the mean and std of each variable. 
df.describe()
print(df.Class.value_counts())
print(df.Class.value_counts(normalize=True))
sns.pairplot(df)
plt.show()
print(f'skewness in Time column: {df.Time.skew():.2f}')
plt.subplots(figsize=(8,6))
# plt.subplot(121)
sns.distplot(df.Time)
# plt.subplot(122)
# stats.probplot(df.Time, plot=plt)
plt.xlabel('Time elapsed in seconds', fontsize=12)
plt.show()
sns.boxplot(df.Time)
plt.show()
bins = [0,25000,50000,75000,100000,125000,150000,175000]
time_bin = pd.cut(df.Time, bins, right=False)
df['Time_Bins'] = time_bin
time_table = pd.crosstab(index=df.Time_Bins, columns=df.Class)

df['Hour'] = df['Time'].apply(lambda x: int(np.ceil(float(x)/3600) % 24))+1
hour_table = df.pivot_table(values='Amount',index='Hour',columns='Class',aggfunc='count', margins=True)

time_table.head()
hour_table.sort_values(by=[1], ascending=False).head(10)
hour_table.plot(kind='bar', stacked=True, figsize=(8,6))
plt.xticks(np.arange(0,25), rotation=0)
plt.xlabel('Hour of the day', fontsize=12)
plt.ylabel('Number of Transactions', fontsize=12)
plt.show()
max_amount_0 = df[df.Class==0].groupby(by='Hour', observed=True).max()['Amount']
min_amount_0 = df[df.Class==0].groupby(by='Hour', observed=True).min()['Amount']
count_0 = df.Hour[df.Class==0].value_counts().sort_index()
max_amount_1 = df[df.Class==1].groupby(by='Hour', observed=True).max()['Amount']
min_amount_1 = df[df.Class==1].groupby(by='Hour', observed=True).min()['Amount']
count_1 = df.Hour[df.Class==1].value_counts().sort_index()
df_time = pd.DataFrame({('0','min_amount_0'):min_amount_0,
                        ('0','max_amount_0'):max_amount_0,
                        ('0','count_0'):count_0,
                        ('1','min_amount_1'):min_amount_1,
                        ('1','max_amount_1'):max_amount_1,
                        ('1','count_1'):count_1}, index=count_0.index)
df_time.sort_values(by=('1','count_1'), ascending=False)
df.drop(['Time_Bins','Hour'], axis=1, inplace=True)
plt.subplots(figsize=(15,10))
plt.subplot(221)
sns.distplot(df.Amount)
plt.title('Distribution plot of Amount')
plt.subplot(222)
stats.probplot(df.Amount, plot=plt)

plt.subplot(223)
sns.distplot(np.log1p(df.Amount))
plt.title('Distribution plot of Amount after Log Transformation')
plt.subplot(224)
stats.probplot(np.log1p(df.Amount), plot=plt)

plt.show()
np.quantile(a=df.Amount, q=[0.25,0.5,0.75])

LW = max(5.6 - (77.165-5.6), 0)
print('LW: ',LW)

UW = 77.165+(77.165-5.6)
print('UW: ',UW)
df1 = df[df.Amount <= 148]
df1.Class.value_counts()
df2 = df1[df1.Amount <= 55]
plt.subplots(figsize=(15,8))
plt.subplot(131)
sns.boxplot(df.Amount, orient='vertical')
plt.title('BoxPlot of original Amount Data')
plt.subplot(132)
sns.boxplot(df1.Amount, orient='vertical')
plt.title('BoxPlot of Amount Data after outlier handling')
plt.subplot(133)
sns.boxplot(df2.Amount, orient='vertical')
plt.title('BoxPlot after handling the ourliers in Data')
plt.show()
df1 = df.copy()
df1.Amount = np.log1p(df1.Amount)

df2 = df1[df1.Amount <= 8]
df2.Class.value_counts()
plt.subplots(figsize=(12,8))
plt.subplot(131)
sns.boxplot(df.Amount, orient='vertical')
plt.title('Given Amount Column')
plt.subplot(132)
sns.boxplot(df1.Amount, orient='vertical')
plt.title('Amount after Log Transformation')
plt.subplot(133)
sns.boxplot(df2.Amount, orient='vertical')
plt.title('Amount after outlier handling')
plt.show()
correlation_matrix = df.corr()
fig = plt.figure(figsize=(15,9))
sns.heatmap(correlation_matrix, vmax=0.8, square = True)
plt.show()
# import necessary modules
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score  #comparison metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor  #feature selection
from statsmodels.tools.tools import add_constant #feature selection computing VIF
ss = StandardScaler()
df2.Time = ss.fit_transform(np.array(df2.Time).reshape(-1,1))
X = add_constant(df2)
# X.drop(['Amount'], axis=1, inplace=True)
# X = pd.get_dummies(X)
X.drop('const', axis=1, inplace=True)
a = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
b = pd.DataFrame(a, columns=['VIF'])
b.sort_values(by='VIF', ascending=False)
X = df2.drop(['Class'], axis=1)
y = df2.Class
X_train = X.sample(frac=0.8, random_state=10)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=10)
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
%%time
rf = RandomForestClassifier(max_depth=15, n_estimators=100, oob_score=True, class_weight='balanced_subsample')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_train = rf.predict(X_train)
print('Train Confusion matrix:\n', confusion_matrix(y_train, y_pred_train))
print('Test Confusion matrix:\n', confusion_matrix(y_test, y_pred_rf))
print('Classification Report:\n', classification_report(y_test, y_pred_rf))
print(f'\nROC Score: {roc_auc_score(y_test, y_pred_rf):.4f}')
print(pd.crosstab(y, y_pred_rf, rownames=['Actual'], colnames=['Predicted'], margins=True))
max_depth = [int(x) for x in np.linspace(5,20,4)]
# max_features = ['auto', 'sqrt']
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]

random_grid = {'max_depth': max_depth}
%%time
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
random_rf = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=10, cv=5, n_jobs=-1, scoring='roc_auc')
%%time
random_rf.fit(X,y)
print(f'grid best params: {random_rf.best_params_}')
print(f'grid best: {random_rf.best_score_}')
%%time
param = {'max_depth':[13,15,18]
         }
xgb = XGBClassifier(subsample=0.7, colsample_bytree=0.8)
grid_xgb = GridSearchCV(estimator=xgb, param_grid=param, scoring='roc_auc', n_jobs=4, cv=5)
grid_xgb.fit(X,y)
print(f'grid best params: {grid_xgb.best_params_}')
print(f'grid best: {grid_xgb.best_score_}')
%%time
xgb=XGBClassifier(max_depth=18, subsample=0.7, scale_pos_weight=1, colsample_bytree=0.8)
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)
y_pred_train = xgb.predict(X_train)
print('Train Confusion matrix:\n', confusion_matrix(y_train, y_pred_train))
print('Test Confusion matrix:\n', confusion_matrix(y_test, y_pred_xgb))
print('Classification Report:\n', classification_report(y_test, y_pred_xgb))
print(f'\nROC Score: {roc_auc_score(y_test, y_pred_xgb):.4f}')
rf_p,rf_r,rf_t = precision_recall_curve(y_test,y_pred_rf)
rf_fpr,rf_tpr,rf_thr = roc_curve(y_test,y_pred_rf)

xgb_p,xgb_r,xgb_t = precision_recall_curve(y_test,y_pred_xgb)
xgb_fpr,xgb_tpr,xgb_thr = roc_curve(y_test,y_pred_xgb)
plt.figure(figsize=(18,8))
plt.subplot(121)
plt.title('ROC Curve', fontsize=16)
plt.plot(rf_fpr, rf_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred_rf)))
plt.plot(xgb_fpr, xgb_tpr, label='XGBoost Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred_xgb)))
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([-0.01, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
            arrowprops=dict(facecolor='#6E726D', shrink=0.05))
plt.legend(loc='lower right', fontsize=12)

plt.subplot(122)
plt.title('Precision Recall Curve', fontsize=16)
plt.plot(rf_r, rf_p, label='Random Forest Classifier Score: {:.4f}'.format(average_precision_score(y_test, y_pred_rf)))
plt.plot(xgb_r, xgb_p, label='XGBoost Classifier Score: {:.4f}'.format(average_precision_score(y_test, y_pred_xgb)))
plt.axis([0, 1.01, 0, 1.01])
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.legend(fontsize=12)

plt.show()