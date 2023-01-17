import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


%matplotlib inline
df = pd.read_csv('../input/HR_comma_sep.csv')
df.head()
df.info()
df.describe()
#could use a lamba expression and apply it to all columns, but here a for loop will be fine

for col in ['Work_accident', 'left', 'promotion_last_5years', 'sales', 'salary']:
    df[col] = df[col].astype('category')
df.info()
df.rename(columns = {'average_montly_hours':'avg_monthly_hours', 'Work_accident':'work_accident', 
                     'sales':'dept'}, inplace=True)
df.head()
df.describe()
#df[['work_accident', 'left', 'promotion_last_5years', 'dept', 'salary']].apply(lambda x: x.value_counts()).T.stack()

for col in ['work_accident', 'left', 'promotion_last_5years', 'dept', 'salary']:
    print("---- %s ----" % col)
    print(df[col].value_counts())
colors = ["#1F77B4FF", "#FF7F0EFF"]
sns.countplot(x="left", data=df, palette = colors);
fig, axs = plt.subplots(ncols=3, figsize=(15,8))
sns.countplot(x='work_accident', data=df, ax=axs[0], palette=colors)
sns.countplot(x='left', data=df, ax=axs[1], palette=colors)
sns.countplot(x='promotion_last_5years', data=df, ax=axs[2], palette=colors);
colors = ["#1F77B4FF", "#FF7F0EFF"]
ax = sns.barplot(x=df.left.value_counts().index, y=df.left.value_counts(normalize=True), palette = colors)
ax.set(xlabel='Left Company', ylabel='Percent')
labels = ['Stayed', 'Left']
ax.set_xticklabels(labels)
plt.show();
df['satisfaction_level'].hist();
from scipy.stats import skew

df['satisfaction_level'].skew()
ax = sns.boxplot(x="left", y="satisfaction_level", data=df, palette=colors)
ax.set(xlabel='Left Company', ylabel='Employee Satisfaction')
labels = ['Stayed', 'Left']
ax.set_xticklabels(labels)
plt.show();
from scipy.stats import ttest_ind

stayed = df[df['left']==0] #create 'stayed' dataframe
left = df[df['left']==1] #create 'left' dataframe

##Welch's two sample t-test (assume unequal variances)
ttest_ind(stayed['satisfaction_level'], left['satisfaction_level'], equal_var=False)
import statsmodels.stats.api as sms

cm = sms.CompareMeans(sms.DescrStatsW(stayed['satisfaction_level']), sms.DescrStatsW(left['satisfaction_level']))
print(cm.ttest_ind(usevar='unequal'))
print(cm.tconfint_diff(usevar='unequal'))
ax = sns.barplot(x="left", y="satisfaction_level", data=df, ci=95, palette=colors)
ax.set(xlabel='Left Company', ylabel='Employee Satisfaction')
ax.set_xticklabels(['Stayed', 'Left'])
plt.show();
ax = sns.boxplot(x="left", y="last_evaluation", data=df, palette=colors)
ax.set(xlabel='Left Company', ylabel="Employee's Last Evaluation")
labels = ['Stayed', 'Left']
ax.set_xticklabels(labels)
plt.show();
df.groupby(['left']).agg({'last_evaluation':'mean'})
#use stayed and left dataframes created earlier
##Welch's two sample t-test (assume unequal variances)
ttest_ind(stayed['last_evaluation'], left['last_evaluation'], equal_var=False)
df.groupby(['left']).agg({'time_spend_company':'mean'})
ttest_ind(stayed['time_spend_company'], left['time_spend_company'], equal_var=False)
dept_left = pd.crosstab(df['left'], df['dept']).apply(lambda r: r/r.sum(), axis=0).T
dept_left.rename(columns = {0:'stayed', 1:'left'}, inplace=True)
dept_left
df_left = dept_left['left']
df_left = df_left.reset_index()
df_left = df_left.sort_values(['left'], ascending=False)
df_left.reset_index(inplace=True)
df_left
from matplotlib import cm

fig, ax = plt.subplots()
fig.set_size_inches(11, 8)
sns.barplot(x="dept", y="left", data=df_left, ax=ax)
ax.set(xlabel='Department', ylabel="Percent Attrition");
df['promotion_last_5years'].value_counts(normalize='all')
pd.crosstab(df['left'], df['promotion_last_5years'], normalize='columns')
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
sns.boxplot(x='work_accident', y='satisfaction_level', hue='left', data=df, palette=colors);
#ggplot(df_hr, aes(x = Work_accident, y = satisfaction_level)) + geom_boxplot(aes(fill = left)) + 
#  ylab("Employee Satisfaction") + xlab("Work Accident")
df.isnull().sum() # no missing values, so nothing to do here
df.info()
#since promotion, work accident, and left are just coded as [0,1], I'll change their type back to integer

for col in ['left', 'work_accident', 'promotion_last_5years']:
    df[col] = df[col].astype(int)
df.info()
#use get_dummies for other categorical features
#alternative is to encode labels, then one hot encoding with scikit

df = pd.get_dummies(data=df, columns=['dept', 'salary']) 
df.head()
df.describe()
X_df = df.drop(['left'], axis = 1) #create df with only X features
y_df = df['left']
X = X_df.values
y = y_df.values
X.shape
y.shape
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 42)
from sklearn.preprocessing import StandardScaler

std_scale = StandardScaler().fit(Xtrain)

Xtrain = std_scale.transform(Xtrain)
Xtest = std_scale.transform(Xtest)
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, ratio = 1.0)
X_train_res, y_train_res = smote.fit_sample(Xtrain, ytrain)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train_res,y_train_res)
print(clf)
print('classes: ',clf.classes_)
print('coefficients: ',clf.coef_)
print('intercept :', clf.intercept_)
print(sorted(zip(X_df.columns, *clf.coef_)))
df_coefs = pd.DataFrame(sorted(zip(X_df.columns, *clf.coef_)))#better viewed in dataframe format
df_coefs.rename(columns = {0:'feature', 1:'coef'}, inplace=True)
df_coefs.sort_values(by='coef')
pps=clf.predict_proba(Xtrain) #predicted class probabilities if needed
ypred=clf.predict(Xtrain) #class membership predictions
from sklearn.metrics import accuracy_score
accuracy_score(ytrain, ypred)
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
print(confusion_matrix(ytrain, ypred))
print(classification_report(ytrain, ypred, target_names=['Stayed', 'Left']))
print(precision_score(ytrain,ypred))
from sklearn.metrics import recall_score
print(recall_score(ytrain,ypred))
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X_train_res, y_train_res, cv=5, scoring="accuracy")
cross_val_score(clf, X_train_res, y_train_res, cv=5, scoring="recall")
cross_val_score(clf, X_train_res, y_train_res, cv=5, scoring="precision")
#without oversampling inside the loop

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=5, random_state=42)

for train_index, test_index in skfolds.split(Xtrain, ytrain): #use training data obtained before resampling
    clone_clf = clone(clf)
    X_train_folds = Xtrain[train_index]
    y_train_folds = ytrain[train_index]
    X_test_fold = Xtrain[test_index]
    y_test_fold = ytrain[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
#with oversampling inside the loop

skfolds = StratifiedKFold(n_splits=5, random_state=42)

for train_index, test_index in skfolds.split(Xtrain, ytrain): #use training data obtained before resampling
    clone_clf = clone(clf)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_sample(Xtrain, ytrain)
    X_train_folds = X_train_res[train_index]
    y_train_folds = y_train_res[train_index]
    X_test_fold = X_train_res[test_index]
    y_test_fold = y_train_res[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))
for train_index, test_index in skfolds.split(Xtrain, ytrain): #use training data obtained before resampling
    clone_clf = clone(clf)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_sample(Xtrain, ytrain)
    X_train_folds = X_train_res[train_index]
    y_train_folds = y_train_res[train_index]
    X_test_fold = X_train_res[test_index]
    y_test_fold = y_train_res[test_index]
    
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    print(precision_score(y_test_fold, y_pred))
    print(recall_score(y_test_fold, y_pred))
ypred = clone_clf.predict(Xtest)
print(confusion_matrix(ytest, ypred))
precision_score(ytest, ypred)
recall_score(ytest, ypred)
accuracy_score(ytest, ypred)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

#train/test split and scale
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 42)
std_scale = StandardScaler().fit(Xtrain)
Xtrain = std_scale.transform(Xtrain)
Xtest = std_scale.transform(Xtest)

#ignore class imbalance for now, let's train an RF
#specify number of estimators,nodes, njobs=-1 to use all CPU cores
rf_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1) 
rf_clf.fit(Xtrain, ytrain)
y_pred_rf = rf_clf.predict(Xtrain)
rf_clf.feature_importances_
list(zip(X_df.columns, rf_clf.feature_importances_))
df_coefs = pd.DataFrame(sorted(zip(X_df.columns, rf_clf.feature_importances_)))#better viewed in dataframe format
df_coefs.rename(columns = {0:'feature', 1:'coef'}, inplace=True)
df_coefs.sort_values(by='coef', ascending=False)
print(confusion_matrix(ytrain, y_pred_rf))
recall_score(ytrain,y_pred_rf)
precision_score(ytrain,y_pred_rf)
accuracy_score(ytrain, y_pred_rf)
cross_val_score(rf_clf, Xtrain, ytrain, cv=5, scoring="accuracy")
cross_val_score(rf_clf, Xtrain, ytrain, cv=5, scoring="recall")
cross_val_score(rf_clf, Xtrain, ytrain, cv=5, scoring="precision")
#With Resampling inside the cross-validation loop

skfolds = StratifiedKFold(n_splits=5, random_state=42)

for train_index, test_index in skfolds.split(Xtrain, ytrain): #use training data obtained before resampling
    clone_rf_clf = clone(rf_clf)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_sample(Xtrain, ytrain)
    X_train_folds = X_train_res[train_index]
    y_train_folds = y_train_res[train_index]
    X_test_fold = X_train_res[test_index]
    y_test_fold = y_train_res[test_index]
    
    clone_rf_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_rf_clf.predict(X_test_fold)
    rf_acc = accuracy_score(y_test_fold, y_pred)
    rf_recall = recall_score(y_test_fold, y_pred)
    rf_precision = precision_score(y_test_fold, y_pred)
    print(rf_acc, rf_recall, rf_precision)
y_pred = clone_rf_clf.predict(Xtest)
print(confusion_matrix(ytest, y_pred))
rf_acc = accuracy_score(ytest, y_pred)
rf_recall = recall_score(ytest, y_pred)
rf_precision = precision_score(ytest, y_pred)
print(rf_acc, rf_recall, rf_precision)
