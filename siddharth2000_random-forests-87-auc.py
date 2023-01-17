# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
df = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')
df.head()
df.info()
df.describe()
sns.set(style='whitegrid')
sns.countplot(df['Response'])
sns.countplot(df['Driving_License'])
df[df['Driving_License'] == 0].describe()
sns.countplot(df['Response'], hue=df['Previously_Insured'])
sns.distplot(df['Age'])
sns.violinplot(df['Age'])
sns.countplot(df['Gender'])
sns.countplot(df['Gender'], hue=df['Response'])
sns.countplot(df['Response'] ,hue=df['Previously_Insured'])
sns.countplot(df['Vehicle_Damage'])
sns.countplot(df['Response'], hue=df['Vehicle_Damage'])
sns.countplot(df['Response'], hue=df['Vehicle_Age'])
df.head()
df.groupby('Region_Code')['Response'].agg('mean').sort_values().head(10)
sns.distplot(df['Annual_Premium'])
sns.boxplot(df['Annual_Premium'])
df['Annual_Premium'].describe()
high_premium = df[df['Annual_Premium'] >39400.00]
high_premium.describe()
print(df['Response'].value_counts())
print(high_premium['Response'].value_counts())
sns.countplot(high_premium['Response'])
sns.scatterplot(df['Annual_Premium'], df['Response'])
# gender
df['Gender'] = df['Gender'].map({'Female':0, 'Male':1}).astype('int')
df=pd.get_dummies(df, drop_first=True)
df = df.rename(columns = {'Vehicle_Age_< 1 Year':'AgeOneYear',
                          'Vehicle_Age_> 2 Years':'AgeTwoYears',
                          'Vehicle_Damage_Yes':'Vehicle_Damage'})
df['AgeOneYear'] = df['AgeOneYear'].astype('int')
df['AgeTwoYears'] = df['AgeTwoYears'].astype('int')
df['Vehicle_Damage'] = df['Vehicle_Damage'].astype('int')
df['HighPremium'] = np.where(df['Annual_Premium'] > 39400.00, 1, 0)
df.describe()
X = df.drop(['id', 'Response'], axis=1)
y = df['Response']

X_train,X_test,y_train,y_test = train_test_split(X,y, random_state = 0, stratify=y)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l1','l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)

grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=3, scoring='f1',error_score=0)
grid_result = grid_search.fit(X_train_sc, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
lr = LogisticRegression(C=10, class_weight='balanced')
lr.fit(X_train_sc, y_train)

y_pred = lr.predict(X_test_sc)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# get importance
importance = lr.coef_[0]
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in X_train.columns], importance)
plt.xticks(rotation=90)
plt.show()
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


# plot no skill and model precision-recall curves
def plot_pr_curve(y_test, model_probs):
    # calculate the no skill line as the proportion of the positive class
    no_skill = len(y_test[y_test==1]) / len(y_test)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # plot model precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, model_probs)
    # convert to f score
#     fscore = (2 * precision * recall) / (precision + recall)
#     # locate the index of the largest f score
#     ix = np.argmax(fscore)
#     print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    plt.plot(recall, precision, marker='.', label='Logistic')
    #plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
yhat = lr.predict_proba(X_test_sc)
model_probs = yhat[:, 1]
# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(y_test, model_probs)
auc_score = auc(recall, precision)
print('Logistic PR AUC: %.3f' % auc_score)
# plot precision-recall curves
plot_pr_curve(y_test, model_probs)
from sklearn.metrics import roc_auc_score, roc_curve

print('Area under curve score for Logistic Regression is: ', roc_auc_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
rf1 = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=4,
                             max_features='auto', bootstrap=True, min_samples_leaf=4,
                            class_weight='balanced_subsample')
rf1.fit(X_train, y_train)
y_pred = rf1.predict(X_test)
print(confusion_matrix(y_train, rf1.predict(X_train)))
print('Accuracy of our model is: ', accuracy_score(y_train, rf1.predict(X_train)))
print(confusion_matrix(y_test, y_pred))
print('Accuracy of our model is: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Area under curve score for Random Forests is: ', roc_auc_score(y_test, y_pred))
print('Kappa score for Random Forests',cohen_kappa_score(y_test, y_pred))
yhat = rf1.predict_proba(X_test)
model_probs = yhat[:, 1]
# calculate the precision-recall auc
precision, recall, _ = precision_recall_curve(y_test, model_probs)
auc_score = auc(recall, precision)
print('Logistic PR AUC: %.3f' % auc_score)
# plot precision-recall curves
plot_pr_curve(y_test, model_probs)
features = pd.DataFrame()
features['Feature'] = X_train.columns
features['Importance'] = rf1.feature_importances_
features.sort_values(by=['Importance'], ascending=False, inplace=True)
features.set_index('Feature', inplace=True)
features.plot(kind='bar', figsize=(20, 10))
# predict probabilities
pred_prob1 = lr.predict_proba(X_test_sc)
pred_prob2 = rf1.predict_proba(X_test)

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

# roc curve for tpr = fpr 
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])

print('AUC for Logistic Regression', auc_score1, 
      'AUC for Random Forests', auc_score2)

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Random Forests')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=200, min_samples_split=5,max_depth=6,
                                max_features = 'auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy of our model is: ', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Area under curve score for GBM is: ', roc_auc_score(y_test, y_pred))
print('Kappa score for GBM',cohen_kappa_score(y_test, y_pred))
