import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import datetime
%matplotlib inline
import time
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.metrics import average_precision_score
from sklearn import model_selection
import re
import io
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
# pd.set_option('display.height', 1000)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
fontsize = 21
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#Clean and transform data

df = df.convert_objects(convert_numeric=True)
df = df.replace('No internet service', 'No Service')
df = df.replace('No phone service', 'No Service')
df['PaymentMethod'] = df['PaymentMethod'].replace({'Electronic check':'Electronic', 'Mailed check':'Mailed'}, regex=True)
df['PaymentMethod'] = df['PaymentMethod'].str.replace(re.escape('Bank transfer (automatic)'), 'Bank Transfer')
df['PaymentMethod'] = df['PaymentMethod'].str.replace(re.escape('Credit (automatic)'), 'Credit Card')
df['Target'] = df['Churn'].map({'Yes':1, 'No':0})
print('Shape before cleaning {}'.format(df.shape))
df = df.dropna()
print('Shape after cleaning {}'.format(df.shape))
#Create Dummy variables for categorical columns
dummies_df = pd.get_dummies(df[['gender',
 'SeniorCitizen',
 'Partner',
 'Dependents',
 'tenure',
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtection',
 'TechSupport',
 'StreamingTV',
 'StreamingMovies',
 'Contract',
 'PaperlessBilling',
 'PaymentMethod',
 'MonthlyCharges',
 'TotalCharges',                               
 'Churn']], drop_first=False)
#Initiate a random state to ensure results are reproducable
X = dummies_df.iloc[:,0:29]
y = dummies_df['Churn_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
models = []
models.append(('LR1', LogisticRegression()))
models.append(('LR2', LogisticRegression(C=100, penalty='l2', solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=20)))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RFC', RandomForestClassifier()))
models.append(('ADA', AdaBoostClassifier()))
models.append(('GBC', GradientBoostingClassifier()))
# models.append(('MLP', MLPClassifier()))
models.append(('ETC', ExtraTreeClassifier()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('BCL', BaggingClassifier()))

# evaluate each model in turn
results = []
names = []
scoring = 'roc_auc'

# parameters = ["accuracy", "average_precision", "f1", "f1_micro", 'f1_macro', 'f1_weighted', 'precision', "roc_auc"]

for name, model in models:
    kfold = model_selection.KFold(n_splits = 15, random_state = 7)
    cv_results = model_selection.cross_val_score(model, X, y, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# boxplot algorithm comparison
fig = plt.figure(figsize=(26,8))
fig.suptitle('Algorithm Comparison', fontsize=fontsize)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.show()
clf = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=500, max_depth=8, max_features='sqrt', subsample=0.8)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
kfold = model_selection.KFold(n_splits = 10, random_state = 42)
scores = cross_val_score(clf, X, y, cv=kfold, scoring='roc_auc')
print('Standard Classifier is prediciting at: {}'.format(metrics.accuracy_score(y_test, predictions)))
print('Cross Validation Scores are: {}'.format(scores))
print('Cross Validation Score Averages are: {}'.format(scores.mean()))
# gb_grid_params = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
#               'max_depth': [4, 6, 8],
#               'min_samples_leaf': [20, 50,100,150],
#               'max_features': [1.0, 0.3, 0.1] 
#               }

# grid = GridSearchCV(clf, param_grid=gb_grid_params, cv=10, scoring="roc_auc", n_jobs=3)

# grid.fit(X,y)
# #grid.cv_results_

# C = grid.cv_results_['param_learning_rate']
# a = grid.cv_results_['mean_test_score']

# plt.plot(C, a, 'b')
# plt.plot(C, a, 'bo')
# plt.xlabel('C')
# plt.ylabel('accuracy')
# plt.show()

# print (grid.cv_results_['rank_test_score'])
# print (np.mean(grid.cv_results_['mean_test_score']))
proba = clf.predict_proba(X_train)[:,1]
confusion_matrix = pd.DataFrame(confusion_matrix(y_test, predictions), columns=["Predicted False", "Predicted True"], index=["Actual False", "Actual True"])
confusion_matrix
fpr, tpr, threshold = roc_curve(y_train, proba)
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b')
plt.plot([0, 1], [0, 1],'g--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive')
plt.xlabel('False Positive')
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.show()
#Assess Feature Importance
df_f = pd.DataFrame(clf.feature_importances_, columns=["Importance"])
df_f['Labels'] = X_train.columns
df_f.sort_values("Importance", inplace=True, ascending=True)
df_f.set_index('Labels').sort_values(by='Importance', ascending=True)[15:].plot(kind='barh', figsize=(20,9), width=0.85)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.tight_layout()
plt.show()
#Group into High Medium Low Value Users
X_train['Churn_Probability'] = clf.predict_proba(X_train)[:,1]*100
X_test['Churn_Probability'] = clf.predict_proba(X_test)[:,1]*100
churn_df = X_train.append(X_test)
churn_df['Churn_Group'] = pd.cut(churn_df["Churn_Probability"], bins=[-1, 33, 66, 100], labels=["Low", "Medium", "High"])
churn_df['Churn_Number'] = np.digitize(churn_df['Churn_Probability'], range(0,100,10))
churn_df = churn_df.merge(df, how='inner')
churn_df_1 = churn_df[churn_df['Churn'] == 'Yes']
#Review Churn Probability by Tenure
#The longer someone stays with a telco, the less likely they are to churn

ax1 = plt.subplot(121)
churn_df.groupby(['tenure'])['Churn_Probability'].median().plot(linewidth=4, label='Median', ax=ax1, figsize=(24,8))
churn_df.groupby(['tenure'])['Churn_Probability'].mean().plot(linewidth=4, label='Mean', ax=ax1, figsize=(24,8))
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.title('All Users', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend()

ax2 = plt.subplot(122)
churn_df_1.groupby(['tenure'])['Churn_Probability'].median().plot(linewidth=4, label='Median', ax=ax2, figsize=(24,8))
churn_df_1.groupby(['tenure'])['Churn_Probability'].mean().plot(linewidth=4, label='Mean', ax=ax2, figsize=(24,8))
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.title('Churn Customers', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend()

plt.tight_layout()
plt.show()
ax1 = plt.subplot(231)
churn_df_1.groupby(['gender', 'Churn_Group']).size().unstack().plot(kind='bar', figsize=(24,14), width=0.85, ax=ax1)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.xlabel(s='Gender', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

ax2 = plt.subplot(232)
churn_df_1.groupby(['TechSupport', 'Churn_Group']).size().unstack().plot(kind='bar', figsize=(24,14), width=0.85, ax=ax2)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.xlabel(s='Tech Support', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

ax3 = plt.subplot(233)
churn_df_1.groupby(['Contract', 'Churn_Group']).size().unstack().plot(kind='bar', figsize=(24,14), width=0.85, ax=ax3)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.xlabel(s='Contract', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

ax4 = plt.subplot(234)
churn_df_1.groupby(['DeviceProtection', 'Churn_Group']).size().unstack().plot(kind='bar', figsize=(24,14), width=0.85, ax=ax4)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.xlabel(s='Device Protection', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

ax5 = plt.subplot(235)
churn_df_1.groupby(['InternetService', 'Churn_Group']).size().unstack().plot(kind='bar', figsize=(24,14), width=0.85, ax=ax5)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.xlabel(s='Internet Service', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

ax6 = plt.subplot(236)
churn_df_1.groupby(['PaymentMethod', 'Churn_Group']).size().unstack().plot(kind='bar', figsize=(24,14), width=0.85, ax=ax6)
plt.grid(linewidth=1, alpha=0.3, color='lightgrey')
plt.xlabel(s='Payment Method', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.tight_layout()
plt.show()
