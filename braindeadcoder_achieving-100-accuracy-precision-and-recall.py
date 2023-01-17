import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp, stats
from itertools import cycle
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
from sklearn.metrics import r2_score, roc_auc_score, roc_curve, auc

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
import seaborn as sns
from pylab import rcParams

%matplotlib inline
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
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
# separate classes into different datasets
normal_class = data.query('Class == 0')
fraudulent_class = data.query('Class == 1')

# randomize the datasets
normal_class = normal_class.sample(frac=1,random_state=69)
fraudulent_class = fraudulent_class.sample(frac=1,random_state=69)
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
normal_class = normal_class.sample(frac=1,random_state=69)
fraudulent_class = fraudulent_class.sample(frac=1,random_state=69)
X = data.drop(['Class'], axis = 1)

y = data['Class']
def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    plt.figure(figsize=(12, 9), dpi=80)
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(X[y==l, 0], X[y==l, 1], c=c, label=l, marker=m)
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
from imblearn.over_sampling import SMOTE

smote = SMOTE(ratio='minority', random_state=69)
X_sm, y_sm = smote.fit_sample(X, y)

plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=69)
# See category counts for test data
category, records = np.unique(y_test, return_counts= True)
cat_counts = dict(zip(category,records))

print(cat_counts)
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train,y_train)
pred_rf = rf_model.predict(X_test)
print(confusion_matrix(y_test,pred_rf))
print()
print(classification_report(y_test,pred_rf))
print("Cohen's Kappa Score:\t",round(cohen_kappa_score(y_test,pred_rf),4)*100)
print()
print("R-Squared Score:\t",round(r2_score(y_test,pred_rf),4)*100)
print()
print("Area Under ROC Curve:\t",round(roc_auc_score(y_test,pred_rf),4)*100)
'''
# Checking 10-fold Cross-Validation Score for this model

kfold = StratifiedKFold(n_splits=5, random_state=69)

# use area under the precision-recall curve to show classification accuracy
scoring = 'roc_auc'
results = cross_val_score(rf_model, X_sm, y_sm, cv=kfold, scoring = scoring)
print( "AUC: %.3f (%.3f)" % (results.mean(), results.std()) )
'''
'''
# change size of Matplotlib plot
fig_size = plt.rcParams["figure.figsize"] # Get current size

old_fig_params = fig_size
# new figure parameters
fig_size[0] = 15
fig_size[1] = 10
   
plt.rcParams["figure.figsize"] = fig_size # set new size
'''
'''
# plot roc-curve
# code adapted from http://scikit-learn.org

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
for (train, test), color in zip(kfold.split(X_sm, y_sm), colors):
    probas_ = rf_model.fit(X_sm[train], y_sm[train]).predict_proba(X_sm[test])
    # Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(y_sm[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= kfold.get_n_splits(X_sm, y_sm)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
'''
