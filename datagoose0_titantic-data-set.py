# Load Libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Machine learning Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

print("Libraries Loaded")
# Load Titanic Data (Training & Test Data)
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.head(5)
#Look at Data Types & Missing Values (Training & Test Data)
print("------------- TRAINING DATA -------------")
train.info()
print("\n------------- TEST DATA -------------")
test.info()
# Missing Value Count & Percentage (Training Data)
df1 = (len(train.index)-train.count()).to_frame()
df2 = (100*(len(train.index)-train.count())/len(train.index)).to_frame()
df2.columns = ['% Missing']
df2['N Missing'] = df1
df2
# Missing Value Count & Percentage (Test Data)
df1 = (len(test.index)-test.count()).to_frame()
df2 = (100*(len(test.index)-test.count())/len(test.index)).to_frame()
df2.columns = ['% Missing']
df2['N Missing'] = df1
df2
# Look at Summary Stats for Numerical Data Types (Training Data)
train.describe()
#Look at Summary Stats for Categorical Data Types (Training Data)
train.describe(include=['O'])
#Look for Simple Trends in Survival Rates

train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).agg(['mean', 'count'])
train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).agg(['mean', 'count'])
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).agg(['mean', 'count'])
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).agg(['mean', 'count'])
train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).agg(['mean', 'count'])
# Age Cohort Analysis, generate age groupings

train['AgeCohort'] = train.groupby(level=0)['Age'].min().apply(lambda x: np.floor(x/10).astype(int))
test['AgeCohort'] = test.groupby(level=0)['Age'].min().apply(lambda x: np.floor(x/10).astype(int))
train.head()
train[["AgeCohort", "Survived"]].groupby(['AgeCohort'], as_index=False).agg(['mean', 'count'])
# Look at Age Distributions for Survived vs. Not Survived
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend();
#Look at Class v Age v Survivals
sns.set(style="ticks")
sns.boxplot(x="Pclass", y="Age", hue="Survived", data=train, palette="PRGn")
sns.despine(offset=10, trim=True)
train[["Pclass","Sex", "Survived"]].groupby(['Pclass','Sex'], as_index=False).agg(['mean', 'count'])
# Map Sex to 1/0
train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
# Drop Columns that have any NaN values
train_Pre = train.drop(['PassengerId','Name','Age','Cabin','Embarked','Ticket','Cabin','AgeCohort'], axis=1)
test_Pre = test.drop(['PassengerId','Name','Age','Cabin','Embarked','Ticket','Cabin','AgeCohort'], axis=1)
#Fix that one missing Fare value in the test set
test_Pre["Fare"].fillna(test["Fare"].median(), inplace=True)
#Check resulting data (Training Data)
train_Pre.head()
#Check resulting data (Test Data)
test_Pre.head()
#Prepare Traning Data into X&Y sets
X_train = train_Pre.drop("Survived", axis=1)
Y_train = train_Pre["Survived"]

X_test = test_Pre
#Initial Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_train)
logFullTrain = logreg.score(X_train, Y_train)
logFullTrain #output accuracy
#Look at factors
coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
#Look at crossfold accuracy
from sklearn import model_selection
from sklearn.model_selection import cross_val_score

kfold = model_selection.StratifiedKFold(n_splits=5, random_state=7)
modelCV = LogisticRegression()
results = model_selection.cross_val_score(modelCV, X_train, Y_train, cv=kfold, scoring='accuracy')
logit_fold_accuracy = results.mean()
print("5-fold cross validation average accuracy: %.3f" % (logit_fold_accuracy))
#Generate ROC Curve & AUC Metric for K Folds

from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

X = X_train
y = Y_train

cv = model_selection.StratifiedKFold(n_splits=5)
classifier = LogisticRegression()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

plt.figure(figsize=(12,8))

#Calculate ROC Curve/AUC for Each Split
for traini, testi in cv.split(X, y):
    
    probas_ = classifier.fit(X.loc[traini], y.loc[traini]).predict_proba(X.loc[testi])
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.loc[testi], probas_[:, 1])
    
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1

#Generate ROC Curve Plot Details
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve, Logistic Regression')
plt.legend(loc="lower right")
plt.show()
#Classification Report
from sklearn.metrics import classification_report
print(classification_report(Y_train, Y_pred))
# Examine probability calculation calibration for model
# Estiate probability score, bin based on predicted survival rate, compare against acutal
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

prob_pos = logreg.predict_proba(X_train)[:,1]
fraction_of_positives, mean_predicted_value = calibration_curve(Y_train, prob_pos, n_bins=10)

fig = plt.figure(0, figsize=(10, 10))
ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated");
ax.plot(mean_predicted_value, fraction_of_positives, "s-");
ax.set_ylabel('Actual Portion Survived')
ax.set_title('Calibration Curve, Logistic Regression')
ax2.hist(prob_pos, range=(0, 1), bins=10,histtype="step", lw=2);
ax2.set_xlabel('Predicted Portion to Survive');
ax2.set_ylabel('Count');
#Generate a simple linear correlation measure, so we can compare the calibration of other models below
logit_cal_corr = np.corrcoef(fraction_of_positives, mean_predicted_value)[0,1]
logit_cal_corr
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_scores = logreg.predict_proba(X_train)[:,1]

precision, recall, _ = precision_recall_curve(Y_train, Y_scores)
average_precision = average_precision_score(Y_train, Y_scores)

plt.figure(figsize=(12,8))
plt.step(recall, precision, color='k', alpha=0.5,where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,color='k')

plt.xlabel('Recall');
plt.ylabel('Precision');
plt.ylim([0.0, 1.05]);
plt.xlim([0.0, 1.0]);
plt.title('Precision-Recall Curve, Logistic Regression: Avg. Precision = {0:0.3f}'.format(average_precision));
trues = Y_scores[Y_train==1]
falses = Y_scores[Y_train==0]

plt.figure(figsize=(12,8));
plt.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="o", label='Perished');
plt.scatter([i for i in range(len(trues))], trues, s=25, c='g', marker="o", label='Suvived');
plt.axhline(.5, color='black');

plt.legend(loc='upper right');
plt.title("Default Decision Boundary (Prob = 0.5)");
plt.xlabel('N');
plt.ylabel('Predicted Probability');
# Model Summary Data
df1 = pd.DataFrame({'Model': ['Logistic Regression'],'Metric':['Mean AUC'], 'Stat_Val':[mean_auc], 'Run':['Initial']})
df2 = pd.DataFrame({'Model': ['Logistic Regression'],'Metric':['Calibration Corr'], 'Stat_Val':[logit_cal_corr], 'Run':['Initial']})
model_summary = pd.merge(df1,df2,how='outer')
df3 = pd.DataFrame({'Model': ['Logistic Regression'],'Metric':['K-fold Accuracy'], 'Stat_Val':[logit_fold_accuracy], 'Run':['Initial']})
model_summary = pd.merge(model_summary,df3,how='outer')
df4 = pd.DataFrame({'Model': ['Logistic Regression'],'Metric':['Full Accuracy'], 'Stat_Val':[logFullTrain], 'Run':['Initial']})
model_summary = pd.merge(model_summary,df4,how='outer')
model_summary
# Predict for Test Set & Make First Submission
Y_pred = logreg.predict(X_test)
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
submission.to_csv('Submission_logit.csv', index=False)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_train)
acc_decision_tree = decision_tree.score(X_train, Y_train)
acc_decision_tree
kfold = model_selection.KFold(n_splits=5, random_state=7)
modelCV = DecisionTreeClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, Y_train, cv=kfold, scoring=scoring)
dtree_fold_accuracy = results.mean()
print("5-fold cross validation average accuracy: %.3f" % (dtree_fold_accuracy))
decision_tree = DecisionTreeClassifier(max_depth=3, random_state=0)
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_train)
acc_decision_tree = decision_tree.score(X_train, Y_train)
acc_decision_tree
import graphviz
from sklearn import tree

dot_data = tree.export_graphviz(decision_tree, out_file=None,
                                feature_names=list(X_train.columns),
                               filled=True, rounded=True,
                               special_characters=True)
graph = graphviz.Source(dot_data)
graph
#Generate ROC Curve & AUC Metric for K Folds

from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

X = X_train
y = Y_train

cv = model_selection.StratifiedKFold(n_splits=5)
classifier = DecisionTreeClassifier()

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

plt.figure(figsize=(12,8))

#Calculate ROC Curve/AUC for Each Split
for traini, testi in cv.split(X, y):
    
    probas_ = classifier.fit(X.loc[traini], y.loc[traini]).predict_proba(X.loc[testi])
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.loc[testi], probas_[:, 1])
    
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1

#Generate ROC Curve Plot Details
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve, Decision Tree')
plt.legend(loc="lower right")
plt.show()
#Classification Report
from sklearn.metrics import classification_report
print(classification_report(Y_train, Y_pred))
# Examine probability calculation calibration for model
# Estiate probability score, bin based on predicted survival rate, compare against acutal
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

prob_pos = decision_tree.predict_proba(X_train)[:,1]
fraction_of_positives, mean_predicted_value = calibration_curve(Y_train, prob_pos, n_bins=10)

fig = plt.figure(0, figsize=(10, 10))
ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated");
ax.plot(mean_predicted_value, fraction_of_positives, "s-");
ax.set_ylabel('Actual Portion Survived')
ax.set_title('Calibration Curve, Decision Tree')
ax2.hist(prob_pos, range=(0, 1), bins=10,histtype="step", lw=2);
ax2.set_xlabel('Predicted Portion to Survive');
ax2.set_ylabel('Count');
#Generate a simple linear correlation measure, so we can compare the calibration of other models below
logit_cal_corr = np.corrcoef(fraction_of_positives, mean_predicted_value)[0,1]
logit_cal_corr
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

Y_scores = decision_tree.predict_proba(X_train)[:,1]

precision, recall, _ = precision_recall_curve(Y_train, Y_scores)
average_precision = average_precision_score(Y_train, Y_scores)

plt.figure(figsize=(12,8))
plt.step(recall, precision, color='k', alpha=0.5,where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,color='k')

plt.xlabel('Recall');
plt.ylabel('Precision');
plt.ylim([0.0, 1.05]);
plt.xlim([0.0, 1.0]);
plt.title('Precision-Recall Curve, Logistic Regression: Avg. Precision = {0:0.3f}'.format(average_precision));
trues = Y_scores[Y_train==1]
falses = Y_scores[Y_train==0]

plt.figure(figsize=(12,8));
plt.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="o", label='Perished');
plt.scatter([i for i in range(len(trues))], trues, s=25, c='g', marker="o", label='Suvived');
plt.axhline(.5, color='black');

plt.legend(loc='upper right');
plt.title("Default Decision Boundary (Prob = 0.5)");
plt.xlabel('N');
plt.ylabel('Predicted Probability');
# Model Summary Data
df1 = pd.DataFrame({'Model': ['Decision Tree'],'Metric':['Mean AUC'], 'Stat_Val':[mean_auc], 'Run':['Initial']})
model_summary = pd.merge(model_summary,df1,how='outer')
df2 = pd.DataFrame({'Model': ['Decision Tree'],'Metric':['Calibration Corr'], 'Stat_Val':[logit_cal_corr], 'Run':['Initial']})
model_summary = pd.merge(model_summary,df2,how='outer')
df3 = pd.DataFrame({'Model': ['Decision Tree'],'Metric':['K-fold Accuracy'], 'Stat_Val':[dtree_fold_accuracy], 'Run':['Initial']})
model_summary = pd.merge(model_summary,df3,how='outer')
df4 = pd.DataFrame({'Model': ['Decision Tree'],'Metric':['Full Accuracy'], 'Stat_Val':[acc_decision_tree], 'Run':['Initial']})
model_summary = pd.merge(model_summary,df4,how='outer')
model_summary
# Predict for Test Set
Y_pred = decision_tree.predict(X_test)
submission = pd.DataFrame({"PassengerId": test["PassengerId"],"Survived": Y_pred})
submission.to_csv('Submission_decTree.csv', index=False)
random_forest = RandomForestClassifier(n_estimators=10, random_state=7)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_train)
acc_random_forest = random_forest.score(X_train, Y_train)
acc_random_forest
kfold = model_selection.KFold(n_splits=5, random_state=7)
modelCV = RandomForestClassifier(n_estimators=10)
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, Y_train, cv=kfold, scoring=scoring)
rForest_fold_accuracy = results.mean()
print("5-fold cross validation average accuracy: %.3f" % (rForest_fold_accuracy))
# Look at feature importance
n_features = X_train.shape[1]
plt.barh(range(n_features), random_forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features),list(X_train.columns))
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()
#Generate ROC Curve & AUC Metric for K Folds

from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

X = X_train
y = Y_train

cv = model_selection.StratifiedKFold(n_splits=5)
classifier = RandomForestClassifier(n_estimators=10, random_state=7)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0

plt.figure(figsize=(12,8))

#Calculate ROC Curve/AUC for Each Split
for traini, testi in cv.split(X, y):
    
    probas_ = classifier.fit(X.loc[traini], y.loc[traini]).predict_proba(X.loc[testi])
    
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y.loc[testi], probas_[:, 1])
    
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1

#Generate ROC Curve Plot Details
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve, Random Forest')
plt.legend(loc="lower right")
plt.show()
# Examine probability calculation calibration for model
# Estiate probability score, bin based on predicted survival rate, compare against acutal
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

prob_pos = random_forest.predict_proba(X_train)[:,1]
fraction_of_positives, mean_predicted_value = calibration_curve(Y_train, prob_pos, n_bins=10)

fig = plt.figure(0, figsize=(10, 10))
ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated");
ax.plot(mean_predicted_value, fraction_of_positives, "s-");
ax.set_ylabel('Actual Portion Survived')
ax.set_title('Calibration Curve, Decision Tree')
ax2.hist(prob_pos, range=(0, 1), bins=10,histtype="step", lw=2);
ax2.set_xlabel('Predicted Portion to Survive');
ax2.set_ylabel('Count');
#Classification Report
from sklearn.metrics import classification_report
print(classification_report(Y_train, Y_pred))
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

Y_scores = random_forest.predict_proba(X_train)[:,1]

precision, recall, _ = precision_recall_curve(Y_train, Y_scores)
average_precision = average_precision_score(Y_train, Y_scores)

plt.figure(figsize=(12,8))
plt.step(recall, precision, color='k', alpha=0.5,where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,color='k')

plt.xlabel('Recall');
plt.ylabel('Precision');
plt.ylim([0.0, 1.05]);
plt.xlim([0.0, 1.0]);
plt.title('Precision-Recall Curve, Logistic Regression: Avg. Precision = {0:0.3f}'.format(average_precision));
trues = Y_scores[Y_train==1]
falses = Y_scores[Y_train==0]

plt.figure(figsize=(12,8));
plt.scatter([i for i in range(len(falses))], falses, s=25, c='r', marker="o", label='Perished');
plt.scatter([i for i in range(len(trues))], trues, s=25, c='g', marker="o", label='Suvived');
plt.axhline(.5, color='black');

plt.legend(loc='upper right');
plt.title("Default Decision Boundary (Prob = 0.5)");
plt.xlabel('N');
plt.ylabel('Predicted Probability');
#x=[i for i in range(len(trues))]
x = np.linspace(0,len(trues),len(trues))
#len(x)
#trues
g = sns.jointplot(x, trues)
knn_model = KNeighborsClassifier(n_neighbors = 10)
knn_model.fit(X_train, Y_train)
Y_pred = knn_model.predict(X_train)
acc_knn_model = knn_model.score(X_train, Y_train)
acc_knn_model
kfold = model_selection.KFold(n_splits=5, random_state=7)
modelCV = KNeighborsClassifier(n_neighbors = 10)
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, Y_train, cv=kfold, scoring=scoring)
rForest_fold_accuracy = results.mean()
print("5-fold cross validation average accuracy: %.3f" % (rForest_fold_accuracy))