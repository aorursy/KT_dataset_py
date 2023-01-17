import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.model_selection import StratifiedKFold, learning_curve, ShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
txn = pd.read_csv('../input/creditcardfraud/creditcard.csv')
txn
txn.describe().T
txn.info()
txn_type = txn['Class'].apply(lambda x: 'Fraud' if x==1 else 'Not Fraud').value_counts()
print('There are {} fraud transactions ({:.2%})'.format(txn_type['Fraud'], txn_type['Fraud']/txn.shape[0]))
print('There are {} safe transactions ({:.2%})'.format(txn_type['Not Fraud'], txn_type['Not Fraud']/txn.shape[0]))
fig, axes = plt.subplots(7,4,figsize=(14,14))
feats = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10','V11', 'V12', 'V13', 'V14', 'V15',
         'V16', 'V17', 'V18', 'V19', 'V20','V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
for i, ax in enumerate(axes.flatten()):
    ax.hist(txn[feats[i]], bins=25, color='green')
    ax.set_title(str(feats[i])+' Distribution', color='brown')
    ax.set_yscale('log')
plt.tight_layout()

max_val = np.max(txn[feats].values)
min_val = np.min(txn[feats].values)
print('All values range: ({:.2f}, {:.2f})'.format(min_val, max_val))
plt.figure(figsize=(14,6))
sns.distplot(txn['Time'])
plt.figure(figsize=(14,6))
sns.distplot(txn['Amount'], hist=False, rug=True)
txn['Amount'] = RobustScaler().fit_transform(txn['Amount'].values.reshape(-1,1))
txn['Time'] = RobustScaler().fit_transform(txn['Time'].values.reshape(-1,1))

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,4))
sns.distplot(txn['Time'], ax=ax1)
sns.distplot(txn['Amount'], hist=False, rug=True, ax=ax2)
X = txn.drop(['Class'], axis=1)
y = txn['Class']

final_Xtrain, final_Xtest, final_ytrain, final_ytest = train_test_split(X,
                                    y, test_size=0.2, stratify=y, random_state=42)

final_Xtrain = final_Xtrain.values
final_Xtest = final_Xtest.values
final_ytrain = final_ytrain.values
final_ytest = final_ytest.values

train_unique_label, train_counts_label = np.unique(final_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(final_ytest, return_counts=True)

print()
print('Proportions [Safe vs Fraud]')
print('Training %: '+ str(100*train_counts_label/len(final_ytrain)))
print('Testing %: '+ str(100*test_counts_label/len(final_ytest)))
# shuffle data first, so it's random
txn = txn.sample(frac=1)

txn_fraud = txn.loc[txn['Class']==1]
txn_safe = txn.loc[txn['Class']==0][:492]

txn_under = pd.concat([txn_fraud, txn_safe])
txn_under = txn_under.sample(frac=1, random_state=41)

txn_under.shape
txn_type = txn_under['Class'].apply(lambda x: 'Fraud' if x==1 else 'Not Fraud').value_counts()
print('Randomly Undersampled Dataset (txn_under):')
print('There are {} fraud transactions ({:.2%})'.format(txn_type['Fraud'], txn_type['Fraud']/txn_under.shape[0]))
print('There are {} safe transactions ({:.2%})'.format(txn_type['Not Fraud'], txn_type['Not Fraud']/txn_under.shape[0]))
fig, axes = plt.subplots(2, 1, figsize=(20,16))

sns.heatmap(txn_under.corr(), annot=True, fmt=".2f", cmap = 'RdYlGn', ax=axes[0])
axes[0].set_title("Balanced Correlation Matrix (Reference This One)", fontsize=20, fontweight='bold')

sns.heatmap(txn.corr(), annot=True, fmt=".2f", cmap = 'RdYlGn', ax=axes[1])
axes[1].set_title('Imbalanced Correlation Matrix', fontsize=20, fontweight='bold')

plt.tight_layout()
fig, axes = plt.subplots(2,3,figsize=(14,8))

high_corr_feats = ['V14', 'V12', 'V10', 'V11', 'V4']

for i, ax in enumerate(axes.flatten()):
    if i == 5:
        ax.axis('off')
        break
    sns.boxplot(x='Class', y=high_corr_feats[i], data=txn_under, ax=ax, palette=sns.color_palette('magma_r', 2))
    ax.set_ylabel(None)
    ax.set_title(label=high_corr_feats[i], fontsize=16, fontweight='bold')
plt.tight_layout()
fig, axes = plt.subplots(2,3,figsize=(14,7))
fig.suptitle('    Fraud Transaction Distributions', fontsize=20, fontweight='bold')

for i, ax in enumerate(axes.flatten()):
    if i == 5:
        ax.axis('off')
        break
    v_fraud = txn_under[txn_under['Class']==1][high_corr_feats[i]].values
    sns.distplot(v_fraud, ax=ax, fit=stats.norm)
    ax.set_title(str(high_corr_feats[i]), fontsize=12)
len(txn_under)
high_corr_feats2 = ['V14', 'V12', 'V10', 'V11', 'V4']

for i in high_corr_feats2:
    v_fraud = txn_under[txn_under['Class']==1][i]

    q75 = np.percentile(v_fraud, 75)
    q25 = np.percentile(v_fraud, 25)
    iqr = q75 - q25

    v_lower, v_upper = q25-1.5*iqr, q75+1.5*iqr
    outliers = [x for x in v_fraud if x > v_upper or x < v_lower]

    print(str(len(outliers))+' '+str(i)+' fraud outliers: '+str(outliers)+'\n')

    txn_under = txn_under.drop(txn_under.index[txn_under[i].isin(outliers) & 
                                     txn_under['Class']==1])
len(txn_under)
fig, axes = plt.subplots(2, 3, figsize=(20,12))
fig.suptitle('    Outlier Reduction', fontsize=20, fontweight='bold')

loc1 = [(0.98, -17.5), (0.98, -17.3), (0.98, -14.5), (0.98, 9.2), (0.98, 10.8)]
loc2 = [(0, -12), (0, -12), (0, -12), (0, 6), (0, 8)]

for i, ax in enumerate(axes.flatten()):
    if i == 5:
        ax.axis('off')
        break
    sns.boxplot(x="Class", y=high_corr_feats[i], data=txn_under, ax=ax, palette=sns.color_palette('magma_r', 2))
    ax.set_title(str(high_corr_feats[i]), fontsize=16, fontweight='bold')
    ax.annotate('Fewer extreme\n     outliers', xy=loc1[i], xytext=loc2[i],
                arrowprops=dict(facecolor='Red'), fontsize=14)
    ax.set_ylabel('')
X = txn_under.drop('Class', axis=1)
y = txn_under['Class']

# Implement dimensionality reductions
X_pca = PCA(n_components=2, random_state=38).fit_transform(X.values)
X_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=37).fit_transform(X.values)
X_tsne = TSNE(n_components=2, random_state=39).fit_transform(X.values)
f, axes = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('    Dimensionality Reductions', fontsize=20, fontweight='bold')

green_patch = mpatches.Patch(color='darkgreen', label='No Fraud')
red_patch = mpatches.Patch(color='darkred', label='Fraud')

dim_red = [X_pca, X_svd, X_tsne]
titles = ['PCA', 'Truncated SVD', 't-SNE']

for i, ax in enumerate(axes):
    ax.scatter(dim_red[i][:,0], dim_red[i][:,1], c=(y == 0), cmap='RdYlGn', label='No Fraud', linewidths=2)
    ax.scatter(dim_red[i][:,0], dim_red[i][:,1], c=(y == 1), cmap='RdYlGn', label='Fraud', linewidths=2)
    ax.set_title(titles[i], fontsize=20)
    ax.grid(True)
    ax.legend(handles=[green_patch, red_patch])
X = txn_under.drop('Class', axis=1)
y = txn_under['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {"Log Reg": LogisticRegression(), "KNN": KNeighborsClassifier(), "SVC": SVC(),
          "D Tree": DecisionTreeClassifier()}

print('Mean cv accuracy on undersampled data. \n')
for name, model in models.items():
    training_acc = cross_val_score(model, X_train, y_train, cv=5)
    print(name+":", str(round(training_acc.mean()*100, 2))+"%")
# Use GridSearchCV to find the best parameters.

print('Mean cv scores on undersampled data after tuning hyperparameters. \n')

# Logistic Regression 
log_reg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]}
grid_log_reg = GridSearchCV(LogisticRegression(max_iter=10000), log_reg_params)
grid_log_reg.fit(X_train, y_train)
log_reg = grid_log_reg.best_estimator_
log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Log Reg: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')

# K Nearest Neighbors
knn_params = {"n_neighbors": list(range(2,6,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params)
grid_knn.fit(X_train, y_train)
knn = grid_knn.best_estimator_
knn_score = cross_val_score(knn, X_train, y_train, cv=5)
print('KNN:     ', round(knn_score.mean() * 100, 2).astype(str) + '%')

# SVC
svc_params = {'C': [0.5, 0.6, 0.7, 0.8, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
grid_svc = GridSearchCV(SVC(), svc_params)
grid_svc.fit(X_train, y_train)
svc = grid_svc.best_estimator_
svc_score = cross_val_score(svc, X_train, y_train, cv=5)
print('SVC:     ', round(svc_score.mean() * 100, 2).astype(str) + '%')

# DescisionTree
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(3,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)
tree = grid_tree.best_estimator_
tree_score = cross_val_score(tree, X_train, y_train, cv=5)
print('D Tree:  ', str(round(tree_score.mean() * 100, 2)) + '%')
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)

fig, axes = plt.subplots(2,2, figsize=(18,12), sharey=True)
classifier = [log_reg, knn, svc, tree]
titles = ["Logistic Regression Learning Curve", "K Nearest Neighbors Learning Curve",
         "Support Vector Classifier Learning Curve", "Decision Tree Classifier Learning Curve"]

for i, ax in enumerate(axes.flatten()):
    train_sizes, train_acc, test_acc = learning_curve(
        classifier[i], X_train, y_train, cv=cv, n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10))
    train_acc_mean = np.mean(train_acc, axis=1)
    train_acc_std = np.std(train_acc, axis=1)
    test_acc_mean = np.mean(test_acc, axis=1)
    test_acc_std = np.std(test_acc, axis=1)
    ax.fill_between(train_sizes, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, alpha=0.3, color="#b2b8b7")
    ax.fill_between(train_sizes, test_acc_mean - test_acc_std, test_acc_mean + test_acc_std, alpha=0.3, color="#46d448")
    ax.plot(train_sizes, train_acc_mean, 'o-', color="#b2b8b7", label="Training accuracy")
    ax.plot(train_sizes, test_acc_mean, 'o-', color="#46d448", label="Cross-validation accuracy")
    ax.set_title(titles[i], fontsize=14)
    ax.set_xlabel('Training size')
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.legend(loc='upper right')
    
plt.ylim(0.86, 1.01);
print ('Model ROC AUC \n')

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5, method="decision_function")
print('Log Reg: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))

knn_pred = cross_val_predict(knn, X_train, y_train, cv=5)
print('KNN: {:.4f}'.format(roc_auc_score(y_train, knn_pred)))

svc_pred = cross_val_predict(svc, X_train, y_train, cv=5, method="decision_function")
print('SVC: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))

tree_pred = cross_val_predict(tree, X_train, y_train, cv=5)
print('D Tree: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knn_fpr, knn_tpr, knn_threshold = roc_curve(y_train, knn_pred)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_train, svc_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)

plt.figure(figsize=(12,6))
plt.title('ROC Curves', fontsize=18)
plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
plt.plot(knn_fpr, knn_tpr, label='K Nearest Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_train, knn_pred)))
plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_train, svc_pred)))
plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
plt.plot([0, 1], [0, 1], 'k--')
plt.axis([-0.01, 1, 0, 1])
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.annotate('Minimum Possible ROC Score (50%)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
            arrowprops=dict(facecolor='Red', shrink=0.05))
plt.legend()
# Predict on X_test
log_reg_pred2 = log_reg.predict(X_test)
knn_pred2 = knn.predict(X_test)
svc_pred2 = svc.predict(X_test)
tree_pred2 = tree.predict(X_test)

log_reg_cf = confusion_matrix(y_test, log_reg_pred2)
knn_cf = confusion_matrix(y_test, knn_pred2)
svc_cf = confusion_matrix(y_test, svc_pred2)
tree_cf = confusion_matrix(y_test, tree_pred2)

fig, axes = plt.subplots(2, 2,figsize=(18,10))
titles = ['Logistic Regression', 'K Nearest Neighbors', 'Suppor Vector Classifier', 'DecisionTree Classifier']
conf_matrix = [log_reg_cf, knn_cf, svc_cf, tree_cf]

fig.suptitle('Confusion Matrices (NearMiss Undersampling)     ', fontsize=20, fontweight='bold')

for i, ax in enumerate(axes.flatten()):
    sns.heatmap(conf_matrix[i], ax=ax, annot=True, fmt='.0f', cmap='magma')
    ax.set_title(titles[i], fontsize=14)
    ax.set_xticklabels(['Predicted\nSafe', 'Predicted\nFraud'], fontsize=10)
    ax.set_yticklabels(['Safe', 'Fraud'], fontsize=10)
print('Logistic Regression:')
print(classification_report(y_test, log_reg_pred2))

print('K Nearest Neighbors:')
print(classification_report(y_test, knn_pred2))

print('Support Vector Classifier:')
print(classification_report(y_test, svc_pred2))

print('DecisionTree Classifier:')
print(classification_report(y_test, tree_pred2))
accuracy_undersample = []
precision_undersample = []
recall_undersample = []
f1_undersample = []
auc_undersample = []

# Cross-Validating correctly to determine real-world performance
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train, test in sss.split(final_Xtrain, final_ytrain):
    pipeline_undersample = imbalanced_make_pipeline(NearMiss(), log_reg)
    model_undersample = pipeline_undersample.fit(final_Xtrain[train], final_ytrain[train])
    prediction_undersample = model_undersample.predict(final_Xtrain[test])
    
    accuracy_undersample.append(pipeline_undersample.score(final_Xtrain[test], final_ytrain[test]))
    precision_undersample.append(precision_score(final_ytrain[test], prediction_undersample))
    recall_undersample.append(recall_score(final_ytrain[test], prediction_undersample))
    f1_undersample.append(f1_score(final_ytrain[test], prediction_undersample))
    auc_undersample.append(roc_auc_score(final_ytrain[test], prediction_undersample))
precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)

y_pred = log_reg.predict(X_train)

# Overfit
print('Cross-Validating on Undersampled/Balanced Data: \n')
print('Accuracy Score: {:.4f}'.format(accuracy_score(y_train, y_pred)))
print('Precision Score: {:.4f}'.format(precision_score(y_train, y_pred)))
print('Recall Score: {:.4f}'.format(recall_score(y_train, y_pred)))
print('F1 Score: {:.4f}'.format(f1_score(y_train, y_pred)))
print('---' * 20)

# True
print('Cross-Validating on Original/Imbalanced Data: \n')
print("Accuracy Score: {:.4f}".format(np.mean(accuracy_undersample)))
print("Precision Score: {:.4f}".format(np.mean(precision_undersample)))
print("Recall Score: {:.4f}".format(np.mean(recall_undersample)))
print("F1 Score: {:.4f}".format(np.mean(f1_undersample)))
y_score_under = log_reg.decision_function(final_Xtest)
undersample_average_precision = average_precision_score(final_ytest, y_score_under)

fig = plt.figure(figsize=(14,5))

precision, recall, _ = precision_recall_curve(final_ytest, y_score_under)

plt.step(recall, precision, color='Green', alpha=0.3, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='Green')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Undersampled Precision-Recall Curve (Avg Score: {0:0.4f})'.format(undersample_average_precision), fontsize=16);
accuracy_oversample = []
precision_oversample = []
recall_oversample = []
f1_oversample = []
auc_oversample = []

# Classifier with optimal parameters - we use RandomizedSearch instead of GridSearch, given large sample size.
log_reg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
rand_log_reg = RandomizedSearchCV(LogisticRegression(random_state=4, max_iter=1000), log_reg_params, n_iter=4)

for train, test in sss.split(final_Xtrain, final_ytrain):
    # Apply SMOTE during training, not cross-validation.
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg)
    model = pipeline.fit(final_Xtrain[train], final_ytrain[train])
    log_reg_sm = rand_log_reg.best_estimator_
    prediction = log_reg_sm.predict(final_Xtrain[test])

    accuracy_oversample.append(pipeline.score(final_Xtrain[test], final_ytrain[test]))
    precision_oversample.append(precision_score(final_ytrain[test], prediction))
    recall_oversample.append(recall_score(final_ytrain[test], prediction))
    f1_oversample.append(f1_score(final_ytrain[test], prediction))
    auc_oversample.append(roc_auc_score(final_ytrain[test], prediction))

print('Cross-Validation on Original/Imbalanced Data (Correct Approach, SMOTE)')
print('')
print("accuracy: {:.4f}".format(np.mean(accuracy_oversample)))
print("precision: {:.4f}".format(np.mean(precision_oversample)))
print("recall: {:.4f}".format(np.mean(recall_oversample)))
print("f1: {:.4f}".format(np.mean(f1_oversample)))
print("auc: {:.4f}".format(np.mean(auc_oversample)))
# Predict on X_test
log_reg_sm_pred = log_reg_sm.predict(X_test)

log_reg_sm_cf = confusion_matrix(y_test, log_reg_sm_pred)

fig, axes = plt.subplots(1, 2,figsize=(18,6))
titles = ['SMOTE Oversampling', 'NearMiss Undersampling']

conf_matrix = [log_reg_sm_cf, log_reg_cf]

fig.suptitle('Logistic Regression Confusion Matrices     ', fontsize=20, fontweight='bold')

for i, ax in enumerate(axes.flatten()):
    sns.heatmap(conf_matrix[i], ax=ax, annot=True, fmt='.0f', cmap='magma')
    ax.set_title(titles[i], fontsize=14)
    ax.set_xticklabels(['Predicted\nSafe', 'Predicted\nFraud'], fontsize=10)
    ax.set_yticklabels(['Safe', 'Fraud'], fontsize=10)
labels = ['No Fraud', 'Fraud']

print('Performance on Undersampled Test Data \n')
print('Logistic Regression, SMOTE Oversampling:')
print(classification_report(y_test, log_reg_sm_pred, target_names=labels))

print('Logistic Regression, NearMiss Undersampling:')
print(classification_report(y_test, log_reg_pred2, target_names=labels))
print('Logistic Regression Performance, Final Testing:\n')
# Logistic regression trained on undersampled data
y_pred = log_reg.predict(final_Xtest)
undersample_accuracy = accuracy_score(final_ytest, y_pred)
print('Undersampling Accuracy: {:.4f}'.format(undersample_accuracy))

# Logistic regression trained on oversampled data
y_pred_sm = log_reg_sm.predict(final_Xtest)
oversample_accuracy = accuracy_score(final_ytest, y_pred_sm)
print('Oversampling Accuracy: {:.4f}'.format(oversample_accuracy))
labels = ['No Fraud', 'Fraud']

print('Logistic Regression Performance, Final Testing: \n')

print('Logistic Regression, NearMiss Undersampling:')
print(classification_report(final_ytest, y_pred, target_names=labels))

print('Logistic Regression, SMOTE Oversampling:')
print(classification_report(final_ytest, y_pred_sm, target_names=labels))
y_score_over = log_reg_sm.decision_function(final_Xtest)
oversample_average_precision = average_precision_score(final_ytest, y_score_over)

fig = plt.figure(figsize=(14,5))

precision, recall, _ = precision_recall_curve(final_ytest, y_score_over)

plt.step(recall, precision, color='Red', alpha=0.3, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='Orange')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('Oversampled Precision-Recall Curve (Avg Score: {0:0.4f})'.format(oversample_average_precision), fontsize=16);
import itertools
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
NN_undersample = Sequential([Dense(X_train.shape[1], input_shape=(X_train.shape[1], ), activation='relu'),
                             Dense(32, activation='relu'),
                             Dense(2, activation='softmax')])
NN_undersample.summary()
NN_undersample.compile(Adam(lr=0.001), metrics=['accuracy'], loss='sparse_categorical_crossentropy')

NN_undersample.fit(X_train, y_train, validation_split=0.2, batch_size=25, epochs=20, shuffle=True, verbose=2)

undersample_pred = NN_undersample.predict_classes(final_Xtest)
def plot_cm(cm, classes, normalize=False, title='Confusion matrix', cmap='Blues'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
undersample_cm = confusion_matrix(final_ytest, undersample_pred)
labels = ['Safe', 'Fraud']

plt.figure(figsize=(6,5))
plot_cm(undersample_cm, labels, title="Random Undersample\nConfusion Matrix")
print(classification_report(final_ytest, undersample_pred, target_names=labels, digits=4))
sm = SMOTE(sampling_strategy='minority', random_state=49)
Xsm_train, ysm_train = sm.fit_sample(final_Xtrain, final_ytrain)

NN_oversample = Sequential([Dense(Xsm_train.shape[1], input_shape=(Xsm_train.shape[1], ), activation='relu'),
                            Dense(32, activation='relu'),
                            Dense(2, activation='softmax')])
NN_oversample.summary()
NN_oversample.compile(Adam(lr=0.001), metrics=['accuracy'], loss='sparse_categorical_crossentropy')

NN_oversample.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)

oversample_pred = NN_oversample.predict_classes(final_Xtest)
oversample_smote = confusion_matrix(final_ytest, oversample_pred)

plt.figure(figsize=(6,5))
plot_cm(oversample_smote, labels, title="SMOTE Oversample\nConfusion Matrix ", cmap=plt.cm.Greens)
print(classification_report(final_ytest, oversample_pred, target_names=labels, digits=4))
