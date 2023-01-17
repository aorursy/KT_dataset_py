import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
breast_cancer = pd.read_csv('../input/data.csv')
breast_cancer.head()
breast_cancer.info()
breast_cancer.shape
breast_cancer.describe()
breast_cancer.groupby('diagnosis').size()
breast_cancer.isnull().sum()
for field in breast_cancer.columns:
    amount = np.count_nonzero(breast_cancer[field] == 0)
    
    if amount > 0:
        print('Number of 0-entries for "{field_name}" feature: {amount}'.format(
            field_name=field,
            amount=amount
        ))
# Features "id" and "Unnamed: 32" are not useful 
feature_names = breast_cancer.columns[2:-1]
X = breast_cancer[feature_names]
# "diagnosis" feature is our class which I wanna predict
y = breast_cancer.diagnosis
class_le = LabelEncoder()
# M -> 1 and B -> 0
y = class_le.fit_transform(breast_cancer.diagnosis.values)
sns.heatmap(
    data=X.corr(),
    annot=True,
    fmt='.2f',
    cmap='RdYlGn'
)

fig = plt.gcf()
fig.set_size_inches(20, 16)

plt.show()
pipe = Pipeline(steps=[
    ('preprocess', StandardScaler()),
    ('feature_selection', PCA()),
    ('classification', LogisticRegression())
])
c_values = [0.1, 1, 10, 100, 1000]
n_values = range(2, 31)
random_state = 42
log_reg_param_grid = [
    {
        'feature_selection__random_state': [random_state],
        'feature_selection__n_components': n_values,
        'classification__C': c_values,
        'classification__penalty': ['l1'],
        'classification__solver': ['liblinear'],
        'classification__multi_class': ['ovr'],
        'classification__random_state': [random_state]
    },
    {
        'feature_selection__random_state': [random_state],
        'feature_selection__n_components': n_values,
        'classification__C': c_values,
        'classification__penalty': ['l2'],
        'classification__solver': ['liblinear', 'newton-cg', 'lbfgs'],
        'classification__multi_class': ['ovr'],
        'classification__random_state': [random_state]
    }
]
strat_k_fold = StratifiedKFold(
    n_splits=10,
    random_state=42
)

log_reg_grid = GridSearchCV(
    pipe,
    param_grid=log_reg_param_grid,
    cv=strat_k_fold,
    scoring='accuracy'
)

log_reg_grid.fit(X, y)

# Best LogisticRegression parameters
print(log_reg_grid.best_params_)
# Best score for LogisticRegression with best parameters
print('Best score for LogisticRegression: {:.2f}%'.format(log_reg_grid.best_score_ * 100))

best_params = log_reg_grid.best_params_
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.20
)

std_scaler = StandardScaler()

X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

pca = PCA(
    n_components=best_params.get('feature_selection__n_components'),
    random_state=best_params.get('feature_selection__random_state')
)

X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

print(pca.explained_variance_ratio_)
print('\nPCA sum: {:.2f}%'.format(sum(pca.explained_variance_ratio_) * 100))

log_reg = LogisticRegression(
    C=best_params.get('classification__C'),
    penalty=best_params.get('classification__penalty'),
    solver=best_params.get('classification__solver'),
    multi_class=best_params.get('classification__multi_class'),
    random_state=best_params.get('classification__random_state'),
)

log_reg.fit(X_train_pca, y_train)

log_reg_predict = log_reg.predict(X_test_pca)
log_reg_predict_proba = log_reg.predict_proba(X_test_pca)[:, 1]

print('LogisticRegression Accuracy: {:.2f}%'.format(accuracy_score(y_test, log_reg_predict) * 100))
print('LogisticRegression AUC: {:.2f}%'.format(roc_auc_score(y_test, log_reg_predict_proba) * 100))
print('LogisticRegression Classification report:\n\n', classification_report(y_test, log_reg_predict))
print('LogisticRegression Training set score: {:.2f}%'.format(log_reg.score(X_train_pca, y_train) * 100))
print('LogisticRegression Testing set score: {:.2f}%'.format(log_reg.score(X_test_pca, y_test) * 100))
outcome_labels = sorted(breast_cancer.diagnosis.unique())

# Confusion Matrix for LogisticRegression
sns.heatmap(
    confusion_matrix(y_test, log_reg_predict),
    annot=True,
    xticklabels=outcome_labels,
    yticklabels=outcome_labels
)
# ROC for LogisticRegression
fpr, tpr, thresholds = roc_curve(y_test, log_reg_predict_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for LogisticRegression')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
strat_k_fold = StratifiedKFold(
    n_splits=10,
    random_state=42
)

std_scaler = StandardScaler()

X_std = std_scaler.fit_transform(X)
X_pca = pca.fit_transform(X_std)

fe_score = cross_val_score(
    log_reg,
    X_pca,
    y,
    cv=strat_k_fold,
    scoring='f1'
)

print("LogisticRegression: F1 after 10-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    fe_score.mean() * 100,
    fe_score.std() * 2
))