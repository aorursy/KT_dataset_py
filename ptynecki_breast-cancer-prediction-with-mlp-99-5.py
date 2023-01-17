import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, LabelEncoder
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
    ('classification', MLPClassifier())
])
random_state = 42
mlp_activation = ['identity', 'logistic', 'tanh', 'relu']
mlp_solver = ['lbfgs', 'sgd', 'adam']
mlp_max_iter = range(1000, 10000, 1000)
mlp_alpha = [1e-4, 1e-3, 0.01, 0.1, 1]
preprocess = [Normalizer(), MinMaxScaler(), StandardScaler(), RobustScaler(), QuantileTransformer()]
mlp_param_grid = [
    {
        'preprocess': preprocess,
        'classification__activation': mlp_activation,
        'classification__solver': mlp_solver,
        'classification__random_state': [random_state],
        'classification__max_iter': mlp_max_iter,
        'classification__alpha': mlp_alpha
    }
]
# strat_k_fold = StratifiedKFold(
#     n_splits=10,
#     random_state=42
# )

# mlp_grid = GridSearchCV(
#     pipe,
#     param_grid=mlp_param_grid,
#     cv=strat_k_fold,
#     scoring='f1',
#     n_jobs=-1,
#     verbose=2
# )

# mlp_grid.fit(X, y)

# # Best MLPClassifier parameters
# print(mlp_grid.best_params_)
# # Best score for MLPClassifier with best parameters
# print('\nBest F1 score for MLP: {:.2f}%'.format(mlp_grid.best_score_ * 100))

# best_params = mlp_grid.best_params_
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=42,
    test_size=0.32
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
scaler = StandardScaler()

print('\nData preprocessing with {scaler}\n'.format(scaler=scaler))

X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)

mlp = MLPClassifier(
    max_iter=1000,
    alpha=0.1,
    activation='logistic',
    solver='adam',
    random_state=42
)
mlp.fit(X_train_scaler, y_train)

mlp_predict = mlp.predict(X_test_scaler)
mlp_predict_proba = mlp.predict_proba(X_test_scaler)[:, 1]

print('MLP Accuracy: {:.2f}%'.format(accuracy_score(y_test, mlp_predict) * 100))
print('MLP AUC: {:.2f}%'.format(roc_auc_score(y_test, mlp_predict_proba) * 100))
print('MLP Classification report:\n\n', classification_report(y_test, mlp_predict))
print('MLP Training set score: {:.2f}%'.format(mlp.score(X_train_scaler, y_train) * 100))
print('MLP Testing set score: {:.2f}%'.format(mlp.score(X_test_scaler, y_test) * 100))
outcome_labels = sorted(breast_cancer.diagnosis.unique())

# Confusion Matrix for MLPClassifier
sns.heatmap(
    confusion_matrix(y_test, mlp_predict),
    annot=True,
    fmt="d",
    xticklabels=outcome_labels,
    yticklabels=outcome_labels
)
# ROC for MLPClassifier
fpr, tpr, thresholds = roc_curve(y_test, mlp_predict_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for MLPClassifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
strat_k_fold = StratifiedKFold(
    n_splits=10,
    random_state=42
)

scaler = StandardScaler()

X_std = scaler.fit_transform(X)

fe_score = cross_val_score(
    mlp,
    X_std,
    y,
    cv=strat_k_fold,
    scoring='f1'
)

print("MLP: F1 after 10-fold cross-validation: {:.2f}% (+/- {:.2f}%)".format(
    fe_score.mean() * 100,
    fe_score.std() * 2
))