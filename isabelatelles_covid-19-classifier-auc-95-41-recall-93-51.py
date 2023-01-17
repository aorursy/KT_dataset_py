import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import numpy as np
dataset = pd.read_excel('dataset.xlsx', index_col=0)
dataset['Urine - pH'].replace('Não Realizado', np.nan, inplace=True)
dataset['Urine - pH'] = dataset['Urine - pH'].astype('float64')
dataset.replace('not_done', np.nan, inplace=True)
dataset['Urine - Leukocytes'].replace('<1000', '999', inplace=True)
dataset['Urine - Leukocytes'] = dataset['Urine - Leukocytes'].astype('float64')
dataset.replace('not_detected', 0, inplace=True)
dataset.replace('detected', 0, inplace=True)
dataset.replace('negative', 0, inplace=True)
dataset.replace('positive', 1, inplace=True)
dataset.replace('absent', 0, inplace=True)
dataset.replace('present', 1, inplace=True)
df_temp = dataset[['Urine - Aspect', 'Urine - Urobilinogen', 'Urine - Crystals', 'Urine - Color']].astype("str").apply(LabelEncoder().fit_transform)
dataset[['Urine - Aspect', 'Urine - Urobilinogen', 'Urine - Crystals', 'Urine - Color']] = df_temp.where(~dataset[['Urine - Aspect', 'Urine - Urobilinogen', 'Urine - Crystals', 'Urine - Color']].isna(), dataset[['Urine - Aspect', 'Urine - Urobilinogen', 'Urine - Crystals', 'Urine - Color']])
dataset['Urine - Aspect'] = dataset['Urine - Aspect'].astype("float64")
dataset['Urine - Urobilinogen'] = dataset['Urine - Urobilinogen'].astype("float64")
dataset['Urine - Crystals'] = dataset['Urine - Crystals'].astype("float64")
dataset['Urine - Color'] = dataset['Urine - Color'].astype("float64")
dataset.drop(columns=['Patient addmited to regular ward (1=yes, 0=no)',
                      'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                      'Patient addmited to intensive care unit (1=yes, 0=no)'], inplace=True)
def plot_missing_data(missing_data, title):
    f, ax = plt.subplots(figsize=(15, 6))
    plt.xticks(rotation='90')
    sns.barplot(x=missing_data.index, y=missing_data['Percent'])
    plt.xlabel('Features', fontsize=15)
    plt.ylabel('Percent of missing values', fontsize=15)
    plt.title(title, fontsize=15)
def get_missing_data(dataset):
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = (dataset.isnull().sum()/dataset.isnull().count()).sort_values(ascending=False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data = get_missing_data(dataset)
plot_missing_data(missing_data, 'Percent missing data by feature')
missing_data.head(10)
dataset_positive = dataset[dataset['SARS-Cov-2 exam result'] == 1]
missing_data_positive = get_missing_data(dataset_positive)

plot_missing_data(missing_data_positive, 'Percent positive missing data by feature')
missing_data_positive.head(10)
dataset_negative = dataset[dataset['SARS-Cov-2 exam result'] == 0]
missing_data_negative = get_missing_data(dataset_negative)

plot_missing_data(missing_data_negative, 'Percent negative missing data by feature')
missing_data_negative.head(10)
corrmat = abs(dataset.corr())
# Correlation with output variable
cor_target = corrmat["SARS-Cov-2 exam result"]
# Selecting highly correlated features
relevant_features = cor_target[cor_target>0.15].index.tolist()
f, ax = plt.subplots(figsize=(16, 8))
sns.heatmap(abs(dataset[relevant_features].corr().iloc[0:1, :]), yticklabels=[relevant_features[0]], xticklabels=relevant_features, vmin = 0.0, square=True, annot=True, vmax=1.0, cmap='RdPu')
nof_positive_cases = len(dataset_positive.index)
nof_negative_cases = len(dataset_negative.index)
fig1, ax1 = plt.subplots()
ax1.pie([nof_positive_cases, nof_negative_cases], labels=['Positive cases', 'Negative cases'], autopct='%1.1f%%', startangle=90, colors=['#c0ffd5', '#ffc0cb'])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
columns_to_exclude = missing_data_positive.index[missing_data_positive['Percent'] > 0.998].tolist()
dataset.drop(columns=columns_to_exclude, inplace=True)
print(columns_to_exclude)
# Redefine dataset positive and negative
dataset_negative = dataset[dataset['SARS-Cov-2 exam result'] == 0]
dataset_positive = dataset[dataset['SARS-Cov-2 exam result'] == 1]
dataset_negative = dataset_negative.dropna(axis=0, thresh=20)
X = pd.concat([dataset_negative, dataset_positive])
nof_positive_cases = len(dataset_positive.index)
nof_negative_cases = len(dataset_negative.index)
fig1, ax1 = plt.subplots()
ax1.pie([nof_positive_cases, nof_negative_cases], labels=['Positive cases', 'Negative cases'], autopct='%1.1f%%', startangle=90, colors=['#c0ffd5', '#ffc0cb'])
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
corrmat = abs(X.corr())
# Correlation with output variable
cor_target = corrmat["SARS-Cov-2 exam result"]
# Selecting highly correlated features
relevant_features = cor_target[cor_target>0.15].index.tolist()
f, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(abs(X[relevant_features].corr().iloc[0:1, :]), yticklabels=[relevant_features[0]], xticklabels=relevant_features, vmin = 0.0, square=True, annot=True, vmax=1.0, cmap='RdPu')
X_with_relevant_features = X[relevant_features]
y_with_relevant_features = X_with_relevant_features['SARS-Cov-2 exam result']
X_with_relevant_features.drop(columns=['SARS-Cov-2 exam result'], inplace=True)
y = X['SARS-Cov-2 exam result']
X.drop(columns=['SARS-Cov-2 exam result'], inplace=True)
def print_scores(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Precision: %.2f%% " % (precision * 100))
    print("Recall: %.2f%% " % (recall * 100))
    print("AUC: %.2f%% " % (roc * 100))
def plot_confusion_matrix(y_test, y_pred):
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(confusion_matrix, annot=True, cmap='RdPu')
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_with_relevant_features, y_with_relevant_features, test_size=0.2, random_state=42)
print("Number of samples in train set: %d" % y_train_rf.shape)
print("Number of positive samples in train set: %d" % (y_train_rf == 1).sum(axis=0))
print("Number of negative samples in train set: %d" % (y_train_rf == 0).sum(axis=0))
print()
print("Number of samples in test set: %d" % y_test_rf.shape)
print("Number of positive samples in test set: %d" % (y_test_rf == 1).sum(axis=0))
print("Number of negative samples in test set: %d" % (y_test_rf == 0).sum(axis=0))
imp = SimpleImputer(strategy='median')
imp = imp.fit(X_with_relevant_features)
rfc = RandomForestClassifier()

# Define parameters and grid search
n_estimators = [100, 300, 500, 800, 1000]
max_depth = [5, 8, 15, 25, 30]
grid = dict(n_estimators=n_estimators, max_depth=max_depth)
grid_search = GridSearchCV(estimator=rfc, param_grid=grid, n_jobs=-1, cv=10, scoring='recall', error_score=0)
grid_result = grid_search.fit(imp.transform(X_train_rf), y_train_rf)
print("Best recall: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
rfc.n_estimators = grid_result.best_params_['n_estimators']
rfc.max_depth = grid_result.best_params_['max_depth']
                                   
model_rfc = rfc.fit(imp.transform(X_train_rf), y_train_rf)
y_pred_rf = model_rfc.predict(imp.transform(X_test_rf))
print_scores(y_test_rf, y_pred_rf)
plot_confusion_matrix(y_test_rf, y_pred_rf)
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X, y, test_size=0.3, random_state=42)
X_test_xgb, X_validation_xgb, y_test_xgb, y_validation_xgb = train_test_split(X_test_xgb, y_test_xgb, test_size=0.5, random_state=42)
print("Number of samples in train set: %d" % y_train_xgb.shape)
print("Number of positive samples in train set: %d" % (y_train_xgb == 1).sum(axis=0))
print("Number of negative samples in train set: %d" % (y_train_xgb == 0).sum(axis=0))
print()
print("Number of samples in validation set: %d" % y_validation_xgb.shape)
print("Number of positive samples in validation set: %d" % (y_validation_xgb == 1).sum(axis=0))
print("Number of negative samples in validation set: %d" % (y_validation_xgb == 0).sum(axis=0))
print()
print("Number of samples in test set: %d" % y_test_rf.shape)
print("Number of positive samples in test set: %d" % (y_test_xgb == 1).sum(axis=0))
print("Number of negative samples in test set: %d" % (y_test_xgb == 0).sum(axis=0))
model_xgb = XGBClassifier()

# Define parameters and grid search
n_estimators = [100, 300, 500, 700]
subsample = [0.5, 0.7, 1.0]
max_depth = [6, 7, 9]
grid = dict(n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
grid_search = GridSearchCV(estimator=model_xgb, param_grid=grid, n_jobs=-1, cv=10, scoring='roc_auc', error_score=0)
grid_result = grid_search.fit(X_train_xgb, y_train_xgb)
print("Best AUC: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
model_xgb.n_estimators = grid_result.best_params_['n_estimators']
model_xgb.subsample = grid_result.best_params_['subsample']
model_xgb.max_depth = grid_result.best_params_['max_depth']
model_xgb.fit(X_train_xgb, y_train_xgb, eval_metric='auc', eval_set=[(X_train_xgb, y_train_xgb), (X_validation_xgb, y_validation_xgb)], verbose=False)
val_predictions_xgb = model_xgb.predict(X_validation_xgb)
print_scores(y_validation_xgb, val_predictions_xgb)
predictions_xgb = model_xgb.predict(X_test_xgb)
print_scores(y_test_xgb, predictions_xgb)
plot_confusion_matrix(y_test_xgb, predictions_xgb)
feature_importances = model_xgb.get_booster().get_fscore()
feature_importances_df = pd.DataFrame({'Feature Score': list(feature_importances.values()), 'Features': list(feature_importances.keys())})
feature_importances_df.sort_values(by='Feature Score', ascending=False, inplace=True)
feature_importances_df = feature_importances_df.head(15)
f, ax = plt.subplots(figsize=(7, 7))
plt.title('Top 15 Feature Importances', fontsize=14)
sns.barplot(x=feature_importances_df['Feature Score'], y=feature_importances_df['Features'])
f, ax = plt.subplots(figsize=(8, 8))
plt.plot([0, 1], [0, 1], '--', color='silver')
plt.title('ROC Curve', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
fpr, tpr, thresholds = roc_curve(y_test_xgb, model_xgb.predict_proba(X_test_xgb)[:,1]) 
sns.lineplot(x=fpr, y=tpr, color=sns.color_palette("husl", 8)[-2], linewidth=2, label="AUC = 95.41%")