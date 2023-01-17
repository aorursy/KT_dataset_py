import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, roc_auc_score
from scipy.stats import ks_2samp
from imblearn.over_sampling import SMOTE
import shap
df = pd.read_excel('../input/covid19/dataset.xlsx')

df.head()
# Transform to numeric
df['SARS-Cov-2 exam result'] = df['SARS-Cov-2 exam result'].map({'positive': 1, 'negative': 0})

# Map detected/not_detected and positive/negative to 1 and 0
df = df.replace({'positive': 1, 'negative': 0, 'detected': 1, 'not_detected': 0})

df['SARS-Cov-2 exam result'].value_counts(normalize=True)
df_null_pct = df.isna().mean().round(4) * 100

df_null_pct.sort_values(ascending=False)
df_null_pct.plot(kind='hist')
nulls = df_null_pct[df_null_pct > 90]

df = df[[col for col in df.columns if col not in nulls]]

df.head()
features = [col for col in df.columns if col not in ['Patient ID', 
                                                    'Patient addmited to regular ward (1=yes, 0=no)',
                                                    'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                                                    'Patient addmited to intensive care unit (1=yes, 0=no)',
                                                    'SARS-Cov-2 exam result']]

df[features].var()
df.drop('Parainfluenza 2', axis=1, inplace=True)
features.remove('Parainfluenza 2')
df['has_disease'] = df[df.columns[20:]].sum(axis=1)

df.loc[df['has_disease'] > 1, 'has_disease'] = 1

df['has_disease'].value_counts(normalize=True)
df[df['has_disease'] == 1]['SARS-Cov-2 exam result'].value_counts(normalize=True)
df_clean = df.copy()

df[df.columns[20:]] = df[df.columns[20:]].fillna(0)
for feature in df[df.columns[6:20]]:
    df_age_var = df.dropna(axis=0, subset=['Patient age quantile', feature]).loc[:, ['Patient age quantile',
                                                                                          feature]]
    missing = df[feature].isnull()
    age_missing = pd.DataFrame(df['Patient age quantile'][missing])

    X = df_age_var[['Patient age quantile']]
    y = df_age_var[feature]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lm = LinearRegression().fit(X_train, y_train)

    df.loc[df[feature].isna(), feature] = lm.predict(age_missing)
    
df.head()
df.isna().sum().sum()
df_age_var = df.dropna(axis=0, subset=['Patient age quantile', 'Hematocrit']).loc[:, ['Patient age quantile',
                                                                                      'Hematocrit']]

missing_hem = df['Hematocrit'].isnull()
age_missing_hem = pd.DataFrame(df['Patient age quantile'][missing_hem])

X = df_age_var[['Patient age quantile']]
y = df_age_var['Hematocrit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lm = LinearRegression().fit(X_train, y_train)
y_pred = lm.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red', lw=2)
plt.xlabel('Patient age quantile')
plt.ylabel('Hematocrit')
plt.show()
df_age_var = df.dropna(axis=0, subset=['Patient age quantile', 'Leukocytes']).loc[:, ['Patient age quantile',
                                                                                      'Leukocytes']]

missing_leu = df['Leukocytes'].isnull()
age_missing_leu = pd.DataFrame(df['Patient age quantile'][missing_leu])

X = df_age_var[['Patient age quantile']]
y = df_age_var['Leukocytes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lm = LinearRegression().fit(X_train, y_train)
y_pred = lm.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red', lw=2)
plt.xlabel('Patient age quantile')
plt.ylabel('Leukocytes')
plt.show()
def fit_and_print(model):
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test)
    #CM and Acc
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))  
    print("Classification Report: \n", classification_report(y_test, y_pred))  
    print("Accuracy Score: ", accuracy_score(y_test, y_pred))   
    print("Recall Score:", recall_score(y_test, y_pred))
    #AUC and KS
    print("AUC: ", roc_auc_score(y_test, y_pred))
    print("KS: ", ks_2samp(y_pred[y_test == 0], y_pred[y_test == 1]).statistic)
    
X = df[features].values 
y = df['SARS-Cov-2 exam result'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print("Split between test and train!")

#Apply RF
rf = RandomForestClassifier(n_estimators=100, max_features='auto', random_state=42)  

fit_and_print(rf)
print('Total Columns: ', df_clean.shape[1])
df_clean.isna().sum(axis=1).value_counts()
df_red = df_clean[df_clean.isna().sum(axis=1) < 26]

df_red.head()
df_null_pct = df_red.isna().mean().round(4) * 100

df_null_pct.sort_values(ascending=False)
df_red = df_red[df_red['Leukocytes'].notna()]

df_null_pct = df_red.isna().mean().round(4) * 100

df_null_pct.sort_values(ascending=False)
df_red.loc[df_red['Mean platelet volume '].isna(), 'Mean platelet volume '] = df_red['Mean platelet volume '].mean()

df_red.loc[df_red['Monocytes'].isna(), 'Monocytes'] = df_red['Monocytes'].mean()
cols_to_remove = [c for c in df_red.columns[20:-1]]
df_feat = df_red.drop(cols_to_remove, axis=1)

#update features
features = [c for c in df_feat.columns if c not in ['Patient ID', 'SARS-Cov-2 exam result',
                                                   'Patient addmited to regular ward (1=yes, 0=no)',
                                                   'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                                                   'Patient addmited to intensive care unit (1=yes, 0=no)']]

df_feat.head()
df_feat.isna().sum().sum()
corr = df_feat.drop(['Patient ID', 'Patient addmited to regular ward (1=yes, 0=no)', 
             'Patient addmited to semi-intensive unit (1=yes, 0=no)',
             'Patient addmited to intensive care unit (1=yes, 0=no)'], axis=1).corr()

plt.figure(figsize=(20,8))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=sns.color_palette('RdBu_r'), 
            annot=True)

plt.show()
df_feat = df_feat.drop(['Mean corpuscular hemoglobin (MCH)', 'Hematocrit', 'Hemoglobin'], axis=1)
#update features
features = [f for f in features if f not in ['Mean corpuscular hemoglobin (MCH)', 'Hematocrit', 'Hemoglobin']]
X = df_feat[features]
y = df_feat['SARS-Cov-2 exam result'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
print("Split between test and train!")

#Apply RF
rf = RandomForestClassifier(n_estimators=100, max_features='auto', random_state=42)  
fit_and_print(rf)
random_grid = {'n_estimators': [10, 50, 100, 200, 500],
            'max_features': ['auto', 'sqrt', 'log2', 5, 10, 30],
            'max_depth': [2, 8, 16, 32, 64, 128],
            'min_samples_split': [1,2,4,8,16,24],
            'min_samples_leaf': [1,2,5,10,15,30]}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                            n_iter = 200, cv = 5, scoring = 'recall', 
                            verbose=0, random_state=42, n_jobs = -1)

# Fit the random search model
fit_and_print(rf_random)
print(rf_random.best_params_)
smt = SMOTE(k_neighbors=5, random_state=42)
X_train, y_train = smt.fit_sample(X_train, y_train)

np.bincount(y_train)
rf = RandomForestClassifier(n_estimators=100, max_features='auto', random_state=42)  
fit_and_print(rf)
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                            n_iter = 300, cv = 5, scoring = 'recall', 
                            verbose=0, random_state=42, n_jobs = -1)
# Fit the random search model
fit_and_print(rf_random)
print(rf_random.best_params_)
random_grid = {
    'penalty': ['l1', 'l2'],
    'C': [100, 10, 1, 0.1, 0.01, 0.001]
    }
lr = LogisticRegression()

lr_random = RandomizedSearchCV(estimator = lr, param_distributions = random_grid, 
                            n_iter = 100, cv = 5, scoring = 'recall', 
                            verbose=0, random_state=42, n_jobs = -1)

# Fit the random search model
fit_and_print(lr_random)
print(lr_random.best_params_)
random_grid = {
    'n_neighbors': [2, 3, 5, 8, 10, 12, 15, 20, 30],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 3]
    }
knn = KNeighborsClassifier()

knn_random = RandomizedSearchCV(estimator = knn, param_distributions = random_grid, 
                            n_iter = 100, cv = 5, scoring = 'recall', 
                            verbose=0, random_state=42, n_jobs = -1)

# Fit the random search model
fit_and_print(knn_random)
print(knn_random.best_params_)
random_grid = {'C': [0.1, 1, 10, 100, 1000], 
               'gamma': ['auto', 1, 0.1, 0.01, 0.001, 0.0001], 
               'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}  

svc = SVC()

svc_random = RandomizedSearchCV(estimator = svc, param_distributions = random_grid, 
                            n_iter = 200, cv = 5, scoring = 'recall', 
                            verbose=0, random_state=42, n_jobs = -1)

# Fit the random search model
fit_and_print(svc_random)
print(svc_random.best_params_)
import xgboost as xgb

random_grid = {"n_estimators": [100, 500, 1000],
              "learning rate": [0.1, 0.05, 0.01],
              "max_depth": [2, 8, 16, 64, 128], 
              "colsample_bytree": [0.3, 0.8, 1],
              "gamma": [0,1,5]}  

xgb_clf = xgb.XGBClassifier(objective='binary:logistic')

xgb_random = RandomizedSearchCV(estimator = xgb_clf, param_distributions = random_grid, 
                            n_iter = 200, cv = 5, scoring = 'recall', 
                            verbose=0, random_state=42, n_jobs = -1)

# Fit the random search model
fit_and_print(xgb_random)
print(xgb_random.best_params_)
feature_importances = pd.DataFrame(rf_random.best_estimator_.feature_importances_,
                                index = features,
                                    columns=['importance']).sort_values('importance', ascending=False)

plt.figure(figsize=(12,6))
feature_importances.importance.plot(kind='barh', color='green')
plt.title('Feature Importance')
plt.show()
feature_importances.style.format({'importance': '{:.1%}'.format})
print(feature_importances.importance*100)
X = df_feat[features] 
y = df_feat['SARS-Cov-2 exam result'].values
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

y_pred = rf_random.best_estimator_.predict(X_test)

explainer = shap.TreeExplainer(rf_random.best_estimator_)
expected_value = explainer.expected_value[1]

shap_values = explainer.shap_values(X_test)[1]
shap_interaction_values = explainer.shap_interaction_values(X_test)[1]
np.save('shap_values.npy', shap_values)

print(expected_value)
shap.decision_plot(expected_value, shap_values[34], X_test.iloc[34])
shap.decision_plot(expected_value, shap_values[87], X_test.iloc[87])
shap.decision_plot(expected_value, shap_values[33], X_test.iloc[33])
shap.decision_plot(expected_value, shap_values[57], X_test.iloc[57])
shap.decision_plot(expected_value, shap_values[40], X_test.iloc[40])
shap.dependence_plot("Platelets", shap_values, X_test, alpha=0.8, show=False)
plt.title("Platelets dependence plot")
plt.axhline(0, lw=3)
plt.show()
shap.dependence_plot("has_disease", shap_values, X_test, alpha=0.8, show=False)
plt.title("Has_disease dependence plot")
plt.axhline(0, lw=3)
plt.show()
shap.dependence_plot("Leukocytes", shap_values, X_test, alpha=0.8, show=False)
plt.title("Leukocytes dependence plot")
plt.axhline(0, lw=3)
plt.show()
shap.dependence_plot("Monocytes", shap_values, X_test, alpha=0.8, show=False)
plt.title("Monocytes")
plt.axhline(0, lw=3)
plt.show()
shap.dependence_plot("Patient age quantile", shap_values, X_test, alpha=0.8, show=False)
plt.title("Patient age quantile dependence plot")
plt.axhline(0, lw=3)
plt.show()
shap.summary_plot(shap_values, X_test)
X_fn = X_test[(y_pred==0) & (y_test == 1)]
shap_fn = shap_values[(y_pred==0) & (y_test == 1)]

shap.decision_plot(expected_value, shap_fn, X_fn, feature_order='hclust')