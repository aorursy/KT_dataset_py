import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/amazon-employee-access-challenge/train.csv')

print(data.shape)

data.head()
data_explore = data.copy()
data_explore.info()
data_explore.nunique()
sns.countplot(x='ACTION', data=data_explore)
data_explore_resources = data_explore[['RESOURCE', "ACTION"]].groupby(by='RESOURCE').count()

data_explore_resources.sort_values('ACTION', ascending=False).head(n=15).transpose()
data_explore_role_dept = data_explore[['ROLE_DEPTNAME', "ACTION"]].groupby(by='ROLE_DEPTNAME').count()

data_explore_role_dept.sort_values('ACTION', ascending=False).head(n=15).transpose()
data_explore_role_codes = data_explore[['ROLE_CODE', "ACTION"]].groupby(by='ROLE_CODE').count()

data_explore_role_codes.sort_values('ACTION', ascending=False).head(n=15).transpose()
data_explore_role_family = data_explore[['ROLE_FAMILY', "ACTION"]].groupby(by='ROLE_FAMILY').count()

data_explore_role_family.sort_values('ACTION', ascending=False).head(n=15).transpose()
plt.figure(figsize=(12, 7))

corr_matrix = data_explore.corr()

sns.heatmap(corr_matrix, mask=np.zeros_like(corr_matrix, dtype=np.bool), square=True, annot=True, cbar=False)

plt.tight_layout()
corr_matrix['ACTION'].sort_values(ascending=False)
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
X = data.drop(columns=['ACTION'], axis=1).copy()

y = data['ACTION'].copy()

X.shape, y.shape
cat_attrs = list(X.columns)

cat_attrs
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(X, y):

    strat_train_set = data.iloc[train_index]

    strat_test_set = data.iloc[test_index]



X_train = strat_train_set.drop('ACTION', axis=1)

y_train = strat_train_set['ACTION'].copy()

X_test = strat_test_set.drop('ACTION', axis=1)

y_test = strat_test_set['ACTION'].copy()

X_train.shape, X_test.shape
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),

                         ('cat_enc', OneHotEncoder(handle_unknown='ignore'))])



pre_process = ColumnTransformer([('cat_process', cat_pipeline, cat_attrs)], remainder='passthrough')



X_train_transformed = pre_process.fit_transform(X_train)

X_test_transformed = pre_process.transform(X_test)

X_train_transformed.shape, X_test_transformed.shape
cat_boost_pre_process = ColumnTransformer([('imputer', SimpleImputer(strategy='most_frequent'), cat_attrs)], remainder='passthrough')



X_cb_train_transformed = cat_boost_pre_process.fit_transform(X_train)

X_cb_test_transformed = cat_boost_pre_process.transform(X_test)

X_cb_train_transformed.shape, X_cb_test_transformed.shape
feature_columns = list(pre_process.transformers_[0][1]['cat_enc'].get_feature_names(cat_attrs))

len(feature_columns)
from sklearn.model_selection import KFold, cross_val_score



kf = KFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score, roc_curve

Matthew = make_scorer(matthews_corrcoef)



results = []



def plot_custom_roc_curve(clf_name, y_true, y_scores):

    auc_score = np.round(roc_auc_score(y_true, y_scores), 3)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    plt.plot(fpr, tpr, linewidth=2, label=clf_name+" (AUC Score: {})".format(str(auc_score)))

    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal

    plt.axis([0, 1, 0, 1])

    plt.xlabel("FPR", fontsize=16)

    plt.ylabel("TPR", fontsize=16)

    plt.legend()

    

    

def performance_measures(model, X_tr=X_train_transformed, y_tr=y_train, X_ts=X_test_transformed, y_ts=y_test,

                         store_results=True):

    train_mcc = cross_val_score(model, X_tr, y_tr, scoring=Matthew, cv=kf, n_jobs=-1)

    test_mcc = cross_val_score(model, X_ts, y_ts, scoring=Matthew, cv=kf, n_jobs=-1)

    print("Mean Train MCC: {}\nMean Test MCC: {}".format(train_mcc.mean(), test_mcc.mean()))



    

    train_roc_auc = cross_val_score(model, X_tr, y_tr, scoring='roc_auc', cv=kf, n_jobs=-1)

    test_roc_auc = cross_val_score(model, X_ts, y_ts, scoring='roc_auc', cv=kf, n_jobs=-1)

    print("Mean Train ROC AUC Score: {}\nMean Test ROC AUC Score: {}".format(train_roc_auc.mean(), test_roc_auc.mean()))

    

    if store_results:

        results.append([model.__class__.__name__, np.round(np.mean(train_roc_auc), 3), np.round(np.mean(test_roc_auc), 3), np.round(np.mean(train_mcc), 3), np.round(np.mean(test_mcc), 3)])
def plot_feature_importance(feature_columns, importance_values, top_n_features=10):

    feature_imp = [ col for col in zip(feature_columns, importance_values)]

    feature_imp.sort(key=lambda x:x[1], reverse=True)

    

    if top_n_features:

        imp = pd.DataFrame(feature_imp[0:top_n_features], columns=['feature', 'importance'])

    else:

        imp = pd.DataFrame(feature_imp, columns=['feature', 'importance'])

    plt.figure(figsize=(10, 8))

    sns.barplot(y='feature', x='importance', data=imp, orient='h')

    plt.title('Most Important Features', fontsize=16)

    plt.ylabel("Feature", fontsize=16)

    plt.xlabel("")

    plt.show()
from sklearn.linear_model import LogisticRegression



logistic_reg = LogisticRegression(solver='liblinear', C=1, penalty='l2', max_iter=1000, random_state=42, n_jobs=-1)

logistic_reg.fit(X_train_transformed, y_train)
plot_feature_importance(feature_columns, logistic_reg.coef_[0], top_n_features=15)
performance_measures(logistic_reg)
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=42,n_jobs=-1)

forest_clf.fit(X_train_transformed, y_train)
plot_feature_importance(feature_columns, forest_clf.feature_importances_, top_n_features=15)
performance_measures(forest_clf)
from xgboost import XGBClassifier



xgb_clf = XGBClassifier(n_estimators=300, max_depth=16, learning_rate=0.1, random_state=42, n_jobs=-1)

xgb_clf.fit(X_train_transformed, y_train)
plot_feature_importance(feature_columns, xgb_clf.feature_importances_, top_n_features=15)
performance_measures(xgb_clf)
from catboost import CatBoostClassifier



catboost_clf = CatBoostClassifier(loss_function='Logloss', iterations=500, depth=6, l2_leaf_reg=1, 

                                  cat_features=list(range(X_cb_train_transformed.shape[1])), 

                                  eval_metric='AUC', random_state=42, verbose=0)

catboost_clf.fit(X_cb_train_transformed, y_train)
performance_measures(catboost_clf, X_tr=X_cb_train_transformed, X_ts=X_cb_test_transformed)
plot_feature_importance(feature_columns, catboost_clf.feature_importances_, top_n_features=15)
logistic_reg_pipeline = Pipeline([('pre_process', pre_process), ('logistic_reg', logistic_reg)])

forest_clf_pipeline = Pipeline([('pre_process', pre_process), ('forest_clf', forest_clf)])

xgb_clf_pipeline = Pipeline([('pre_process', pre_process), ('xgb_clf', xgb_clf)])

catboost_clf_pipeline = Pipeline([('pre_process', cat_boost_pre_process), ('catboost_clf', catboost_clf)])



named_estimators = [('logistic_reg', logistic_reg_pipeline), ('forest_clf', forest_clf_pipeline), 

                    ('xgb_clf', xgb_clf_pipeline), ('catboost_clf', catboost_clf_pipeline)]
from sklearn.ensemble import VotingClassifier



voting_reg = VotingClassifier(estimators=named_estimators, voting='soft', n_jobs=-1)

voting_reg.fit(X_train, y_train)
performance_measures(voting_reg, X_tr=X_train, X_ts=X_test)
result_df = pd.DataFrame(results, columns=['Model', 'CV Train AUC Score', 'CV Test AUC Score', 'CV Train MCC', 'CV Test MCC'])

result_df
plt.figure(figsize=(8, 5))

plot_custom_roc_curve('Logistic Regression', y_test, logistic_reg.decision_function(X_test_transformed))

plot_custom_roc_curve('Random Forest', y_test, forest_clf.predict_proba(X_test_transformed)[:,1])

plot_custom_roc_curve('XGBoost', y_test, xgb_clf.predict_proba(X_test_transformed)[:,1])

plot_custom_roc_curve('CatBoost', y_test, catboost_clf.predict_proba(X_cb_test_transformed)[:,1])

plot_custom_roc_curve('Soft Voting', y_test, voting_reg.predict_proba(X_test)[:,1])

plt.show()
final_model = Pipeline([('pre_process', cat_boost_pre_process),

                        ('catboost', catboost_clf)])

final_model.fit(X_train, y_train)
test_data = pd.read_csv('../input/amazon-employee-access-challenge/test.csv')

test_data.head()
output = pd.DataFrame(test_data['id'])

test_data = test_data.drop('id', axis=1)
test_data.info()
predictions = final_model.predict(test_data)
output['ACTION'] = predictions.copy()
output.head()
output.to_csv("./submission.csv", index=False)