import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from skopt import gp_minimize
from skopt.plots import plot_convergence

from imblearn.under_sampling import NearMiss

from collections import Counter
import joblib
df = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
print(df.shape)
df.head()
print(df["target"].value_counts()/df.shape[0])
sns.countplot("target", data = df)
y = df["target"]
x = df.drop(columns=["target", "ID_code"]).values
nm = NearMiss(version = 3, n_neighbors_ver3 = 3)
nm_x, nm_y = nm.fit_resample(x, y)
print(Counter(nm_y))
sns.countplot(nm_y)
x_train, x_test_val, y_train, y_test_val = train_test_split(nm_x, nm_y, test_size = .3, 
                                                            stratify = nm_y, random_state = 42)
x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size = .3, stratify = y_test_val,
                                               random_state = 42)


print("Train size: %i"%(len(y_train)))
print("Test size: %i"%(len(y_test)))
print("Validation size: %i"%(len(y_val)))
baseline_sgd_pipeline = Pipeline([("scaler", MaxAbsScaler()),
                                  ("sgd", SGDClassifier(loss = "log", random_state = 42))
                                 ])
baseline_sgd_pipeline.fit(x_train, y_train)

y_pred = baseline_sgd_pipeline.predict(x_test)
print(classification_report(y_test, y_pred))

y_pred_prob = baseline_sgd_pipeline.predict_proba(x_test)[:,1]
print("ROC AUC: %f"%(roc_auc_score(y_test, y_pred_prob)))
print("AUPRC: %f"%(average_precision_score(y_test, y_pred_prob)))
def create_sgd(params):
    penalty = params[0]
    max_iter = params[1]
    alpha = params[2]
    
    sgd_pipeline =  Pipeline([("scaler", MaxAbsScaler()),
                     ("sgd", SGDClassifier(loss = "log", penalty = penalty, 
                                           max_iter= max_iter, alpha = alpha, 
                                           random_state = 42))
                    ])
    
    return sgd_pipeline

def train_sgd(params):
    print("\n", params)
    
    sgd_pipeline = create_sgd(params)
    
    sgd_pipeline.fit(x_train, y_train)
    
    y_percs = sgd_pipeline.predict_proba(x_test)[:,1]
    
    return -roc_auc_score(y_test, y_percs)
    
params = [(["l1", "l2"]), #penalty
          (500, 10000), #max_iter
          (1e-5, 1e-1, "log-uniform") #alpha
         ]

gp_sgd = gp_minimize(train_sgd, params, verbose = 1, n_calls = 50, n_random_starts = 10, random_state = 42)
print(gp_sgd.x)
plot_convergence(gp_sgd)
k_scores = []
for k in range(5, 201, 5):
    selector = SelectKBest(f_classif, k = k)
    
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)
    
    sgd_pipeline_selected = create_sgd(gp_sgd.x)
    sgd_pipeline_selected.fit(x_train_selected, y_train)
    
    y_percs_selected = sgd_pipeline_selected.predict_proba(x_test_selected)[:,1]
    
    roc_score = roc_auc_score(y_test, y_percs_selected)
    k_scores.append(roc_score)
    
    print("K = %i -> ROC AUC %f"%(k, roc_score))

ind = k_scores.index(max(k_scores))
opt_k = 5 + 5 * ind
print("Max score = %f; K = %i"%(k_scores[ind], opt_k))

sns.lineplot(x = range(2, 201, 5), y = k_scores)
selector_sgd = SelectKBest(f_classif, k = opt_k)
x_train_sgd = selector_sgd.fit_transform(x_train, y_train)

df_sgd = df.drop(columns=["target", "ID_code"]).iloc[:, selector_sgd.get_support()]

print("Removed cols ->", set(df.columns) - set(df_sgd.columns))
df_sgd.head()
x_test_sgd = selector_sgd.transform(x_test)
x_val_sgd = selector_sgd.transform(x_val)
    
opt_sgd_pipeline = create_sgd(gp_sgd.x)
opt_sgd_pipeline.fit(x_train_sgd, y_train)
y_pred = opt_sgd_pipeline.predict(x_test_sgd)
print(classification_report(y_test, y_pred))

y_pred_prob = opt_sgd_pipeline.predict_proba(x_test_sgd)[:,1]
print("ROC AUC: %f"%(roc_auc_score(y_test, y_pred_prob)))
print("AUPRC: %f"%(average_precision_score(y_test, y_pred_prob)))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap = "BuGn")
y_pred = opt_sgd_pipeline.predict(x_val_sgd)
print(classification_report(y_val, y_pred))

y_pred_prob = opt_sgd_pipeline.predict_proba(x_val_sgd)[:,1]
print("ROC AUC: %f"%(roc_auc_score(y_val, y_pred_prob)))
print("AUPRC: %f"%(average_precision_score(y_val, y_pred_prob)))

sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt="d", cmap = "BuGn")
baseline_xgb_pipeline = Pipeline([("scaler", MaxAbsScaler()),
                                  ("gb", XGBClassifier(random_state = 42))
                                 ])
baseline_xgb_pipeline.fit(x_train, y_train)

y_pred = baseline_xgb_pipeline.predict(x_test)
print(classification_report(y_test, y_pred))

y_pred_prob = baseline_xgb_pipeline.predict_proba(x_test)[:,1]
print("ROC AUC: %f"%(roc_auc_score(y_test, y_pred_prob)))
print("AUPRC: %f"%(average_precision_score(y_test, y_pred_prob)))
def create_xgb(params):
    learning_rate = params[0]
    max_depth = params[1]
    subsample = params[2]
    colsample_bytree = params[3]
    max_leaves = params[4]
    min_child_weight = params[5]
    
    xgb_pipeline =  Pipeline([("scaler", MaxAbsScaler()),
                     ("gb", XGBClassifier(random_state = 42, 
                                          learning_rate = learning_rate, 
                                          max_depth = max_depth, 
                                          subsample = subsample,
                                          colsample_bytree = colsample_bytree,
                                          max_leaves = max_leaves, 
                                          min_child_weight = min_child_weight))
                    ])
    
    return xgb_pipeline

def train_xgb(params):
    print("\n", params)
    
    xgb_pipeline = create_xgb(params)
    
    xgb_pipeline.fit(x_train, y_train)
    
    y_percs = xgb_pipeline.predict_proba(x_test)[:,1]
    
    return -roc_auc_score(y_test, y_percs)
    
params = [(1e-3, 1e-1, "log-uniform"), #learning_rate
          (1, 50), #max_depth
          (0.01, .95), #subsample
          (0.01, .95), #colsample_bytree
          (2, 512), #max_leaves
          (1, 120) #min_child_weight
         ]

gp_xgb = gp_minimize(train_xgb, params, verbose = 1, n_calls = 65, n_random_starts = 15, random_state = 42)
print(gp_xgb.x)
plot_convergence(gp_xgb)
k_scores = []
for k in range(5, 201, 5):
    selector = SelectKBest(f_classif, k = k)
    
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)
    
    xgb_pipeline_selected = create_xgb(gp_xgb.x)
    xgb_pipeline_selected.fit(x_train_selected, y_train)
    
    y_percs_selected = xgb_pipeline_selected.predict_proba(x_test_selected)[:,1]
    
    roc_score = roc_auc_score(y_test, y_percs_selected)
    k_scores.append(roc_score)
    
    print("K = %i -> ROC AUC %f"%(k, roc_score))

ind = k_scores.index(max(k_scores))
opt_k = 5 + 5 * ind
print("Max score = %f; K = %i"%(k_scores[ind], opt_k))

sns.lineplot(x = range(2, 201, 5), y = k_scores)
selector_xgb = SelectKBest(f_classif, k = opt_k)
x_train_xgb = selector_xgb.fit_transform(x_train, y_train)

df_xgb = df.drop(columns=["target", "ID_code"]).iloc[:, selector_xgb.get_support()]

print("Removed cols ->", set(df.columns) - set(df_xgb.columns))
df_xgb.head()
x_test_xgb = selector_xgb.transform(x_test)
x_val_xgb = selector_xgb.transform(x_val)
    
opt_xgb_pipeline = create_xgb(gp_xgb.x)
opt_xgb_pipeline.fit(x_train_xgb, y_train)
y_pred = opt_xgb_pipeline.predict(x_test_xgb)
print(classification_report(y_test, y_pred))

y_pred_prob = opt_xgb_pipeline.predict_proba(x_test_xgb)[:,1]
print("ROC AUC: %f"%(roc_auc_score(y_test, y_pred_prob)))
print("AUPRC: %f"%(average_precision_score(y_test, y_pred_prob)))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap = "BuGn")
y_pred = opt_xgb_pipeline.predict(x_val_xgb)
print(classification_report(y_val, y_pred))

y_pred_prob = opt_xgb_pipeline.predict_proba(x_val_xgb)[:,1]
print("ROC AUC: %f"%(roc_auc_score(y_val, y_pred_prob)))
print("AUPRC: %f"%(average_precision_score(y_val, y_pred_prob)))

sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt="d", cmap = "BuGn")
baseline_lgbm_pipeline = Pipeline([("scaler", MaxAbsScaler()),
                                   ("lgbm", LGBMClassifier(random_state = 42))
                                  ])
baseline_lgbm_pipeline.fit(x_train, y_train)

y_pred = baseline_lgbm_pipeline.predict(x_test)
print(classification_report(y_test, y_pred))

y_pred_prob = baseline_lgbm_pipeline.predict_proba(x_test)[:,1]
print("ROC AUC: %f"%(roc_auc_score(y_test, y_pred_prob)))
print("AUPRC: %f"%(average_precision_score(y_test, y_pred_prob)))
def create_lgbm(params):
    learning_rate = params[0]
    n_estimators = params[1]
    num_leaves = params[2]
    min_child_samples = params[3]
    subsample = params[4]
    colsample_bytree = params[5]
    
    lgbm_pipeline =  Pipeline([("scaler", MaxAbsScaler()),
                     ("lgbm", LGBMClassifier(random_state = 42, 
                                                     learning_rate = learning_rate,
                                                     n_estimators = n_estimators, 
                                                     num_leaves = num_leaves, 
                                                     min_child_samples = min_child_samples, 
                                                     subsample = subsample, 
                                                     colsample_bytree = colsample_bytree))
                    ])
    
    return lgbm_pipeline

def train_lgbm(params):
    print("\n", params)
    
    lgbm_pipeline = create_lgbm(params)
    
    lgbm_pipeline.fit(x_train, y_train)
    
    y_percs = lgbm_pipeline.predict_proba(x_test)[:,1]
    
    return -roc_auc_score(y_test, y_percs)
    
params = [(1e-3, 1e-1, 'log-uniform'), #learning_rate
          (50, 2500), #n_estimators
          (2, 256), #num_leaves
          (2, 250), #min_child_samples
          (0.05, 1.0), #subsample
          (0.05, 1.0) #colsample_bytree
         ]

gp_lgbm = gp_minimize(train_lgbm, params, verbose = 1, n_calls = 40, n_random_starts = 10, random_state = 42)
print(gp_lgbm.x)
plot_convergence(gp_lgbm)
k_scores = []
for k in range(5, 201, 5):
    selector = SelectKBest(f_classif, k = k)
    
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)
    
    lgbm_pipeline_selected = create_lgbm(gp_lgbm.x)
    lgbm_pipeline_selected.fit(x_train_selected, y_train)
    
    y_percs_selected = lgbm_pipeline_selected.predict_proba(x_test_selected)[:,1]
    
    roc_score = roc_auc_score(y_test, y_percs_selected)
    k_scores.append(roc_score)
    
    print("K = %i -> ROC AUC %f"%(k, roc_score))

ind = k_scores.index(max(k_scores))
opt_k = 5 + 5 * ind
print("Max score = %f; K = %i"%(k_scores[ind], opt_k))

sns.lineplot(x = range(2, 201, 5), y = k_scores)
selector_lgbm = SelectKBest(f_classif, k = opt_k)
x_train_lgbm = selector_lgbm.fit_transform(x_train, y_train)

df_lgbm = df.drop(columns=["target", "ID_code"]).iloc[:, selector_lgbm.get_support()]

print("Removed cols ->", set(df.columns) - set(df_lgbm.columns))
df_lgbm.head()
x_test_lgbm = selector_lgbm.transform(x_test)
x_val_lgbm = selector_lgbm.transform(x_val)
    
opt_lgbm_pipeline = create_lgbm(gp_lgbm.x)
opt_lgbm_pipeline.fit(x_train_lgbm, y_train)
y_pred = opt_lgbm_pipeline.predict(x_test_lgbm)
print(classification_report(y_test, y_pred))

y_pred_prob = opt_lgbm_pipeline.predict_proba(x_test_lgbm)[:,1]
print("ROC AUC: %f"%(roc_auc_score(y_test, y_pred_prob)))
print("AUPRC: %f"%(average_precision_score(y_test, y_pred_prob)))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap = "BuGn")
y_pred = opt_lgbm_pipeline.predict(x_val_lgbm)
print(classification_report(y_val, y_pred))

y_pred_prob = opt_lgbm_pipeline.predict_proba(x_val_lgbm)[:,1]
print("ROC AUC: %f"%(roc_auc_score(y_val, y_pred_prob)))
print("AUPRC: %f"%(average_precision_score(y_val, y_pred_prob)))

sns.heatmap(confusion_matrix(y_val, y_pred), annot=True, fmt="d", cmap = "BuGn")

