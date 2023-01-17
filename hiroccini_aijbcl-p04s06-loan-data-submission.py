import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()

df = pd.read_csv('../input/av_loan_u6lujuX_CVtuZ9i.csv', header=0) ## read_csv for Check
df_s = pd.read_csv('../input/av_loan_test_Y3wMUE5_7gLdaTN.csv', header=0)  ## test data for score

print(f"df.shape: {df.shape}, df_s.shape: {df_s.shape}")
print(f"{[col for col in df.columns]}\n{[col for col in df_s.columns]}")
for i, elm in enumerate(zip(df.columns, df.dtypes, df_s.dtypes)):
    if elm[1] != elm[2]:
        print(f"{i}: {elm[0]}, {elm[1]}, {elm[2]}, {elm[1] == elm[2]}")
object_col = [col for col in df.columns if df[col].dtypes == object] # 'Dependents': object
num_col = [col for col in df.columns if df[col].dtypes != object]
print(f"{object_col}\n{num_col}")
[col for col in object_col + num_col if col not in df.columns] # df.columns との一致を確認
# set(object_col + num_col) == set(df.columns)
df = pd.read_csv('../input/av_loan_u6lujuX_CVtuZ9i.csv', header=0
                , dtype={
                    'Gender': object, 
                    'Married': object, 
                    'Dependents': object,     # 'Dependents': object
                    'Education': object, 
                    'Self_Employed': object, 
                    'Property_Area': object
                    }
                )
df_s = pd.read_csv('../input//av_loan_test_Y3wMUE5_7gLdaTN.csv', header=0
                  , dtype={
                    'Gender': object, 
                    'Married': object, 
                    'Dependents': object,     # 'Dependents': object
                    'Education': object, 
                    'Self_Employed': object, 
                    'Property_Area': object
                    }
                )
print(f"{[col for col in df.columns if df[col].dtypes == object]}\n{[col for col in df_s.columns if df_s[col].dtypes == object]}")
[col for col in df.columns if col not in [c for c in df_s.columns]]
df[object_col].head()
object_col_plot = [col for col in object_col if df[col].nunique() < 10]
print(f"{len(object_col_plot)}")
fig, axes = plt.subplots(2, 4, figsize=(12,3))
for i, col in enumerate(object_col_plot):
    ax = axes.ravel()[i]
    df_plot = df[col].value_counts(dropna=False, sort=False)
    df_plot.plot.bar(ax=ax, color='b', alpha=.5)
    ax.set_title(col); ax.set_xticklabels(s[:3] for s in df_plot.index.astype(str))
#     print(f"{col}: {df[col].unique()}")
fig.tight_layout()
df[num_col].head(3)
# print(f"{len(num_col)}")
fig, axes = plt.subplots(1, 5, figsize=(18,2.5))
bins = 20         # df.hist() と np.histogram() の bins 一致のため
for i, col in enumerate(num_col):
    ax = axes.ravel()[i]
    df[col].hist(ax=ax, bins=bins, alpha=.5, label='hist')
    hist, _ = np.histogram(df[col].dropna(), bins=bins)  # ax.vlines() ymax 取得のため
    ax.set_title(col, fontsize=16)
    ax.vlines(x = df[col].mean(), ymax=hist.max(), ymin=0, color = 'r', alpha=.7, label='mean')
    ax.set_xlabel("mean: {:,.2f}".format(df[col].mean()), fontsize=14)
    ax.legend()
#     print(f"{df[col].nunique()}")
fig.tight_layout()
if set(df['Loan_Status']) == {'N', 'Y'}:
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 0, 'N': 1})
df.corr()
sns.heatmap(df.corr(), cmap='Blues')
df.describe()
pivot_col = [col for col in df.columns if df[col].nunique() < 10 and col not in 'Loan_Status']
print(f"{pivot_col}")
# for col in pivot_col:
#     print(f"{col}: {df[col].unique()}")
print(f"{len(pivot_col)}")
fig, axis = plt.subplots(1, len(pivot_col), figsize=(20,2.5))
for i, col in enumerate(pivot_col):
    ax = axis.ravel()[i]
    df_v = df[col].value_counts(dropna=False)
    df_v.plot.pie(ax=ax, autopct='%.2f%%', colors = ['pink', 'grey', 'silver'])
    ax.set_title('')
fig, axes = plt.subplots(1, len(pivot_col), figsize=(18,1.5))
for i, col in enumerate(pivot_col):
    ax = axes.ravel()[i]
    df_pvt = pd.pivot_table(df.fillna('nan'), index=col, columns='Loan_Status', aggfunc={'Loan_ID': [len]})
    df_fin = df_pvt.apply(lambda x: x/sum(x), axis=1).iloc[:, 1]
    df_fin.plot.bar(ax=ax, color='b', alpha=.5); ax.set_title(col); ax.set_xlabel('')
    ax.set_xticklabels([i.upper()[:5] for i in df_fin.index.astype(str)])
X = df.iloc[:, 1:-1]
ID = df.iloc[:, [0]]
y = df.iloc[:, [-1]]

print(f"all_shapes: {df.shape}, {X.shape}, {ID.shape}, {y.shape}")
print(f"correspondence: {(ID.join(X).join(y).columns == df.columns).all()}\n")

print(f"{y['Loan_Status'].value_counts(dropna=False)}\n")
y = y.copy()

if [v for v in y['Loan_Status'].unique()] != [0, 1]:
    y.loc[:, 'Loan_Status'] = y.loc[:, 'Loan_Status'].map({'Y': 0, 'N': 1})
print(f"{y['Loan_Status'].value_counts()}\n")
print(f"y['Loan_Status'].isnull().any(): {y['Loan_Status'].isnull().any()}")

y.head()
ID_s = df_s.iloc[:, [0]]
X_s =df_s.iloc[:, 1:]
print(f"{df_s.shape}\n{ID_s.shape}\n{X_s.shape}")
ID_s.join(X_s).head()
X.isnull().any()
print(f"X.shape: {X.shape}")
ohe_cols = [col for col in X.columns if X[col].dtypes == object]
print(f"{ohe_cols}\n")
for col in ohe_cols:
    print(f"{col}: {X[col].unique()}")

X_ohe = pd.get_dummies(X, dummy_na=True, columns=ohe_cols)
print(f"X_ohe.shape: {X_ohe.shape}")
print(f"{[c for c in X_ohe.columns.values]}")
X_ohe.head()
combine, comb_name = [X, X_s], ['X', 'X_s']
for i, dataset in enumerate(combine):
    num_cols_with_missing_values = [col for col in dataset.columns if dataset[col].dtypes != object and dataset[col].isnull().any()]
    df_null = dataset.loc[:, num_cols_with_missing_values].isnull().apply(lambda x: x.value_counts())
    df_null.index = ['no_Null', 'Null']
    print(f"{df_null}\n")
X_ohe.isnull().any()
nullcol = [col for col in X_ohe.columns if X_ohe[col].isnull().any()]
nullrow = [i for i in X_ohe.index if X_ohe.iloc[i, :].isnull().any()]
X_ohe.loc[nullrow, nullcol].head()
from sklearn.preprocessing import Imputer
X_ohe_columns = X_ohe.columns
print(f"### Check mean() for each features before Imputing:\n{X_ohe[nullcol].mean()}")

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) # All Parameters: default
imp.fit(X_ohe)
X_ohe = pd.DataFrame(imp.transform(X_ohe), columns=X_ohe_columns)
X_ohe.loc[nullrow, nullcol].head()
X_ohe
y.values.ravel().shape
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
selector = RFE(GradientBoostingClassifier(random_state=1), n_features_to_select=10, step=.05)
selector.fit(X_ohe, y.values.ravel())
X_fin = pd.DataFrame(selector.transform(X_ohe), columns=X_ohe_columns[selector.support_])
print(f"selector.ranking_: {selector.ranking_}, {len(selector.ranking_)}")
X_fin.head()
ranks = {X_ohe_columns[i]: rank for i, rank in enumerate(selector.ranking_)}
print(f"sorted_ranks:\n{sorted(ranks.items(), key=lambda item: item[1])}")
print(f"selected_columns:\n{[k for k, v in ranks.items() if v == 1]}")
# print(f"{[c for c in X_ohe_columns[selector.support_]]}")
key_max = max(ranks, key=ranks.get)
print(f"key_max: {key_max}")
print(f"ranks[key_max]: {ranks[key_max]}")
# print(f"{[c for c in X_s.columns if X_s[c].dtypes == object]}")
# print(f"{ohe_cols}\n")

X_ohe_s = pd.get_dummies(X_s, dummy_na=True, columns=ohe_cols)
print(f"{X_ohe_s.shape}")

# Check the difference of columns
only_in_model_col = [col for col in X_ohe.columns if col not in X_ohe_s.columns]
only_in_score_col = [col for col in X_ohe_s.columns if col not in X_ohe.columns]
print(f"only_in_model_col: {only_in_model_col}")
print(f"only_in_score_col: {only_in_score_col}")
p, q, r, s = 3, 4, 2, 5
pq = np.arange(p*q).reshape(p,q)
rs = np.arange(p*q, p*q+r*s).reshape(r,s)
df_pq = pd.DataFrame(pq, columns=['a', 'b', 'c', 'd'])
df_rs = pd.DataFrame(rs, columns=['b', 'e', 'd', 'f', 'g'])
print(f"{df_pq} ---- カラム名 b, d が共通\n{df_rs}")
join_list = ['inner', 'left', 'right', 'outer']; axis_list = [0, 1, None]
for join in join_list:
    for axis in axis_list:
        train, test = df_pq.align(df_rs, join=join, axis=axis); print(f", join='{join}', axis={axis}):\n{train} ----\n{test}\n")
X_ohe, X_ohe_s2 = X_ohe.align(X_ohe_s, join='left', axis=1) ### axis=1: Very Important!!!
print(f"X_ohe.shape, X_ohe_s2.shape: {X_ohe.shape}, {X_ohe_s2.shape}")
print(f"new column in X_ohe_s2: {[col for col in X_ohe_s2.columns if col not in X_ohe_s.columns]}")
print(f"'{only_in_model_col[0]}' in X_ohe_s2.columns: {only_in_model_col[0] in X_ohe_s2.columns}")
print(f"'{only_in_score_col[0]}' in X_ohe_s2.columns: {only_in_score_col[0] in X_ohe_s2.columns}")
print(f"(X_ohe.columns == X_ohe_s2.columns).all(): {(X_ohe.columns == X_ohe_s2.columns).all()}")
print(f"{[c for c in X_ohe.columns if c not in X_ohe_s2.columns]}, {[c for c in X_ohe_s2.columns if c not in X_ohe.columns]}\n")

print(f"X_ohe_s2[only_in_model_col[0]].value_counts(dropna=False):\n{X_ohe_s2[only_in_model_col[0]].value_counts(dropna=False)}\n")
print(f"X_ohe_s2.shape: {X_ohe_s2.shape}")
X_ohe_s2[only_in_model_col[0]] = X_ohe_s2[only_in_model_col[0]].fillna(0)
X_ohe_s2[[only_in_model_col[0]]].head()
for elm in zip([c for c in X_ohe.columns], [c for c in X_ohe_s2.columns]):
    if elm[0] != elm[1]:
        print(f"{false}")
[elm for elm in zip([c for c in X_ohe.columns], [c for c in X_ohe_s2.columns]) if elm[0] != elm[1]]
# [elm for elm in zip([c for c in X_ohe.columns], [c for c in X_ohe_s2.columns])] # display All in zip 表示するとこうなる 
nullcol_s2 = [col for col in X_ohe_s2.columns if X_ohe_s2[col].isnull().any()]
nullrow_s2 = [indx for indx in X_ohe_s2.index if X_ohe_s2.loc[indx, :].isnull().any()]
print(f"{X_ohe_s2.loc[nullrow_s2, nullcol_s2].head(6)}")
print(f"means of modeling data: X_ohe:\n{X_ohe[nullcol].mean()}\n")
X_ohe_s3 = pd.DataFrame(imp.transform(X_ohe_s2), columns=X_ohe_columns)
print(f"{X_ohe_s3.loc[nullrow_s2, nullcol_s2].head(6)}")
X_ohe_s3.head()
print(f"{selector.support_}\n{[c for c in X_ohe_columns[selector.support_]]}")
X_fin_s = pd.DataFrame(selector.transform(X_ohe_s3), columns=X_ohe_columns[selector.support_])
print(f"{X_fin_s.shape}")
X_fin_s.head()
from sklearn.model_selection import train_test_split, GridSearchCV
print(f"X_fin.shape, y.shape: {X_fin.shape}, {y.shape}")
indices = np.arange(X_fin.shape[0]) # make inidces
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_fin, y, indices, random_state=64)
print(f"X_train.shape, X_test.shape: {X_train.shape}, {X_test.shape}\ny_train.shape, y_test.shape: {y_train.shape}, {y_test.shape}\n\
indices_train.shape, indices_test.shape: {indices_train.shape}, {indices_test.shape}")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_recall_curve, classification_report, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier

from imblearn.over_sampling import SMOTE
pipe_knn = Pipeline([('scl', StandardScaler()), ('est', KNeighborsClassifier())])
pipe_decision_tree = Pipeline([('scl', StandardScaler()), ('est', DecisionTreeClassifier(random_state=0))])
pipe_logreg = Pipeline([('scl', StandardScaler()), ('est', LogisticRegression(random_state=0))])
pipe_perceptron = Pipeline([('scl', StandardScaler()), ('est', Perceptron(max_iter=5, tol=None))])
pipe_sgd = Pipeline([('scl', StandardScaler()), ('est', SGDClassifier(random_state=0, max_iter=1000, tol=1e-3))])
pipe_rf = Pipeline([('scl', StandardScaler()), ('est', RandomForestClassifier(random_state=3))])
pipe_gb = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingClassifier(random_state=1))])
pipe_svc = Pipeline([('scl', StandardScaler()), ('est', SVC(random_state=0))])
pipe_linear_svc = Pipeline([('scl', StandardScaler()), ('est', LinearSVC())])
pipe_gaussian = Pipeline([('scl', StandardScaler()), ('est', GaussianNB())])
pipe_xgb = Pipeline([('scl', StandardScaler()), ('est', XGBClassifier())])
pipes = [pipe_knn, pipe_decision_tree, pipe_logreg, pipe_perceptron, pipe_sgd, pipe_rf, pipe_gb, pipe_gaussian, pipe_xgb]
param_grid_knn = {'est__n_neighbors': range(1,101),
                  'est__weights': ['uniform', 'distance']}

param_grid_decision_tree = {'est__criterion': ['gini', 'entropy'],
                    'est__max_depth': range(1, 11),
                    'est__min_samples_split': range(2, 21),
                    'est__min_samples_leaf': range(1, 21),}

param_grid_logreg = {
                    'est__penalty': ['l1', 'l2'],
                    'est__C': [1e-4, 1e-3, 1e-2, 0.05, 0.1, 1.0, 10.0, 100.0],
#                     'est__dual': [True, False]  ## エラー発生
                    }

param_grid_perceptron = {'est__penalty': [None, 'l2', 'l1', 'elasticnet'],
                'est__alpha': [0.00001, 0.0001, 0.001, 0.01],}

param_grid_sgd = {'est__loss': ['hinge', 'log', 'modified_huber'],
                'est__penalty': ['none', 'l2', 'l1', 'elasticnet'],
}

param_grid_rf = {
                'est__n_estimators': [10, 50, 100, 150, 200],
                'est__criterion': ['gini', 'entropy'],
                'est__max_features': np.arange(0.05, 1.01, 0.05),
#                 'est__min_samples_split': range(2, 21),
}

param_grid_gb = {
                'est__n_estimators': [25, 50, 100, 150],
                'est__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
                'est__max_depth': range(1, 11),
                'est__min_samples_split': range(2, 11),
#                 '': [],
}

param_grid_svc = {
#                 'est__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'est__C': [1, 10, 100, 1000],
                'est__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

param_grid_linear_svc = {
#                 'est__penalty': ['l1', 'l2'],
#                 'est__loss': ['hinge', 'squared_hinge'], 
                'est__dual': [True, False],
                'est__tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                'est__C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]}

param_grid_gaussian = {}

param_grid_xgb = {
#       'est__n_estimators': [100]
     'est__learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.] 
#      'est__learning_rate': [1e-1, 0.5, 1.] 
    , 'est__max_depth': range(1, 11) 
#     , 'est__max_depth': range(1, 11, 2) 
#     , 'est__min_child_weight': range(1, 21) 
    , 'est__min_child_weight': range(1, 11) 
    , 'est__subsample': np.arange(0.05, 1.01, 0.05) 
#     , 'est__nthread': [1] 
}
param_grids = [param_grid_knn, param_grid_decision_tree, param_grid_logreg, param_grid_perceptron, param_grid_sgd, 
               param_grid_rf, param_grid_gb, param_grid_gaussian, param_grid_xgb]
est_names = ['KNeighbors', 'DecisionTree', 'LogisticRegression', 'Perceptron', 'SGDClassifier', 
             'RandomForest', 'GradientBoosting', 'GaussianNB', 'XGBClassifier']
# check for correspondence
for i, pipe in enumerate(pipes):
    print(f"{i}: {est_names[i]}\n{pipe.named_steps['est']}\n{param_grids[i]}\n")
## for test
## リスト3つを再度記載して
pipes = [pipe_knn, pipe_decision_tree, pipe_logreg, pipe_perceptron, pipe_sgd, pipe_rf, pipe_gb, pipe_gaussian, pipe_xgb]

param_grids = [param_grid_knn, param_grid_decision_tree, param_grid_logreg, param_grid_perceptron, param_grid_sgd, 
               param_grid_rf, param_grid_gb, param_grid_gaussian, param_grid_xgb]

est_names = ['KNeighbors', 'DecisionTree', 'LogisticRegression', 'Perceptron', 'SGDClassifier', 
             'RandomForest', 'GradientBoosting', 'GaussianNB', 'XGBClassifier']

n_to_select = [0, 1, 2, 3, 4, 5, 6, 7, 8] ### ここでインデックスを指定するのがミソ
# n_to_select = [0, 1, 2, 3, 4, 5, 7] ### ここでインデックスを指定するのがミソ

pipes = [pipes[i] for i in n_to_select]
param_grids = [param_grids[i] for i in n_to_select]
est_names = [est_names[i] for i in n_to_select]

print({i: name for i, name in enumerate(est_names)})
X_train.shape, y_train.shape, X_test.shape, y_test.shape
import time
start_time = time.time()
import warnings
warnings.filterwarnings('ignore')

best_scores, acc_scores, f1_scores, best_paramses, best_estimators = [], [], [], [], []
for i, pipe in enumerate(pipes):
    iter_time = time.time()
    param_grid = param_grids[i]
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=3)
    gs.fit(X_train, y_train.values.ravel())
    best_score = gs.best_score_
    acc_score_train = accuracy_score(y_train.values.ravel(), gs.predict(X_train))
    acc_score_test = accuracy_score(y_test.values.ravel(), gs.predict(X_test))
    f1_score_train = f1_score(y_train.values.ravel(), gs.predict(X_train))
    f1_score_test = f1_score(y_test.values.ravel(), gs.predict(X_test))  ##
    best_scores.append(best_score)
    acc_scores.append(acc_score_test)
    f1_scores.append(f1_score_test)
    best_paramses.append(gs.best_params_)
    best_estimators.append(gs.best_estimator_)
    
    print(f"{i}: {est_names[i]} ------------\n{pipe.named_steps['est']}")
    print(f"{gs.best_params_}")
    print(f"{gs.best_estimator_.named_steps['est']}")
    print(f"gs.best_score_: {best_score:.4f}")
    print(f"acc_train_test: {acc_score_train:.4f}, {acc_score_test:.4f}")
    print(f"f1_train_test : {f1_score_train:.4f}, {f1_score_test:.4f}")
    
    print(f"{time.time() - iter_time:.2f} sec.")
    print(f"{time.time() - start_time:.2f} sec.")
    print("")
print(f"{best_scores}\n{acc_scores}\n{f1_scores}\n{est_names}")
df_scores = pd.DataFrame([best_scores, acc_scores, f1_scores]
            , index=['best_score', 'acc_score', 'f1_score']
            , columns=est_names).T
df_scores
df_scores.sort_values(by=['best_score'], ascending=False)
df_scores.sort_values(by=['acc_score'], ascending=False)
df_scores.sort_values(by=['f1_score'], ascending=False)
for i, score in enumerate(best_scores):
    print(f"{i}: {score}")
[i for i, score in enumerate(best_scores) if score == max(best_scores)][-1]

# [1, 0, 8, 9, 7, 2, 4][-1]
best_score_indx = [i for i, score in enumerate(best_scores) if score == max(best_scores)][-1] # 複数ある場合は最後を取得
final_best_name = est_names[best_score_indx]
print(f"{final_best_name}: {max(best_scores)}, ({best_score_indx})")

final_best_estimator = best_estimators[best_score_indx]
print(f"{final_best_estimator.named_steps['est']}")
best_acc_indx = [i for i, score in enumerate(acc_scores) if score == max(acc_scores)][-1]
final_acc_name = est_names[best_acc_indx]
final_acc_estimator = best_estimators[best_acc_indx]
print(f"{final_acc_name}: {max(acc_scores)}, ({best_acc_indx})")
print(f"{final_acc_estimator.named_steps['est']}")
best_f1_index = [i for i, score in enumerate(f1_scores) if score == max(f1_scores)][-1]
final_f1_name = est_names[best_f1_index]
final_f1_estimator = best_estimators[best_f1_index]
print(f"{final_f1_name}: {max(f1_scores)}, ({best_f1_index})")
print(f"{final_f1_estimator.named_steps['est']}")
print(f"final_best_name: {final_best_name}\nfinal_acc_name:  {final_acc_name}\nfinal_f1_name:   {final_f1_name}")
from sklearn.externals import joblib
for i, est in enumerate(best_estimators):
    joblib.dump(est, est_names[i] + '.pkl')
#     print(f"{i}: {est_names[i]}\n{est.named_steps['est']}\n")
print(f"{final_best_name}")
load_best = joblib.load(final_best_name + '.pkl')
load_best.named_steps['est']
# predict_proba()
final_best_estimator.predict_proba(X_fin_s)[:5, :]
# get the right column by slicing （スライスで右のカラムを取得）
best_proba_right = final_best_estimator.predict_proba(X_fin_s)[:, 1]  # not [:, [1]], but [:, 1] 
print(f"{best_proba_right.shape}, ndim: {best_proba_right.ndim}")
print(f"{best_proba_right[:5]}")
print(f"{type(ID_s.iloc[:, 0])}, ndim: {ID_s.iloc[:, 0].ndim}, shape: {ID_s.iloc[:, 0].shape}")
final_best_submission = pd.DataFrame({
    'Loan_ID': ID_s.iloc[:, 0],        # not [:, [0]], but [:, 0] then ndim: 1
    'proba_right': best_proba_right
})
final_best_submission.to_csv('final_best_submission.csv', index=False)
final_best_submission.head()
final_estimators = [final_best_estimator, final_acc_estimator, final_f1_estimator]
names_to_csv =['final_best_submission', 'final_acc_submission', 'final_f1_submission']
final_scoring_names = [final_best_name, final_acc_name, final_f1_name]

for i, estimator in enumerate(final_estimators):
    proba_right_iter = estimator.predict_proba(X_fin_s)[:, 1]
    df_submission = pd.DataFrame({
        'Loan_ID': ID_s.iloc[:, 0],
        'proba_right': proba_right_iter
        })
    df_submission.to_csv(names_to_csv[i] + '.csv', index=False)
    print(f"{i}: {names_to_csv[i]}: {final_scoring_names[i]}; \n{proba_right_iter[:5]}\n{df_submission.head()}\n")
for i, estimator in enumerate(final_estimators):
    print(f"{i}: {names_to_csv[i][6:]:<15}: {final_scoring_names[i]}\n{confusion_matrix(y_test.values.ravel(), estimator.predict(X_test))}\n{classification_report(y_test.values.ravel(), estimator.predict(X_test))}")
y_test.iloc[:, 0].value_counts()
pred_proba_right = final_f1_estimator.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), pred_proba_right)
thresholds = np.append(thresholds, 1)
print(f"precision.shape, recall.shape, thresholds.shape: {precision.shape, recall.shape, thresholds.shape}")
# print(f"precision:  {precision[:6]}\nrecall:     {recall[:6]}\nthresholds: {thresholds[:6]}, {thresholds.min()}, {thresholds.max()}")
fig, axes = plt.subplots(figsize=(6,2))
df_pre_rec = pd.DataFrame({'recall': recall}, index=precision)
# print(f"\n{df_pre_rec.head()}\n")
df_pre_rec.plot(ax=axes)
for i in range(11):
    close_point = np.argmin(abs(thresholds - (0.1 * i)))
#     print(f"{i}: {0.1 *i}, {close_point}")
    axes.plot(precision[close_point], recall[close_point], 'o')
pred_proba_right = final_acc_estimator.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), pred_proba_right)
thresholds = np.append(thresholds, 1)

queue_rate = []
for i, threshold in enumerate(thresholds):
#     print(f"{i:>2}: {threshold:<20}, {(pred_proba_right > threshold).mean()}") # 閾値よりも大きい pred_proba_right の平均
    queue_rate.append((pred_proba_right > threshold).mean())

print(f"precision.shape, recall.shape, thresholds.shape, len(queue_rate): \
{precision.shape, recall.shape, thresholds.shape, len(queue_rate)}")
# print(f"precision:  {precision[:6]}\nrecall:     {recall[:6]}\nthresholds: {thresholds[:6]}, {thresholds.min()}, \
# {thresholds.max()}\n")

fig, axes = plt.subplots(figsize=(6,2))
df_pre_rec = pd.DataFrame({'precision': precision, 'recall': recall, 'queue_rate': queue_rate, 'thresholds': thresholds}, 
                          index=thresholds)
# print(f"{df_pre_rec.head()}\n")
df_pre_rec.plot(ax=axes)

# 0から1まで0.1刻みで○をプロット
for i in range(11):
    close_point = np.argmin(abs(thresholds - (0.1 * i)))
#     print(f"{i}: {0.1 *i}, {close_point}")
    axes.plot(thresholds[close_point], queue_rate[close_point], 'o')
fig, axes = plt.subplots(figsize=(6, 4))
for i, estimator in enumerate(final_estimators):
    proba = estimator.predict_proba(X_test)
    proba_right = estimator.predict_proba(X_test)[:, 1]
#     print(f"{i}:\n{proba[:5]} \n{proba_right[:5]}")
    precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), proba_right)
    thresholds = np.append(thresholds, 1)
#     df_pre_recall = pd.DataFrame({'recall': recall}, index=precision)
#     df_pre_recall.plot(ax = axes, label=i)
    axes.plot(precision, recall, label=final_scoring_names[i])
    axes.set_xlabel('precision'); axes.set_ylabel('recall')
    axes.legend()
#     print(f"{i}: {final_scoring_names[i]}\n{precision.shape, recall.shape, thresholds.shape}\n{precision[:5]}\n{recall[:5]}")
    
    for i in range(11):
        close_point = np.argmin(abs(thresholds - (0.1 * i)))
#         print(f"{i}: {close_point}")
        axes.plot(precision[close_point], recall[close_point], 'o', color = 'r', alpha=.4)
#     print("")
fig, axes = plt.subplots(len(final_estimators), 1, figsize=(6, 10))
for i, estimator in enumerate(final_estimators):
    proba_right = estimator.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test.values.ravel(), proba_right)
    thresholds = np.append(thresholds, 1)

    queue_rate = []
    for j, threshold in enumerate(thresholds):
        queue_rate.append((proba_right > threshold).mean())
#         print(f"{j:>3}: {(proba_right > threshold).mean():<20}, {threshold}")
    
    df_queue = pd.DataFrame({
        'precision': precision, 'recall': recall, 'thresholds': thresholds, 'queue_rate': queue_rate
    }, index=thresholds)
    ax = axes.ravel()[i]
    df_queue.plot(ax = ax); ax.set_title(final_scoring_names[i])
    
    for k in range(11):
        close_point = np.argmin(abs(thresholds - 0.1 * k))
        ax.plot(thresholds[close_point], queue_rate[close_point], 'o', color='r', alpha=.3)
#         print(f"{k}: {0.1 * k}, {close_point}")
    
#     print(f"{i}: {final_scoring_names[i]}\n{precision.shape, recall.shape, thresholds.shape, len(queue_rate)}")
#     print(f"{df_queue.head()}")
    print("")

fig.tight_layout()
tree_list = [1, 5, 6, 8]
# tree_list = [1, 5] ###
fig, axes = plt.subplots(1, len(tree_list), figsize=(16, 3))
for i, j in enumerate(tree_list):
    df_imp = pd.DataFrame(
        best_estimators[j].named_steps['est'].feature_importances_,
        index=X_train.columns,
        columns=['f_importances_']
    )
    ax = axes.ravel()[i]
    df_imp.iloc[::-1, :].plot.barh(ax=ax, title=est_names[j])
    ax.set_yticklabels("")
axes.ravel()[0].set_yticklabels([c.upper()[:] for c in df_imp.index[::-1]])
    
    
#     print(f"{est_names[j]}: \n{df_imp}\n{df_imp.iloc[::-1, :]}\n")

fig.tight_layout()
print({i: name for i, name in enumerate(est_names)})
i_selected = 6
pipe_best = best_estimators[i_selected]
df_imp = pd.DataFrame(pipe_best.named_steps['est'].feature_importances_, index=X_test.columns, columns=['importance'])
fig, axes = plt.subplots(figsize=(8,3))
df_imp.plot.bar(ax=axes); axes.set_xticklabels([i.upper()[:9] for i in df_imp.index])
axes.set_title("feature_importances_of_{}".format(est_names[i_selected]))
fig.tight_layout()
from sklearn.ensemble.partial_dependence import plot_partial_dependence
n_to_select = 6 # 重要度上位の特徴量数の選択
df_sort = df_imp.reset_index().sort_values(by='importance', ascending=False).iloc[: n_to_select, :]
print(f"{df_sort}")
fig, axes = plt.subplots(figsize=(12,4))
plot_partial_dependence(pipe_best.named_steps['est'], pipe_best.named_steps['scl'].transform(X_train), features=df_sort.index, 
                        feature_names=df_sort['index'], ax=axes)
fig.tight_layout()
data = pd.read_csv('../input/melb_data.csv')
data.columns = [col.lower() for col in data.columns]
print(f"{data.shape}")
col_to_drop = [col for col in data.columns if data[col].dtypes == object and data[col].nunique() > 10] # 削除するカラム
# print(f"{col_to_drop}")
# for col in col_to_drop:
#     print(f"{col}:\n{data[col].unique()}")
data = data.drop(col_to_drop, axis=1)
print(f"{data.shape}\n{[col for col in data.columns]}\n")
y = data[['price']]
X = data.drop(['price'], axis=1)

ohe_columns = [col for col in X.columns if X[col].dtypes == object]
# print(f"{ohe_columns}")
# for col in ohe_columns:
#     print(f"{col}: \n{X[col].unique()}")
X_ohe = pd.get_dummies(X, columns=ohe_columns, dummy_na=True)    
# print(f"{X_ohe.shape}\n{[col for col in X_ohe.columns]}\n")

nullrow = [row for row in X_ohe.index if X_ohe.iloc[row, :].isnull().any()]
nullcol = [col for col in X_ohe.columns if X_ohe[col].isnull().any()]
# print(X_ohe.loc[nullrow, nullcol].loc[13516:13531, :])
# print({col: X_ohe[col].mean() for col in nullcol})

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_ohe_columns = X_ohe.columns
imp.fit(X_ohe)
X_ohe = pd.DataFrame(imp.transform(X_ohe), columns=X_ohe_columns)
# print(X_ohe.loc[nullrow, nullcol].loc[13516:13531, :])
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
indices = np.arange(X_ohe.shape[0])
# holdout
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X_ohe, y, indices, random_state=0)
print(f"{X_train.shape, X_test.shape, y_train.shape, y_test.shape, indices_train.shape, indices_test.shape}")
# modeling
pipe_gbr = Pipeline([('scl', StandardScaler()), ('est', GradientBoostingRegressor(random_state=0))])
pipe_gbr.fit(X_train, y_train.values.ravel())

print(f"r2_train: {r2_score(y_train.values.ravel(), pipe_gbr.predict(X_train)):.4f}")
print(f"r2_test:  {r2_score(y_test.values.ravel(), pipe_gbr.predict(X_test)):.4f}")
X_train.columns.shape
df_imp = pd.DataFrame(pipe_gbr.named_steps['est'].feature_importances_, index=X_train.columns, columns=['importance'])
print({i.upper()[:8]: df_imp.loc[i, 'importance'] for i in df_imp.index})
fig, axes = plt.subplots(figsize=(12, 3))
df_imp.plot.bar(ax=axes)
axes.set_xticklabels([c.upper()[:8] for c in X_train.columns])
fig.tight_layout()
pipe_gbr.named_steps['scl'].transform(X_train)
n_to_select = 6
df_sort = df_imp.sort_values(by='importance', ascending=False).reset_index().iloc[: n_to_select, :]
print(f"{df_sort}")
fig, axes = plt.subplots(figsize=(12, 4))
plot_partial_dependence(pipe_gbr.named_steps['est'], X=pipe_gbr.named_steps['scl'].transform(X_train), features=df_sort.index, 
                        feature_names=df_sort['index'], ax=axes)
fig.tight_layout()
