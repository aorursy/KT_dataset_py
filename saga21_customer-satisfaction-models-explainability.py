import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

import time

import warnings

warnings.filterwarnings('ignore')



# ML models

import lightgbm as lgb

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



# Additional libraries related to ML tasks

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, normalize

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_validate, cross_val_score

from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support

from sklearn.metrics import mean_squared_error as mse

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from imblearn.under_sampling import RandomUnderSampler

from imblearn.under_sampling import OneSidedSelection

import eli5

from eli5.sklearn import PermutationImportance

import shap
train = pd.read_csv("../input/santander-customer-satisfaction/train.csv")

test = pd.read_csv("../input/santander-customer-satisfaction/test.csv")

submission_example = pd.read_csv("../input/santander-customer-satisfaction/sample_submission.csv")



print("Train dataset:", len(train), "rows and", len(train.columns), "columns")

print("Test dataset:", len(test), "rows and", len(test.columns), "columns")

display(train.head(5))

display(train.describe())
len_train = len(train)

target_0 = len(train.loc[train['TARGET']==0])/len_train

target_1 = 1-target_0



fig, axes = plt.subplots(1, 2, figsize=(12,5))



# TARGET distribution count

sns.countplot(x='TARGET', ax=axes[0], data=train, palette='Set2')

axes[0].set_title('Target count')



# TARGET distribution pie chart

axes[1].pie([target_0, target_1], colors=['mediumaquamarine', 'coral'], autopct='%1.2f%%', shadow=True, startangle=90, wedgeprops={'alpha':.5})

axes[1].set_title('Target distribution')

plt.savefig('target_counts.png')
train[['var3','var15','var21','var36','var38']].hist(bins=100, figsize=(10, 8), alpha=0.5)

plt.savefig('var_columns_all.png')

plt.show()
fig, ax = plt.subplots(1, 3, sharex=False, sharey=False, figsize=(18,4))



train.loc[train.var3.between(5, 300), 'var3'].hist(bins=50, range=(5, 300), ax=ax[0], alpha=0.7)

ax[0].set_title("Var3 distribution")

ax[0].set_ylabel("Count")

ax[0].set_xlabel("var3")

ax[0].set_ylim(0,600)



train.loc[train.var15.between(5, 105), 'var15'].hist(bins=50, range=(5, 105), ax=ax[1], alpha=0.7)

ax[1].set_title("Var15 distribution")

ax[1].set_xlabel("var15")

ax[1].set_ylim(0,27000)



train.var38.hist(bins=100, range=(0, 500000), ax=ax[2], alpha=0.7)

ax[2].set_title("Var38 distribution")

ax[2].set_xlabel("var38")

ax[2].set_ylim(0,18000)



plt.savefig('var3_15_38.png')

plt.show()
print("Var3")

print("Max: ", train['var3'].max())

print("Min: ", train['var3'].min())

print("Unique values: ", train['var3'].nunique())



print("\nVar15")

print("Max: ", train['var15'].max())

print("Min: ", train['var15'].min())

print("Unique values: ", train['var15'].nunique())



print("\nVar38")

print("Max: ", train['var38'].max())

print("Min: ", train['var38'].min())

print("Unique values: ", train['var38'].nunique())
fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(7,4))

train.var21.hist(bins=100, range=(0, 30000), ax=ax, alpha=0.7)

ax.set_title("Var21 distribution")

ax.set_xlabel("var21")

ax.set_ylim(0,250)

ax.set_xlim(300,30000)

plt.savefig('var21_36.png')

plt.show()
print("Var21")

print("Max: ", train['var21'].max())

print("Min: ", train['var21'].min())

print("Unique values: ", train['var21'].nunique())

print("List of unique values: ", train['var21'].unique())



print("\nVar36")

print("Max: ", train['var36'].max())

print("Min: ", train['var36'].min())

print("Unique values: ", train['var36'].nunique())

print("List of unique values: ", train['var36'].unique())
# Correlation matrix of relevant features

corr = train.corr()

top20_corr = corr.nlargest(20, 'TARGET')['TARGET']



# Plot top20 correlations

fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=(7,4))

plt.bar(top20_corr[1:].index.values, top20_corr[1:].values, alpha=0.7)

plt.title("Top 20 most correlated columns with TARGET")

plt.ylabel("Correlation")

plt.xlabel("Features")

plt.xticks(rotation=90)

plt.savefig('correlation_target.png')

plt.show()
print("Number of mssing data in the dataset: ", train.isna().any().sum())
# Train/valid split

def split_dataset(data, split_size):

    y = data['TARGET']

    X = data.drop(['TARGET'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=split_size, random_state=21)

    return X_train, X_valid, y_train, y_valid



X_train, X_valid, y_train, y_valid = split_dataset(train, 0.2) 
ts = time.time()



LR = LogisticRegression()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LR, X_train, y_train, scoring=scoring, cv=5)



sorted(scores.keys())

LR_fit_time = scores['fit_time'].mean()

LR_score_time = scores['score_time'].mean()

LR_accuracy = scores['test_accuracy'].mean()

LR_precision = scores['test_precision_macro'].mean()

LR_recall = scores['test_recall_macro'].mean()

LR_f1 = scores['test_f1_weighted'].mean()

LR_roc = scores['test_roc_auc'].mean()



print("Time spent: ", time.time()-ts)
ts = time.time()



SVM = SVC(probability = True)



scoring = ['accuracy','precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(SVM, X_train, y_train, scoring=scoring, cv=5)



sorted(scores.keys())

SVM_fit_time = scores['fit_time'].mean()

SVM_score_time = scores['score_time'].mean()

SVM_accuracy = scores['test_accuracy'].mean()

SVM_precision = scores['test_precision_macro'].mean()

SVM_recall = scores['test_recall_macro'].mean()

SVM_f1 = scores['test_f1_weighted'].mean()

SVM_roc = scores['test_roc_auc'].mean()



print("Time spent: ", time.time()-ts)
ts = time.time()



LDA = LinearDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(LDA, X_train, y_train, scoring=scoring, cv=5)



sorted(scores.keys())

LDA_fit_time = scores['fit_time'].mean()

LDA_score_time = scores['score_time'].mean()

LDA_accuracy = scores['test_accuracy'].mean()

LDA_precision = scores['test_precision_macro'].mean()

LDA_recall = scores['test_recall_macro'].mean()

LDA_f1 = scores['test_f1_weighted'].mean()

LDA_roc = scores['test_roc_auc'].mean()



print("Time spent: ", time.time()-ts)
ts = time.time()



QDA = QuadraticDiscriminantAnalysis()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(QDA, X_train, y_train, scoring=scoring, cv=5)



sorted(scores.keys())

QDA_fit_time = scores['fit_time'].mean()

QDA_score_time = scores['score_time'].mean()

QDA_accuracy = scores['test_accuracy'].mean()

QDA_precision = scores['test_precision_macro'].mean()

QDA_recall = scores['test_recall_macro'].mean()

QDA_f1 = scores['test_f1_weighted'].mean()

QDA_roc = scores['test_roc_auc'].mean()



print("Time spent: ", time.time()-ts)
ts = time.time()



random_forest = RandomForestClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(random_forest, X_train, y_train, scoring=scoring, cv=5)



sorted(scores.keys())

forest_fit_time = scores['fit_time'].mean()

forest_score_time = scores['score_time'].mean()

forest_accuracy = scores['test_accuracy'].mean()

forest_precision = scores['test_precision_macro'].mean()

forest_recall = scores['test_recall_macro'].mean()

forest_f1 = scores['test_f1_weighted'].mean()

forest_roc = scores['test_roc_auc'].mean()



print("Time spent: ", time.time()-ts)
ts = time.time()

KNN = KNeighborsClassifier()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(KNN, X_train, y_train, scoring=scoring, cv=5)



sorted(scores.keys())

KNN_fit_time = scores['fit_time'].mean()

KNN_score_time = scores['score_time'].mean()

KNN_accuracy = scores['test_accuracy'].mean()

KNN_precision = scores['test_precision_macro'].mean()

KNN_recall = scores['test_recall_macro'].mean()

KNN_f1 = scores['test_f1_weighted'].mean()

KNN_roc = scores['test_roc_auc'].mean()



print("Time spent: ", time.time()-ts)
ts = time.time()



bayes = GaussianNB()



scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(bayes, X_train, y_train, scoring=scoring, cv=5)



sorted(scores.keys())

bayes_fit_time = scores['fit_time'].mean()

bayes_score_time = scores['score_time'].mean()

bayes_accuracy = scores['test_accuracy'].mean()

bayes_precision = scores['test_precision_macro'].mean()

bayes_recall = scores['test_recall_macro'].mean()

bayes_f1 = scores['test_f1_weighted'].mean()

bayes_roc = scores['test_roc_auc'].mean()



print("Time spent: ", time.time()-ts)
models_initial = pd.DataFrame({

    'Model'       : ['Logistic Regression', 'Support Vector Machine', 'Linear Discriminant Analysis', 'Quadratic Discriminant Analysis', 'Random Forest', 'K-Nearest Neighbors', 'Bayes'],

    'Fitting time': [LR_fit_time, SVM_fit_time, LDA_fit_time, QDA_fit_time, forest_fit_time, KNN_fit_time, bayes_fit_time],

    'Scoring time': [LR_score_time, SVM_score_time, LDA_score_time, QDA_score_time, forest_score_time, KNN_score_time, bayes_score_time],

    'Accuracy'    : [LR_accuracy, SVM_accuracy, LDA_accuracy, QDA_accuracy, forest_accuracy, KNN_accuracy, bayes_accuracy],

    'Precision'   : [LR_precision, SVM_precision, LDA_precision, QDA_precision, forest_precision, KNN_precision, bayes_precision],

    'Recall'      : [LR_recall, SVM_recall, LDA_recall, QDA_recall, forest_recall, KNN_recall, bayes_recall],

    'F1_score'    : [LR_f1,SVM_f1, LDA_f1, QDA_f1, forest_f1, KNN_f1, bayes_f1],

    'AUC_ROC'     : [LR_roc, SVM_roc, LDA_roc, QDA_roc, forest_roc, KNN_roc, bayes_roc],

    }, columns = ['Model', 'Fitting time', 'Scoring time', 'Accuracy', 'Precision', 'Recall', 'F1_score', 'AUC_ROC'])



models_initial.sort_values(by='AUC_ROC', ascending=False)
# Hyperparametrization of LGB model

def optimize_lgb(X_train, y_train, X_valid, y_valid): 

    

    n_HP_points_to_test = 100



    fit_params={"early_stopping_rounds":30, 

            "eval_metric" : 'auc', 

            "eval_set" : [(X_valid, y_valid)],

            'eval_names': ['valid'],

            'verbose': 0,

            'categorical_feature': 'auto'}

    

    param_test ={'max_depth': [4,5,6,7],

             'num_leaves': sp_randint(6, 50), 

             'min_child_samples': sp_randint(100, 500), 

             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

             'subsample': sp_uniform(loc=0.2, scale=0.8), 

             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),

             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

    

    clf = lgb.LGBMClassifier(max_depth=-1, random_state=21, silent=True, metric='auc', n_jobs=4, n_estimators=1000)

    

    gs = RandomizedSearchCV(

            estimator=clf, param_distributions=param_test, 

            n_iter=n_HP_points_to_test,

            scoring='roc_auc',

            cv=5,

            refit=True,

            random_state=21,

            verbose=0)

    

    gs.fit(X_train, y_train, **fit_params)

    print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))

    

    return gs, gs.best_score_, gs.best_params_, fit_params





# Raw data

X_train, X_valid, y_train, y_valid = split_dataset(train, 0.2)

clf_1, best_score_1, optimal_params_1, fit_params = optimize_lgb(X_train, y_train, X_valid, y_valid)
def clean_duplicates(df1, df2):

    remove = []

    cols = df1.columns

    for i in range(len(cols)-1):

        v = df1[cols[i]].values

        for j in range(i+1,len(cols)):

            if np.array_equal(v,df1[cols[j]].values):

                remove.append(cols[j])

    df1.drop(remove, axis=1, inplace=True)

    df2.drop(remove, axis=1, inplace=True)

    return df1, df2





def clean_constant_columns(df1, df2, threshold):

    constant_cols = []

    for i in df1.columns:

        counts = df1[i].value_counts()

        zeros = counts.iloc[0]

        if zeros / len(df1) * 100 > threshold:

            constant_cols.append(i)

    df1 = df1.drop(constant_cols, axis=1)

    df2 = df2.drop(constant_cols, axis=1)

    return df1, df2





# Duplicates of train/test for modification purposes

train_df = train.copy()

test_df = test.copy()



# Replace -999999 by -0.5 in var3

train_df.var3.replace(-999999, -0.5, inplace=True)

test_df.var3.replace(-999999, -0.5, inplace=True)



# Drop ID column

train_id = train_df['ID']

test_id = test['ID']

train_df.drop('ID', axis=1, inplace=True)

test_df.drop('ID', axis=1, inplace=True)



# Irrelevant columns cleaning 

train_df, test_df = clean_duplicates(train_df, test_df)

train_df, test_df = clean_constant_columns(train_df, test_df, 99.9)



# Split train dataset and find the best parameters for LGB

X_train, X_valid, y_train, y_valid = split_dataset(train_df, 0.20)

clf_2, best_score_2, optimal_params_2, fit_params = optimize_lgb(X_train, y_train, X_valid, y_valid)
def fix_skewness(df1, df2, nunique, max_skew):

    numeric_cols = [cname for cname in df1.columns if df1[cname].dtype in ['int64', 'float64']]

    skewed_feats = df1[numeric_cols].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)



    # Apply log1p to all columns with >nunique values, |skewness|>max_skew and x>-0.99

    log_col = []

    for col in skewed_feats.index:

        if(df1[col].nunique()>nunique):

            if(abs(skewed_feats[col])>max_skew): 

                if(df1[col].min()>=-0.99):

                    log_col.append(col)

                    df1[col]=df1[col].apply(lambda x: np.log1p(x))

                    df2[col]=df2[col].apply(lambda x: np.log1p(x))

    return df1, df2, log_col





def var38_flag(df1, df2):

    df1['var38_flag'], df2['var38_flag'] = 0, 0

    var38_mode = df1.var38.mode()

    df1.loc[df1['var38']==var38_mode[0], ['var38', 'var38_flag']] = 0, 1

    df2.loc[df2['var38']==var38_mode[0], ['var38', 'var38_flag']] = 0, 1

    return df1, df2

    

    

ts = time.time()



train_df_skw, test_df_skw = train_df.copy(), test_df.copy()



# Transform skewed features (log1p), create flag for var38 mode

train_df_skw, test_df_skw = var38_flag(train_df_skw, test_df_skw)

train_df_skw, test_df_skw, cols_skw = fix_skewness(train_df_skw, test_df_skw, 50, 0.7)



# Split train dataset and find the best parameters for LGB

X_train_skw, X_valid_skw, y_train_skw, y_valid_skw = split_dataset(train_df_skw, 0.20)

clf_3, best_score_3, optimal_params_3, fit_params = optimize_lgb(X_train_skw, y_train_skw, X_valid_skw, y_valid_skw)



print("Time spent: ", time.time()-ts)
clf_lgb_1 = lgb.LGBMClassifier(   colsample_bytree= 0.5041066856718041, 

                                max_depth= 6, 

                                min_child_samples= 215, 

                                min_child_weight= 0.01, 

                                num_leaves= 47, 

                                reg_alpha= 2, 

                                reg_lambda= 5, 

                                subsample= 0.7631144227290101, 

                                random_state=21, 

                                silent=True, 

                                metric='auc', 

                                n_jobs=4, 

                                n_estimators=1000)

clf_lgb_1.fit(X_train, y_train)



# Feature importance

feature_imp = pd.DataFrame(sorted(zip(clf_lgb_1.feature_importances_,X_train.columns)), columns=['Value','Feature']).sort_values(by=['Value'], ascending=False)[:50]

plt.figure(figsize=(20, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGB feature importance')

plt.tight_layout()

plt.savefig('lgb_feature_importance.png')

plt.show()
perm = PermutationImportance(clf_lgb_1, random_state=21).fit(X_valid, y_valid)

eli5.show_weights(perm, feature_names = X_valid.columns.tolist())
X_train_red = X_train

X_valid_red = X_valid



clf_lgb_1.fit(X_train_red, y_train)



explainer = shap.TreeExplainer(clf_lgb_1)

shap_values = explainer.shap_values(X_valid_red)

shap.summary_plot(shap_values[1], X_valid_red)

plt.savefig('shap_summary.png')
shap.dependence_plot('var38', shap_values[1], X_valid_red)

plt.savefig('shap_var38.png')

shap.dependence_plot('var15', shap_values[1], X_valid_red)

plt.savefig('shap_var15.png')

shap.dependence_plot('var36', shap_values[1], X_valid_red)

plt.savefig('shap_var36.png')

shap.dependence_plot('var3', shap_values[1], X_valid_red)

plt.savefig('shap_var3.png')
shap.dependence_plot('saldo_var5', shap_values[1], X_valid_red)

plt.savefig('shap_saldo_var5.png')

shap.dependence_plot('saldo_medio_var5_hace3', shap_values[1], X_valid_red)

plt.savefig('shap_saldo_medio_var5_hace3.png')

shap.dependence_plot('saldo_medio_var5_ult3', shap_values[1], X_valid_red)

plt.savefig('shap_saldo_medio_var5_ult3.png')
shap.dependence_plot('num_var30', shap_values[1], X_valid_red)

plt.savefig('shap_num_var30.png')

shap.dependence_plot('saldo_var30', shap_values[1], X_valid_red)

plt.savefig('shap_saldo_var30.png')
shap.dependence_plot('num_var45_hace2', shap_values[1], X_valid_red)

plt.savefig('shap_num_var45_hace2.png')

shap.dependence_plot('num_var45_hace3', shap_values[1], X_valid_red)

plt.savefig('shap_num_var45_hace3.png')
clf_lgb = lgb.LGBMClassifier(   colsample_bytree= 0.6641632439141401, 

                                max_depth= 5, 

                                min_child_samples= 224, 

                                min_child_weight= 0.1, 

                                num_leaves= 28, 

                                reg_alpha= 2, 

                                reg_lambda= 10, 

                                subsample= 0.9244622777209361,  

                                random_state=21,

                                silent=True, 

                                metric='auc', 

                                n_jobs=4, 

                                n_estimators=1000)





clf_lgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=10)

test_df = test_df[X_train.columns]

probs = clf_lgb.predict_proba(test_df)



submission = pd.DataFrame({"ID":test_id, "TARGET": probs[:,1]})

submission.to_csv("submission.csv", index=False)