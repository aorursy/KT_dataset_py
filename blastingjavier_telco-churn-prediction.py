import pandas as pd

import numpy as np

import seaborn as sns



import matplotlib.pyplot as plt
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
train.head()
traincopy = train.copy()

testcopy = test.copy()
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA#, FastICA

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, learning_curve, RandomizedSearchCV

from sklearn.svm import LinearSVC,SVC

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB

from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score



from imblearn.pipeline import make_pipeline, Pipeline

from imblearn.over_sampling import SMOTE, ADASYN, KMeansSMOTE



from sklearn.metrics import accuracy_score,roc_auc_score, f1_score



SEED = 17 # specify seed for reproducable results

import xgboost as xgb

import lightgbm as lgb
RANDOM_FOREST_PARAMS = {

    'clf__max_depth': [25, 50, 75],

    'clf__max_features': ["sqrt"], # just sqrt is used because values of log2 and sqrt are very similar for our number of features (10-19) 

    'clf__criterion': ['gini', 'entropy'],

    'clf__n_estimators': [100, 300, 500, 1000]

}



DECISION_TREE_PARAMS = {

    'clf__max_depth': [25, 50, 75],

    'clf__max_features': ["sqrt"], # just sqrt is used because values of log2 and sqrt are very similar for our number of features (10-19)

    'clf__criterion': ['gini', 'entropy'],

    'clf__min_samples_split': [6, 10, 14],

}



LOGISTIC_REGRESSION_PARAMS = {

    'clf__solver': ['liblinear'],

    'clf__C': [0.1, 1, 10],

    'clf__penalty': ['l2', 'l1']

}



KNN_PARAMS = {

    'clf__n_neighbors': [5, 15, 25, 35, 45, 55, 65],

    'clf__weights': ['uniform', 'distance'],

    'clf__p': [1, 2, 10]

}



KNN_PARAMS_UNIFORM = {

    'clf__n_neighbors': [5, 15, 25, 35, 45, 55, 65],

    'clf__weights': ['uniform'],

    'clf__p': [1, 2, 10]

}



COMPLEMENT_NB_PARAMS = {

    'clf__alpha': [0, 1]

}



ADA_PARAMS = {

    'clf__n_estimators': [50, 100, 200, 300]

}



GB_PARAMS = {

    'clf__learning_rate': [0.1, 0.5, 1],

    'clf__n_estimators': [50, 100, 180, 200, 240, 300],

    'clf__max_depth': [4, 8, 12],

    'clf__min_samples_split': [0.2, 0.4, 0.8, 1.0],

    'clf__min_samples_leaf': [0.2, 0.4]

}



ET_PARAMS = {

    

}



SVM_PARAMS = [

{

    'clf__kernel': ['linear'],

    'clf__C': [0.1, 1, 10],

}, 

{

    'clf__kernel': ['rbf'],

    'clf__C': [0.01, 0.1, 1, 10, 100],

    'clf__gamma': [0.01, 0.1, 1, 10, 100],

}]



# A parameter grid for XGBoost

# A parameter grid for XGBoost

XGB_PARAMS = {

        'clf__lambda': [1.5, 2, 3, 4],

        'clf__min_child_weight': [1, 2, 5, 7, 10],

        'clf__gamma': [0.5, 1, 2, 4, 8],

        'clf__subsample': [0.6, 0.8, 1.0],

        'clf__colsample_bytree': [0.7, 0.8],

        'clf__max_depth': [4, 5, 6, 7],

        'clf__scale_pos_weight': [5, 7, 8, 12]

        #'clf__n_estimators':[50, 100, 150, 200, 300, 400, 500, 600, 750, 1000],

        #'clf__learning_rate':[0.01, 0.1, 0.5, 1]

        }



LGB_PARAMS = {

    'clf__learning_rate': [0.07],

    'clf__n_estimators': [8,16, 64, 120, 200, 300],

    'clf__num_leaves': [20, 24, 27],

    'clf__boosting_type' : ['gbdt'],

    'clf__objective' : ['binary'],

    'clf__colsample_bytree' : [0.64, 0.65],

    'clf__min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

    'clf__reg_alpha': [0, 1e-3, 1e-1, 1, 10, 50, 100],

    'clf__reg_lambda': [0, 1e-3, 1e-1, 1, 10, 50, 100]

    }
sns.countplot(traincopy['Churn'])
sns.countplot(traincopy['Voice mail plan'], hue=traincopy['Churn'])
sns.countplot(traincopy['International plan'], hue=traincopy['Churn'])
def preprocessing_ds(ds):

    le = LabelEncoder()

    objecttypes = [key for key in dict(ds.dtypes) if dict(ds.dtypes)[key] in ['object']]

    print("Objects types in the dataset: {}".format(objecttypes))

    for column in objecttypes:

        le.fit(ds[column])

        ds[column] = le.fit_transform(ds[column])
preprocessing_ds(traincopy)
traincopy.head()
traincopy.describe()
plt.figure(figsize=(20,10))

corrmat = traincopy.corr()

k = 25 #number of variables for heatmap

cols = corrmat.nlargest(k, 'Churn')['Churn'].index

cm = np.corrcoef(traincopy[cols].values.T)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

from sklearn import cluster
from scipy.cluster import hierarchy as hc

names = traincopy.columns

inverse_correlation = 1 - abs(traincopy.corr())

Z = hc.linkage(inverse_correlation.values,'average')

plt.figure(figsize=(15, 10))

dendo = hc.dendrogram(Z, labels=names, orientation='left')
duplicates_feats = ['Total day charge', 'Total intl charge', 'Total eve charge', 'Total night charge']
df_y = traincopy['Churn']

df_x = traincopy.drop(['Churn'], axis=1)
df_x_normed = (df_x - df_x.mean()) / df_x.std()
df_x_normed.head()
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE
#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(df_x,df_y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(df_x.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
etmodel = ExtraTreesClassifier()

etmodel.fit(df_x,df_y)

feat_importances = pd.Series(etmodel.feature_importances_, index=df_x.columns).sort_values(kind="quicksort", ascending=False).reset_index()

print(feat_importances)
# feature extraction

model = LogisticRegression()

rfe = RFE(model, 10)

fit = rfe.fit(df_x, df_y)

print(dict(zip(df_x.columns, fit.support_)))

print("Feature Ranking: {}".format(fit.ranking_))
fe = list(featureScores.nlargest(10,'Score').Specs.values)

feExtra = list(feat_importances['index'].values)

lrfe = [column for index,column in enumerate(df_x) if fit.ranking_[index] == 1]
dicctofBestColumns = {}

for column in feExtra:

    dicctofBestColumns[column] = 0

    if column in fe and column in lrfe:

        dicctofBestColumns[column] +=1

dicctofBestColumns
dicctofBest = {}

for column in fe:

    dicctofBest[column] = 0

    if column in lrfe:

        dicctofBest[column] += 1

dicctofBest
# calculate the principal components

pca = PCA(random_state=SEED)

df_x_pca = pca.fit_transform(df_x_normed)
print(LGB_PAR)
n_components = 10

df_x_reduced = np.dot(df_x_normed.values, pca.components_[:n_components,:].T)

df_x_reduced = pd.DataFrame(df_x_reduced, columns=["PC#%d" % (x + 1) for x in range(n_components)])
df_x_reduced.head()
# prints the best grid search scores along with their parameters.

def print_best_grid_search_scores_with_params(grid_search, n=5):

    if not hasattr(grid_search, 'best_score_'):

        raise KeyError('grid_search is not fitted.')

    print("Best grid scores on validation set:")

    indexes = np.argsort(grid_search.cv_results_['mean_test_score'])[::-1][:n]

    means = grid_search.cv_results_['mean_test_score'][indexes]

    stds = grid_search.cv_results_['std_test_score'][indexes]

    params = np.array(grid_search.cv_results_['params'])[indexes]

    for mean, std, params in zip(means, stds, params):

        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
def do_gridsearch_with_cv(clf, params, name, X_trainn, y_trainn, cv, smote=None, random=None, n_sampling=0):



    if smote is None:

        pipeline = Pipeline([('clf', clf)])

    else:

        pipeline = Pipeline([('imb', smote), ('clf', clf)])

        

    if random is None:

        gs = GridSearchCV(pipeline, params, cv=kf, n_jobs=-1, scoring='f1', return_train_score=True, verbose=4)

        gs.fit(X_trainn, y_trainn)

        print(gs.best_params_)

        return gs

    else:

        gsrandom = RandomizedSearchCV(pipeline, params, cv=kf, n_iter=n_sampling, n_jobs=-1, scoring='f1', return_train_score=True, verbose=4)

        gsrandom.fit(X_trainn, y_trainn)

        print(gsrandom.best_params_)

        return gsrandom





#Return the scores in the test data

def score_on_test_set(clfs, datasets):

    scores = []

    for c, (X_test, y_test) in zip(clfs, datasets):

        scores.append(c.score(X_test, y_test))

    return scores
# split data into train and test set in proportion 4:1 for all differntly preprocessed datasets

X_train, X_test, y_train, y_test = train_test_split(df_x_normed, df_y, test_size=0.15, random_state=SEED, stratify=df_y)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(df_x_reduced, df_y, test_size=0.15, random_state=SEED, stratify=df_y)

cols_without_duplicate = [x for x in df_x_normed.columns if x not in duplicates_feats]

X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(df_x_normed[cols_without_duplicate], df_y, test_size=0.15, random_state=SEED, stratify=df_y)
sm = SMOTE(random_state=SEED)

kf = StratifiedKFold(n_splits=5, random_state=SEED)

kmeansmote = KMeansSMOTE(random_state=SEED)

adasyn = ADASYN(random_state=SEED)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
dsbalanced = DecisionTreeClassifier(random_state=SEED, class_weight='balanced')

ds = DecisionTreeClassifier(random_state=SEED)

clf_rf = RandomForestClassifier(random_state=SEED)

clf_rf_balanced = RandomForestClassifier(random_state=SEED, class_weight='balanced')

ada = AdaBoostClassifier(random_state=SEED)

gb = GradientBoostingClassifier()

etrees = ExtraTreesClassifier()



knn = KNeighborsClassifier()

rnn = RadiusNeighborsClassifier()

lr = LogisticRegression(random_state=SEED)

svm = SVC(random_state=SEED, probability=True)

gnb = ComplementNB()



xgbc = xgb.XGBClassifier(nthread=1, n_estimators=200, learning_rate=0.5, random_state=SEED)

lgbc = lgb.LGBMClassifier(nthread=1, n_estimators=200, learning_rate=0.5, random_state=SEED)
gsclf_rf_balanced = do_gridsearch_with_cv(clf_rf_balanced, RANDOM_FOREST_PARAMS,'balanced', X_train, y_train, cv=kf, smote=None)
gsclf_rf = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS,'RandomForest', X_train, y_train, cv=kf, smote=sm)
gsclf_rfadasyn = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS,'Random Forest Adasyn', X_train, y_train, cv=kf, smote=adasyn)
gsds = do_gridsearch_with_cv(ds, DECISION_TREE_PARAMS, 'DecisisonTree', X_train, y_train, cv=kf, smote=sm)
gsdsada = do_gridsearch_with_cv(ds, DECISION_TREE_PARAMS, 'DecisisonTree', X_train, y_train, cv=kf, smote=adasyn)
gsdsbalanced = do_gridsearch_with_cv(dsbalanced, DECISION_TREE_PARAMS, 'DecisisonTree', X_train, y_train, cv=kf, smote=sm)
gsada_smote = do_gridsearch_with_cv(ada, ADA_PARAMS, 'AdaBoost', X_train, y_train, cv=kf, smote=sm)
gsada_adasyn = do_gridsearch_with_cv(ada, ADA_PARAMS, 'AdaBoost', X_train, y_train, cv=kf, smote=adasyn)
gsgb_smote =  do_gridsearch_with_cv(gb, GB_PARAMS, 'Gradient Boosting', X_train, y_train, cv=kf, smote=sm)
gsgb_adasyn = do_gridsearch_with_cv(gb, GB_PARAMS, 'Gradient Boosting ada', X_train, y_train, cv=kf, smote=adasyn)
gsetrees_smote = do_gridsearch_with_cv(etrees, ET_PARAMS, 'Extra trees', X_train, y_train, cv=kf, smote=sm)
gsetrees_ada = do_gridsearch_with_cv(etrees, ET_PARAMS, 'Extra trees ada', X_train, y_train, cv=kf, smote=adasyn)
gsknn = do_gridsearch_with_cv(knn, KNN_PARAMS, 'KNN', X_train, y_train, cv=kf, smote=sm)
gsknn_adasyn = do_gridsearch_with_cv(knn, KNN_PARAMS, 'KNN', X_train, y_train, cv=kf, smote=adasyn)
gslr = do_gridsearch_with_cv(lr, LOGISTIC_REGRESSION_PARAMS, 'Log Regression', X_train, y_train, cv=kf, smote=sm)
gslrada = do_gridsearch_with_cv(lr, LOGISTIC_REGRESSION_PARAMS, 'Log Regression', X_train, y_train, cv=kf, smote=adasyn)
gssvm = do_gridsearch_with_cv(svm, SVM_PARAMS,'Support Vector Machine', X_train, y_train, cv=kf, smote=sm)
gssvmada = do_gridsearch_with_cv(svm, SVM_PARAMS,'Support Vector Machine', X_train, y_train, cv=kf, smote=adasyn)
#gsgnb = do_gridsearch_with_cv(gnb, COMPLEMENT_NB_PARAMS,'Naive Bayes', X_train, y_train, cv=kf, smote=sm)
#gsgnbada = do_gridsearch_with_cv(gnb, COMPLEMENT_NB_PARAMS,'Naive Bayes', X_train, y_train, cv=kf, smote=adasyn)
gsxgbc = do_gridsearch_with_cv(xgbc, XGB_PARAMS,'XGBOOST', X_train, y_train, cv=kf, smote=None, random=True, n_sampling=120)
gslgbc = do_gridsearch_with_cv(lgbc, LGB_PARAMS,'LGBOOST', X_train, y_train, cv=kf, smote=None, random=True, n_sampling=140)
print_best_grid_search_scores_with_params(gslgbc)
from joblib import dump, load
dump(gslgbc, 'gslgbc.joblib')

dump(gsxgbc, 'gsxgbc.joblib')

dump(gsclf_rf_balanced, 'gsclf_rf_balanced.joblib')

dump(gsclf_rf, 'gsclf_rf.joblib')

dump(gsclf_rfadasyn, 'gsclf_rfadasyn.joblib')

dump(gsds, 'gsds.joblib')

dump(gsdsbalanced, 'gsdsbalanced.joblib')

dump(gsada_smote, 'gsada_smote.joblib')

dump(gsada_adasyn, 'gsada_adasyn.joblib')

dump(gsgb_smote, 'gsgb_smote.joblib')



dump(gsgb_adasyn, 'gsgb_adasyn.joblib')

dump(gsetrees_smote, 'gsetrees_smote.joblib')

dump(gsetrees_ada, 'gsetrees_ada.joblib')

dump(gsknn, 'gsknn.joblib')

dump(gslr, 'gslr.joblib')

dump(gslrada, 'gslrada.joblib')

dump(gssvm, 'gssvm.joblib')

dump(gssvmada, 'gssvmada.joblib')
gs_rf_score = gsclf_rfadasyn.score(X_test, y_test)

y_pred_rf = gsclf_rfadasyn.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)

cm_rf = cm_rf.astype('float') / cm_rf.sum(axis=1)[:, np.newaxis] # normalize the confusion matri
cm_df = pd.DataFrame(cm_rf.round(3), index=["true no churn", "true churn"], columns=["predicted no churn", "predicted churn"])

cm_df
classifiers = [gslgbc, gsxgbc, gsclf_rf_balanced, gsclf_rf, gsclf_rfadasyn,

              gsds, gsdsbalanced, gsada_smote, gsada_adasyn, gsgb_smote, gsgb_adasyn,

              gsetrees_smote, gsetrees_ada, gsknn, gslr, gslrada, gssvm, gssvmada]

classifier_names = ["gslgbc", "gsxgbc", "gsclf_rf_balanced", "gsclf_rf Tree", "gsclf_rfadasyn",

                   'gsds', 'gsdsbalanced', 'gsada_smote', 'gsada_adasyn', 'gsgb_smote', 'gsgb_adasyn',

                   'gsetrees_smote', 'gsetrees_ada', 'gsknn', 'gslr', 'gslrada', 'gssvm', 'gssvmada']

dump(classifiers,'classifiers.joblib')
accs = []

recalls = []

precision = []

results_table = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1", "auc"])

for (i, clf), name in zip(enumerate(classifiers), classifier_names):

    y_pred = clf.predict(X_test)

    row = []

    row.append(accuracy_score(y_test, y_pred))

    row.append(precision_score(y_test, y_pred))

    row.append(recall_score(y_test, y_pred))

    row.append(f1_score(y_test, y_pred))

    row.append(roc_auc_score(y_test, y_pred))



    row = ["%.3f" % r for r in row]

    results_table.loc[name] = row
results_table
test_results_raw = score_on_test_set(gss_raw, [(X_test, y_test), (X_test_red, y_test_red), (X_test_pca, y_test_pca)])
gs_full_balanced = do_gridsearch_with_cv(clf_rf_balanced, RANDOM_FOREST_PARAMS, X_train, y_train, kf, smote=None)

#gs_red_balanced = do_gridsearch_with_cv(clf_rf_balanced, RANDOM_FOREST_PARAMS, X_train_red, y_train_red, kf, smote=None)

#gs_pca_balanced = do_gridsearch_with_cv(clf_rf_balanced, RANDOM_FOREST_PARAMS, X_train_pca, y_train_pca, kf, smote=None)

gss_balanced_weights = [gs_full_balanced]
test_results_balanced_weights = score_on_test_set(gss_balanced_weights, [(X_test, y_test), (X_test_red, y_test_red), (X_test_pca, y_test_pca)])
gs_full_smote = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train, y_train, kf, smote=sm)

#gs_red_smote = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train_red, y_train_red, kf, smote=sm)

#gs_pca_smote = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train_pca, y_train_pca, kf, smote=sm)

gss_smote = [gs_full_smote]
test_results_smote = score_on_test_set(gss_smote, [(X_test, y_test), (X_test_red, y_test_red), (X_test_pca, y_test_pca)])
gs_full_kmean = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train, y_train, kf, smote=kmeansmote)

gs_red_kmean = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train_red, y_train_red, kf, smote=kmeansmote)

gs_pca_kmean = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train_pca, y_train_pca, kf, smote=kmeansmote)

gss_kmean = [gs_full_kmean, gs_red_kmean, gs_pca_kmean]
test_results_kmean = score_on_test_set(gss_kmean, [(X_test, y_test), (X_test_red, y_test_red), (X_test_pca, y_test_pca)])
gs_full_adasyn = do_gridsearch_with_cv(clf_rf, RANDOM_FOREST_PARAMS, X_train, y_train, kf, smote=adasyn)

gss_adasyn = [gs_full_adasyn]
test_results_adasyn = score_on_test_set(gss_adasyn, [(X_test, y_test), (X_test_red, y_test_red), (X_test_pca, y_test_pca)])
dataset_strings = ["full dataset", "data set with reduced features", "dataset with first 10 principal components"]

method_strings = ["without any balancing", "using balanced class weights", "using SMOTE", "using ADASYN"]



result_strings = dict()

for ms, results in zip(method_strings, [test_results_raw, test_results_balanced_weights, test_results_smote, test_results_adasyn]):

    for ds, res in zip(dataset_strings, results):

        string = "%.3f" % res + "     " + ds + " " + ms

        result_strings[string] = res

        2

result_strings = sorted(result_strings.items(), key=lambda kv: kv[1], reverse=True)

print("F1 score  dataset and method")

for k, _ in result_strings:

    print(k)
clf_xgb = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 

                        subsample=0.8, nthread=10, learning_rate=0.1)
xgb_full = do_gridsearch_with_cv(clf_xgb, XGB_PARAMS, X_train, y_train, kf, smote=None)

xgb_red = do_gridsearch_with_cv(clf_xgb, XGB_PARAMS, X_train_red, y_train_red, kf, smote=None)

test_results_xgb = [xgb_full, xgb_red]
test_results_xgb_full = score_on_test_set(test_results_xgb, [(X_test, y_test), (X_test_red, y_test_red)])

print(test_results_xgb_full)
xgb3 = xgb.XGBClassifier(nthread=1, n_estimators=200, learning_rate=0.5)
xgb_full = do_gridsearch_with_cv(xgb3, XGB_PARAMS, X_train, y_train, kf, smote=None, random=True, n_sampling=100)

gg2 = [xgb_full]
xgb_full2 = do_gridsearch_with_cv(xgb3, XGB_PARAMS, X_train, y_train, kf, smote=None, random=True, n_sampling=100)

gg2 = [xgb_full2]
test_xgb2 = score_on_test_set(gg2, [(X_test, y_test)])
xgb_adasyn = do_gridsearch_with_cv(xgb3, XGB_PARAMS, X_train, y_train, kf, smote=adasyn, random=True, n_sampling=100)

ggadasyn = [xgb_adasyn]