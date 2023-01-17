import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('../input/clinvar_conflicting.csv')

df.head()
df.info()
df = df.drop(['CLNDISDBINCL', 'CLNDNINCL', 'CLNSIGINCL', 'SSR', 'DISTANCE', 'MOTIF_NAME', 'MOTIF_POS', 'HIGH_INF_POS', 'MOTIF_SCORE_CHANGE'], axis = 1)
for var in ['CLNVI', 'INTRON', 'BAM_EDIT', 'SIFT', 'PolyPhen', 'BLOSUM62']:

    df[var] = df[var].apply(lambda x: 1 if x == x else 0).astype('category')

    print(df[var].value_counts())
df = df.rename({'CLASS': 'target'}, axis = 1)

df['target'] = df['target'].astype('category')

df['target'].value_counts()
df['CHROM'].value_counts()
df['CHROM'] = df['CHROM'].astype('str').apply(lambda x: x.strip())

df['CHROM'] = df['CHROM'].astype('category')
df['POS'].describe()
for var in ['REF', 'ALT', 'Allele']:

    print(df[var].value_counts()[0:10])
for var in ['REF', 'ALT', 'Allele']:

    df[var] = df[var].apply(lambda x: 'O' if x not in ['A', 'C', 'G', 'T'] else x).astype('category')
df[['AF_ESP', 'AF_EXAC', 'AF_TGP']].describe()
df[['AF_ESP', 'AF_EXAC', 'AF_TGP']].hist()
df['AF_ESP'] = df['AF_ESP'].apply(lambda x: 1 if x > 0 else 0).astype('category')

df['AF_EXAC'] = df['AF_EXAC'].apply(lambda x: 1 if x > 0 else 0).astype('category')

df['AF_TGP'] = df['AF_TGP'].apply(lambda x: 1 if x > 0 else 0).astype('category')
print(len(df['CLNDISDB'].unique()))

df['CLNDISDB'].value_counts()[0:10]
df = df.drop('CLNDISDB', axis = 1)
print(len(df['CLNDN'].unique()))

df['CLNDN'].value_counts()[0:20]
name_df = df['CLNDN'].str.split(pat = '|', expand = True)

name_df.head()

top_100_dn = name_df.apply(pd.value_counts).sum(axis=1).sort_values(ascending = False)[0:100]

print(top_100_dn[0:10])



top_100_dn_list = list(top_100_dn.index)

print(top_100_dn_list[0:10])
for dn in top_100_dn_list:

    df[dn] = df['CLNDN'].apply(lambda x: 1 if dn in x else 0).astype('category')

df = df.drop('CLNDN', axis = 1)
print(df.columns)
print(len(df['CLNHGVS'].unique()))

df = df.drop('CLNHGVS', axis = 1)
print(df['CLNVC'].value_counts())
clnvc_types = ['single_nucleotide_variant', 'Deletion', 'Duplication']

df['CLNVC'] = df['CLNVC'].apply(lambda x: x if x in clnvc_types else 'Other').astype('category')
df['MC'].value_counts()[0:10]
name_df = df['MC'].str.split(pat = '[|,]', expand = True)

name_df.head()

top_mc = name_df.apply(pd.value_counts).sum(axis=1).sort_values(ascending = False)[0:20]

print(top_mc)



top_mc_list = [x for x in list(top_mc.index) if 'SO:' not in x]

print(top_mc_list)
df['MC'] = df['MC'].fillna('unknown')

for mc in top_mc_list:

    df[mc] = df['MC'].apply(lambda x: 1 if mc in x else 0).astype('category')

    print(df[mc].value_counts())

df = df.drop('MC', axis = 1)
df['ORIGIN'] = df['ORIGIN'].fillna(0).apply(lambda x: 1 if x == 1.0 else 0).astype('category')
name_df = df['Consequence'].str.split(pat = '&', expand = True)

name_df.head()

top_mc = name_df.apply(pd.value_counts).sum(axis=1).sort_values(ascending = False)

print(top_mc[0:20])
for mc in top_mc_list:

    mc2 = mc + '2'

    df[mc2] = df['Consequence'].apply(lambda x: 1 if mc in x else 0).astype('category')

    df[mc] = df[[mc, mc2]].apply(lambda x: max(x[mc], x[mc2]), axis = 1).astype('category')

    print(df[mc].value_counts())

    df=df.drop(mc2, axis = 1)

df = df.drop('Consequence', axis = 1)
df['IMPACT'].value_counts()
df['IMPACT'] = df['IMPACT'].astype('category')
len(df['SYMBOL'].unique())
df['SYMBOL'].value_counts()[0:10]
top_100_symb = df['SYMBOL'].value_counts()[0:100].index

df['SYMBOL'] = df['SYMBOL'].apply(lambda x: x if x in top_100_symb else 'Other').astype('category')
df['SYMBOL'].value_counts()[0:100]
df = df.drop('Feature', axis = 1)
for var in ['Feature_type', 'BIOTYPE']:

    print(df[var].value_counts())

    df = df.drop(var, axis = 1)
len(df['EXON'].unique())
df = df.drop('EXON', axis = 1)
df = df.drop(['cDNA_position', 'CDS_position', 'Protein_position'], axis = 1)
df = df.drop(['Amino_acids', 'Codons'], axis = 1)
df['STRAND'].value_counts()
df['STRAND'] = df['STRAND'].fillna(df['STRAND'].mode())

df['STRAND'] = df['STRAND'].astype('category')
df['LoFtool'] = df['LoFtool'].fillna(df['LoFtool'].median())
df['CADD_PHRED'] = df['CADD_PHRED'].fillna(df['CADD_PHRED'].median())
df['CADD_RAW'] = df['CADD_RAW'].fillna(df['CADD_RAW'].median())
from sklearn.preprocessing import StandardScaler



num_var_list = ['POS', 'LoFtool', 'CADD_PHRED', 'CADD_RAW']

scl = StandardScaler()

df[num_var_list] = scl.fit_transform(df[num_var_list])
target = df['target']

features = df.drop('target', axis = 1)
#Original columns

list(df.columns[0:23])
df.iloc[:, 0:23].info()
#Original feature set

orig_feat = list(features.columns[0:22])

orig_feat_cat = [x for x in orig_feat if x not in num_var_list]
features[num_var_list].describe()
features[num_var_list].hist()
plt.figure(figsize=(8,8))

sns.heatmap(features[num_var_list].corr(),

            vmin=0,

            vmax=1,

            cmap='YlGnBu',

            annot=np.round(features[num_var_list].corr(), 2))
features = features.drop('CADD_RAW', axis = 1)
import scipy.stats as ss



def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
num_feat = len(orig_feat_cat)

cat_corr_arr = np.empty((num_feat, num_feat))

for i, row in enumerate(orig_feat_cat):

    for j, col in enumerate(orig_feat_cat):

        #print((i, j))

        cat_corr_arr[i, j] = cramers_v(features[row], features[col])

print(cat_corr_arr[0:5, 0:5])
plt.figure(figsize=(16, 14))

sns.heatmap(cat_corr_arr,

            vmin=0,

            vmax=1,

            cmap='YlGnBu',

            xticklabels = orig_feat_cat,

            yticklabels = orig_feat_cat,

            annot=np.round(cat_corr_arr, 2))
features = features.drop(['Allele', 'IMPACT', 'SYMBOL', 'PolyPhen'], axis = 1)
from sklearn.dummy import DummyClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.model_selection import GridSearchCV, train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, make_scorer
features = pd.get_dummies(features, drop_first = True)

print(features.columns)
f1_scorer = make_scorer(f1_score)
dm_clf = DummyClassifier(random_state = 42)

mean_dm_cv_score = cross_val_score(dm_clf, features, target, scoring = f1_scorer, cv = 3).mean()

print("Mean Cross Validation F1 Score for Dummy Classifier: {:.3}".format(mean_dm_cv_score))
gnb_clf = GaussianNB()

mean_gnb_cv_score = cross_val_score(gnb_clf, features, target, scoring = f1_scorer, cv = 3).mean()

print("Mean Cross Validation F1 Score for Gaussian Naive Bayes Classifier: {:.3}".format(mean_gnb_cv_score))
bnb_clf = BernoulliNB()

mean_bnb_cv_score = cross_val_score(bnb_clf, features, target, scoring = f1_scorer, cv = 3).mean()

print("Mean Cross Validation F1 Score for Bernoulli Naive Bayes Classifier: {:.3}".format(mean_bnb_cv_score))
adb_clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(), random_state = 42)

mean_adb_cv_score = cross_val_score(adb_clf, features, target, scoring = f1_scorer, cv = 3).mean()

print("Mean Cross Validation F1 Score for AdaBoost Decision Tree Classifier: {:.3}".format(mean_adb_cv_score))
adb_clf = AdaBoostClassifier(base_estimator = LogisticRegression(solver = 'lbfgs'), random_state = 42)

mean_adb_cv_score = cross_val_score(adb_clf, features, target, scoring = f1_scorer, cv = 3).mean()

print("Mean Cross Validation F1 Score for AdaBoost Logistic Regression Classifier: {:.3}".format(mean_adb_cv_score))
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(seed = 123)

mean_xgb_cv_score = cross_val_score(xgb_clf, features, target, scoring = f1_scorer, cv = 3).mean()

print("Mean Cross Validation F1 Score for XGBoost Classifier: {:.3}".format(mean_xgb_cv_score))
rf_clf = RandomForestClassifier(n_estimators = 100, random_state = 42)

mean_rf_cv_score = cross_val_score(rf_clf, features, target, scoring = f1_scorer, cv = 3).mean()

print("Mean Cross Validation F1 Score for Random Forest Classifier: {:.3}".format(mean_rf_cv_score))
bnb_param_grid = {

'alpha': [0.1, 0.5, 1, 2, 5],

'fit_prior': [True, False]

}
import time

start = time.time()

bnb_grid_search = GridSearchCV(bnb_clf, bnb_param_grid, scoring = f1_scorer, cv = 3)

bnb_grid_search.fit(features, target)



print("Cross Validation F1 Score: {:.3}".format(bnb_grid_search.best_score_))

print("Total Runtime for Grid Search on Bernoulli Naive Bayes: {:.4} seconds".format(time.time() - start))

print("")

print("Optimal Parameters: {}".format(bnb_grid_search.best_params_))
best_bnb = bnb_grid_search.best_estimator_
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)

best_bnb.fit(X_train, y_train)

y_hat_test = best_bnb.predict(X_test) 

bnb_confusion_matrix = confusion_matrix(y_test, y_hat_test)

print(bnb_confusion_matrix)

bnb_classification_report = classification_report(y_test, y_hat_test)

print(bnb_classification_report)
feat_df = pd.DataFrame()

feat_df['prob_0'] = np.exp(best_bnb.feature_log_prob_[0])

feat_df['prob_1'] = np.exp(best_bnb.feature_log_prob_[1])

feat_df.index = features.columns

feat_df.head()
feat_df['ave_prob'] = feat_df.apply(lambda x: (x[0] + x[1])/2, axis = 1)

feat_df['prob_diff'] = feat_df.apply(lambda x: np.abs(x[0] - x[1]), axis = 1)

feat_df.head()
feat_df.sort_values('prob_diff', ascending=False).head(10)