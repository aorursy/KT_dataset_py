import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
%matplotlib inline

# Ignore deprecation warnings from scikit-learn 0.20.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
train_raw = pd.read_csv('../input/train.csv',
                        sep=r'\s*,\s*',
                        engine='python')

test_raw = pd.read_csv('../input/test.csv',
                       sep=r'\s*,\s*',
                       engine='python')
train_raw.info()
train_raw.head()
train_raw.describe()
# Columns with missing data
cols_with_na = train_raw.columns[train_raw.isnull().any(axis=0)]
cols_with_na
# Frequency of missing values in each column
# High for v2a1, v18q1, rez_esc
train_raw[cols_with_na].isnull().sum(axis=0) / train_raw.shape[0]
# All column dtypes
print('Feature types:', *{train_raw[col].dtype.name for col in train_raw.columns})

print('Non-numeric features:', *train_raw.select_dtypes(exclude=[np.number]).columns)
train_raw.select_dtypes(exclude=[np.number]).head(10)
# Number of 0/1 columns (one-hot encoded)
sum(set(train_raw[col].unique()) == {0,1} for col in train_raw.columns)
# Frequency of each target value
train_raw['Target'].value_counts() / train_raw.shape[0]
def preprocess(data):
    data = data.copy()
    dep = data['dependency'].copy()
    dep[dep == 'no'] = 0
    dep[(dep != 0) & (~dep.isnull())] = 1
    data['dependency'] = pd.to_numeric(dep)
    
    for col in ['edjefe', 'edjefa']:
        edjef = data[col].copy()
        edjef[edjef == 'yes'] = 1
        edjef[edjef == 'no'] = 0
        data[col] = pd.to_numeric(edjef)

    return data
# After preprocessing, only the Id and idhogar features are not numeric
preprocess(train_raw).select_dtypes(exclude=[np.number]).columns
train = preprocess(train_raw)
test = preprocess(train_raw)
numeric_columns = list(train.select_dtypes(include=[np.number]).columns)
columns = list(set(numeric_columns) - {'v2a1', 'v18q1', 'rez_esc', 'Target'})
train_initial = train.copy()[columns + ['Target']]
train_initial.dropna(inplace=True)
x = train_initial[columns]
y = train_initial['Target']
# Use f1_macro scoring by default, since it's the one used in the competition

def cross_val(knn, x, y, cv, scoring='f1_macro'):
    scores = cross_val_score(knn, x, y, cv=cv, scoring=scoring)
    return sum(scores)/len(scores)
knn = KNeighborsClassifier(n_neighbors=30, p=2)
print('Accuracy:', cross_val(knn, x, y, cv=5, scoring='accuracy'))
print('F1:', cross_val(knn, x, y, cv=5))
# Add imputation (slightly better results)

x = Imputer().fit_transform(train[columns])
y = train['Target']

print('Accuracy:', cross_val(knn, x, y, cv=5, scoring='accuracy'))
print('F1:', cross_val(knn, x, y, cv=5))
corrs = {}
pvals = {}
print('Correlations and p-values between the features and the target.')
print(f'{"feature":<15}{"corr":>6}{"pval":>9}')

for col in columns:
    # Ignore the feature elimbasu5, since it always has value 0
    dropped = train[[col, 'Target']].dropna()
    corrs[col], pvals[col] = pearsonr(dropped[col], dropped['Target'])
    print(f'{col:15}{corrs[col]:6.2f}{pvals[col]:9.6f}')
print()
print('Min p-value:', min(pvals.values()))
print('Max abs(correlation):', max(abs(c) for c in corrs.values()));
# The above warning suggests that a division by zero.
# Since the correlation for elimbasu5 is nan, we now analyse that feature.
set(train['elimbasu5'])
# Remove elimbasu5 from the used columns
columns.remove('elimbasu5')
# Most of the p-values are tiny.
np.median(list(pvals.values()))
plt.axhline(color='black')
plt.plot(corrs.keys(), corrs.values())
plt.title('pearson correlation coefficients')
plt.show()

plt.plot(pvals.keys(), pvals.values())
plt.title('p-values')
plt.show()
# Columns with pval < .1 and abs(corr) > .15
filtered_columns = [col for col in columns
                    if pvals[col] < .1
                    and abs(corrs[col]) > .15]
filtered_columns
# New attempt, removing columns based on pval and corr.
# This improves the quality of the features used, avoids the curse of dimensionality and lowers the runtime.
# The F1 score and accuracy are higher.

x = Imputer().fit_transform(train[filtered_columns])
y = train['Target']

print('Accuracy:', cross_val(knn, x, y, cv=5, scoring='accuracy'))
print('F1:', cross_val(knn, x, y, cv=5))
# Now using p=1 (manhattan distance)
knn.set_params(p=1)
print('Accuracy:', cross_val(knn, x, y, cv=5, scoring='accuracy'))
print('F1:', cross_val(knn, x, y, cv=5))
# TransformerMixin provides the method fit_transform.
# BaseEstimator provides get_params, set_params.

class CorrelationSelector(sklearn.base.TransformerMixin, sklearn.base.BaseEstimator):
    """A transformer that removes columns based on pearson correlation and the frequency of each value.
    
    Calculates the pearson correlation and p-values between each feature and the target.
    Removes any column with abs(correlation) < min_corr or pvalue > max_pval.
    """
    def __init__(self, min_corr=0, max_pval=1):
        self.min_corr = min_corr
        self.max_pval = max_pval
    
    def fit(self, x, y):
        x, y = sklearn.utils.check_X_y(x, y, dtype='numeric', y_numeric=True)
        cols = []
        for i in range(x.shape[1]):
            # If x[:, i] has only one value, pearsonr will raise a warning.
            # Therefore, set cols[i] to False instead of calling pearsonr.
            if len(np.unique(x[:, i])) == 1:
                cols.append(False)
            else:
                corr, pval = pearsonr(x[:, i], y)
                cols.append(abs(corr) >= self.min_corr and pval <= self.max_pval)
            
        self.columns_ = cols
        return self
        
    def transform(self, x):
        sklearn.utils.validation.check_is_fitted(self, 'columns_')
        x = sklearn.utils.check_array(x, dtype='numeric')
        if x.shape[1] != len(self.columns_):
            raise ValueError('x has different shape than during fitting.')
        
        x = x[:, self.columns_]
        return x
from sklearn.utils.estimator_checks import check_estimator
check_estimator(CorrelationSelector)
p = Pipeline([
    ('imputer', Imputer()),
    ('corr', CorrelationSelector()),
    ('knn', KNeighborsClassifier())
])
p.get_params()
x = train[columns]
y = train['Target']

params = {
    'corr__max_pval': [.1],
    'corr__min_corr': [.2],
    'knn__n_neighbors': [10, 20, 30, 40],
    'knn__p': [1, 2],
}

# Initially, identify the best values of p (p=1 for euclidean distance, p=2 for manhattan distance)
gs = GridSearchCV(p, params, scoring='f1_macro', cv=3, return_train_score=True, refit=False)
gs.fit(x, y)
for i in range(len(gs.cv_results_['params'])):
    print('k={params[knn__n_neighbors]}  p={params[knn__p]}  score={score:.4f}'
          .format(params=gs.cv_results_['params'][i], score=gs.cv_results_['mean_test_score'][i]))
params = {
    'corr__max_pval': [1e-50, 1e-25, .1],
    'corr__min_corr': [.1, .15, .2, .25, .3], # max correlation in this dataset is .335 (as seen before)
    'knn__n_neighbors': [1, 3, 5, 7, 10, 12, 15, 25],
    'knn__p': [1],
}

# Now look for the best params
gs = GridSearchCV(p, params, scoring='f1_macro', cv=3, return_train_score=True, refit=False)
gs.fit(x, y)
# Create a dataframe in order to analyse the results

def gs_results_to_dataframe(gs):
    scores = pd.DataFrame(gs.cv_results_['params'],
                          columns=['knn__n_neighbors', 'corr__max_pval', 'corr__min_corr'])

    scores.rename(columns={'knn__n_neighbors': 'k',
                           'corr__max_pval': 'max_pval',
                           'corr__min_corr': 'min_corr'}, inplace=True)

    scores['score'] = gs.cv_results_['mean_test_score']
    scores.sort_values('score', ascending=False, inplace=True)
    return scores

scores = gs_results_to_dataframe(gs)
scores.head(20)
params = {
    'corr__max_pval': [.1],
    'corr__min_corr': [.14, .15, .16, .29, .30, .31],
    'knn__n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'knn__p': [1],
}

# Now look for the best params
gs = GridSearchCV(p, params, scoring='f1_macro', cv=10, return_train_score=True, refit=False)
gs.fit(x, y)
gs_results_to_dataframe(gs).head(20)
params = {
    'corr__max_pval': [.1],
    'corr__min_corr': [.14, .15, .16],
    'knn__n_neighbors': [3, 4, 5],
    'knn__p': [1],
}

# Now look for the best params
gs = GridSearchCV(p, params, scoring='f1_macro', cv=20, return_train_score=True, refit=False)
gs.fit(x, y)
gs_results_to_dataframe(gs).head(20)
# Features used above
print(columns)
base_features = ['r4t3', 'instlevel5', 'SQBage', 'hogar_nin', 'r4t1', 'sanitario6', 'energcocinar1', 'SQBescolari', 'abastaguafuera', 'estadocivil4', 'paredfibras', 'paredzocalo', 'eviv1', 'tipovivi1', 'pisonotiene', 'instlevel3', 'hogar_mayor', 'paredblolad', 'energcocinar2', 'estadocivil1', 'lugar5', 'elimbasu6', 'eviv2', 'parentesco8', 'r4h2', 'edjefa', 'SQBhogar_nin', 'epared3', 'abastaguano', 'qmobilephone', 'elimbasu2', 'paredother', 'dis', 'etecho3', 'cielorazo', 'elimbasu1', 'estadocivil7', 'parentesco6', 'techozinc', 'abastaguadentro', 'tamhog', 'v18q', 'pisoother', 'energcocinar4', 'r4t2', 'lugar3', 'tipovivi2', 'refrig', 'instlevel9', 'rooms', 'r4h3', 'area2', 'lugar4', 'estadocivil6', 'female', 'male', 'tipovivi4', 'area1', 'instlevel6', 'parentesco7', 'r4m1', 'parentesco10', 'SQBedjefe', 'computer', 'r4h1', 'techocane', 'estadocivil5', 'instlevel8', 'etecho1', 'parentesco1', 'parentesco4', 'tipovivi3', 'sanitario3', 'age', 'public', 'planpri', 'elimbasu3', 'tamviv', 'epared1', 'etecho2', 'lugar2', 'pisonatur', 'pisomadera', 'r4m2', 'television', 'lugar6', 'hogar_total', 'parentesco5', 'estadocivil3', 'parentesco2', 'hogar_adul', 'instlevel2', 'parentesco9', 'instlevel4', 'paredpreb', 'coopele', 'sanitario5', 'energcocinar3', 'r4m3', 'dependency', 'parentesco12', 'techoentrepiso', 'mobilephone', 'instlevel7', 'SQBdependency', 'estadocivil2', 'techootro', 'meaneduc', 'bedrooms', 'parentesco3', 'instlevel1', 'sanitario2', 'noelec', 'SQBovercrowding', 'eviv3', 'hacapo', 'sanitario1', 'tipovivi5', 'SQBhogar_total', 'pisocemento', 'epared2', 'paredmad', 'hacdor', 'paredzinc', 'elimbasu4', 'overcrowding', 'pareddes', 'hhsize', 'edjefe', 'parentesco11', 'pisomoscer', 'escolari', 'SQBmeaned', 'v14a', 'agesq', 'lugar1']

def make_submission(k, p, min_corr, max_pval, out):
    imp = Imputer()
    train_processed = preprocess(train_raw)
    test_processed = preprocess(test_raw)
    
    xtrain = train_processed[base_features]
    ytrain = train_processed['Target']
    xtest = test_processed[base_features]
    
    pipeline = Pipeline([
        ('imputer', Imputer()),
        ('corr', CorrelationSelector(min_corr=min_corr, max_pval=max_pval)),
        ('knn', KNeighborsClassifier(n_neighbors=k, p=p))
    ])
    
    scores_f1 = cross_val_score(pipeline, xtrain, ytrain, scoring='f1_macro', cv=20)
    score_f1 = sum(scores_f1) / len(scores_f1)
    scores_acc = cross_val_score(pipeline, xtrain, ytrain, scoring='accuracy', cv=20)
    score_acc = sum(scores_acc) / len(scores_acc)
    
    pipeline.fit(xtrain, ytrain)
    features = list(xtrain.columns[pipeline.get_params()['corr'].columns_])
    ytest = pipeline.predict(xtest)
    df = pd.DataFrame({'Id': test_processed['Id'], 'Target': ytest})
    df.to_csv(out, index=False)
    print(f'{out}: k={k}, p={p}, min_corr={min_corr}, max_pval={max_pval}, f1={score_f1:.6f}, acc={score_acc:.6f}')
    print(f'    features={features}')
    print()
make_submission(k=4, p=1, min_corr=0.16, max_pval=0.1, out='sub1.csv')
make_submission(k=4, p=1, min_corr=0.14, max_pval=0.1, out='sub2.csv')
make_submission(k=4, p=1, min_corr=0.15, max_pval=0.1, out='sub3.csv')
make_submission(k=3, p=1, min_corr=0.15, max_pval=0.1, out='sub4.csv')
make_submission(k=5, p=1, min_corr=0.15, max_pval=0.1, out='sub5.csv')