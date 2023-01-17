import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_val_score
def extract_categories(df, col):
    categories = pd.get_dummies(df[col], prefix=col)
    return pd.concat([df, categories], axis=1).drop(col, axis=1)
def extract_categories_from_str(df, col):
    categories = df[col].str.get_dummies(', ')
    categories = categories.T.rename(lambda c: col + ':' + c).T
    return pd.concat([df, categories], axis=1).drop(col, axis=1)
def join_essays(df):
    return (df['project_essay_1'].fillna('') 
        + '\n' + df['project_essay_2'].fillna('') 
        + '\n' + df['project_essay_3'].fillna('') 
        + '\n' + df['project_essay_4'].fillna('')).str.strip()
def count_essays(df):
    return (df['project_essay_1'].fillna(0).astype(bool).astype(int)
           + df['project_essay_2'].fillna(0).astype(bool).astype(int)
           + df['project_essay_3'].fillna(0).astype(bool).astype(int)
           + df['project_essay_4'].fillna(0).astype(bool).astype(int))
def extract_words(df, col):
    cv = CountVectorizer(
        stop_words='english', 
        binary=True, 
        strip_accents='ascii', 
        max_df=0.99, 
        min_df=0.01
    )
    words = cv.fit_transform(df[col])
    words = pd.DataFrame(words.toarray(), columns=cv.vocabulary_)
    words = words.T.rename(lambda c: col + ':' + c).T
    return pd.concat([df, words], axis=1).drop(col, axis=1)
def extract_dt(df, col):
    dt = pd.to_datetime(df[col])
    dt_df = pd.DataFrame()
    dt_df[col + ':y'] = dt.dt.year
    dt_df[col + ':m'] = dt.dt.month
    dt_df[col + ':w'] = dt.dt.week
    dt_df[col + ':wd'] = dt.dt.dayofweek
    dt_df[col + ':d'] = dt.dt.day
    dt_df[col + ':h'] = dt.dt.hour
    return pd.concat([df, dt_df], axis=1).drop(col, axis=1)
data = pd.concat([
    pd.read_csv('../input/train.csv'),
    pd.read_csv('../input/test.csv')
]).reset_index(drop=True)

# resources = pd.read_csv('../input/resources.csv')
# resource_prices = resources.groupby('id')['price'].sum().reset_index()
# data = data.merge(resource_prices, on='id')

data = extract_categories(data, 'project_grade_category')
data = extract_categories(data, 'school_state')
data = extract_categories(data, 'teacher_prefix')

data = extract_categories_from_str(data, 'project_subject_categories')
data = extract_categories_from_str(data, 'project_subject_subcategories')

data['project_essay'] = join_essays(data)
# data['essay_count'] = count_essays(data)
data = data.drop(['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4'], axis=1)

data = extract_words(data, 'project_title')
data = extract_words(data, 'project_essay')
data = extract_words(data, 'project_resource_summary')

# data = extract_dt(data, 'project_submitted_datetime')

data = data.drop([
    'teacher_id', 
    'teacher_number_of_previously_posted_projects', 
    'project_submitted_datetime'
], axis=1)

data.head()
X = data.drop(['id', 'project_is_approved'], axis=1)
y = data['project_is_approved']
variances = (X.sample(frac=0.1).var().sort_values() * 100).astype(int)
features = variances[variances > 1].keys().tolist()
X = X[features]
corr = X.sample(frac=0.1).corr()
corr = corr.unstack().sort_values()

correlated_features = corr[(corr < 1) & (abs(corr) > 0.5)][::2].keys().values
new_feature_dict = {}
for a, b in correlated_features:
    if (a not in new_feature_dict) and (b not in new_feature_dict):
        new_feature = a + ' | ' + b
        X[new_feature] = (X[a].astype(bool) | X[b].astype(bool)).astype(int)
        new_feature_dict[a] = new_feature
        new_feature_dict[b] = new_feature
    else:
        if b in new_feature_dict:
            existing_new_feature = new_feature_dict[b]
            new_feature = existing_new_feature + ' | ' + a
            X[new_feature] = (X[a].astype(bool) | X[existing_new_feature].astype(bool)).astype(int)
            new_feature_dict[a] = new_feature
        elif a in new_feature_dict:
            existing_new_feature = new_feature_dict[a]
            new_feature = existing_new_feature + ' | ' + b
            X[new_feature] = (X[b].astype(bool) | X[existing_new_feature].astype(bool)).astype(int)
            new_feature_dict[b] = new_feature

correlated_features = list(set(np.array(list(map(list, correlated_features))).flatten()))
X = X.drop(correlated_features, axis=1)
X_train = X[~y.isnull()]
y_train = y[~y.isnull()]
X_test = X[y.isnull()]
X_test_ids = data[y.isnull()]['id'].values
print('CV Score:', cross_val_score(BernoulliNB(fit_prior=False), X_train, y_train, scoring=make_scorer(roc_auc_score)).mean() * 10000 // 1 / 100, '%')
m = BernoulliNB(fit_prior=False).fit(X_train, y_train)
y_pred = m.predict_proba(X_test)[:,1]
result = pd.DataFrame({
    'id': X_test_ids,
    'project_is_approved': y_pred
})
result.to_csv('result.csv', index=False)