import pandas as pd
import numpy as np
import gc
import os
import string
import seaborn as sns
from matplotlib import pyplot as plt
import lightgbm as lgb
import warnings
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy import stats
from nltk.stem.snowball import EnglishStemmer
import statsmodels.api as sm
%matplotlib inline
pd.options.display.float_format = '{:.2f}'.format
sns.set()
warnings.filterwarnings('ignore')
def get_plot_instance(x=10, y=8, nrows=1, ncols=1):
    f, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(x, y))
    return f, ax
df = pd.read_csv('../input/ks-projects-201801.csv', parse_dates=['deadline', 'launched'])
df.head()
print('There are {} rows and {} columns'.format(*df.shape))
df.select_dtypes(include='object').describe()
df.select_dtypes(exclude='object').describe()
print('There are {} unique IDs and {} rows'.format(df['ID'].nunique(), df.shape[0]))
df['name'].value_counts().head(20)
df.info()
f, ax = get_plot_instance()
sns.set_context("notebook", font_scale=2)
ax.set_ylabel('% of projects')
print(df['state'].value_counts(normalize=True))
df['state'].value_counts(normalize=True).plot(kind='bar', ax=ax)
f.show()
target_map = {'failed': 'failed',
             'successful': 'successful',
             'canceled': 'failed'}
mask = np.logical_or(np.logical_or(df['state'] == 'successful', df['state'] == 'failed'), df['state'] == 'canceled')
df = df.loc[mask, :]
df['state'] = df['state'].map(target_map)
f, ax = get_plot_instance()
print(df['state'].value_counts(normalize=True))
df['state'].value_counts(normalize=True).plot(kind='bar', ax=ax)
f.show()
f, ax = get_plot_instance(12, 8)
sns.set_context("notebook", font_scale=2)
sns.set_palette("Set3")
df['main_category'].value_counts(normalize=True).plot(kind='bar', ax=ax)
f.show()
f, ax = get_plot_instance(12, 8)
sns.set_context("notebook", font_scale=2)
sns.set_palette("Set3")
df[df['main_category']=='Film & Video']['category'].value_counts(normalize=True).plot(kind='bar', ax=ax)
f.show()
f, ax = get_plot_instance(12, 8)
sns.set_context("notebook", font_scale=2)
sns.set_palette("Set3")
df[df['main_category']=='Music']['category'].value_counts(normalize=True).plot(kind='bar', ax=ax)
f.show()
f, ax = get_plot_instance(12, 8)
sns.set_context("notebook", font_scale=2)
sns.set_palette("Set3")
df[df['main_category']=='Publishing']['category'].value_counts(normalize=True).plot(kind='bar', ax=ax)
f.show()
def plot_normalized_bars(in_df, in_groups, label, x_c, y_c):
    main_categories_frac = in_df.groupby(in_groups).size().reset_index(name='count')
    main_categories_frac['count_sum'] = main_categories_frac.groupby(in_groups[0])['count'].transform('sum')
    main_categories_frac['percent_frac'] = main_categories_frac['count']/main_categories_frac['count_sum'] * 100
    f, ax = get_plot_instance(x_c, y_c)
    sns.barplot(data=main_categories_frac, x=in_groups[0], y='percent_frac', hue='state', ax=ax)
    ax.set_ylabel(label)
    f.show()
sns.set_context("notebook", font_scale=1.1)
sns.set_palette("Paired")

plot_normalized_bars(df, ['main_category', 'state'], '% of failed/successfull per category', 17, 8)
sns.set_context("notebook", font_scale=1.5)
sns.set_palette("Paired")

plot_normalized_bars(df, ['currency', 'state'], '% of failed/successfull per category', 14, 8)
sns.set_context("notebook", font_scale=1.5)
sns.set_palette("Paired")

plot_normalized_bars(df, ['country', 'state'], '% of failed/successfull per category', 14, 8)
print(df.groupby('state')['goal'].median().reset_index(name='goal median'))
print(df.groupby('state')['goal'].mean().reset_index(name='goal mean'))
def bootstrap_ci(x, n_iter, sample_size, cl):
    means = []
    for i in range(n_iter):
        sample = x.sample(n=sample_size, replace=True)
        means.append(sample.mean())
    means = np.sort(np.array(means))
    left_limit = int(means.size * cl/2)
    right_limit = int(means.size * (1-cl/2))
    return means.mean(), means[left_limit], means[right_limit]

failed_mean, failed_left, failed_right = bootstrap_ci(df[df['state']=='failed']['goal'], 1000, 50000, 0.05)
print('95% confidence interval for failed mean: {:.2f} <= {:.2f} <= {:.2f}'.format(failed_left, failed_mean, failed_right))

suc_mean, suc_left, suc_right = bootstrap_ci(df[df['state']=='successful']['goal'], 1000, 50000, 0.05)
print('95% confidence interval for successful mean: {:.2f} <= {:.2f} <= {:.2f}'.format(suc_left, suc_mean, suc_right))
df['log_goal'] = df['goal'].apply(lambda x: np.log1p(x))
f, ax = get_plot_instance(16, 8)
sns.boxplot(data=df, x='state', y='log_goal', ax=ax)
ax.set_ylabel(r'$\log(goal + 1)$')
f.show()
sns.pairplot(data=df[df.state == 'successful'][['goal', 'pledged']].apply(lambda x: np.log1p(x)),
             vars=['goal', 'pledged'], size=6)
sns.pairplot(data=df[df.state == 'failed'][['goal', 'pledged']].apply(lambda x: np.log1p(x)),
             vars=['goal', 'pledged'], size=6)
f, ax = get_plot_instance(16, 8)
sns.set_context("notebook", font_scale=1.1)
sns.boxplot(data=df, x='main_category', y='log_goal', ax=ax)
ax.set_ylabel(r'$\log(goal + 1)$')
f.show()
df['goal_pledged_diff'] = df['goal'] - df['pledged']
sns.set_context("notebook", font_scale=1.5)
f, ax = get_plot_instance(8, 10)
sns.boxplot(data=df, x='state', y='goal_pledged_diff')
ax.set_ylabel('Difference between goals and pledges')
f.show()
df['goal_pledged_ratio'] = np.log1p(df['pledged']/df['goal'])
f, ax = get_plot_instance(12, 8)
sns.set_context("notebook", font_scale=1.5)
sns.distplot(df[df['state']=='failed']['goal_pledged_ratio'], ax=ax)
sns.distplot(df[df['state']=='successful']['goal_pledged_ratio'], ax=ax)
ax.set_xlabel('Ratio of pledges over goals')
f.show()
f, ax = get_plot_instance(12, 8)
sns.set_context("notebook", font_scale=1.5)
sns.violinplot(data=df, x='state', y='goal_pledged_ratio', ax=ax)
ax.set_ylabel('Ratio of pledges over goals')
f.show()
backers_by_cat = df.groupby(['main_category', 'state'])['backers'].agg({'backers_sum_state': np.sum}).reset_index()
backers_by_cat['backers_category'] = backers_by_cat.groupby('main_category')['backers_sum_state'].transform('sum')
backers_by_cat['backers_state_frac'] = backers_by_cat['backers_sum_state']/backers_by_cat['backers_category'] * 100
f, ax = get_plot_instance(18, 8)
sns.set_context("notebook", font_scale=1.1)
sns.barplot(data=backers_by_cat, x='main_category', y='backers_state_frac', hue='state', ax=ax)
f.show()
f, ax = get_plot_instance()
sns.set_context("notebook", font_scale=2)
sns.distplot(np.log1p(df[df['state']=='failed']['backers']), ax=ax)
sns.distplot(np.log1p(df[df['state']=='successful']['backers']), ax=ax)
f.show()
f, ax = get_plot_instance()
sns.set_context("notebook", font_scale=2)
df['log_backers'] = np.log1p(df['backers'])
sns.violinplot(data=df[['state', 'log_backers']], x='state', y='log_backers', ax=ax)
ax.set_ylabel(r'$\log$(backers + 1)')
f.show()
backers_sum_over_category = df.groupby('main_category')['backers'].agg({'sum': np.sum}).reset_index()
backers_sum_over_category['sum_frac'] = backers_sum_over_category['sum']/backers_sum_over_category['sum'].sum() * 100
f, ax = get_plot_instance(16, 8)
sns.set_context("notebook", font_scale=1.0)
sns.barplot(data=backers_sum_over_category, x='main_category', y='sum_frac', ax=ax)
ax.set_ylabel('% of backers in category')
f.show()
f, ax = get_plot_instance(16, 8)
sns.set_context("notebook", font_scale=1.0)
sns.violinplot(data=df, x='main_category', y='log_backers')
ax.set_ylabel(r'$\log$(backers + 1)')
f.show()
df['launched_year'] = df['launched'].dt.year
df['deadline_year'] = df['deadline'].dt.year
df['months_diff'] = (df['deadline'] - df['launched'])/ np.timedelta64(1, 'M')
df['years_diff'] = (df['deadline'] - df['launched'])/ np.timedelta64(1, 'Y')
f, ax = get_plot_instance(16, 8)
sns.set_context("notebook", font_scale=1.0)
sns.violinplot(data=df[df['launched'].dt.year > 1970], x='main_category', y='years_diff')
f.show()
f, ax = get_plot_instance(16, 8)
sns.set_context("notebook", font_scale=1.0)
sns.violinplot(data=df[df['launched'].dt.year > 1970], x='main_category', y='months_diff')
f.show()
project_count_over_years = df.groupby('launched_year').size().reset_index(name='number_of_projects')
f, ax = get_plot_instance(16, 8)
ax.plot(project_count_over_years['launched_year'], project_count_over_years['number_of_projects'])
f.show()
backers_sum_over_years = df.groupby('launched_year')['backers'].agg({'backers_overall': np.sum}).reset_index()
f, ax = get_plot_instance(16, 8)
ax.plot(backers_sum_over_years['launched_year'], backers_sum_over_years['backers_overall'])
f.show()
goals_sum_over_years = df.groupby('launched_year')['goal'].agg({'goals_overall': np.sum}).reset_index()
pledged_sum_over_years = df.groupby('launched_year')['pledged'].agg({'pledged_overall': np.sum}).reset_index()
f, ax = get_plot_instance(16, 8)
ax.plot(goals_sum_over_years['launched_year'], goals_sum_over_years['goals_overall'], color='b')
ax.plot(pledged_sum_over_years['launched_year'], pledged_sum_over_years['pledged_overall'], color='g')
f.show()
goals_j_sum_over_years = df[df['main_category']=='Journalism'].groupby('launched_year')['goal'].agg({'goals_overall': np.sum}).reset_index()
pledged_j_sum_over_years = df[df['main_category']=='Journalism'].groupby('launched_year')['pledged'].agg({'pledged_overall': np.sum}).reset_index()
f, ax = get_plot_instance(16, 8)
ax.plot(goals_j_sum_over_years['launched_year'], goals_j_sum_over_years['goals_overall'], color='b')
ax.plot(pledged_j_sum_over_years['launched_year'], pledged_j_sum_over_years['pledged_overall'], color='g')
f.show()
print(np.corrcoef(df['goal'].values, df['usd_goal_real'].values))
# Delete DataFrame and reload it
del df
gc.collect()

df = pd.read_csv('../input/ks-projects-201801.csv', parse_dates=['deadline', 'launched'])
target_map = {'failed': 'failed',
             'successful': 'successful',
             'canceled': 'failed'}
mask = np.logical_or(np.logical_or(df['state'] == 'successful', df['state'] == 'failed'), df['state'] == 'canceled')
df = df.loc[mask, :]
df['state'] = df['state'].map(target_map)

# Map new target
# Drop some raw variables like ID, name, time variables 
# and variables that come from the future like backers, pledged, etc
df_model = df.drop(['ID', 'name', 'deadline', 'launched', 'backers', 'usd pledged', 'usd_pledged_real', 'pledged'],
                    axis='columns')
target = df['state'].copy()
df_model.drop('state', axis='columns', inplace=True)
target = target.map({'failed': 0, 'successful': 1})

# Scale real value variables, encode categorical variables
df_model = pd.get_dummies(df_model)
df_model['goal'] = df_model['goal'].transform(lambda x: (x-x.mean())/x.std())
df_model['usd_goal_real'] = df_model['usd_goal_real'].transform(lambda x: (x-x.mean())/x.std())

# Model
# Sample randomly because there is no underlying business problem
X_train, X_valid, y_train, y_valid = train_test_split(df_model, target, test_size=0.3, random_state=42)
clf = LogisticRegression(n_jobs=-1, C=2, penalty='l2', random_state=42)
clf.fit(X_train, y_train)
print('Area under roc curve train: {}'.format(roc_auc_score(y_train, clf.predict(X_train))))
print('Area under roc curve valid: {}'.format(roc_auc_score(y_valid, clf.predict(X_valid))))
lreg_coeffs = pd.DataFrame(list(zip(df_model.columns, clf.coef_[0].tolist()))).sort_values(by=1)
pd.options.display.max_columns = 100
print(pd.concat([lreg_coeffs.head(10).rename(columns={0: 'feature', 1: 'neg_coef'}).reset_index().drop('index',axis=1),
                 lreg_coeffs.tail(10).rename(columns={0: 'feature', 1: 'pos_coef'}).reset_index()], axis=1).drop('index',axis=1).sort_values(by='pos_coef', ascending=False))
# Reload data
del df
gc.collect()

df = pd.read_csv('../input/ks-projects-201801.csv', parse_dates=['deadline', 'launched'])
target_map = {'failed': 'failed',
             'successful': 'successful',
             'canceled': 'failed'}
mask = np.logical_or(np.logical_or(df['state'] == 'successful', df['state'] == 'failed'), df['state'] == 'canceled')
df = df.loc[mask, :]
df['state'] = df['state'].map(target_map)

# Prepare text
df['name'] = df['name'].astype('str').apply(lambda x: nltk.word_tokenize(x))

stop_words = nltk.corpus.stopwords.words('english')
table = str.maketrans('', '', string.punctuation)
stemmer = EnglishStemmer()
df['name'] = df['name']. \
apply(lambda x: ' '.join([stemmer.stem(word.lower().translate(table)) \
                 for word in x if word not in string.punctuation and word not in stop_words]))

bow_vectorizer = CountVectorizer(ngram_range=(1, 1))
sparse_embeddings = bow_vectorizer.fit_transform(df['name'])
class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool=True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

def get_oof(clf, x_train, y):
    kf = KFold(n_splits=10, random_state=42)
    oof_train = np.zeros((x_train.shape[0],))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)

    return oof_train.reshape(-1, 1)
lr = SklearnWrapper(clf=LogisticRegression, seed=42, params={'C': 0.08, 'penalty': 'l2'})
lr_oof_train = get_oof(
    lr, sparse_embeddings, target.values)

rocauc = roc_auc_score(target.values, lr_oof_train)
print('Ridge OOF ROCAUC: {}'.format(rocauc))

df['lr_preds'] = lr_oof_train
#  Extract month, weekday, year and quarter
df['launched_month'] = df['launched'].dt.month
df['deadline_month'] = df['deadline'].dt.month
df['launched_weekday'] = df['launched'].dt.weekday
df['deadline_weekday'] = df['deadline'].dt.weekday
df['launched_year'] = df['launched'].dt.year
df['deadline_year'] = df['deadline'].dt.year
df['launched_quarter'] = df['launched'].dt.quarter
df['deadline_quarter'] = df['deadline'].dt.quarter

# Extract business days
def apply_busday_count(row):
    return np.busday_count(row['launched'], row['deadline'])
df['business_days'] = df[['launched', 'deadline']].apply(apply_busday_count, axis=1)

# Extract time deltas
df['months_diff'] = (df['deadline'] - df['launched'])/ np.timedelta64(1, 'M')
df['years_diff'] = (df['deadline'] - df['launched'])/ np.timedelta64(1, 'Y')
df['days_diff'] = (df['deadline'] - df['launched'])/ np.timedelta64(1, 'D')

# Recode categories with additional information
groupby = ['main_category', ['main_category', 'category']]
transformations = ['mean', 'min', 'max', 'median']

for idx, group in enumerate(groupby):
    for stat in transformations:
        if idx == 0:
            df['main_cat_goal' + '_' + stat] = df.groupby(group)['goal'].transform(stat)
        if idx == 1:
            df['cat_cat_goal' + '_' + stat] = df.groupby(group)['goal'].transform(stat)

# Deviations, booleans and ratios
df['main_cat_goal_mean_dev'] = df['goal'] - df['main_cat_goal_mean']
df['main_cat_goal_median_dev'] = df['goal'] - df['main_cat_goal_median']
df['main_cat_goal_mean_ratio'] = df['goal']/df['main_cat_goal_mean']
df['main_cat_goal_median_ratio'] = df['goal']/df['main_cat_goal_median']
df['goal_gt_mean'] = df['goal'] > df['main_cat_goal_mean']
df['goals_diff'] = df['goal'] - df['usd_goal_real']
df['goals_ratio'] = df['goal']/df['usd_goal_real'] * 100
df['main_cat_goal_sum'] = df.groupby('main_category')['goal'].transform('sum')
df['main_cat_goal_sum_year'] = df.groupby(['main_category', 'launched_year'])['goal'].transform('sum')
df['goal_to_goal_sum_prc'] = df['goal']/df['main_cat_goal_sum'] * 100
df['goal_to_goal_sum_prc_year'] = df['goal']/df['main_cat_goal_sum_year'] * 100

df['cat_cat_goal_mean_dev'] = df['goal'] - df['main_cat_goal_mean']
df['cat_cat_goal_median_dev'] = df['goal'] - df['main_cat_goal_median']
df['cat_cat_goal_mean_ratio'] = df['goal']/df['main_cat_goal_mean'] * 100
df['cat_cat_goal_median_ratio'] = df['goal']/df['main_cat_goal_median'] * 100
df_model = df.drop(['ID', 'name', 'deadline', 'launched', 'backers', 'usd pledged', 'usd_pledged_real', 'pledged'], axis='columns')
target = df_model['state'].copy()
df_model.drop('state', axis='columns', inplace=True)
target = target.map({'failed': 0, 'successful': 1})
cat_cols = df_model.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
# Sample randomly because there is no underlying business problem
X_train, X_valid, y_train, y_valid = train_test_split(df_model, target, test_size=0.3, random_state=42)
lgbm_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 100,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'learning_rate': 0.01,
    'verbose': 0,
    'num_threads': 10,
    'lambda_l2': 4
}

validation_curve = {}

lgtrain = lgb.Dataset(X_train, y_train,
                      categorical_feature=cat_cols)
lgvalid = lgb.Dataset(X_valid, y_valid,
                      categorical_feature=cat_cols)
lgb_clf = lgb.train(
    lgbm_params,
    lgtrain,
    num_boost_round=4000,
    valid_sets=[lgtrain, lgvalid],
    valid_names=['train', 'valid'],
    early_stopping_rounds=5,
    verbose_eval=200,
    evals_result=validation_curve
)
f, ax = get_plot_instance()
lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
f.show()
train_pred = lgb_clf.predict(data=X_train, num_iteration=lgb_clf.best_iteration)
valid_pred = lgb_clf.predict(data=X_valid, num_iteration=lgb_clf.best_iteration)
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, train_pred)
fpr_test, tpr_test, thresholds_test = roc_curve(y_valid, valid_pred)
f, ax = get_plot_instance()
sns.set_context("notebook", font_scale=1.5)
no_information_x_ref = [0, 1]
no_information_y_ref = [0, 1]
ax.plot(fpr_train, tpr_train, label='Train curve')
ax.plot(fpr_test, tpr_test, label='Test curve')
ax.plot(no_information_x_ref, no_information_y_ref, label='No information reference')
ax.legend(loc='best')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_xlim(left=0.0, right=1.0)
ax.set_ylim(bottom=0.0, top=1.0)
f.show()
ks_stat, pvalue = stats.ks_2samp(valid_pred, train_pred)
print('Kolmogorov-Smirnov goodness-of-fit statistic: {:.5f}, p-value: {:.3f}'.format(ks_stat, pvalue))
# Empirical cdfs based on model scores
ecdf_train = sm.distributions.ECDF(train_pred)
ecdf_test = sm.distributions.ECDF(valid_pred )

train_x = np.linspace(np.min(train_pred), np.max(train_pred), 1000)
valid_x = np.linspace(np.min(valid_pred), np.max(valid_pred), 1000)
f, ax = get_plot_instance()
sns.set_context("notebook", font_scale=1.5)
ax.plot(train_x, ecdf_train(train_x), label='Train curve')
ax.plot(train_x, ecdf_test(valid_x), label='Test curve')
ax.legend(loc='best')
ax.set_xlabel('Score')
ax.set_ylabel('Probability')
f.show()
f, ax = get_plot_instance()
sns.set_context("notebook", font_scale=1.5)
ax.plot(range(1, len(validation_curve['train']['auc'])+1), validation_curve['train']['auc'], label='Train curve')
ax.plot(range(1, len(validation_curve['train']['auc'])+1), validation_curve['valid']['auc'], label='Valid curve')
best_iter_reference_x = [lgb_clf.best_iteration, lgb_clf.best_iteration]
best_iter_reference_y = [0, 1]
ax.plot(best_iter_reference_x, best_iter_reference_y, label='Early stopping iteration')
ax.set_ylim(bottom=0.735, top=0.81)
ax.legend(loc='best')
ax.set_xlabel('iteration')
ax.set_ylabel('ROC AUC')
f.show()
def cumulative_gain_curve(y_true, y_score, pos_label=None):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]
    gains = np.cumsum(y_true)

    percentages = np.arange(start=1, stop=len(y_true) + 1)

    gains = gains / float(np.sum(y_true))
    percentages = percentages / float(len(y_true))

    gains = np.insert(gains, 0, [0])
    percentages = np.insert(percentages, 0, [0])

    return percentages, gains

def plot_cumulative_gain(y_true, y_probas, title='Cumulative Gains Curve',
                         ax=None, figsize=None, title_fontsize="large",
                         text_fontsize="medium"):
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    classes = np.unique(y_true)
    if len(classes) != 2:
        raise ValueError('Cannot calculate Cumulative Gains for data with '
                         '{} category/ies'.format(len(classes)))

    # Compute Cumulative Gain Curves
    percentages, gains1 = cumulative_gain_curve(y_true, 1-y_probas,
                                                classes[0])
    percentages, gains2 = cumulative_gain_curve(y_true, y_probas,
                                                classes[1])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    ax.plot(percentages, gains1, lw=3, label='Class {}'.format(classes[0]))
    ax.plot(percentages, gains2, lw=3, label='Class {}'.format(classes[1]))

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Baseline')

    ax.set_xlabel('Percentage of sample', fontsize=text_fontsize)
    ax.set_ylabel('Gain', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.grid('on')
    ax.legend(loc='lower right', fontsize=text_fontsize)

    return ax
f, ax = get_plot_instance()
plot_cumulative_gain(y_train, train_pred, ax=ax)
f.show()