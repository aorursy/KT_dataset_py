import warnings
warnings.filterwarnings('ignore')

import string
from tqdm import tqdm
from collections import Counter
from itertools import tee

from IPython.display import display

import numpy as np
from scipy import stats
from scipy.sparse import hstack as sparse_hstack
import pandas as pd

import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 100
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer

import eli5
# statistic methods
def tconfint(sample, alpha=0.05):
    '''Confidence interval based on Student t distribution.'''
    mean = np.mean(sample)
    S = np.std(sample, ddof=1)
    n = len(sample)

    t = stats.t.ppf(1 - alpha / 2, n - 1)
    left_boundary = mean - t * S / np.sqrt(n)
    right_boundary = mean + t * S / np.sqrt(n)

    return left_boundary, right_boundary


def tconfint_diff(sample1, sample2, alpha=0.05):
    '''Confidence interval based on Student t distribution for
    the difference in means of two samples.'''
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    s1 = np.std(sample1, ddof=1)
    s2 = np.std(sample2, ddof=1)
    n1 = len(sample1)
    n2 = len(sample2)

    sem1 = np.var(sample1) / (n1 - 1)
    sem2 = np.var(sample2) / (n2 - 1)
    semsum = sem1 + sem2
    z1 = (sem1 / semsum) ** 2 / (n1 - 1)
    z2 = (sem2 / semsum) ** 2 / (n2 - 1)
    dof = 1 / (z1 + z2)

    t = stats.t.ppf(1 - alpha / 2, dof)
    left_boundary = (mean1 - mean2) - t * np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    right_boundary = (mean1 - mean2) + t * np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)

    return left_boundary, right_boundary


def bootstrap_statint(sample, stat=np.mean, n_samples=5000, alpha=0.05):
    '''Statistical interval for a `stat` of a `sample` calculation
    using bootstrap sampling mechanism. `stat` is a numpy function
    like np.mean, np.std, np.median, np.max, np.min, etc.'''
    indices = np.random.randint(0, len(sample), (n_samples, len(sample)))
    samples = sample[indices]

    stat_scores = stat(samples, axis=1)
    boundaries = np.percentile(stat_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return boundaries


def bootstrap_statint_diff(sample1, sample2, stat=np.mean, n_samples=5000, alpha=0.05):
    '''Statistical interval for a difference in `stat` of two samples
    calculation using bootstrap sampling mechanism. `stat` is a numpy
    function like np.mean, np.std, np.median, np.max, np.min, etc.'''
    indices1 = np.random.randint(0, len(sample1), (n_samples, len(sample1)))
    indices2 = np.random.randint(0, len(sample2), (n_samples, len(sample2)))
    samples1 = sample1[indices1]
    samples2 = sample2[indices2]

    stat_scores1 = stat(samples1, axis=1)
    stat_scores2 = stat(samples2, axis=1)
    stat_scores_diff = stat_scores1 - stat_scores2
    boundaries = np.percentile(stat_scores_diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return boundaries


def proportion_confint(sample, alpha=0.05):
    '''Wilson\'s сonfidence interval for a proportion.'''
    p = np.mean(sample)
    n = len(sample)

    z = stats.norm.ppf(1 - alpha / 2)
    left_boundary = 1 / (1 + z ** 2 / n) * (p + z ** 2 / (2 * n) \
                                            - z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)))
    right_boundary = 1 / (1 + z ** 2 / n) * (p + z ** 2 / (2 * n) \
                                             + z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)))

    return left_boundary, right_boundary


def proportions_diff_confint_ind(sample1, sample2, alpha=0.05):
    '''Confidence interval for the difference of two independent proportions.'''
    z = stats.norm.ppf(1 - alpha / 2)
    p1 = np.mean(sample1)
    p2 = np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    return left_boundary, right_boundary


def permutation_test_ind(sample1, sample2, max_permutations=None, alternative='two-sided'):
    '''Permutation test for two independent samples.'''
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError('Alternative not recognized, should be \'two-sided\', \'less\' or \'greater\'.')

    t_stat = np.mean(sample1) - np.mean(sample2)

    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n = len(joined_sample)

    if max_permutations:
        index = list(range(n))
        indices = set([tuple(index)])
        for _ in range(max_permutations - 1):
            np.random.shuffle(index)
            indices.add(tuple(index))

        indices = [(index[:n1], index[n1:]) for index in indices]
    else:
        indices = [(list(index), list(filter(lambda i: i not in index, range(n)))) \
                    for index in itertools.combinations(range(n), n1)]

    zero_distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean() \
                  for i in indices]

    if alternative == 'two-sided':
        p_value = sum([abs(x) >= abs(t_stat) for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        p_value = sum([x <= t_stat for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        p_value = sum([x >= t_stat for x in zero_distr]) / len(zero_distr)

    return t_stat, p_value


def proportions_ztest_ind(sample1, sample2, alternative='two-sided'):
    '''Z-test for two independent proportions.'''
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError('Alternative not recognized, should be \'two-sided\', \'less\' or \'greater\'.')

    p1 = np.mean(sample1)
    p2 = np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)

    P = (p1 * n1 + p2 * n2) / (n1 + n2)
    z_stat = (p1 - p2) / np.sqrt(P * (1 - P) * (1 / n1 + 1 / n2))

    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        p_value = stats.norm.cdf(z_stat)

    if alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)

    return z_stat, p_value


def cramers_v(contingency_table):
    '''Cramer\'s V coefficient.'''
    n = np.sum(contingency_table)
    ct_nrows, ct_ncols = contingency_table.shape
    if n < 40 or np.sum(contingency_table < 5) / (ct_nrows * ct_ncols) > 0.2:
        raise ValueError('Contingency table isn\'t suitable for Cramers\'s V coefficient calculation.')

    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    corr = np.sqrt(chi2 / (n * (min(ct_nrows, ct_ncols) - 1)))
    return corr, p_value
data = pd.read_csv('../input/real-or-fake-fake-jobposting-prediction/fake_job_postings.csv')
data.head(10)
data.info()
bin_features = ['telecommuting', 'has_company_logo', 'has_questions']
cat_features = ['department', 'employment_type', 'required_experience', 
                'required_education', 'industry', 'function']

text_features = ['title', 'company_profile', 'description', 'requirements', 'benefits']
complex_features = ['location', 'salary_range']
data.drop('job_id', axis=1, inplace=True)
data[text_features].head()
for feature_name in text_features[1:]:
    unspec_feature_name = f'{feature_name}_specified'
    data[unspec_feature_name] = (~data[feature_name].isna()).astype('int')
    bin_features += [unspec_feature_name]
data.head()[text_features + bin_features[-4:]]
for feature_name in text_features[1:]:
    data[feature_name].fillna('', inplace=True)
# nltk.download('stopwords')
# nltk.download('punkt')
nltk_supported_languages = ['hungarian', 'swedish', 'kazakh', 'norwegian',
                            'finnish', 'arabic', 'indonesian', 'portuguese',
                            'turkish', 'azerbaijani', 'slovene', 'spanish',
                            'danish', 'nepali', 'romanian', 'greek', 'dutch',
                            'tajik', 'german', 'english', 'russian',
                            'french', 'italian']
# stop words list
stop_words = set(stopwords.words(nltk_supported_languages))
# stemmer
porter = PorterStemmer()
def preprocess_texts(texts):
    '''Returns a list of clean and word-stemmed strings.'''
    preprocessed_texts = []
    for text in tqdm(texts):
        # punctuation marks cleaning
        text = ''.join([sym.lower() for sym in text if sym.isalpha() or sym == ' '])
        
        # tokenization
        tokenized_text = word_tokenize(text)
        
        # stop words cleaning
        tokenized_text_wout_sw = [word for word in tokenized_text if word not in stop_words]
        
        # stemming
        tokenized_text_wout_sw_stem = [porter.stem(word) for word in tokenized_text_wout_sw]
        
        # saving result
        preprocessed_texts += [' '.join(tokenized_text_wout_sw_stem)]
    
    return preprocessed_texts
%%time
for feature_name in text_features:
    data[feature_name] = preprocess_texts(data[feature_name])

data[text_features].head()
location = data['location'].copy()
location.head(15)
location_splitted = list(location.str.split(', ').values)
location_splitted[:15]
for loc_ind, loc in enumerate(location_splitted):
    if loc is np.nan:
        location_splitted[loc_ind] = ['Unpecified'] * 3
    else:
        for el_ind, el in enumerate(loc):
            if el == '':
                loc[el_ind] = 'Unpecified'
location_splitted[:15]
any([len(loc) > 3 for loc in location_splitted])
any([len(loc) < 3 for loc in location_splitted])
for loc_ind, loc in enumerate(location_splitted):
    if len(loc) > 3:
        print(loc_ind, loc)
for loc_ind, loc in enumerate(location_splitted):
    if len(loc) < 3:
        print(loc_ind, loc)
location_splitted[0] is list
type(location_splitted[0])
location_splitted = list(map(lambda loc: list(loc), location_splitted))
for loc_ind, loc in enumerate(location_splitted):
    if len(loc) > 3:
        location_splitted[loc_ind] = loc[:2] + [', '.join(loc[2:])]
    if len(loc) < 3:
        location_splitted[loc_ind] += ['Unpecified'] * 2
any([len(loc) != 3 for loc in location_splitted])
data_location = pd.DataFrame(location_splitted, columns=['country', 'state', 'city'])
data_location.head(15)
# complementing the list of categorical features
cat_features += ['country', 'state', 'city']
data = pd.concat([data, data_location], axis=1)
data.head()
data.drop('location', axis=1, inplace=True)
salary_range = data.salary_range.copy()
salary_range.head(15)
salary_range.fillna('0-0', inplace=True)
salary_range_sep = list(salary_range.str.split('-').values)
salary_range_sep[:5]
for range_ind, s_range in enumerate(salary_range_sep):
    if len(s_range) < 2 or len(s_range) > 2:
        print(range_ind, s_range)
salary_range_sep[5538] = ['40000', '40000']
error_range_inds = []
for range_ind, s_range in enumerate(salary_range_sep):
    min_value, max_value = s_range
    if not min_value.isdigit() or not max_value.isdigit():
        print(range_ind, (min_value, max_value))
        error_range_inds += [range_ind]
for range_ind in error_range_inds:
    salary_range_sep[range_ind] = ['0', '0']
data_salary_range = pd.DataFrame(np.array(salary_range_sep, dtype='int64'), 
                                 columns=['min_salary', 'max_salary'])
data_salary_range.head(15)
data_salary_range['salary_specified'] = ((data_salary_range.min_salary != 0) | 
                                         (data_salary_range.max_salary != 0)).astype('int64')
data_salary_range.head(15)
# creating the list of numerical features names and complementing the list of binary ones
num_features = ['min_salary', 'max_salary']
bin_features += ['salary_specified']
data = pd.concat([data, data_salary_range], axis=1)
data.head()
data.drop('salary_range', axis=1, inplace=True)
data.info()
data.fillna('Unspecified', inplace=True)
data.info()
plt.figure(figsize=(6, 4))
ax = sns.countplot(data.fraudulent)
plt.title('The distribution of the target feature (fraudulent)')
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.33, p.get_height()))

plt.show()
fig = plt.figure(figsize=(25, 30))
outer = gridspec.GridSpec(4, 2, wspace=0.2, hspace=0.1)

for feature_ind, feature_name in enumerate(bin_features):
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[feature_ind], 
                                             wspace=0.5, hspace=0.7)
    
    ax = plt.Subplot(fig, outer[feature_ind])
    ax.set_title(f'The distribution of fraudulent for each {feature_name}\'s class')
    ax.axis('off')
    fig.add_subplot(ax)
    
    for feature_class in [0, 1]:
        ax = plt.Subplot(fig, inner[feature_class])
        feature_cl_vc = data[data[feature_name] == feature_class].fraudulent.value_counts().sort_index()
        if len(feature_cl_vc) == 2:
            feature_cl_vc.index = ['non-fraudulent', 'fraudulent']
        else:
            feature_cl_vc.index = ['fraudulent']
        
        ax.pie(feature_cl_vc.values, labels=feature_cl_vc.index, autopct='%1.1f%%')
        ax.set_title(f'{feature_name} = {feature_class}')
        fig.add_subplot(ax)

fig.suptitle('Distributions of fraudulent for the binary features')
fig.subplots_adjust(top=0.95)
fig.show()
cont_table = pd.crosstab(data.fraudulent, data.description_specified)
print('Contingency table (fraudulent x description_specified):')
display(cont_table)
def show_feature1_x_feature2_info(feature_name1, feature_name2, figsize=(12, 4), is_binxcat=False):
    '''Shows info about a combination of two binary/categorical features.'''
    cont_table = pd.crosstab(data[feature_name1], data[feature_name2]).fillna(0)
    prop_table = pd.pivot_table(data, index=feature_name1, columns=feature_name2, 
                                values='fraudulent', aggfunc=np.mean).fillna(0)
    
    corr, p = cramers_v(cont_table.values)
    
    if is_binxcat:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    sns.heatmap(cont_table, annot=True, fmt='d', ax=axes[0])
    axes[0].set_title(f'Contingency table:')
    if is_binxcat:
        axes[0].set_xlabel('')
    
    sns.heatmap(prop_table, annot=True, ax=axes[1])
    axes[1].set_title(f'Proportion of fraudulent posts:')
    
    fig_title = f'{feature_name1} x {feature_name2} (Correlation: {round(corr, 4)}, p-value: {round(p, 4)}))'
    if is_binxcat:
        fig.suptitle(fig_title, y=1.05, x=0.45)
    else:
        fig.suptitle(fig_title, y=1.05)
    
    fig.show()
show_feature1_x_feature2_info('has_company_logo', 'company_profile_specified')
show_feature1_x_feature2_info('benefits_specified', 'has_questions')
show_feature1_x_feature2_info('telecommuting', 'has_questions')
show_feature1_x_feature2_info('telecommuting', 'benefits_specified')
show_feature1_x_feature2_info('benefits_specified', 'salary_specified')
round_confint = lambda confint: list(map(lambda lim: round(lim, 4), confint))
def print_stats_for_proportions(feature_name):
    fraudulent_0 = data[data[feature_name] == 0].fraudulent
    fraudulent_1 = data[data[feature_name] == 1].fraudulent
    
    prop_0 = round(np.mean(fraudulent_0), 4)
    prop_1 = round(np.mean(fraudulent_1), 4)
    prop_0_confint = round_confint(proportion_confint(fraudulent_0))
    prop_1_confint = round_confint(proportion_confint(fraudulent_1))
    
    bigger_prop, smaller_prop = (fraudulent_0, fraudulent_1) if prop_0 > prop_1 else (fraudulent_1, fraudulent_0)
    props_diff = round(np.mean(bigger_prop) - np.mean(smaller_prop), 4)
    props_diff_confint = round_confint(proportions_diff_confint_ind(bigger_prop, smaller_prop))
    z_test_p = proportions_ztest_ind(fraudulent_0, fraudulent_1)[1]
    
    print(f'Feature: {feature_name}\n======')
    print(f'Proportion of fraudulent posts for 0: {prop_0}')
    print(f'Proportion of fraudulent posts for 1: {prop_1}')
    print(f'Confidence interval for the proportion of fraudulent posts for 0: {prop_0_confint}')
    print(f'Confidence interval for the proportion of fraudulent posts for 1: {prop_1_confint}')
    print(f'Difference in these proportions: {props_diff}')
    print(f'Confidence interval for the difference in these proportions: {props_diff_confint}')
    print(f'Z-test result: {z_test_p} (p-value)')
print_stats_for_proportions('has_questions')
round((0.0331 / 0.0284) * 100, 1)
print_stats_for_proportions('salary_specified')
round((0.0273 / 0.0427) * 100, 1)
for feature_name in cat_features:
    print(f'Count of {feature_name}\'s unique values: {data[feature_name].unique().shape[0]}')
def plot_cat_feature_distribution(feature_name):
    '''Makes a plotly chart with categorical feature\'s distribution.'''
    feature_0f = data[data.fraudulent == 0][feature_name].value_counts()
    feature_1f = data[data.fraudulent == 1][feature_name].value_counts()
    
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], 
                        subplot_titles=['non-fraudulent', 'fraudulent'])
    fig.add_trace(go.Pie(labels=feature_0f.index, 
                         values=feature_0f.values), 
                  row=1, col=1)
    fig.add_trace(go.Pie(labels=feature_1f.index, 
                         values=feature_1f.values), 
                  row=1, col=2)
    
    fig.update_layout(title_text=f'The distribution of {feature_name}')
    fig.show()
plot_cat_feature_distribution('employment_type')
plot_cat_feature_distribution('required_experience')
plot_cat_feature_distribution('required_education')
func_meanfr_pt = pd.pivot_table(data, index='function', values='fraudulent', 
                                aggfunc=np.mean).sort_values(by='fraudulent', ascending=False)
func_meanfr_pt.columns = ['Proportion of fraudulent posts']
print('Top-15 function\'s values with the biggest proportions of fraudulent posts:')
display(func_meanfr_pt.head(15))
country_meanfr_pt = pd.pivot_table(data, index='country', values='fraudulent', 
                                   aggfunc=np.mean).sort_values(by='fraudulent', ascending=False)
country_meanfr_pt.columns = ['Proportion of fraudulent posts']
print('Top-15 country\'s values with the biggest proportions of fraudulent posts:')
display(country_meanfr_pt.head(15))
show_feature1_x_feature2_info('employment_type', 'required_experience', (18, 5))
show_feature1_x_feature2_info('benefits_specified', 'required_education', (14, 4.5), True)
show_feature1_x_feature2_info('has_questions', 'required_education', (14, 4.5), True)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

for ind, feature_name in enumerate(num_features):
    sns.boxplot(y=feature_name, x='fraudulent', data=data[data.salary_specified == 1], ax=axes[ind])
    axes[ind].set_ylim([-1e4, 2e5])
    axes[ind].set_xticklabels(['non-fraudulent', 'fraudulent'])
    axes[ind].set_title(f'Distributions of specified {feature_name}')

fig.suptitle('Distributions of min_salary and max_salary')
fig.show()
diff_salary = data[data.salary_specified == 1]['max_salary'] - data[data.salary_specified == 1]['min_salary']
plt.figure(figsize=(5, 5))
sns.boxplot(y=diff_salary, x='fraudulent', data=data[data.salary_specified == 1])
plt.ylim([-1e4, 1e5])
plt.xticks([0, 1], ['non-fraudulent', 'fraudulent'])
plt.ylabel('Difference')
plt.title('Distribution of difference between\n min and max salary')
plt.show()
specified_salaries = data[data.salary_specified == 1][num_features]
specified_salaries['difference'] = diff_salary
specified_salaries['fraudulent'] = data.fraudulent
specified_salaries.head()
np.sum(np.unique(specified_salaries.min_salary, return_counts=True)[1] > 10)
np.sum(np.unique(specified_salaries.max_salary, return_counts=True)[1] > 10)
def print_stats_for_salary(feature_name):
    '''Calculates statistics for fraudulent and non-fraudulent salary-feature.'''
    np.random.seed(42)
    feature_0f = specified_salaries[specified_salaries.fraudulent == 0][feature_name]
    feature_1f = specified_salaries[specified_salaries.fraudulent == 1][feature_name]
    
    med_0f = np.median(feature_0f)
    med_1f = np.median(feature_1f)
    med_0f_confint = bootstrap_statint(feature_0f.values, stat=np.median)
    med_1f_confint = bootstrap_statint(feature_1f.values, stat=np.median)
    
    bigger_med, smaller_med = (feature_0f, feature_1f) if med_0f > med_1f else (feature_1f, feature_0f)
    med_diff = np.median(bigger_med) - np.median(smaller_med)
    med_diff_confint = bootstrap_statint_diff(bigger_med.values, smaller_med.values, stat=np.median)
    perm_test_p = permutation_test_ind(feature_0f, feature_1f, max_permutations=5000)[1]
    
    print(f'Feature: {feature_name}\n======')
    print(f'Median of {feature_name} in non-fraudulent posts: {med_0f}')
    print(f'Median of {feature_name} in fraudulent posts:     {med_1f}')
    print(f'Statistical interval for the median of {feature_name} in non-fraudulent posts: {med_0f_confint}')
    print(f'Statistical interval for the median of {feature_name} in fraudulent posts:     {med_1f_confint}')
    print(f'Difference in these medians: {med_diff}')
    print(f'Statistical interval for the difference in these medians: {med_diff_confint}')
    print(f'Permutation test result: {perm_test_p} (p-value)')
print_stats_for_salary('min_salary')
print_stats_for_salary('max_salary')
print_stats_for_salary('difference')
fig, axes = plt.subplots(3, 2, figsize=(15, 18))

text_features_gen = iter(text_features)

for row in range(3):
    for col in range(2):
        try:
            feature_name = next(text_features_gen)
        except StopIteration:
            break
        
        if feature_name == 'title':
            feature_values_0f = data[(data.fraudulent == 0)][feature_name].astype(str)
            feature_values_1f = data[(data.fraudulent == 1)][feature_name].astype(str)
        else:
            feature_values_0f = data[(data.fraudulent == 0) & data[f'{feature_name}_specified']][feature_name].astype(str)
            feature_values_1f = data[(data.fraudulent == 1) & data[f'{feature_name}_specified']][feature_name].astype(str)

        fv_0f_len = feature_values_0f.str.split(' ').apply(len)
        fv_1f_len = feature_values_1f.str.split(' ').apply(len)
        
        sns.distplot(fv_0f_len, label='non-fraudulent', ax=axes[row, col])
        sns.distplot(fv_1f_len, label='fraudulent', ax=axes[row, col])
        axes[row, col].set_title(f'The distribution of {feature_name}\'s count of words')
        axes[row, col].legend()
        
fig.suptitle('Distributions of count of words for each text feature', y=0.92)
fig.show()
def print_stats_for_texts(feature_name):
    '''Calculates statistics for fraudulent and non-fraudulent count of words in feature\'s texts.'''
    if feature_name == 'title':
        feature_values_0f = data[(data.fraudulent == 0)][feature_name].astype(str)
        feature_values_1f = data[(data.fraudulent == 1)][feature_name].astype(str)
    else:
        feature_values_0f = data[(data.fraudulent == 0) & data[f'{feature_name}_specified']][feature_name].astype(str)
        feature_values_1f = data[(data.fraudulent == 1) & data[f'{feature_name}_specified']][feature_name].astype(str)
    
    lens_0f = feature_values_0f.str.split(' ').apply(len)
    lens_1f = feature_values_1f.str.split(' ').apply(len)
    
    mean_lens_0f = round(np.mean(lens_0f), 4)
    mean_lens_1f = round(np.mean(lens_1f), 4)
    mean_lens_0f_confint = round_confint(tconfint(lens_0f.values))
    mean_lens_1f_confint = round_confint(tconfint(lens_1f.values))
    
    bigger_mean, smaller_mean = (lens_0f, lens_1f) if mean_lens_0f > mean_lens_1f else (lens_1f, lens_0f)
    mean_diff = round(np.mean(bigger_mean) - np.mean(smaller_mean), 4)
    
    mean_diff_confint = round_confint(tconfint_diff(bigger_mean.values, smaller_mean.values))
    perm_test_p = permutation_test_ind(lens_0f, lens_1f, max_permutations=5000)[1]
    
    print(f'Feature: {feature_name}\n======')
    print(f'Mean of {feature_name}\'s count of words in non-fraudulent posts: {mean_lens_0f}')
    print(f'Mean of {feature_name}\'s count of words in fraudulent posts:     {mean_lens_1f}')
    print(f'Confidence interval for the mean of {feature_name}\'s count of words in non-fraudulent posts: {mean_lens_0f_confint}')
    print(f'Confidence interval for the mean of {feature_name}\'s count of words in fraudulent posts:     {mean_lens_1f_confint}')
    print(f'Difference in these means: {mean_diff}')
    print(f'Confidence interval for the difference in these means: {mean_diff_confint}')
    print(f'Permutation test result: {perm_test_p} (p-value)')
for feature_name in text_features:
    print_stats_for_texts(feature_name)
    print()
data['company_profile_count_of_words'] = data['company_profile'].astype(str).str.split(' ').apply(len)
data['requirements_count_of_words'] = data['requirements'].astype(str).str.split(' ').apply(len)
data.head()[['company_profile_count_of_words', 'requirements_count_of_words']]
num_features += ['company_profile_count_of_words', 'requirements_count_of_words']
data_1f = data[data.fraudulent == 1]
original_data = data.copy()
data = pd.concat([data] + [data_1f] * 7, axis=0)
plt.figure(figsize=(6, 4))
ax = sns.countplot(data.fraudulent)
plt.title('The distribution of the target feature (fraudulent)')
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x()+0.33, p.get_height()))

plt.show()
skf = StratifiedKFold(n_splits=4, random_state=42)
X, y = data.drop('fraudulent', axis=1), data.fraudulent
num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(ngram_range=(1, 2)))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features),
        *[(feature_name, text_transformer, feature_name) 
          for feature_name in text_features]
    ]
)
log_reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression())])
%%time
cv_scores = cross_validate(log_reg_pipe, X, y, return_train_score=True, cv=skf, 
                           scoring=['accuracy', 'roc_auc'], n_jobs=-1)

print(f'Accuracy on train part: {cv_scores["train_accuracy"]}, mean: {cv_scores["train_accuracy"].mean()}')
print(f'Accuracy on test part:  {cv_scores["test_accuracy"]}, mean: {cv_scores["test_accuracy"].mean()}')
print(f'ROC AUC on train part: {cv_scores["train_roc_auc"]}, mean: {cv_scores["train_roc_auc"].mean()}')
print(f'ROC AUC on test part:  {cv_scores["test_roc_auc"]}, mean: {cv_scores["test_roc_auc"].mean()}')
%%time
feature_names = num_features.copy()

num_features_scaled = StandardScaler().fit_transform(data[num_features])
X = num_features_scaled

feature_names += bin_features
X = np.hstack([X, data[bin_features]])


for feature_name in cat_features:
    encoder = OneHotEncoder()
    encoded_feature = encoder.fit_transform(data[feature_name].values.reshape(-1, 1))
    
    X = sparse_hstack([X, encoded_feature])
    f_names = list(map(lambda cat: f'{feature_name}:{cat}', encoder.categories_[0]))
    feature_names += f_names

for feature_name in text_features:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorized_feature = vectorizer.fit_transform(data[feature_name])
    
    X = sparse_hstack([X, vectorized_feature])
    sorted_phrases = [pair[0] for pair in list(sorted(vectorizer.vocabulary_.items(), 
                                                      key=lambda pair: pair[1]))]
    f_names = list(map(lambda phrase: f'{feature_name}:{phrase}', sorted_phrases))
    feature_names += f_names
X.shape[1], len(feature_names)
log_reg = LogisticRegression(random_state=42, n_jobs=-1).fit(X, y)
eli5.explain_weights(log_reg, feature_names=feature_names, top=(30, 30))
original_data[original_data.country == 'MY'].fraudulent.value_counts().sort_index()
original_data[original_data.country == 'GR'].fraudulent.value_counts().sort_index()
original_data[original_data.industry == 'Accounting'].fraudulent.value_counts().sort_index()
original_data[original_data.industry == 'Internet'].fraudulent.value_counts().sort_index()
original_data[original_data.city == 'london'].fraudulent.value_counts().sort_index()
original_data[original_data.city == 'London'].fraudulent.value_counts().sort_index()
original_data[original_data.city == 'chicago'].fraudulent.value_counts().sort_index()
original_data[original_data.city == 'Chicago'].fraudulent.value_counts().sort_index()