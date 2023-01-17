from IPython.display import display

import itertools

import numpy as np
from scipy import stats
import pandas as pd

import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 100
import matplotlib.gridspec as gridspec
import seaborn as sns; sns.set()

import statsmodels.formula.api as smf
data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data.info()
np.all((data.status == 'Not Placed') == data.salary.isna())
np.all(data.sl_no == np.arange(1, 216))
data.drop(['sl_no', 'salary'], axis=1, inplace=True)
bin_features = ['gender', 'ssc_b', 'hsc_b', 'workex', 'specialisation']
cat_features = ['hsc_s', 'degree_t']
num_features = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

feature_names = iter(bin_features + cat_features + ['status'])
for row in range(2):
    for col in range(4):
        feature_name = next(feature_names)
        sns.countplot(data[feature_name], ax=axes[row, col])
        axes[row, col].set_title(f'The distribution of {feature_name}')

fig.suptitle('Distributions of the binary and categorical features (+ target feature)')
fig.subplots_adjust(top=0.92)
fig.show()
fig, axes = plt.subplots(3, 2, figsize=(22, 17))

feature_names = iter(num_features)
for row in range(3):
    for col in range(2):
        try:
            feature_name = next(feature_names)
        except StopIteration:
            break
        
        sns.distplot(data[feature_name], ax=axes[row, col])    
        axes[row, col].set_title(f'The distribution of {feature_name}')
        

fig.suptitle('Distributions of the numerical features')
fig.subplots_adjust(top=0.93)
fig.show()
fig = plt.figure(figsize=(23, 18))
outer = gridspec.GridSpec(3, 2, wspace=0.2, hspace=0.2)

for feature_ind, feature_name in enumerate(bin_features):
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[feature_ind], 
                                             wspace=0.3, hspace=0.3)
    
    ax = plt.Subplot(fig, outer[feature_ind])
    ax.set_title(f'The distribution of status for each {feature_name}\'s class')
    ax.axis('off')
    fig.add_subplot(ax)
    
    for pie_ind, f_class in enumerate(data[feature_name].unique()):
        ax = plt.Subplot(fig, inner[pie_ind])
        f_class_status_vc = data[data[feature_name] == f_class]['status'].value_counts().sort_index()
        ax.pie(f_class_status_vc.values, labels=f_class_status_vc.index, autopct='%1.1f%%')
        ax.set_title(f_class)
        fig.add_subplot(ax)

fig.suptitle('Distributions of status for the binary features')
fig.subplots_adjust(top=0.93)
fig.show()
fig = plt.figure(figsize=(20, 15))
outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.1)

for feature_ind, feature_name in enumerate(cat_features):
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[feature_ind], 
                                             wspace=0.2, hspace=0.2)
    
    ax = plt.Subplot(fig, outer[feature_ind])
    ax.set_title(f'The distribution of status for each {feature_name}\'s class')
    ax.axis('off')
    fig.add_subplot(ax)
    
    for pie_ind, f_class in enumerate(data[feature_name].unique()):
        ax = plt.Subplot(fig, inner[pie_ind])
        f_class_status_vc = data[data[feature_name] == f_class]['status'].value_counts().sort_index()
        ax.pie(f_class_status_vc.values, labels=f_class_status_vc.index, autopct='%1.1f%%')
        ax.set_title(f_class)
        fig.add_subplot(ax)

fig.suptitle('Distributions of status for the categorical features')
fig.subplots_adjust(top=0.93)
fig.show()
# transforming status into binary feature to calculate proportions
data_01_status = data.copy()
data_01_status['status'].replace({'Placed': 1, 'Not Placed': 0}, inplace=True)
# proportions of placed people for different binary/categorical features values combinations
bin_x_cat_status_props = pd.pivot_table(data_01_status, index=cat_features, columns=bin_features, 
                                        values='status', aggfunc=np.mean)
plt.figure(figsize=(18, 5))
sns.heatmap(bin_x_cat_status_props, annot=True).tick_params(labelsize=9)
plt.title('Proportions of placed people for the different binary/categorical features values combinations')
plt.show()
print('Top of the binary features combinations:')
display(pd.DataFrame((
    bin_x_cat_status_props.sum(axis=0) / (bin_x_cat_status_props.shape[0] - bin_x_cat_status_props.isna().sum(axis=0))
).sort_values(ascending=False), columns=['Mean proportion of placed people']))

print('\n\nTop of the categorical features combinations:')
display(pd.DataFrame((
    bin_x_cat_status_props.sum(axis=1) / (bin_x_cat_status_props.shape[1] - bin_x_cat_status_props.isna().sum(axis=1))
).sort_values(ascending=False), columns=['Mean proportion of placed people']))
orange = (0.8666666666666667, 0.5176470588235295, 0.3215686274509804)
blue = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
pp = sns.pairplot(data[num_features + ['status']], hue='status', palette={'Placed': orange, 'Not Placed': blue})
pp.fig.suptitle('Distributions and pairplots of the numerical features divided by status', y=1.03)
pp.fig.show()
def matthews_correlation(contingency_table):
    '''Matthews correlation calculation.'''
    a, b = contingency_table[0]
    c, d = contingency_table[1]

    n = np.sum(contingency_table)
    acabn = (a + c) * (a + b) / n
    accdn = (a + c) * (c + d) / n
    bdabn = (b + d) * (a + b) / n
    bdcdn = (b + d) * (c + d) / n
    if n < 40 or np.any(np.array([acabn, accdn, bdabn, bdcdn]) < 5):
        raise ValueError('Contingency table isn\'t suitable for Matthews correlation calculation.')

    p_value = stats.chi2_contingency(contingency_table)[1]
    corr = (a * d - b * c) / np.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    return corr, p_value


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
    '''Confidence interval for the difference in two independent proportions.'''
    z = stats.norm.ppf(1 - alpha / 2)
    p1 = np.mean(sample1)
    p2 = np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    return left_boundary, right_boundary


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
print('BINARY FEATURES\n'
      '===============\n\n')
for feature_name in bin_features:
    class1, class2 = data_01_status[feature_name].unique()
    class1_status = data_01_status[data[feature_name] == class1].status
    class2_status = data_01_status[data[feature_name] == class2].status
    
    print(f'Feature: {feature_name}\n------')
    
    print('Contingency table:')
    contingency_table = pd.crosstab(data['status'], data[feature_name])
    display(contingency_table)
    corr, p = matthews_correlation(contingency_table.values)
    print(f'Correlation between status and {feature_name}: {round(corr, 4)}, p-value: {p}')
    if p < 0.05:
        print(f'There is a correlation between {feature_name} and status.')
    else:
        print(f'There isn\'t any correlation between {feature_name} and status.')
        
    print()
    
    print(f'The proportion of placed people for {class1}: {round(np.mean(class1_status), 4)}')
    print(f'The proportion of placed people for {class2}: {round(np.mean(class2_status), 4)}')
    
    class1_confint = list(map(lambda lim: round(lim, 4), proportion_confint(class1_status)))
    class2_confint = list(map(lambda lim: round(lim, 4), proportion_confint(class2_status)))
    print(f'The confidence interval (95%) for {class1}: {class1_confint}')
    print(f'The confidence interval (95%) for {class2}: {class2_confint}')
    if class2_confint[0] < class1_confint[0] < class2_confint[1] or \
       class2_confint[0] < class1_confint[1] < class2_confint[1]:
        print('The intervals overlap.')
    else:
        print('The intervals don\'t overlap.')
    
    print()
    
    if np.mean(class1_status) > np.mean(class2_status):
        bigger_prop, smaller_prop = class1_status, class2_status
    else:
        bigger_prop, smaller_prop = class2_status, class1_status
    
    print(f'The difference in the proportions: {round(np.mean(bigger_prop) - np.mean(smaller_prop), 4)}')
    prop_diff_confint = list(map(lambda lim: round(lim, 4), proportions_diff_confint_ind(bigger_prop, 
                                                                                         smaller_prop)))
    print(f'The confidence interval (95%) for the difference in the proportions: {prop_diff_confint}')
    if prop_diff_confint[0] > 0:
        print(f'The proportions may differ by at least {prop_diff_confint[0]}.')
    else:
        print('The difference between the proportions may be 0.')
    
    print()
    
    p = proportions_ztest_ind(class1_status, class2_status)[1]
    print(f'Z-test result (two-sided): {p} (p-value)')
    if p < 0.05:
        print('The proportions are probably unequal.')
    else:
        print('The proportions are probably equal.')
    
    print('\n\n')
def cramers_v(contingency_table):
    '''Cramer\'s V coefficient.'''
    n = np.sum(contingency_table)
    ct_nrows, ct_ncols = contingency_table.shape
    if n < 40 or np.sum(contingency_table < 5) / (ct_nrows * ct_ncols) > 0.2:
        raise ValueError('Contingency table isn\'t suitable for Cramers\'s V coefficient calculation.')

    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    corr = np.sqrt(chi2 / (n * (min(ct_nrows, ct_ncols) - 1)))
    return corr, p_value
print('CATEGORICAL FEATURES\n'
      '====================\n\n')
for feature_name in cat_features:
    print(f'Feature: {feature_name}\n------')
    print('Contingency table:')
    contingency_table = pd.crosstab(data['status'], data[feature_name])
    display(contingency_table)
    corr, p = cramers_v(contingency_table.values)
    print(f'Correlation between status and {feature_name}: {round(corr, 4)}, p-value: {p}')
    if p < 0.05:
        print(f'There is a correlation between {feature_name} and status.')
    else:
        print(f'There isn\'t any correlation between {feature_name} and status.')
    
    print('\n\n')
norm_features = [feature_name for feature_name in num_features if stats.shapiro(data[feature_name])[1] < 0.05]
print(f'Numerical features: {", ".join(num_features)}')
print(f'Features that are probably normally distributed: {", ".join(norm_features)}')
for feature_name in num_features:
    feature_placed = data[data.status == 'Placed'][feature_name]
    feature_not_placed = data[data.status == 'Not Placed'][feature_name]
    
    p_placed = stats.shapiro(feature_placed)[1]
    p_not_placed = stats.shapiro(feature_not_placed)[1]
    if p_placed < 0.05 and p_not_placed < 0.05:
        print(f'The both distributions of {feature_name} for the different status are probably normally distributed.')
norm_num_features = ['degree_p', 'etest_p']
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
    
    t = np.abs(stats.t.ppf(alpha / 2, dof))
    left_boundary = (mean1 - mean2) - t * np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    right_boundary = (mean1 - mean2) + t * np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    
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
        indices = [(list(index), list(filter(lambda i: i not in index, range(n))))
                    for index in itertools.combinations(range(n), n1)]

    zero_distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean()
                  for i in indices]

    if alternative == 'two-sided':
        p_value = sum([abs(x) >= abs(t_stat) for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        p_value = sum([x <= t_stat for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        p_value = sum([x >= t_stat for x in zero_distr]) / len(zero_distr)

    return t_stat, p_value
print('NUMERICAL FEATURES\n'
      '==================\n\n')
for feature_name in num_features:
    feature_placed = data[data.status == 'Placed'][feature_name]
    feature_not_placed = data[data.status == 'Not Placed'][feature_name]
    
    print(f'Feature: {feature_name}\n------')
    
    corr, p = stats.pointbiserialr(data_01_status.status, data[feature_name])
    print(f'Correlation between status and {feature_name}: {round(corr, 4)}, p-value: {p}')
    if p < 0.05:
        print(f'There is a correlation between {feature_name} and status.')
    else:
        print(f'There isn\'t any correlation between {feature_name} and status.')
        
    print()
    
    print(f'Mean of {feature_name} for placed people:     {round(np.mean(feature_placed), 4)}')
    print(f'Mean of {feature_name} for not placed people: {round(np.mean(feature_not_placed), 4)}')
    
    feature_placed_confint = list(map(lambda lim: round(lim, 4), tconfint(feature_placed)))
    feature_not_placed_confint = list(map(lambda lim: round(lim, 4), tconfint(feature_not_placed)))
    
    print(f'Confidence interval (95%) of mean {feature_name} for placed people:     {feature_placed_confint}')
    print(f'Confidence interval (95%) of mean {feature_name} for not placed people: {feature_not_placed_confint}')
    if feature_not_placed_confint[0] < feature_placed_confint[0] < feature_not_placed_confint[1] or \
       feature_not_placed_confint[0] < feature_placed_confint[1] < feature_not_placed_confint[1]:
        print('The intervals overlap.')
    else:
        print('The intervals don\'t overlap.')
    
    print()
    
    if np.mean(feature_placed) > np.mean(feature_not_placed):
        bigger_mean, smaller_mean = feature_placed, feature_not_placed
    else:
        bigger_mean, smaller_mean = feature_not_placed, feature_placed
    
    mean_diff_confint = list(map(lambda lim: round(lim, 4), tconfint_diff(bigger_mean, smaller_mean)))
    print(f'Difference in means: {round(np.mean(bigger_mean) - np.mean(smaller_mean), 4)}')
    print(f'Confidence interval (95%) for the difference in means: {mean_diff_confint}')
    if mean_diff_confint[0] > 0:
        print(f'The means may differ by at least {mean_diff_confint[0]}.')
    else:
        print('The difference in means between the samples may be 0.')
        
    print()

    if feature_name in norm_num_features:
        comparison_subject = 'means'
        p = stats.ttest_ind(feature_placed, feature_not_placed, equal_var=False)[1]
        print(f'Student\'s t-test result (two-sided): {p} (p-value)')
    else:
        comparison_subject = 'distributions'
        p = permutation_test_ind(feature_placed, feature_not_placed, max_permutations=5000)[1]
        print(f'Permutation test result (two-sided): {p} (p-value)')
    
    if p < 0.05:
        print(f'The {comparison_subject} of the samples are probably unequal.')
    else:
        print(f'The {comparison_subject} of the samples are probably equal.')
    
    print('\n\n')
data_01_status[num_features] = data[num_features] / 100
data_01_status[num_features].describe()
data_01_status.head()
formula = 'status ~ C(' + ') + C('.join(bin_features + cat_features) + ') + ' + ' + '.join(num_features)
formula
model = smf.logit(formula, data=data_01_status)
fitted = model.fit()
print(fitted.summary())
formula = 'status ~ C(degree_t) + C(workex) + ssc_p + hsc_p + degree_p + mba_p'
model = smf.logit(formula, data=data_01_status)
fitted = model.fit()
print(fitted.summary())