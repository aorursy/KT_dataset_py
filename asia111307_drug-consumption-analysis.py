import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import pearsonr, spearmanr, chi2_contingency, ttest_ind, mannwhitneyu
sns.set(rc={'figure.figsize':(10,5)})

sns.set_style('whitegrid')

%matplotlib inline  
def value_counts_percentage(dataset, column):

    ''' value.counts() method extended by displaying percentage '''

    

    a = dataset[column].value_counts()

    b = dataset[column].value_counts(normalize=True) * 100

    

    return pd.concat([a,b.round(2)], axis=1, keys=['N', '%'])
def heatmap_corr(dataset, method='spearman', ready=False, mask=True, nominal=False):

    ''' Extended sns.heatmap() method. 

    

    dataset - can be 'pure' data (without calculated correlations) or a DataFrame with already calcuateg 

        correlations (in that case attribute 'ready' should be set to True);

    method - mainly pearson or spearman; nominal correlations should be calculated externally 

        and be delivered with attribute ready=True; 

    mask - if dataset is NOT a cross-valued DataFrame of one type, mask should be set to False;

    nominal - for nominal data correlations values are in range (0, 1) instead of (-1, -1). 

        nominal=True should be folowed by ready=True 

    '''

    

    if not ready:

        corr = dataset.corr(method=method)

    elif ready:

        corr = dataset

    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    vmax = corr.max().max()

    if nominal:

        center = 0.5

        cmap=None

    elif not nominal:

        center = 0

    if mask:

        mask = np.zeros_like(corr, dtype=np.bool)

        mask[np.triu_indices_from(mask)] = True

        vmax = corr.replace(1, -2).max().max()

    elif not mask:

        mask=None

    f, ax = plt.subplots(figsize=(20,9))

    

    return sns.heatmap(corr, cmap=cmap, mask=mask, vmax=vmax, center=center, annot=True, square=True, 

                       linewidths=0.5, cbar_kws={'shrink': 0.5})
def calculate_r(df1, df2, method='spearman', p=0.05, pvalues=True):

    ''' Returns correlation matrix extended by statistical significance index. Used for non-nominal data.

    

    df1, df2 - DataFrames of data to correlate;

    method - mainly pearson and spearman;

    p - significance level;

    pvalues - if set to False, only correlation values will be returned in DataFrame 

        (without '**' marks for significant observations)

    '''

    

    data_corr_table = pd.DataFrame()

    data_pvalues = pd.DataFrame()

    for x in df1.columns:

        for y in df2.columns:

            if method == 'pearson':

                corr = pearsonr(df1[x], df2[y])

            elif method == 'spearman':

                corr = spearmanr(df1[x], df2[y])

            else:

                raise ValueError('Unknown method')

            if pvalues:

                data_corr_table.loc[x,y] = '{} {}'.format(round(corr[0], 3), '**' if round(corr[1], 3) < p else '')

            elif not pvalues:

                data_corr_table.loc[x,y] = round(corr[0], 3)

            data_pvalues.loc[x,y] = round(corr[1], 3)

    

    return data_corr_table, data_pvalues
def cramers_v(crosstab):

    ''' Returns Cramer's V correlation coefficient (and statistic significance) for data 

        delivered as a crosstable. Used for nominal data.

    '''

    

    chi2 = chi2_contingency(crosstab)[0]

    n = crosstab.sum().sum()

    phi2 = chi2/n

    r,k = crosstab.shape

    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    

    rcorr = r - ((r-1)**2)/(n-1)

    kcorr = k - ((k-1)**2)/(n-1)

    

    return round(np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))), 3), '**' if chi2_contingency(crosstab)[1] < 0.05 else ''
def nominal_corrs(dataset, col_1_names, col_2_names, pvalues=True):

    ''' Returns Cramer's V coefficients matrix for the whole dataset.

    

    col_1_names, col_2_names - lists of names of columns to correlate. Function creates crosstables for every 

        columns' combination and returns a matrix with single Cramer's V coefficients of every combination;

    pvalues - if set to False, only correlation values will be returned in DataFrame 

        (without '**' marks for significant observations)

    '''

    

    corr_table = pd.DataFrame()

    for i in range(len(col_1_names)):

        for j in range(len(col_2_names)):

            crosstab = pd.crosstab(dataset[col_1_names[i]], [dataset[col_2_names[j]]])

            if pvalues:

                v = ' '.join([str(i) for i in cramers_v(crosstab)])

            elif not pvalues:

                v = cramers_v(crosstab)[0]

            corr_table.loc[i, j] = v

    corr_table.index = col_1_names

    corr_table.columns = col_2_names

    

    return corr_table
def diff_test(dataset, sample_attr, diff_attr, sample_attr_type='ordered', diff_attr_type='ordered'):

    ''' Difference significance test on dataset. Returns a text summary.

    

    sample_attr - column which will be divided into two samples with median value;

    diff_attr - attribute, which value will be checked in two sample groups;

    diff_attr_type - determines type of data which medians will be compared (ordered, interval)

    '''

    

    sample_attr_central = dataset[sample_attr].median() if sample_attr_type=='ordered' else dataset[sample_attr].mean()

    group_1 = dataset[dataset[sample_attr] > sample_attr_central]

    group_2 = dataset[dataset[sample_attr] <= sample_attr_central]

    

    group_1_diff_attr_central = group_1[diff_attr].median() if diff_attr_type=='ordered' else group_1[diff_attr].mean()

    group_2_diff_attr_central = group_2[diff_attr].median() if diff_attr_type=='ordered' else group_2[diff_attr].mean()

    

    if diff_attr_type == 'ordered':

        diff_sign, p = mannwhitneyu(group_1[diff_attr], group_2[diff_attr])

    elif diff_attr_type == 'interval':

        diff_sign, p = ttest_ind(group_1[diff_attr], group_2[diff_attr])

    are = 'are' if p < 0.05 else 'are not'

    sample_central = 'median' if sample_attr_type=='ordered' else 'mean'

    diff_central = 'median' if diff_attr_type=='ordered' else 'mean'

    

    return f'First group: {sample_attr} above {sample_central} value {round(sample_attr_central, 3)}\n Second group: {sample_attr} equal or below {sample_central} value {round(sample_attr_central, 3)} \n First group {diff_attr} {diff_central}: {round(group_1_diff_attr_central, 3)} \n Second group {diff_attr} {diff_central}: {round(group_2_diff_attr_central, 3)} \n Difference significance for samples: {round(diff_sign, 3)} with p-value: {round(p, 3)} \n Samples {are} statistically different.'
all_columns = [

    'ID', 

    'Age', 

    'Gender', 

    'Education', 

    'Country',

    'Ethnicity',

    'Neuroticism',

    'Extraversion',

    'Openness to experience',

    'Agreeableness',

    'Conscientiousness',

    'Impulsiveness',

    'Sensation seeking',

    'Alcohol consumption',

    'Amphetamines consumption',

    'Amyl nitrite consumption',

    'Benzodiazepine consumption',

    'Caffeine consumption',

    'Cannabis consumption',

    'Chocolate consumption',

    'Cocaine consumption',

    'Crack consumption',

    'Ecstasy consumption',

    'Heroin consumption',

    'Ketamine consumption',

    'Legal highs consumption',

    'Lysergic acid diethylamide consumption',

    'Methadone consumption',

    'Magic mushrooms consumption',

    'Nicotine consumption',

    'Fictitious drug Semeron consumption',

    'Volatile substance abuse consumption'

]



demographic_columns = [

    'Age', 

    'Gender', 

    'Education', 

    'Country',

    'Ethnicity',

]



personality_columns = [

    'Neuroticism',

    'Extraversion',

    'Openness to experience',

    'Agreeableness',

    'Conscientiousness',

    'Impulsiveness',

    'Sensation seeking'

]



drugs_columns = [

    'Alcohol consumption',

    'Amphetamines consumption',

    'Amyl nitrite consumption',

    'Benzodiazepine consumption',

    'Caffeine consumption',

    'Cannabis consumption',

    'Chocolate consumption',

    'Cocaine consumption',

    'Crack consumption',

    'Ecstasy consumption',

    'Heroin consumption',

    'Ketamine consumption',

    'Legal highs consumption',

    'Lysergic acid diethylamide consumption',

    'Methadone consumption',

    'Magic mushrooms consumption',

    'Nicotine consumption',

    'Fictitious drug Semeron consumption',

    'Volatile substance abuse consumption'

]
data = pd.read_csv('../input/drug_consumption.data', header=None, names=all_columns)

data.shape
for i in drugs_columns:

    data[i] = data[i].map({'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6})
semerons = data[data['Fictitious drug Semeron consumption'] != 0]

semerons
data = data[data['Fictitious drug Semeron consumption'] == 0]

drugs_columns.remove('Fictitious drug Semeron consumption')

data.shape
demo_data = data.copy()
age = ['18-24' if a <= -0.9 else 

       '25-34' if a >= -0.5 and a < 0 else 

       '35-44' if a > 0 and a < 1 else 

       '45-54' if a > 1 and a < 1.5 else 

       '55-64' if a > 1.5 and a < 2 else 

       '65+' 

       for a in demo_data['Age']]



gender = ['Female' if g > 0 else "Male" for g in demo_data['Gender']]



education = ['Left school before 16 years' if e <-2 else 

             'Left school at 16 years' if e > -2 and e < -1.5 else 

             'Left school at 17 years' if e > -1.5 and e < -1.4 else 

             'Left school at 18 years' if e > -1.4 and e < -1 else 

             'Some college or university, no certificate or degree' if e > -1 and e < -0.5 else 

             'Professional certificate/ diploma' if e > -0.5 and e < 0 else 

             'University degree' if e > 0 and e < 0.5 else 

             'Masters degree' if e > 0.5 and e < 1.5 else 

             'Doctorate degree' 

             for e in demo_data['Education']]



country = ['USA' if c < -0.5 else 

           'New Zealand' if c > -0.5 and c < -0.4 else 

           'Other' if c > -0.4 and c < -0.2 else 

           'Australia' if c > -0.2 and c < 0 else 

           'Ireland' if c > 0 and c < 0.23 else 

           'Canada' if c > 0.23 and c < 0.9 else 

           'UK' 

           for c in demo_data['Country']]



ethnicity = ['Black' if e < -1 else 

             'Asian' if e > -1 and e < -0.4 else 

             'White' if e > -0.4 and e < -0.25 else 

             'Mixed-White/Black' if e >= -0.25 and e < 0.11 else 

             'Mixed-White/Asian' if e > 0.12 and e < 1 else 

             'Mixed-Black/Asian' if e > 1.9 else 

             'Other' 

             for e in demo_data['Ethnicity']]





demo_data['Age'] = age

demo_data['Gender'] = gender

demo_data['Education'] = education

demo_data['Country'] = country

demo_data['Ethnicity'] = ethnicity
value_counts_percentage(demo_data, 'Gender')
value_counts_percentage(demo_data, 'Age')
sns.countplot(x='Age', palette='ch:.25', data=demo_data.sort_values(by=['Age']))
value_counts_percentage(demo_data, 'Education')
edu_plot = sns.countplot(x='Education', palette='ch:.25', data=demo_data.sort_values(by=['Education']))

edu_plot.set_xticklabels(edu_plot.get_xticklabels(), rotation=40, ha="right")

edu_plot
value_counts_percentage(demo_data, 'Ethnicity')
value_counts_percentage(demo_data, 'Country')
sns.countplot(x='Age', hue='Gender', palette='ch:.25', data=demo_data.sort_values(by=['Age']))
sns.countplot(x='Country', hue='Gender', palette='ch:.25', data=demo_data.sort_values(by=['Country']))
pd.crosstab(demo_data['Age'], [demo_data['Gender'], demo_data['Country']])
demo_data.pivot_table(index='Education', columns=['Gender', 'Ethnicity'], aggfunc='size', fill_value=0)
pd.crosstab(demo_data['Country'], [demo_data['Gender'], demo_data['Ethnicity']])
corr_table_demo = nominal_corrs(demo_data, demographic_columns, demographic_columns)

corr_table_demo
corr_table_demo_no_pvalues = nominal_corrs(demo_data, demographic_columns, demographic_columns, pvalues=False)

heatmap_corr(corr_table_demo_no_pvalues, nominal=True, ready=True)
drug_data = data[drugs_columns]
d1 = drug_data['Alcohol consumption'].value_counts()

d2 = drug_data['Amphetamines consumption'].value_counts()

d3 = drug_data['Amyl nitrite consumption'].value_counts()

d4 = drug_data['Benzodiazepine consumption'].value_counts()

d5 = drug_data['Caffeine consumption'].value_counts()

d6 = drug_data['Cannabis consumption'].value_counts()

d7 = drug_data['Chocolate consumption'].value_counts()

d8 = drug_data['Cocaine consumption'].value_counts()

d9 = drug_data['Crack consumption'].value_counts()

d10 = drug_data['Ecstasy consumption'].value_counts()

d11 = drug_data['Heroin consumption'].value_counts()

d12 = drug_data['Ketamine consumption'].value_counts()

d13 = drug_data['Legal highs consumption'].value_counts()

d14 = drug_data['Lysergic acid diethylamide consumption'].value_counts()

d15 = drug_data['Methadone consumption'].value_counts()

d16 = drug_data['Magic mushrooms consumption'].value_counts()

d17 = drug_data['Nicotine consumption'].value_counts()

d18 = drug_data['Volatile substance abuse consumption'].value_counts()



drug_table = pd.concat([d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18], axis=1, 

          keys=['Alcohol', 'Amphetamines', 'Amyl nitrite', 'Benzodiazepine', 'Caffeine', 'Cannabis', 'Chocolate', 

                'Cocaine', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legal highs', 'LSD', 

                'Methadone', 'Magic mushrooms', 'Nicotine', 'VSA'], sort=True)

drug_table.T.fillna(0).astype(np.int64)
heatmap_corr(drug_data, method='spearman')
drug_data_corr_table, drug_data_pvalues = calculate_r(drug_data, drug_data, method='spearman')

drug_data_corr_table
pers_data = data[personality_columns]

pers_data.describe()
heatmap_corr(pers_data, method='spearman')
pers_data_corr_table, pers_data_pvalues = calculate_r(pers_data, pers_data, method='spearman')

pers_data_corr_table
drug_pers_corr_table, drug_pers_pvalues = calculate_r(drug_data, pers_data, method='spearman')

drug_pers_corr_table
drug_pers_corr_table_no_pvalues, drug_pers_pvalues_no_pvalues = calculate_r(drug_data, pers_data, pvalues=False, method='spearman')

heatmap_corr(drug_pers_corr_table_no_pvalues, ready=True, mask=False)
corr_table = nominal_corrs(data, demographic_columns, drugs_columns)

corr_table
corr_table_no_pvalues = nominal_corrs(data, demographic_columns, drugs_columns, pvalues=False)

heatmap_corr(corr_table_no_pvalues, ready=True, mask=False, nominal=True)
gender_data_female = demo_data[demo_data['Gender']=='Female']

gender_data_male = demo_data[demo_data['Gender']=='Male']

print(f"Female cannabis consumption median: {gender_data_female['Cannabis consumption'].median()}")

print(f"Male cannabis consumption median: {gender_data_male['Cannabis consumption'].median()}")

print(f"Female legal highs consumption median: {gender_data_female['Legal highs consumption'].median()}")

print(f"Male legal highs consumption median: {gender_data_male['Legal highs consumption'].median()}")
corr_table_demo_psycho = nominal_corrs(data, demographic_columns, personality_columns)

corr_table_demo_psycho
corr_table_demo_psycho_no_pvalues = nominal_corrs(data, demographic_columns, personality_columns, pvalues=False)

heatmap_corr(corr_table_demo_psycho_no_pvalues, ready=True, nominal=True, mask=False)
print(f"Female Agreeableness median: {gender_data_female['Agreeableness'].median()}")

print(f"Male Agreeableness median: {gender_data_male['Agreeableness'].median()}")

print(f"Female Sensation seeking median: {gender_data_female['Sensation seeking'].median()}")

print(f"Male Sensation seeking median: {gender_data_male['Sensation seeking'].median()}")
diff_data = data.copy()

value_diff = diff_test(diff_data, 'Education', 'Cannabis consumption', sample_attr_type='ordered', diff_attr_type='ordered')

print(value_diff)
value_diff = diff_test(diff_data, 'Sensation seeking', 'Cannabis consumption', sample_attr_type='ordered', diff_attr_type='ordered')

print(value_diff)
value_diff = diff_test(diff_data, 'Education', 'Sensation seeking', diff_attr_type='ordered')

print(value_diff)
value_diff = diff_test(diff_data, 'Gender', 'Caffeine consumption', diff_attr_type='ordered')

print(value_diff)