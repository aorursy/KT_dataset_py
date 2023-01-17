# import all packages and set plots to be embedded inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

from matplotlib.ticker import FuncFormatter

from scipy.stats import norm

from scipy import stats

import warnings

warnings.simplefilter('ignore')



%matplotlib inline

# intial formatting of the data before we read the file in

# helps reduce clutter and noise when displaying numbers

pd.set_option('float_format', '{:.02f}'.format)
# load in dataset into a pandas dataframe

sf_data = pd.read_csv('../input/SALARIES_2.csv', low_memory = False)
sf_data.info()
sf_data.sample(5)
# cleaning up of column headers

sf_data.columns = sf_data.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('-', '_')
sf_data.columns
# adjust columns names for easier code later on and consistency

sf_data.rename(index=str, columns={"employee_identifier": "emp_id",

                                   "organization_group": "org_group",

                                   "department": "dept",

                                   "total_compensation": "total_comp",

                                   "salaries": 'base_salary',

                                   "other_salaries": "other_salary",

                                   "total_benefits": "total_benefit",

                                   "other_benefits": "other_benefit",

                                   "health_and_dental": "health_dental"}, inplace = True)
sf_data.columns
# make year categorical for analysis later

# make emp_id an object since the numerical representation cannot be evaluated statistically

sf_data['year'] = sf_data['year'].astype('object')

sf_data['emp_id'] = sf_data['emp_id'].astype('object')
# set continuous variables

variables = ['base_salary', 'overtime', 'other_salary', 'total_salary', 'retirement',

       'health_dental', 'other_benefit', 'total_benefit', 'total_comp']
sf_data.info()
# find descriptive statistics for the numeric variable in the dataset

sf_data.describe()
# using pandas query function to filter out negative entries for total_salary and total_benefit

zero_comp = sf_data[(sf_data.total_benefit == 0) & (sf_data.total_salary == 0) &

                     (sf_data.other_benefit == 0) & (sf_data.other_salary == 0) &

                     (sf_data.retirement == 0) & (sf_data.overtime == 0) &

                     (sf_data.health_dental == 0) & (sf_data.base_salary == 0)]

zero_comp_index = zero_comp.index
# dropping negative entries

sf_data.drop(index = zero_comp_index, inplace = True)
# reset the index in order for it to be easier to work with later

sf_data.reset_index(inplace = True, drop = True)
sf_data.describe()
duplicates = sf_data[sf_data.duplicated(subset = variables,keep = 'first')]

duplicates.head()
dupe_index = duplicates.index
# dropped these in place

sf_data.drop(index = dupe_index, inplace = True)
sf_data.reset_index(inplace = True, drop = True)
sf_data.describe()
# tick converter for x-axis

def thousands(x, pos):

    '''The two args are the value and tick position'''

    return '%1.fK' % (x * 1e-3)

format_x = FuncFormatter(thousands)

def x_format():

    x_tick_form = ax.xaxis.set_major_formatter(format_x)

    return x_tick_form
# tick converter for y-axis

def thousands(y, pos):

    '''The two args are the value and tick position'''

    return '%1.fK' % (y * 1e-3)

format_y = FuncFormatter(thousands)

def y_format():

    y_tick_form = ax.yaxis.set_major_formatter(format_y)

    return y_tick_form
# cuberoot converter

def cr_trans(x, inverse = False):

    if not inverse:

        return x ** (1/3)

    else:

        return x ** 3
# log converter

def log_trans(x, inverse = False):

    if not inverse:

        return np.log10(x)

    else:

        return np.power(10, x)
sf_data.describe()
# start with a histogram to see distribution

fig, ax = plt.subplots(figsize=[8, 5])

bins = np.arange(0, 797635.20 + 5000, 5000)

plt.hist(data = sf_data, x = 'total_comp', bins = bins)

plt.xlabel('Total Compensation ($)');

x_format();
# filtering out part-time work by total_comp and total_salary

part_time = sf_data[(sf_data.total_comp < 31200) | (sf_data.total_salary < 31200)].index
sf_data.drop(index = part_time, inplace = True)
# reset the index in order for it to be easier to work with later

sf_data.reset_index(inplace = True, drop = True)
# used to find the bins for total_comp log

log_trans(sf_data.total_comp.describe())
fig, ax = plt.subplots(figsize = [9, 6])

x_ticks = [50000, 70000, 90000, 110000, 150000, 200000, 300000]

bins = 10 ** np.arange(4.50, 5.90 + 0.01, 0.01)

plt.hist(data = sf_data, x = 'total_comp', bins = bins)

plt.xlabel('Total Compensation ($)');

plt.xscale('log');

x_format();

plt.xticks(x_ticks);

plt.xlim(32100, 400000);
sf_data.describe()
# showing total salary with standard plot

fig, ax = plt.subplots(figsize = [8, 5])

bins = np.arange(31200, 641374.64 + 5000, 5000)

plt.hist(data = sf_data, x = "total_salary", bins = bins);

plt.xlabel('Total Salary ($)');

plt.xticks(rotation = 15);

x_format();
# did a search for total_salary above 400K because that is where the bars above die off. 

high_outliers = sf_data[(sf_data.total_salary > 400000)]

high_outliers.sort_values(by = 'total_salary',ascending = False).head()
# investigating Police Officer 3 since that doesnt match high positions like the others

pol_3 = sf_data[sf_data.job == "Police Officer 3"]

pol_3.total_salary.sort_values(ascending = False).head()
# looking at the lower end as well

pol_3.total_salary.sort_values(ascending = False).tail()
# dropped high police officer 3 entry

irr_sal = sf_data[sf_data.index == 215652]
# found index of entry

irr_sal_index = irr_sal.index
# dropped in place

sf_data.drop(index = irr_sal_index, inplace = True)
# reset index

sf_data.reset_index(inplace = True, drop = True)
# used to find bins for total_salary log

np.log10(sf_data.total_salary.describe())
# both distributions are gonna take on a log-scale transformaton 

# added x limits in zoom in on distributions

fig, ax = plt.subplots(figsize = [8, 5])

x_ticks = [30000, 50000, 80000, 110000, 150000, 200000, 300000]

bins = 10 ** np.arange(4.49, 5.73 + 0.01, 0.01)

plt.hist(data = sf_data, x = "total_salary", bins = bins);

plt.xlabel('Total Salary ($)');

plt.xscale('log');

x_format();

plt.xticks(x_ticks);

plt.xlim(30000, 400000);
sf_data.total_benefit.describe()
# showing total benefits with standard plot

fig, ax = plt.subplots(figsize = [8, 5])

bins = np.arange(0, 151681.38 + 2000, 2000)

plt.hist(data = sf_data, x = "total_benefit", bins = bins);

plt.xlabel('Total Benefits ($)');

x_format();
high_outliers = sf_data[sf_data['total_benefit'] > 100000]

high_outliers.sort_values(by = 'total_benefit', ascending = False).head()
low_outliers = sf_data[sf_data['total_benefit'] == 0]

low_outliers.sample(5)
# used to find the bins

np.cbrt(sf_data.describe())
# needed to get the cube root of total_benefits isolated

tot_bene = sf_data.total_benefit

cube_tot_bene = np.cbrt(tot_bene)
fig, ax = plt.subplots(figsize = [8, 5])

x_ticks = [10, 15, 20, 25, 30, 35, 40, 45, 50]

tick_labels = ['1K', '3K', '8K', '15K', '27K', '43K', '64K', '91K', '125K']

bins = np.arange(0, 53.33 + 0.25, 0.25)

plt.hist(x = cube_tot_bene, bins = bins);

plt.xlabel('Total Benefits ($)');

plt.xticks(x_ticks, tick_labels);

plt.xlim(10, 50);
sf_data.describe()
plt.figure(figsize=[15,5])



# base_salary

bins = np.arange(0, 537847.86 + 5000, 5000)

plt.subplot(1, 3, 1)

plt.hist(data = sf_data, x = "base_salary", bins = bins);

plt.xlabel('Salary ($)');

plt.xticks(rotation = 15)



# overtime

bins2 = np.arange(0, 304546.25 + 5000, 5000)

plt.subplot(1, 3, 2)

plt.hist(data = sf_data, x = "overtime", bins = bins2);

plt.xlabel('Overtime ($)');



# other_salary

bins3 = np.arange(0, 342802.63 + 5000, 5000)

plt.subplot(1, 3, 3)

plt.hist(data = sf_data, x = "other_salary", bins = bins3);

plt.xlabel('Other Salary ($)');

plt.xticks(rotation = 15);
high_outliers = ((sf_data['base_salary'] > 250000) | (sf_data['overtime'] > 60000) | (sf_data['other_salary'] > 40000))
sf_data[high_outliers].sort_values(by = ['base_salary'], ascending = False).head(10)
sf_data[high_outliers].sort_values(by = 'overtime', ascending= False).head()
sf_data[high_outliers].sort_values(by = 'other_salary', ascending= False).head()
base_outliers = sf_data[sf_data.base_salary > 250000]

base_outliers.sort_values(ascending = False, by = ['job','base_salary']).head()
sf_data[sf_data.job == "Transit Operator"].base_salary.sort_values(ascending = False).head()
sf_data[sf_data.job == "Transit Operator"].base_salary.sort_values(ascending = False).tail()
irr_base_sal = sf_data[(sf_data.index == 215628)]
irr_base_sal_index = irr_base_sal.index
# dropped these in place

sf_data.drop(index = irr_base_sal_index, inplace = True)
sf_data.reset_index(inplace = True, drop = True)
sf_data.describe()
low_outliers = (sf_data['base_salary'] < 20000)

sf_data[low_outliers].sample(5)
irregular_incomes = sf_data[((sf_data['base_salary']) <= (sf_data['other_salary'])) 

                            & ((sf_data['base_salary'] == 0))]

irregular_incomes.head()
index_irr_income = irregular_incomes.index
# dropped these in place

sf_data.drop(index = index_irr_income, inplace = True)
sf_data.reset_index(inplace = True, drop = True)
sf_data.describe()
# log bins for base_salary

np.log10(sf_data.base_salary.describe())
np.cbrt(sf_data.describe())
# setting data for overtime and other_salary cube root

cube_ot = np.cbrt(sf_data['overtime'])

cube_other_salary = np.cbrt(sf_data['other_salary'])
plt.figure(figsize=[15,6])



# base_salary log transformation

x_ticks = [15000, 20000, 30000, 50000, 70000, 110000, 170000, 250000]

bins = 10 ** np.arange(4.07, 5.73 + 0.01, 0.01)

plt.subplot(1, 3, 1)

plt.hist(data = sf_data, x = "base_salary", bins = bins);

plt.xlabel('Base Salary ($)');

plt.xscale('log')

plt.xticks(rotation = 15)

plt.xticks(x_ticks, ['15K', '20K', '30K', '50K', '70K', '110K', '170K', '250K'])





# overtime cuberoot transformation

x_ticks = [0, 10, 20, 30, 40, 50, 60]

labels = [0, '1K', '8K', '27K', '64K', '125K', '216K']

bins2 = np.arange(0, 67.28 + 1, 1)

plt.subplot(1, 3, 2)

plt.hist(x = cube_ot, bins = bins2);

plt.xlabel('Overtime ($)');

plt.xticks(x_ticks, labels);







# other_salary cuberoot transformation

x_ticks = [0, 10, 20, 30, 40, 50, 60, 70]

labels = [0, '1K', '8K', '27K', '64K', '125K', '216K', '343K']

bins3 = np.arange(0, 69.99 + 1, 1)

plt.subplot(1, 3, 3)

plt.hist(x = cube_other_salary, bins = bins3);

plt.xlabel('Other Salary ($)');

plt.xticks(x_ticks, labels);
sf_data[['retirement', 'health_dental', 'other_benefit' ]].describe()
plt.figure(figsize=[15,5])



# retirement

bins = np.arange(0, 105052.98 + 1000, 1000)

plt.subplot(1, 3, 1)

plt.hist(data = sf_data, x = "retirement", bins = bins);

plt.xlabel('Retirement ($)');





# health_dental

bins2 = np.arange(0, 36609.50 + 500, 500)

plt.subplot(1, 3, 2)

plt.hist(data = sf_data, x = "health_dental", bins = bins2);

plt.xlabel('Health & Dental ($)');





# other_benefit

bins3 = np.arange(0, 37198.60 + 500, 500)

plt.subplot(1, 3, 3)

plt.hist(data = sf_data, x = "other_benefit", bins = bins3);

plt.xlabel('Other Benefits ($)');

x_format();

plt.xticks(rotation = 0);
sf_data[sf_data['retirement'] == 0].sample(5)
# bins for retirement and other_benefit cube root

np.cbrt(sf_data[['retirement', 'other_benefit']].describe())
# setting data cube root transformation for both variables

cube_retire = np.cbrt(sf_data['retirement'])

cube_benefit = np.cbrt(sf_data['other_benefit'])
# applied transformations

plt.figure(figsize=[15,5])



# retirement

x_ticks = [0, 10, 20, 30, 40]

labels = [0, '1K', '8K', '27K', '64K']

bins = np.arange(0, 49.18 + 1, 1)

plt.subplot(1, 3, 1)

plt.hist(x = cube_retire, bins = bins);

plt.xlabel('Retirement ($)');

plt.xticks(x_ticks, labels);





# health_dental

bins2 = np.arange(0, 21715.08 + 500, 500)

plt.subplot(1, 3, 2)

plt.hist(data = sf_data, x = "health_dental", bins = bins2);

plt.xlabel('Health & Dental ($)');





# other_benefit

x_ticks = [0, 5, 10, 15, 20, 25, 30]

labels = [0, '125', '1K', '3K', '8K', '16K', '27K']

bins3 = np.arange(0, 33.38 + 1, 1)

plt.subplot(1, 3, 3)

plt.hist(x = cube_benefit, bins = bins3);

plt.xlabel('Other Benefits ($)');

plt.xticks(x_ticks, labels);



sf_data.org_group.value_counts()
# made these category names easier to work with

org_series = sf_data.org_group

org_series.replace({'Public Works, Transportation & Commerce': 'Pub Wrks, Tran & Comm.',

                    'Community Health': 'Comm. Health',

                    'Public Protection': 'Pub. Protect',

                    'Human Welfare & Neighborhood Development': 'Hmn Wlfr & Nbrhd Dev.',

                    'General Administration & Finance': 'Gen Admin & Fin.',

                    'Culture & Recreation': 'Culture & Rec.'}, inplace=True)
sf_data.org_group.value_counts()
group_cats = ['Pub Wrks, Tran & Comm.', 'Pub. Protect', 'Comm. Health',

              'Gen Admin & Fin.', 'Hmn Wlfr & Nbrhd Dev.', 'Culture & Rec.']

categories = pd.api.types.CategoricalDtype(ordered = True, categories = group_cats)

sf_data['org_group'] = sf_data['org_group'].astype(categories)
group_cats = sf_data.org_group.value_counts()

labels = group_cats.index
# Pie Chart shows distributions of Proportions of Employees in each Organizational Group

plt.figure(figsize= [6,5])

plt.pie(group_cats, autopct='%1.1f%%', labels = labels, pctdistance = 0.83);
# EXPLORING JOB

top_five_job = sf_data.job.value_counts().sort_values(ascending = False).head(5).index

top_five_job
# getting top jobs

top_five_by_job = (sf_data[sf_data.job.isin(top_five_job)].job.value_counts())

top_five_by_job
# countplot top five jobs

top_five_by_job.plot(kind='barh');
# exploring year

base_color = sb.color_palette()[0]

sb.countplot(data= sf_data, x = 'year', color = base_color);
data_clean = sf_data.copy()
# log columns

data_clean['log_salary'] = data_clean['base_salary'].apply(log_trans)

data_clean['log_total_comp'] = data_clean['total_comp'].apply(log_trans)

data_clean['log_total_salary'] = data_clean['total_salary'].apply(log_trans)

# cuberoot columns

data_clean['cr_overtime'] = data_clean['overtime'].apply(cr_trans)

data_clean['cr_other_salary'] = data_clean['other_salary'].apply(cr_trans)

data_clean['cr_retirement'] = data_clean['retirement'].apply(cr_trans)

data_clean['cr_other_benefit'] = data_clean['other_benefit'].apply(cr_trans)

data_clean['cr_total_benefit'] = data_clean['total_benefit'].apply(cr_trans)
# saving cleaned copy for explanatory analysis

data_clean.to_csv(path_or_buf='master_sf_salary.csv', index = False)
# take a sample of the dataset so the scatterplot will process quicker

np.random.seed(2018)

sample = np.random.choice(data_clean.shape[0], 2000, replace = False)

sf_subset = data_clean.iloc[sample]
# set continuous variables

variables = ['base_salary', 'overtime', 'other_salary', 'total_salary', 'retirement',

       'health_dental', 'other_benefit', 'total_benefit', 'total_comp']
#scatter plot

sb.set_context('talk')

sb.pairplot(data = sf_subset, vars = variables);
# heat map of continuous variables UNTRANSFORMED

sb.set_context('notebook')

plt.figure(figsize = [10,7])

sb.heatmap(data_clean[variables].corr(), annot = True, fmt = '.3f',

          cmap = 'vlag_r', center = 0);
# relplot `overtime` vs `other_benefit`

sb.relplot(data = sf_subset, x = 'overtime', y = 'other_benefit');

plt.xticks(rotation = 15);
sf_subset[['other_benefit', 'overtime', 'cr_other_benefit',

             'cr_overtime']].corr()
# relplot `base_salary` vs `overtime`

sb.relplot(data = sf_subset, x = 'base_salary', y = 'overtime');

plt.xticks(rotation = 15);
sf_subset[['base_salary', 'overtime', 'log_salary',

             'cr_overtime']].corr()
# relplot `other_salary` vs `other_benefit`

sb.relplot(data = sf_subset, x = 'other_salary', y = 'other_benefit');

plt.xticks(rotation = 15);
sf_subset[['other_salary', 'other_benefit', 'cr_other_salary',

             'cr_other_benefit']].corr()
fig, ax = plt.subplots(figsize = [8, 5])

x_ticks = log_trans(np.array([30000, 45000, 65000, 100000, 160000, 250000, 400000, 630000]))

sb.violinplot(data = sf_subset, y = 'org_group',

              x = 'log_total_comp', color = base_color)

ax.tick_params(axis = 'both', which = 'major',labelsize = 10);

plt.xlabel('Total Comp ($)')

plt.ylabel('Organizational Group');

plt.xticks(x_ticks, ['30K', '45K', '65K', '100K',

                     '160K', '250K', '400K', '630K' ]);
# to get job subset

job_subset = (sf_subset[sf_subset.job.isin(top_five_job)])
# BY JOB BY TOTAL_COMP

fig, ax = plt.subplots(figsize = [7, 5])

sb.violinplot(data = job_subset, y = 'job', x = 'log_total_comp', color = base_color)

ax.tick_params(axis = 'both', which = 'major',labelsize = 10)

plt.xticks(log_trans(np.array([30000, 40000, 60000, 100000, 150000, 250000, 400000])),

          ['30K', '40K', '60K', '100K', '150K', '250K', '400K'])

plt.xlabel('Total Comp ($)')

plt.ylabel('Top Five Jobs');
job_sub = data_clean[data_clean['job'].isin(top_five_job)]

g = sb.FacetGrid(job_sub, col = 'job', height = 3, aspect = 1.5,

                 col_wrap = 3, legend_out = True)

g.map(plt.hist, 'log_total_comp', alpha = 0.9);

g.add_legend();
# salary variables

mean_org_sal = data_clean.groupby('org_group')[['base_salary', 'overtime', 'other_salary']].agg('mean')

mean_org_sal
mean_org_sal.plot(kind='bar');
# found percentage to see proportions

salary_percents = mean_org_sal.div(mean_org_sal.sum(1), axis = 0)

salary_percents
# charted proportions

salary_percents.plot(kind = 'barh', stacked = True);

plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);

plt.ylabel('Organizational Group');
# benefits Variables

mean_org_bene = data_clean.groupby('org_group')[['retirement',

                                              'health_dental',

                                              'other_benefit']].agg('mean')

# found percentage to see proportions

benefit_percents = mean_org_bene.div(mean_org_bene.sum(1), axis = 0)

benefit_percents
# charted proportions

benefit_percents.plot(kind = 'barh', stacked = True);

plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);

plt.ylabel('Organizational Group');
# ALL Variables

mean_org_total = data_clean.groupby('org_group')[['base_salary',

                                              'overtime',

                                              'other_salary',

                                              'retirement',

                                              'health_dental',

                                              'other_benefit']].agg('mean')

# found percentage to see proportions

total_percents = mean_org_total.div(mean_org_total.sum(1), axis = 0)

total_percents
# charted proportions

total_percents.plot(kind = 'barh', stacked = True);

plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);

plt.ylabel('Organizational Group');
# salary variables

mean_job_sal = job_sub.groupby('job')[['base_salary',

                                       'overtime',

                                       'other_salary']].agg('mean')

# found percentage to see proportions

sal_percent_job = mean_job_sal.div(mean_job_sal.sum(1), axis = 0)

sal_percent_job
# charted proportions

sal_percent_job.plot(kind = 'barh', stacked = True);

plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);

plt.ylabel('Top Five Jobs');
# salary variables

mean_job_bene = job_sub.groupby('job')[['retirement',

                                        'health_dental',

                                        'other_benefit']].agg('mean')

# found percentage to see proportions

bene_percent_job = mean_job_bene.div(mean_job_bene.sum(1), axis = 0)

bene_percent_job
# charted proportions

bene_percent_job.plot(kind = 'barh', stacked = True);

plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);

plt.ylabel('Top Five Jobs');
# ALL variables

mean_job_total = job_sub.groupby('job')[['base_salary', 

                                         'overtime', 

                                         'other_salary',

                                         'retirement',

                                         'health_dental',

                                         'other_benefit']].agg('mean')

# found percentage to see proportions

sal_percent_total = mean_job_total.div(mean_job_total.sum(1), axis = 0)

sal_percent_total
# charted proportions

sal_percent_total.plot(kind = 'barh', stacked = True);

plt.legend(bbox_to_anchor=(1.05, 1), loc = 'upper left', borderaxespad = 0.);

plt.ylabel('Top Five Jobs');
g = sb.FacetGrid(data = job_sub, hue = 'job', hue_order = top_five_job,

                height = 10, aspect = 1.5)

g.map(sb.scatterplot, 'total_benefit',

      'log_total_salary').set(yticks=log_trans(np.array([30000, 40000, 60000,

                                                         100000, 150000, 

                                                         250000, 400000])));

g.add_legend(title='Jobs');

g.set_ylabels('Total Salary ($)');

g.set_yticklabels(['30K', '40K', '60K', '100K', '150K', '250K', '400K'])

g.set_xticklabels([0, '0k', '10K', '20K', '30K', '40K', '50K', '60K', '70K'])

g.set_xlabels('Total Benefits ($)');
g = sb.FacetGrid(data = job_sub, col = 'job',

                 hue = 'job', height = 4, aspect = 1.25, hue_order = top_five_job)

g.map(sb.scatterplot, 'total_benefit', 'log_total_salary');

g.add_legend();
sb.pairplot(job_sub, hue = 'job', vars = ['log_total_salary', 'total_benefit', 'log_total_comp'],

            hue_order = top_five_job);