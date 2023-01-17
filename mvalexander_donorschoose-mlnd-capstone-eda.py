# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
train_test_dtypes = {'id':str, 'teacher_id':str, 'teacher_prefix':str, 'school_state':str, 'project_submitted_datetime':str, 'project_grade_category':str, 'project_subject_categories':str,
                     'project_subject_subcategories':str, 'project_title':str, 'project_essay_1':str, 'project_essay_2':str, 'project_essay_3':str, 'project_essay_4':str, 'project_resource_summary':str, 
                     'teacher_number_of_previously_posted_projects':int, 'project_is_approved':int}
train_data_raw = pd.read_csv('../input/train.csv', sep=',', dtype=train_test_dtypes, low_memory=True)
test_data_raw = pd.read_csv('../input/test.csv', sep=',', dtype=train_test_dtypes, low_memory=True)
resource_data_raw = pd.read_csv('../input/resources.csv', sep=',')
train_data_raw.info()
test_data_raw.info()
resource_data_raw.info()
cat_features = ['project_grade_category', 'project_subject_categories', 'project_subject_subcategories', 'teacher_prefix', 'school_state']
text_features = ['project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'project_resource_summary']
temp_text_features1 = ['project_title', 'project_essay_1', 'project_essay_2']
temp_text_features2 = ['project_essay_3', 'project_essay_4', 'project_resource_summary']
train_data_raw[cat_features].head(2)
resource_data_raw.head(2)
train_data_raw[text_features].head(2)
train_data_raw['year'] = train_data_raw.project_submitted_datetime.apply(lambda x: x.split("-")[0])
train_data_raw['month'] = train_data_raw.project_submitted_datetime.apply(lambda x: x.split("-")[1])
train_data_raw['project_submitted_datetime'] = pd.to_datetime(train_data_raw['project_submitted_datetime'], format="%Y-%m-%d %H:%M:%S")
test_data_raw['year'] = test_data_raw.project_submitted_datetime.apply(lambda x: x.split("-")[0])
test_data_raw['month'] = test_data_raw.project_submitted_datetime.apply(lambda x: x.split("-")[1])
test_data_raw['project_submitted_datetime'] = pd.to_datetime(test_data_raw['project_submitted_datetime'], format="%Y-%m-%d %H:%M:%S")
sns.distplot(train_data_raw.project_submitted_datetime.dt.month, kde=False, bins=12)
plt.title("Dist plot of Months in which projects are submitted");
print(train_data_raw['project_is_approved'].value_counts())
print("\nPercentage of proposals approved = {}%".format(train_data_raw['project_is_approved'].value_counts()[1] / len(train_data_raw['project_is_approved'])))
plt.figure(figsize=(10,3));
plt.title('Project approval imbalance in training data')
sns.countplot(x=train_data_raw['project_is_approved']);
resource_data_raw['total'] = resource_data_raw['quantity'] * resource_data_raw['price']
resource_data_raw.head()
totals_by_prop_id = resource_data_raw[['id', 'total']].groupby('id').total.agg(sum)
print("Max proposal amount request: {}".format(totals_by_prop_id.max()))
print("Min proposal amount request: {}".format(totals_by_prop_id.min()))
print("Avg proposal amount request: {}".format(totals_by_prop_id.mean()))
print("Median proposal amount request: {}".format(totals_by_prop_id.median()))
resource_data_raw[['id', 'price']].groupby('id').max().max()
res = resource_data_raw[['id', 'total']].groupby('id').total.agg(\
    [
        'count', 
        'sum', 
        'min', 
        'max', 
        'mean', 
        'median',
        'std',
    ]).reset_index()
print(res.head())
train_data_raw = train_data_raw.merge(res, on='id')
test_data_raw = test_data_raw.merge(res, on='id')
train_data_raw[train_data_raw.isnull().any(axis=1)].head(2)
values = {'std': 0.0}
train_data_raw.fillna(value=values, inplace=True)
test_data_raw.fillna(value=values, inplace=True)
print('Max sum requested for rejected and approved proposals.')
train_data_raw[['project_is_approved', 'sum']].groupby('project_is_approved').max().rename(columns={'sum':'max'})
print('Average sum requested for rejected and approved proposals.')
train_data_raw[['project_is_approved', 'sum']].groupby('project_is_approved').mean().rename(columns={'sum':'average'})
print('Max number of previous proposals: approved vs. rejected')
train_data_raw[['project_is_approved','teacher_number_of_previously_posted_projects']].groupby('project_is_approved').max().rename(columns={'teacher_number_of_previously_posted_projects':'teacher_number_of_previously_posted_projects (max)'})
print('Average number of previous proposals: approved vs. rejected')
train_data_raw[['project_is_approved','teacher_number_of_previously_posted_projects']].groupby('project_is_approved').mean().rename(columns={'teacher_number_of_previously_posted_projects':'teacher_number_of_previously_posted_projects (avg)'})
teacher_max_number_of_previous_proposals = train_data_raw[['teacher_id', 'teacher_number_of_previously_posted_projects']].groupby('teacher_id').teacher_number_of_previously_posted_projects.agg(max)
print("Highest number of previous proposals: {}".format(teacher_max_number_of_previous_proposals.max()))
print("Lowest number of previous proposals: {}".format(teacher_max_number_of_previous_proposals.min()))
print("Avg number of previous proposals: {}".format(teacher_max_number_of_previous_proposals.mean()))
print("Median number of previous proposals: {}".format(teacher_max_number_of_previous_proposals.median()))

teacher_number_previous_submissions_mean = train_data_raw[['teacher_number_of_previously_posted_projects', 'project_is_approved']].groupby('teacher_number_of_previously_posted_projects').mean()
#sns.distplot(teacher_number_previous_submissions_mean, kde=True, bins=15);
sns.distplot(teacher_number_previous_submissions_mean['project_is_approved'], bins=15)
sns.despine()
plt.yticks([])
plt.xticks([])

plt.ylabel('Approval Rate');
plt.xlabel('Number of submissions')
plt.title('Approval rates as number of previously submitted proposals increases');
train_data_raw.dropna(subset=['teacher_prefix'], inplace=True)
values = {'teacher_prefix': 'Teacher'}
test_data_raw.fillna(value=values, inplace=True)
plt.figure(figsize=(30,2));
plt.title('Histogram of proposals submitted by state')
sns.countplot(x=train_data_raw['school_state'], order=train_data_raw['school_state'].value_counts().index, hue=train_data_raw['project_is_approved'])
train_data_raw[['school_state', 'project_is_approved']].groupby('school_state').mean().sort_values(by='project_is_approved', ascending=False).plot.bar(figsize=(30,2), grid=True, title='Approval Rates by school_state');
plt.figure(figsize=(30,2))
sns.countplot(x=train_data_raw['project_subject_categories'], hue=train_data_raw['project_is_approved'], order=train_data_raw['project_subject_categories'].value_counts().index);
plt.xticks(rotation=90);
plt.title('Histogram of proposals submitted by project_subject_categories')
train_data_raw[['project_subject_categories', 'project_is_approved']].groupby('project_subject_categories').mean().sort_values(by='project_is_approved', ascending=False).plot.bar(figsize=(30,2), grid=True, title='Approval Rates by project_subject_categories');
train_data_raw[['project_subject_categories', 'project_is_approved']].groupby('project_subject_categories')['project_is_approved'].agg(['mean','count']).sort_values('mean', ascending=False).tail(5)
plt.figure(figsize=(30,2))
sns.countplot(x=train_data_raw['project_subject_subcategories'], hue=train_data_raw['project_is_approved'], order=train_data_raw['project_subject_subcategories'].value_counts().iloc[:30].index);
plt.xticks(rotation=90);
plt.title('Histogram of proposals submitted by project_subject_subcategories (30 highest)')
train_data_raw[['project_subject_subcategories', 'project_is_approved']].groupby('project_subject_subcategories').mean().sort_values(by='project_is_approved', ascending=False).plot.bar(figsize=(30,2), grid=True, title='Approval Rates by project_subject_subcategories (X labels not shown)');
plt.axis('off')
plt.show()
train_data_raw[['project_subject_subcategories', 'project_is_approved']].groupby('project_subject_subcategories')['project_is_approved'].agg(['mean','count']).sort_values('mean', ascending=False).head(10)
train_data_raw[['project_subject_subcategories', 'project_is_approved']].groupby('project_subject_subcategories')['project_is_approved'].agg(['mean','count']).sort_values('mean', ascending=False).tail(10)
plt.figure(figsize=(30,2))
sns.countplot(x=train_data_raw['project_grade_category'], hue=train_data_raw['project_is_approved'], order=train_data_raw['project_grade_category'].value_counts().index);
plt.title('Histogram of proposals submitted by project_grade_category')
train_data_raw[['project_grade_category', 'project_is_approved']].groupby('project_grade_category').mean().sort_values(by='project_is_approved', ascending=False).plot.bar(figsize=(30,2), grid=True, title='Approval Rates by project_grade_category');
plt.figure(figsize=(30,2))
sns.countplot(x=train_data_raw['teacher_prefix'], hue=train_data_raw['project_is_approved'], order=train_data_raw['teacher_prefix'].value_counts().index);
plt.title('Histogram of proposals submitted by teacher_prefix')
train_data_raw[['teacher_prefix', 'project_is_approved']].groupby('teacher_prefix').mean().sort_values(by='project_is_approved', ascending=False).plot.bar(figsize=(30,2), grid=True, title='Approval Rates by teacher_prefix');

resource_data_raw.info()
resource_data_raw[resource_data_raw.isnull().any(axis=1)].head()
resource_data_raw.fillna('', inplace=True)
resource_data_raw.info()
pivot_table = resource_data_raw.groupby('id').description.apply(lambda x: "%s" % ';'.join(x)).reset_index()
train_data_raw = train_data_raw.merge(pivot_table, on='id')
test_data_raw = test_data_raw.merge(pivot_table, on='id')
essay_3_4_nonull_filter = train_data_raw.project_essay_3.notnull()

train_data_raw.loc[essay_3_4_nonull_filter,'project_essay_1'] = train_data_raw[essay_3_4_nonull_filter].project_essay_1.str.cat(train_data_raw[essay_3_4_nonull_filter].project_essay_2)
train_data_raw.loc[essay_3_4_nonull_filter, 'project_essay_2'] = train_data_raw[essay_3_4_nonull_filter].project_essay_3.str.cat(train_data_raw[essay_3_4_nonull_filter].project_essay_4)

train_data_raw.drop(['project_essay_3', 'project_essay_4'], axis=1, inplace=True)

test_essay_3_4_nonull_filter = test_data_raw.project_essay_3.notnull()

test_data_raw.loc[test_essay_3_4_nonull_filter,'project_essay_1'] = test_data_raw[test_essay_3_4_nonull_filter].project_essay_1.str.cat(test_data_raw[test_essay_3_4_nonull_filter].project_essay_2)
test_data_raw.loc[test_essay_3_4_nonull_filter, 'project_essay_2'] = test_data_raw[test_essay_3_4_nonull_filter].project_essay_3.str.cat(test_data_raw[test_essay_3_4_nonull_filter].project_essay_4)

test_data_raw.drop(['project_essay_3', 'project_essay_4'], axis=1, inplace=True)
train_data_raw.info()
text_features_final = ['project_title', 'project_essay_1', 'project_essay_2', 'project_resource_summary', 'description']

str_df_approved = pd.DataFrame()
for col in text_features_final:
    str_df_approved[col] = train_data_raw[train_data_raw.project_is_approved==1][col].str.len()
    
str_df_rejected = pd.DataFrame()
for col in text_features_final:
    str_df_rejected[col] = train_data_raw[train_data_raw.project_is_approved==0][col].str.len()
str_stats_approved_df = pd.DataFrame()
for col in str_df_approved:
    str_stats_approved_df[col] = str_df_approved[col].agg(['min', 'max', 'mean'])

str_stats_rejected_df = pd.DataFrame()
for col in str_df_rejected:
    str_stats_rejected_df[col] = str_df_rejected[col].agg(['min', 'max', 'mean'])
str_stats_approved_df
str_stats_rejected_df
train_data_raw[['project_is_approved', 'project_title']].groupby('project_is_approved').describe()
from sklearn.feature_extraction.text import CountVectorizer
corpus = train_data_raw.project_essay_1
vec = CountVectorizer(stop_words='english').fit(corpus)
bag_of_words = vec.transform(corpus)
sum_words = bag_of_words.sum(axis=0) 
full_words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
full_words_freq = sorted(full_words_freq, key = lambda x: x[1], reverse=True)
corpus = train_data_raw[train_data_raw.project_is_approved==1].project_essay_1
vec = CountVectorizer(stop_words='english').fit(corpus)
bag_of_words = vec.transform(corpus)
sum_words = bag_of_words.sum(axis=0) 
approved_words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
approved_words_freq = sorted(approved_words_freq, key = lambda x: x[1], reverse=True)
corpus = train_data_raw[train_data_raw.project_is_approved==0].project_essay_1
vec = CountVectorizer(stop_words='english').fit(corpus)
bag_of_words = vec.transform(corpus)
sum_words = bag_of_words.sum(axis=0) 
rejected_words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
rejected_words_freq = sorted(rejected_words_freq, key = lambda x: x[1], reverse=True)
full_list_x = [x for x,y in full_words_freq[0:20]]
full_list_y = [y for x,y in full_words_freq[0:20]]

approved_list_x = [x for x,y in approved_words_freq[0:20]]
approved_list_y = [y for x,y in approved_words_freq[0:20]]

rejected_list_x = [x for x,y in rejected_words_freq[0:20]]
rejected_list_y = [y for x,y in rejected_words_freq[0:20]]
#plt.figure(figsize=(40,10))
sns.set(font_scale=2);
f, (ax2, ax3) = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(40,10));
#sns.barplot(x=full_list_y, y=full_list_x, ax=ax1);
#ax1.set(xlabel='Word densities for ')
sns.barplot(x=approved_list_y, y=approved_list_x, ax=ax2);
ax2.set(xlabel="Essay1 top word frequencies in approved proposals");
sns.barplot(x=rejected_list_y, y=rejected_list_x, ax=ax3);
ax3.set(xlabel="Essay1 top word frequencies in rejected proposals");
plt.suptitle('Most frequent words in essay1');
str_df_charachters = pd.DataFrame()
str_df_charachters['project_is_approved'] = train_data_raw.project_is_approved
for col in text_features_final:
    str_df_charachters[col] = train_data_raw[col].str.len()
str_df_num_words = pd.DataFrame()
str_df_num_words['project_is_approved'] = train_data_raw.project_is_approved
for col in text_features_final:
    str_df_num_words[col] = train_data_raw[col].str.split().str.len()
str_df_word_density = pd.DataFrame()
str_df_word_density['project_is_approved'] = train_data_raw.project_is_approved
for col in text_features_final:
    str_df_word_density[col] =  str_df_num_words[col] / str_df_charachters[col]
str_df_word_density.head()
str_df_word_density.groupby('project_is_approved').agg(['min', 'max', 'mean'])
fig, axes = plt.subplots();
axes.violinplot(dataset=[str_df_word_density[str_df_word_density.project_is_approved==1]['project_title'], str_df_word_density[str_df_word_density.project_is_approved==0]['project_title']]);
axes.set_title('project_title Word Densities');
axes.yaxis.grid(True);
axes.set_xlabel('Rejected/Approved');
axes.set_ylabel('');
axes.set_xticklabels([]);
fig, axes = plt.subplots();
axes.violinplot(dataset=[str_df_word_density[str_df_word_density.project_is_approved==1]['project_essay_1'], str_df_word_density[str_df_word_density.project_is_approved==0]['project_essay_1']]);
axes.set_title('project_essay_1 Word Densities');
axes.yaxis.grid(True);
axes.set_xlabel('Project is approved');
axes.set_ylabel('snark');
fig, axes = plt.subplots();
axes.violinplot(dataset=[str_df_word_density[str_df_word_density.project_is_approved==1]['project_essay_2'], str_df_word_density[str_df_word_density.project_is_approved==0]['project_essay_2']]);
axes.set_title('project_essay_2 Word Densities');
axes.yaxis.grid(True);
axes.set_xlabel('Rejected/Approved');
axes.set_ylabel('');
axes.set_xticklabels([]);
fig, axes = plt.subplots();
axes.violinplot(dataset=[str_df_word_density[str_df_word_density.project_is_approved==1]['project_resource_summary'], str_df_word_density[str_df_word_density.project_is_approved==0]['project_resource_summary']]);
axes.set_title('project_resource_summary Word Densities');
axes.yaxis.grid(True);
axes.set_xlabel('Rejected/Approved');
axes.set_ylabel('');
axes.set_xticklabels([]);
fig, axes = plt.subplots();
axes.violinplot(dataset=[str_df_word_density[str_df_word_density.project_is_approved==1]['description'], str_df_word_density[str_df_word_density.project_is_approved==0]['description']]);
axes.set_title('description Word Densities');
axes.yaxis.grid(True);
axes.set_xlabel('Rejected/Approved');
axes.set_ylabel('');
axes.set_xticklabels([]);
fig, axes = plt.subplots();
axes.violinplot(dataset=[np.log(train_data_raw[train_data_raw.project_is_approved==0]['mean']), np.log(train_data_raw[train_data_raw.project_is_approved==1]['mean'])]);
axes.set_title('Project price means');
axes.yaxis.grid(True);
axes.set_xlabel('Rejected/Approved');
axes.set_ylabel('');
axes.set_xticklabels([]);
approval_rates_by_month = train_data_raw[['project_is_approved', 'month']].groupby('month').mean().reset_index()
approval_rates_by_month
ax1 = sns.barplot(x=approval_rates_by_month.month, y=approval_rates_by_month.project_is_approved)
ax1.set(xlabel='Months', ylabel='Approval Rates');
approval_rates_by_subcategory = train_data_raw[['project_subject_subcategories', 'project_is_approved']].groupby('project_subject_subcategories').mean().reset_index()
submission_counts_by_subcategory = train_data_raw[['project_subject_subcategories', 'project_is_approved']].groupby('project_subject_subcategories').count().reset_index()
subcategory_df = approval_rates_by_subcategory.merge(submission_counts_by_subcategory, on='project_subject_subcategories')
subcategory_df = subcategory_df.sort_values(by='project_is_approved_x', ascending=True)
plt.figure(figsize=(30,10));
ax1 = sns.barplot(x=subcategory_df.project_is_approved_x, y=subcategory_df.project_is_approved_y, palette='dark');
plt.ylim(0,2000)
labels = [item.get_text() for item in ax1.get_xticklabels()]
for i in range(len(labels)):
    labels[i]=''
labels[0]=0
labels[-1] = 1
halfwaythere = len(labels)//2
labels[halfwaythere] = .5
ax1.set_xticklabels(labels);
ax1.set(ylabel='Number of submissions', xlabel='Approval Rates')

ax1.set_title('Number of submissions per approval rate');
axes.yaxis.grid(True);
plt.figure(figsize=(30,10));
axes = sns.distplot(subcategory_df.project_is_approved_y, bins=500);
axes.set(xlabel='Number of proposal submissions', ylabel='');
plt.figure(figsize=(30,20))
g = sns.countplot(x=train_data_raw['project_subject_subcategories'], order=train_data_raw['project_subject_subcategories'].value_counts().index);
#g.set_yscale('log')
g.set(xlabel='Project Subcategories, sorted by number of submissions (407 subcategory labels turned off)', ylabel='Number of proposal submissions')
plt.xticks([]);
plt.title('Histogram of number of proposals submitted by project_subject_subcategories')
print("Categories with 10 lowest approval rates, and the number of submissions for those subcategories")
subcategory_df[:10]
print("Categories with 50 highest approval rates, and the number of submissions for those subcategories")
subcategory_df[-50:]
len(subcategory_df)
subcategory_base_values = sorted(train_data_raw[~train_data_raw.project_subject_subcategories.str.contains(',')].project_subject_subcategories.unique(), key=len)
import operator

subcat_dict = {}
for subcat in subcategory_base_values:
    subcat_dict[subcat] = train_data_raw[train_data_raw.project_subject_subcategories.str.contains(subcat)][['project_is_approved', 'project_subject_subcategories']].project_is_approved.count()                     

subcat_list = sorted(subcat_dict.items(), key=operator.itemgetter(1))         
#train_data_raw[['school_state', 'project_is_approved']].groupby('school_state').mean()
#subcat_df = pd.DataFrame.from_dict(subcat_dict, orient='index').reset_index()
#subcat_df.columns=['project_subject_subcategories', 'num_submissions']

subcat_approval_rate_dict = {}
for subcat in subcategory_base_values:
    subcat_approval_rate_dict[subcat] = train_data_raw[train_data_raw.project_subject_subcategories.str.contains(subcat)][['project_is_approved', 'project_subject_subcategories']].project_is_approved.mean()                     

subcat_approval_rate_list = sorted(subcat_approval_rate_dict.items(), key=operator.itemgetter(1)) 
subcat_df = pd.DataFrame(subcat_list, columns=['project_subject_subcategories', 'num_submissions'])
subcat_df
plt.figure(figsize=(30,5))
g = sns.barplot(x=subcat_df.project_subject_subcategories, y=subcat_df.num_submissions);
g.set(xlabel='Subcategories', ylabel='Number of submissions');
g.set_xticklabels(g.get_xticklabels(), rotation=9);
plt.title('Number of submissions per base category in project_subject_subcategories');
subcat_approval_rate_df = pd.DataFrame(subcat_approval_rate_list, columns=['project_subject_subcategories', 'approval_rates'])
subcat_approval_rate_df
plt.figure(figsize=(30,5));
g = sns.barplot(x=subcat_approval_rate_df.project_subject_subcategories, y=subcat_approval_rate_df.approval_rates);
g.set(xlabel='Subcategories', ylabel='Approval Rates');
g.set_xticklabels(g.get_xticklabels(), rotation=90);
plt.title('Approval rate per base category in project_subject_subcategories');
subcat_df = subcat_df.merge(subcat_approval_rate_df, on='project_subject_subcategories')
subcat_df
