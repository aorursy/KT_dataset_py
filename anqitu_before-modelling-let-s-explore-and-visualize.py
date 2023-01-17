import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
data_dir = '../input/'

# Load Data
dtype = {
    'id': str,
    'teacher_id': str,
    'teacher_prefix': str,
    'school_state': str,
    'project_submitted_datetime': str,
    'project_grade_category': str,
    'project_subject_categories': str,
    'project_subject_subcategories': str,
    'project_title': str,
    'project_essay_1': str,
    'project_essay_2': str,
    'project_essay_3': str,
    'project_essay_4': str,
    'project_resource_summary': str,
    'teacher_number_of_previously_posted_projects': int,
    'project_is_approved': np.uint8}

resources_df = pd.read_csv(data_dir + 'resources.csv')
test_df = pd.read_csv(data_dir + 'test.csv', dtype=dtype, low_memory=True)
train_df = pd.read_csv(data_dir + 'train.csv', dtype=dtype, low_memory=True)

Full_df = pd.concat([test_df,train_df])
print('train_df shape:' + str(train_df.shape))
print('test_df shape:' + str(test_df.shape))
print('Full_df shape:' + str(Full_df.shape))
print('resources_df shape:' + str(Full_df.shape))
print('Full_df info:')
Full_df.info()

print('-'*60)

print('resources_df info:')
Full_df.info()
print('Full_df null:')
print(Full_df.isnull().sum())

print('-'*60)

print('resources_df null:')
print(resources_df.isnull().sum())
Full_df.head()
Full_df.describe(include = 'all')
resources_df.head()
resources_df.describe(include = 'all')
# Converting project_submitted_datetime from object to date time
Full_df['project_submitted_datetime']= pd.to_datetime(Full_df['project_submitted_datetime'])

# The later essay 1 question is similar to previous essay 1&2
# The later essay 2 question is similar to previous essay 3&4
# We can combine them
Full_df['project_essay_full'] = Full_df['project_essay_1'].map(str) + ' ' + Full_df['project_essay_2'].map(str) + ' ' + Full_df['project_essay_3'].map(str) + ' ' + Full_df['project_essay_4'].map(str)
Full_df = Full_df.drop(columns = ['project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4'])
Full_df.groupby('teacher_prefix',as_index=False)['id'].count().sort_values('id')
Full_df['teacher_prefix'] = Full_df['teacher_prefix'].fillna('Mrs.')
train_df = Full_df[pd.notna(Full_df['project_is_approved'])]
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.patches as mpatches
import seaborn as sns

%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')

import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
def get_barplot(df, factor):
    rate_df = df.groupby(factor)['project_is_approved'].mean()
    print((1-rate_df).iplot(kind='bar', xTitle=factor, title='project rejected rate'))
#     count_df = df.groupby(factor)['project_is_approved'].count()
#     print(count_df.iplot(kind='bar', xTitle=factor, title='project count'))
cat_df = train_df[['project_is_approved','teacher_id', 'id']]
approved_df = cat_df.groupby('teacher_id',as_index=False)['project_is_approved'].sum()
submitted_df = cat_df.groupby('teacher_id')['project_is_approved'].count().reset_index()
approved_df = approved_df.rename(columns = {'project_is_approved': 'teacher_projects_approved_sum'})
submitted_df = submitted_df.rename(columns = {'project_is_approved': 'teacher_projects_count'})
cat_df = cat_df.merge(approved_df)
cat_df = cat_df.merge(submitted_df)

# approval rate for the teacher for other projects
cat_df['teacher_projects_approved_rate'] = (cat_df['teacher_projects_approved_sum'] - cat_df['project_is_approved']) / (cat_df['teacher_projects_count'] - 1)
teacher_rate_mean = cat_df['teacher_projects_approved_rate'].mean()
cat_df['teacher_projects_approved_rate'] = cat_df['teacher_projects_approved_rate'].fillna(teacher_rate_mean)

# only consider those submitting more than 5 projects 
more_than_5_df = cat_df[cat_df['teacher_projects_count'] > 5]
more_than_5_df['teacher_projects_approved_rate' + '_round_1dp_clip(0.6,1)'] = round(more_than_5_df['teacher_projects_approved_rate'], 1).clip(0.6,1)
get_barplot(more_than_5_df, 'teacher_projects_approved_rate' + '_round_1dp_clip(0.6,1)')
train_df['teacher_number_of_previously_posted_projects' + '_clip(0,22)'] = round(train_df['teacher_number_of_previously_posted_projects'] / 1).clip(0,22)
get_barplot(train_df, 'teacher_number_of_previously_posted_projects' + '_clip(0,22)')
get_barplot(train_df, 'school_state')
firstDate = train_df['project_submitted_datetime'].min()
train_df['year'] = train_df['project_submitted_datetime'].map(lambda x: x.year)
train_df['month'] = train_df['project_submitted_datetime'].map(lambda x: x.month)
train_df['week'] = train_df['project_submitted_datetime'].map(lambda x: x.week)
train_df['day_num'] = train_df['project_submitted_datetime'].map(lambda x: (x - firstDate).days)
Full_df['month'] = Full_df['project_submitted_datetime'].map(lambda x: x.month)
Full_df['week'] = Full_df['project_submitted_datetime'].map(lambda x: x.week)
get_barplot(train_df,'month')
get_barplot(train_df,'day_num')
time_unit = 'week'
rate_df = train_df[[time_unit,"project_is_approved"]].groupby(time_unit).mean()
rate_df['project_is_rejected'] = 1 - rate_df['project_is_approved']
rate_df = rate_df.drop(columns = ['project_is_approved'])
count_df = Full_df[[time_unit,"project_is_approved"]].groupby(time_unit).count()

fig = plt.figure(figsize=(20,6))
plt.title("Acceptance rate per date and number of applications")
ax1 = plt.subplot(1,1,1)
plt.plot(rate_df , 'blue')
ax2 = plt.subplot(1,1,1)
ax2 = ax1.twinx()
plt.plot(count_df, "red"  )
red_patch = mpatches.Patch(color='red', label='Total number of applications')
blue_patch = mpatches.Patch(color='blue', label='Acceptance rate')
plt.legend(handles=[blue_patch, red_patch])
time_unit = 'month'
rate_df = train_df[[time_unit,"project_is_approved"]].groupby(time_unit).mean()
rate_df['project_is_rejected'] = 1 - rate_df['project_is_approved']
rate_df = rate_df.drop(columns = ['project_is_approved'])
count_df = Full_df[[time_unit,"project_is_approved"]].groupby(time_unit).count()

fig = plt.figure(figsize=(20,6))
plt.title("Acceptance rate per date and number of applications")
ax1 = plt.subplot(1,1,1)
plt.plot(rate_df , 'blue')
ax2 = plt.subplot(1,1,1)
ax2 = ax1.twinx()
plt.plot(count_df, "red"  )
red_patch = mpatches.Patch(color='red', label='Total number of applications')
blue_patch = mpatches.Patch(color='blue', label='Acceptance rate')
plt.legend(handles=[blue_patch, red_patch])
get_barplot(train_df,'project_grade_category')
new_cat,new_approve = [],[]
for i, row in train_df.iterrows():
    cats = row['project_subject_categories'].split(', ')
    for j in range(len(cats)):
        new_cat.append(cats[j])
        new_approve.append(row['project_is_approved'])
new_cat = pd.DataFrame({'project_subject_categories':new_cat})
new_approve = pd.DataFrame({'project_is_approved':new_approve})
cat_total_df = pd.concat([new_cat,new_approve],axis=1).reset_index()
get_barplot(cat_total_df,'project_subject_categories')
new_sub,new_approve_sub = [],[]
for i, row in train_df.iterrows():
    subs = row['project_subject_subcategories'].split(', ')
    for k in range(len(subs)):
        new_sub.append(subs[k])
        new_approve_sub.append(row['project_is_approved'])

new_sub = pd.DataFrame({'project_subject_subcategories':new_sub})
new_approve_sub = pd.DataFrame({'project_is_approved':new_approve_sub})
cat_total_df = pd.concat([new_sub,new_approve_sub],axis=1).reset_index()
get_barplot(cat_total_df,'project_subject_subcategories')
resources_df['resource_price_total'] = resources_df['quantity'] * resources_df['price']
resources_price_total_df = resources_df.groupby(['id'],as_index=False)['resource_price_total'].sum()
resources_quantity_total_df = resources_df.groupby(['id'],as_index=False)['quantity'].sum()
resources_quantity_total_df = resources_quantity_total_df.rename(columns = {'quantity': 'resources_quantity_total'})
resources_count_df = resources_df.groupby(['id'],as_index=False)['quantity'].count()
resources_count_df = resources_count_df.rename(columns = {'quantity': 'resources_variety_count'})
resources_money_df = resources_price_total_df.merge(resources_quantity_total_df)
resources_money_df = resources_money_df.merge(resources_count_df)
resources_money_df['resources_price_ave'] = resources_money_df['resource_price_total'] / resources_money_df['resources_quantity_total']
cat_df = train_df[['id','project_is_approved']].merge(resources_money_df)
cat_df['resource_price_total' + '_scaled_50_clip(0,20)'] = round(cat_df['resource_price_total'] / 50).clip(0,20)
get_barplot(cat_df, 'resource_price_total_scaled_50_clip(0,20)')
cat_df['resources_quantity_total' + '_scaled_20_clip(0,50)'] = round(cat_df['resources_quantity_total'] / 4).clip(0,15)
get_barplot(cat_df, 'resources_quantity_total_scaled_20_clip(0,50)')
cat_df['resources_price_ave' + '_scaled_20_clip(0,16)'] = round(cat_df['resources_price_ave'] / 20).clip(0,16)
get_barplot(cat_df, 'resources_price_ave_scaled_20_clip(0,16)')
cat_df['resources_variety_count' + '_clip(0,20)'] = round(cat_df['resources_variety_count']).clip(0,25)
get_barplot(cat_df, 'resources_variety_count_clip(0,20)')
resources_df['description'] = resources_df['description'].astype(str)
resources_description_series = resources_df.groupby('id')['description'].apply(lambda x: '. '.join(x))
resources_description_df = pd.DataFrame({'id':resources_description_series.index, 'all_resources_description':resources_description_series.values})
train_df = train_df.merge(resources_description_df)

textColumnList = ['project_title', 'project_essay_full', 'project_resource_summary', 'all_resources_description']
cat_df = train_df[['id', 'project_title', 'project_essay_full', 'project_resource_summary', 'all_resources_description', 'project_is_approved' ]]

# Length of words/letters in texts
for textColumn in textColumnList:
    cat_df[textColumn + '_len'] = cat_df[textColumn].map(lambda x: len(str(x)))
#     cat_df[textColumn + '_word_count'] = cat_df[textColumn].map(lambda x: len(str(x).split(' ')))
cat_df['project_title' + '_len' + 'scaled_in_5_clip(0,15)'] = round(cat_df['project_title' + '_len'] / 5).clip(2,15)
get_barplot(cat_df, 'project_title' + '_len' + 'scaled_in_5_clip(0,15)')
cat_df['project_essay_full' + '_len' + 'scaled_in_24_clip(25,55)'] = round(cat_df['project_essay_full' + '_len'] / 40).clip(25,55)
get_barplot(cat_df, 'project_essay_full' + '_len' + 'scaled_in_24_clip(25,55)')
cat_df['project_resource_summary' + '_len' + 'scaled_in_10_clip(5,24)'] = round(cat_df['project_resource_summary' + '_len'] / 10).clip(5,24)
get_barplot(cat_df, 'project_resource_summary' + '_len' + 'scaled_in_10_clip(5,24)')
cat_df['all_resources_description' + '_len' + 'scaled_in_40_clip(1,24)'] = round(cat_df['all_resources_description' + '_len'] / 40).clip(1,24)
get_barplot(cat_df, 'all_resources_description' + '_len' + 'scaled_in_40_clip(1,24)')