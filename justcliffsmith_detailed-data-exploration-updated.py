# Let's import everything we'll be using. Keep it all at the top to make your life easy.
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
# Where are our files? Check down below! Don't forget the ../input when you try to load them in.
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
resource = pd.read_csv('../input/resources.csv')
print(train.shape)
print(resource.shape)
train.head()
resource.head()
train.dtypes
resource.dtypes
for label in list(train.columns.values):
    print(f'{label} has {sum(train[label].isna())}')
# Using this pandas code you can infer how many nan's there are. 
# One line of code instead of my 2 above, plus this gives more info!

#train.info()
# If we wanted to look at them in depth then we'd use the following code. 
# For the sake of length I'll omit the output since it isn't particularly interesting.
#idx = np.where(train['teacher_prefix'].isna() == True)[0]
#print(train.iloc[idx])
for label in ['teacher_prefix', 'school_state', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories']:
    total = 0
    print(f'=== {label} ===')
    for i, item in enumerate(train[label].unique()):
        count = len(np.where(train[label] == item)[0])
        print('{}: {}'.format(item, count))
        total += count
    print(f'== Total categories: {i+1} and {total} values out of {len(train)}===\n')
# Here is the slick pandas way to do the above code. 2 lines versus 8! 
# Note that doing it this way doesn't show the nan's in the teacher_prefix, 
# so you still want to make sure you use something to count nan's!

#for label in ['teacher_prefix', 'school_state', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories']:
#    print(train[label].value_counts())
train.describe(include=['O'])
train.describe()
plt.figure()
plt.hist(train['teacher_number_of_previously_posted_projects'], bins=30)
plt.title('Histogram Counting # of Teachers that Previously Posted Projects')
plt.xlabel('Projects')
plt.ylabel('Count')
plt.show()

nonzeroidx = np.where(train['teacher_number_of_previously_posted_projects'] != 0 )[0]
plt.figure()
plt.hist(train['teacher_number_of_previously_posted_projects'].iloc[nonzeroidx], bins=30)
plt.title('Histogram Counting # of Teachers that Previously Posted Projects (teachers with 0 removed)')
plt.xlabel('Projects')
plt.ylabel('Count')
plt.show()

testidx = np.where(train['teacher_number_of_previously_posted_projects'] > 5 )[0]
plt.figure()
plt.hist(train['teacher_number_of_previously_posted_projects'].iloc[testidx], bins=30)
plt.title('Histogram Counting # of Teachers that Previously Posted Projects (teachers with 0 removed)')
plt.xlabel('Projects')
plt.ylabel('Count')
plt.show()
len(np.where(train['teacher_id'] == train['teacher_id'][0])[0])
train['teacher_number_of_previously_posted_projects'].iloc[np.where(train['teacher_id'] == train['teacher_id'][0])]
plt.figure()
plt.scatter(train['teacher_number_of_previously_posted_projects'], train['project_is_approved'])
plt.show()
reject = np.where(train['project_is_approved'] == 0)[0] # Indexes where an application failed.
big_submit = np.where(train['teacher_number_of_previously_posted_projects'] > 375)[0] # Indexes where a teacher has sent more than 350 applications before.
idx = np.intersect1d(reject, big_submit)
id_bigsubmit_but_fail = train['teacher_id'].iloc[idx] # All the ids for teachers that have submitted a bunch but didn't suceed on one.

print(id_bigsubmit_but_fail.head(1)) # Let's look at just a single id.
print(train['teacher_number_of_previously_posted_projects'].iloc[np.where(train['teacher_id'] == id_bigsubmit_but_fail.iloc[0])])
print(train['project_is_approved'].iloc[np.where(train['teacher_id'] == id_bigsubmit_but_fail.iloc[0])])
# Now that I've changed my analysis above, this is redundant. But I'll keep it in for posterity.

rejected_rate = len(reject)/len(train)
print(f'The acceptance rate is {(1 - rejected_rate)*100}%')
for label in list(resource.columns.values):
    print(f'{label} has {sum(resource[label].isna())}')
id_cost = pd.DataFrame({'id': resource['id'], 'total_cost': resource['quantity'] * resource['price']})
id_cost.head()
id_total_cost = id_cost.groupby(id_cost['id'], sort=False).sum().reset_index()

# Small note, I originally wrote the above code as a for loop that looped 
# through all unique ids and and summed up every instance.
# However, it ran slooowwww. I projected it'd take aboout 3.5 hours to run. 
# The above code runs in under a second. Use pandas built-in methods!
id_total_cost.head()
id_total_cost.describe()
id_total_cost.sort_values(by=['total_cost'])
plt.figure()
plt.hist(id_total_cost['total_cost'], bins=50)
plt.show()
print(len(id_total_cost['id']))
print(len(train['id']))
train['id'].isin(id_total_cost['id']).head()
print(sum(train['id'].isin(id_total_cost['id'])))
print(len(train['id'].isin(id_total_cost['id'])))
train_aug = pd.merge(train, id_total_cost, on='id', sort=False)
sum(pd.merge(train, id_total_cost, on='id', sort=False)['total_cost'].isna())
train_aug.head()
plt.figure()
plt.scatter(train_aug['teacher_number_of_previously_posted_projects'], train_aug['total_cost'])
plt.ylabel('Total Cost')
plt.xlabel('Number of Previous Submissions')
plt.show()

plt.figure()
plt.scatter(train_aug['project_is_approved'], train_aug['total_cost'])
plt.ylabel('Total Cost')
plt.xlabel('Approved')
plt.show()
ax = sns.heatmap(train_aug.corr(), annot=True, cmap='coolwarm')