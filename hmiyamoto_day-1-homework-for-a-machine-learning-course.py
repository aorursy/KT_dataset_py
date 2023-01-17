# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_recall_fscore_support
df_kick = pd.read_csv("../input/ks-projects-201801.csv")
display(df_kick.head())
display(df_kick.tail())
print(df_kick.columns.values)
df_kick.describe()
df_kick.describe(include=['O'])
df_kick.info()
print(df_kick['state'].value_counts(dropna=False))
print(df_kick['state'].value_counts(dropna=False, normalize=True))
g = sns.countplot(df_kick['state'],
                 order = df_kick['state'].value_counts().index)
g = sns.distplot(df_kick['backers'])
plt.hist(df_kick['backers'], bins=10**np.linspace(0, 7, 8)) 
plt.xscale('log') 
plt.xlabel('backers log-scale')
plt.ylabel('count')
plt.title('backers')
plt.show() 
display(df_kick[['backers']].sort_values(by='backers', ascending=False).head())
plt.hist(df_kick['backers'][df_kick['state']=='failed'], bins=10**np.linspace(0, 7, 8)) 
plt.xscale('log') 
plt.xlabel('backers log-scale')
plt.ylabel('count')
plt.title('failed')
plt.show() 
plt.hist(df_kick['backers'][df_kick['state']=='successful'], bins=10**np.linspace(0, 7, 8)) 
plt.xscale('log') 
plt.xlabel('backers log-scale')
plt.ylabel('count')
plt.title('successful')
plt.show() 
plt.hist(df_kick['backers'][df_kick['state']=='canceled'], bins=10**np.linspace(0, 7, 8)) 
plt.xscale('log') 
plt.xlabel('backers log-scale')
plt.ylabel('count')
plt.title('canceled')
plt.show() 
plt.hist(df_kick['backers'][df_kick['state']=='live'], bins=10**np.linspace(0, 7, 8)) 
plt.xscale('log') 
plt.xlabel('backers log-scale')
plt.ylabel('count')
plt.title('live')
plt.show() 
plt.hist(df_kick['backers'][df_kick['state']=='suspended'], bins=10**np.linspace(0, 7, 8)) 
plt.xscale('log') 
plt.xlabel('backers log-scale')
plt.ylabel('count')
plt.title('suspended')
plt.show() 
g = sns.distplot(df_kick['usd_goal_real'])
plt.hist(df_kick['usd_goal_real'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_goal_real log-scale')
plt.ylabel('count')
plt.title('usd_goal_real')
plt.show() 
display(df_kick[['usd_goal_real']].sort_values(by='usd_goal_real', ascending=False).head())
plt.hist(df_kick['usd_goal_real'][df_kick['state']=='failed'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_goal_real log-scale')
plt.ylabel('count')
plt.title('failed')
plt.show() 
plt.hist(df_kick['usd_goal_real'][df_kick['state']=='successful'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_goal_real log-scale')
plt.ylabel('count')
plt.title('successful')
plt.show() 
plt.hist(df_kick['usd_goal_real'][df_kick['state']=='canceled'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_goal_real log-scale')
plt.ylabel('count')
plt.title('canceled')
plt.show() 
plt.hist(df_kick['usd_goal_real'][df_kick['state']=='live'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_goal_real log-scale')
plt.ylabel('count')
plt.title('live')
plt.show() 
plt.hist(df_kick['usd_goal_real'][df_kick['state']=='suspended'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_goal_real log-scale')
plt.ylabel('count')
plt.title('suspended')
plt.show() 
g = sns.distplot(df_kick['usd_pledged_real'])
plt.hist(df_kick['usd_pledged_real'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_pledged_real log-scale')
plt.ylabel('count')
plt.title('usd_pledged_real')
plt.show() 
display(df_kick[['usd_pledged_real']].sort_values(by='usd_pledged_real', ascending=False).head())
plt.hist(df_kick['usd_pledged_real'][df_kick['state']=='failed'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_pledged_real log-scale')
plt.ylabel('count')
plt.title('failed')
plt.show() 
plt.hist(df_kick['usd_pledged_real'][df_kick['state']=='successful'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_pledged_real log-scale')
plt.ylabel('count')
plt.title('successful')
plt.show() 
plt.hist(df_kick['usd_pledged_real'][df_kick['state']=='canceled'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_pledged_real log-scale')
plt.ylabel('count')
plt.title('canceled')
plt.show() 
plt.hist(df_kick['usd_pledged_real'][df_kick['state']=='live'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_pledged_real log-scale')
plt.ylabel('count')
plt.title('live')
plt.show() 
plt.hist(df_kick['usd_pledged_real'][df_kick['state']=='suspended'], bins=10**np.linspace(0, 8, 9)) 
plt.xscale('log') 
plt.xlabel('usd_pledged_real log-scale')
plt.ylabel('count')
plt.title('suspended')
plt.show() 
print(df_kick['main_category'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['main_category'],
                 order = df_kick['main_category'].value_counts().index)
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['main_category'][df_kick['state'] == 'successful'],
                 order = df_kick['main_category'].value_counts().index)
df_main_category_successful_count = df_kick['main_category'][df_kick['state'] == 'successful'].value_counts()
# df_main_category_successful_count.to_dict()
df_main_category_all_count = df_kick['main_category'].value_counts()
# df_main_category_all_count.to_dict()
category_success_rate = {}
success_rate = {}
other_rate = {}

for main_category in df_main_category_all_count.keys():
    success_rate[main_category] = df_main_category_successful_count[main_category] / df_main_category_all_count[main_category]
    other_rate[main_category] = 1.0 - success_rate[main_category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('main_category')
plt.ylabel('Success rate')
plt.title('Success rate of each main_category')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'].value_counts(dropna=False).head(20))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'],
                 order = df_kick['category'].value_counts().index)
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['state'] == 'successful'],
                 order = df_kick['category'].value_counts().index)
df_category_successful_count = df_kick['category'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Film & Video'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Film & Video'],
                 order = df_kick['category'][df_kick['main_category'] == 'Film & Video'].value_counts().index) \
                    .set_title("main_category == 'Film & Video'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Film & Video'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Film & Video'].value_counts().index) \
                    .set_title("main_category == 'Film & Video'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Film & Video'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Film & Video'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Film & Video)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Music'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Music'],
                 order = df_kick['category'][df_kick['main_category'] == 'Music'].value_counts().index) \
                    .set_title("main_category == 'Music'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Music'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Music'].value_counts().index) \
                    .set_title("main_category == 'Music'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Music'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Music'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Music)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Publishing'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Publishing'],
                 order = df_kick['category'][df_kick['main_category'] == 'Publishing'].value_counts().index) \
                    .set_title("main_category == 'Publishing'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Publishing'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Publishing'].value_counts().index) \
                    .set_title("main_category == 'Publishing'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Publishing'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Publishing'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Publishing)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Games'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Games'],
                 order = df_kick['category'][df_kick['main_category'] == 'Games'].value_counts().index) \
                    .set_title("main_category == 'Games'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Games'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Games'].value_counts().index) \
                    .set_title("main_category == 'Games'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Games'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Games'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Games)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Technology'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Technology'],
                 order = df_kick['category'][df_kick['main_category'] == 'Technology'].value_counts().index) \
                    .set_title("main_category == 'Technology'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Technology'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Technology'].value_counts().index) \
                    .set_title("main_category == 'Technology'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Technology'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Technology'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Technology)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Design'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Design'],
                 order = df_kick['category'][df_kick['main_category'] == 'Design'].value_counts().index) \
                    .set_title("main_category == 'Design'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Design'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Design'].value_counts().index) \
                    .set_title("main_category == 'Design'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Design'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Design'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Design)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Art'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Art'],
                 order = df_kick['category'][df_kick['main_category'] == 'Art'].value_counts().index) \
                    .set_title("main_category == 'Art'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Art'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Art'].value_counts().index) \
                    .set_title("main_category == 'Art'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Art'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Art'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Art)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Food'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Food'],
                 order = df_kick['category'][df_kick['main_category'] == 'Food'].value_counts().index) \
                    .set_title("main_category == 'Food'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Food'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Food'].value_counts().index) \
                    .set_title("main_category == 'Food'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Food'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Food'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Food)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Fashion'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Fashion'],
                 order = df_kick['category'][df_kick['main_category'] == 'Fashion'].value_counts().index) \
                    .set_title("main_category == 'Fashion'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Fashion'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Fashion'].value_counts().index) \
                    .set_title("main_category == 'Fashion'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Fashion'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Fashion'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Fashion)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Theater'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Theater'],
                 order = df_kick['category'][df_kick['main_category'] == 'Theater'].value_counts().index) \
                    .set_title("main_category == 'Theater'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Theater'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Theater'].value_counts().index) \
                    .set_title("main_category == 'Theater'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Theater'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Theater'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Theater)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Comics'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Comics'],
                 order = df_kick['category'][df_kick['main_category'] == 'Comics'].value_counts().index) \
                    .set_title("main_category == 'Comics'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Comics'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Comics'].value_counts().index) \
                    .set_title("main_category == 'Comics'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Comics'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Comics'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Comics)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Photography'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Photography'],
                 order = df_kick['category'][df_kick['main_category'] == 'Photography'].value_counts().index) \
                    .set_title("main_category == 'Photography'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Photography'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Photography'].value_counts().index) \
                    .set_title("main_category == 'Photography'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Photography'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Photography'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Photography)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Crafts'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Crafts'],
                 order = df_kick['category'][df_kick['main_category'] == 'Crafts'].value_counts().index) \
                    .set_title("main_category == 'Crafts'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Crafts'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Crafts'].value_counts().index) \
                    .set_title("main_category == 'Crafts'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Crafts'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Crafts'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Crafts)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Journalism'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Journalism'],
                 order = df_kick['category'][df_kick['main_category'] == 'Journalism'].value_counts().index) \
                    .set_title("main_category == 'Journalism'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Journalism'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Journalism'].value_counts().index) \
                    .set_title("main_category == 'Journalism'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Journalism'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Journalism'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Journalism)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['category'][df_kick['main_category'] == 'Dance'].value_counts(dropna=False))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Dance'],
                 order = df_kick['category'][df_kick['main_category'] == 'Dance'].value_counts().index) \
                    .set_title("main_category == 'Dance'")
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['category'][df_kick['main_category'] == 'Dance'][df_kick['state'] == 'successful'],
                 order = df_kick['category'][df_kick['main_category'] == 'Dance'].value_counts().index) \
                    .set_title("main_category == 'Dance'")
df_category_successful_count = df_kick['category'][df_kick['main_category'] == 'Dance'][df_kick['state'] == 'successful'].value_counts()
# df_category_successful_count.to_dict()
df_category_all_count = df_kick['category'][df_kick['main_category'] == 'Dance'].value_counts()
# df_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for category in df_category_all_count.keys():
    success_rate[category] = df_category_successful_count[category] / df_category_all_count[category]
    other_rate[category] = 1.0 - success_rate[category]
# success_rate

category_success_rate.update(success_rate)
# category_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('category')
plt.ylabel('Success rate')
plt.title('Success rate of each category (main_category == Dance)')
plt.legend(loc='upper left')
plt.show()
print(df_kick['currency'].value_counts(dropna=False))
print(df_kick['currency'].value_counts(dropna=False, normalize=True))
g = sns.countplot(df_kick['currency'],
                 order = df_kick['currency'].value_counts().index)
g = sns.countplot(df_kick['currency'][df_kick['state'] == 'successful'],
                 order = df_kick['currency'].value_counts().index)
df_main_category_successful_count = df_kick['currency'][df_kick['state'] == 'successful'].value_counts()
# df_main_category_successful_count.to_dict()
df_main_category_all_count = df_kick['currency'].value_counts()
# df_main_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for main_category in df_main_category_all_count.keys():
    success_rate[main_category] = df_main_category_successful_count[main_category] / df_main_category_all_count[main_category]
    other_rate[main_category] = 1.0 - success_rate[main_category]
# success_rate

currency_success_rate = success_rate
currency_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('currency')
plt.ylabel('Success rate')
plt.title('Success rate of each currency')
plt.legend(loc='upper left')
plt.show()
print(df_kick['country'].value_counts(dropna=False))
print(df_kick['country'].value_counts(dropna=False, normalize=True))
g = sns.countplot(df_kick['country'],
                 order = df_kick['country'].value_counts().index)
g = sns.countplot(df_kick['country'][df_kick['state'] == 'successful'],
                 order = df_kick['country'].value_counts().index)
df_main_category_successful_count = df_kick['country'][df_kick['state'] == 'successful'].value_counts()
# df_main_category_successful_count.to_dict()
df_main_category_all_count = df_kick['country'].value_counts()
# df_main_category_all_count.to_dict()
success_rate = {}
other_rate = {}

for main_category in df_main_category_all_count.keys():
    success_rate[main_category] = df_main_category_successful_count[main_category] / df_main_category_all_count[main_category]
    other_rate[main_category] = 1.0 - success_rate[main_category]
# success_rate

country_success_rate = success_rate
# country_success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('currency')
plt.ylabel('Success rate')
plt.title('Success rate of each country')
plt.legend(loc='upper left')
plt.show()
df_kick['launched'] = pd.to_datetime(df_kick['launched'])
df_kick['laun_month_year'] = df_kick['launched'].dt.to_period("M")
df_kick['laun_year'] = df_kick['launched'].dt.to_period("A")
df_kick['laun_hour'] = df_kick['launched'].dt.hour

df_kick['deadline'] = pd.to_datetime(df_kick['deadline'])
df_kick['dead_month_year'] = df_kick['deadline'].dt.to_period("M")
df_kick['dead_year'] = df_kick['launched'].dt.to_period("A")
#Creating a new columns with Campaign total months
df_kick['time_campaign'] = df_kick['dead_month_year'] - df_kick['laun_month_year']
df_kick['time_campaign'] = df_kick['time_campaign'].astype(int)
display(df_kick.head())
df_kick.info()
df_kick.describe()
df_kick.describe(include=['O'])
print(df_kick['laun_year'].value_counts(dropna=False))
print(df_kick['laun_year'].value_counts(dropna=False).index.sort_values(ascending=True))
g = sns.countplot(df_kick['laun_year'],
                  order=df_kick['laun_year'].value_counts(dropna=False).index.sort_values(ascending=True))
print(df_kick['laun_month_year'].value_counts(dropna=False).index.sort_values(ascending=True))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['laun_month_year'],
                  order=df_kick['laun_month_year'].value_counts(dropna=False).index.sort_values(ascending=True))
print(df_kick['dead_year'].value_counts(dropna=False))
print(df_kick['dead_year'].value_counts(dropna=False).index.sort_values(ascending=True))
g = sns.countplot(df_kick['dead_year'],
                  order=df_kick['dead_year'].value_counts(dropna=False).index.sort_values(ascending=True))
print(df_kick['dead_month_year'].value_counts(dropna=False).index.sort_values(ascending=True))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['dead_month_year'],
                  order=df_kick['dead_month_year'].value_counts(dropna=False).index.sort_values(ascending=True))
print(df_kick['time_campaign'].value_counts(dropna=False))
g = sns.countplot(df_kick['time_campaign'],
                  order=df_kick['time_campaign'].value_counts(dropna=False).index.sort_values(ascending=True))
plt.figure(figsize=(15, 10))
g = sns.countplot(df_kick['time_campaign'][df_kick['state'] == 'successful'],
                  order=df_kick['time_campaign'].value_counts(dropna=False).index.sort_values(ascending=True))
df_time_campaign_successful_count = df_kick['time_campaign'][df_kick['state'] == 'successful'].value_counts()
df_time_campaign_successful_count.to_dict()
df_time_campaign_all_count = df_kick['time_campaign'].value_counts()
df_time_campaign_all_count.to_dict()
success_rate = {}
other_rate = {}

print(sorted(df_time_campaign_all_count.keys()))

for time_campaign in sorted(df_time_campaign_all_count.keys()):
#     time_campaign = str(time_campaign)
#     print(time_campaign, df_time_campaign_successful_count[time_campaign])
#     print(df_time_campaign_all_count[time_campaign])
    if time_campaign not in df_time_campaign_successful_count.keys():
        df_time_campaign_successful_count[time_campaign] = 0
    
    success_rate[time_campaign] = df_time_campaign_successful_count[time_campaign] / df_time_campaign_all_count[time_campaign]
    other_rate[time_campaign] = 1.0 - success_rate[time_campaign]
# success_rate
plt.figure(figsize=(15, 10))
names = list(success_rate.keys())
values_success = list(success_rate.values())
values_other = list(other_rate.values())
plt.bar(range(len(success_rate)),values_success,tick_label=names, label='successful')
plt.bar(range(len(success_rate)),values_other,tick_label=names, label='other', bottom=values_success)
plt.xlabel('main_category')
plt.ylabel('Success rate')
plt.title('Success rate of each main_category')
plt.legend(loc='upper left')
plt.show()
display(df_kick.head())
df_kick['backers_log10'] = np.log10(df_kick['backers'] + 1e-8)
df_kick['usd_pledged_real_log10'] = np.log10(df_kick['usd_pledged_real'] + 1e-8)
df_kick['usd_goal_real_log10'] = np.log10(df_kick['usd_goal_real'] + 1e-8)
display(df_kick.head())
len(category_success_rate)
df_kick['category_dummy'] = df_kick['category'].replace(category_success_rate)
display(df_kick.head())
len(currency_success_rate)
df_kick['currency_dummy'] = df_kick['currency'].replace(currency_success_rate)
display(df_kick.head())
len(country_success_rate)
df_kick['country_dummy'] = df_kick['country'].replace(country_success_rate)
display(df_kick.head())
df_kick['time_campaign_dummy'] = df_kick['time_campaign']
df_kick['time_campaign_dummy'].loc[df_kick['time_campaign_dummy'] >= 5] = 5
display(df_kick.head())
df_kick.describe()
df_kick['state_dummy'] = df_kick['state']
df_kick['state_dummy'].loc[df_kick['state_dummy'] != 'successful'] = 0
df_kick['state_dummy'].loc[df_kick['state_dummy'] == 'successful'] = 1
display(df_kick.head())
df_kick.describe()
y = df_kick["state_dummy"].values
X = df_kick[["backers_log10", "usd_pledged_real_log10", "usd_goal_real_log10", "category_dummy", "currency_dummy", "time_campaign_dummy"]].values
clf = SGDClassifier(loss='log', penalty='none', max_iter=100, fit_intercept=True, random_state=1234)
clf.fit(X, y)

# Weight
w0 = clf.intercept_[0]
w1 = clf.coef_[0, 0]
w2 = clf.coef_[0, 1]
w3 = clf.coef_[0, 2]
w4 = clf.coef_[0, 3]
w5 = clf.coef_[0, 4]
w6 = clf.coef_[0, 5]
print('w0 = {:.3f}, w1 = {:.3f}, w2 = {:.3f}, w3 = {:.3f}, w4 = {:.3f}, w5 = {:.3f}, w6 = {:.3f}'.format(w0, w1, w2, w3, w4, w5, w6))
# Predict labels
y_est = clf.predict(X)

# Log-likelihood
print('Log-likelihood = {:.3f}'.format(- log_loss(y, y_est)))

# Accuracy
print('Accuracy = {:.3f}%'.format(100 * accuracy_score(y, y_est)))

# Precision, Recall, F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(y, y_est)

print('Precision = {:.3f}%'.format(100 * precision[0]))
print('Recal = {:.3f}%'.format(100 * recall[0]))
print('F1-score = {:.3f}%'.format(100 * f1_score[0]))
# Confusion matrix
conf_mat = confusion_matrix(y, y_est)
conf_mat = pd.DataFrame(conf_mat, 
                        index=['Correct = Other', 'Correct = Successful'], 
                        columns=['Predict = Other', 'Predict = Successful'])
conf_mat
