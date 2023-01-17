import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

#plt.xkcd()
df_ppl = pd.read_csv('../input/people.csv')
df_act_train = pd.read_csv('../input/act_train.csv')
df_act_test = pd.read_csv('../input/act_test.csv')

new_columns = []
for col in df_ppl.columns:
    if 'char' in col or 'date' in col:
        new_columns.append('ppl_' + col)
    else:
        new_columns.append(col)
df_ppl.columns = new_columns

new_columns = []
for col in df_act_train.columns:
    if 'char' in col or 'date' in col:
        new_columns.append('act_' + col)
    else:
        new_columns.append(col)
df_act_train.columns = new_columns
del(new_columns[-1])
df_act_test.columns = new_columns

df = pd.merge(df_act_train, df_ppl, on='people_id')
df_validate = pd.merge(df_act_test, df_ppl, on='people_id')

df.to_csv('merged.csv')
df_validate.to_csv('merged_test.csv')

del(df_ppl, df_act_train, df_act_test)

print('Memory usage of training DataFrame: ' + str(sum(df.memory_usage())))
print('Columns: ' + str(df.columns))
sns.countplot(x='outcome', data=df)
plt.suptitle('Customer Value - Binary Outcomes', fontsize=20)
plt.show()
row_counts = []
for col in df.columns:
    rows = len(df[col].value_counts())
    row_counts.append((col + ': ' + str(rows) + ' unique values.', rows))
row_counts.sort(key=lambda tup: tup[1], reverse=True)
for col in row_counts:
    print(col[0])
def null_percentage(column):
    df_name = column.name
    nans = np.count_nonzero(column.isnull().values)
    total = column.size
    frac = nans / total
    perc = int(frac * 100)
    print('%d%% of values or %d missing from %s column.' % (perc, nans, df_name))

def check_null(df, columns):
    for col in columns:
        null_percentage(df[col])
        
check_null(df, df.columns)
overlap_count = 0
for non_null_feature in [df.act_char_9.notnull(), 
                df.act_char_8.notnull(), 
                df.act_char_7.notnull(), 
                df.act_char_6.notnull(), 
                df.act_char_5.notnull(),
                df.act_char_4.notnull(),
                df.act_char_3.notnull(),
                df.act_char_2.notnull(),
                df.act_char_1.notnull()]:
    overlap_count += df.loc[df.act_char_10.notnull() & non_null_feature].shape[0]
print('%d rows have overlap between char_10 and any other characteristic features.' % overlap_count)
overlap = df.loc[df.act_char_9.notnull() & df.act_char_8.notnull() & df.act_char_7.notnull() & 
                 df.act_char_6.notnull() & df.act_char_5.notnull() & df.act_char_4.notnull() & 
                 df.act_char_3.notnull() & df.act_char_2.notnull() & df.act_char_1.notnull()]
print('%d rows have overlap between ALL characteristic columns besides char_10.' % overlap.shape[0])
del(overlap)
df.act_date = df.act_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
#df.ppl_date = df.ppl_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
#df['date_diff'] = df.act_date.sub(df.ppl_date, axis=0)
#df.date_diff = df.date_diff.apply(lambda x: int(x.days))

weekday_map = {0:'1 Sunday', 1:'2 Monday', 2:'3 Tuesday', 3:'4 Wednesday', 4:'5 Thursday', 5:'6 Friday', 
              6:'7 Saturday'}
#df['ppl_weekday'] = df.ppl_date.apply(lambda x: x.weekday())
#df.ppl_weekday = df.ppl_weekday.replace(weekday_map)
df['act_weekday'] = df.act_date.apply(lambda x: x.weekday())
df.act_weekday = df.act_weekday.replace(weekday_map)
df['act_year'] = df.act_date.apply(lambda x: x.year)
df['act_month'] = df.act_date.apply(lambda x: x.month)
df['act_day'] = df.act_date.apply(lambda x: x.day)
tab = pd.crosstab(df.act_weekday, df.outcome)
tab['ratio'] = tab[0] + tab[1]
tab.ratio = (tab[1] / tab.ratio) * 100
bar = sns.barplot(x = list(tab.index), y = list(tab.ratio))
bar.set(ylabel="Percentage", xlabel="Day of Week")
plt.xticks(rotation = 45)
plt.show()

print('Range: ' + str(max(list(tab.ratio))-min(list(tab.ratio))))
bar = sns.barplot(x = list(weekday_map.values()), y = list(df.act_weekday.value_counts().sort_index()))
plt.xticks(rotation = 45)
plt.show()
def crosstab_heatmap(*args, title='', size=(6.4, 4.8), ant=True, color='Blues'):
    tab = pd.crosstab(*args)
    plt.figure(title, figsize=size)
    plt.title(title)
    hmap = sns.heatmap(tab, annot=ant, fmt='g', cmap=color)
    loc, ylabels = plt.yticks()
    #hmap.set_xticklabels(labels, rotation=45)
    hmap.set_yticklabels(ylabels, rotation=45)
    plt.show()

crosstab_heatmap(df.act_weekday, df.outcome)
crosstab_heatmap(df.act_year, df.act_month, title='chart', size=(20,6))
crosstab_heatmap(df.act_month, df.act_day, title='chart', ant=False, size=(15,8), color='YlGnBu')
def plot_ecdf(data, label):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    _ = plt.plot(x, y, marker='.', linestyle='none')
    _ = plt.xlabel(label)
    _ = plt.ylabel('ECDF')
    plt.margins(0.02)
    plt.show()

def percentages(data, top=10):
    s = data.iloc[:,0] 
    s = s.value_counts()
    s = s.index
    s = s[0:top]
    col = data.columns[0]
    data = data.loc[df[col].isin(s)]
    tab = pd.crosstab(data.iloc[:,0], data.iloc[:,1]).apply(lambda r: r/r.sum(), axis=1)
    tab.plot(kind='bar', stacked=True, color=['red','blue'], grid=False, figsize=(30, 8), legend=None)
    plt.show()

len(df.group_1.value_counts())
percentages(df[['group_1', 'outcome']], top = 10)
df.group_1.value_counts().head(10)
percentages(df[['group_1', 'outcome']], top = 200)
overlap = 0
df_groups, df_test_groups = df.group_1.value_counts().index, df_validate.group_1.value_counts().index
for group in df_groups:
    if group in df_test_groups:
        overlap += 1
print('Trainign set groups: %d' % len(df_groups))
print('Test set groups: ' + str(len(df_test_groups)))
print('Overlap: ' + str(overlap))
overlap = 0
df_groups, df_test_groups = df.group_1.value_counts().index[0:1000], df_validate.group_1.value_counts()[0:1000].index
for group in df_groups:
    if group in df_test_groups:
        overlap += 1
print('Trainign set groups: %d' % len(df_groups))
print('Test set groups: %d' % len(df_test_groups))
print('Overlap: ' + str(overlap))
print('Training top five:')
print(df.group_1.value_counts().head())
print()
print('Test top five:')
print(df_validate.group_1.value_counts().head())
activities = list(df.columns[5:15])
#print(activities)

people = sorted(df.columns)
people = people[21:59] 
#people.remove('people_id')
#print(people)

def perc_tab(data, i, j, top=10):
    s = data.iloc[:,0] 
    s = s.value_counts()
    s = s.index
    s = s[0:top]
    col = data.columns[0]
    #print(col)
    data = data.loc[df[col].isin(s)]
    tab = pd.crosstab(data.iloc[:,0], data.iloc[:,1]).apply(lambda r: r/r.sum(), axis=1)
    tab.plot(kind='bar', stacked=True, color=['red','blue'], grid=False, ax=axes[i, j], legend=None)

crosstab_heatmap(df.activity_category, df.outcome)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25,12))
i = 0
j = 0
for c, col in enumerate(activities):
    if c > 4:
        i = c % 5
    else:
        i = c
    if c > 0 and c % 5 == 0:
        j += 1
    #print(str(i) + ' ' + str(j))

    perc_tab(df[[col, 'outcome']], j, i, top = 10)
plt.show()
fig, axes = plt.subplots(nrows=8, ncols=5, figsize=(25,45))
i = 0
j = 0
for c, col in enumerate(people):
    if c > 4:
        i = c % 5
    else:
        i = c
    if c > 0 and c % 5 == 0:
        j += 1
    #print(str(i) + ' ' + str(j))

    perc_tab(df[[col, 'outcome']], j, i, top = 10)
plt.show()
print(df.people_id.value_counts().head(15))
power_users = df.people_id.value_counts().head(500).index
df_power = df[df.people_id.isin(list(power_users))]
df_power.head()
percentages(df_power[['people_id', 'outcome']], top = 13)
percentages(df[['people_id', 'outcome']], top = 150)
del(df_power)
one_person = df.loc[df.people_id == 'ppl_337688']
one_person.iloc[0:20]