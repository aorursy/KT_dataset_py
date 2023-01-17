import pandas as pd
import os
import matplotlib.pyplot as plt
%matplotlib inline

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 18
PATH_TO_DATA = '../input'

train = pd.read_csv(os.path.join(PATH_TO_DATA, 'train.csv.gz'))
UNITS = ['кг', '%', 'мл', 'шт']

train_units = train[['name', 'category']].copy()
for u in UNITS:
    train_units['{0}'.format(u)] = train['name'].str.lower().str.contains(u)
train_units[UNITS].sum().reset_index().rename(columns={'index': 'unit', 0: 'count'})
units_categories = train_units[['category'] + UNITS].groupby('category').sum()

for u in UNITS:
    plt.figure(figsize=(20, 10))
    units_categories[u].sort_values(ascending=False).plot(kind='bar', color='darkblue')
    plt.title('Category distribution for {0}'.format(u))
def get_one_char_word_count(name):
    clean_name = ''.join([c if c.isalpha() else ' ' for c in name])
    one_char_words = [w for w in clean_name.split() if len(w) == 1]
    return len(one_char_words)

train_names = train[['name']].copy()
train_names['one_char_word_count'] = train_names['name'].fillna('').apply(get_one_char_word_count)
train_names.one_char_word_count.max()
train_names[train_names.one_char_word_count > 7]
check_size = train.groupby('check_id').count()['category']
check_size_value_counts = check_size.value_counts()[:10]
check_size_value_counts.reset_index().rename(columns={'index': 'check_size', 'category': 'count'})
train_check_size = train[['check_id', 'category']].merge(
    check_size.reset_index().rename(columns={'category': 'check_size'}), on='check_id', how='left')
def plot_check_size_category_distribution(df, size_description):
    plt.figure(figsize=(20, 10))
    df.groupby('category').count()['check_id'].sort_values(ascending=False).plot(kind='bar', color='darkblue')
    plt.title('Category distibution for check sizes {0}'.format(size_description))
    
plot_check_size_category_distribution(
    train_check_size[train_check_size.check_size == 1], '== 1')
plot_check_size_category_distribution(
    train_check_size[(train_check_size.check_size > 1) & (train_check_size.check_size <= 5)], '> 1 and <= 5')
plot_check_size_category_distribution(
    train_check_size[(train_check_size.check_size > 5)], '> 5')