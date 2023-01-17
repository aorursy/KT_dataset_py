import pandas as pd

import matplotlib.pyplot as plt

import re

CSV_PATH = '../input/tweet5foldoutputs'
df_ls = [] 

for i in range(5):

    df = pd.read_csv(f'{CSV_PATH}/fold_{i}.csv')

    df_ls.append(df)

df = pd.concat(df_ls)

df = df.drop(['Unnamed: 0'], axis=1)

df.head()
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



def calculate_jac(df):

    ls = []

    for row in df.iterrows():

        row = row[1]

        a = row.selected_text

        b = row.selected_text_out

        jac = jaccard(a,b)

        ls.append(jac)

    return ls
jaccard_scores = calculate_jac(df)

df['jaccard'] = jaccard_scores
df['selected_text_2'] = df['selected_text'].apply(lambda x: re.findall(r"[\w']+|[.,!?;]",x))

df['selected_text_2_out'] = df['selected_text_out'].apply(lambda x: re.findall(r"[\w']+|[.,!?;]",x))
def jaccard2(a,b):

    ls_a = []

    ls_b = []

    for word in a:

        ls_a.append(word.lower())

    for word in b:

        ls_b.append(word.lower())

    a = set(ls_a) 

    b = set(ls_b)

    c = a.intersection(b)

    denom = (len(a) + len(b) - len(c))

    if denom == 0:

        return 1

    return float(len(c)) / denom

def calculate_jac2(df):

    ls = []

    for row in df.iterrows():

        row = row[1]

        a = row.selected_text_2

        b = row.selected_text_2_out

        jac = jaccard2(a,b)

        ls.append(jac)

    return ls
df['jaccard2'] = calculate_jac2(df)
print(df['jaccard'].mean())
print(df['jaccard2'].mean())
for i in range(5):

    df_i = df[df.kfold == i]

    jac = df_i['jaccard'].mean()

    print(f'For out of fold {i}, jaccard is {jac}')
for i in range(5):

    df_i = df[df.kfold == i]

    jac = df_i['jaccard2'].mean()

    print(f'For out of fold {i}, jaccard2 is {jac}')
df_positive = df[df['sentiment']=='positive']

df_negative = df[df['sentiment']=='negative']

df_neutral = df[df['sentiment']=='neutral']
print('Jaccard for positive',df_positive['jaccard'].mean())

print('Jaccard for negative',df_negative['jaccard'].mean())

print('Jaccard for neutral',df_neutral['jaccard'].mean())
print('Jaccard2 for positive',df_positive['jaccard2'].mean())

print('Jaccard2 for negative',df_negative['jaccard2'].mean())

print('Jaccard2 for neutral',df_neutral['jaccard2'].mean())
def print_bad_examples(df,num=10):

    df = df.sort_values(by=['jaccard'])

    for i in range(num):

        row = df.iloc[i]

        print('text:             ', row.text.strip())

        print('selected text:    ', row.selected_text.strip())

        print('my selected text: ',row.selected_text_out.strip())

        print('-'*50)
def print_bad_examples2(df,num=10):

    df = df.sort_values(by=['jaccard2'])

    for i in range(num):

        row = df.iloc[i]

        print('text:             ', row.text.strip())

        print('selected text:    ', row.selected_text.strip())

        print('my selected text: ',row.selected_text_out.strip())

        print('-'*50)
print('Jaccard Worst Examples positive sentiment \n')

print_bad_examples(df_positive)
print('Jaccard2 Worst Examples positive sentiment \n')

print_bad_examples2(df_positive)
print('Jaccard Worst Examples negative sentiment \n')

print_bad_examples(df_negative)
print('Jaccard2 Worst Examples negative sentiment \n')

print_bad_examples2(df_negative)
print('Jaccard Worst Examples neutral sentiment \n')

print_bad_examples(df_neutral)
print('Jaccard2 Worst Examples neutral sentiment \n')

print_bad_examples2(df_neutral)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))

axes[0].hist(df.jaccard.values)

axes[0].set_title('Histogram of all jaccard scores')

axes[1].hist(df.jaccard2.values)

axes[1].set_title('Histogram of all jaccard2 scores')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))

axes[0].hist(df_neutral.jaccard.values)

axes[0].set_title('Histogram of all neutral jaccard scores')

axes[1].hist(df.jaccard2.values)

axes[1].set_title('Histogram of all neutral jaccard2 scores')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))

axes[0].hist(df_positive.jaccard.values)

axes[0].set_title('Histogram of all positive jaccard scores')

axes[1].hist(df.jaccard2.values)

axes[1].set_title('Histogram of all positive jaccard2 scores')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))

axes[0].hist(df_negative.jaccard.values)

axes[0].set_title('Histogram of all negative jaccard scores')

axes[1].hist(df.jaccard2.values)

axes[1].set_title('Histogram of all negative jaccard2 scores')