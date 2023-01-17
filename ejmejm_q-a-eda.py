from collections import Counter

import json

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats

import seaborn as sns

import sentencepiece as spm

import tqdm



%matplotlib inline

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def data_generator(path, chunk_size=30000):

    curr_pos = 0

    last_line = False

    with open(path, 'rt') as f:

        while not last_line:

            df = []

            for i in range(curr_pos, curr_pos+chunk_size):

                line = f.readline()

                if line is None:

                    last_line = True

                    break

                df.append(json.loads(line))

            curr_pos = i + 1

            yield pd.DataFrame(df)
data_path = '/kaggle/input/tensorflow2-question-answering/'

train_path = os.path.join(data_path, 'simplified-nq-train.jsonl')

test_path = os.path.join(data_path, 'simplified-nq-test.jsonl')



train_gen = data_generator(train_path, chunk_size=5000)



df = next(train_gen)

df.head()
# Uncomment for a demonstration of trainig and using a Sentencepiece model



# tmp_data_gen = data_generator(train_path, chunk_size=5000)

# with open('wiki_text.txt', 'w+') as f:

#     print('Generating corpus for sentencepiece model...')

#     for i in tqdm.tqdm(range(5)):

#         docs = next(tmp_data_gen)['document_text']

#         for text in docs:

#             f.write(text + '\n')

        

# sp = spm.SentencePieceTrainer.Train('--input=wiki_text.txt --model_prefix=test_sp --vocab_size=8000 --character_coverage=1.0 --model_type=unigram')



# sp = spm.SentencePieceProcessor()

# sp.Load('test_sp.model');



# print(sp.EncodeAsIds('this is a test'))

# print(sp.encode_as_pieces('this is a test'))

# print(sp.encode_as_pieces('that is a test'))

# print(sp.encode_as_pieces('Some of the tokenizing here is quite strange, but I guess itll be okay :)'))

# print(sp.decode_ids([1, 2, 3]))
# Split documents up into informal tokens for analysis

df['tokens'] = df['document_text'].apply(lambda x: [w.lower() for w in x.split(' ')])
word_counts = {}

for tokens in df['tokens']:

    for token in tokens:

        if token in word_counts:

            word_counts[token] += 1

        else:

            word_counts[token] = 1

            

top_word_counts = sorted(word_counts.items(), key=lambda i: i[1], reverse=True)
plt.figure(figsize=(12, 5))

plt.title('Most Common Tokens')

plt.bar(x=[x[0] for x in top_word_counts[:20]], height=[x[1] for x in top_word_counts[:20]])
plt.figure(figsize=(12, 5))

plt.title('Words Per Page')

sns.distplot(df['tokens'].apply(len).values, kde=False);
n_tags = df['tokens'].apply(lambda x: sum([1 if t.startswith('<') and t.endswith('>') else 0 for t in x])).values

tag_ratios = n_tags / df['tokens'].apply(len)



plt.figure(figsize=(12, 5))

plt.title('Tag Percentage Distribution')

sns.distplot(tag_ratios, kde=False)
plt.figure(figsize=(12, 5))

plt.title('Number of Long Answers Candidates')

plt.xlim(0, 1000)

sns.distplot(df['long_answer_candidates'].apply(len), kde=False);
plt.figure(figsize=(12, 6))

plt.title('Number of Tokens per Long Answer Candidate')

sns.distplot(df['tokens'].apply(len) / df['long_answer_candidates'].apply(len), kde=False);
plt.figure(figsize=(12, 6))

plt.title('Question Text Length')

sns.distplot(df['question_text'].apply(lambda s: s.split(' ')).apply(len), kde=False);
fig = plt.figure(figsize=(12, 6))

plt.title('Long Answer Text Log Length')



long_answer_lengths = df.apply(lambda row: row['annotations'][0]['long_answer']['end_token'] - \

                               row['annotations'][0]['long_answer']['start_token'], axis=1).values

long_answer_lengths = [x for x in long_answer_lengths if x != 0]



# fig.axes[0].set_xscale('log')



sns.distplot(np.log(long_answer_lengths), kde=False);
plt.figure(figsize=(12, 6))

plt.title('Short Answer Text Length')



short_answer_lengths = df.apply(

    lambda row: [x['end_token'] - x['start_token'] for x in row['annotations'][0]['short_answers']],

    axis=1).values

short_answer_lengths = np.concatenate(short_answer_lengths)



sns.distplot(short_answer_lengths, kde=False);
fig = plt.figure(figsize=(12, 6))

plt.title('Short Answer Text Length')



sns.boxplot(short_answer_lengths);
print(stats.describe(short_answer_lengths))

print(np.quantile(short_answer_lengths, 0.99))

for i in range(10):

    print(df['annotations'][i])
df['annotations'][0][0]
# Every annotations entry is a list of rank 1

sum([1 if x != 1 else 0 for x in df['annotations'].apply(len)])
f = 0

for i in range(len(df)):

    print(i)

    print(df['annotations'][i][0])

    f += 1

    if f >= 5:

        break



print('-------------------------------------------')



f = 0

for i in range(len(df)):

    if df['annotations'][i][0]['yes_no_answer'] != 'NONE':

        print(i)

        print(df['annotations'][i][0])

        f += 1

    if f >= 5:

        break

        

print('-------------------------------------------')

        

f = 0

for i in range(len(df)):

    if len(df['annotations'][i][0]['short_answers']) > 1:

        print(i)

        print(df['annotations'][i][0])

        f += 1

    if f >= 5:

        break



print('-------------------------------------------')



# It looks like a short answer will probably only exist if a long answer also exists

f = 0

for i in range(len(df)):

    if len(df['annotations'][i][0]['short_answers']) >= 1 and df['annotations'][i][0]['long_answer']['start_token'] == -1:

        print(i)

        print(df['annotations'][i][0])

        f += 1

    if f >= 5:

        break



print('-------------------------------------------')



# It looks like a YES/NO will probably only exist if a long answer also exists

f = 0

for i in range(len(df)):

    if df['annotations'][i][0]['yes_no_answer'] != 'NONE' and df['annotations'][i][0]['long_answer']['start_token'] == -1:

        print(i)

        print(df['annotations'][i][0])

        f += 1

    if f >= 5:

        break
y_count = 0

n_count = 0

la_count = 0

sa_count = 0

none_count = 0

for an in df['annotations']:

    an = an[0]

    none = True

    if an['yes_no_answer'] == 'YES':

        y_count += 1

        none = False

    elif an['yes_no_answer'] == 'NO':

        n_count += 1

        none = False

        

    if an['long_answer']['start_token'] != -1:

        la_count += 1

        none = False

        

    if len(an['short_answers']) > 0:

        sa_count += 1

        none = False

        

    if none:

        none_count += 1

        

n = float(len(df))



plt.figure(figsize=(12, 6))

plt.title('Answer Possibilities by Category')

plt.bar(x=['% Yes', '% No', '% Long Answer', '% Short Answer', '% No Answer'],

        height=[y_count/n, n_count/n, la_count/n, sa_count/n, none_count/n]);
df['has_long_answer'] = df['annotations'].apply(lambda x: x[0]['long_answer']['start_token'] > -1)

df['has_short_answer'] = df['annotations'].apply(lambda x: len(x[0]['short_answers']) > 0)

df['has_yn_answer'] = df['annotations'].apply(lambda x: x[0]['yes_no_answer'] != 'NONE')



la_with_answer_counts = []

la_no_answer_counts = []



for i, row in df.iterrows():

    n_candidates = len(row['long_answer_candidates'])

    if row['has_long_answer']:

        la_with_answer_counts.append(len(row['long_answer_candidates']))

    else:

        la_no_answer_counts.append(len(row['long_answer_candidates']))

    

t, p = stats.ttest_ind(la_with_answer_counts, la_no_answer_counts)

print(f'p-val: {p} | t-stat: {t}')

if p > 0.05:

    print('No significant difference between distributions')

else:

    print('Significant difference between distributions')

    



plt.figure(figsize=(6, 10))

plt.ylim(0, 1200)

plt.title('Amount of Long Answer Candidates for Questions With and Without Answers')

plt.boxplot([la_no_answer_counts, la_with_answer_counts]);
df['n_article_tokens'] = df['tokens'].apply(len)

df['n_candidate_long_answers'] = df['long_answer_candidates'].apply(len)



df['has_long_answer_int'] = df['has_long_answer'].apply(np.int8)

df['has_short_answer_int'] = df['has_short_answer'].apply(np.int8)

df['has_yn_answer_int'] = df['has_yn_answer'].apply(np.int8)



sns.pairplot(df, vars=['n_article_tokens', 'n_candidate_long_answers', 'has_long_answer_int', 'has_short_answer_int', 'has_yn_answer_int'],

             kind='scatter', size=3, plot_kws={'alpha': 0.075});
for i, row in df.iterrows():

    if not row['has_short_answer'] or not row['has_long_answer']:

        continue

    

    annotations = row['annotations'][0]

    long_range = (annotations['long_answer']['start_token'], annotations['long_answer']['end_token'])

    for short_ans in annotations['short_answers']:

        if not (short_ans['start_token'] >= long_range[0] and short_ans['end_token'] <= long_range[1]):

            print(f'**{i}**')

            print('Short Answer: ' + ' '.join(df['tokens'][short_ans['start_token']:short_ans['end_token']]))

            print('Long Answer: ' + ' '.join(df['tokens'][long_range[0]:long_range[1]]))