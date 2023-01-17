from pathlib import Path
import warnings
from collections import Counter
import regex
import tqdm
from tqdm import tqdm, tqdm_notebook

import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (8, 6)
warnings.filterwarnings('ignore')

#tqdm_notebook().pandas()
tqdm().pandas()

DATA_PATH = Path('../input/')
train_df = pd.read_csv(DATA_PATH/'train.csv')
train_df.head()
test_df = pd.read_csv(DATA_PATH/'test.csv')
test_df.head()
train_df.info()
train_df['target'] = train_df['target'].astype('category')
train_df.info()
train_df.describe(include='all')
test_df.info()
test_df.describe(include='all')
# Duplicated question texts
train_df.duplicated('question_text').sum(), test_df.duplicated('question_text').sum()
X_train = train_df.copy()
X_test = test_df.copy()
X_train['target'].value_counts(normalize=True)
X_train.head()
X_test['qid'].str.len().unique()
X_train.groupby(X_train['qid'].str[:1])['target'].apply(lambda x: np.sum(x.astype('int8')) / len(x)).rolling(1).mean().plot()
X_train.groupby(X_train['qid'].str[:2])['target'].apply(lambda x: np.sum(x.astype('int8')) / len(x)).rolling(10).mean().plot()
X_train.groupby(X_train['qid'].str[:3])['target'].apply(lambda x: np.sum(x.astype('int8')) / len(x)).rolling(100).mean().plot()
X_train.groupby(X_train['qid'].str[:4])['target'].apply(lambda x: np.sum(x.astype('int8')) / len(x)).rolling(1000).mean().plot()
X_train['question_length'] = X_train['question_text'].str.len()
X_test['question_length'] = X_test['question_text'].str.len()
fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
sns.boxplot('question_length', data=X_train, ax=axes[0])
sns.boxplot('question_length', data=X_test, ax=axes[1])
fig, axes = plt.subplots(figsize=(16, 6), sharex=True)
sns.kdeplot(X_train['question_length'], label='Train')
sns.kdeplot(X_test['question_length'], label='Test')
agg_stats = ['min', 'max', 'mean', 'std', 'median']
X_train['question_length'].agg(agg_stats)
X_test['question_length'].agg(agg_stats)
sns.boxplot('question_length', 'target', data=X_train, orient='h')
X_train.loc[X_train['question_length'] < 11, 'target'].value_counts(normalize=True)
X_train.loc[X_train['question_length'] > 588, 'target'].value_counts(normalize=True)
questions_outliers = X_train.loc[(X_train['question_length'] < 11) | (X_train['question_length'] > 588)]
questions_outliers
# Original questions lengths distribution
plt.figure(figsize=(14, 6))
sns.kdeplot(X_train.loc[X_train['target'] == 0, 'question_length'], label='Train - good')
sns.kdeplot(X_train.loc[X_train['target'] == 1, 'question_length'], label='Train - bad')
sns.kdeplot(X_train['question_length'], label='Train')
sns.kdeplot(X_test['question_length'], label='Test')
plt.show()
X_train.groupby('target')['question_length'].agg(agg_stats)
plt.figure(figsize=(14, 6))
X_train.groupby('question_length')['target'].apply(lambda x: np.sum(x.astype('int')) / len(x)).plot()
sns.kdeplot(X_train['question_length'], label='Train')
plt.show()
plt.figure(figsize=(14, 6))
X_train.groupby('question_length')['target'].apply(lambda x: np.sum(x.astype('int')) / len(x)).rolling(10).mean().plot()
sns.kdeplot(X_train['question_length'], label='Train')
plt.show()
train_question_lengths_bins = pd.cut(X_train['question_length'], bins=5, include_lowest=True)
X_train.groupby(train_question_lengths_bins)['target'].apply(lambda x: np.sum(x.astype('int8')) / len(x))
X_train[(X_train['question_length'] < 10) | (X_train['question_length'] > 800)]
X_test.loc[(X_test['question_length'] < 11) | (X_test['question_length'] > 588)]
plt.figure(figsize=(14, 6))
sns.kdeplot(X_train.loc[X_train['target'] == 0, 'question_length'], label='Train - good')
# sns.kdeplot(X_train.loc[X_train['target'] == 1, 'question_length'], label='Train - bad')
sns.kdeplot(X_train.loc[(X_train['question_length'] <= 588) & (X_train['question_length'] >= 11),
                        'question_length'], label='Train - bad (no outliers)')
sns.kdeplot(X_train['question_length'], label='Train')
sns.kdeplot(X_test['question_length'], label='Test')
plt.show()
X_train['first_word'] = X_train['question_text'].str.extract('(.*?)\s.*') 
X_test['first_word'] = X_test['question_text'].str.extract('(.*?)\s.*')
X_train['first_word'].value_counts(normalize=True).head()
X_train[X_train['first_word'].isnull()]
X_test['first_word'].value_counts(normalize=True).head()
X_test[X_test['first_word'].isnull()]
first_words_0 = X_train.loc[X_train['target'] == 0, 'first_word'].value_counts(normalize=True)
first_words_1 = X_train.loc[X_train['target'] == 1, 'first_word'].value_counts(normalize=True)
first_words_df = pd.concat([first_words_0, first_words_1], axis=1, join='outer')

first_words_df.columns = ['good', 'bad']
first_words_df.head()
first_words_df.sort_values('good', ascending=False)[['good']].T
first_words_df.sort_values('bad', ascending=False)[['bad']].T
def color_cells(value):
    if value < 0:
        color = 'red'
    elif value > 0:
        color = 'green'
    else:
        color = 'black'

    return f'color: {color}'
# Top first words with maximum absolute proportion difference by target
first_words_df['diff'] = first_words_df['bad'] - first_words_df['good']
first_words_df['abs_diff'] = np.abs(first_words_df['bad'] - first_words_df['good'])

top_first_words_df = first_words_df[first_words_df['abs_diff'] >= 0.01].sort_values('diff', ascending=False)
top_first_words_df.style.applymap(color_cells, subset=['diff'])
top_first_words_df['diff'][::-1].plot.barh();
first_words_df[first_words_df['bad'].isnull()].sort_values('good', ascending=False)[['good']].T
good_only_first_words = first_words_df[first_words_df['bad'].isnull()].sort_values('good', ascending=False)[['good']].index
good_only_first_words
first_words_df[first_words_df['good'].isnull()].sort_values('bad', ascending=False)[['bad']].T
bad_only_first_words = first_words_df[first_words_df['good'].isnull()].sort_values('bad', ascending=False)[['bad']].index
bad_only_first_words
X_test[X_test['first_word'].isin(good_only_first_words)].head()
X_test[X_test['first_word'].isin(bad_only_first_words)].head()
X_train['last_char'] = X_train['question_text'].str[-1]
X_test['last_char'] = X_test['question_text'].str[-1]
X_train['last_char'].value_counts(normalize=True).head()
X_test['last_char'].value_counts(normalize=True).head()
last_chars_0 = X_train.loc[X_train['target'] == 0, 'last_char'].value_counts(normalize=True)
last_chars_1 = X_train.loc[X_train['target'] == 1, 'last_char'].value_counts(normalize=True)
last_chars_df = pd.concat([last_chars_0, last_chars_1], axis=1, join='outer')

last_chars_df.columns = ['good', 'bad']
last_chars_df.head()
last_chars_df.sort_values('good', ascending=False)[['good']].T
last_chars_df.sort_values('bad', ascending=False)[['bad']].T
last_chars_df['diff'] = last_chars_df['bad'] - last_chars_df['good']
last_chars_df['abs_diff'] = np.abs(last_chars_df['bad'] - last_chars_df['good'])

top_last_chars_df = last_chars_df[last_chars_df['abs_diff'] > 0.001].sort_values('diff', ascending=False)
top_last_chars_df.style.applymap(color_cells, subset=['diff'])
top_last_chars_df['diff'][::-1].plot.barh()
last_chars_df[last_chars_df['bad'].isnull()].sort_values('good', ascending=False)[['good']].T
good_only_last_chars = last_chars_df[last_chars_df['bad'].isnull()].sort_values('good', ascending=False)[['good']].index.values
good_only_last_chars
last_chars_df[last_chars_df['good'].isnull()].sort_values('bad', ascending=False)[['bad']].T
bad_only_last_chars = last_chars_df[last_chars_df['good'].isnull()].sort_values('bad', ascending=False)[['bad']].index.values
bad_only_last_chars
train_chars_freq = Counter(X_train['question_text'].str.cat())
train_good_chars_freq = Counter(X_train.loc[X_train['target'] == 0, 'question_text'].str.cat())
train_bad_chars_freq = Counter(X_train.loc[X_train['target'] == 1, 'question_text'].str.cat())
test_chars_freq = Counter(X_test['question_text'].str.cat())
train_chars_len = len(X_train['question_text'].str.cat())
train_good_chars_len = len(X_train.loc[X_train['target'] == 0, 'question_text'].str.cat())
train_bad_chars_len = len(X_train.loc[X_train['target'] == 1, 'question_text'].str.cat())
test_chars_len = len(X_test['question_text'].str.cat())
len(set(train_chars_freq)), len(set(train_good_chars_freq)), len(set(train_bad_chars_freq)), len(set(test_chars_freq))
train_chars = set(train_chars_freq)
train_good_chars = set(train_good_chars_freq)
train_bad_chars = set(train_bad_chars_freq)
test_chars = set(test_chars_freq)
print(sorted(train_chars - test_chars))
print(sorted(test_chars - train_chars))
print(sorted(train_good_chars - train_bad_chars))
print(sorted(set(train_bad_chars) - set(train_good_chars)))
train_chars_df = pd.Series(dict(train_chars_freq)) / train_chars_len
test_chars_df = pd.Series(dict(test_chars_freq)) / test_chars_len
chars_df = pd.concat([train_chars_df, test_chars_df], axis=1, join='outer')
chars_df.columns = ['train', 'test']
chars_df.head()
chars_df.sort_values('train', ascending=False)[['train']].T
chars_df.sort_values('test', ascending=False)[['test']].T
chars_df['diff'] = chars_df['train'] - chars_df['test']
chars_df['abs_diff'] = np.abs(chars_df['train'] - chars_df['test'])

top_chars_df = chars_df[chars_df['abs_diff'] > 0.0001].sort_values('diff', ascending=False)
top_chars_df.style.applymap(color_cells, subset=['diff'])
top_chars_df['diff'][::-1].plot.barh()
chars_df[chars_df['test'].isnull()].sort_values('train', ascending=False)[['train']].T
train_only_chars = chars_df[chars_df['test'].isnull()].sort_values('train', ascending=False)[['train']].index.values.tolist()
print(train_only_chars)
chars_df[chars_df['train'].isnull()].sort_values('test', ascending=False)[['test']].T
test_only_chars = chars_df[chars_df['train'].isnull()].sort_values('test', ascending=False)[['test']].index.values
test_only_chars
train_good_chars_df = pd.Series(dict(train_good_chars_freq)) / train_good_chars_len
train_bad_chars_df = pd.Series(dict(train_bad_chars_freq)) / train_bad_chars_len
chars_df = pd.concat([train_good_chars_df, train_bad_chars_df], axis=1, join='outer')
chars_df.columns = ['good', 'bad']
chars_df.head()
chars_df.sort_values('good', ascending=False)[['good']].T
chars_df.sort_values('bad', ascending=False)[['bad']].T
chars_df['diff'] = chars_df['bad'] - chars_df['good']
chars_df['abs_diff'] = np.abs(chars_df['bad'] - chars_df['good'])

top_chars_df = chars_df[chars_df['abs_diff'] > 0.001].sort_values('diff', ascending=False)
top_chars_df.style.applymap(color_cells, subset=['diff'])
top_chars_df['diff'][::-1].plot.barh()
chars_df[chars_df['bad'].isnull()].sort_values('good', ascending=False)[['good']].T
good_only_chars = chars_df[chars_df['bad'].isnull()].sort_values('good', ascending=False)[['good']].index.values.tolist()
print(good_only_chars)
chars_df[chars_df['good'].isnull()].sort_values('bad', ascending=False)[['bad']].T
bad_only_chars = chars_df[chars_df['good'].isnull()].sort_values('bad', ascending=False)[['bad']].index.values
bad_only_chars
X_train['question_text'].loc[1]
# Any kind of letter from any language
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Letter}', x)).head()
# A lowercase letter that has an uppercase variant
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Lowercase_Letter}', x)).head()
# An uppercase letter that has a lowercase variant
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Uppercase_Letter}', x)).head()
# Alternative method for title-case words
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\b\p{Uppercase_Letter}\p{Lowercase_Letter}+\b', x)).head()
# A letter that exists in lowercase and uppercase variants (combination of Ll, Lu and Lt)
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Cased_Letter}', x)).head()
# A special character that is used like a letter
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Modifier_Letter}', x)).loc[[14498, 18460, 81490]]
# A letter or ideograph that does not have lowercase and uppercase variants
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Letter}', x)).loc[[905, 1360, 2194]]
# A character intended to be combined with another character (e.g. accents, umlauts, enclosing boxes, etc.)
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Mark}', x)).loc[[441, 2194, 9479]]
# A character intended to be combined with another character without taking up extra space (e.g. accents, umlauts, etc.)
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Non_Spacing_Mark}', x)).loc[[441, 2194, 9479]]
# Any kind of whitespace or invisible separator
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Separator}', x)).head()
# A whitespace character that is invisible, but does take up space
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Space_Separator}', x)).head()
# Math symbols, currency signs, dingbats, box-drawing characters, etc.
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Symbol}', x)).loc[[83, 94, 97]]
# Any mathematical symbol
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Math_Symbol}', x)).loc[[94, 97, 274]]
# Any currency sign
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Currency_Symbol}', x)).loc[[83, 345, 1897]]
# A combining character (mark) as a full character on its own
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Modifier_Symbol}', x)).loc[[540, 1035, 1322]]
# Various symbols that are not math symbols, currency signs, or combining characters
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Symbol}', x)).loc[[2805, 2922, 4480]]
# Any kind of numeric character in any script
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Number}', x)).loc[[0, 14, 27]]
# A number that looks like a letter, such as a Roman numeral
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Letter_Number}', x)).loc[887588]
# A superscript or subscript digit, or a number that is not a digit 0–9 (excluding numbers from ideographic scripts)
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Number}+', x)).loc[[3605, 12830, 29910]]
# Any kind of punctuation character
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Punctuation}', x)).head()
# Any kind of hyphen or dash
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Dash_Punctuation}', x)).loc[[33, 48, 94]]
# Any kind of opening bracket
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Open_Punctuation}', x)).loc[[35, 53, 86]]
# Any kind of closing bracket
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Close_Punctuation}', x)).loc[[35, 53, 86]]
# Any kind of opening quote
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Initial_Punctuation}', x)).loc[[903, 986, 1075]]
# Any kind of closing quote
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Final_Punctuation}', x)).loc[[173, 260, 317]]
# A punctuation character such as an underscore that connects words
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Connector_Punctuation}', x)).loc[[7331, 14208, 21426]]
# Any kind of punctuation character that is not a dash, bracket, quote or connector
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Punctuation}', x)).head()
# Invisible control characters and unused code points
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other}', x)).loc[[1048, 1655, 4572]].values
# An ASCII or Latin-1 control character: 0x00–0x1F and 0x7F–0x9F
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Control}', x)).loc[[344457, 522266, 566346]]
# Invisible formatting indicator
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Format}', x)).loc[[1048, 1655, 4572]].values
# Invisible formatting indicator
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Private_Use}', x)).loc[[814877, 1135120, 1192382]].values

train_question_length = X_train['question_length']
test_question_length = X_test['question_length']
X_train['prop_category_letter'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Letter}', x)).str.len() / train_question_length
X_train['prop_category_lowercase_letter'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Lowercase_Letter}', x)).str.len() / train_question_length
X_train['prop_category_uppercase_letter'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Uppercase_Letter}', x)).str.len() / train_question_length
X_train['prop_category_titlecase_letter'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\b\p{Uppercase_Letter}\p{Lowercase_Letter}+\b', x)).str.len() / train_question_length
X_train['prop_category_cased_letter'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Cased_Letter}', x)).str.len() / train_question_length
X_train['prop_category_modifier_letter'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Modifier_Letter}', x)).str.len() / train_question_length
X_train['prop_category_other_letter'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Letter}', x)).str.len() / train_question_length
X_train['prop_category_mark'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Mark}', x)).str.len() / train_question_length
X_train['prop_category_non_spacing_mark'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Non_Spacing_Mark}', x)).str.len() / train_question_length
X_train['prop_category_separator'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Separator}', x)).str.len() / train_question_length
X_train['prop_category_space_separator'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Space_Separator}', x)).str.len() / train_question_length
X_train['prop_category_symbol'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Symbol}', x)).str.len() / train_question_length
X_train['prop_category_math_symbol'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Math_Symbol}', x)).str.len() / train_question_length
X_train['prop_category_currency_symbol'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Currency_Symbol}', x)).str.len() / train_question_length
X_train['prop_category_modifier_symbol'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Modifier_Symbol}', x)).str.len() / train_question_length
X_train['prop_category_other_symbol'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Symbol}', x)).str.len() / train_question_length
X_train['prop_category_number'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Number}', x)).str.len() / train_question_length
X_train['prop_category_letter_number'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Letter_Number}', x)).str.len() / train_question_length
X_train['prop_category_other_number'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Number}', x)).str.len() / train_question_length
X_train['prop_category_punctuation'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Punctuation}', x)).str.len() / train_question_length
X_train['prop_category_dash_punctuation'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Dash_Punctuation}', x)).str.len() / train_question_length
X_train['prop_category_open_punctuation'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Open_Punctuation}', x)).str.len() / train_question_length
X_train['prop_category_close_punctuation'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Close_Punctuation}', x)).str.len() / train_question_length
X_train['prop_category_initial_punctuation'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Initial_Punctuation}', x)).str.len() / train_question_length
X_train['prop_category_final_punctuation'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Final_Punctuation}', x)).str.len() / train_question_length
X_train['prop_category_connector_punctuation'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Connector_Punctuation}', x)).str.len() / train_question_length
X_train['prop_category_other_punctuation'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Punctuation}', x)).str.len() / train_question_length
X_train['prop_category_other'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other}', x)).str.len() / train_question_length
X_train['prop_category_control'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Control}', x)).str.len() / train_question_length
X_train['prop_category_format'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Format}', x)).str.len() / train_question_length
X_train['prop_category_format'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Private_Use}', x)).str.len() / train_question_length
X_test['prop_category_letter'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Letter}', x)).str.len() / test_question_length
X_test['prop_category_lowercase_letter'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Lowercase_Letter}', x)).str.len() / test_question_length
X_test['prop_category_uppercase_letter'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Uppercase_Letter}', x)).str.len() / test_question_length
X_test['prop_category_titlecase_letter'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\b\p{Uppercase_Letter}\p{Lowercase_Letter}+\b', x)).str.len() / test_question_length
X_test['prop_category_cased_letter'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Cased_Letter}', x)).str.len() / test_question_length
X_test['prop_category_modifier_letter'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Modifier_Letter}', x)).str.len() / test_question_length
X_test['prop_category_other_letter'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Letter}', x)).str.len() / test_question_length
X_test['prop_category_mark'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Mark}', x)).str.len() / test_question_length
X_test['prop_category_non_spacing_mark'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Non_Spacing_Mark}', x)).str.len() / test_question_length
X_test['prop_category_separator'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Separator}', x)).str.len() / test_question_length
X_test['prop_category_space_separator'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Space_Separator}', x)).str.len() / test_question_length
X_test['prop_category_symbol'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Symbol}', x)).str.len() / test_question_length
X_test['prop_category_math_symbol'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Math_Symbol}', x)).str.len() / test_question_length
X_test['prop_category_currency_symbol'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Currency_Symbol}', x)).str.len() / test_question_length
X_test['prop_category_modifier_symbol'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Modifier_Symbol}', x)).str.len() / test_question_length
X_test['prop_category_other_symbol'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Symbol}', x)).str.len() / test_question_length
X_test['prop_category_number'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Number}', x)).str.len() / test_question_length
X_test['prop_category_letter_number'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Letter_Number}', x)).str.len() / test_question_length
X_test['prop_category_other_number'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Number}', x)).str.len() / test_question_length
X_test['prop_category_punctuation'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Punctuation}', x)).str.len() / test_question_length
X_test['prop_category_dash_punctuation'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Dash_Punctuation}', x)).str.len() / test_question_length
X_test['prop_category_open_punctuation'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Open_Punctuation}', x)).str.len() / test_question_length
X_test['prop_category_close_punctuation'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Close_Punctuation}', x)).str.len() / test_question_length
X_test['prop_category_initial_punctuation'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Initial_Punctuation}', x)).str.len() / test_question_length
X_test['prop_category_final_punctuation'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Final_Punctuation}', x)).str.len() / test_question_length
X_test['prop_category_connector_punctuation'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Connector_Punctuation}', x)).str.len() / test_question_length
X_test['prop_category_other_punctuation'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other_Punctuation}', x)).str.len() / test_question_length
X_test['prop_category_other'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Other}', x)).str.len() / test_question_length
X_test['prop_category_control'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Control}', x)).str.len() / test_question_length
X_test['prop_category_format'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Format}', x)).str.len() / test_question_length
X_test['prop_category_format'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Private_Use}', x)).str.len() / test_question_length

X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Common}', x)).head()
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Arabic}', x)).loc[[3135, 24545, 46644]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Armenian}', x)).loc[[157715, 706999]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Bengali}', x)).loc[[83908, 226715, 511566]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Canadian_Aboriginal}', x)).loc[[919693]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Cyrillic}', x)).loc[[19906, 22628, 37574]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Devanagari}', x)).loc[[905, 2194, 26412]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Ethiopic}', x)).loc[[236161, 936861]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Greek}', x)).loc[[7570, 12077, 13683]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Gujarati}', x)).loc[87519]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Gurmukhi}', x)).loc[[286009, 518150, 672715]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Han}', x)).loc[[1360, 9457, 15165]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Hangul}', x)).loc[[56312, 134870, 138183]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Hebrew}', x)).loc[[97812, 259482, 345541]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Hiragana}', x)).loc[[18235, 101614, 109595]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Inherited}', x)).loc[[441, 9479, 19836]].values
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Kannada}', x)).loc[[232429, 1002928]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Katakana}', x)).loc[[81490, 109595, 219625]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Khmer}', x)).loc[[391253]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Latin}', x)).head()
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Malayalam}', x)).loc[[633873]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Mongolian}', x)).loc[[163714]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Myanmar}', x)).loc[[619484]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Oriya}', x)).loc[[468196, 769870]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Tamil}', x)).loc[[46721, 148815, 262266]]
X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Thai}', x)).loc[[41733, 403251, 782676]]
X_train['prop_script_common'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Common}', x)).str.len() / train_question_length
X_train['prop_script_arabic'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Arabic}', x)).str.len() / train_question_length
X_train['prop_script_armenian'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Armenian}', x)).str.len() / train_question_length
X_train['prop_script_bengali'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Bengali}', x)).str.len() / train_question_length
X_train['prop_script_canadian_aboriginal'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Canadian_Aboriginal}', x)).str.len() / train_question_length
X_train['prop_script_cyrillic'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Cyrillic}', x)).str.len() / train_question_length
X_train['prop_script_devanagari'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Devanagari}', x)).str.len() / train_question_length
X_train['prop_script_ethiopic'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Ethiopic}', x)).str.len() / train_question_length
X_train['prop_script_greek'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Greek}', x)).str.len() / train_question_length
X_train['prop_script_gujarati'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Gujarati}', x)).str.len() / train_question_length
X_train['prop_script_gurmukhi'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Gurmukhi}', x)).str.len() / train_question_length
X_train['prop_script_han'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Han}', x)).str.len() / train_question_length
X_train['prop_script_hangul'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Hangul}', x)).str.len() / train_question_length
X_train['prop_script_hebrew'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Hebrew}', x)).str.len() / train_question_length
X_train['prop_script_hiragana'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Hiragana}', x)).str.len() / train_question_length
X_train['prop_script_inherited'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Inherited}', x)).str.len() / train_question_length
X_train['prop_script_kannada'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Kannada}', x)).str.len() / train_question_length
X_train['prop_script_katakana'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Katakana}', x)).str.len() / train_question_length
X_train['prop_script_khmer'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Khmer}', x)).str.len() / train_question_length
X_train['prop_script_latin'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Latin}', x)).str.len() / train_question_length
X_train['prop_script_malayalam'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Malayalam}', x)).str.len() / train_question_length
X_train['prop_script_mongolian'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Mongolian}', x)).str.len() / train_question_length
X_train['prop_script_myanmar'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Myanmar}', x)).str.len() / train_question_length
X_train['prop_script_oriya'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Oriya}', x)).str.len() / train_question_length
X_train['prop_script_tamil'] = X_train['question_text'].progress_apply(lambda x: regex.findall(r'\p{Tamil}', x)).str.len() / train_question_length
X_test['prop_script_common'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Common}', x)).str.len() / test_question_length
X_test['prop_script_arabic'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Arabic}', x)).str.len() / test_question_length
X_test['prop_script_armenian'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Armenian}', x)).str.len() / test_question_length
X_test['prop_script_bengali'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Bengali}', x)).str.len() / test_question_length
X_test['prop_script_canadian_aboriginal'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Canadian_Aboriginal}', x)).str.len() / test_question_length
X_test['prop_script_cyrillic'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Cyrillic}', x)).str.len() / test_question_length
X_test['prop_script_devanagari'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Devanagari}', x)).str.len() / test_question_length
X_test['prop_script_ethiopic'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Ethiopic}', x)).str.len() / test_question_length
X_test['prop_script_greek'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Greek}', x)).str.len() / test_question_length
X_test['prop_script_gujarati'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Gujarati}', x)).str.len() / test_question_length
X_test['prop_script_gurmukhi'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Gurmukhi}', x)).str.len() / test_question_length
X_test['prop_script_han'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Han}', x)).str.len() / test_question_length
X_test['prop_script_hangul'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Hangul}', x)).str.len() / test_question_length
X_test['prop_script_hebrew'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Hebrew}', x)).str.len() / test_question_length
X_test['prop_script_hiragana'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Hiragana}', x)).str.len() / test_question_length
X_test['prop_script_inherited'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Inherited}', x)).str.len() / test_question_length
X_test['prop_script_kannada'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Kannada}', x)).str.len() / test_question_length
X_test['prop_script_katakana'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Katakana}', x)).str.len() / test_question_length
X_test['prop_script_khmer'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Khmer}', x)).str.len() / test_question_length
X_test['prop_script_latin'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Latin}', x)).str.len() / test_question_length
X_test['prop_script_malayalam'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Malayalam}', x)).str.len() / test_question_length
X_test['prop_script_mongolian'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Mongolian}', x)).str.len() / test_question_length
X_test['prop_script_myanmar'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Myanmar}', x)).str.len() / test_question_length
X_test['prop_script_oriya'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Oriya}', x)).str.len() / test_question_length
X_test['prop_script_tamil'] = X_test['question_text'].progress_apply(lambda x: regex.findall(r'\p{Tamil}', x)).str.len() / test_question_length

train_binary_categories_df = (X_train.filter(like='prop_category') > 0).astype('int').astype('category')
train_binary_categories_df.columns = train_binary_categories_df.columns.str.replace(r'prop_', 'is_')
train_binary_scripts_df = (X_train.filter(like='prop_script') > 0).astype('int').astype('category')
train_binary_scripts_df.columns = train_binary_scripts_df.columns.str.replace(r'prop_', 'is_')

X_train = pd.merge(X_train, train_binary_categories_df, left_index=True, right_index=True)
X_train = pd.merge(X_train, train_binary_scripts_df, left_index=True, right_index=True)
X_train.info()
test_binary_categories_df = (X_test.filter(like='prop_category') > 0).astype('int').astype('category')
test_binary_categories_df.columns = test_binary_categories_df.columns.str.replace(r'prop_', 'is_')
test_binary_scripts_df = (X_test.filter(like='prop_script') > 0).astype('int').astype('category')
test_binary_scripts_df.columns = test_binary_scripts_df.columns.str.replace(r'prop_', 'is_')

X_test = pd.merge(X_test, test_binary_categories_df, left_index=True, right_index=True)
X_test = pd.merge(X_test, test_binary_scripts_df, left_index=True, right_index=True)
X_test.info()
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
fig.suptitle('Train/Test mean unicode categories/scripts proportion distribution (absolute values)', y=1.05, fontsize=14)

X_train.filter(like='prop_category').mean().sort_values(ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[0])
X_test.filter(like='prop_category').mean().sort_values(ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[1])
axes[0].set_title('Train - Unicode Categories Proportion (log10)')
axes[1].set_title('Test - Unicode Categories Proportion (log10)')
plt.tight_layout()

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
X_train.filter(like='prop_script').mean().sort_values(ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[0])
X_test.filter(like='prop_script').mean().sort_values(ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[1])
axes[0].set_title('Train - Unicode Scripts Proportion (log10)')
axes[1].set_title('Test - Unicode Scripts Proportion (log10)')
plt.tight_layout()
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
fig.suptitle('Train/Test mean unicode categories/scripts boolean distribution (absolute values)', y=1.05, fontsize=14)

X_train.filter(like='is_category').mean().sort_values(ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[0])
X_test.filter(like='is_category').mean().sort_values(ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[1])
axes[0].set_title('Train - Unicode Categories Boolean (log10)')
axes[1].set_title('Test - Unicode Categories Boolean (log10)')
plt.tight_layout()

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
X_train.filter(like='is_script').mean().sort_values(ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[0])
X_test.filter(like='is_script').mean().sort_values(ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[1])
axes[0].set_title('Train - Unicode Scripts Boolean (log10)')
axes[1].set_title('Test - Unicode Scripts Boolean (log10)')
plt.tight_layout()
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
fig.suptitle('Train mean unicode categories/scripts proportion/boolean distribution by target (absolute values)', y=1.05, fontsize=14)

X_train.filter(like='prop_category').groupby(X_train['target']).mean().T.sort_values(1, ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[0])
X_train.filter(like='is_category').astype('int').groupby(X_train['target']).mean().T.sort_values(1, ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[1])
axes[0].set_title('Train - Unicode Categories Proportion (log10)')
axes[1].set_title('Train - Unicode Categories Boolean (log10)')
plt.tight_layout()

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
X_train.filter(like='prop_script').groupby(X_train['target']).mean().T.sort_values(1, ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[0])
X_train.filter(like='is_script').astype('int').groupby(X_train['target']).mean().T.sort_values(1, ascending=False)[::-1].apply(np.log10).plot.barh(ax=axes[1])
axes[0].set_title('Train - Unicode Scripts Proportion (log10)')
axes[1].set_title('Train - Unicode Scripts Boolean (log10))')
plt.tight_layout()
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
fig.suptitle('Train mean unicode categories/scripts proportion/boolean distribution by target (relative values)', y=1.05, fontsize=14)

X_train.filter(like='prop_category').groupby(X_train['target']).mean().apply(lambda x: x / x.sum()).T.sort_values(1, ascending=False)[::-1].plot.barh(ax=axes[0])
X_train.filter(like='is_category').astype('int').groupby(X_train['target']).mean().apply(lambda x: x / x.sum()).T.sort_values(1, ascending=False)[::-1].plot.barh(ax=axes[1])
axes[0].set_title('Train - Unicode Categories Proportion')
axes[1].set_title('Train - Unicode Categories Boolean')
plt.tight_layout()

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
X_train.filter(like='prop_script').groupby(X_train['target']).mean().apply(lambda x: x / x.sum()).T.sort_values(1, ascending=False)[::-1].plot.barh(ax=axes[0])
X_train.filter(like='is_script').astype('int').groupby(X_train['target']).mean().apply(lambda x: x / x.sum()).T.sort_values(1, ascending=False)[::-1].plot.barh(ax=axes[1])
axes[0].set_title('Train - Unicode Scripts Proportion')
axes[1].set_title('Train - Unicode Scripts Boolean')
plt.tight_layout()
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True)
fig.suptitle('Train/test correlation of unicode categories/scripts proportion features', y=1.05, fontsize=14)

sns.heatmap(X_train.filter(like='prop_category').corr(), ax=axes[0])
sns.heatmap(X_test.filter(like='prop_category').corr(), ax=axes[1])
axes[0].set_title('Train - Unicode Categories Proportion')
axes[1].set_title('Test - Unicode Categories Proportion')
plt.tight_layout()

fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True)
sns.heatmap(X_train.filter(like='prop_script').corr(), ax=axes[0])
sns.heatmap(X_test.filter(like='prop_script').corr(), ax=axes[1])
axes[0].set_title('Train - Unicode Scripts Proportion')
axes[1].set_title('Test - Unicode Scripts Proportion')
plt.tight_layout()
fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True)
fig.suptitle('Train correlation of unicode categories/scripts proportion by target', y=1.05, fontsize=14)

sns.heatmap(X_train.filter(like='prop_category').groupby(X_train['target']).corr().loc[0], ax=axes[0])
sns.heatmap(X_train.filter(like='prop_category').groupby(X_train['target']).corr().loc[1], ax=axes[1])
axes[0].set_title('Train - Unicode Categories Proportion (Good)')
axes[1].set_title('Train - Unicode Categories Proportion (Bad)')
plt.tight_layout()

fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharex=True)
sns.heatmap(X_train.filter(like='prop_script').groupby(X_train['target']).corr().loc[0], ax=axes[0])
sns.heatmap(X_train.filter(like='prop_script').groupby(X_train['target']).corr().loc[1], ax=axes[1])
axes[0].set_title('Train - Unicode Scripts Proportion (Good)')
axes[1].set_title('Train - Unicode Scripts Proportion (Bad)')
plt.tight_layout()