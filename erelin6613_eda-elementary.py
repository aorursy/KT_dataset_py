import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 5);
sns.set_style('whitegrid')
root_dir = '../input/contradictory-my-dear-watson'
train_path = 'train.csv'
test_path = 'test.csv'
sub_path = 'sample_submission.csv'
train_df = pd.read_csv(os.path.join(root_dir, train_path))
test_df = pd.read_csv(os.path.join(root_dir, test_path))
train_df.head()
sorted(train_df.language.unique()) == sorted(test_df.language.unique())
train_df.language.value_counts()
train_df.label.hist(color='orange')
train_df.isna().sum()
for each in ['premise', 'hypothesis']:
    print(f'Mean symbols in {each}:', 
          train_df[each].apply(lambda x: len(x)).mean())
    print(f'Maximum symbols in {each}:', 
          train_df[each].apply(lambda x: len(x)).max())
    print(f'Minimum symbols in {each}:', 
          train_df[each].apply(lambda x: len(x)).min())
    print(f'Median symbols in {each}:', 
          train_df[each].apply(lambda x: len(x)).median())
for each in ['premise', 'hypothesis']:
    print(f'Mean number of words in {each}:', 
          train_df[each].apply(lambda x: len(x.split(' '))).mean())
    print(f'Maximum number of words in {each}:', 
          train_df[each].apply(lambda x: len(x.split(' '))).max())
    print(f'Minimum number of words in {each}:', 
          train_df[each].apply(lambda x: len(x.split(' '))).min())
    print(f'Median number of words in {each}:', 
          train_df[each].apply(lambda x: len(x.split(' '))).median())
train_df['premise_len'] = train_df['premise'].apply(lambda x: len(x.split(' ')))
train_df['hypothesis_len'] = train_df['hypothesis'].apply(lambda x: len(x.split(' ')))
fig, ax = plt.subplots(1, 3)
train_df[train_df.label==0].premise_len.hist(ax=ax[0], color='gray', label='entailment', bins=10)
ax[0].legend();
train_df[train_df.label==1].premise_len.hist(ax=ax[1], color='gold', label='neutral', bins=10)
ax[1].legend();
train_df[train_df.label==2].premise_len.hist(ax=ax[2], color='olive', label='contradiction', bins=10)
ax[2].legend();
fig, ax = plt.subplots(1, 3)
train_df[train_df.label==0].hypothesis_len.hist(ax=ax[0], color='gray', label='entailment', bins=10)
ax[0].legend();
train_df[train_df.label==1].hypothesis_len.hist(ax=ax[1], color='gold', label='neutral', bins=10)
ax[1].legend();
train_df[train_df.label==2].hypothesis_len.hist(ax=ax[2], color='olive', label='contradiction', bins=10)
ax[2].legend();
lang_en = train_df[train_df.language=='English']
lang_en.describe()
from nltk.probability import FreqDist
from nltk.corpus import stopwords
sw = stopwords.words('english')

lang_en.loc[:, 'premise'] = lang_en['premise'].apply(lambda x: x.lower())
lang_en.loc[:, 'hypothesis'] = lang_en['hypothesis'].apply(lambda x: x.lower())

p = ' '.join(lang_en['premise'].tolist()).split(' ')
h = ' '.join(lang_en['hypothesis'].tolist()).split(' ')
f_dist_p = FreqDist([x for x in p if x.replace('.', '') not in sw and len(x)>1])
f_dist_h = FreqDist([x for x in h if x.replace('.', '') not in sw and len(x)>1])
p_common = f_dist_p.most_common(20)
plt.bar([x[0] for x in p_common], [x[1] for x in p_common], 
        color='purple', label='most common in premise');
plt.legend();
p_common = f_dist_h.most_common(20)
plt.bar([x[0] for x in p_common], [x[1] for x in p_common], 
        color='purple', label='most common in hypothesis');
plt.legend();
import spacy
nlp = spacy.load('en')
doc = nlp(lang_en.loc[17, 'premise'])
spacy.displacy.render(doc, style='dep', options={'distance':80})
doc = nlp(lang_en.loc[17, 'hypothesis'])
spacy.displacy.render(doc, style='dep', options={'distance':80})
lang_en.loc[17, 'label']
doc = nlp(lang_en.loc[321, 'premise'])
spacy.displacy.render(doc, style='dep', options={'distance':60})
doc = nlp(lang_en.loc[321, 'hypothesis'])
spacy.displacy.render(doc, style='dep', options={'distance':60})
lang_en.loc[321, 'label']
def leave_diff(string1, string2):
    string1 = string1.lower().replace('.', '')
    string2 = string2.lower().replace('.', '')
    string1 = string1.replace(',', '')
    string2 = string2.replace(',', '')
    tokens1 = string1.split(' ')
    tokens2 = string2.split(' ')
    diff = set(tokens1).difference(tokens2)
    return ' '.join(list(diff))
for i in lang_en.index:
    lang_en.loc[i, 'diff'] = leave_diff(
        lang_en.loc[i, 'premise'], lang_en.loc[i, 'hypothesis'])
lang_en.head()
lang_en.loc[7, ['premise', 'hypothesis', 'diff']]
lang_en[lang_en['diff']=='']['label'].value_counts()