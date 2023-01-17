import numpy as np
import pandas as pd
import os, re

# The end at the beginnging!  Here are # of misspellings per application.
result = pd.read_csv('../input/corpus-misspellings/corpus_misspellings_feature.csv')
result.tail(10)

text_cols = ['project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'project_resource_summary']
id_col = 'id'
target_col = 'project_is_approved'
spell_col = 'misspellings'

input_folder = '../input/donorschoose-application-screening'
train = pd.read_csv(os.path.join(input_folder, 'train.csv'))[[id_col] + text_cols + [target_col]]
test = pd.read_csv(os.path.join(input_folder, 'test.csv'))[[id_col] + text_cols]

df = pd.concat([train, test], axis=0, ignore_index=True)

# piece together 'project_essay'
df.loc[df['project_essay_3'].notnull(), 'project_essay_1'] = df['project_essay_1'] + ' ' + df['project_essay_2']
df.loc[df['project_essay_4'].notnull(), 'project_essay_2'] = df['project_essay_3'] + ' ' + df['project_essay_4']
df['project_essay'] = df['project_essay_1'] + ' ' + df['project_essay_2']

def clean_text(phrase):
  # specific
  q = "[\'\’\´\ʻ]"
  
  phrase = re.sub(re.compile("won%st" % q), "will not", phrase)
  phrase = re.sub(re.compile("can%st" % q), "can not", phrase)
  
  # general
  phrase = re.sub(re.compile("n%st" % q), " not", phrase)
  phrase = re.sub(re.compile("%sre" % q), " are", phrase)
  phrase = re.sub(re.compile("%ss" % q), " is", phrase)
  phrase = re.sub(re.compile("%sd" % q), " would", phrase)
  phrase = re.sub(re.compile("%sll" % q), " will", phrase)
  phrase = re.sub(re.compile("%st" % q), " not", phrase)
  phrase = re.sub(re.compile("%sve" % q), " have", phrase)
  phrase = re.sub(re.compile("%sm" % q), " am", phrase)
  
  phrase = re.sub(r"\\r|\\n", " ", phrase)
  phrase = re.sub(r"\.\.+", ". ", phrase) # ellipsis ... or .. to .
  
  # all chars except ;.?!
  phrase = re.sub(re.compile(q + "+"), "", phrase)   
  phrase = re.sub(r"[\'\"\#\$\%\&\(\)\*\+\,\-\/\:\<\=\>\@\[\\\]\^\_\`\{\|\}\~\“\”\″\ʺ\¨\‘\…\—\―\–\•\®]+", " ", phrase)   
  
  # add space after EOS if missing
  phrase = re.sub(r"([\;\.\?\!])([^\s])", "\\1 \\2", phrase)
  # squeeze space before EOS
  phrase = re.sub(r"\s+([\;\.\?\!])", "\\1", phrase)
  
  # space squeezer: \u200b is UNICODE space
  phrase = re.sub(r"['\u200b\s]+", " ", phrase).strip()
  
  return phrase  

text_cols = ['project_title', 'project_essay', 'project_resource_summary']
for col in text_cols:
  df[col] = df[col].apply(clean_text)
  df[col] = df[col].apply(lambda x: x if x[-1] in ';.?!' else x + '.') # EOS marker

df['project_corpus'] = df['project_title'] + ' ' + df['project_essay'] + ' ' + df['project_resource_summary']
df = df[[id_col] + ['project_corpus'] + [target_col]]

df.head()
import hunspell
hunspell_folder = '../input/hunspellenuslarge'

# https://aur.archlinux.org/packages/hunspell-en-us-large/
hobj = hunspell.HunSpell(os.path.join(hunspell_folder, 'en_US-large.dic'), 
                         os.path.join(hunspell_folder, 'en_US-large.aff'))

# give it a try
hobj.spell('donors')

isok = pd.read_csv('../input/corpus-misspellings/corpus_misspellings.csv', encoding='utf-8')

# pass through these words as OK even though hunspell flags them as not
passthru = set(isok[isok.ok == 1].word)
df[spell_col] = 0
misspell = {}

isok.head()
pd.options.mode.chained_assignment = None
for ci, s in enumerate(df.project_corpus):
  sw = s.split(' ') # split into words
  
  w_ok = set()
  w_misspell = set() # per word misspelled
  for i, w in enumerate(sw):
    w = re.sub(r'\W+', '', w) # clean the word
    
    if len(w) <= 3: # ignore short words
      continue
    
    # mark any word that contains chars outside a-z as OK:
    # if a product name like "Brite" is capitalized once, this means other occurrences in the
    # application like "brite" will be considered OK
    if bool(re.search(r'[^a-z]', w)):
      w_ok.add(w.upper())
      continue
    
    w = w.upper() # hobj.spell() works better with CAPS
    if hobj.spell(w.upper()) or \
       (w.endswith('S') and (hobj.spell(w[:-1].upper())) or (w[:-1] in passthru)) or \
       (w in passthru) or \
       ((i > 0) and (sw[i-1] in ['MR', 'MRS'])):
      continue
    
    w_misspell.add(w) # word is misspelled!
    
  lw_misspell = list(w_misspell - w_ok)
  for w in lw_misspell:
    # print(ci, w)
    misspell[w] = misspell.get(w, 0) + 1
    df[spell_col][ci] += 1
    
# same as we saw in "result"
df[[id_col, spell_col]].tail(10)

from collections import Counter
Counter(df[spell_col])
list(df[df[spell_col] == 13].project_corpus)
# clipped at 8
print('% approval given # of misspellings:')
for n in range(0,9):
  print('%d: %.4f' % (n, np.mean(df[df[spell_col].clip(0,8) == n][target_col])))
# combine plural misspelling with singular
todelete = []
for w, n in misspell.items():
  if not w.endswith('S') or not w[:-1] in misspell:
    continue
  
  misspell[w[:-1]] += n
  todelete.append(w)

# delete plurals
for w in todelete:
  misspell.pop(w)  
  
misspell = pd.DataFrame.from_dict(misspell, orient='index')
misspell.reset_index(level=0, inplace=True)
misspell.columns = ['word', 'n']
misspell['ok'] = 0 # not OK, unless manually set to 1
misspell = misspell.sort_values(by='n', ascending=False)  

# misspell.to_csv('../output/corpus_misspellings.csv', index=False)

misspell.head()