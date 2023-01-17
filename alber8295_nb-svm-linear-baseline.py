import pandas as pd, numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

subm = pd.read_csv('../input/sample_submission.csv')



# List all identities

identity_columns = [

    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',

    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']



len(train),len(test)
train.head(3)
print('Toxicity = 0.0 (No toxic) : {}\n'.format(train['comment_text'][0]))

print('0.0 < Toxicity < 0.4 (Hard to say) : {}\n'.format(train['comment_text'][11]))

print('0.5 < Toxicity < 0.9 (Toxic) : {}\n'.format(train['comment_text'][13]))

print('Toxicity > 0.9 (Very Toxic) : {}\n'.format(train['comment_text'][31]))
lens = train.comment_text.str.len()

lens.min(), lens.mean(), lens.std(), lens.max()
lens.hist();
# Make sure all comment_text values are strings

train['comment_text'] = train['comment_text'].astype(str)



# And no missing values

COMMENT = 'comment_text'

train[COMMENT].fillna("unknown", inplace=True)

test[COMMENT].fillna("unknown", inplace=True)



# Convert taget and identity columns to booleans

def convert_to_bool(df, col_name):

    df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    

def convert_dataframe_to_bool(df):

    bool_df = df.copy()

    for col in ['target'] + identity_columns:

        convert_to_bool(bool_df, col)

    return bool_df



train = convert_dataframe_to_bool(train)
plt.bar(x=train['target'].value_counts().keys(), height=train['target'].value_counts().values, tick_label=train['target'].value_counts().keys());
import re, string

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return re_tok.sub(r' \1 ', s).split()
n = train.shape[0]

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )

trn_term_doc = vec.fit_transform(train[COMMENT])

test_term_doc = vec.transform(test[COMMENT])
trn_term_doc, test_term_doc
def pr(y_i, y):

    p = x[y==y_i].sum(0)

    return (p+1) / ((y==y_i).sum()+1)
x = trn_term_doc

test_x = test_term_doc
def get_mdl(y):

    y = y.values

    r = np.log(pr(1,y) / pr(0,y))

    m = LogisticRegression(C=4, dual=True)

    x_nb = x.multiply(r)

    return m.fit(x_nb, y), r
label_cols = ['target']
preds = np.zeros((len(test), len(label_cols)))



for i, j in enumerate(label_cols):

    print('fit', j)

    m,r = get_mdl(train[j])

    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
submid = pd.DataFrame({'id': subm["id"]})

submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)

submission.columns =  ['id','prediction']

submission.to_csv('submission.csv', index=False)
submission.head()