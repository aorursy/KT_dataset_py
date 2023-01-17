import numpy as np

import pandas as pd

import seaborn as sns

pd.set_option('max_colwidth', None)
question_names = ['first_year','last_year','is_mandatory','timing','strain','has_revaccinations','revaccination_timing','location', 'manufacturer', 'supplier', 'groups']
def read_text(row):

    code = row['alpha_2_code']

    filename=row['filename'].replace('.txt', '')

    filename = f'/kaggle/input/hackathon/task_1-google_search_txt_files_v2/{code}/{filename}.txt'

    

    with open(filename, 'r') as file:

        data = file.read()#.replace('\n', ' ')

    return data



import spacy

nlp = spacy.load('en_core_web_sm')



def get_snippets(text):

    '''

        Returns sentences in the text which contain more than 5 tokens and at least one verb.

    '''

    return [sent.text.strip() for sent in nlp(text).sents 

                 if len(sent.text.strip()) < 350 and len(sent.text.strip().split()) > 5 and any([token.pos_ == 'VERB' for token in sent])]
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import tqdm



def get_negative_examples(df, q):

    df = df[df[q].notna()][['alpha_2_code','country','url', 'filename',q]]

    negative_examples = []



    for _, row in df.iterrows():

        text = read_text(row)

        snippets = get_snippets(text)



        tfidf_vectorizer = TfidfVectorizer()

        tfidf_matrix = tfidf_vectorizer.fit_transform(snippets)



        sim = cosine_similarity(tfidf_vectorizer.transform([row[q]]),tfidf_matrix)

        res = pd.DataFrame()

        res['snippet'] = snippets

        res['sim'] = sim[0]

        low_sim = res[res['sim']<0.1]['snippet'].values

        negative_examples.extend(low_sim)

    return negative_examples
path = '/kaggle/input/bcg-manually-reviewed-cleaned'

file = f'{path}/manually_reviewed_cleaned.csv'

df_man = pd.read_csv(file, encoding = "ISO-8859-1")



df_man.columns = ['alpha_2_code', 'country', 'url', 'filename', 'is_pdf','Comments',

              'Snippet'] + question_names + ['snippet_len', 'text_len']
datasets = []

for q in question_names:

    print(q)

    file = f'{path}/{q}_labeled.csv'

    df_labeled = pd.read_csv(file, encoding = "ISO-8859-1")

    neg = get_negative_examples(df_man, q)

    

    df_data = pd.DataFrame({'snippet': df_labeled['sentence'], 'label': df_labeled['label']})

    df_data = df_data.append(pd.DataFrame({'snippet': neg, 'label': 0}), ignore_index=True)

    

    print(df_data.shape)

    display(df_data['label'].value_counts(normalize=True))

    print()

    

    datasets.append(df_data)
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.model_selection import cross_val_score



from sklearn.pipeline import Pipeline



from sklearn.dummy import DummyClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn.svm import SVC





def eval_model(X, y):

    clfs = [('Dummy', DummyClassifier(strategy='prior')),

            ('LogReg', LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced')),]

    

    for name, clf in clfs:

        pipeline = Pipeline([

        ('tfidf', TfidfVectorizer()),

        ('clf', clf),

        ])

    

        scores = cross_validate(pipeline, X, y, cv=4, scoring=('accuracy', 'f1', 'roc_auc'), return_train_score=True)



        print("{:10s} {:5s} | Train: {:.3f}, Test: {:.3f}".format(name, 'ACC', np.mean(scores['train_accuracy']), np.mean(scores['test_accuracy'])))



        print("{:10s} {:5s} | Train: {:.3f}, Test: {:.3f}".format(name, 'F1', np.mean(scores['train_f1']), np.mean(scores['test_f1'])))



        print("{:10s} {:5s} | Train: {:.3f}, Test: {:.3f}".format(name, 'AUC', np.mean(scores['train_roc_auc']), np.mean(scores['test_roc_auc'])))

    

    

def get_model(X, y):

    pipeline = Pipeline([

    ('tfidf', TfidfVectorizer()),

    ('clf', LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced')),

    ])

    

    return pipeline.fit(X, y)
for q, dataset in zip(question_names, datasets):

    print(q)

    eval_model(dataset['snippet'], dataset['label'])
models = [get_model(d['snippet'], d['label']) for d in datasets]
path = '/kaggle/input/hackathon'

files = [f'{path}/task_1-google_search_english_original_metadata.csv',

         f'{path}/task_1-google_search_translated_to_english_metadata.csv']

dfs = []

for file in files:

    df = pd.read_csv(file, encoding = "ISO-8859-1")

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df.drop(['Is Processed', 'Comments', 'language', 'query'], axis=1, inplace=True)

df.drop(df[df['is_downloaded']==False].index, inplace=True)

df['char_number'] = pd.to_numeric(df['char_number'], errors='coerce')

df.drop(df[df['char_number']==0].index, inplace=True)

df.drop_duplicates('url', keep=False, inplace=True)

df.drop(df[df['url'].str.contains('researchgate.net')].index, inplace=True)

df.drop(df[df['is_pdf']].index, inplace=True)

assert all(df[df['alpha_2_code'].isna()]['country']=='Namibia')

df['alpha_2_code'].fillna('NA', inplace=True)
from urllib.parse import urlparse

df['url_domain'] = df['url'].apply(lambda x: urlparse(x).netloc)
df_filtered = df[(df['url'].str.contains('vaccin')) |

                (df['url'].str.contains('bcg')) |

                 (df['url_domain']=='www.sciencedirect.com') |

                 (df['url_domain']=='www.ncbi.nlm.nih.gov')]
f"Working with {df_filtered.shape[0]} sources"
! pip install pandarallel
import bs4 as bs

import urllib.request

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)



def get_url_title(url):

    try:

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

        source = urllib.request.urlopen(req).read()

        soup = bs.BeautifulSoup(source,'lxml')

        if not soup.title:

            print('No title')

            print(url)

            return ""

        return soup.title.text

    except Exception as e:

        print(e)

        print(url)

        return ""
df_filtered['url_title'] = df_filtered['url'].parallel_apply(get_url_title)
pd.options.mode.chained_assignment = None
df_filtered['title_has_country'] = df_filtered.apply(lambda row: row['country'] in row['url_title'], axis=1)

df_filtered.drop(df_filtered[df_filtered['title_has_country'] == False].index, inplace=True)
df_filtered.drop_duplicates('url_title', inplace=True)
f"Working with {df_filtered.shape[0]} sources"
import tqdm



dfs = []

for _, row in tqdm.tqdm(df_filtered.iterrows()):

    data = read_text(row)

    

    snippets = get_snippets(data)

    

    result = pd.DataFrame()

    result['sentence'] = snippets

    result['len'] = result['sentence'].apply(len)

    result['country'] = row['country']

    result['url'] = row['url']

    result['filename'] = row['filename']

    result['alpha_2_code'] = row['alpha_2_code']

    

    for q, model in zip(question_names, models):

        result[q] = model.predict_proba(snippets)[:,1]

    

    result = result.replace(0, np.nan)

    result = result.dropna(how='all', axis=0, subset=question_names)

    

    dfs.append(result)
result = pd.concat(dfs, ignore_index=True)
f"Considering {result.shape[0]} snippets"
result.groupby('country')['url'].unique().apply(len).value_counts()
def get_source_score(row, min_score=0.65):

    res = {'score': sum([row[q].max() for q in question_names if row[q].max() > min_score])}

    for q in question_names:

        if row[q].max() > min_score:

            res[q] = row.loc[row[q].idxmax()]['sentence']

            res[f'{q}_score'] = row[q].max()

        else:

            res[q] = np.nan

            res[f'{q}_score'] = np.nan

    return res
final_result = result.groupby(['country', 'alpha_2_code', 'url', 'filename']).apply(lambda x: pd.Series(get_source_score(x))).sort_values('score', ascending=False).groupby(['country']).head(2)
final_result = final_result.replace(0, np.nan)

final_result.dropna(how='all', inplace=True)

final_result.reset_index(inplace=True)
f"Final result has {final_result.shape[0]} different sources"
final_result.head()
final_result['country'].value_counts()
total_n = 0

for q in question_names:

    n = final_result[final_result[q].notna()].shape[0]

    total_n += n

    print(f'{q}: {n}')
f"Total number of extracted answers: {total_n}"
path = '/kaggle/input/hackathon'

file = f'{path}/BCG_world_atlas_data-2020.csv'

df_atlas = pd.read_csv(file)
final_result.insert(3, 'atlas', 'no')

final_result['comments'] = ''
cols = [c for c in final_result.columns if 'score' not in c]

assert len(cols) == len(df_atlas.columns)

res_to_append = final_result[cols]

res_to_append.columns = df_atlas.columns
df_atlas_ext = df_atlas.append(res_to_append.fillna(''), ignore_index=True)
df_atlas_ext.sort_values('Contry Name (Mandatory field)', inplace=True)
f"Original dataset has {df_atlas.shape[0]} entries"
f"Extended dataset has {df_atlas_ext.shape[0]} entries"
df_atlas_ext.to_csv(f'/kaggle/working/df_atlas_extended.csv', index=False)