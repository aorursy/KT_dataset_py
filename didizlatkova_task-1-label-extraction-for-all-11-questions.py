import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
pd.set_option('max_colwidth', None)
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
df_ncbi = df[df['url_domain']=='www.ncbi.nlm.nih.gov']
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
    except urllib.error.HTTPError as e:
        print(e)
        print(url)
        return ""
df_ncbi['url_title'] = df_ncbi['url'].parallel_apply(get_url_title)
pd.options.mode.chained_assignment = None
df_ncbi['title_has_country'] = df_ncbi.apply(lambda row: row['country'] in row['url_title'], axis=1)
df_ncbi.drop(df_ncbi[df_ncbi['title_has_country'] == False].index, inplace=True)
df_ncbi.drop_duplicates('url_title', inplace=True)
f"Working with {df_ncbi.shape[0]} sources"
import spacy
nlp = spacy.load('en_core_web_sm')

def get_snippets(text):
    '''
        Returns sentences in the text which contain more than 5 tokens and at least one verb.
    '''
    return [sent.text.strip() for sent in nlp(text).sents 
                 if len(sent.text.strip()) < 350 and len(sent.text.strip().split()) > 5 and any([token.pos_ == 'VERB' for token in sent])]
keywords = {
    'first_year': ["first", "bcg", "policy", "initiated", "bacille", "calmette", "guerin", 
                   "introduced", "distributed", "vaccination", "included", "immunisation", 
                   "started", "since", "schedule", "recommended", "vaccine", "programme", "program",
                  "start", "universal"],
    'last_year': ["abolished", "suspended", "until", "continued", "withdrawn"],
    'is_mandatory': ["voluntary", "mandatory", "compulsory", "few"],
    'timing': ["birth", "aged", "children", "age", "child", "given", "target", "weeks", "obstetric", "newborns"],
    'strain': ["strain", "substrain", "strains", "mycobacterium", "pasteur", 
               "1173p2", "ssi", "danish", "1331"],
    'has_revaccinations': ["revaccination", "revaccinations"],
    'revaccination_timing': ["revaccination", "revaccinations"],
    'location': ["intradermal", "injection", "arm", "left", "right", "deltoid", "muscle"],
    'manufacturer': ["producer", "manufacturer", "sanofi", "pasteur", "ltd", 
                    'unk', 'serum', 'mikrogen', 'rivm', 'merieux', 'institute', 'statens', 'laboratory', 'biomed', 'ncipd',
       'ssi',  'biofarma', 'bulbio', 'intervax', 'intervac', 'pfizer', 'aventis'],
    'supplier': ['ltd', 'serum', 'institute','mikrogen',
       'rivm', 'merieux', 'sanofi', 'pasteur', 'statens','unk', 'laboratory', 'paho',
       'medoka', 'biomed', 'wholesaler', 'unicef', 'bulbio', 'intervax', 'aventis', 'ncipd'],
    'groups': ['high-risk', 'risk', 'travel', 'high incidence', 'high risk', 'TB incidence']
}
from sklearn.feature_extraction.text import TfidfVectorizer

def get_score(corpus, vocab):
    vec = TfidfVectorizer(vocabulary=vocab).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.mean(axis=1)
    return sum_words
def read_text(row):
    code = row['alpha_2_code']
    filename=row['filename'].replace('.txt', '')
    filename = f'/kaggle/input/hackathon/task_1-google_search_txt_files_v2/{code}/{filename}.txt'
    
    with open(filename, 'r') as file:
        data = file.read()
    return data
import tqdm

dfs = []
for _, row in tqdm.tqdm(df_ncbi.iterrows()):
    data = read_text(row)
    
    snippets = get_snippets(data)
    
    result = pd.DataFrame()
    result['sentence'] = snippets
    result['len'] = result['sentence'].apply(len)
    result['country'] = row['country']
    result['url'] = row['url']
    
    for question, words in keywords.items():
        result[question] = get_score(snippets, words)
    
    result = result.replace(0, np.nan)
    result = result.dropna(how='all', axis=0, subset=keywords.keys())
    
    dfs.append(result)
result = pd.concat(dfs, ignore_index=True)
result.drop_duplicates('sentence', inplace=True)
f"Working with {result.shape[0]} snippets"
for k in keywords.keys():
    print(k)
    display(result.sort_values(k, ascending=False)[[k, 'sentence','country','len']].head(3))
path = '/kaggle/input/bcg-manually-reviewed-cleaned'
file = f'{path}/manually_reviewed_cleaned.csv'
df_man = pd.read_csv(file, encoding = "ISO-8859-1")
question_names = ['first_year','last_year','is_mandatory','timing','strain','has_revaccinations','revaccination_timing','location','manufacturer', 'supplier', 'groups']
df_man.columns = ['alpha_2_code', 'country', 'url', 'filename', 'is_pdf','Comments',
              'Snippet'] + question_names + ['snippet_len', 'text_len']
for k in keywords.keys():
    print(k)
    df_ncbi_ans = result.sort_values(k, ascending=False)[[k, 'sentence','country', 'url']].head(30)
    df_man_ans = df_man[df_man[k].notna()][[k, 'country', 'url']]
    df_man_ans['sentence'] = df_man_ans[k]
    df_man_ans[k] = 1
    
    df_ans = pd.concat([df_man_ans, df_ncbi_ans], ignore_index=True)
    df_ans['label'] = 1
    
    df_ans.to_csv(f'/kaggle/working/{k}_labeled.csv', index=False)
