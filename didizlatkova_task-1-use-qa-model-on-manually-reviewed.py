import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option('max_colwidth', None)
path = '/kaggle/input/bcg-manually-reviewed-cleaned'
file = f'{path}/manually_reviewed_cleaned.csv'
df = pd.read_csv(file, encoding = "ISO-8859-1")
df.head(1)
f"Using {df.shape[0]} entries"
import spacy
nlp = spacy.load('en_core_web_sm')

def get_snippets(text):
    '''
        Returns sentences in the text which contain more than 5 tokens and at least one verb.
    '''
    return [sent.text.strip() for sent in nlp(text).sents 
                 if len(sent.text.strip().split()) > 5 and any([token.pos_ == 'VERB' for token in sent])]
!pip3 install tensorflow_text>=2.0.0rc0
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
module = hub.load('https://tfhub.dev/google/universal-sentence-encoder-qa/3')
questions = ["Which is the First Year of the BCG Policy?",
             "Which is the last year of the BCG Policy?",
             "Is BCG vaccination mandatory for all children?",
             "What is the timing for the BCG vaccination (age)?",
             "Which BCG Strain has been used?",
             "Are revaccinations (boosters) recommended for BCG?",
             "What is the timing of BCG revaccination?",
             "Which in the body (arm) is the BCG Vaccine administered?"]

question_embeddings = module.signatures['question_encoder'](
            tf.constant(questions))
def read_text(row):
    code = row['alpha_2_code']
    filename=row['filename'].replace('.txt', '')
    filename = f'/kaggle/input/hackathon/task_1-google_search_txt_files_v2/{code}/{filename}.txt'
    
    with open(filename, 'r') as file:
        data = file.read().replace('\n', ' ')
    return data

question_names = ['first_year','last_year','is_mandatory','timing','strain','has_revaccinations','revaccination_timing','location']

def apply_USE_model(row):
    data = read_text(row)
    
    snippets = get_snippets(data)
    
    response_embeddings = module.signatures['response_encoder'](
        input=tf.constant(snippets),
        context=tf.constant(snippets))
    scores = np.inner(question_embeddings['outputs'], response_embeddings['outputs'])

    result = pd.DataFrame(scores.T, columns=question_names)
    result['sentence'] = snippets
    result['len'] = result['sentence'].apply(len)
    result['country'] = row['country']
    
    return result
dfs = []
import tqdm
for _, row in tqdm.tqdm(df.iterrows()):
    result = apply_USE_model(row)
    dfs.append(result)
final_eval = pd.concat(dfs, ignore_index=True)
f"The evaluation is performed on {final_eval.shape[0]} snippets"
final_eval.iloc[final_eval[question_names].idxmax()]
for k in question_names:
    print(k)
    display(final_eval.sort_values(k, ascending=False)[[k, 'sentence','country','len']].head(3))
final_eval.drop('len', axis=1).plot.box(figsize=(15,5))
df.columns = ['alpha_2_code', 'country', 'url', 'filename', 'is_pdf','Comments',
              'Snippet'] + question_names + ['snippet_len', 'text_len']
for _, row in df.head(10).iterrows():
    print('----' * 10)
    print('ACTUAL:')
    print('\n'.join([f"<{i}>: {v}" for i, v in row[question_names].dropna().iteritems()]))
    
    cols = [i for i, v in row[question_names].dropna().iteritems()]
    
    result = apply_USE_model(row)
    print(f"Total snippets: {result.shape[0]}")
    
    for k in cols:
        display(result.sort_values(k, ascending=False)[[k, 'sentence','country','len']].head(3))
    
