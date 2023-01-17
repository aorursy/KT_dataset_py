import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option('max_colwidth', None)
path = '/kaggle/input/bcg-manually-reviewed-cleaned'
file = f'{path}/manually_reviewed_cleaned.csv'
df = pd.read_csv(file, encoding = "ISO-8859-1")
question_names = ['first_year','last_year','is_mandatory','timing','strain','has_revaccinations','revaccination_timing','location', 'manufacturer', 'company', 'groups']

df.columns = ['alpha_2_code', 'country', 'url', 'filename', 'is_pdf','Comments',
              'Snippet'] + question_names + ['snippet_len', 'text_len']
df.info()
df_fy = df[df['first_year'].notna()][['alpha_2_code','country','url', 'filename','first_year']].reset_index()
f"Working with {df_fy.shape[0]} positive examples."
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
                 if len(sent.text.strip().split()) > 5 and any([token.pos_ == 'VERB' for token in sent])]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

negative_examples = []

for _, row in tqdm.tqdm(df_fy.iterrows()):
    text = read_text(row)
    snippets = get_snippets(text)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(snippets)

    sim = cosine_similarity(tfidf_vectorizer.transform([row['first_year']]),tfidf_matrix)
    res = pd.DataFrame()
    res['snippet'] = snippets
    res['sim'] = sim[0]
    low_sim = res[res['sim']<0.1]['snippet'].values
    negative_examples.extend(low_sim)
df_data = pd.DataFrame({'snippet': df_fy['first_year'], 'label': 1})
df_data = df_data.append(pd.DataFrame({'snippet': negative_examples, 'label': 0}), ignore_index=True)
f"Working with {df_data.shape[0]} examples in total."
df_data['snippet_len'] = df_data['snippet'].apply(len)
df_data.drop(df_data[df_data['snippet_len'] > 350].index, inplace=True)
f"Working with {df_data.shape[0]} examples in total."
df_data['label'].value_counts(normalize=True)
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(df_data['snippet'], df_data['label'], test_size=0.33, random_state=42, stratify=df_data['label'])
C = 1
clfs = [('Dummy', DummyClassifier(strategy='prior')),
        ('RF', RandomForestClassifier(n_estimators=100, max_depth=2, class_weight='balanced')),
       ('LogReg', LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced')),
       ('SVM SGD', SGDClassifier(max_iter=1000, tol=1e-3, class_weight='balanced')),
       ('SVM linear', SVC(kernel='linear', C=C, class_weight='balanced')),
       ('SVM RBF', SVC(kernel='rbf', C=C, class_weight='balanced')),
       ('SVM Poly', SVC(kernel='poly', C=C, class_weight='balanced'))
       ]


for name, clf in clfs:
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', clf),
    ])
    
    scores = cross_validate(pipeline, X_train, y_train, cv=5, scoring=('accuracy', 'f1', 'roc_auc'), return_train_score=True)

    print("{:10s} {:5s} | Train: {:.3f}, Test: {:.3f}".format(name, 'ACC', np.mean(scores['train_accuracy']), np.mean(scores['test_accuracy'])))
    
    print("{:10s} {:5s} | Train: {:.3f}, Test: {:.3f}".format(name, 'F1', np.mean(scores['train_f1']), np.mean(scores['test_f1'])))
    
    print("{:10s} {:5s} | Train: {:.3f}, Test: {:.3f}".format(name, 'AUC', np.mean(scores['train_roc_auc']), np.mean(scores['test_roc_auc'])))
    
    print()
from sklearn.metrics import accuracy_score, f1_score, classification_report

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(random_state=0, solver='lbfgs', class_weight='balanced')),
])
    
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
pred_proba = pipeline.predict_proba(X_test)[:,1]
result = pd.DataFrame({'proba': pred_proba, 'label': y_test, 'text': X_test})
result.sort_values('proba', ascending=False).head(10)
accuracy_score(y_test, pred)
f1_score(y_test, pred)
print(classification_report(y_test, pred))
