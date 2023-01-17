import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
biorxiv_dir = '/kaggle/input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json'
filenames = os.listdir(biorxiv_dir)
print("Number of articles retrieved from biorxiv:", len(filenames))
def load_files(dirname):
    filenames = os.listdir(dirname)
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    
    return raw_files

def generate_clean_df(all_files):
    cleaned_files = []
    
    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'], 
                           with_affiliation=True),
            format_body(file['abstract']),
            format_body(file['body_text']),
            format_bib(file['bib_entries']),
            file['metadata']['authors'],
            file['bib_entries']
        ]

        cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text', 
                 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df.head()
    
    return clean_df
def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

def format_bib(bibs):
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []
    
    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'], 
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)
all_files = []

for filename in filenames:
    filename = biorxiv_dir +"/"+ filename
    file =json.load(open(filename, 'rb'))
    all_files.append(file)

    


def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))
    
    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    name_ls = []
    
    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)
    
    return ", ".join(name_ls)

def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    
    return body

from tqdm.notebook import tqdm

cleaned_files = []
    
for file in tqdm(all_files):
    features = [
        file['paper_id'],
        file['metadata']['title'],
        format_authors(file['metadata']['authors']),
        format_authors(file['metadata']['authors'], 
                       with_affiliation=True),
        format_body(file['abstract']),
        format_body(file['body_text']),

    ]

    cleaned_files.append(features)

col_names = ['paper_id', 'title', 'authors',
             'affiliations', 'abstract', 'text',]

df = pd.DataFrame(cleaned_files, columns=col_names)
df.head()
df['abstract_word_count'] = df['abstract'].apply(lambda x: len(x.strip().split()))
df['body_word_count'] = df['text'].apply(lambda x: len(x.strip().split()))
df.head()
df.drop_duplicates(['abstract', 'text'], inplace=True)
df.head(5)
df.drop_duplicates(['title','abstract'], inplace=True)
df.shape
#df = df.head(10000)
import re

df['text'] = df['text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
df['abstract'] = df['abstract'].apply(lambda x: re.sub('[^a-zA-z0-9\s]','',x))
df['text'] = df['text'].apply(lambda x: re.sub('\n\n',' ',x))
df['abstract'] = df['abstract'].apply(lambda x: re.sub('\n\n',' ',x))
df['text'] = df['text'].apply(lambda x: re.sub('\d+', '',x))
df['abstract'] = df['abstract'].apply(lambda x: re.sub('\d+', '',x))


df['text'] = df['text'].apply(lambda x: x.lower())
df['abstract'] = df['abstract'].apply(lambda x: x.lower())
df.head(5)
text = df.drop(["paper_id", "abstract", "abstract_word_count", "body_word_count", "authors", "title", "affiliations"], axis=1)
text.head(5)
docs = []
for x in range(0,len(text)):
    docs.append(str(text.iloc[x]['text']))
print(docs[5])
from sklearn.feature_extraction.text import TfidfVectorizer


with open('../input/stopwords/englishStopwords.txt', 'r') as f:
    myLists = [line.strip() for line in f]

               
vectorizer = TfidfVectorizer(stop_words=myLists)
X = vectorizer.fit_transform(docs)
from sklearn.model_selection import train_test_split

# test set size of 20% of the data and the random seed 42 <3
X_train, X_test = train_test_split(X.toarray(), test_size=0.2, random_state=42)

print("X_train size:", len(X_train))
print("X_test size:", len(X_test), "\n")
from sklearn.cluster import KMeans

k = 10
kmeans = KMeans(n_clusters=k, n_jobs=4, verbose=10)
y_pred = kmeans.fit_predict(X_train)
y_train = y_pred
y_test = kmeans.predict(X_test)
outerlist = []
while len(outerlist) < k:
    outerlist.append([])
for x in docs:
    Y = vectorizer.transform([x])
    prediction = kmeans.predict(Y)
    outerlist[int(prediction)].append(x)

order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()


index = 0


for iter in outerlist:
    print("DOCUMENTS GROUP %d" % index)
    print(iter[:1], sep=', ')
    print(" ")
    
    print("-----------------")


    print("GROUP DESCRIPTIVE KEYWORDS" )
    for ind in order_centroids[index, :10]:
        print(' %s' % terms[ind]),
    index = index + 1
    print("-----------------")


from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
    
print(model_name, ":\n")
print("Accuracy Score: ", '{:,.3f}'.format(float(accuracy_score(y_test, y_pred)) * 100), "%")
print("F1 score: ", '{:,.3f}'.format(float(f1_score(test, pred, average='micro')) * 100), "%")