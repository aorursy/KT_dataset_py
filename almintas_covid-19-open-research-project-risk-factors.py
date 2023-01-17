import os
import json; print('JSON version:', json.__version__)
import pandas as pd; print('Pandas version:', pd.__version__)
import numpy as np; print('Numpy version:', np.__version__)
from tqdm import tqdm; print('tqdm version:', pd.__version__)
import nltk; print('NLTK version:', nltk.__version__)
from nltk.corpus import stopwords
# nltk.download('stopwords') # may need to download depending on the local setup
from nltk.tokenize import word_tokenize, sent_tokenize
# nltk.download('punkt') # may need to download depending on the local setup
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet') # may need to download depending on the local setup
import string
import re; print('Regex version:', re.__version__)
import wordcloud; print('Wordcloud version:', wordcloud.__version__)
from wordcloud import WordCloud
import gensim; print('Gensim version:', gensim.__version__)
from gensim.models import Word2Vec
from collections import Counter
import matplotlib.pyplot as plt
%matplotlib inline

os.getcwd()
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
file_dir = '/kaggle/input/'
with open(file_dir + '/CORD-19-research-challenge/metadata.readme', 'r') as fm:
    data_meta = fm.read()
    print(data_meta)
meta = pd.read_csv(file_dir + "/CORD-19-research-challenge/metadata.csv", low_memory=False)
meta.head()
count = 0
file_extensions = []
for dirname, _, filenames in os.walk(file_dir + "/CORD-19-research-challenge"):
    for filename in filenames:
        count += 1
        file_extension = filename.split(".")[-1]
        file_extensions.append(file_extension)

file_extset = set(file_extensions)

print(f"Files: {count}")
print(f"Files extensions: {file_extset}\n\n-------------------------------\nFiles extension count:\n")
file_extlist = list(file_extset)
for fext in file_extlist:
    fext_count = file_extensions.count(fext)
    print(f"{fext}: {fext_count}")
counter = 0
file_list = []
for dirname, _, filenames in os.walk(file_dir + "/CORD-19-research-challenge"):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        if filename[-5:]==".json":
            file_list.append(os.path.join(dirname, filename))

file_list.sort()
total_files = len(file_list)
all_abstracts = []

for file in tqdm(file_list):
    j = json.load(open(file, "rb"))
       
    abstract = ""
    
    try:
        if j['abstract']:
                for entry in j['abstract']:
                    abstract += entry['text'] +'\n\n'
    except KeyError:
            pass 
            
    all_abstracts.append([abstract])
with open('/kaggle/working/all_abstracts.txt', 'w', encoding="utf-8") as f:
    for words in all_abstracts:
        f.write("%s\n" % words)
with open('/kaggle/working/all_abstracts.txt', encoding="utf-8") as f, open('/kaggle/working/all_abstracts_tokens.txt', 'w', encoding="utf-8") as out_f:
    text = f.read().lower() #read the file and convert text to lower-case
    short_words = re.compile(r'\W*\b\w{1,3}\b') #get rid of words less than 4 letters long
    text = short_words.sub('', text)
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = stopwords.words('english')
    new_stop_words = ['preprint', 'copyright', 'holder', 'peerreviewed', 'authorfunder', 'license', 'medrxiv', 'biorxiv',
                     'righta', 'reuse', 'reserved', 'also', 'used', 'found', 'using', 'however'] 
    stop_words.extend(new_stop_words) #add additional stop workds to NLTK's stop words vocabulary
    words = [w for w in words if not w in stop_words]
    new_text = ' '.join(words)
    plt.figure(figsize=(16, 7))
    fd = nltk.FreqDist(words)
    fd.plot(40,title = "40 Most Frequent Words in Abstracts", cumulative=False)
    out_f.write(new_text)
bigrams = nltk.bigrams(words)
freq_bigrams = nltk.FreqDist(bigrams)
figure = plt.figure(figsize=(14, 7))
freq_bigrams.plot(20)
with open('/kaggle/working/all_abstracts_tokens.txt', encoding = "utf-8") as f, open('/kaggle/working/all_abstracts_lemmas.txt', 'w', encoding = "utf-8") as out_f:
    text = f.read()
    tokens = word_tokenize(text)
    lemma = WordNetLemmatizer()
    lemmed = [lemma.lemmatize(word) for word in tokens]
    new_lem_text = ' '.join(lemmed)
    out_f.write(new_lem_text)
lemma_text = open('/kaggle/working//all_abstracts_lemmas.txt', 'rt', encoding="utf-8").read()

wc = WordCloud(max_font_size=200,
                      width=2500,
                      height=2000,
                      max_words=4000,
                      random_state=44,
                      collocations = False,
                     ).generate(lemma_text)

figure = plt.figure(figsize=(32, 14))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("CORD abstracts", fontsize= 20)
n = 4
word = r'\W*([\w]+)'
text_search = re.findall(r'{}\W*{}{}'.format(word*n,'(?:comorbid|comorbidity|comorbidities|comorbid |comorbidity |comorbidities | comorbid| comorbidity| comorbidities| comorbid | comorbidity | comorbidities )',word*n), lemma_text)
flatten_text_search = [element for sublist in text_search for element in sublist if len(element) >3] 
sorted_counts = pd.DataFrame(Counter(flatten_text_search).most_common())
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # show all sorted counts data
    print(sorted_counts)
plt.rcParams.update({'figure.figsize':(24, 10), 'figure.dpi':300})

counts = dict(Counter(flatten_text_search).most_common(150))
labels, values = zip(*counts.items())

# sort your values in descending order
indSort = np.argsort(values)[::-1]

# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]

indexes = np.arange(len(labels))

width = 0.35

plt.bar(indexes, values)

# add labels
plt.xticks(rotation=90)
plt.xticks(indexes + width * 0.05, labels)
all_docs = []

for file in tqdm(file_list):
    j = json.load(open(file, "rb"))
    paper_id = j['paper_id']
    title = j['metadata']['title']
       
    abstract = ""
    
    try:
        if j['abstract']:
                for entry in j['abstract']:
                    abstract += entry['text'] +'\n\n'
    except KeyError:
            pass 
            
    all_bodytext = ""
        
    for text in j['body_text']:
        all_bodytext += text['text'] +'\n\n'

    all_docs.append([paper_id, title, abstract, all_bodytext])
with open('/kaggle/working/all_docs.txt', 'w', encoding="utf-8") as f:
    for words in all_docs:
        f.write("%s\n" % words)
with open('/kaggle/working/all_docs.txt', encoding="utf-8") as f, open('/kaggle/working/all_docs_tokens.txt', 'w', encoding="utf-8") as out_f:
    text = f.read().lower()
    short_words = re.compile(r'\W*\b\w{1,3}\b')
    text = short_words.sub('', text)
    tokens = word_tokenize(text)
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = stopwords.words('english')
    new_stop_words = ['preprint', 'copyright', 'holder', 'peerreviewed', 'authorfunder', 'license', 'medrxiv', 'biorxiv',
                     'righta', 'reuse', 'reserved', 'also', 'used', 'found', 'using', 'however']
    stop_words.extend(new_stop_words)
    words = [w for w in words if not w in stop_words]
    new_text = ' '.join(words)
    figure = plt.figure(figsize=(16, 7))
    fd = nltk.FreqDist(words)
    fd.plot(40,title = "40 Most Frequent Words", cumulative=False)
    out_f.write(new_text)
bigrams = nltk.bigrams(words)
freq_bigrams = nltk.FreqDist(bigrams)

figure = plt.figure(figsize=(16, 7))
freq_bigrams.plot(40)
with open('/kaggle/working/all_docs_tokens.txt', encoding = "utf-8") as f, open('/kaggle/working/all_docs_lemmas.txt', 'w', encoding = "utf-8") as out_f:
    text = f.read()
    tokens = word_tokenize(text)
    lemma = WordNetLemmatizer()
    lemmed = [lemma.lemmatize(word) for word in tokens]
    new_lem_text = ' '.join(lemmed)
    out_f.write(new_lem_text)
lemma_text_all = open('/kaggle/working/all_docs_lemmas.txt', 'rt', encoding="utf-8").read()

wc = WordCloud(max_font_size=200,
                      width=2500,
                      height=2000,
                      max_words=4000,
                      random_state=44,
                      collocations = False,
                     ).generate(lemma_text)

figure = plt.figure(figsize=(32, 14))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("CORD documents", fontsize= 20)
n = 4
word = r'\W*([\w]+)'
text_search_all = re.findall(r'{}\W*{}{}'.format(word*n,'(?:comorbid|comorbidity|comorbidities|comorbid |comorbidity |comorbidities | comorbid| comorbidity| comorbidities| comorbid | comorbidity | comorbidities )',word*n), lemma_text_all)
flatten_text_search_all = [element for sublist in text_search_all for element in sublist if len(element) >3] 
sorted_counts = pd.DataFrame(Counter(flatten_text_search_all).most_common())
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # show all sorted counts data
    print(sorted_counts)
plt.rcParams.update({'figure.figsize':(24, 10), 'figure.dpi':300})

counts = dict(Counter(flatten_text_search_all).most_common(150))
labels, values = zip(*counts.items())

# sort your values in descending order
indSort = np.argsort(values)[::-1]

# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]

indexes = np.arange(len(labels))

width = 0.35

plt.bar(indexes, values)

# add labels
plt.xticks(rotation=90)
plt.xticks(indexes + width * 0.05, labels)
df_all_docs = pd.DataFrame(all_docs, columns=['paper_id', 'title', 'abstract', 'all_bodytext'])
df_all_docs.head()
df_risk = df_all_docs[(df_all_docs['abstract'].str.contains('comorbid')) | (df_all_docs['abstract'].str.contains('Comorbid')) | (df_all_docs['abstract'].str.contains('comorbidity')) | (df_all_docs['abstract'].str.contains('Comorbidity')) | (df_all_docs['abstract'].str.contains('comorbidities')) | (df_all_docs['abstract'].str.contains('Comorbidities'))
                  | (df_all_docs['all_bodytext'].str.contains('comorbid')) | (df_all_docs['all_bodytext'].str.contains('Comorbid')) | (df_all_docs['all_bodytext'].str.contains('comorbidity')) | (df_all_docs['all_bodytext'].str.contains('Comorbidity')) | (df_all_docs['all_bodytext'].str.contains('comorbidities')) | (df_all_docs['all_bodytext'].str.contains('Comorbidities'))]

df_risk.head()
len(df_risk)
abstract_searched = df_risk['abstract'].values
bodytext_searched = df_risk['all_bodytext'].values
df_risk_sentences = pd.DataFrame([])

for s in tqdm(abstract_searched):
    for sentence in s.split('. '):
        if "comorbid" in sentence:
            risk_sentences = pd.DataFrame([sentence])
            df_risk_sentences  = df_risk_sentences.append(risk_sentences)
#             df_risk_sentences.to_csv("df_risk_sentences.csv", encoding='utf-8', index=False)

with pd.option_context('display.max_rows', None):  # show all risk sentences
    print(df_risk_sentences)
len(df_risk_sentences)
search_term = df_risk[(df_risk['abstract'].str.contains('It has been noted that elderly patients'))]
with pd.option_context('display.max_rows', None):  
    print(search_term)
paper_id = "179df1e769292dd113cef1b54b0b43213e6b5c97.json"

counter = 0
file_select = []
for dirname, _, filenames in os.walk(file_dir):
    for filename in filenames:
#         print(os.path.join(dirname, filename))
        if filename==paper_id:
            file_select.append(os.path.join(dirname, filename))

file_select = ''.join(file_select) #convert to string

with open(file_select) as json_file:
    json_data = json.load(json_file)
for excerpt in json_data['abstract']:
    with pd.option_context('display.max_rows', None):  # show all risk sentences
        print(excerpt)
df_risk_sentences_bodytext = pd.DataFrame([])

for s in tqdm(bodytext_searched):
    for sentence in s.split('. '):
        if "comorbid" in sentence:
            risk_sentences = pd.DataFrame([sentence])
            df_risk_sentences_bodytext  = df_risk_sentences_bodytext.append(risk_sentences)
#             df_risk_sentences_bodytext.to_csv("df_risk_sentences_alldocs.csv", encoding='utf-8', index=False)

with pd.option_context('display.max_rows', None):  # show all risk sentences
    print(df_risk_sentences_bodytext)
len(df_risk_sentences_bodytext)
search_term_bodytext = df_risk[(df_risk['all_bodytext'].str.contains('The association between comorbidities and ALI'))]
with pd.option_context('display.max_rows', None):  
    print(search_term_bodytext)
paper_id = "061ffcdd4d674c4d7ce24e4aa7c5037c68596864.json"

counter = 0
file_select = []
for dirname, _, filenames in os.walk(file_dir):
    for filename in filenames:
#         print(os.path.join(dirname, filename))
        if filename==paper_id:
            file_select.append(os.path.join(dirname, filename))

file_select = ''.join(file_select) #convert to string

with open(file_select) as json_file:
    json_data = json.load(json_file)
for excerpt in json_data['body_text']:
    with pd.option_context('display.max_rows', None):  # show all risk sentences
        print(excerpt)
sample = open('kaggle/working/all_docs.txt', 'r', encoding = 'utf-8') 
s = sample.read() 
  
# Replaces escape character with space 
f = s.replace("\n", " ") 
  
data = [] 
  
# iterate through each sentence in the file 
for i in tqdm(sent_tokenize(f)): 
    temp = [] 
      
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp) 
model_skipgram = gensim.models.Word2Vec(data, min_count = 1, size = 100, window = 5, sg = 1) 
similar_words_skipgram = {search_term: [item for item in model_skipgram.wv.most_similar([search_term], topn=300)]
                  for search_term in ['comorbidity']}
similar_words_skipgram
print("Cosine similarity between 'comorbidity' " + "and 'asthma' - SG : ", 
    round(model_skipgram.wv.similarity('comorbidity', 'asthma'),2)) 
