import pandas as pd
# Meta data :
meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
meta.head(3)
meta.shape
meta.columns
meta.dtypes
# Charge all .Json files:
import glob
papers = glob.glob(f'/kaggle/input/CORD-19-research-challenge/**/*.json', recursive=True)
len(papers)
import json
papers_data = pd.DataFrame(columns=['PaperID','Title','Section','Text','Affilations'], index=range(len(papers)*50))

# # Remove duplicates in a list:
def my_function(x):
    return list(dict.fromkeys(x))

i=0
for j in range(len(papers)):
    with open(papers[j]) as file:
        content = json.load(file)
        
        # ID and Title:
        pap_id = content['paper_id']
        title =  content['metadata']['title']
        
        # Affiations:
        affiliation = []
        for sec in content['metadata']['authors']:
            try:
                affiliation.append(sec['affiliation']['institution'])
            except:
                pass
        affiliation = my_function(affiliation)
        
#         # Abstract
        for sec in content['abstract']:
            papers_data.iloc[i, 0] = pap_id
            papers_data.iloc[i, 1] = title
            papers_data.iloc[i, 2] = sec['section']
            papers_data.iloc[i, 3] = sec['text']
            papers_data.iloc[i, 4] = affiliation
            i = i + 1
            
#         # Body text
        for sec in content['body_text']:
            papers_data.iloc[i, 0] = pap_id
            papers_data.iloc[i, 1] = title
            papers_data.iloc[i, 2] = sec['section']
            papers_data.iloc[i, 3] = sec['text']
            papers_data.iloc[i, 4] = affiliation
            i = i + 1

papers_data.dropna(inplace=True)
papers_data = papers_data.astype(str).drop_duplicates() 

# # Text processing:
import nltk
nltk.download('punkt')
# Lowercase:
for i in range(len(papers_data)):
    try:
        papers_data.iloc[i, 1] = papers_data.iloc[i, 1].lower()
        papers_data.iloc[i, 2] = papers_data.iloc[i, 2].lower()
        papers_data.iloc[i, 3] = papers_data.iloc[i, 3].lower()
        papers_data.iloc[i, 4] = papers_data.iloc[i, 4].lower()
    except:
        pass
    
# # Tokenization:

from nltk.tokenize import word_tokenize, sent_tokenize , RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+') # remove punctuation
papers_data["Title_Tokens_words"] = [list() for x in range(len(papers_data.index))]
papers_data["Text_Tokens_words"] = [list() for x in range(len(papers_data.index))]

for i in range(len(papers_data)):
    try:
        papers_data.iloc[i, 5] = tokenizer.tokenize(papers_data.iloc[i, 1])
        papers_data.iloc[i, 6] = tokenizer.tokenize(papers_data.iloc[i, 3])
    except:
        pass
    
# Remove stopwords:
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 

for i in range(len(papers_data)):
    try:
        papers_data.iloc[i, 5] = [w for w in papers_data.iloc[i, 5] if not w in stop_words] 
        papers_data.iloc[i, 6] = [w for w in papers_data.iloc[i, 6] if not w in stop_words]
    except:
        pass
    
# # Words count:  
papers_data["Words_count"] = 0

# # for i in range(len(papers_data)):
# #     try:
# #         papers_data.iloc[i, 7] = len(papers_data.iloc[i, 6])
# #     except:
# #         pass
    
# # Lemmatization :
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

papers_data["Text_Lem_words"] = [list() for x in range(len(papers_data))]

for i in range(len(papers_data)):
    for j in range(len(papers_data.iloc[i, 6])):
        papers_data.iloc[i, 8].append(wordnet_lemmatizer.lemmatize(papers_data.iloc[i, 6][j]))
        
papers_data["Title_Lem_words"] = [list() for x in range(len(papers_data))]

for i in range(len(papers_data)):
    for j in range(len(papers_data.iloc[i, 5])):
        papers_data.iloc[i, 9].append(wordnet_lemmatizer.lemmatize(papers_data.iloc[i, 5][j]))
        
papers_data.to_csv("/kaggle/input/processed-researches-data/papers_data_final.csv")
print("Preprocessing done!")
papers_data = pd.read_csv("/kaggle/input/processed-researches-data/papers_data_final.csv")
del papers_data['Unnamed: 0']
import ast
papers_data['Affilations'] = papers_data['Affilations'].apply(lambda x: ast.literal_eval(x))
papers_data['Text_Lem_words'] = papers_data['Text_Lem_words'].apply(lambda x: ast.literal_eval(x))
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

wordnet_lemmatizer = WordNetLemmatizer()
def my_function(x):
    return list(dict.fromkeys(x))


keywords =['widespread','exposure', 'policy', 'recommendation','mitigation','measures','denominator','mechanism','test','sharing','demographic','asymptomatic','disease','serosurvey','convalescent','detection','screening','neutralizing','antibodies','elisas']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 11 of words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 11: 
        return(True)  
    return(False)    
  
task6_1 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_1.append(i)
    
len(task6_1)
## Results for task 6.1 :
for i in task6_1:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
# Task 6.1:
print(task6_1)
task6_1_rank = papers_data.iloc[task6_1, :]
task6_1_rank.reset_index(inplace=True,drop=True)

# Grab the publish year from the meta data file:
meta = meta.rename(columns={"title": "Title"})
for i in range(len(meta)):
    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()
meta['Title'] = meta['Title'].astype(str) 
task6_1_rank['Title'] = task6_1_rank['Title'].astype(str) 

task6_1_rank = pd.merge(task6_1_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_1_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task6_1_rank['publish_time'] = task6_1_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task6_1_rank['publish_time'] = task6_1_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_1_rank['publish_time'] = pd.to_numeric(task6_1_rank['publish_time'])
task6_1_rank = task6_1_rank.sort_values(by='publish_time', ascending=False)
task6_1_rank.reset_index(inplace=True,drop=True)
rank = pd.read_csv("/kaggle/input/shanghai-ranking/rank-univ.csv")
del rank['Unnamed: 0']
rank.head(5)
# Extract the affiliations score to the task's results:
task6_1_rank['Aff_Score'] = 0
for i in range(len(task6_1_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_1_rank.iloc[i, 4]:
            task6_1_rank.iloc[i, 11] = rank.iloc[j, 3]
task6_1_rank["Ranking_Score"] = task6_1_rank["publish_time"]*0.8 + task6_1_rank["Aff_Score"]*0.2
task6_1_rank = task6_1_rank.sort_values(by='Ranking_Score', ascending=False)
task6_1_rank.reset_index(inplace=True,drop=True)
task6_1_rank
## 20 - Ranked Results for task 1.1 :

for i in range(len(task6_1_rank)):
    print("\n")
    print("PaperID: ", task6_1_rank.iloc[i, 0])
    print("Title: ", task6_1_rank.iloc[i, 1])
    print("Section: ", task6_1_rank.iloc[i, 2])
    print("Text: ", task6_1_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['efforts','capacity', 'diagnostic', 'platforms','surveillance','covid19','covid-19']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 5 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 4: 
        return(True)  
    return(False)    
  
task6_2 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_2.append(i)
    
len(task6_2)
## Results for task 6.2 :
for i in task6_2:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
# Task 6.2:
print(task6_2)
task6_2_rank = papers_data.iloc[task6_2, :]
task6_2_rank.reset_index(inplace=True,drop=True)

# Grab the publish year from the meta data file:

task6_2_rank['Title'] = task6_2_rank['Title'].astype(str) 

task6_2_rank = pd.merge(task6_2_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_2_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_2_rank['publish_time'] = task6_2_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_2_rank['publish_time'] = pd.to_numeric(task6_2_rank['publish_time'])
task6_2_rank = task6_2_rank.sort_values(by='publish_time', ascending=False)
task6_2_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_2_rank['Aff_Score'] = 0
for i in range(len(task6_2_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_2_rank.iloc[i, 4]:
            task6_2_rank.iloc[i, 11] = rank.iloc[j, 3]
task6_2_rank["Ranking_Score"] = task6_2_rank["publish_time"]*0.8 + task6_2_rank["Aff_Score"]*0.2
task6_2_rank = task6_2_rank.sort_values(by='Ranking_Score', ascending=False)
task6_2_rank.reset_index(inplace=True,drop=True)
task6_2_rank
## 20 - Ranked Results for task 1.2 :

for i in range(len(task6_2_rank)):
    print("\n")
    print("PaperID: ", task6_2_rank.iloc[i, 0])
    print("Title: ", task6_2_rank.iloc[i, 1])
    print("Section: ", task6_2_rank.iloc[i, 2])
    print("Text: ", task6_2_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['recruitment', 'support','coordination','local','expertise','capacity','public','private','commercial','non-profit','academic','legal','ethical','communication','operational','issues']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 12 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 11: 
        return(True)  
    return(False)    
  
task6_3 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_3.append(i)
    
len(task6_3)
## Results for task 6.3 :
for i in task6_3:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
# Task 6.3:
print(task6_3)
task6_3_rank = papers_data.iloc[task6_3, :]
task6_3_rank.reset_index(inplace=True,drop=True)

# Grab the publish year from the meta data file:
task6_3_rank['Title'] = task6_3_rank['Title'].astype(str) 

task6_3_rank = pd.merge(task6_3_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_3_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_3_rank['publish_time'] = task6_3_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_3_rank['publish_time'] = pd.to_numeric(task6_3_rank['publish_time'])
task6_3_rank = task6_3_rank.sort_values(by='publish_time', ascending=False)
task6_3_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_3_rank['Aff_Score'] = 0
for i in range(len(task6_3_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_3_rank.iloc[i, 4]:
            task6_3_rank.iloc[i, 11] = rank.iloc[j, 3]
task6_3_rank["Ranking_Score"] = task6_3_rank["publish_time"]*0.8 + task6_3_rank["Aff_Score"]*0.2
task6_3_rank = task6_3_rank.sort_values(by='Ranking_Score', ascending=False)
task6_3_rank.reset_index(inplace=True,drop=True)
task6_3_rank
## 20 - Ranked Results for task 1.3 :

for i in range(len(task6_3_rank)):
    print("\n")
    print("PaperID: ", task6_3_rank.iloc[i, 0])
    print("Title: ", task6_3_rank.iloc[i, 1])
    print("Section: ", task6_3_rank.iloc[i, 2])
    print("Text: ", task6_3_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['national', 'guidance','guidelines','practices','states','universities','private','laboratories','test','public','health','officials','public','covid19','covid-19','virus']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 12 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 11: 
        return(True)  
    return(False)    
  
task6_4 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_4.append(i)
    
len(task6_4)
## Results for task 6.4 :
for i in task6_4:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
# Task 6.4:
print(task6_4)
task6_4_rank = papers_data.iloc[task6_4, :]
task6_4_rank.reset_index(inplace=True,drop=True)

# Grab the publish year from the meta data file:

task6_4_rank['Title'] = task6_4_rank['Title'].astype(str) 

task6_4_rank = pd.merge(task6_4_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_4_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_4_rank['publish_time'] = task6_4_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))
task6_4_rank['publish_time'] = task6_4_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_4_rank['publish_time'] = pd.to_numeric(task6_4_rank['publish_time'])
task6_4_rank = task6_4_rank.sort_values(by='publish_time', ascending=False)
task6_4_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_4_rank['Aff_Score'] = 0
for i in range(len(task6_4_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_4_rank.iloc[i, 4]:
            task6_4_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_4_rank["Ranking_Score"] = task6_4_rank["publish_time"]*0.8 + task6_4_rank["Aff_Score"]*0.2
task6_4_rank = task6_4_rank.sort_values(by='Ranking_Score', ascending=False)
task6_4_rank.reset_index(inplace=True,drop=True)
task6_4_rank
## 20 - Ranked Results for task 1.4 :

for i in range(len(task6_4_rank)):
    print("\n")
    print("PaperID: ", task6_4_rank.iloc[i, 0])
    print("Title: ", task6_4_rank.iloc[i, 1])
    print("Section: ", task6_4_rank.iloc[i, 2])
    print("Text: ", task6_4_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['development', 'point-of-care','covid19','covid-19','test','influenza','bed-side','recognizing','dischtradeoffs','speed','accessibility','accuracy']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 7 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 6: 
        return(True)  
    return(False)    
  
task6_5 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_5.append(i)
    
len(task6_5)
## Results for task 6.5 :
for i in task6_5:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
# Task 6.5 
print(task6_5)
task6_5_rank = papers_data.iloc[task6_5, :]
task6_5_rank.reset_index(inplace=True,drop=True)

# Grab the publish year from the meta data file:

task6_5_rank['Title'] = task6_5_rank['Title'].astype(str) 

task6_5_rank = pd.merge(task6_5_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_5_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_5_rank['publish_time'] = task6_5_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_5_rank['publish_time'] = pd.to_numeric(task6_5_rank['publish_time'])
task6_5_rank = task6_5_rank.sort_values(by='publish_time', ascending=False)
task6_5_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_5_rank['Aff_Score'] = 0
for i in range(len(task6_5_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_5_rank.iloc[i, 4]:
            task6_5_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_5_rank["Ranking_Score"] = task6_5_rank["publish_time"]*0.8 + task6_5_rank["Aff_Score"]*0.2
task6_5_rank = task6_5_rank.sort_values(by='Ranking_Score', ascending=False)
task6_5_rank.reset_index(inplace=True,drop=True)
task6_5_rank
## 20 - Ranked Results for task 1.5 :

for i in range(len(task6_5_rank)):
    print("\n")
    print("PaperID: ", task6_5_rank.iloc[i, 0])
    print("Title: ", task6_5_rank.iloc[i, 1])
    print("Section: ", task6_5_rank.iloc[i, 2])
    print("Text: ", task6_5_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['rapid', 'design','execution','target','surveillance','experiments','testers','pcr','area','report','entity','aid','longitudinal','sample','critical','impact','ad-hoc','intervention','local','recorded','covid19','covidd-19','virus']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 16 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 15: 
        return(True)  
    return(False)    
  
task6_6 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_6.append(i)
    
len(task6_6)
## Results for task 6.6 :
for i in task6_6:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
# Task 6.6:
print(task6_6)
task6_6_rank = papers_data.iloc[task6_6, :]
task6_6_rank.reset_index(inplace=True,drop=True)

task6_6_rank['Title'] = task6_6_rank['Title'].astype(str) 

task6_6_rank = pd.merge(task6_6_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_6_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_6_rank['publish_time'] = task6_6_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_6_rank['publish_time'] = pd.to_numeric(task6_6_rank['publish_time'])
task6_6_rank = task6_6_rank.sort_values(by='publish_time', ascending=False)
task6_6_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_6_rank['Aff_Score'] = 0
for i in range(len(task6_6_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_6_rank.iloc[i, 4]:
            task6_6_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_6_rank["Ranking_Score"] = task6_6_rank["publish_time"]*0.8 + task6_6_rank["Aff_Score"]*0.2
task6_6_rank = task6_6_rank.sort_values(by='Ranking_Score', ascending=False)
task6_6_rank.reset_index(inplace=True,drop=True)
task6_6_rank
## 20 - Ranked Results for task 1.6 :

for i in range(len(task6_6_rank)):
    print("\n")
    print("PaperID: ", task6_6_rank.iloc[i, 0])
    print("Title: ", task6_6_rank.iloc[i, 1])
    print("Section: ", task6_6_rank.iloc[i, 2])
    print("Text: ", task6_6_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['separation', 'assay','virus','development','issues','instruments','role','private','sector','devices','covid19','coronavirus','covid-19']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 13 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 12: 
        return(True)  
    return(False)    
  
task6_7 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_7.append(i)
    
len(task6_7)
## Results for task 6.7 :
for i in task6_7:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
task6_7_rank = papers_data.iloc[task6_7, :]
task6_7_rank.reset_index(inplace=True,drop=True)

task6_7_rank['Title'] = task6_7_rank['Title'].astype(str) 

task6_7_rank = pd.merge(task6_7_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_7_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_7_rank['publish_time'] = task6_7_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_7_rank['publish_time'] = pd.to_numeric(task6_7_rank['publish_time'])
task6_7_rank = task6_7_rank.sort_values(by='publish_time', ascending=False)
task6_7_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_7_rank['Aff_Score'] = 0
for i in range(len(task6_7_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_7_rank.iloc[i, 4]:
            task6_7_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_7_rank["Ranking_Score"] = task6_7_rank["publish_time"]*0.8 + task6_7_rank["Aff_Score"]*0.2
task6_7_rank = task6_7_rank.sort_values(by='Ranking_Score', ascending=False)
task6_7_rank.reset_index(inplace=True,drop=True)
task6_7_rank
## 20 - Ranked Results for task 1.7 :

for i in range(len(task6_7_rank)):
    print("\n")
    print("PaperID: ", task6_7_rank.iloc[i, 0])
    print("Title: ", task6_7_rank.iloc[i, 1])
    print("Section: ", task6_7_rank.iloc[i, 2])
    print("Text: ", task6_7_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['efforts', 'track','evolution','virus','genetic','mutation','locking','reagent','surveillance','detection','scheme','covid19','covid-19','coronavirus']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 9 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 9: 
        return(True)  
    return(False)    
  
task6_8 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_8.append(i)
    
len(task6_8)
## Results for task 6.8 :
for i in task6_8:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
task6_8_rank = papers_data.iloc[task6_8, :]
task6_8_rank.reset_index(inplace=True,drop=True)

task6_8_rank['Title'] = task6_8_rank['Title'].astype(str) 

task6_8_rank = pd.merge(task6_8_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_8_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task6_8_rank['publish_time'] = task6_8_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))
task6_8_rank['publish_time'] = task6_8_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_8_rank['publish_time'] = pd.to_numeric(task6_8_rank['publish_time'])
task6_8_rank = task6_8_rank.sort_values(by='publish_time', ascending=False)
task6_8_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_8_rank['Aff_Score'] = 0
for i in range(len(task6_8_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_8_rank.iloc[i, 4]:
            task6_8_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_8_rank["Ranking_Score"] = task6_8_rank["publish_time"]*0.8 + task6_8_rank["Aff_Score"]*0.2
task6_8_rank = task6_8_rank.sort_values(by='Ranking_Score', ascending=False)
task6_8_rank.reset_index(inplace=True,drop=True)
task6_8_rank
## 20 - Ranked Results for task 1.8 :

for i in range(len(task6_8_rank)):
    print("\n")
    print("PaperID: ", task6_8_rank.iloc[i, 0])
    print("Title: ", task6_8_rank.iloc[i, 1])
    print("Section: ", task6_8_rank.iloc[i, 2])
    print("Text: ", task6_8_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['latency', 'issues','viral','detect','pathogen', 'needed','biological','environment','sample']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 11 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 10: 
        return(True)  
    return(False)    
  
task6_9 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_9.append(i)
    
len(task6_9)
## Results for task 6.9 :
for i in task6_9:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
task6_9_rank = papers_data.iloc[task6_9, :]
task6_9_rank.reset_index(inplace=True,drop=True)

task6_9_rank['Title'] = task6_9_rank['Title'].astype(str) 

task6_9_rank = pd.merge(task6_9_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_9_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_9_rank['publish_time'] = task6_9_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_9_rank['publish_time'] = pd.to_numeric(task6_9_rank['publish_time'])
task6_9_rank = task6_9_rank.sort_values(by='publish_time', ascending=False)
task6_9_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_9_rank['Aff_Score'] = 0
for i in range(len(task6_9_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_9_rank.iloc[i, 4]:
            task6_9_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_9_rank["Ranking_Score"] = task6_9_rank["publish_time"]*0.8 + task6_9_rank["Aff_Score"]*0.2
task6_9_rank = task6_9_rank.sort_values(by='Ranking_Score', ascending=False)
task6_9_rank.reset_index(inplace=True,drop=True)
task6_9_rank
## 20 - Ranked Results for task 1.9 :

for i in range(len(task6_9_rank)):
    print("\n")
    print("PaperID: ", task6_9_rank.iloc[i, 0])
    print("Title: ", task6_9_rank.iloc[i, 1])
    print("Section: ", task6_9_rank.iloc[i, 2])
    print("Text: ", task6_9_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['diagnostics', 'host','response','markers','cytokines','detect','early','disease','predict','progression','clinical','practice','efficacy','therapeutic','interventions','covid-19','covid19','virus']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 11 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 11: 
        return(True)  
    return(False)    
  
task6_10 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_10.append(i)
    
len(task6_10)
## Results for task 6.10 :
for i in task6_10:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
task6_10_rank = papers_data.iloc[task6_10, :]
task6_10_rank.reset_index(inplace=True,drop=True)

task6_10_rank['Title'] = task6_10_rank['Title'].astype(str) 

task6_10_rank = pd.merge(task6_10_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_10_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_10_rank['publish_time'] = task6_10_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_10_rank['publish_time'] = pd.to_numeric(task6_10_rank['publish_time'])
task6_10_rank = task6_10_rank.sort_values(by='publish_time', ascending=False)
task6_10_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_10_rank['Aff_Score'] = 0
for i in range(len(task6_10_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_10_rank.iloc[i, 4]:
            task6_10_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_10_rank["Ranking_Score"] = task6_10_rank["publish_time"]*0.8 + task6_10_rank["Aff_Score"]*0.2
task6_10_rank = task6_10_rank.sort_values(by='Ranking_Score', ascending=False)
task6_10_rank.reset_index(inplace=True,drop=True)
task6_10_rank
## 20 - Ranked Results for task 1.10 :

for i in range(len(task6_10_rank)):
    print("\n")
    print("PaperID: ", task6_10_rank.iloc[i, 0])
    print("Title: ", task6_10_rank.iloc[i, 1])
    print("Section: ", task6_10_rank.iloc[i, 2])
    print("Text: ", task6_10_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['policies','protocol','coronavirus','corona','covid','testing','screening','covid19','covid-19']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 5 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 4: 
        return(True)  
    return(False)    
  
task6_11 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_11.append(i)
    
len(task6_11)
## Results for task 6.11 :
for i in task6_11:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
task6_11_rank = papers_data.iloc[task6_11, :]
task6_11_rank.reset_index(inplace=True,drop=True)

task6_11_rank['Title'] = task6_11_rank['Title'].astype(str) 

task6_11_rank = pd.merge(task6_11_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_11_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_11_rank['publish_time'] = task6_11_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_11_rank['publish_time'] = pd.to_numeric(task6_11_rank['publish_time'])
task6_11_rank = task6_11_rank.sort_values(by='publish_time', ascending=False)
task6_11_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_11_rank['Aff_Score'] = 0
for i in range(len(task6_11_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_11_rank.iloc[i, 4]:
            task6_11_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_11_rank["Ranking_Score"] = task6_11_rank["publish_time"]*0.8 + task6_11_rank["Aff_Score"]*0.2
task6_11_rank = task6_11_rank.sort_values(by='Ranking_Score', ascending=False)
task6_11_rank.reset_index(inplace=True,drop=True)
task6_11_rank
## 20 - Ranked Results for task 1.11 :

for i in range(len(task6_11_rank)):
    print("\n")
    print("PaperID: ", task6_11_rank.iloc[i, 0])
    print("Title: ", task6_11_rank.iloc[i, 1])
    print("Section: ", task6_11_rank.iloc[i, 2])
    print("Text: ", task6_11_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['policies', 'effects','control','supplies','mass','testing','reagents','swabs','covid19','virus','covid-19']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 10 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 10: 
        return(True)  
    return(False)    
  
task6_12 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_12.append(i)
    
len(task6_12)
## Results for task 6.12 :
for i in task6_12:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
task6_12_rank = papers_data.iloc[task6_12, :]
task6_12_rank.reset_index(inplace=True,drop=True)

task6_12_rank['Title'] = task6_12_rank['Title'].astype(str) 

task6_12_rank = pd.merge(task6_12_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_12_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task6_12_rank['publish_time'] = task6_12_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))
task6_12_rank['publish_time'] = task6_12_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_12_rank['publish_time'] = pd.to_numeric(task6_12_rank['publish_time'])
task6_12_rank = task6_12_rank.sort_values(by='publish_time', ascending=False)
task6_12_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_12_rank['Aff_Score'] = 0
for i in range(len(task6_12_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_12_rank.iloc[i, 4]:
            task6_12_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_12_rank["Ranking_Score"] = task6_12_rank["publish_time"]*0.8 + task6_12_rank["Aff_Score"]*0.2
task6_12_rank = task6_12_rank.sort_values(by='Ranking_Score', ascending=False)
task6_12_rank.reset_index(inplace=True,drop=True)
task6_12_rank
## 20 - Ranked Results for task 1.12 :

for i in range(len(task6_12_rank)):
    print("\n")
    print("PaperID: ", task6_12_rank.iloc[i, 0])
    print("Title: ", task6_12_rank.iloc[i, 1])
    print("Section: ", task6_12_rank.iloc[i, 2])
    print("Text: ", task6_12_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['technology', 'roadmap','diagnostics','covid19','virus','covid-19']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 3 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 2: 
        return(True)  
    return(False)    
  
task6_13 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_13.append(i)
    
len(task6_13)
## Results for task 6.13 :
for i in task6_13:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
task6_13_rank = papers_data.iloc[task6_13, :]
task6_13_rank.reset_index(inplace=True,drop=True)

task6_13_rank['Title'] = task6_13_rank['Title'].astype(str) 

task6_13_rank = pd.merge(task6_13_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_13_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_13_rank['publish_time'] = task6_13_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_13_rank['publish_time'] = pd.to_numeric(task6_13_rank['publish_time'])
task6_13_rank = task6_13_rank.sort_values(by='publish_time', ascending=False)
task6_13_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_13_rank['Aff_Score'] = 0
for i in range(len(task6_13_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_13_rank.iloc[i, 4]:
            task6_13_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_13_rank["Ranking_Score"] = task6_13_rank["publish_time"]*0.8 + task6_13_rank["Aff_Score"]*0.2
task6_13_rank = task6_13_rank.sort_values(by='Ranking_Score', ascending=False)
task6_13_rank.reset_index(inplace=True,drop=True)
task6_13_rank
## 20 - Ranked Results for task 1.13 :

for i in range(len(task6_13_rank)):
    print("\n")
    print("PaperID: ", task6_13_rank.iloc[i, 0])
    print("Title: ", task6_13_rank.iloc[i, 1])
    print("Section: ", task6_13_rank.iloc[i, 2])
    print("Text: ", task6_13_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['barriers','scaling','diagnostic','market','coalition','accelerator','model','epidemic','preparedness','innovation','critical','funding','opportunities','streamlined','regulatory','environment']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 9 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 8: 
        return(True)  
    return(False)    
  
task6_14 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_14.append(i)
    
len(task6_14)
## Results for task 6.14 :
for i in task6_14:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
    
task6_14_rank = papers_data.iloc[task6_14, :]
task6_14_rank.reset_index(inplace=True,drop=True)

task6_14_rank['Title'] = task6_14_rank['Title'].astype(str) 

task6_14_rank = pd.merge(task6_14_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_14_rank.dropna(inplace=True)

# Extract the year from the string publish time
import dateutil.parser as parser
task6_14_rank['publish_time'] = task6_14_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))
task6_14_rank['publish_time'] = task6_14_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_14_rank['publish_time'] = pd.to_numeric(task6_14_rank['publish_time'])
task6_14_rank = task6_14_rank.sort_values(by='publish_time', ascending=False)
task6_14_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_14_rank['Aff_Score'] = 0
for i in range(len(task6_14_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_14_rank.iloc[i, 4]:
            task6_14_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_14_rank["Ranking_Score"] = task6_14_rank["publish_time"]*0.8 + task6_14_rank["Aff_Score"]*0.2
task6_14_rank = task6_14_rank.sort_values(by='Ranking_Score', ascending=False)
task6_14_rank.reset_index(inplace=True,drop=True)
task6_14_rank
## 20 - Ranked Results for task 6.14 :

for i in range(len(task6_14_rank)):
    print("\n")
    print("PaperID: ", task6_14_rank.iloc[i, 0])
    print("Title: ", task6_14_rank.iloc[i, 1])
    print("Section: ", task6_14_rank.iloc[i, 2])
    print("Text: ", task6_14_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['platforms','technology','response','time','employ','holistic','approaches','epidemic','covid19','didease','covid-19','future']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   
# At least 8 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 7: 
        return(True)  
    return(False)    
  
task6_15 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_15.append(i)
    
len(task6_15)
## Results for task 6.15 :
for i in task6_15:
    print("\n")
    print("PaperID: ", papers_data.iloc[i, 0])
    print("Title: ", papers_data.iloc[i, 1])
    print("Section: ", papers_data.iloc[i, 2])
    print("Text: ", papers_data.iloc[i, 3])  
    print("\n")
    
task6_15_rank = papers_data.iloc[task6_15, :]
task6_15_rank.reset_index(inplace=True,drop=True)

task6_15_rank['Title'] = task6_15_rank['Title'].astype(str) 

task6_15_rank = pd.merge(task6_15_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_15_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_15_rank['publish_time'] = task6_15_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_15_rank['publish_time'] = pd.to_numeric(task6_15_rank['publish_time'])
task6_15_rank = task6_15_rank.sort_values(by='publish_time', ascending=False)
task6_15_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:
task6_15_rank['Aff_Score'] = 0
for i in range(len(task6_15_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_15_rank.iloc[i, 4]:
            task6_15_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_15_rank["Ranking_Score"] = task6_15_rank["publish_time"]*0.8 + task6_15_rank["Aff_Score"]*0.2
task6_15_rank = task6_15_rank.sort_values(by='Ranking_Score', ascending=False)
task6_15_rank.reset_index(inplace=True,drop=True)
task6_15_rank
## 20 - Ranked Results for task 6.14 :

for i in range(len(task6_15_rank)):
    print("\n")
    print("PaperID: ", task6_15_rank.iloc[i, 0])
    print("Title: ", task6_15_rank.iloc[i, 1])
    print("Section: ", task6_15_rank.iloc[i, 2])
    print("Text: ", task6_15_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['coupling','genomic','diagnostic','test','large-scale','covid19','covid-19','epidemic','didease']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   

# At least 5 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 4: 
        return(True)  
    return(False)    
  
task6_16 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_16.append(i)
    
len(task6_16)
task6_16_rank = papers_data.iloc[task6_16, :]
task6_16_rank.reset_index(inplace=True,drop=True)

task6_16_rank['Title'] = task6_16_rank['Title'].astype(str) 

task6_16_rank = pd.merge(task6_16_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_16_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_16_rank['publish_time'] = task6_16_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_16_rank['publish_time'] = pd.to_numeric(task6_16_rank['publish_time'])
task6_16_rank = task6_16_rank.sort_values(by='publish_time', ascending=False)
task6_16_rank.reset_index(inplace=True,drop=True)

## Ranking by affiliations

# Extract the affiliations score to the task's results:
task6_16_rank['Aff_Score'] = 0
for i in range(len(task6_16_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_16_rank.iloc[i, 4]:
            task6_16_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_16_rank["Ranking_Score"] = task6_16_rank["publish_time"]*0.8 + task6_16_rank["Aff_Score"]*0.2
task6_16_rank = task6_16_rank.sort_values(by='Ranking_Score', ascending=False)
task6_16_rank.reset_index(inplace=True,drop=True)


## 20 - Ranked Results for task 6.14 :

for i in range(len(task6_16_rank)):
    print("\n")
    print("PaperID: ", task6_16_rank.iloc[i, 0])
    print("Title: ", task6_16_rank.iloc[i, 1])
    print("Section: ", task6_16_rank.iloc[i, 2])
    print("Text: ", task6_16_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['enhance','capabilities','sequencing','bioinformatics','target','region','genome','variant','covid19','covid-19','epidemic','didease']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   

# At least 7 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 9: 
        return(True)  
    return(False)    
  
task6_17 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_17.append(i)
    
len(task6_17)
task6_17_rank = papers_data.iloc[task6_17, :]
task6_17_rank.reset_index(inplace=True,drop=True)

task6_17_rank['Title'] = task6_17_rank['Title'].astype(str) 

task6_17_rank = pd.merge(task6_17_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_17_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_17_rank['publish_time'] = task6_17_rank['publish_time'].apply(lambda x: str(x).replace('Feb 5 Nov-Dec',''))

task6_17_rank['publish_time'] = task6_17_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_17_rank['publish_time'] = pd.to_numeric(task6_17_rank['publish_time'])
task6_17_rank = task6_17_rank.sort_values(by='publish_time', ascending=False)
task6_17_rank.reset_index(inplace=True,drop=True)

## Ranking by affiliations

# Extract the affiliations score to the task's results:
task6_17_rank['Aff_Score'] = 0
for i in range(len(task6_17_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_17_rank.iloc[i, 4]:
            task6_17_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_17_rank["Ranking_Score"] = task6_17_rank["publish_time"]*0.8 + task6_17_rank["Aff_Score"]*0.2
task6_17_rank = task6_17_rank.sort_values(by='Ranking_Score', ascending=False)
task6_17_rank.reset_index(inplace=True,drop=True)


## 20 - Ranked Results for task 6.14 :

for i in range(len(task6_17_rank)):
    print("\n")
    print("PaperID: ", task6_17_rank.iloc[i, 0])
    print("Title: ", task6_17_rank.iloc[i, 1])
    print("Section: ", task6_17_rank.iloc[i, 2])
    print("Text: ", task6_17_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['enhance','capacity','people','technology','data','sequencing','analytics','pathogens','covid19','covid-19','epidemic','didease','capabilities','natural','intentional']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   

# At least 7 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 6: 
        return(True)  
    return(False)    
  
task6_18 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_18.append(i)
    
len(task6_18)
task6_18_rank = papers_data.iloc[task6_18, :]
task6_18_rank.reset_index(inplace=True,drop=True)

task6_18_rank['Title'] = task6_18_rank['Title'].astype(str) 

task6_18_rank = pd.merge(task6_18_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_18_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_18_rank['publish_time'] = task6_18_rank['publish_time'].apply(lambda x: str(x).replace('Feb 5 Nov-Dec',''))

task6_18_rank['publish_time'] = task6_18_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_18_rank['publish_time'] = pd.to_numeric(task6_18_rank['publish_time'])
task6_18_rank = task6_18_rank.sort_values(by='publish_time', ascending=False)
task6_18_rank.reset_index(inplace=True,drop=True)

## Ranking by affiliations

# Extract the affiliations score to the task's results:
task6_18_rank['Aff_Score'] = 0
for i in range(len(task6_18_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_18_rank.iloc[i, 4]:
            task6_18_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_18_rank["Ranking_Score"] = task6_18_rank["publish_time"]*0.8 + task6_18_rank["Aff_Score"]*0.2
task6_18_rank = task6_18_rank.sort_values(by='Ranking_Score', ascending=False)
task6_18_rank.reset_index(inplace=True,drop=True)


## 20 - Ranked Results for task 6.14 :

for i in range(len(task6_18_rank)):
    print("\n")
    print("PaperID: ", task6_18_rank.iloc[i, 0])
    print("Title: ", task6_18_rank.iloc[i, 1])
    print("Section: ", task6_18_rank.iloc[i, 2])
    print("Text: ", task6_18_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
keywords =['health','surveillance','humans','sources','spillover','future','exposure','pathogens','covid19','covid-19','epidemic','didease','organism','evolutionary','host','bats','transmission','trafficked','farm','wildlife','domestic','food','companion','species','environment','demographic','occupation','risk','factor']
kw = []
for i in keywords:
    kw.append(wordnet_lemmatizer.lemmatize(i))
         
# Gett synonyms: 
synonyms = []
for k in kw:
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(wordnet_lemmatizer.lemmatize(l.name()))
for i in synonyms:
    kw.append(i)
    
kw = [ x for x in kw if "_" not in x ]
kw = my_function(kw)

print(kw)   

# At least 15 words in common
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if len(a_set.intersection(b_set)) > 14: 
        return(True)  
    return(False)    
  
task6_19 =[]
for i in range(len(papers_data)):
    if common_member(kw, papers_data.iloc[i, 8]):
        task6_19.append(i)
    
len(task6_19)
task6_19_rank = papers_data.iloc[task6_19, :]
task6_19_rank.reset_index(inplace=True,drop=True)

task6_19_rank['Title'] = task6_19_rank['Title'].astype(str) 

task6_19_rank = pd.merge(task6_19_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')
task6_19_rank.dropna(inplace=True)

# Extract the year from the string publish time
task6_19_rank['publish_time'] = task6_19_rank['publish_time'].apply(lambda x: str(x).replace('Feb 5 Nov-Dec',''))

task6_19_rank['publish_time'] = task6_19_rank['publish_time'].apply(lambda x: parser.parse(x).year)

# Rank the task's results by time (Freshness)
task6_19_rank['publish_time'] = pd.to_numeric(task6_19_rank['publish_time'])
task6_19_rank = task6_19_rank.sort_values(by='publish_time', ascending=False)
task6_19_rank.reset_index(inplace=True,drop=True)

## Ranking by affiliations

# Extract the affiliations score to the task's results:
task6_19_rank['Aff_Score'] = 0
for i in range(len(task6_19_rank)):
    for j in range(len(rank)):
        if rank.iloc[j, 1] in task6_19_rank.iloc[i, 4]:
            task6_19_rank.iloc[i, 11] = rank.iloc[j, 3]
            
task6_19_rank["Ranking_Score"] = task6_19_rank["publish_time"]*0.8 + task6_19_rank["Aff_Score"]*0.2
task6_19_rank = task6_19_rank.sort_values(by='Ranking_Score', ascending=False)
task6_19_rank.reset_index(inplace=True,drop=True)


## 20 - Ranked Results for task 6.14 :

for i in range(len(task6_19_rank)):
    print("\n")
    print("PaperID: ", task6_19_rank.iloc[i, 0])
    print("Title: ", task6_19_rank.iloc[i, 1])
    print("Section: ", task6_19_rank.iloc[i, 2])
    print("Text: ", task6_19_rank.iloc[i, 3])  
    print("\n")
    if i == 19:
        break
task6_1_rank.to_csv("task6_1_rank.csv")
task6_2_rank.to_csv("task6_2_rank.csv")
task6_3_rank.to_csv("task6_3_rank.csv")
task6_4_rank.to_csv("task6_4_rank.csv")
task6_5_rank.to_csv("task6_5_rank.csv")
task6_6_rank.to_csv("task6_6_rank.csv")
task6_7_rank.to_csv("task6_7_rank.csv")
task6_8_rank.to_csv("task6_8_rank.csv")
task6_9_rank.to_csv("task6_9_rank.csv")
task6_10_rank.to_csv("task6_10_rank.csv")
task6_11_rank.to_csv("task6_11_rank.csv")
task6_12_rank.to_csv("task6_12_rank.csv")
task6_13_rank.to_csv("task6_13_rank.csv")
task6_14_rank.to_csv("task6_14_rank.csv")
task6_15_rank.to_csv("task6_15_rank.csv")
task6_16_rank.to_csv("task6_16_rank.csv")
task6_17_rank.to_csv("task6_17_rank.csv")
task6_18_rank.to_csv("task6_18_rank.csv")
task6_19_rank.to_csv("task6_19_rank.csv")