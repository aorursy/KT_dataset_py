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
# import json

# papers_data = pd.DataFrame(columns=['PaperID','Title','Section','Text','Affilations'], index=range(len(papers)*50))



# # Remove duplicates in a list:

# def my_function(x):

#     return list(dict.fromkeys(x))



# i=0

# for j in range(len(papers)):

#     with open(papers[j]) as file:

#         content = json.load(file)

        

#         # ID and Title:

#         pap_id = content['paper_id']

#         title =  content['metadata']['title']

        

#         # Affiations:

#         affiliation = []

#         for sec in content['metadata']['authors']:

#             try:

#                 affiliation.append(sec['affiliation']['institution'])

#             except:

#                 pass

#         affiliation = my_function(affiliation)

        

#         # Abstract

#         for sec in content['abstract']:

#             papers_data.iloc[i, 0] = pap_id

#             papers_data.iloc[i, 1] = title

#             papers_data.iloc[i, 2] = sec['section']

#             papers_data.iloc[i, 3] = sec['text']

#             papers_data.iloc[i, 4] = affiliation

#             i = i + 1

            

#         # Body text

#         for sec in content['body_text']:

#             papers_data.iloc[i, 0] = pap_id

#             papers_data.iloc[i, 1] = title

#             papers_data.iloc[i, 2] = sec['section']

#             papers_data.iloc[i, 3] = sec['text']

#             papers_data.iloc[i, 4] = affiliation

#             i = i + 1



# papers_data.dropna(inplace=True)

# papers_data = papers_data.astype(str).drop_duplicates() 



# # Text processing:

# import nltk

# nltk.download('punkt')

# # Lowercase:

# for i in range(len(papers_data)):

#     try:

#         papers_data.iloc[i, 1] = papers_data.iloc[i, 1].lower()

#         papers_data.iloc[i, 2] = papers_data.iloc[i, 2].lower()

#         papers_data.iloc[i, 3] = papers_data.iloc[i, 3].lower()

#         papers_data.iloc[i, 4] = papers_data.iloc[i, 4].lower()

#     except:

#         pass

    

# # Tokenization:



# from nltk.tokenize import word_tokenize, sent_tokenize , RegexpTokenizer



# tokenizer = RegexpTokenizer(r'\w+') # remove punctuation

# papers_data["Title_Tokens_words"] = [list() for x in range(len(papers_data.index))]

# papers_data["Text_Tokens_words"] = [list() for x in range(len(papers_data.index))]



# for i in range(len(papers_data)):

#     try:

#         papers_data.iloc[i, 5] = tokenizer.tokenize(papers_data.iloc[i, 1])

#         papers_data.iloc[i, 6] = tokenizer.tokenize(papers_data.iloc[i, 3])

#     except:

#         pass

    

# # Remove stopwords:

# nltk.download('stopwords')

# from nltk.corpus import stopwords

# stop_words = set(stopwords.words('english')) 



# for i in range(len(papers_data)):

#     try:

#         papers_data.iloc[i, 5] = [w for w in papers_data.iloc[i, 5] if not w in stop_words] 

#         papers_data.iloc[i, 6] = [w for w in papers_data.iloc[i, 6] if not w in stop_words]

#     except:

#         pass

    

# # Words count:  

# papers_data["Words_count"] = 0



# # for i in range(len(papers_data)):

# #     try:

# #         papers_data.iloc[i, 7] = len(papers_data.iloc[i, 6])

# #     except:

# #         pass

    

# # Lemmatization :

# nltk.download('wordnet')



# from nltk.stem import WordNetLemmatizer



# wordnet_lemmatizer = WordNetLemmatizer()



# papers_data["Text_Lem_words"] = [list() for x in range(len(papers_data))]



# for i in range(len(papers_data)):

#     for j in range(len(papers_data.iloc[i, 6])):

#         papers_data.iloc[i, 8].append(wordnet_lemmatizer.lemmatize(papers_data.iloc[i, 6][j]))

        

# papers_data["Title_Lem_words"] = [list() for x in range(len(papers_data))]



# for i in range(len(papers_data)):

#     for j in range(len(papers_data.iloc[i, 5])):

#         papers_data.iloc[i, 9].append(wordnet_lemmatizer.lemmatize(papers_data.iloc[i, 5][j]))

        

# papers_data.to_csv("/kaggle/input/processed-researches-data/papers_data_final.csv")

print("Preprocessing done!")
papers_data = pd.read_csv("/kaggle/input/processed-researches-data/papers_data_final.csv")
del papers_data['Unnamed: 0']
papers_data.head()
import ast

papers_data['Affilations'] = papers_data['Affilations'].apply(lambda x: ast.literal_eval(x))

papers_data['Text_Lem_words'] = papers_data['Text_Lem_words'].apply(lambda x: ast.literal_eval(x))
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet



wordnet_lemmatizer = WordNetLemmatizer()

def my_function(x):

    return list(dict.fromkeys(x))



keywords =['smoking','disease', 'pre-existing', 'pulmonary','virus','covid-19','covid19','risk','factor']

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
# At least 7 of words in common

def common_member(a, b): 

    a_set = set(a) 

    b_set = set(b) 

    if len(a_set.intersection(b_set)) > 6: 

        return(True)  

    return(False)    

  

task2_1_1 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task2_1_1.append(i)

    

len(task2_1_1)
## Results for task 1.1 :

for i in task2_1_1:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 2.1.1:

print(task2_1_1)
task2_1_1_rank = papers_data.iloc[task2_1_1, :]

task2_1_1_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

meta = meta.rename(columns={"title": "Title"})

for i in range(len(meta)):

    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()

    

meta['Title'] = meta['Title'].astype(str) 

task2_1_1_rank['Title'] = task2_1_1_rank['Title'].astype(str) 



task2_1_1_rank = pd.merge(task2_1_1_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task2_1_1_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task2_1_1_rank['publish_time'] = task2_1_1_rank['publish_time'].apply(lambda x:  str(x).replace('Nov-Dec',''))

task2_1_1_rank['publish_time'] = task2_1_1_rank['publish_time'].apply(lambda x:  str(x).replace('Oct 28 Mar-Apr',''))

task2_1_1_rank['publish_time'] = task2_1_1_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task2_1_1_rank['publish_time'] = pd.to_numeric(task2_1_1_rank['publish_time'])

task2_1_1_rank = task2_1_1_rank.sort_values(by='publish_time', ascending=False)

task2_1_1_rank.reset_index(inplace=True,drop=True)
rank = pd.read_csv("/kaggle/input/shanghai-ranking/rank-univ.csv")

del rank['Unnamed: 0']

rank.head(5)
# Extract the affiliations score to the task's results:

task2_1_1_rank['Aff_Score'] = 0

for i in range(len(task2_1_1_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task2_1_1_rank.iloc[i, 4]:

            task2_1_1_rank.iloc[i, 11] = rank.iloc[j, 3]
task2_1_1_rank
task2_1_1_rank["Ranking_Score"] = task2_1_1_rank["publish_time"]*0.8 + task2_1_1_rank["Aff_Score"]*0.2
task2_1_1_rank.head(5)
task2_1_1_rank = task2_1_1_rank.sort_values(by='Ranking_Score', ascending=False)

task2_1_1_rank.reset_index(inplace=True,drop=True)

task2_1_1_rank
## 20 - Ranked Results for task 2.1.1 :



for i in range(len(task2_1_1_rank)):

    print("\n")

    print("PaperID: ", task2_1_1_rank.iloc[i, 0])

    print("Title: ", task2_1_1_rank.iloc[i, 1])

    print("Section: ", task2_1_1_rank.iloc[i, 2])

    print("Text: ", task2_1_1_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['co-infections','co-existing', 'respiratory', 'viral','infection','virus','risk','factor','covid19','covid-19','transmissible','virulent','co-morbiditie']

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

  

task2_1_2 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task2_1_2.append(i)

    

len(task2_1_2)
## Results for task 2.1.2 :

for i in task2_1_2:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 2.1.2:

print(task2_1_2)
task2_1_2_rank = papers_data.iloc[task2_1_2, :]

task2_1_2_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

meta = meta.rename(columns={"title": "Title"})

for i in range(len(meta)):

    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()

meta['Title'] = meta['Title'].astype(str) 

task2_1_2_rank['Title'] = task2_1_2_rank['Title'].astype(str) 



task2_1_2_rank = pd.merge(task2_1_2_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task2_1_2_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task2_1_2_rank['publish_time'] = task2_1_2_rank['publish_time'].apply(lambda x:  str(x).replace('Oct 28 Mar-Apr',''))

task2_1_2_rank['publish_time'] = task2_1_2_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task2_1_2_rank['publish_time'] = pd.to_numeric(task2_1_2_rank['publish_time'])

task2_1_2_rank = task2_1_2_rank.sort_values(by='publish_time', ascending=False)

task2_1_2_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task2_1_2_rank['Aff_Score'] = 0

for i in range(len(task2_1_2_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task2_1_2_rank.iloc[i, 4]:

            task2_1_2_rank.iloc[i, 11] = rank.iloc[j, 3]
task2_1_2_rank["Ranking_Score"] = task2_1_2_rank["publish_time"]*0.8 + task2_1_2_rank["Aff_Score"]*0.2

task2_1_2_rank = task2_1_2_rank.sort_values(by='Ranking_Score', ascending=False)

task2_1_2_rank.reset_index(inplace=True,drop=True)

task2_1_2_rank
## 20 - Ranked Results for task 2.1.2 :



for i in range(len(task2_1_2_rank)):

    print("\n")

    print("PaperID: ", task2_1_2_rank.iloc[i, 0])

    print("Title: ", task2_1_2_rank.iloc[i, 1])

    print("Section: ", task2_1_2_rank.iloc[i, 2])

    print("Text: ", task2_1_2_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['neonates', 'pregnant','women','covid19','covid-19','virus','risk','factor']

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
# At least 6 words in common

def common_member(a, b): 

    a_set = set(a) 

    b_set = set(b) 

    if len(a_set.intersection(b_set)) > 6: 

        return(True)  

    return(False)    

  

task2_1_3 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task2_1_3.append(i)

    

len(task2_1_3)
## Results for task 2.1.3 :

for i in task2_1_3:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 2.1.3:

print(task2_1_3)
task2_1_3_rank = papers_data.iloc[task2_1_3, :]

task2_1_3_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

task2_1_3_rank['Title'] = task2_1_3_rank['Title'].astype(str) 



task2_1_3_rank = pd.merge(task2_1_3_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task2_1_3_rank.dropna(inplace=True)



# Extract the year from the string publish time

task2_1_3_rank['publish_time'] = task2_1_3_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task2_1_3_rank['publish_time'] = task2_1_3_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task2_1_3_rank['publish_time'] = pd.to_numeric(task2_1_3_rank['publish_time'])

task2_1_3_rank = task2_1_3_rank.sort_values(by='publish_time', ascending=False)

task2_1_3_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task2_1_3_rank['Aff_Score'] = 0

for i in range(len(task2_1_3_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task2_1_3_rank.iloc[i, 4]:

            task2_1_3_rank.iloc[i, 11] = rank.iloc[j, 3]
task2_1_3_rank["Ranking_Score"] = task2_1_3_rank["publish_time"]*0.8 + task2_1_3_rank["Aff_Score"]*0.2

task2_1_3_rank = task2_1_3_rank.sort_values(by='Ranking_Score', ascending=False)

task2_1_3_rank.reset_index(inplace=True,drop=True)

task2_1_3_rank
## 20 - Ranked Results for task 2.1.3 :



for i in range(len(task2_1_3_rank)):

    print("\n")

    print("PaperID: ", task2_1_3_rank.iloc[i, 0])

    print("Title: ", task2_1_3_rank.iloc[i, 1])

    print("Section: ", task2_1_3_rank.iloc[i, 2])

    print("Text: ", task2_1_3_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['socio-economic', 'behavior','economic','impact','virus','covid19','covid-19','risk']

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

  

task2_1_4 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task2_1_4.append(i)

    

len(task2_1_4)
## Results for task 2.1.4 :

for i in task2_1_4:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 2.1.4:

print(task2_1_4)
task2_1_4_rank = papers_data.iloc[task2_1_4, :]

task2_1_4_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

task2_1_4_rank['Title'] = task2_1_4_rank['Title'].astype(str) 



task2_1_4_rank = pd.merge(task2_1_4_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task2_1_4_rank.dropna(inplace=True)



# Extract the year from the string publish time

task2_1_4_rank['publish_time'] = task2_1_4_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task2_1_4_rank['publish_time'] = pd.to_numeric(task2_1_4_rank['publish_time'])

task2_1_4_rank = task2_1_4_rank.sort_values(by='publish_time', ascending=False)

task2_1_4_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task2_1_4_rank['Aff_Score'] = 0

for i in range(len(task2_1_4_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task2_1_4_rank.iloc[i, 4]:

            task2_1_4_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task2_1_4_rank["Ranking_Score"] = task2_1_4_rank["publish_time"]*0.8 + task2_1_4_rank["Aff_Score"]*0.2

task2_1_4_rank = task2_1_4_rank.sort_values(by='Ranking_Score', ascending=False)

task2_1_4_rank.reset_index(inplace=True,drop=True)

task2_1_4_rank
## 20 - Ranked Results for task 2.1.4 :



for i in range(len(task2_1_4_rank)):

    print("\n")

    print("PaperID: ", task2_1_4_rank.iloc[i, 0])

    print("Title: ", task2_1_4_rank.iloc[i, 1])

    print("Section: ", task2_1_4_rank.iloc[i, 2])

    print("Text: ", task2_1_4_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['transmission', 'dynamics','virus','reproductive','incubation','period','serial','factor','risk','covid-19','covid19']

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

  

task2_2 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task2_2.append(i)

    

len(task2_2)
## Results for task 2.2 :

for i in task2_2:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 2.2 

print(task2_2)
task2_2_rank = papers_data.iloc[task2_2, :]

task2_2_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

task2_2_rank['Title'] = task2_2_rank['Title'].astype(str) 



task2_2_rank = pd.merge(task2_2_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task2_2_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task2_2_rank['publish_time'] = task2_2_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task2_2_rank['publish_time'] = task2_2_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task2_2_rank['publish_time'] = pd.to_numeric(task2_2_rank['publish_time'])

task2_2_rank = task2_2_rank.sort_values(by='publish_time', ascending=False)

task2_2_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task2_2_rank['Aff_Score'] = 0

for i in range(len(task2_2_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task2_2_rank.iloc[i, 4]:

            task2_2_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task2_2_rank["Ranking_Score"] = task2_2_rank["publish_time"]*0.8 + task2_2_rank["Aff_Score"]*0.2

task2_2_rank = task2_2_rank.sort_values(by='Ranking_Score', ascending=False)

task2_2_rank.reset_index(inplace=True,drop=True)

task2_2_rank
## 20 - Ranked Results for task 2.2 :



for i in range(len(task2_2_rank)):

    print("\n")

    print("PaperID: ", task2_2_rank.iloc[i, 0])

    print("Title: ", task2_2_rank.iloc[i, 1])

    print("Section: ", task2_2_rank.iloc[i, 2])

    print("Text: ", task2_2_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['severity', 'virus','disease','risk','fatality','symptomatic','hospitalized','patients','high-risk','group','covid-19','covid19','factor']

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

  

task2_3 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task2_3.append(i)

    

len(task2_3)
## Results for task 2.3 :

for i in task2_3:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 2.3:

print(task2_3)
task2_3_rank = papers_data.iloc[task2_3, :]

task2_3_rank.reset_index(inplace=True,drop=True)



task2_3_rank['Title'] = task2_3_rank['Title'].astype(str) 



task2_3_rank = pd.merge(task2_3_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task2_3_rank.dropna(inplace=True)



# Extract the year from the string publish time

task2_3_rank['publish_time'] = task2_3_rank['publish_time'].apply(lambda x:  str(x).replace('Oct 28 Mar-Apr',''))

task2_3_rank['publish_time'] = task2_3_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task2_3_rank['publish_time'] = pd.to_numeric(task2_3_rank['publish_time'])

task2_3_rank = task2_3_rank.sort_values(by='publish_time', ascending=False)

task2_3_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task2_3_rank['Aff_Score'] = 0

for i in range(len(task2_3_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task2_3_rank.iloc[i, 4]:

            task2_3_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task2_3_rank["Ranking_Score"] = task2_3_rank["publish_time"]*0.8 + task2_3_rank["Aff_Score"]*0.2

task2_3_rank = task2_3_rank.sort_values(by='Ranking_Score', ascending=False)

task2_3_rank.reset_index(inplace=True,drop=True)

task2_3_rank
## 20 - Ranked Results for task 2.3 :



for i in range(len(task2_3_rank)):

    print("\n")

    print("PaperID: ", task2_3_rank.iloc[i, 0])

    print("Title: ", task2_3_rank.iloc[i, 1])

    print("Section: ", task2_3_rank.iloc[i, 2])

    print("Text: ", task2_3_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['susceptibility', 'populations','virus','covid19','covid-19','risk','factor']

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
# At least 6 words in common

def common_member(a, b): 

    a_set = set(a) 

    b_set = set(b) 

    if len(a_set.intersection(b_set)) > 5: 

        return(True)  

    return(False)    

  

task2_4 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task2_4.append(i)

    

len(task2_4)
## Results for task 2.4 :

for i in task2_4:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task2_4_rank = papers_data.iloc[task2_4, :]

task2_4_rank.reset_index(inplace=True,drop=True)



task2_4_rank['Title'] = task2_4_rank['Title'].astype(str) 



task2_4_rank = pd.merge(task2_4_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task2_4_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task2_4_rank['publish_time'] = task2_4_rank['publish_time'].apply(lambda x:  str(x).replace('Oct 28 Mar-Apr',''))

task2_4_rank['publish_time'] = task2_4_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task2_4_rank['publish_time'] = pd.to_numeric(task2_4_rank['publish_time'])

task2_4_rank = task2_4_rank.sort_values(by='publish_time', ascending=False)

task2_4_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task2_4_rank['Aff_Score'] = 0

for i in range(len(task2_4_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task2_4_rank.iloc[i, 4]:

            task2_4_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task2_4_rank["Ranking_Score"] = task2_4_rank["publish_time"]*0.8 + task2_4_rank["Aff_Score"]*0.2

task2_4_rank = task2_4_rank.sort_values(by='Ranking_Score', ascending=False)

task2_4_rank.reset_index(inplace=True,drop=True)

task2_4_rank
## 20 - Ranked Results for task 2.4 :



for i in range(len(task2_4_rank)):

    print("\n")

    print("PaperID: ", task2_4_rank.iloc[i, 0])

    print("Title: ", task2_4_rank.iloc[i, 1])

    print("Section: ", task2_4_rank.iloc[i, 2])

    print("Text: ", task2_4_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['public', 'health','mitigation','measures','effective','control', 'covid19','covid-19','risk','factor']

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

  

task2_5 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task2_5.append(i)

    

len(task2_5)
## Results for task 2.5 :

for i in task2_5:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task2_5_rank = papers_data.iloc[task2_5, :]

task2_5_rank.reset_index(inplace=True,drop=True)



task2_5_rank['Title'] = task2_5_rank['Title'].astype(str) 



task2_5_rank = pd.merge(task2_5_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task2_5_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task2_5_rank['publish_time'] = task2_5_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task2_5_rank['publish_time'] = task2_5_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task2_5_rank['publish_time'] = pd.to_numeric(task2_5_rank['publish_time'])

task2_5_rank = task2_5_rank.sort_values(by='publish_time', ascending=False)

task2_5_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task2_5_rank['Aff_Score'] = 0

for i in range(len(task2_5_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task2_5_rank.iloc[i, 4]:

            task2_5_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task2_5_rank["Ranking_Score"] = task2_5_rank["publish_time"]*0.8 + task2_5_rank["Aff_Score"]*0.2

task2_5_rank = task2_5_rank.sort_values(by='Ranking_Score', ascending=False)

task2_5_rank.reset_index(inplace=True,drop=True)

task2_5_rank
## 20 - Ranked Results for task 1.8 :



for i in range(len(task2_5_rank)):

    print("\n")

    print("PaperID: ", task2_5_rank.iloc[i, 0])

    print("Title: ", task2_5_rank.iloc[i, 1])

    print("Section: ", task2_5_rank.iloc[i, 2])

    print("Text: ", task2_5_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
task2_1_1_rank.to_csv("task2_1_1_rank.csv")

task2_1_2_rank.to_csv("task2_1_2_rank.csv")

task2_1_3_rank.to_csv("task2_1_3_rank.csv")

task2_1_4_rank.to_csv("task2_1_4_rank.csv")

task2_2_rank.to_csv("task2_2_rank.csv")

task2_3_rank.to_csv("task2_3_rank.csv")

task2_4_rank.to_csv("task2_4_rank.csv")

task2_5_rank.to_csv("task2_5_rank.csv")