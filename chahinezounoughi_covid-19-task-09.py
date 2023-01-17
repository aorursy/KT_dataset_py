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

import ast

papers_data['Affilations'] = papers_data['Affilations'].apply(lambda x: ast.literal_eval(x))

papers_data['Text_Lem_words'] = papers_data['Text_Lem_words'].apply(lambda x: ast.literal_eval(x))
from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet



wordnet_lemmatizer = WordNetLemmatizer()

def my_function(x):

    return list(dict.fromkeys(x))





keywords =['efforts','articulate', 'covid-2019','translate', 'ethical','principles','standards','salient','disease','covid-19','virus']

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

  

task9_1 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task9_1.append(i)

    

len(task9_1)
## Results for task 9.1 :

for i in task9_1:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 9.1:

print(task9_1)
task9_1_rank = papers_data.iloc[task9_1, :]

task9_1_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

meta = meta.rename(columns={"title": "Title"})

for i in range(len(meta)):

    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()

meta['Title'] = meta['Title'].astype(str) 

task9_1_rank['Title'] = task9_1_rank['Title'].astype(str) 



task9_1_rank = pd.merge(task9_1_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task9_1_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task9_1_rank['publish_time'] = task9_1_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task9_1_rank['publish_time'] = task9_1_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task9_1_rank['publish_time'] = pd.to_numeric(task9_1_rank['publish_time'])

task9_1_rank = task9_1_rank.sort_values(by='publish_time', ascending=False)

task9_1_rank.reset_index(inplace=True,drop=True)
rank = pd.read_csv("/kaggle/input/shanghai-ranking/rank-univ.csv")

del rank['Unnamed: 0']

rank.head(5)
# Extract the affiliations score to the task's results:

task9_1_rank['Aff_Score'] = 0

for i in range(len(task9_1_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task9_1_rank.iloc[i, 4]:

            task9_1_rank.iloc[i, 11] = rank.iloc[j, 3]
task9_1_rank["Ranking_Score"] = task9_1_rank["publish_time"]*0.8 + task9_1_rank["Aff_Score"]*0.2
task9_1_rank = task9_1_rank.sort_values(by='Ranking_Score', ascending=False)

task9_1_rank.reset_index(inplace=True,drop=True)

task9_1_rank
## 20 - Ranked Results for task 9.1 :



for i in range(len(task9_1_rank)):

    print("\n")

    print("PaperID: ", task9_1_rank.iloc[i, 0])

    print("Title: ", task9_1_rank.iloc[i, 1])

    print("Section: ", task9_1_rank.iloc[i, 2])

    print("Text: ", task9_1_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['efforts','embed', 'ethics', 'thematic','area','covid19','covid-19','novel','issues','minimize','duplication','oversight']

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

  

task9_2 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task9_2.append(i)

    

len(task9_2)
## Results for Task 9.2 :

for i in task9_2:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 9.2:

print(task9_2)
task9_2_rank = papers_data.iloc[task9_2, :]

task9_2_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task9_2_rank['Title'] = task9_2_rank['Title'].astype(str) 



task9_2_rank = pd.merge(task9_2_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task9_2_rank.dropna(inplace=True)



# Extract the year from the string publish time

task9_2_rank['publish_time'] = task9_2_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task9_2_rank['publish_time'] = pd.to_numeric(task9_2_rank['publish_time'])

task9_2_rank = task9_2_rank.sort_values(by='publish_time', ascending=False)

task9_2_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task9_2_rank['Aff_Score'] = 0

for i in range(len(task9_2_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task9_2_rank.iloc[i, 4]:

            task9_2_rank.iloc[i, 11] = rank.iloc[j, 3]
task9_2_rank["Ranking_Score"] = task9_2_rank["publish_time"]*0.8 + task9_2_rank["Aff_Score"]*0.2

task9_2_rank = task9_2_rank.sort_values(by='Ranking_Score', ascending=False)

task9_2_rank.reset_index(inplace=True,drop=True)

task9_2_rank
## 20 - Ranked Results for task 9.2 :



for i in range(len(task9_2_rank)):

    print("\n")

    print("PaperID: ", task9_2_rank.iloc[i, 0])

    print("Title: ", task9_2_rank.iloc[i, 1])

    print("Section: ", task9_2_rank.iloc[i, 2])

    print("Text: ", task9_2_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['efforts', 'support','sustained','education','access','capacity','area','ethics','covid19','covid-19','covid-2019','virus','disease']

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

    if len(a_set.intersection(b_set)) > 9: 

        return(True)  

    return(False)    

  

task9_3 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task9_3.append(i)

    

len(task9_3)
## Results for Task 9.3 :

for i in task9_3:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 9.3:

print(task9_3)
task9_3_rank = papers_data.iloc[task9_3, :]

task9_3_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

task9_3_rank['Title'] = task9_3_rank['Title'].astype(str) 



task9_3_rank = pd.merge(task9_3_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task9_3_rank.dropna(inplace=True)



# Extract the year from the string publish time

task9_3_rank['publish_time'] = task9_3_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task9_3_rank['publish_time'] = pd.to_numeric(task9_3_rank['publish_time'])

task9_3_rank = task9_3_rank.sort_values(by='publish_time', ascending=False)

task9_3_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task9_3_rank['Aff_Score'] = 0

for i in range(len(task9_3_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task9_3_rank.iloc[i, 4]:

            task9_3_rank.iloc[i, 11] = rank.iloc[j, 3]
task9_3_rank["Ranking_Score"] = task9_3_rank["publish_time"]*0.8 + task9_3_rank["Aff_Score"]*0.2

task9_3_rank = task9_3_rank.sort_values(by='Ranking_Score', ascending=False)

task9_3_rank.reset_index(inplace=True,drop=True)

task9_3_rank
## 20 - Ranked Results for task 9.3 :



for i in range(len(task9_3_rank)):

    print("\n")

    print("PaperID: ", task9_3_rank.iloc[i, 0])

    print("Title: ", task9_3_rank.iloc[i, 1])

    print("Section: ", task9_3_rank.iloc[i, 2])

    print("Text: ", task9_3_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['efforts', 'team','who','World','health','organization','multidisciplinary','research','covid-19','covid19','virus','patients','operational','platform','global','networks','social','sciences']

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

    if len(a_set.intersection(b_set)) > 12: 

        return(True)  

    return(False)    

  

task9_4 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task9_4.append(i)

    

len(task9_4)
## Results for Task 9.4 :

for i in task9_4:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 9.4:

print(task9_4)
task9_4_rank = papers_data.iloc[task9_4, :]

task9_4_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task9_4_rank['Title'] = task9_4_rank['Title'].astype(str) 



task9_4_rank = pd.merge(task9_4_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task9_4_rank.dropna(inplace=True)



# Extract the year from the string publish time

task9_4_rank['publish_time'] = task9_4_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))

task9_4_rank['publish_time'] = task9_4_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task9_4_rank['publish_time'] = pd.to_numeric(task9_4_rank['publish_time'])

task9_4_rank = task9_4_rank.sort_values(by='publish_time', ascending=False)

task9_4_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task9_4_rank['Aff_Score'] = 0

for i in range(len(task9_4_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task9_4_rank.iloc[i, 4]:

            task9_4_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task9_4_rank["Ranking_Score"] = task9_4_rank["publish_time"]*0.8 + task9_4_rank["Aff_Score"]*0.2

task9_4_rank = task9_4_rank.sort_values(by='Ranking_Score', ascending=False)

task9_4_rank.reset_index(inplace=True,drop=True)

task9_4_rank
## 20 - Ranked Results for task 9.4 :



for i in range(len(task9_4_rank)):

    print("\n")

    print("PaperID: ", task9_4_rank.iloc[i, 0])

    print("Title: ", task9_4_rank.iloc[i, 1])

    print("Section: ", task9_4_rank.iloc[i, 2])

    print("Text: ", task9_4_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['efforts', 'develop','qualitative','assessment','framework','systematically','collect','local','barriers','uptake','adherence','public','health','measures','prevention','control','identification','impact','surgical','mask','modification','seeking','behavior','srh','school','closures','covid19','covid-19','mechanical','ventilation','adjust','age']

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
# At least 17 words in common

def common_member(a, b): 

    a_set = set(a) 

    b_set = set(b) 

    if len(a_set.intersection(b_set)) > 16: 

        return(True)  

    return(False)    

  

task9_5 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task9_5.append(i)

    

len(task9_5)
## Results for Task 9.5 :

for i in task9_5:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 9.5 

print(task9_5)
task9_5_rank = papers_data.iloc[task9_5, :]

task9_5_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task9_5_rank['Title'] = task9_5_rank['Title'].astype(str) 



task9_5_rank = pd.merge(task9_5_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task9_5_rank.dropna(inplace=True)



# Extract the year from the string publish time

task9_5_rank['publish_time'] = task9_5_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task9_5_rank['publish_time'] = pd.to_numeric(task9_5_rank['publish_time'])

task9_5_rank = task9_5_rank.sort_values(by='publish_time', ascending=False)

task9_5_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task9_5_rank['Aff_Score'] = 0

for i in range(len(task9_5_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task9_5_rank.iloc[i, 4]:

            task9_5_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task9_5_rank["Ranking_Score"] = task9_5_rank["publish_time"]*0.8 + task9_5_rank["Aff_Score"]*0.2

task9_5_rank = task9_5_rank.sort_values(by='Ranking_Score', ascending=False)

task9_5_rank.reset_index(inplace=True,drop=True)

task9_5_rank
## 20 - Ranked Results for task 9.5 :



for i in range(len(task9_5_rank)):

    print("\n")

    print("PaperID: ", task9_5_rank.iloc[i, 0])

    print("Title: ", task9_5_rank.iloc[i, 1])

    print("Section: ", task9_5_rank.iloc[i, 2])

    print("Text: ", task9_5_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['efforts', 'burden','responding','outbreak','public','health','measures','affects','physical','psychological','covid19','covidd-19','virus','care','patient','needs']

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

  

task9_6 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task9_6.append(i)

    

len(task9_6)
## Results for Task 9.6 :

for i in task9_6:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 9.6:

print(task9_6)
task9_6_rank = papers_data.iloc[task9_6, :]

task9_6_rank.reset_index(inplace=True,drop=True)



task9_6_rank['Title'] = task9_6_rank['Title'].astype(str) 



task9_6_rank = pd.merge(task9_6_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task9_6_rank.dropna(inplace=True)



# Extract the year from the string publish time

task9_6_rank['publish_time'] = task9_6_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task9_6_rank['publish_time'] = pd.to_numeric(task9_6_rank['publish_time'])

task9_6_rank = task9_6_rank.sort_values(by='publish_time', ascending=False)

task9_6_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task9_6_rank['Aff_Score'] = 0

for i in range(len(task9_6_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task9_6_rank.iloc[i, 4]:

            task9_6_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task9_6_rank["Ranking_Score"] = task9_6_rank["publish_time"]*0.8 + task9_6_rank["Aff_Score"]*0.2

task9_6_rank = task9_6_rank.sort_values(by='Ranking_Score', ascending=False)

task9_6_rank.reset_index(inplace=True,drop=True)

task9_6_rank
## 20 - Ranked Results for task 9.6 :



for i in range(len(task9_6_rank)):

    print("\n")

    print("PaperID: ", task9_6_rank.iloc[i, 0])

    print("Title: ", task9_6_rank.iloc[i, 1])

    print("Section: ", task9_6_rank.iloc[i, 2])

    print("Text: ", task9_6_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['efforts', 'underlying','drivers','fear','anxiety','stigma','virus','fuel','misinformation','rumor','social','covid19','coronavirus','covid-19','media']

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

    if len(a_set.intersection(b_set)) > 6: 

        return(True)  

    return(False)    

  

task9_7 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task9_7.append(i)

    

len(task9_7)
## Results for Task 9.7 :

for i in task9_7:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task9_7_rank = papers_data.iloc[task9_7, :]

task9_7_rank.reset_index(inplace=True,drop=True)



task9_7_rank['Title'] = task9_7_rank['Title'].astype(str) 



task9_7_rank = pd.merge(task9_7_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task9_7_rank.dropna(inplace=True)



# Extract the year from the string publish time

task9_7_rank['publish_time'] = task9_7_rank['publish_time'].apply(lambda x: str(x).replace('May 8 Summer',''))

task9_7_rank['publish_time'] = task9_7_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task9_7_rank['publish_time'] = pd.to_numeric(task9_7_rank['publish_time'])

task9_7_rank = task9_7_rank.sort_values(by='publish_time', ascending=False)

task9_7_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task9_7_rank['Aff_Score'] = 0

for i in range(len(task9_7_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task9_7_rank.iloc[i, 4]:

            task9_7_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task9_7_rank["Ranking_Score"] = task9_7_rank["publish_time"]*0.8 + task9_7_rank["Aff_Score"]*0.2

task9_7_rank = task9_7_rank.sort_values(by='Ranking_Score', ascending=False)

task9_7_rank.reset_index(inplace=True,drop=True)

task9_7_rank
## 20 - Ranked Results for task 9.7 :



for i in range(len(task9_7_rank)):

    print("\n")

    print("PaperID: ", task9_7_rank.iloc[i, 0])

    print("Title: ", task9_7_rank.iloc[i, 1])

    print("Section: ", task9_7_rank.iloc[i, 2])

    print("Text: ", task9_7_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
task9_1_rank.to_csv("task9_1_rank.csv")

task9_2_rank.to_csv("task9_2_rank.csv")

task9_3_rank.to_csv("task9_3_rank.csv")

task9_4_rank.to_csv("task9_4_rank.csv")

task9_5_rank.to_csv("task9_5_rank.csv")

task9_6_rank.to_csv("task9_6_rank.csv")

task9_7_rank.to_csv("task9_7_rank.csv")