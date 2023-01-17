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







keywords =['effectiveness','drugs', 'developed', 'treat','covid-19','covid19','patient', 'clinical', 'bench','trial','investigation','viral','inhibitor','naproxen','clarithromycin','minocycline','effect','replication']

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
# At least 13 of words in common

def common_member(a, b): 

    a_set = set(a) 

    b_set = set(b) 

    if len(a_set.intersection(b_set)) > 12: 

        return(True)  

    return(False)    

  

task4_1 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task4_1.append(i)

    

len(task4_1)
## Results for task 4.1 :

for i in task4_1:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 4.1:

print(task4_1)
task4_1_rank = papers_data.iloc[task4_1, :]

task4_1_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

meta = meta.rename(columns={"title": "Title"})

for i in range(len(meta)):

    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()

meta['Title'] = meta['Title'].astype(str) 

task4_1_rank['Title'] = task4_1_rank['Title'].astype(str) 



task4_1_rank = pd.merge(task4_1_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task4_1_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task4_1_rank['publish_time'] = task4_1_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task4_1_rank['publish_time'] = task4_1_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task4_1_rank['publish_time'] = pd.to_numeric(task4_1_rank['publish_time'])

task4_1_rank = task4_1_rank.sort_values(by='publish_time', ascending=False)

task4_1_rank.reset_index(inplace=True,drop=True)
rank = pd.read_csv("/kaggle/input/shanghai-ranking/rank-univ.csv")

del rank['Unnamed: 0']

rank.head(5)
# Extract the affiliations score to the task's results:

task4_1_rank['Aff_Score'] = 0

for i in range(len(task4_1_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task4_1_rank.iloc[i, 4]:

            task4_1_rank.iloc[i, 11] = rank.iloc[j, 3]
task4_1_rank["Ranking_Score"] = task4_1_rank["publish_time"]*0.8 + task4_1_rank["Aff_Score"]*0.2
task4_1_rank.head(5)
task4_1_rank = task4_1_rank.sort_values(by='Ranking_Score', ascending=False)

task4_1_rank.reset_index(inplace=True,drop=True)

task4_1_rank
## 20 - Ranked Results for task 4.1 :



for i in range(len(task4_1_rank)):

    print("\n")

    print("PaperID: ", task4_1_rank.iloc[i, 0])

    print("Title: ", task4_1_rank.iloc[i, 1])

    print("Section: ", task4_1_rank.iloc[i, 2])

    print("Text: ", task4_1_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['method','evaluating', 'complication', 'antibody','dependent','enhancement','ade','vaccine','recipient','therapeutics']

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

  

task4_2 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task4_2.append(i)

    

len(task4_2)
## Results for task 4.2 :

for i in task4_2:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 4.2:

print(task4_2)
task4_2_rank = papers_data.iloc[task4_2, :]

task4_2_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task4_2_rank['Title'] = task4_2_rank['Title'].astype(str) 



task4_2_rank = pd.merge(task4_2_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task4_2_rank.dropna(inplace=True)



# Extract the year from the string publish time

task4_2_rank['publish_time'] = task4_2_rank['publish_time'].apply(lambda x:  str(x).replace('Jan 13 May-Jun',''))

task4_2_rank['publish_time'] = task4_2_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task4_2_rank['publish_time'] = pd.to_numeric(task4_2_rank['publish_time'])

task4_2_rank = task4_2_rank.sort_values(by='publish_time', ascending=False)

task4_2_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task4_2_rank['Aff_Score'] = 0

for i in range(len(task4_2_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task4_2_rank.iloc[i, 4]:

            task4_2_rank.iloc[i, 11] = rank.iloc[j, 3]
task4_2_rank["Ranking_Score"] = task4_2_rank["publish_time"]*0.8 + task4_2_rank["Aff_Score"]*0.2

task4_2_rank = task4_2_rank.sort_values(by='Ranking_Score', ascending=False)

task4_2_rank.reset_index(inplace=True,drop=True)

task4_2_rank
## 20 - Ranked Results for task 4.2 :



for i in range(len(task4_2_rank)):

    print("\n")

    print("PaperID: ", task4_2_rank.iloc[i, 0])

    print("Title: ", task4_2_rank.iloc[i, 1])

    print("Section: ", task4_2_rank.iloc[i, 2])

    print("Text: ", task4_2_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['exploration', 'animal','model','predictive','human','vaccine']

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

  

task4_3 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task4_3.append(i)

    

len(task4_3)
## Results for task 4.3 :

for i in task4_3:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 4.3:

print(task4_3)
task4_3_rank = papers_data.iloc[task4_3, :]

task4_3_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task4_3_rank['Title'] = task4_3_rank['Title'].astype(str) 



task4_3_rank = pd.merge(task4_3_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task4_3_rank.dropna(inplace=True)



# Extract the year from the string publish time

#task4_3_rank['publish_time'] = task4_3_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task4_3_rank['publish_time'] = task4_3_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task4_3_rank['publish_time'] = pd.to_numeric(task4_3_rank['publish_time'])

task4_3_rank = task4_3_rank.sort_values(by='publish_time', ascending=False)

task4_3_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task4_3_rank['Aff_Score'] = 0

for i in range(len(task4_3_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task4_3_rank.iloc[i, 4]:

            task4_3_rank.iloc[i, 11] = rank.iloc[j, 3]
task4_3_rank["Ranking_Score"] = task4_3_rank["publish_time"]*0.8 + task4_3_rank["Aff_Score"]*0.2

task4_3_rank = task4_3_rank.sort_values(by='Ranking_Score', ascending=False)

task4_3_rank.reset_index(inplace=True,drop=True)

task4_3_rank
## 20 - Ranked Results for task 1.3 :



for i in range(len(task4_3_rank)):

    print("\n")

    print("PaperID: ", task4_3_rank.iloc[i, 0])

    print("Title: ", task4_3_rank.iloc[i, 1])

    print("Section: ", task4_3_rank.iloc[i, 2])

    print("Text: ", task4_3_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['capabilities', 'therapeutic','disease','covid-19','covid19','clinical','antiviral','agent','effectiveness']

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

  

task4_4 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task4_4.append(i)

    

len(task4_4)
## Results for task 4.4 :

for i in task4_4:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 4.4:

print(task4_4)
task4_4_rank = papers_data.iloc[task4_4, :]

task4_4_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task4_4_rank['Title'] = task4_4_rank['Title'].astype(str) 



task4_4_rank = pd.merge(task4_4_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task4_4_rank.dropna(inplace=True)



# Extract the year from the string publish time

#task4_4_rank['publish_time'] = task4_4_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task4_4_rank['publish_time'] = task4_4_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task4_4_rank['publish_time'] = pd.to_numeric(task4_4_rank['publish_time'])

task4_4_rank = task4_4_rank.sort_values(by='publish_time', ascending=False)

task4_4_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task4_4_rank['Aff_Score'] = 0

for i in range(len(task4_4_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task4_4_rank.iloc[i, 4]:

            task4_4_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task4_4_rank["Ranking_Score"] = task4_4_rank["publish_time"]*0.8 + task4_4_rank["Aff_Score"]*0.2

task4_4_rank = task4_4_rank.sort_values(by='Ranking_Score', ascending=False)

task4_4_rank.reset_index(inplace=True,drop=True)

task4_4_rank
## 20 - Ranked Results for task 4.4 :



for i in range(len(task4_4_rank)):

    print("\n")

    print("PaperID: ", task4_4_rank.iloc[i, 0])

    print("Title: ", task4_4_rank.iloc[i, 1])

    print("Section: ", task4_4_rank.iloc[i, 2])

    print("Text: ", task4_4_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['alternative', 'model','aid','prioritize','decision','scarce','therapeutics','ramps','population','need']

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

  

task4_5 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task4_5.append(i)

    

    

len(task4_5)
## Results for task 4.5 :

for i in task4_5:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 4.5 

print(task4_5)
task4_5_rank = papers_data.iloc[task4_5, :]

task4_5_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

task4_5_rank['Title'] = task4_5_rank['Title'].astype(str) 



task4_5_rank = pd.merge(task4_5_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task4_5_rank.dropna(inplace=True)



# Extract the year from the string publish time

task4_5_rank['publish_time'] = task4_5_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task4_5_rank['publish_time'] = task4_5_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task4_5_rank['publish_time'] = pd.to_numeric(task4_5_rank['publish_time'])

task4_5_rank = task4_5_rank.sort_values(by='publish_time', ascending=False)

task4_5_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task4_5_rank['Aff_Score'] = 0

for i in range(len(task4_5_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task4_5_rank.iloc[i, 4]:

            task4_5_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task4_5_rank["Ranking_Score"] = task4_5_rank["publish_time"]*0.8 + task4_5_rank["Aff_Score"]*0.2

task4_5_rank = task4_5_rank.sort_values(by='Ranking_Score', ascending=False)

task4_5_rank.reset_index(inplace=True,drop=True)

task4_5_rank
## 20 - Ranked Results for task 1.5 :



for i in range(len(task4_5_rank)):

    print("\n")

    print("PaperID: ", task4_5_rank.iloc[i, 0])

    print("Title: ", task4_5_rank.iloc[i, 1])

    print("Section: ", task4_5_rank.iloc[i, 2])

    print("Text: ", task4_5_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['efforts', 'target','universal','coronavirus','vaccine','covid19','covid-19']

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

  

task4_6 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task4_6.append(i)

    

len(task4_6)
## Results for task 4.6 :

for i in task4_6:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 4.6:

print(task4_6)
task4_6_rank = papers_data.iloc[task4_6, :]

task4_6_rank.reset_index(inplace=True,drop=True)



task4_6_rank['Title'] = task4_6_rank['Title'].astype(str) 



task4_6_rank = pd.merge(task4_6_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task4_6_rank.dropna(inplace=True)



# Extract the year from the string publish time

task4_6_rank['publish_time'] = task4_6_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task4_6_rank['publish_time'] = pd.to_numeric(task4_6_rank['publish_time'])

task4_6_rank = task4_6_rank.sort_values(by='publish_time', ascending=False)

task4_6_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task4_6_rank['Aff_Score'] = 0

for i in range(len(task4_6_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task4_6_rank.iloc[i, 4]:

            task4_6_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task4_6_rank["Ranking_Score"] = task4_6_rank["publish_time"]*0.8 + task4_6_rank["Aff_Score"]*0.2

task4_6_rank = task4_6_rank.sort_values(by='Ranking_Score', ascending=False)

task4_6_rank.reset_index(inplace=True,drop=True)

task4_6_rank
## 20 - Ranked Results for task 1.6 :



for i in range(len(task4_6_rank)):

    print("\n")

    print("PaperID: ", task4_6_rank.iloc[i, 0])

    print("Title: ", task4_6_rank.iloc[i, 1])

    print("Section: ", task4_6_rank.iloc[i, 2])

    print("Text: ", task4_6_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['effort', 'develop','animan','standardize','challenge','covid-19','covid19','vaccine']

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

    if len(a_set.intersection(b_set)) > 5: 

        return(True)  

    return(False)    

  

task4_7 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task4_7.append(i)

    

len(task4_7)
## Results for task 4.7 :

for i in task4_7:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task4_7_rank = papers_data.iloc[task4_7, :]

task4_7_rank.reset_index(inplace=True,drop=True)



task4_7_rank['Title'] = task4_7_rank['Title'].astype(str) 



task4_7_rank = pd.merge(task4_7_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task4_7_rank.dropna(inplace=True)



# Extract the year from the string publish time

task4_7_rank['publish_time'] = task4_7_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task4_7_rank['publish_time'] = pd.to_numeric(task4_7_rank['publish_time'])

task4_7_rank = task4_7_rank.sort_values(by='publish_time', ascending=False)

task4_7_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task4_7_rank['Aff_Score'] = 0

for i in range(len(task4_7_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task4_7_rank.iloc[i, 4]:

            task4_7_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task4_7_rank["Ranking_Score"] = task4_7_rank["publish_time"]*0.8 + task4_7_rank["Aff_Score"]*0.2

task4_7_rank = task4_7_rank.sort_values(by='Ranking_Score', ascending=False)

task4_7_rank.reset_index(inplace=True,drop=True)

task4_7_rank
## 20 - Ranked Results for task 1.7 :



for i in range(len(task4_7_rank)):

    print("\n")

    print("PaperID: ", task4_7_rank.iloc[i, 0])

    print("Title: ", task4_7_rank.iloc[i, 1])

    print("Section: ", task4_7_rank.iloc[i, 2])

    print("Text: ", task4_7_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['effort', 'develop','clinical','prophylaxis','prioritize','healthcare','worker']

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

  

task4_8 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task4_8.append(i)

    

len(task4_8)
## Results for task 4.8 :

for i in task4_8:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task4_8_rank = papers_data.iloc[task4_8, :]

task4_8_rank.reset_index(inplace=True,drop=True)



task4_8_rank['Title'] = task4_8_rank['Title'].astype(str) 



task4_8_rank = pd.merge(task4_8_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task4_8_rank.dropna(inplace=True)



# Extract the year from the string publish time

task4_8_rank['publish_time'] = task4_8_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task4_8_rank['publish_time'] = pd.to_numeric(task4_8_rank['publish_time'])

task4_8_rank = task4_8_rank.sort_values(by='publish_time', ascending=False)

task4_8_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task4_8_rank['Aff_Score'] = 0

for i in range(len(task4_8_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task4_8_rank.iloc[i, 4]:

            task4_8_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task4_8_rank["Ranking_Score"] = task4_8_rank["publish_time"]*0.8 + task4_8_rank["Aff_Score"]*0.2

task4_8_rank = task4_8_rank.sort_values(by='Ranking_Score', ascending=False)

task4_8_rank.reset_index(inplace=True,drop=True)

task4_8_rank
## 20 - Ranked Results for task 1.8 :



for i in range(len(task4_8_rank)):

    print("\n")

    print("PaperID: ", task4_8_rank.iloc[i, 0])

    print("Title: ", task4_8_rank.iloc[i, 1])

    print("Section: ", task4_8_rank.iloc[i, 2])

    print("Text: ", task4_8_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['approaches', 'evaluete','risk','enhanced','disease', 'vaccination','vaccine','covid19','covid-19']

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

  

task4_9 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task4_9.append(i)

    

len(task4_9)
## Results for task 4.9 :

for i in task4_9:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task4_9_rank = papers_data.iloc[task4_9, :]

task4_9_rank.reset_index(inplace=True,drop=True)



task4_9_rank['Title'] = task4_9_rank['Title'].astype(str) 



task4_9_rank = pd.merge(task4_9_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task4_9_rank.dropna(inplace=True)



# Extract the year from the string publish time

task4_9_rank['publish_time'] = task4_9_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task4_9_rank['publish_time'] = pd.to_numeric(task4_9_rank['publish_time'])

task4_9_rank = task4_9_rank.sort_values(by='publish_time', ascending=False)

task4_9_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task4_9_rank['Aff_Score'] = 0

for i in range(len(task4_9_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task4_9_rank.iloc[i, 4]:

            task4_9_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task4_9_rank["Ranking_Score"] = task4_9_rank["publish_time"]*0.8 + task4_9_rank["Aff_Score"]*0.2

task4_9_rank = task4_9_rank.sort_values(by='Ranking_Score', ascending=False)

task4_9_rank.reset_index(inplace=True,drop=True)

task4_9_rank
## 20 - Ranked Results for task 4.9 :



for i in range(len(task4_9_rank)):

    print("\n")

    print("PaperID: ", task4_9_rank.iloc[i, 0])

    print("Title: ", task4_9_rank.iloc[i, 1])

    print("Section: ", task4_9_rank.iloc[i, 2])

    print("Text: ", task4_9_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['assays', 'evaluate','vaccine','immune','response','development','animal','model','conjunction','therapeutics','covid19','covid-19']

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

  

task4_10 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task4_10.append(i)

    

len(task4_10)
## Results for task 4.10 :

for i in task4_10:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task4_10_rank = papers_data.iloc[task4_10, :]

task4_10_rank.reset_index(inplace=True,drop=True)



task4_10_rank['Title'] = task4_10_rank['Title'].astype(str) 



task4_10_rank = pd.merge(task4_10_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task4_10_rank.dropna(inplace=True)



# Extract the year from the string publish time

task4_10_rank['publish_time'] = task4_10_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task4_10_rank['publish_time'] = pd.to_numeric(task4_10_rank['publish_time'])

task4_10_rank = task4_10_rank.sort_values(by='publish_time', ascending=False)

task4_10_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task4_10_rank['Aff_Score'] = 0

for i in range(len(task4_10_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task4_10_rank.iloc[i, 4]:

            task4_10_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task4_10_rank["Ranking_Score"] = task4_10_rank["publish_time"]*0.8 + task4_10_rank["Aff_Score"]*0.2

task4_10_rank = task4_10_rank.sort_values(by='Ranking_Score', ascending=False)

task4_10_rank.reset_index(inplace=True,drop=True)

task4_10_rank
## 20 - Ranked Results for task 1.10 :



for i in range(len(task4_10_rank)):

    print("\n")

    print("PaperID: ", task4_10_rank.iloc[i, 0])

    print("Title: ", task4_10_rank.iloc[i, 1])

    print("Section: ", task4_10_rank.iloc[i, 2])

    print("Text: ", task4_10_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
task4_1_rank.to_csv("task4_1_rank.csv")

task4_2_rank.to_csv("task4_2_rank.csv")

task4_3_rank.to_csv("task4_3_rank.csv")

task4_4_rank.to_csv("task4_4_rank.csv")

task4_5_rank.to_csv("task4_5_rank.csv")

task4_6_rank.to_csv("task4_6_rank.csv")

task4_7_rank.to_csv("task4_7_rank.csv")

task4_8_rank.to_csv("task4_8_rank.csv")

task4_9_rank.to_csv("task4_9_rank.csv")

task4_10_rank.to_csv("task4_10_rank.csv")