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





keywords =['methods','data-gathering', 'gathering', 'data','standardized','nomenclature','disease','covid19','covid-19','virus']

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

  

task10_1 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_1.append(i)

    

len(task10_1)
## Results for Task 10.1 :

for i in task10_1:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 10.1:

print(task10_1)
task10_1_rank = papers_data.iloc[task10_1, :]

task10_1_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

meta = meta.rename(columns={"title": "Title"})

for i in range(len(meta)):

    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()

meta['Title'] = meta['Title'].astype(str) 

task10_1_rank['Title'] = task10_1_rank['Title'].astype(str) 



task10_1_rank = pd.merge(task10_1_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_1_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task10_1_rank['publish_time'] = task10_1_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task10_1_rank['publish_time'] = task10_1_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_1_rank['publish_time'] = pd.to_numeric(task10_1_rank['publish_time'])

task10_1_rank = task10_1_rank.sort_values(by='publish_time', ascending=False)

task10_1_rank.reset_index(inplace=True,drop=True)
rank = pd.read_csv("/kaggle/input/shanghai-ranking/rank-univ.csv")

del rank['Unnamed: 0']

rank.head(5)
# Extract the affiliations score to the task's results:

task10_1_rank['Aff_Score'] = 0

for i in range(len(task10_1_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_1_rank.iloc[i, 4]:

            task10_1_rank.iloc[i, 11] = rank.iloc[j, 3]
task10_1_rank["Ranking_Score"] = task10_1_rank["publish_time"]*0.8 + task10_1_rank["Aff_Score"]*0.2
task10_1_rank = task10_1_rank.sort_values(by='Ranking_Score', ascending=False)

task10_1_rank.reset_index(inplace=True,drop=True)

task10_1_rank
## 20 - Ranked Results for task 10.1 :



for i in range(len(task10_1_rank)):

    print("\n")

    print("PaperID: ", task10_1_rank.iloc[i, 0])

    print("Title: ", task10_1_rank.iloc[i, 1])

    print("Section: ", task10_1_rank.iloc[i, 2])

    print("Text: ", task10_1_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['sharing','response', 'information', 'planners','providers','virus','covid19','covid-19']

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

  

task10_2 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_2.append(i)

    

len(task10_2)
## Results for Task 10.2 :

for i in task10_2:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 10.2:

print(task10_2)
task10_2_rank = papers_data.iloc[task10_2, :]

task10_2_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task10_2_rank['Title'] = task10_2_rank['Title'].astype(str) 



task10_2_rank = pd.merge(task10_2_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_2_rank.dropna(inplace=True)



# Extract the year from the string publish time

task10_2_rank['publish_time'] = task10_2_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_2_rank['publish_time'] = pd.to_numeric(task10_2_rank['publish_time'])

task10_2_rank = task10_2_rank.sort_values(by='publish_time', ascending=False)

task10_2_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_2_rank['Aff_Score'] = 0

for i in range(len(task10_2_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_2_rank.iloc[i, 4]:

            task10_2_rank.iloc[i, 11] = rank.iloc[j, 3]
task10_2_rank["Ranking_Score"] = task10_2_rank["publish_time"]*0.8 + task10_2_rank["Aff_Score"]*0.2

task10_2_rank = task10_2_rank.sort_values(by='Ranking_Score', ascending=False)

task10_2_rank.reset_index(inplace=True,drop=True)

task10_2_rank
## 20 - Ranked Results for task 10.2 :



for i in range(len(task10_2_rank)):

    print("\n")

    print("PaperID: ", task10_2_rank.iloc[i, 0])

    print("Title: ", task10_2_rank.iloc[i, 1])

    print("Section: ", task10_2_rank.iloc[i, 2])

    print("Text: ", task10_2_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['understanding', 'mitigating','barriers','information','sharing',' information-sharing','covid19','covid-19','virus']

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

  

task10_3 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_3.append(i)

    

len(task10_3)
## Results for Task 10.3 :

for i in task10_3:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 10.3:

print(task10_3)
task10_3_rank = papers_data.iloc[task10_3, :]

task10_3_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

task10_3_rank['Title'] = task10_3_rank['Title'].astype(str) 



task10_3_rank = pd.merge(task10_3_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_3_rank.dropna(inplace=True)



# Extract the year from the string publish time

task10_3_rank['publish_time'] = task10_3_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_3_rank['publish_time'] = pd.to_numeric(task10_3_rank['publish_time'])

task10_3_rank = task10_3_rank.sort_values(by='publish_time', ascending=False)

task10_3_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_3_rank['Aff_Score'] = 0

for i in range(len(task10_3_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_3_rank.iloc[i, 4]:

            task10_3_rank.iloc[i, 11] = rank.iloc[j, 3]
task10_3_rank["Ranking_Score"] = task10_3_rank["publish_time"]*0.8 + task10_3_rank["Aff_Score"]*0.2

task10_3_rank = task10_3_rank.sort_values(by='Ranking_Score', ascending=False)

task10_3_rank.reset_index(inplace=True,drop=True)

task10_3_rank
## 20 - Ranked Results for task 10.3 :



for i in range(len(task10_3_rank)):

    print("\n")

    print("PaperID: ", task10_3_rank.iloc[i, 0])

    print("Title: ", task10_3_rank.iloc[i, 1])

    print("Section: ", task10_3_rank.iloc[i, 2])

    print("Text: ", task10_3_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['recruit', 'support','coordinate','local','non-federal','expertise','capacity','relevant','public','health','emergency','private','covid19','covid-19','virus','commercial','non-profit','academic']

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

  

task10_4 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_4.append(i)

    

len(task10_4)
## Results for Task 10.4 :

for i in task10_4:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 10.4:

print(task10_4)
task10_4_rank = papers_data.iloc[task10_4, :]

task10_4_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task10_4_rank['Title'] = task10_4_rank['Title'].astype(str) 



task10_4_rank = pd.merge(task10_4_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_4_rank.dropna(inplace=True)



# Extract the year from the string publish time

task10_4_rank['publish_time'] = task10_4_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))

task10_4_rank['publish_time'] = task10_4_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_4_rank['publish_time'] = pd.to_numeric(task10_4_rank['publish_time'])

task10_4_rank = task10_4_rank.sort_values(by='publish_time', ascending=False)

task10_4_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_4_rank['Aff_Score'] = 0

for i in range(len(task10_4_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_4_rank.iloc[i, 4]:

            task10_4_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task10_4_rank["Ranking_Score"] = task10_4_rank["publish_time"]*0.8 + task10_4_rank["Aff_Score"]*0.2

task10_4_rank = task10_4_rank.sort_values(by='Ranking_Score', ascending=False)

task10_4_rank.reset_index(inplace=True,drop=True)

task10_4_rank
## 20 - Ranked Results for task 10.4 :



for i in range(len(task10_4_rank)):

    print("\n")

    print("PaperID: ", task10_4_rank.iloc[i, 0])

    print("Title: ", task10_4_rank.iloc[i, 1])

    print("Section: ", task10_4_rank.iloc[i, 2])

    print("Text: ", task10_4_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['integration', 'federal','state','local','public','health','surveillance','systems','covid19','covid-19']

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

    if len(a_set.intersection(b_set)) > 8: 

        return(True)  

    return(False)    

  

task10_5 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_5.append(i)

    

len(task10_5)
## Results for Task 10.5 :

for i in task10_5:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 10.5 

print(task10_5)
task10_5_rank = papers_data.iloc[task10_5, :]

task10_5_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task10_5_rank['Title'] = task10_5_rank['Title'].astype(str) 



task10_5_rank = pd.merge(task10_5_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_5_rank.dropna(inplace=True)



# Extract the year from the string publish time 

task10_5_rank['publish_time'] = task10_5_rank['publish_time'].apply(lambda x: str(x).replace('May 8 Summer',''))

task10_5_rank['publish_time'] = task10_5_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_5_rank['publish_time'] = pd.to_numeric(task10_5_rank['publish_time'])

task10_5_rank = task10_5_rank.sort_values(by='publish_time', ascending=False)

task10_5_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_5_rank['Aff_Score'] = 0

for i in range(len(task10_5_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_5_rank.iloc[i, 4]:

            task10_5_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task10_5_rank["Ranking_Score"] = task10_5_rank["publish_time"]*0.8 + task10_5_rank["Aff_Score"]*0.2

task10_5_rank = task10_5_rank.sort_values(by='Ranking_Score', ascending=False)

task10_5_rank.reset_index(inplace=True,drop=True)

task10_5_rank
## 20 - Ranked Results for task 10.5 :



for i in range(len(task10_5_rank)):

    print("\n")

    print("PaperID: ", task10_5_rank.iloc[i, 0])

    print("Title: ", task10_5_rank.iloc[i, 1])

    print("Section: ", task10_5_rank.iloc[i, 2])

    print("Text: ", task10_5_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['value', 'investment','baseline','public','health','response','infrastructure','preparedness','covid19','covidd-19','virus']

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

  

task10_6 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_6.append(i)

    

len(task10_6)
## Results for Task 10.6 :

for i in task10_6:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 10.6:

print(task10_6)
task10_6_rank = papers_data.iloc[task10_6, :]

task10_6_rank.reset_index(inplace=True,drop=True)



task10_6_rank['Title'] = task10_6_rank['Title'].astype(str) 



task10_6_rank = pd.merge(task10_6_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_6_rank.dropna(inplace=True)



# Extract the year from the string publish time May 8 Summer

task10_6_rank['publish_time'] = task10_6_rank['publish_time'].apply(lambda x: str(x).replace('May 8 Summer',''))

task10_6_rank['publish_time'] = task10_6_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_6_rank['publish_time'] = pd.to_numeric(task10_6_rank['publish_time'])

task10_6_rank = task10_6_rank.sort_values(by='publish_time', ascending=False)

task10_6_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_6_rank['Aff_Score'] = 0

for i in range(len(task10_6_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_6_rank.iloc[i, 4]:

            task10_6_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task10_6_rank["Ranking_Score"] = task10_6_rank["publish_time"]*0.8 + task10_6_rank["Aff_Score"]*0.2

task10_6_rank = task10_6_rank.sort_values(by='Ranking_Score', ascending=False)

task10_6_rank.reset_index(inplace=True,drop=True)

task10_6_rank
## 20 - Ranked Results for task 10.6 :



for i in range(len(task10_6_rank)):

    print("\n")

    print("PaperID: ", task10_6_rank.iloc[i, 0])

    print("Title: ", task10_6_rank.iloc[i, 1])

    print("Section: ", task10_6_rank.iloc[i, 2])

    print("Text: ", task10_6_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['mode', 'communication','virus','target','high-risk','risk','population','elderly','health','care','workers','covid-19','coronavirus','covid19']

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

  

task10_7 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_7.append(i)

    

len(task10_7)
## Results for Task 10.7 :

for i in task10_7:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task10_7_rank = papers_data.iloc[task10_7, :]

task10_7_rank.reset_index(inplace=True,drop=True)



task10_7_rank['Title'] = task10_7_rank['Title'].astype(str) 



task10_7_rank = pd.merge(task10_7_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_7_rank.dropna(inplace=True)



# Extract the year from the string publish time

task10_7_rank['publish_time'] = task10_7_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_7_rank['publish_time'] = pd.to_numeric(task10_7_rank['publish_time'])

task10_7_rank = task10_7_rank.sort_values(by='publish_time', ascending=False)

task10_7_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_7_rank['Aff_Score'] = 0

for i in range(len(task10_7_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_7_rank.iloc[i, 4]:

            task10_7_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task10_7_rank["Ranking_Score"] = task10_7_rank["publish_time"]*0.8 + task10_7_rank["Aff_Score"]*0.2

task10_7_rank = task10_7_rank.sort_values(by='Ranking_Score', ascending=False)

task10_7_rank.reset_index(inplace=True,drop=True)

task10_7_rank
## 20 - Ranked Results for task 10.7 :



for i in range(len(task10_7_rank)):

    print("\n")

    print("PaperID: ", task10_7_rank.iloc[i, 0])

    print("Title: ", task10_7_rank.iloc[i, 1])

    print("Section: ", task10_7_rank.iloc[i, 2])

    print("Text: ", task10_7_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['risk', 'communication','guidlines','virus','understand','follow','targeting','population','family','covid19','covid-19','coronavirus']

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

    if len(a_set.intersection(b_set)) > 7: 

        return(True)  

    return(False)    

  



task10_8 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_8.append(i)

    

len(task10_8)
## Results for Task 10.8 :

for i in task10_8:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task10_8_rank = papers_data.iloc[task10_8, :]

task10_8_rank.reset_index(inplace=True,drop=True)



task10_8_rank['Title'] = task10_8_rank['Title'].astype(str) 



task10_8_rank = pd.merge(task10_8_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_8_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task10_8_rank['publish_time'] = task10_8_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task10_8_rank['publish_time'] = task10_8_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_8_rank['publish_time'] = pd.to_numeric(task10_8_rank['publish_time'])

task10_8_rank = task10_8_rank.sort_values(by='publish_time', ascending=False)

task10_8_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_8_rank['Aff_Score'] = 0

for i in range(len(task10_8_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_8_rank.iloc[i, 4]:

            task10_8_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task10_8_rank["Ranking_Score"] = task10_8_rank["publish_time"]*0.8 + task10_8_rank["Aff_Score"]*0.2

task10_8_rank = task10_8_rank.sort_values(by='Ranking_Score', ascending=False)

task10_8_rank.reset_index(inplace=True,drop=True)

task10_8_rank
## 20 - Ranked Results for task 10.8 :



for i in range(len(task10_8_rank)):

    print("\n")

    print("PaperID: ", task10_8_rank.iloc[i, 0])

    print("Title: ", task10_8_rank.iloc[i, 1])

    print("Section: ", task10_8_rank.iloc[i, 2])

    print("Text: ", task10_8_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['communication', 'indicates','risk','disease','population', 'groups','covid-19','covid19','virus','coronavirus','corona']

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

  

task10_9 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_9.append(i)

    

len(task10_9)
## Results for Task 10.9 :

for i in task10_9:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task10_9_rank = papers_data.iloc[task10_9, :]

task10_9_rank.reset_index(inplace=True,drop=True)



task10_9_rank['Title'] = task10_9_rank['Title'].astype(str) 



task10_9_rank = pd.merge(task10_9_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_9_rank.dropna(inplace=True)



# Extract the year from the string publish time

task10_9_rank['publish_time'] = task10_9_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_9_rank['publish_time'] = pd.to_numeric(task10_9_rank['publish_time'])

task10_9_rank = task10_9_rank.sort_values(by='publish_time', ascending=False)

task10_9_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_9_rank['Aff_Score'] = 0

for i in range(len(task10_9_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_9_rank.iloc[i, 4]:

            task10_9_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task10_9_rank["Ranking_Score"] = task10_9_rank["publish_time"]*0.8 + task10_9_rank["Aff_Score"]*0.2

task10_9_rank = task10_9_rank.sort_values(by='Ranking_Score', ascending=False)

task10_9_rank.reset_index(inplace=True,drop=True)

task10_9_rank
## 20 - Ranked Results for task 10.9 :



for i in range(len(task10_9_rank)):

    print("\n")

    print("PaperID: ", task10_9_rank.iloc[i, 0])

    print("Title: ", task10_9_rank.iloc[i, 1])

    print("Section: ", task10_9_rank.iloc[i, 2])

    print("Text: ", task10_9_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['misunderstanding', 'containment','mitigation','virus','covid-19','covid19','desease']

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

    if len(a_set.intersection(b_set)) > 2: 

        return(True)  

    return(False)    

  

task10_10 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_10.append(i)

    

len(task10_10)
## Results for Task 10.10 :

for i in task10_10:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task10_10_rank = papers_data.iloc[task10_10, :]

task10_10_rank.reset_index(inplace=True,drop=True)



task10_10_rank['Title'] = task10_10_rank['Title'].astype(str) 



task10_10_rank = pd.merge(task10_10_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_10_rank.dropna(inplace=True)



# Extract the year from the string publish time

task10_10_rank['publish_time'] = task10_10_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_10_rank['publish_time'] = pd.to_numeric(task10_10_rank['publish_time'])

task10_10_rank = task10_10_rank.sort_values(by='publish_time', ascending=False)

task10_10_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_10_rank['Aff_Score'] = 0

for i in range(len(task10_10_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_10_rank.iloc[i, 4]:

            task10_10_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task10_10_rank["Ranking_Score"] = task10_10_rank["publish_time"]*0.8 + task10_10_rank["Aff_Score"]*0.2

task10_10_rank = task10_10_rank.sort_values(by='Ranking_Score', ascending=False)

task10_10_rank.reset_index(inplace=True,drop=True)

task10_10_rank
## 20 - Ranked Results for task 10.10 :



for i in range(len(task10_10_rank)):

    print("\n")

    print("PaperID: ", task10_10_rank.iloc[i, 0])

    print("Title: ", task10_10_rank.iloc[i, 1])

    print("Section: ", task10_10_rank.iloc[i, 2])

    print("Text: ", task10_10_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['plan','mitigation','gaps','corona','problem','inequity','covid19','covid-19','nation','public','health','capability','capacity','funding','citizens','needs','support','information','access','treatment','surveillance']

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
# At least 14 words in common

def common_member(a, b): 

    a_set = set(a) 

    b_set = set(b) 

    if len(a_set.intersection(b_set)) > 14: 

        return(True)  

    return(False)    

  

task10_11 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_11.append(i)

    

len(task10_11)
## Results for Task 10.11 :

for i in task10_11:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task10_11_rank = papers_data.iloc[task10_11, :]

task10_11_rank.reset_index(inplace=True,drop=True)



task10_11_rank['Title'] = task10_11_rank['Title'].astype(str) 



task10_11_rank = pd.merge(task10_11_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_11_rank.dropna(inplace=True)



# Extract the year from the string publish time

task10_11_rank['publish_time'] = task10_11_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_11_rank['publish_time'] = pd.to_numeric(task10_11_rank['publish_time'])

task10_11_rank = task10_11_rank.sort_values(by='publish_time', ascending=False)

task10_11_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_11_rank['Aff_Score'] = 0

for i in range(len(task10_11_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_11_rank.iloc[i, 4]:

            task10_11_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task10_11_rank["Ranking_Score"] = task10_11_rank["publish_time"]*0.8 + task10_11_rank["Aff_Score"]*0.2

task10_11_rank = task10_11_rank.sort_values(by='Ranking_Score', ascending=False)

task10_11_rank.reset_index(inplace=True,drop=True)

task10_11_rank
## 20 - Ranked Results for task 10.11 :



for i in range(len(task10_11_rank)):

    print("\n")

    print("PaperID: ", task10_11_rank.iloc[i, 0])

    print("Title: ", task10_11_rank.iloc[i, 1])

    print("Section: ", task10_11_rank.iloc[i, 2])

    print("Text: ", task10_11_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['measures', 'marginalized','disadvantaged','populations','data','system','priorities','research','covid19','virus','covid-19','agenda','needs','circumstances','underrepresented','minorities']

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

  

task10_12 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_12.append(i)

    

len(task10_12)
## Results for Task 10.12 :

for i in task10_12:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task10_12_rank = papers_data.iloc[task10_12, :]

task10_12_rank.reset_index(inplace=True,drop=True)



task10_12_rank['Title'] = task10_12_rank['Title'].astype(str) 



task10_12_rank = pd.merge(task10_12_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_12_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task10_12_rank['publish_time'] = task10_12_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))

task10_12_rank['publish_time'] = task10_12_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_12_rank['publish_time'] = pd.to_numeric(task10_12_rank['publish_time'])

task10_12_rank = task10_12_rank.sort_values(by='publish_time', ascending=False)

task10_12_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_12_rank['Aff_Score'] = 0

for i in range(len(task10_12_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_12_rank.iloc[i, 4]:

            task10_12_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task10_12_rank["Ranking_Score"] = task10_12_rank["publish_time"]*0.8 + task10_12_rank["Aff_Score"]*0.2

task10_12_rank = task10_12_rank.sort_values(by='Ranking_Score', ascending=False)

task10_12_rank.reset_index(inplace=True,drop=True)

task10_12_rank
## 20 - Ranked Results for task 10.12 :



for i in range(len(task10_12_rank)):

    print("\n")

    print("PaperID: ", task10_12_rank.iloc[i, 0])

    print("Title: ", task10_12_rank.iloc[i, 1])

    print("Section: ", task10_12_rank.iloc[i, 2])

    print("Text: ", task10_12_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['mitigating', 'threats','diagnosis','covid19','virus','covid-19','incarcerated','people','information','prevention','treatment']

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

    if len(a_set.intersection(b_set)) > 6: 

        return(True)  

    return(False)    

  

task10_13 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_13.append(i)

    

len(task10_13)
## Results for Task 10.13 :

for i in task10_13:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task10_13_rank = papers_data.iloc[task10_13, :]

task10_13_rank.reset_index(inplace=True,drop=True)



task10_13_rank['Title'] = task10_13_rank['Title'].astype(str) 



task10_13_rank = pd.merge(task10_13_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_13_rank.dropna(inplace=True)



# Extract the year from the string publish time

task10_13_rank['publish_time'] = task10_13_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_13_rank['publish_time'] = pd.to_numeric(task10_13_rank['publish_time'])

task10_13_rank = task10_13_rank.sort_values(by='publish_time', ascending=False)

task10_13_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_13_rank['Aff_Score'] = 0

for i in range(len(task10_13_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_13_rank.iloc[i, 4]:

            task10_13_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task10_13_rank["Ranking_Score"] = task10_13_rank["publish_time"]*0.8 + task10_13_rank["Aff_Score"]*0.2

task10_13_rank = task10_13_rank.sort_values(by='Ranking_Score', ascending=False)

task10_13_rank.reset_index(inplace=True,drop=True)

task10_13_rank
## 20 - Ranked Results for task 10.13 :



for i in range(len(task10_13_rank)):

    print("\n")

    print("PaperID: ", task10_13_rank.iloc[i, 0])

    print("Title: ", task10_13_rank.iloc[i, 1])

    print("Section: ", task10_13_rank.iloc[i, 2])

    print("Text: ", task10_13_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['understanding','coverage','policies','barriers','opportunities','testing','treatment','epidemic','care','covid19','covid-19','virus']

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

  

task10_14 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task10_14.append(i)

    

len(task10_14)
## Results for Task 10.14 :

for i in task10_14:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")

    
task10_14_rank = papers_data.iloc[task10_14, :]

task10_14_rank.reset_index(inplace=True,drop=True)



task10_14_rank['Title'] = task10_14_rank['Title'].astype(str) 



task10_14_rank = pd.merge(task10_14_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task10_14_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task10_14_rank['publish_time'] = task10_14_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))

task10_14_rank['publish_time'] = task10_14_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task10_14_rank['publish_time'] = pd.to_numeric(task10_14_rank['publish_time'])

task10_14_rank = task10_14_rank.sort_values(by='publish_time', ascending=False)

task10_14_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task10_14_rank['Aff_Score'] = 0

for i in range(len(task10_14_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task10_14_rank.iloc[i, 4]:

            task10_14_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task10_14_rank["Ranking_Score"] = task10_14_rank["publish_time"]*0.8 + task10_14_rank["Aff_Score"]*0.2

task10_14_rank = task10_14_rank.sort_values(by='Ranking_Score', ascending=False)

task10_14_rank.reset_index(inplace=True,drop=True)

task10_14_rank
## 20 - Ranked Results for Task 10.14 :



for i in range(len(task10_14_rank)):

    print("\n")

    print("PaperID: ", task10_14_rank.iloc[i, 0])

    print("Title: ", task10_14_rank.iloc[i, 1])

    print("Section: ", task10_14_rank.iloc[i, 2])

    print("Text: ", task10_14_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
task10_1_rank.to_csv("task10_1_rank.csv")

task10_2_rank.to_csv("task10_2_rank.csv")

task10_3_rank.to_csv("task10_3_rank.csv")

task10_4_rank.to_csv("task10_4_rank.csv")

task10_5_rank.to_csv("task10_5_rank.csv")

task10_6_rank.to_csv("task10_6_rank.csv")

task10_7_rank.to_csv("task10_7_rank.csv")

task10_8_rank.to_csv("task10_8_rank.csv")

task10_9_rank.to_csv("task10_9_rank.csv")

task10_10_rank.to_csv("task10_10_rank.csv")

task10_11_rank.to_csv("task10_11_rank.csv")

task10_12_rank.to_csv("task10_12_rank.csv")

task10_13_rank.to_csv("task10_13_rank.csv")

task10_14_rank.to_csv("task10_14_rank.csv")