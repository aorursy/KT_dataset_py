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





keywords =['guidance','npis', 'funding', 'infrastructure','authorities','real-time','participants','collaboration','states','consensus','mobilize','resources','disease','geographic','area','critical','shortfalls','health','care','system','capacity','cases','covid19','covid-19']

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
# At least 17 of words in common

def common_member(a, b): 

    a_set = set(a) 

    b_set = set(b) 

    if len(a_set.intersection(b_set)) > 16: 

        return(True)  

    return(False)    

  

task7_1 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task7_1.append(i)

    

len(task7_1)
## Results for task 7.1 :

for i in task7_1:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 7.1:

print(task7_1)
task7_1_rank = papers_data.iloc[task7_1, :]

task7_1_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

meta = meta.rename(columns={"title": "Title"})

for i in range(len(meta)):

    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()

meta['Title'] = meta['Title'].astype(str) 

task7_1_rank['Title'] = task7_1_rank['Title'].astype(str) 



task7_1_rank = pd.merge(task7_1_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task7_1_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task7_1_rank['publish_time'] = task7_1_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))

task7_1_rank['publish_time'] = task7_1_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task7_1_rank['publish_time'] = pd.to_numeric(task7_1_rank['publish_time'])

task7_1_rank = task7_1_rank.sort_values(by='publish_time', ascending=False)

task7_1_rank.reset_index(inplace=True,drop=True)
rank = pd.read_csv("/kaggle/input/shanghai-ranking/rank-univ.csv")

del rank['Unnamed: 0']

rank.head(5)
# Extract the affiliations score to the task's results:

task7_1_rank['Aff_Score'] = 0

for i in range(len(task7_1_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task7_1_rank.iloc[i, 4]:

            task7_1_rank.iloc[i, 11] = rank.iloc[j, 3]
task7_1_rank["Ranking_Score"] = task7_1_rank["publish_time"]*0.8 + task7_1_rank["Aff_Score"]*0.2
task7_1_rank = task7_1_rank.sort_values(by='Ranking_Score', ascending=False)

task7_1_rank.reset_index(inplace=True,drop=True)

task7_1_rank
## 20 - Ranked Results for task 1.1 :



for i in range(len(task7_1_rank)):

    print("\n")

    print("PaperID: ", task7_1_rank.iloc[i, 0])

    print("Title: ", task7_1_rank.iloc[i, 1])

    print("Section: ", task7_1_rank.iloc[i, 2])

    print("Text: ", task7_1_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['rapid','design', 'execution', 'experiment','examine','npi','npis','compare','dhs','centers','covid19','covid-19']

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

  

task7_2 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task7_2.append(i)

    

len(task7_2)
## Results for task 7.2 :

for i in task7_2:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 7.2:

print(task7_2)
task7_2_rank = papers_data.iloc[task7_2, :]

task7_2_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task7_2_rank['Title'] = task7_2_rank['Title'].astype(str) 



task7_2_rank = pd.merge(task7_2_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task7_2_rank.dropna(inplace=True)



# Extract the year from the string publish time

task7_2_rank['publish_time'] = task7_2_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task7_2_rank['publish_time'] = pd.to_numeric(task7_2_rank['publish_time'])

task7_2_rank = task7_2_rank.sort_values(by='publish_time', ascending=False)

task7_2_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task7_2_rank['Aff_Score'] = 0

for i in range(len(task7_2_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task7_2_rank.iloc[i, 4]:

            task7_2_rank.iloc[i, 11] = rank.iloc[j, 3]
task7_2_rank["Ranking_Score"] = task7_2_rank["publish_time"]*0.8 + task7_2_rank["Aff_Score"]*0.2

task7_2_rank = task7_2_rank.sort_values(by='Ranking_Score', ascending=False)

task7_2_rank.reset_index(inplace=True,drop=True)

task7_2_rank
## 20 - Ranked Results for task 1.2 :



for i in range(len(task7_2_rank)):

    print("\n")

    print("PaperID: ", task7_2_rank.iloc[i, 0])

    print("Title: ", task7_2_rank.iloc[i, 1])

    print("Section: ", task7_2_rank.iloc[i, 2])

    print("Text: ", task7_2_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['rapid', 'assessment','efficacy','school','closure','travel','bans','sizes','social','distancing','approaches','covid19','covid-19','virus']

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

  

task7_3 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task7_3.append(i)

    

len(task7_3)
## Results for task 7.3 :

for i in task7_3:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 7.3:

print(task7_3)
task7_3_rank = papers_data.iloc[task7_3, :]

task7_3_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

task7_3_rank['Title'] = task7_3_rank['Title'].astype(str) 



task7_3_rank = pd.merge(task7_3_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task7_3_rank.dropna(inplace=True)



# Extract the year from the string publish time

task7_3_rank['publish_time'] = task7_3_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task7_3_rank['publish_time'] = pd.to_numeric(task7_3_rank['publish_time'])

task7_3_rank = task7_3_rank.sort_values(by='publish_time', ascending=False)

task7_3_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task7_3_rank['Aff_Score'] = 0

for i in range(len(task7_3_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task7_3_rank.iloc[i, 4]:

            task7_3_rank.iloc[i, 11] = rank.iloc[j, 3]
task7_3_rank["Ranking_Score"] = task7_3_rank["publish_time"]*0.8 + task7_3_rank["Aff_Score"]*0.2

task7_3_rank = task7_3_rank.sort_values(by='Ranking_Score', ascending=False)

task7_3_rank.reset_index(inplace=True,drop=True)

task7_3_rank
## 20 - Ranked Results for task 1.3 :



for i in range(len(task7_3_rank)):

    print("\n")

    print("PaperID: ", task7_3_rank.iloc[i, 0])

    print("Title: ", task7_3_rank.iloc[i, 1])

    print("Section: ", task7_3_rank.iloc[i, 2])

    print("Text: ", task7_3_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['method', 'control','spread','communities','barriers','compliance','population','vary','covid19','covid-19','virus']

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

  

task7_4 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task7_4.append(i)

    

len(task7_4)
## Results for task 7.4 :

for i in task7_4:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 7.4:

print(task7_4)
task7_4_rank = papers_data.iloc[task7_4, :]

task7_4_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task7_4_rank['Title'] = task7_4_rank['Title'].astype(str) 



task7_4_rank = pd.merge(task7_4_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task7_4_rank.dropna(inplace=True)



# Extract the year from the string publish time

task7_4_rank['publish_time'] = task7_4_rank['publish_time'].apply(lambda x:  str(x).replace('Oct 28 Mar-Apr',''))

task7_4_rank['publish_time'] = task7_4_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task7_4_rank['publish_time'] = pd.to_numeric(task7_4_rank['publish_time'])

task7_4_rank = task7_4_rank.sort_values(by='publish_time', ascending=False)

task7_4_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task7_4_rank['Aff_Score'] = 0

for i in range(len(task7_4_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task7_4_rank.iloc[i, 4]:

            task7_4_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task7_4_rank["Ranking_Score"] = task7_4_rank["publish_time"]*0.8 + task7_4_rank["Aff_Score"]*0.2

task7_4_rank = task7_4_rank.sort_values(by='Ranking_Score', ascending=False)

task7_4_rank.reset_index(inplace=True,drop=True)

task7_4_rank
## 20 - Ranked Results for task 1.4 :



for i in range(len(task7_4_rank)):

    print("\n")

    print("PaperID: ", task7_4_rank.iloc[i, 0])

    print("Title: ", task7_4_rank.iloc[i, 1])

    print("Section: ", task7_4_rank.iloc[i, 2])

    print("Text: ", task7_4_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['model', 'intervention','predict','cost','benefit','factor','race','geographic','immigration','location','status','housing','employment','health','insurance','covid19','covid-19','test','influenza','bed-side','recognizing','dischtradeoffs','speed','accessibility','accuracy']

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

    if len(a_set.intersection(b_set)) > 13: 

        return(True)  

    return(False)    

  

task7_5 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task7_5.append(i)

    

len(task7_5)
## Results for task 7.5 :

for i in task7_5:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 7.5 

print(task7_5)
task7_5_rank = papers_data.iloc[task7_5, :]

task7_5_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task7_5_rank['Title'] = task7_5_rank['Title'].astype(str) 



task7_5_rank = pd.merge(task7_5_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task7_5_rank.dropna(inplace=True)



# Extract the year from the string publish time

task7_5_rank['publish_time'] = task7_5_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))



task7_5_rank['publish_time'] = task7_5_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task7_5_rank['publish_time'] = pd.to_numeric(task7_5_rank['publish_time'])

task7_5_rank = task7_5_rank.sort_values(by='publish_time', ascending=False)

task7_5_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task7_5_rank['Aff_Score'] = 0

for i in range(len(task7_5_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task7_5_rank.iloc[i, 4]:

            task7_5_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task7_5_rank["Ranking_Score"] = task7_5_rank["publish_time"]*0.8 + task7_5_rank["Aff_Score"]*0.2

task7_5_rank = task7_5_rank.sort_values(by='Ranking_Score', ascending=False)

task7_5_rank.reset_index(inplace=True,drop=True)

task7_5_rank
## 20 - Ranked Results for task 1.5 :



for i in range(len(task7_5_rank)):

    print("\n")

    print("PaperID: ", task7_5_rank.iloc[i, 0])

    print("Title: ", task7_5_rank.iloc[i, 1])

    print("Section: ", task7_5_rank.iloc[i, 2])

    print("Text: ", task7_5_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['policy', 'necessary','compliance','individual','limited','resources','underserved','npis','npi','covid19','covidd-19','virus']

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

  

task7_6 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task7_6.append(i)

    

len(task7_6)
## Results for task 7.6 :

for i in task7_6:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 7.6:

print(task7_6)
task7_6_rank = papers_data.iloc[task7_6, :]

task7_6_rank.reset_index(inplace=True,drop=True)



task7_6_rank['Title'] = task7_6_rank['Title'].astype(str) 



task7_6_rank = pd.merge(task7_6_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task7_6_rank.dropna(inplace=True)



# Extract the year from the string publish time

task7_6_rank['publish_time'] = task7_6_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task7_6_rank['publish_time'] = pd.to_numeric(task7_6_rank['publish_time'])

task7_6_rank = task7_6_rank.sort_values(by='publish_time', ascending=False)

task7_6_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task7_6_rank['Aff_Score'] = 0

for i in range(len(task7_6_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task7_6_rank.iloc[i, 4]:

            task7_6_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task7_6_rank["Ranking_Score"] = task7_6_rank["publish_time"]*0.8 + task7_6_rank["Aff_Score"]*0.2

task7_6_rank = task7_6_rank.sort_values(by='Ranking_Score', ascending=False)

task7_6_rank.reset_index(inplace=True,drop=True)

task7_6_rank
## 20 - Ranked Results for task 1.6 :



for i in range(len(task7_6_rank)):

    print("\n")

    print("PaperID: ", task7_6_rank.iloc[i, 0])

    print("Title: ", task7_6_rank.iloc[i, 1])

    print("Section: ", task7_6_rank.iloc[i, 2])

    print("Text: ", task7_6_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['research', 'people','virus','fail','comply','public','health','advice','social','financial','cost','covid19','coronavirus','covid-19']

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

    if len(a_set.intersection(b_set)) > 7: 

        return(True)  

    return(False)    

  

task7_7 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task7_7.append(i)

    

len(task7_7)
## Results for task 7.7 :

for i in task7_7:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task7_7_rank = papers_data.iloc[task7_7, :]

task7_7_rank.reset_index(inplace=True,drop=True)



task7_7_rank['Title'] = task7_7_rank['Title'].astype(str) 



task7_7_rank = pd.merge(task7_7_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task7_7_rank.dropna(inplace=True)



# Extract the year from the string publish time 

task7_7_rank['publish_time'] = task7_7_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))



task7_7_rank['publish_time'] = task7_7_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task7_7_rank['publish_time'] = pd.to_numeric(task7_7_rank['publish_time'])

task7_7_rank = task7_7_rank.sort_values(by='publish_time', ascending=False)

task7_7_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task7_7_rank['Aff_Score'] = 0

for i in range(len(task7_7_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task7_7_rank.iloc[i, 4]:

            task7_7_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task7_7_rank["Ranking_Score"] = task7_7_rank["publish_time"]*0.8 + task7_7_rank["Aff_Score"]*0.2

task7_7_rank = task7_7_rank.sort_values(by='Ranking_Score', ascending=False)

task7_7_rank.reset_index(inplace=True,drop=True)

task7_7_rank
## 20 - Ranked Results for task 1.7 :



for i in range(len(task7_7_rank)):

    print("\n")

    print("PaperID: ", task7_7_rank.iloc[i, 0])

    print("Title: ", task7_7_rank.iloc[i, 1])

    print("Section: ", task7_7_rank.iloc[i, 2])

    print("Text: ", task7_7_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['research', 'economic','impact','pandemic','policy','programmatic','alternatives','lessen','mitigate','risk','critical','government','ervices','food','distribution','supplies','household','health','diagnoses','treatment','care','ability','pay','covid19','covid-19','coronavirus']

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

  

task7_8 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task7_8.append(i)

    

len(task7_8)
## Results for task 7.8 :

for i in task7_8:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task7_8_rank = papers_data.iloc[task7_8, :]

task7_8_rank.reset_index(inplace=True,drop=True)



task7_8_rank['Title'] = task7_8_rank['Title'].astype(str) 



task7_8_rank = pd.merge(task7_8_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task7_8_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task7_8_rank['publish_time'] = task7_8_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))

task7_8_rank['publish_time'] = task7_8_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task7_8_rank['publish_time'] = pd.to_numeric(task7_8_rank['publish_time'])

task7_8_rank = task7_8_rank.sort_values(by='publish_time', ascending=False)

task7_8_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task7_8_rank['Aff_Score'] = 0

for i in range(len(task7_8_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task7_8_rank.iloc[i, 4]:

            task7_8_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task7_8_rank["Ranking_Score"] = task7_8_rank["publish_time"]*0.8 + task7_8_rank["Aff_Score"]*0.2

task7_8_rank = task7_8_rank.sort_values(by='Ranking_Score', ascending=False)

task7_8_rank.reset_index(inplace=True,drop=True)

task7_8_rank
## 20 - Ranked Results for task 1.8 :



for i in range(len(task7_8_rank)):

    print("\n")

    print("PaperID: ", task7_8_rank.iloc[i, 0])

    print("Title: ", task7_8_rank.iloc[i, 1])

    print("Section: ", task7_8_rank.iloc[i, 2])

    print("Text: ", task7_8_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
task7_1_rank.to_csv("task7_1_rank.csv")

task7_2_rank.to_csv("task7_2_rank.csv")

task7_3_rank.to_csv("task7_3_rank.csv")

task7_4_rank.to_csv("task7_4_rank.csv")

task7_5_rank.to_csv("task7_5_rank.csv")

task7_6_rank.to_csv("task7_6_rank.csv")

task7_7_rank.to_csv("task7_7_rank.csv")

task7_8_rank.to_csv("task7_8_rank.csv")