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





keywords =['resources','support', 'skill', 'nursing','facilities','long-term','care','disease','covid-19','virus']

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
# At least 9 of words in common

def common_member(a, b): 

    a_set = set(a) 

    b_set = set(b) 

    if len(a_set.intersection(b_set)) > 8: 

        return(True)  

    return(False)    

  

task8_1 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_1.append(i)

    

len(task8_1)
## Results for task 8.1 :

for i in task8_1:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 8.1:

print(task8_1)
task8_1_rank = papers_data.iloc[task8_1, :]

task8_1_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

meta = meta.rename(columns={"title": "Title"})

for i in range(len(meta)):

    meta.iloc[i, 2] = str(meta.iloc[i, 2]).lower()

meta['Title'] = meta['Title'].astype(str) 

task8_1_rank['Title'] = task8_1_rank['Title'].astype(str) 



task8_1_rank = pd.merge(task8_1_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_1_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task8_1_rank['publish_time'] = task8_1_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task8_1_rank['publish_time'] = task8_1_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_1_rank['publish_time'] = pd.to_numeric(task8_1_rank['publish_time'])

task8_1_rank = task8_1_rank.sort_values(by='publish_time', ascending=False)

task8_1_rank.reset_index(inplace=True,drop=True)
rank = pd.read_csv("/kaggle/input/shanghai-ranking/rank-univ.csv")

del rank['Unnamed: 0']

rank.head(5)
# Extract the affiliations score to the task's results:

task8_1_rank['Aff_Score'] = 0

for i in range(len(task8_1_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_1_rank.iloc[i, 4]:

            task8_1_rank.iloc[i, 11] = rank.iloc[j, 3]
task8_1_rank["Ranking_Score"] = task8_1_rank["publish_time"]*0.8 + task8_1_rank["Aff_Score"]*0.2
task8_1_rank = task8_1_rank.sort_values(by='Ranking_Score', ascending=False)

task8_1_rank.reset_index(inplace=True,drop=True)

task8_1_rank
## 20 - Ranked Results for task 1.1 :



for i in range(len(task8_1_rank)):

    print("\n")

    print("PaperID: ", task8_1_rank.iloc[i, 0])

    print("Title: ", task8_1_rank.iloc[i, 1])

    print("Section: ", task8_1_rank.iloc[i, 2])

    print("Text: ", task8_1_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['mobilization','surge', 'medical', 'staff','address','covid19','covid-19','shortages','overwhelmed','community']

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

  

task8_2 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_2.append(i)

    

len(task8_2)
## Results for task 8.2 :

for i in task8_2:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 8.2:

print(task8_2)
task8_2_rank = papers_data.iloc[task8_2, :]

task8_2_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task8_2_rank['Title'] = task8_2_rank['Title'].astype(str) 



task8_2_rank = pd.merge(task8_2_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_2_rank.dropna(inplace=True)



# Extract the year from the string publish time

task8_2_rank['publish_time'] = task8_2_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_2_rank['publish_time'] = pd.to_numeric(task8_2_rank['publish_time'])

task8_2_rank = task8_2_rank.sort_values(by='publish_time', ascending=False)

task8_2_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_2_rank['Aff_Score'] = 0

for i in range(len(task8_2_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_2_rank.iloc[i, 4]:

            task8_2_rank.iloc[i, 11] = rank.iloc[j, 3]
task8_2_rank["Ranking_Score"] = task8_2_rank["publish_time"]*0.8 + task8_2_rank["Aff_Score"]*0.2

task8_2_rank = task8_2_rank.sort_values(by='Ranking_Score', ascending=False)

task8_2_rank.reset_index(inplace=True,drop=True)

task8_2_rank
## 20 - Ranked Results for task 1.2 :



for i in range(len(task8_2_rank)):

    print("\n")

    print("PaperID: ", task8_2_rank.iloc[i, 0])

    print("Title: ", task8_2_rank.iloc[i, 1])

    print("Section: ", task8_2_rank.iloc[i, 2])

    print("Text: ", task8_2_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['age-adjusted', 'mortality','data','acute','respiratory','distress','syndrome','ards','organ','failure','viral','etiologies']

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

    if len(a_set.intersection(b_set)) > 8: 

        return(True)  

    return(False)    

  

task8_3 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_3.append(i)

    

len(task8_3)
## Results for task 8.3 :

for i in task8_3:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 8.3:

print(task8_3)
task8_3_rank = papers_data.iloc[task8_3, :]

task8_3_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:

task8_3_rank['Title'] = task8_3_rank['Title'].astype(str) 



task8_3_rank = pd.merge(task8_3_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_3_rank.dropna(inplace=True)



# Extract the year from the string publish time

task8_3_rank['publish_time'] = task8_3_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_3_rank['publish_time'] = pd.to_numeric(task8_3_rank['publish_time'])

task8_3_rank = task8_3_rank.sort_values(by='publish_time', ascending=False)

task8_3_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_3_rank['Aff_Score'] = 0

for i in range(len(task8_3_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_3_rank.iloc[i, 4]:

            task8_3_rank.iloc[i, 11] = rank.iloc[j, 3]
task8_3_rank["Ranking_Score"] = task8_3_rank["publish_time"]*0.8 + task8_3_rank["Aff_Score"]*0.2

task8_3_rank = task8_3_rank.sort_values(by='Ranking_Score', ascending=False)

task8_3_rank.reset_index(inplace=True,drop=True)

task8_3_rank
## 20 - Ranked Results for task 1.3 :



for i in range(len(task8_3_rank)):

    print("\n")

    print("PaperID: ", task8_3_rank.iloc[i, 0])

    print("Title: ", task8_3_rank.iloc[i, 1])

    print("Section: ", task8_3_rank.iloc[i, 2])

    print("Text: ", task8_3_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['extracorporeal', 'membrane','oxygenation','ecmo','outcomes','data','covid-19','covid19','virus','patients']

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

  

task8_4 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_4.append(i)

    

len(task8_4)
## Results for task 8.4 :

for i in task8_4:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 8.4:

print(task8_4)
task8_4_rank = papers_data.iloc[task8_4, :]

task8_4_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task8_4_rank['Title'] = task8_4_rank['Title'].astype(str) 



task8_4_rank = pd.merge(task8_4_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_4_rank.dropna(inplace=True)



# Extract the year from the string publish time

task8_4_rank['publish_time'] = task8_4_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))

task8_4_rank['publish_time'] = task8_4_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_4_rank['publish_time'] = pd.to_numeric(task8_4_rank['publish_time'])

task8_4_rank = task8_4_rank.sort_values(by='publish_time', ascending=False)

task8_4_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_4_rank['Aff_Score'] = 0

for i in range(len(task8_4_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_4_rank.iloc[i, 4]:

            task8_4_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_4_rank["Ranking_Score"] = task8_4_rank["publish_time"]*0.8 + task8_4_rank["Aff_Score"]*0.2

task8_4_rank = task8_4_rank.sort_values(by='Ranking_Score', ascending=False)

task8_4_rank.reset_index(inplace=True,drop=True)

task8_4_rank
## 20 - Ranked Results for task 1.4 :



for i in range(len(task8_4_rank)):

    print("\n")

    print("PaperID: ", task8_4_rank.iloc[i, 0])

    print("Title: ", task8_4_rank.iloc[i, 1])

    print("Section: ", task8_4_rank.iloc[i, 2])

    print("Text: ", task8_4_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['outcomes', 'data','covid19','covid-19','mechanical','ventilation','adjust','age']

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

  

task8_5 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_5.append(i)

    

len(task8_5)
## Results for task 8.5 :

for i in task8_5:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 8.5 

print(task8_5)
task8_5_rank = papers_data.iloc[task8_5, :]

task8_5_rank.reset_index(inplace=True,drop=True)



# Grab the publish year from the meta data file:



task8_5_rank['Title'] = task8_5_rank['Title'].astype(str) 



task8_5_rank = pd.merge(task8_5_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_5_rank.dropna(inplace=True)



# Extract the year from the string publish time

task8_5_rank['publish_time'] = task8_5_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_5_rank['publish_time'] = pd.to_numeric(task8_5_rank['publish_time'])

task8_5_rank = task8_5_rank.sort_values(by='publish_time', ascending=False)

task8_5_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_5_rank['Aff_Score'] = 0

for i in range(len(task8_5_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_5_rank.iloc[i, 4]:

            task8_5_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_5_rank["Ranking_Score"] = task8_5_rank["publish_time"]*0.8 + task8_5_rank["Aff_Score"]*0.2

task8_5_rank = task8_5_rank.sort_values(by='Ranking_Score', ascending=False)

task8_5_rank.reset_index(inplace=True,drop=True)

task8_5_rank
## 20 - Ranked Results for task 1.5 :



for i in range(len(task8_5_rank)):

    print("\n")

    print("PaperID: ", task8_5_rank.iloc[i, 0])

    print("Title: ", task8_5_rank.iloc[i, 1])

    print("Section: ", task8_5_rank.iloc[i, 2])

    print("Text: ", task8_5_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['knowledge', 'frequency','manifestation','course','extrapulmonary','limited','possible','cardiomyopathy','cardiac','arrest','covid19','covidd-19','virus']

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

    if len(a_set.intersection(b_set)) > 9: 

        return(True)  

    return(False)    

  

task8_6 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_6.append(i)

    

len(task8_6)
## Results for task 8.6 :

for i in task8_6:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
# Task 8.6:

print(task8_6)
task8_6_rank = papers_data.iloc[task8_6, :]

task8_6_rank.reset_index(inplace=True,drop=True)



task8_6_rank['Title'] = task8_6_rank['Title'].astype(str) 



task8_6_rank = pd.merge(task8_6_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_6_rank.dropna(inplace=True)



# Extract the year from the string publish time

task8_6_rank['publish_time'] = task8_6_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_6_rank['publish_time'] = pd.to_numeric(task8_6_rank['publish_time'])

task8_6_rank = task8_6_rank.sort_values(by='publish_time', ascending=False)

task8_6_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_6_rank['Aff_Score'] = 0

for i in range(len(task8_6_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_6_rank.iloc[i, 4]:

            task8_6_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_6_rank["Ranking_Score"] = task8_6_rank["publish_time"]*0.8 + task8_6_rank["Aff_Score"]*0.2

task8_6_rank = task8_6_rank.sort_values(by='Ranking_Score', ascending=False)

task8_6_rank.reset_index(inplace=True,drop=True)

task8_6_rank
## 20 - Ranked Results for task 1.6 :



for i in range(len(task8_6_rank)):

    print("\n")

    print("PaperID: ", task8_6_rank.iloc[i, 0])

    print("Title: ", task8_6_rank.iloc[i, 1])

    print("Section: ", task8_6_rank.iloc[i, 2])

    print("Text: ", task8_6_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['application', 'regulatory','level','standards','eua','clia','virus','ability','adapt','crisis','care','covid19','coronavirus','covid-19']

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

  

task8_7 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_7.append(i)

    

len(task8_7)
## Results for task 8.7 :

for i in task8_7:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task8_7_rank = papers_data.iloc[task8_7, :]

task8_7_rank.reset_index(inplace=True,drop=True)



task8_7_rank['Title'] = task8_7_rank['Title'].astype(str) 



task8_7_rank = pd.merge(task8_7_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_7_rank.dropna(inplace=True)



# Extract the year from the string publish time

task8_7_rank['publish_time'] = task8_7_rank['publish_time'].apply(lambda x: str(x).replace('May 8 Summer',''))

task8_7_rank['publish_time'] = task8_7_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_7_rank['publish_time'] = pd.to_numeric(task8_7_rank['publish_time'])

task8_7_rank = task8_7_rank.sort_values(by='publish_time', ascending=False)

task8_7_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_7_rank['Aff_Score'] = 0

for i in range(len(task8_7_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_7_rank.iloc[i, 4]:

            task8_7_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_7_rank["Ranking_Score"] = task8_7_rank["publish_time"]*0.8 + task8_7_rank["Aff_Score"]*0.2

task8_7_rank = task8_7_rank.sort_values(by='Ranking_Score', ascending=False)

task8_7_rank.reset_index(inplace=True,drop=True)

task8_7_rank
## 20 - Ranked Results for task 1.7 :



for i in range(len(task8_7_rank)):

    print("\n")

    print("PaperID: ", task8_7_rank.iloc[i, 0])

    print("Title: ", task8_7_rank.iloc[i, 1])

    print("Section: ", task8_7_rank.iloc[i, 2])

    print("Text: ", task8_7_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['approaches', 'encouraging','facilitating','virus','production','elastomeric','respirators','save','thousands','n95','masks','covid19','covid-19','coronavirus']

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

  

task8_8 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_8.append(i)

    

len(task8_8)
## Results for task 8.8 :

for i in task8_8:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task8_8_rank = papers_data.iloc[task8_8, :]

task8_8_rank.reset_index(inplace=True,drop=True)



task8_8_rank['Title'] = task8_8_rank['Title'].astype(str) 



task8_8_rank = pd.merge(task8_8_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_8_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task8_8_rank['publish_time'] = task8_8_rank['publish_time'].apply(lambda x:  str(x).replace('Winter',''))

task8_8_rank['publish_time'] = task8_8_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_8_rank['publish_time'] = pd.to_numeric(task8_8_rank['publish_time'])

task8_8_rank = task8_8_rank.sort_values(by='publish_time', ascending=False)

task8_8_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_8_rank['Aff_Score'] = 0

for i in range(len(task8_8_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_8_rank.iloc[i, 4]:

            task8_8_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_8_rank["Ranking_Score"] = task8_8_rank["publish_time"]*0.8 + task8_8_rank["Aff_Score"]*0.2

task8_8_rank = task8_8_rank.sort_values(by='Ranking_Score', ascending=False)

task8_8_rank.reset_index(inplace=True,drop=True)

task8_8_rank
## 20 - Ranked Results for task 1.8 :



for i in range(len(task8_8_rank)):

    print("\n")

    print("PaperID: ", task8_8_rank.iloc[i, 0])

    print("Title: ", task8_8_rank.iloc[i, 1])

    print("Section: ", task8_8_rank.iloc[i, 2])

    print("Text: ", task8_8_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['telemedicine', 'best','practices','barriers','facilitators', 'specific','action','expand','state','boundries']

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

  

task8_9 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_9.append(i)

    

len(task8_9)
## Results for task 8.9 :

for i in task8_9:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task8_9_rank = papers_data.iloc[task8_9, :]

task8_9_rank.reset_index(inplace=True,drop=True)



task8_9_rank['Title'] = task8_9_rank['Title'].astype(str) 



task8_9_rank = pd.merge(task8_9_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_9_rank.dropna(inplace=True)



# Extract the year from the string publish time May 8 Summer

task8_9_rank['publish_time'] = task8_9_rank['publish_time'].apply(lambda x: str(x).replace('May 8 Summer',''))

task8_9_rank['publish_time'] = task8_9_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_9_rank['publish_time'] = pd.to_numeric(task8_9_rank['publish_time'])

task8_9_rank = task8_9_rank.sort_values(by='publish_time', ascending=False)

task8_9_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_9_rank['Aff_Score'] = 0

for i in range(len(task8_9_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_9_rank.iloc[i, 4]:

            task8_9_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_9_rank["Ranking_Score"] = task8_9_rank["publish_time"]*0.8 + task8_9_rank["Aff_Score"]*0.2

task8_9_rank = task8_9_rank.sort_values(by='Ranking_Score', ascending=False)

task8_9_rank.reset_index(inplace=True,drop=True)

task8_9_rank
## 20 - Ranked Results for task 1.9 :



for i in range(len(task8_9_rank)):

    print("\n")

    print("PaperID: ", task8_9_rank.iloc[i, 0])

    print("Title: ", task8_9_rank.iloc[i, 1])

    print("Section: ", task8_9_rank.iloc[i, 2])

    print("Text: ", task8_9_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['guidance', 'simple','simple','things','people','home','care','sick','manage','disease','covid-19','covid19','virus']

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

  

task8_10 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_10.append(i)

    

len(task8_10)
## Results for task 8.10 :

for i in task8_10:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task8_10_rank = papers_data.iloc[task8_10, :]

task8_10_rank.reset_index(inplace=True,drop=True)



task8_10_rank['Title'] = task8_10_rank['Title'].astype(str) 



task8_10_rank = pd.merge(task8_10_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_10_rank.dropna(inplace=True)



# Extract the year from the string publish time

task8_10_rank['publish_time'] = task8_10_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_10_rank['publish_time'] = pd.to_numeric(task8_10_rank['publish_time'])

task8_10_rank = task8_10_rank.sort_values(by='publish_time', ascending=False)

task8_10_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_10_rank['Aff_Score'] = 0

for i in range(len(task8_10_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_10_rank.iloc[i, 4]:

            task8_10_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_10_rank["Ranking_Score"] = task8_10_rank["publish_time"]*0.8 + task8_10_rank["Aff_Score"]*0.2

task8_10_rank = task8_10_rank.sort_values(by='Ranking_Score', ascending=False)

task8_10_rank.reset_index(inplace=True,drop=True)

task8_10_rank
## 20 - Ranked Results for task 1.10 :



for i in range(len(task8_10_rank)):

    print("\n")

    print("PaperID: ", task8_10_rank.iloc[i, 0])

    print("Title: ", task8_10_rank.iloc[i, 1])

    print("Section: ", task8_10_rank.iloc[i, 2])

    print("Text: ", task8_10_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['oral','medications','coronavirus','corona','covid','potentially','work','covid19','covid-19']

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

  

task8_11 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_11.append(i)

    

len(task8_11)
## Results for task 8.11 :

for i in task8_11:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task8_11_rank = papers_data.iloc[task8_11, :]

task8_11_rank.reset_index(inplace=True,drop=True)



task8_11_rank['Title'] = task8_11_rank['Title'].astype(str) 



task8_11_rank = pd.merge(task8_11_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_11_rank.dropna(inplace=True)



# Extract the year from the string publish time

task8_11_rank['publish_time'] = task8_11_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_11_rank['publish_time'] = pd.to_numeric(task8_11_rank['publish_time'])

task8_11_rank = task8_11_rank.sort_values(by='publish_time', ascending=False)

task8_11_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_11_rank['Aff_Score'] = 0

for i in range(len(task8_11_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_11_rank.iloc[i, 4]:

            task8_11_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_11_rank["Ranking_Score"] = task8_11_rank["publish_time"]*0.8 + task8_11_rank["Aff_Score"]*0.2

task8_11_rank = task8_11_rank.sort_values(by='Ranking_Score', ascending=False)

task8_11_rank.reset_index(inplace=True,drop=True)

task8_11_rank
## 20 - Ranked Results for task 1.11 :



for i in range(len(task8_11_rank)):

    print("\n")

    print("PaperID: ", task8_11_rank.iloc[i, 0])

    print("Title: ", task8_11_rank.iloc[i, 1])

    print("Section: ", task8_11_rank.iloc[i, 2])

    print("Text: ", task8_11_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['ai', 'real-time','health','care','delivery','evaluate','intervention','risk','factor','outcome','manually','covid19','virus','covid-19']

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

  

task8_12 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_12.append(i)

    

len(task8_12)
## Results for task 8.12 :

for i in task8_12:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task8_12_rank = papers_data.iloc[task8_12, :]

task8_12_rank.reset_index(inplace=True,drop=True)



task8_12_rank['Title'] = task8_12_rank['Title'].astype(str) 



task8_12_rank = pd.merge(task8_12_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_12_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task8_12_rank['publish_time'] = task8_12_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))

task8_12_rank['publish_time'] = task8_12_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_12_rank['publish_time'] = pd.to_numeric(task8_12_rank['publish_time'])

task8_12_rank = task8_12_rank.sort_values(by='publish_time', ascending=False)

task8_12_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_12_rank['Aff_Score'] = 0

for i in range(len(task8_12_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_12_rank.iloc[i, 4]:

            task8_12_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_12_rank["Ranking_Score"] = task8_12_rank["publish_time"]*0.8 + task8_12_rank["Aff_Score"]*0.2

task8_12_rank = task8_12_rank.sort_values(by='Ranking_Score', ascending=False)

task8_12_rank.reset_index(inplace=True,drop=True)

task8_12_rank
## 20 - Ranked Results for task 1.12 :



for i in range(len(task8_12_rank)):

    print("\n")

    print("PaperID: ", task8_12_rank.iloc[i, 0])

    print("Title: ", task8_12_rank.iloc[i, 1])

    print("Section: ", task8_12_rank.iloc[i, 2])

    print("Text: ", task8_12_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['practices', 'critical','challenges','innovative','solution','technology','hostipal','flow','organization','workforce','protection','allocation','community','resources','payment','supply','chain','management','capacity','efficiency','outcomes','covid19','virus','covid-19']

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

  

task8_13 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_13.append(i)

    

len(task8_13)
## Results for task 8.13 :

for i in task8_13:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")
task8_13_rank = papers_data.iloc[task8_13, :]

task8_13_rank.reset_index(inplace=True,drop=True)



task8_13_rank['Title'] = task8_13_rank['Title'].astype(str) 



task8_13_rank = pd.merge(task8_13_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_13_rank.dropna(inplace=True)



# Extract the year from the string publish time

task8_13_rank['publish_time'] = task8_13_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_13_rank['publish_time'] = pd.to_numeric(task8_13_rank['publish_time'])

task8_13_rank = task8_13_rank.sort_values(by='publish_time', ascending=False)

task8_13_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_13_rank['Aff_Score'] = 0

for i in range(len(task8_13_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_13_rank.iloc[i, 4]:

            task8_13_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_13_rank["Ranking_Score"] = task8_13_rank["publish_time"]*0.8 + task8_13_rank["Aff_Score"]*0.2

task8_13_rank = task8_13_rank.sort_values(by='Ranking_Score', ascending=False)

task8_13_rank.reset_index(inplace=True,drop=True)

task8_13_rank
## 20 - Ranked Results for task 1.13 :



for i in range(len(task8_13_rank)):

    print("\n")

    print("PaperID: ", task8_13_rank.iloc[i, 0])

    print("Title: ", task8_13_rank.iloc[i, 1])

    print("Section: ", task8_13_rank.iloc[i, 2])

    print("Text: ", task8_13_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['efforts','natural','history','disease','clinical','care','public','health','epidemic','intervention','infection','prevention','control','transmission','trials','covid19','covid-19','virus']

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

    if len(a_set.intersection(b_set)) > 15: 

        return(True)  

    return(False)    

  

task8_14 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_14.append(i)

    

len(task8_14)
## Results for task 8.14 :

for i in task8_14:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")

    
task8_14_rank = papers_data.iloc[task8_14, :]

task8_14_rank.reset_index(inplace=True,drop=True)



task8_14_rank['Title'] = task8_14_rank['Title'].astype(str) 



task8_14_rank = pd.merge(task8_14_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_14_rank.dropna(inplace=True)



# Extract the year from the string publish time

import dateutil.parser as parser

task8_14_rank['publish_time'] = task8_14_rank['publish_time'].apply(lambda x:  str(x).replace('May 8 Summer',''))

task8_14_rank['publish_time'] = task8_14_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_14_rank['publish_time'] = pd.to_numeric(task8_14_rank['publish_time'])

task8_14_rank = task8_14_rank.sort_values(by='publish_time', ascending=False)

task8_14_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_14_rank['Aff_Score'] = 0

for i in range(len(task8_14_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_14_rank.iloc[i, 4]:

            task8_14_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_14_rank["Ranking_Score"] = task8_14_rank["publish_time"]*0.8 + task8_14_rank["Aff_Score"]*0.2

task8_14_rank = task8_14_rank.sort_values(by='Ranking_Score', ascending=False)

task8_14_rank.reset_index(inplace=True,drop=True)

task8_14_rank
## 20 - Ranked Results for task 6.14 :



for i in range(len(task8_14_rank)):

    print("\n")

    print("PaperID: ", task8_14_rank.iloc[i, 0])

    print("Title: ", task8_14_rank.iloc[i, 1])

    print("Section: ", task8_14_rank.iloc[i, 2])

    print("Text: ", task8_14_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['efforts','develop','core','clinical','outcomes','maximize','usability','data','epidemic','covid19','didease','covid-19','trials']

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

  

task8_15 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_15.append(i)

    

len(task8_15)
## Results for task 8.15 :

for i in task8_15:

    print("\n")

    print("PaperID: ", papers_data.iloc[i, 0])

    print("Title: ", papers_data.iloc[i, 1])

    print("Section: ", papers_data.iloc[i, 2])

    print("Text: ", papers_data.iloc[i, 3])  

    print("\n")

    
task8_15_rank = papers_data.iloc[task8_15, :]

task8_15_rank.reset_index(inplace=True,drop=True)



task8_15_rank['Title'] = task8_15_rank['Title'].astype(str) 



task8_15_rank = pd.merge(task8_15_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_15_rank.dropna(inplace=True)



# Extract the year from the string publish time

task8_15_rank['publish_time'] = task8_15_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_15_rank['publish_time'] = pd.to_numeric(task8_15_rank['publish_time'])

task8_15_rank = task8_15_rank.sort_values(by='publish_time', ascending=False)

task8_15_rank.reset_index(inplace=True,drop=True)
# Extract the affiliations score to the task's results:

task8_15_rank['Aff_Score'] = 0

for i in range(len(task8_15_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_15_rank.iloc[i, 4]:

            task8_15_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_15_rank["Ranking_Score"] = task8_15_rank["publish_time"]*0.8 + task8_15_rank["Aff_Score"]*0.2

task8_15_rank = task8_15_rank.sort_values(by='Ranking_Score', ascending=False)

task8_15_rank.reset_index(inplace=True,drop=True)

task8_15_rank
## 20 - Ranked Results for task 6.14 :



for i in range(len(task8_15_rank)):

    print("\n")

    print("PaperID: ", task8_15_rank.iloc[i, 0])

    print("Title: ", task8_15_rank.iloc[i, 1])

    print("Section: ", task8_15_rank.iloc[i, 2])

    print("Text: ", task8_15_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
keywords =['efforts','adjunctive','supportive','interventions','clinical','outcomes','infected','patients','steriod','oxygen','covid19','covid-19','epidemic','didease']

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

    if len(a_set.intersection(b_set)) > 9: 

        return(True)  

    return(False)    

  

task8_16 =[]

for i in range(len(papers_data)):

    if common_member(kw, papers_data.iloc[i, 8]):

        task8_16.append(i)

    

len(task8_16)
task8_16_rank = papers_data.iloc[task8_16, :]

task8_16_rank.reset_index(inplace=True,drop=True)



task8_16_rank['Title'] = task8_16_rank['Title'].astype(str) 



task8_16_rank = pd.merge(task8_16_rank, meta[['Title','publish_time']], left_on='Title', right_on='Title')

task8_16_rank.dropna(inplace=True)



# Extract the year from the string publish time

task8_16_rank['publish_time'] = task8_16_rank['publish_time'].apply(lambda x: parser.parse(x).year)



# Rank the task's results by time (Freshness)

task8_16_rank['publish_time'] = pd.to_numeric(task8_16_rank['publish_time'])

task8_16_rank = task8_16_rank.sort_values(by='publish_time', ascending=False)

task8_16_rank.reset_index(inplace=True,drop=True)



## Ranking by affiliations



# Extract the affiliations score to the task's results:

task8_16_rank['Aff_Score'] = 0

for i in range(len(task8_16_rank)):

    for j in range(len(rank)):

        if rank.iloc[j, 1] in task8_16_rank.iloc[i, 4]:

            task8_16_rank.iloc[i, 11] = rank.iloc[j, 3]

            

task8_16_rank["Ranking_Score"] = task8_16_rank["publish_time"]*0.8 + task8_16_rank["Aff_Score"]*0.2

task8_16_rank = task8_16_rank.sort_values(by='Ranking_Score', ascending=False)

task8_16_rank.reset_index(inplace=True,drop=True)





## 20 - Ranked Results for task 6.14 :



for i in range(len(task8_16_rank)):

    print("\n")

    print("PaperID: ", task8_16_rank.iloc[i, 0])

    print("Title: ", task8_16_rank.iloc[i, 1])

    print("Section: ", task8_16_rank.iloc[i, 2])

    print("Text: ", task8_16_rank.iloc[i, 3])  

    print("\n")

    if i == 19:

        break
task8_1_rank.to_csv("task8_1_rank.csv")

task8_2_rank.to_csv("task8_2_rank.csv")

task8_3_rank.to_csv("task8_3_rank.csv")

task8_4_rank.to_csv("task8_4_rank.csv")

task8_5_rank.to_csv("task8_5_rank.csv")

task8_6_rank.to_csv("task8_6_rank.csv")

task8_7_rank.to_csv("task8_7_rank.csv")

task8_8_rank.to_csv("task8_8_rank.csv")

task8_9_rank.to_csv("task8_9_rank.csv")

task8_10_rank.to_csv("task8_10_rank.csv")

task8_11_rank.to_csv("task8_11_rank.csv")

task8_12_rank.to_csv("task8_12_rank.csv")

task8_13_rank.to_csv("task8_13_rank.csv")

task8_14_rank.to_csv("task8_14_rank.csv")

task8_15_rank.to_csv("task8_15_rank.csv")

task8_16_rank.to_csv("task8_16_rank.csv")