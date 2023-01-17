from tqdm.notebook import tqdm
import csv
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models import Word2Vec
import gensim 
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings(action = 'ignore')
def format_name(author):
    middle_name = " ".join(author['middle'])
    
    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])

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
        body += text
    
    return body
# articles_dir = '../input/CORD-19-research-challenge/document_parses/pdf_json/'
# filenames = os.listdir(articles_dir)

# all_files = []

# for filename in filenames:
#     filename = articles_dir + filename
#     file = json.load(open(filename, 'rb'))
#     all_files.append(file)
    
# cleaned_files = []

# for file in tqdm(all_files):
#     features = [
#         file['paper_id'],
#         format_authors(file['metadata']['authors']),
#         format_body(file['body_text'])
#         ]
    
#     cleaned_files.append(features)
# col_names = ['paper_id','authors','text']
# complete_df = pd.DataFrame(cleaned_files, columns=col_names)
# complete_df.head(2)
# print("All articles found in the dataset of the CORD-19-research-challenge on Kaggle.")
# complete_df.shape
# #keeping documents with (cov, covid-19, corona, coronavirus, cov2, ncov) substrings in their titles
# #all summed up in 2 substrings: (cov, corona) to focus the search on the papers and avoid unrelated documents
# def search_focus(df):
#     dfb = df[df['abstract'].str.contains('cov')]
#     dfc = df[df['abstract'].str.contains('corona')]
#     frames=[dfb,dfc]
#     df = pd.concat(frames)
#     df=df.drop_duplicates(subset='title', keep="first")
#     return df

# # load the meta data from the CSV file
# df=pd.read_csv('../input/CORD-19-research-challenge/metadata.csv', usecols=['sha','title','abstract','publish_time'])
# print("Shape of articles DF found in the metadata.csv file: ")
# print (df.shape)

# df.rename(columns={'sha':'paper_id'},inplace=True)

# #drop na fields
# df=df.dropna()
# #convert to lower case and drop duplicate titles
# df["title"] = df["title"].str.lower()
# df = df.drop_duplicates(subset='title', keep="first")
# df=df[df['publish_time'].str.contains('2020')]
# # converting to lowercase
# df["abstract"] = df["abstract"].str.lower()
# df=search_focus(df)
# print("Shape of articles DF after extracting only the needed articles from the metadata.csv file: ")
# print (df.shape)
# df.head(2)
# final_df= pd.merge(df,complete_df,on='paper_id').drop(['paper_id','title','abstract','publish_time','authors'],axis=1)
# final_df = final_df[final_df['text'].apply(lambda x: len(re.findall(r"(?i)\b[a-z]+\b", x))) > 1000]
# final_df.reset_index(inplace=True, drop=True)
# print("Articles with full text found in both metadata.csv file & Kaggle CORD-19-research-challenge dataset.")
# final_df.shape
# final_df.head(2)
# f = []
# f = final_df['text']
# data = []

# for x in range(len(f)):
#         # tokenize the corpus into sentences 
#     for i in sent_tokenize(f[x]): 
#         temp = [] 

#         # tokenize the sentence into words 
#         for j in word_tokenize(i): 
#             temp.append(j) 

#         data.append(temp)
with open('../input/tokenized-words-cord19-challenge/data.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
model2 = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
                                             window = 5, sg = 1)
# a function to round decimals to the nearest 100th
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

print("Cosine similarity using Skip Gram model between:")
a=truncate((model2.similarity('risk', 'smoking')),2)
b=truncate((model2.similarity('risk', 'heart')),2)
c=truncate((model2.similarity('risk', 'pregnant')),2)
d=truncate((model2.similarity('risk', 'cancer')),2)
e=truncate((model2.similarity('risk', 'diabetes')),2)
f=truncate((model2.similarity('risk', 'age')),2)
g=truncate((model2.similarity('risk', 'asthma')),2)
h=truncate((model2.similarity('risk', 'HIV')),2)
i=truncate((model2.similarity('risk', 'transplant')),2)
j=truncate((model2.similarity('risk', 'obesity')),2)
k=truncate((model2.similarity('risk', 'immunocompromised')),2)
l=truncate((model2.similarity('risk', 'underweight')),2)
m=truncate((model2.similarity('risk', 'liver')),2)
n=truncate((model2.similarity('risk', 'bronchitis')),2)
o=truncate((model2.similarity('risk', 'COPD')),2)

print("risk and smoking : ",a)
print("risk and heart : ",b)
print("risk and pregnant  : ",c)
print("risk and cancer : ",d)
print("risk and diabetes : ",e)
print("risk and age : ",f)
print("risk and asthma : ",g)
print("risk and HIV : ",h)
print("risk and transplant : ",i)
print("risk and obesity : ",j)
print("risk and immunocompromised : ",k)
print("risk and underweight : ",l)
print("risk and liver : ",m)
print("risk and bronchitis : ",n)
print("risk and COPD : ",o)
objects = ('Smoking', 'Heart Disease', 'Pregnancy', 'Cancer', 'Diabetes' ,'Age', 'Asthma','HIV','Transplant','Obesity','Immunocompromised','Underweight','Liver Disease','Chronic Bronchitis','COPD')
y_pos = np.arange(len(objects))
similarity = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o]

fig = plt.figure(1, [15, 10])
axes = plt.gca()
axes.set_ylim([0,1])
plt.bar(y_pos, similarity, align='center', alpha=0.5)
plt.xticks(y_pos, objects,rotation=90)
plt.ylabel('Cosine similarity with the word "risk" using Skip Gram model')
plt.title('Risk Factors')

count=-0.27
for i in similarity:
    plt.text(count,i-0.05,str(i))
    count+=1

plt.show()