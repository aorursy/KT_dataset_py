import pandas as pd
import numpy as np 

"""
I got  starter  code that filters the articles from another kernel
https://www.kaggle.com/mlconsult/summary-page-covid-19-risk-factors 

"""



import re
import os
import json
# keep only documents with covid -cov-2 and cov2
def search_focus(df):
    dfa = df[df['abstract'].str.contains('covid')]
    dfb = df[df['abstract'].str.contains('-cov-2')]
    dfc = df[df['abstract'].str.contains('cov2')]
    dfd = df[df['abstract'].str.contains('ncov')]
    frames=[dfa,dfb,dfc,dfd]
    df = pd.concat(frames)
    df=df.drop_duplicates(subset='title', keep="first")
    return df


df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','journal','abstract','authors','doi','publish_time','sha','pdf_json_files'])
print ('All CORD19 documents ',df.shape)
#fill na fields
df=df.fillna('no data provided')
#drop duplicate titles
df = df.drop_duplicates(subset='title', keep="first")
#keep only 2020 dated papers
df=df[df['publish_time'].str.contains('2020')]
# convert abstracts to lowercase
df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()
#show 5 lines of the new dataframe
df=search_focus(df)
print ("COVID-19 focused docuemnts ",df.shape)



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
for index, row in df.iterrows():
    if ';' not in row['sha'] and os.path.exists('/kaggle/input/CORD-19-research-challenge/'+row['pdf_json_files']+'/'+row['pdf_json_files']+'/pdf_json/'+row['sha']+'.json')==True:
        with open('/kaggle/input/CORD-19-research-challenge/'+row['pdf_json_files']+'/'+row['pdf_json_files']+'/pdf_json/'+row['sha']+'.json') as json_file:
            data = json.load(json_file)
            body=format_body(data['body_text'])
            keyword_list=['TB','incidence','age']
            #print (body)
            body=body.replace("\n", " ")

            df.loc[index, 'abstract'] = body.lower()

df=df.drop(['pdf_json_files'], axis=1)
df=df.drop(['sha'], axis=1)
# df.head()
df.reset_index(inplace=True)
df.drop("index",axis=1,inplace=True)
import nltk
nltk.download("punkt")

from nltk import word_tokenize,sent_tokenize
from nltk.stem  import PorterStemmer


from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download('stopwords')
stops = stopwords.words("english")


def removepunc(my_str):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct

def hasNumbers(inputString):
    return (bool(re.search(r'\d', inputString)))
snowstem = SnowballStemmer("english")
portstem = PorterStemmer()



"""
These are the queries related to risk factors

"""

usequeries = sent_tokenize("""Smoking, pre-existing pulmonary disease
Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities.
cardiovascular disease , chronic obstructive pulmonary disease and diabetes.
Neonates and pregnant women.
Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.
Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high risk patient groups
Susceptibility of populations.
Public health mitigation measures that could be effective for control.
immune system disorders.
heart failure.
drinking.
diabetes.

""")
queryarticle = [" ".join([snowstem.stem(removepunc(i.lower())) for i in word_tokenize(x) if i not in stops ]) for x in usequeries]

"""

had to reduce the vocabulary of the data


"""
df["usetext"] = df.abstract.apply(lambda x: " ".join([snowstem.stem(i) for i in word_tokenize(removepunc(x.lower())) if not hasNumbers(i) if i not in stops]))

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
encArticles = vectorizer.fit_transform(df.usetext)
encQueries = vectorizer.transform(queryarticle)
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix  = cosine_similarity(encQueries,encArticles)

print(np.sort(similarity_matrix[1])[-5:][::-1]) #sorting to get the most similar articles for a given query
np.argsort(similarity_matrix[1])[-5:][::-1]
import torch
from transformers import  AutoTokenizer,AutoModelForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")

model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")

def ask(question,context):
  input_ids = tokenizer.encode(question, context)
  sep_index = input_ids.index(tokenizer.sep_token_id)

  num_seg_a = sep_index + 1

  num_seg_b = len(input_ids) - num_seg_a
  segment_ids = [0]*num_seg_a + [1]*num_seg_b
  assert len(segment_ids) == len(input_ids)
  tokens = tokenizer.convert_ids_to_tokens(input_ids)


  start_scores, end_scores = model(torch.tensor([input_ids]),
                                 token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text
  answer_end = 0
  answer_start = torch.argmax(start_scores)
  answer_ends = torch.argsort(end_scores).numpy()[::-1]
  for i in answer_ends[0]:
    if answer_start<= i:
      answer_end= i

  answer = ' '.join(tokens[answer_start:answer_end+1])
  answer = answer.replace(" ##","").replace("[CLS] ","")

  pack = [answer,answer_start,answer_end,torch.max(start_scores),end_scores[0][answer_end],(torch.max(start_scores)+end_scores[0][answer_end]),context]
  return pack
from IPython.display import display, HTML
"""
this function is used to visualize the answers with their contexts 
"""

def highlightTextInContext(answer, context):
    if "?"  in answer:
        answer =" ".join(answer[answer.index("?")+1:].split(" "))
    
    antokens = word_tokenize(answer)
    cotokens = word_tokenize(context)
    startword= ""
    startindex= ""
    for i,w in enumerate(antokens):
        for c in cotokens:
            if c==w:
                startword = c 
                selectedText = context[context.index(w):context.index(antokens[-1])+len(antokens[-1])]
                highlighted = f'<span style="color: green; font-weight: bold">{selectedText}</span>'
                return context.replace(selectedText,highlighted)
                # return is an easy way to break two nested loops
def showTopAnswers(answers):
        for i in np.argsort(answers[:,5])[-8:][::-1]:
            display(HTML("<p>"+highlightTextInContext(answers[i,0],answers[i,6])+"</p>"))
def getanswers(question):
  recommendations = []
  for i in range(len(usequeries)):
    indecies = np.argsort(similarity_matrix[i])[-7:][::-1] ## I choose to show N recommended queries from every query
    for t in indecies:
        recommendations.append(word_tokenize(df.abstract[t]))
  
  processedQuestion =   " ".join([snowstem.stem(i) for i in word_tokenize(removepunc(question)) if i not in stops])
  vector = vectorizer.transform([processedQuestion])
  questionSimilarityMatrix = cosine_similarity(vector,encArticles)
  indecies = np.argsort(questionSimilarityMatrix[0])[-7:][::-1] 
  for t in indecies:
    recommendations.append(word_tokenize(df.abstract[t]))
          
  questions= []
  contexts= []
  for bigcontext in recommendations:
    for i in range(int(len(bigcontext)/60)):
      contexts.append(" ".join(bigcontext[i*60:60*(i+1)]))
      questions.append(question)

  answers = []
  for  question, context in zip(questions,contexts):
    result = ask(question,context)
    if len(result[0]) < 7 and "[CLS]" in result[0] :
      continue
    answers.append(result)
  answers = np.array(answers)

  return answers
answers = getanswers("are pregnant women at risk ?")
showTopAnswers(answers)
answers = getanswers("what are the risk factors ?")
showTopAnswers(answers)
answers = getanswers("how will the virus affect neonates ?")
showTopAnswers(answers)
answers = getanswers("are infected diabetic patients at risk?")
showTopAnswers(answers)
answers = getanswers("how will hypertension affect patients?")
showTopAnswers(answers)