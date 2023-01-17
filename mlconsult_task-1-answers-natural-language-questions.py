###################### LOAD PACKAGES ##########################

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk import PorterStemmer

from IPython.core.display import display, HTML

import torch

#!pip install -q transformers --upgrade

from transformers import *

!pip install bert-extractive-summarizer

from summarizer import Summarizer

#from transformers import pipeline

import pandas as pd



#https://colab.research.google.com/drive/1rN0CS0hoxeByoPZu6_zF-AFJijYLsPw3
def remove_stopwords(query,stopwords):

    qstr=''

    qstr=qstr.join(query)

    #remove punctuaiton

    qstr = "".join(c for c in qstr if c not in ('!','.',',','?','(',')','-'))

    text_tokens = word_tokenize(qstr)

    #remove stopwords

    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

    #stem words

    tokens_without_sw=stem_words(tokens_without_sw)

    str1=''

    str1=' '.join(word for word in tokens_without_sw)

    return str1



def stem_words(words):

    stemmer = PorterStemmer()

    singles=[]

    for w in words:

        singles.append(stemmer.stem(w))

    return singles



# query MySQL database returns most relevant paper sentences in dataframe

def get_search_table(query):

    query=query.replace(" ","+")

    urls=r"https://edocdiscovery.com/covid_19/covid_19_search_api.php?search_string="+query

    table = pd.read_csv(urls,encoding= 'unicode_escape')

    return table



# BERT pretrained question answering module

def answer_question(question,text):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"

    input_ids = tokenizer.encode(input_text)

    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]

    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    #print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    answer=(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    # show qeustion and text

    #tokenizer.decode(input_ids)

    return answer



def prepare_summary_answer(text):

    model = Summarizer()

    #model = pipeline(task="summarization")

    

    return model(text)





###################### MAIN PROGRAM ###########################



### NL questions

#'''

questions = [

'What is the range of incubation period in days?',

'How long is the period from first positive test to test clearance or communicable period?',

'Does the incubation period differ or is it the same across age groups or children?',

'What was the basic reproduction number?',

'What was the highest basic reproduction number?',

'Can the virus be spread asymptomatically?',

'How is transmission affected by seasonal changes in humidity and temperature?',

'What agents are best to disinfect surfaces and inactivate virus?',

'How long of period is viral shedding in sputum, fecal matter, urine?',

'How many days does the virus persist on inanimate surfaces?',

'What is the median viral shedding duration?',

'What is the longest duration of viral shedding?',

'What models for infections predict?',

'What do models for transmission predict?',

'What do we know about phenotypes?',

'What do we know about immune response and immunity?',

'How effective has movement control been in controlling spread?',

'What is the minimum recommended PPE that should worn?',

'How does environment affect transmission or spread?'

]

#'''

# use QA or summarize for the realted NL question?

a_type=['qa','qa','qa','qa','qa','qa','qa','qa','qa','qa','qa','qa','sum','sum','sum','sum','qa','qa','sum']



#test one question

#questions=['What do we know about phenotypes?']

#test one question type of answer

#a_type=['sum']



q=0



# loop through the list of questions

for question in questions:



    #remove punctuation, stop words and stem words from NL question

    search_words=remove_stopwords(question,stopwords)

    

    #clean up bad stems that do not render search results

    bad_stems=['phenotyp','deadli','contagi','recoveri','rout','viru', 'surfac','immun','respons','person','protect','includ']

    replace_with=['phenotype','dead','contagious','recovery','route','virus','surface','immune','response','personal','protective','include']

    r=0

    for words in bad_stems:

        search_words=search_words.replace(words,replace_with[r])

        r=r+1

    # use to see stemmed query for troubleshooting

    #print (search_words)



    # get best sentences

    df_table=get_search_table(search_words)

    df_answers=df_table

    

    # if qa limit dataframe search rows to consider

    if a_type[q]=='qa':

        df_answers=df_table.head(5)

    

    # if sum expand dataframe search rows to consider

    if a_type[q]=='sum':

        df_answers=df_table.head(100)

    

    text=''

    

    for index, row in df_answers.iterrows():

        text=text+' '+row['excerpt']

        

    display(HTML('<h3>'+question+'</h3>'))

    

    #if qa use the question answering function

    if a_type[q]=='qa':

        answer=answer_question(question,text)

        answer=answer.replace("#", "")

        answer=answer.replace(" . ", ".")

        display(HTML('<h4> Answer:</h4> '+ answer))

         

    #if sum use the summarizer function

    if a_type[q]=='sum':

        summary_answer=prepare_summary_answer(text)

        #summary_answer=summary_answer[0]['summary_text']

        display(HTML('<h4> Summarized Answer: </h4><i>'+summary_answer+'</i>'))

    

    #print (text)

    

    #limit the size of the df for the html table

    df_table=df_table.head(5)

    

    #convert df to html

    df_table=HTML(df_table.to_html(escape=False,index=False))

    

    # show the HTML table with responses

    display(df_table)

    

    # link to web based CORD search preloaded

    sstr=search_words.replace(" ","+")

    cord_link='<a href="http://edocdiscovery.com/covid_19/xscript_serp.php?search_string='+sstr+'">see more web-based results</a>'

    display(HTML(cord_link))

    

    q=q+1

    

print ('done')