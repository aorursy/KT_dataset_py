###################### LOAD PACKAGES ##########################

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk import PorterStemmer

from IPython.core.display import display, HTML

import torch

!pip install -q transformers --upgrade

from transformers import *

!pip install bert-extractive-summarizer

from summarizer import Summarizer

#from transformers import pipeline

import pandas as pd

modelqa = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

model = Summarizer()

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

    #tokens_without_sw=stem_words(tokens_without_sw)

    str1=''

    #str1=' '.join(word for word in tokens_without_sw)

    str1=' '.join(word for word in text_tokens)

    return str1



def stem_words(words):

    stemmer = PorterStemmer()

    singles=[]

    for w in words:

        singles.append(stemmer.stem(w))

    return singles



# query MySQL database returns most relevant paper sentences in dataframe

def get_search_table(query,keyword):

    query=query.replace(" ","+")

    urls=r"https://edocdiscovery.com/covid_19/covid_19_search_api_v2.php?search_string="+query+'&keyword='+keyword

    table = pd.read_csv(urls,encoding= 'unicode_escape')

    return table



# BERT pretrained question answering module

def answer_question(question,text,model):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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



def prepare_summary_answer(text,model):

    #model = pipeline(task="summarization")

    return model(text)





###################### MAIN PROGRAM ###########################



### NL questions

#'''

questions = [

'What do we know about the genome and  evolution of the virus?',

'Where is information or data about the genome shared?',

'Are diverse genome sample sets available?',

'How many COV 2 strains are circulating?',

'What agreement exist to share data or information?',

'What do we know about COV 2 and livestock?',

'What do we know about animal reservoirs?',

'What do we know about framers being infected?',

'What do we know about human wildlife interface or interaction and the infected?',

'What do we know about experiments for the host range?',

'What do we know about wild animal as reservoir or hosts and spillover?',

'What are the socioeconomic and behavioral risks for this spillover infection?',

'What are some sustainable risk reduction strategies to avoid spillover from animals to humans?'

]

#'''

### focus quesiton with single keyword

keyword=['genome','genome data','genome','strain','data sharing','livestock','reservoir','farmer','wildlife','host range','hosts','spillover','animal','spillover']

# use QA or summarize for the realted NL question?

a_type=['sum','qa','qa','qa','sum','sum','sum','sum','sum','sum','sum','sum','sum']



#test one question

#questions=['Can livestock such as cows, horses and sheep be infected?']

#keyword=['livestock']

#test one question type of answer

#a_type=['sum']



q=0



# loop through the list of questions

for question in questions:



    #remove punctuation, stop words and stem words from NL question

    search_words=remove_stopwords(question,stopwords)

    

    #clean up bad stems that do not render search results

    bad_stems=['phenotyp','deadli','contagi','recoveri','rout','viru', 'surfac','immun','respons','person','protect','includ','smoke','diabet']

    replace_with=['phenotype','dead','contagious','recovery','route','virus','surface','immune','response','personal','protective','include','smoking','diabetes']

    r=0

    for words in bad_stems:

        search_words=search_words.replace(words,replace_with[r])

        r=r+1

    # use to see stemmed query for troubleshooting

    #print (search_words)



    # get best sentences

    df_table=get_search_table(search_words,keyword[q])

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

        

    display(HTML('<h1>'+question+'</h1>'))

    

    #if qa use the question answering function

    if a_type[q]=='qa':

        answer=answer_question(question,text,modelqa)

        answer=answer.replace("#", "")

        answer=answer.replace(" . ", ".")

        display(HTML('<h4> Answer:</h4> '+ answer))

         

    #if sum use the summarizer function

    if a_type[q]=='sum':

        summary_answer=prepare_summary_answer(text,model)

        #summary_answer=summary_answer[0]['summary_text']

        display(HTML('<h4> Summarized Answer: </h4><i>'+summary_answer+'</i>'))

    

    #print (text)

    

    #limit the size of the df for the html table

    df_table=df_table.head(5)

    

    #convert df to html

    df_table=HTML(df_table.to_html(escape=False,index=False))

    

    display(HTML('<h5>results limited to 5 for ease of scanning</h5>'))

    # show the HTML table with responses

    display(df_table)

    

    # link to web based CORD search preloaded

    sstr=search_words.replace(" ","+")

    cord_link='<a href="http://edocdiscovery.com/covid_19/xscript_serp.php?search_string='+sstr+'">see more web-based results</a>'



    display(HTML(cord_link))

    

    q=q+1

    

print ('done')