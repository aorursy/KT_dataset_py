###################### LOAD PACKAGES ##########################

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk import PorterStemmer

from IPython.core.display import display, HTML

import pandas as pd

import torch

from transformers import *

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# query MySQL database returns most relevant paper scored by sentence sentences in csv

def get_search_table(query,keyword):

    query=query.replace(" ","+")

    urls=r"https://edocdiscovery.com/covid_19/covid_19_risk_table_api.php?search_string="+query+'&keyword='+keyword

    table = pd.read_csv(urls,encoding= 'unicode_escape')

    return table



# BERT pretrained question answering module

def answer_question(question,text, model,tokenizer):

    input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"

    input_ids = tokenizer.encode(input_text)

    token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]

    start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    #print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    answer=(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))

    # show qeustion and text

    #tokenizer.decode(input_ids)

    answer=answer.replace(" ##", "")

    answer=answer.replace(" · ", "·")

    answer=answer.replace(" . ", ".")

    answer=answer.replace(" , ", ",")

    if '[SEP]'in answer or '[CLS]' in answer or answer=='':

        answer='unk'

        

    return answer





###################### MAIN PROGRAM ###########################





### focus quesiton with single keyword

keywords = ['diabetes']

#'diabetes','heart disease','male gender','copd','smoking','age','stroke','cerbrovascular','cancer','kidney disease','drinking','tuberculosis','bmi'



q=0



# loop through the list of questions

for keyword in keywords:

    # limit results to severe risk factors

    search_words = keyword+' risk factor severe'

    

    # get best sentences

    df_table=get_search_table(search_words,keyword)

    df_answers=df_table

        

    display(HTML('<h3>'+search_words+'</h3>'))

    

    #print (text)

    

    #limit the size of the df for the html table

    #df_table=df_table.head(100)

    df_table=df_table.drop_duplicates(subset='study', keep="first")

    df_table = df_table.sort_values(by=['date'], ascending=False)

    

    df_table

    for index, row in df_table.iterrows():

        text=row['method'][0:1000]

        ## get sample size

        question='how many patients or cases were included this group study anlaysis report review?'

        sample=answer_question(question,text,model,tokenizer)

        df_table.loc[index,'sample']=sample

        

        question='what kind or type of study anlaysis report review was conducted?'

        design=answer_question(question,text,model,tokenizer)

        design=design[0:100]

        df_table.loc[index,'design']=design

        

        ### get sever numbers

        text2=row['odds'][0:1000]

        question='what is the '+keyword+' HR OR RR AOR hazard odds ratio ()?'

        severe=answer_question(question,text2,model,tokenizer)

        df_table.loc[index,'severe']=severe

        

        

    df_table=df_table.drop(['odds', 'method','excerpt'], axis = 1) 

    

    df_allriskcsv=df_table

    

    #convert df to html

    df_table=HTML(df_table.to_html(escape=False,index=False))

    

    # show the HTML table with responses

    display(df_table)

    

    

    q=q+1

df_allriskcsv.to_csv('diabetes_risk_factors.csv',index = False)

print ('done')