!pip install googletrans
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

import os

import torch

from tqdm import tqdm,tqdm_notebook

from IPython.core.display import display, HTML

from googletrans import Translator

import gc

# Input data files are available in the "../input/" directory.

#####################################################################################

#thanks for your work vasuji https://www.kaggle.com/vasuji/i-covid19-nlp-data-parsing

#####################################################################################



datafiles = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        ifile = os.path.join(dirname, filename)

        if ifile.split(".")[-1] == "json":

            datafiles.append(ifile)
id2abstract = []

for file in datafiles[:10000]:

    with open(file,'r')as f:

        doc = json.load(f)

    id = doc['paper_id'] 

    abstract = ''

    try:

        for item in doc['abstract']:

            abstract = abstract + item['text']

            

        id2abstract.append({id:abstract})

    except KeyError:

        None

    

print ("finish load all paper's abstract")    
from transformers import BertForQuestionAnswering

from transformers import BertTokenizer

import torch

device = torch.device("cuda")

model_QA = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

tokenizer_QA = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

model_QA = model_QA.to(device)

model_QA = model_QA.eval()



from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

# see ``examples/summarization/bart/evaluate_cnn.py`` for a longer example

model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')

tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
def reconstructText(tokens, start=0, stop=-1):

    tokens = tokens[start: stop]

    if '[SEP]' in tokens:

        sepind = tokens.index('[SEP]')

        tokens = tokens[sepind+1:]

    txt = ' '.join(tokens)

    txt = txt.replace(' ##', '')

    txt = txt.replace('##', '')

    txt = txt.strip()

    txt = " ".join(txt.split())

    txt = txt.replace(' .', '.')

    txt = txt.replace('( ', '(')

    txt = txt.replace(' )', ')')

    txt = txt.replace(' - ', '-')

    txt_list = txt.split(' , ')

    txt = ''

    nTxtL = len(txt_list)

    if nTxtL == 1:

        return txt_list[0]

    newList =[]

    for i,t in enumerate(txt_list):

        if i < nTxtL -1:

            if t[-1].isdigit() and txt_list[i+1][0].isdigit():

                newList += [t,',']

            else:

                newList += [t, ', ']

        else:

            newList += [t]

    return ''.join(newList)



def extract_sentence(abstract,answer):

    abstract = abstract.lower()

    answer = answer.lower()

    split_byans = abstract.split(answer)

    if len(split_byans)==1:

        return split_byans[0][split_byans[0].rfind(". ")+1:]+" "+answer

    else: 

        return split_byans[0][split_byans[0].rfind(". ")+1:]+" "+answer+split_byans[1][:split_byans[1].find(". ")+1]

    

def Add_contaner(text,index,query,Summary_text,Summary_text_TW,Summary_text_CN,Summary_text_JP,Summary_text_KO,ans_pd):

    text = text.replace("demo","demo"+index).replace("en_s","en_s"+index).replace("tw_s","tw_s"+index).replace("cn_s","cn_s"+index).replace("jp_s","jp_s"+index).replace("ko_s","ko_s"+index).replace("topextract","topextract"+index)

    text = text.format(query,Summary_text,Summary_text_TW,Summary_text_CN,Summary_text_JP,Summary_text_KO,ans_pd)

    return text

def Summary_Model(pd,count,model):

    ### Top confident answer to article

    total_abstract=''

    for i in range(len(pd[:count])):

        abss, ans = pd.loc[i,['abstract_by_ans','Answer']].values

        total_abstract+=(abss+".")

    ARTICLE_TO_SUMMARIZE = total_abstract



    ### Token the article, if larger than 1024, then split the article

    tokens= tokenizer.tokenize(ARTICLE_TO_SUMMARIZE)

    max_seq_length = 1024

    longer = 0

    all_tokens=[]

    if len(tokens)>1024:

        for i in range(0,len(tokens),max_seq_length):

            tokens_a = tokens[i:i+max_seq_length]

            one_token = tokenizer.batch_encode_plus([tokens_a], max_length=1024, return_tensors='pt')

            all_tokens.append(one_token)

    

    

    Summary_text = []

    

    ## decode the model output as summary text

    def decode_text(sum_ids):

        text =''

        for g in sum_ids:

            text = text+tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return text

    

    ## Summary model

    model = model.to(device)

    model.eval()

    Summary_text = ''

    for inputs in all_tokens:

        summary_ids = model.generate(inputs['input_ids'].to(device), num_beams=2, max_length=1000, early_stopping=True)

        Summary_text = Summary_text+" "+decode_text(summary_ids) 

    

    if Summary_text == '':

        Summary_text = "Can't find summary of answer"

    #print (Summary_text)

    ## Translate to zh-TW, zh-CN, JP, KO

    translator = Translator()

    Summary_text_TW = translator.translate(Summary_text,dest='zh-tw').text

    Summary_text_CN = translator.translate(Summary_text,dest='zh-cn').text

    Summary_text_JP = translator.translate(Summary_text,dest='ja').text

    Summary_text_KO = translator.translate(Summary_text,dest='ko').text

    return Summary_text,Summary_text_TW,Summary_text_CN,Summary_text_JP,Summary_text_KO
def ANS_Model(ques,count):

    question = ques

    answer_all=[]

    cons_all = []

    ID_list = []

    abstract_list = []

    for i in range(len(id2abstract[:count])):

        document = list(id2abstract[i].values())[0]

        ID = list(id2abstract[i].keys())[0]

        nWords = len(document.split()) ## check how many tokens

        input_ids_all = tokenizer_QA.encode(question, document) ## Encode the document to ids

        tokens_all = tokenizer_QA.convert_ids_to_tokens(input_ids_all)

        if len(input_ids_all) > 512:

            nFirstPiece = int(np.ceil(nWords/2))

            docSplit = document.split()

            docPieces = [' '.join(docSplit[:nFirstPiece]), ' '.join(docSplit[nFirstPiece:])]

            input_ids = [tokenizer_QA.encode(question, dp) for dp in docPieces]

        else:

            input_ids = [input_ids_all]   



        answers = []

        cons = []

        for iptIds in input_ids:

            tokens = tokenizer_QA.convert_ids_to_tokens(iptIds)

            sep_index = iptIds.index(tokenizer_QA.sep_token_id)

            num_seg_a = sep_index + 1

            num_seg_b = len(iptIds) - num_seg_a

            segment_ids = [0]*num_seg_a + [1]*num_seg_b

            assert len(segment_ids) == len(iptIds)

            number_ids = len(segment_ids)



            if number_ids < 512:

                start_scores, end_scores = model_QA(torch.tensor([iptIds]).to(device), 

                                         token_type_ids=torch.tensor([segment_ids]).to(device))

            else:

                start_scores, end_scores = model_QA(torch.tensor([iptIds[:512]]).to(device), 

                                         token_type_ids=torch.tensor([segment_ids[:512]]).to(device))

            

            start_scores = start_scores[:,1:-1]

            end_scores = end_scores[:,1:-1]

            answer_start = torch.argmax(start_scores)

            answer_end = torch.argmax(end_scores)

            answer = reconstructText(tokens, answer_start, answer_end+2)



            if answer.startswith('. ') or answer.startswith(', '):

                answer = answer[2:]



            c = start_scores[0,answer_start].item()+end_scores[0,answer_end].item()

            answers.append(answer)

            cons.append(c)



        maxC = max(cons)

        iMaxC = [i for i, j in enumerate(cons) if j == maxC][0]

        confidence = cons[iMaxC]

        answer = answers[iMaxC]



        sep_index = tokens_all.index('[SEP]')

        full_txt_tokens = tokens_all[sep_index+1:]



        abs_returned = reconstructText(full_txt_tokens)

        if answer!="":

            answer_all.append(answer)

            cons_all.append(confidence)

            ID_list.append(ID)

            abstract_list.append(document)

            

    ans_pd = pd.DataFrame({"PaperID":ID_list,'abstract':abstract_list,"Answer":answer_all,"Confident":cons_all})

    gc.collect()

    extrac_list = []

    for i in range(len(ans_pd)):

        abss, ans = ans_pd.loc[i,['abstract','Answer']].values

        extract_sen = extract_sentence(abss,ans)

        extrac_list.append(extract_sen)

    ans_pd['abstract_by_ans'] = extrac_list

    ans_pd = ans_pd.sort_values(by=['Confident'],ascending=False).reset_index(drop=True)

    Summary_text_EN,Summary_text_TW,Summary_text_CN,Summary_text_JP,Summary_text_KO = Summary_Model(ans_pd,100,model)

    gc.collect()

    return ans_pd,Summary_text_EN,Summary_text_TW,Summary_text_CN,Summary_text_JP,Summary_text_KO
from tqdm import tqdm_notebook

all_tasks_count = 0 

def topic_all_display(topic, ques, count_papers):

    global all_tasks_count

    HTML_Question= """

    <div>

      <button type="button" class="btn btn-info" data-toggle="collapse" data-target="#demo">{}</button>

      <div id="demo" class="collapse">

        <nav class="navbar navbar-light" style="background-color: #e3f2fd;width: 350px">

            <div>

                <ul class="nav navbar-nav nav-tabs">

                    <li class="active"><a data-toggle="tab" href="#en_s">EN</a></li>

                    <li><a data-toggle="tab" href="#tw_s">ZH-TW</a></li>

                    <li><a data-toggle="tab" href="#cn_s">ZH-CN</a></li>

                    <li><a data-toggle="tab" href="#jp_s">JP</a></li>

                    <li><a data-toggle="tab" href="#ko_s">KO</a></li>

                </ul>

            </div>

        </nav>

        <h1>Summary : </h1>

        <div class="tab-content">

            <div id="en_s" class="tab-pane fade in active">

                <p>{}</p>

            </div>

            <div id="tw_s" class="tab-pane fade">

                <p>{}</p>

            </div>

            <div id="cn_s" class="tab-pane fade">

                <p>{}</p>

            </div>

            <div id="jp_s" class="tab-pane fade">

                <p>{}</p>

            </div>

            <div id="ko_s" class="tab-pane fade">

                <p>{}</p>

            </div>

        </div>

        <br>

        <button type="button" class="btn-warning" data-toggle="collapse" data-target="#topextract">Top Condfident Papers</button>

        <div id="topextract" class="collapse">

        {}

        </div>

      </div>

    </div>"""

    



    topic_header = """

    <div>

      <button type="button" class="btn" data-toggle="collapse" data-target="#topic_id" style="font-size:20px">&#8226 topic</button>

      <div id="topic_id" class="collapse">

      {}

      </div>

    </div>"""



    topic_result = ""

    for index,i in enumerate(ques):

        question = i

        ans_pd,Summary_text_EN,Summary_text_TW,Summary_text_CN,Summary_text_JP,Summary_text_KO = ANS_Model(question,len(id2abstract[:count_papers]))

        ans_pd_html = ans_pd.drop(['abstract'],axis = 1)[:10].to_html(render_links=True, escape=False)

        topic_result = topic_result+Add_contaner(HTML_Question,str(all_tasks_count),i,Summary_text_EN,Summary_text_TW,Summary_text_CN,Summary_text_JP,Summary_text_KO,ans_pd_html)

        all_tasks_count+=1

        

    topic_header = topic_header.replace("topic_id",topic.replace(" ","").replace("?","")).replace(",","").replace("topic",topic).format(topic_result)

    #HTML_String = HTML_Header+topic_header

    #display(HTML(HTML_String))

    return topic_header
task_dic={}



all_tasks=[

    'What is known about transmission, incubation, and environmental stability?',

    'What do we know about COVID-19 risk factors?',

    'What do we know about virus genetics, origin, and evolution?',

    'What do we know about vaccines and therapeutics?',

    'What has been published about medical care?',

    'What do we know about non-pharmaceutical interventions?',

    'What do we know about diagnostics and surveillance?',

    'What has been published about ethical and social science considerations?',

    'What has been published about information sharing and inter-sectoral collaboration?'

]



#1 task

question_list = []

question_list.append("What is known about transmission, incubation, and environmental stability for the 2019-nCoV")

question_list.append("What do we know about natural history, transmission, and diagnostics for the 2019-nCoV")

question_list.append("What have we learned about infection prevention and control for the 2019-nCoV")

question_list.append("What is the range of incubation periods for the 2019-nCoV in humans")

question_list.append("How does temperature and humidity affect the tramsmission of 2019-nCoV")

question_list.append("How long can 2019-nCoV remain viable on inanimate, environmental, or common surfaces")

question_list.append("What types of inanimate or environmental surfaces affect transmission, survival, or  inactivation of 2019-nCov")

question_list.append("What is tools and studies to monitor phenotypic change and potential adaptation of the virus")

task_dic[all_tasks[0]] = question_list





#2 task

question_list = []

question_list.append("What risk factors contribute to the severity of 2019-nCoV")

question_list.append("How does hypertension affect patients")

question_list.append("How does heart disease affect patients")

question_list.append("How does copd affect patients")

question_list.append("How does smoking affect 2019-nCoV patients")

question_list.append("How does pregnancy affect patients")

question_list.append("What are the case fatality rates for 2019-nCoV patients")

question_list.append("What is the case fatality rate in Italy")

question_list.append("What public health policies prevent or control the spread of 2019-nCoV")

task_dic[all_tasks[1]] = question_list



#3 task

question_list = []

question_list.append("Can animals transmit 2019-nCoV")

question_list.append("What do we know about the virus origin and management measures at the human-animal interface")

question_list.append("What animal did 2019-nCoV come from")

question_list.append("What real-time genomic tracking tools exist")

question_list.append("What regional genetic variations (mutations) exist")

question_list.append("What effors are being done in asia to prevent further outbreaks")

task_dic[all_tasks[2]] = question_list



#4 task

question_list = []

question_list.append("What do we know about vaccines and therapeutics")

question_list.append("What has been published concerning research and development and evaluation efforts of vaccines and therapeutics")

question_list.append("What drugs or therapies are being investigated")

question_list.append("What clinical trials for hydroxychloroquine have been completed")

question_list.append("What antiviral drug clinical trials have been completed")

question_list.append("Are anti-inflammatory drugs recommended")

task_dic[all_tasks[3]] = question_list





#5 task

question_list = []

question_list.append("What has been published about medical care for 2019-nCoV")

question_list.append("What has been published concerning surge capacity and nursing homes for 2019-nCoV")

question_list.append("What has been published concerning efforts to inform allocation of scarce resources for 2019-nCoV")

question_list.append("What do we know about the clinical characterization and management of the 2019-nCoV")

question_list.append("How does extracorporeal membrane oxygenation affect 2019-nCoV patients")

question_list.append("What telemedicine and cybercare methods are most effective")

question_list.append("How is artificial intelligence being used in real time health delivery")

question_list.append("What adjunctive or supportive methods can help patients")

task_dic[all_tasks[4]] = question_list



#6 task

question_list = []

question_list.append("Which non-pharmaceutical interventions limit tramsission")

question_list.append("What are most important barriers to compliance")

task_dic[all_tasks[5]] = question_list



#7 task

question_list = []

question_list.append("What diagnostic tests (tools) exist or are being developed to detect 2019-nCoV")

question_list.append("What is being done to increase testing capacity or throughput")

question_list.append("What point of care tests are exist or are being developed")

question_list.append("What is the minimum viral load for detection")

question_list.append("What markers are used to detect or track COVID-19")

task_dic[all_tasks[6]] = question_list





#8 task

question_list = []

question_list.append('What collaborations are happening within the research community')

task_dic[all_tasks[7]] = question_list



#9 task

question_list = []

question_list.append("What are the major ethical issues related pandemic outbreaks")

question_list.append("How do pandemics affect the physical and/or psychological health of doctors and nurses")

question_list.append("What strategies can help doctors and nurses cope with stress in a pandemic")

question_list.append("What factors contribute to rumors and misinformation")

task_dic[all_tasks[8]] = question_list

#ques = ["What do we know about COVID19 risk factors","What is COVID19"]

#ans_pd,Summary_text_EN,Summary_text_TW,Summary_text_CN,Summary_text_JP,Summary_text_KO = ANS_Model(ques,len(id2abstract[:100]))
HTML_Header="""

    <!DOCTYPE html>

    <html>

    <head>

      <meta name="viewport" content="width=device-width, initial-scale=1">

      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">

      <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>

      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/js/bootstrap.min.js"></script>

    </head>

    <body> """



topic_body = ""

for i in task_dic:

    task = i

    question_list = task_dic[task]

    topic_body += topic_all_display(task,question_list,3000) # Only read 3000 papers to in the kaggle run-time 

All_HTML = HTML_Header + topic_body

display(HTML(All_HTML))