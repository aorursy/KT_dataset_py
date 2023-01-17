# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import time

import torch

from transformers import T5ForConditionalGeneration,T5Tokenizer





def set_seed(seed):

  torch.manual_seed(seed)

  if torch.cuda.is_available():

    torch.cuda.manual_seed_all(seed)



set_seed(42)



model = T5ForConditionalGeneration.from_pretrained('../input/question-generation-multiple/result')

tokenizer = T5Tokenizer.from_pretrained('t5-base')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print ("device ",device)

model = model.to(device)
text = '''Coronaviruses were discovered in the 1960s. They are a group of viruses that cause diseases in mammals and birds. In humans, coronaviruses cause respiratory tract infections that are typically mild, such as the common cold.



The name "coronavirus" is derived from the Latin corona, meaning crown or halo. The name refers to the characteristic appearance of the infective form of the virus, which is reminiscent of a crown or a solar corona.'''

answer = '''coronavirus'''
con =  "context:%s answer:%s</s>" %(text,answer)
# input_ = "passage:%s \n input:%s</s>" % (context,answer)
max_len = 40


encoding = tokenizer.encode_plus(con, return_tensors="pt")

input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

def greedy_decoding (inp_ids,attn_mask):

    greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=80)

    Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)

    return Question.strip().capitalize()
print ("passage: "+text+"\n"+"answer: "+answer)

# print ("\nGenerated Question: ",truefalse)



output = greedy_decoding(input_ids,attention_masks)

print (output)




def beam_search_decoding (inp_ids,attn_mask):

  beam_output = model.generate(input_ids=inp_ids,

                                 attention_mask=attn_mask,

                                 max_length=80,

                               num_beams=10,

                               num_return_sequences=2,

                               no_repeat_ngram_size=2,

                               early_stopping=True

                               )

  Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in

               beam_output]

#   return Question.strip().capitalize() for Question in Questions

  return  Questions







def topkp_decoding (inp_ids,attn_mask):

  topkp_output = model.generate(input_ids=inp_ids,

                                 attention_mask=attn_mask,

                                 max_length=80,

                               do_sample=True,

                               top_k=20,

                               top_p=0.80,

                               num_return_sequences=3,

                                no_repeat_ngram_size=2,

                                early_stopping=True

                               )

  Questions = [tokenizer.decode(out, skip_special_tokens=True,clean_up_tokenization_spaces=True) for out in topkp_output]

  return list(Question.strip().capitalize() for Question in Questions)



output = beam_search_decoding(input_ids,attention_masks)

print ("\nBeam decoding [Most accurate questions] ::\n")

for out in output:

    print(out)





# output = topkp_decoding(input_ids,attention_masks)

# print ("\nTopKP decoding [Not very accurate but more variety in questions] ::\n")

# for out in output:

#     print (out)

output = topkp_decoding(input_ids,attention_masks)

print ("\nTopKP decoding [Not very accurate but more variety in questions] ::\n")

for out in output:

    print (out)

def t5_answer(text,answer):

    con = "context:%s answer:%s</s>" %(text,answer)

    encoding = tokenizer.encode_plus(con, return_tensors="pt")

    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    output = beam_search_decoding(input_ids, attention_masks)

    return str(output)

text = '''Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people who fall sick with COVID-19 will experience mild to moderate symptoms and recover without special treatment. Coronavirus is mainly transmitted through droplets generated when an infected person coughs, sneezes, or exhales. The Victorian Government announced Stay at Home Directions for metropolitan Melbourne and Mitchell Shire from 11:59pm on Wednesday 8 July until 11.59pm on Wednesday 19 August. There are 4 reasons people in metropolitan Melbourne and the Mitchell Shire can leave home. The reasons to leave home can be either shopping, work or study, medical treatment or for exercise. The New South Wales border with Victoria closed at 12:01am on Wednesday 8 July 2020. There is a permit system in place for people who need to travel from Victoria into NSW. the Victorian Government has announced that if you live in metropolitan Melbourne or Mitchell Shire, you must wear a face covering when you leave your home from 11.59pm on Wednesday 22 July 2020. the Victorian Government has announced that metropolitan Melbourne and Mitchell Shire will return to Stage 3 Stay at Home restrictions from 11.59pm on Wednesday 8 July 2020. All Year 11 and Year 12 students will go back to school for Term 3 as planned from 13 July, along with students at special schools. Prep to Year 10 students will return to school on Monday 20 July through remote and flexible learning. The JobKeeper Payment scheme is a temporary subsidy for businesses significantly affected by coronavirus. Eligible employers, sole traders and other entities affected by coronavirus can apply to receive $1,500 per eligible employee per fortnight.'''
text = '''

Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus.

Most people who fall sick with COVID-19 will experience mild to moderate symptoms and recover without special treatment. 

Coronavirus is mainly transmitted through droplets generated when an infected person coughs, sneezes, or exhales. 

The Victorian Government announced Stay at Home Directions for metropolitan Melbourne and Mitchell Shire from 11:59pm on Wednesday 8 July until 11.59pm on Wednesday 19 August. 

There are 4 reasons people in metropolitan Melbourne and the Mitchell Shire can leave home.

The reasons to leave home can be either shopping, work or study, medical treatment or for exercise.

The New South Wales border with Victoria closed at 12:01am on Wednesday 8 July 2020. There is a permit system in place for people who need to travel from Victoria into NSW. 

the Victorian Government has announced that if you live in metropolitan Melbourne or Mitchell Shire, you must wear a face covering when you leave your home from 11.59pm on Wednesday 22 July 2020. 

the Victorian Government has announced that metropolitan Melbourne and Mitchell Shire will return to Stage 3 Stay at Home restrictions from 11.59pm on Wednesday 8 July 2020.

All Year 11 and Year 12 students will go back to school for Term 3 as planned from 13 July, along with students at special schools.

Prep to Year 10 students will return to school on Monday 20 July through remote and flexible learning. 

The JobKeeper Payment scheme is a temporary subsidy for businesses significantly affected by coronavirus. Eligible employers, 

sole traders and other entities affected by coronavirus can apply to receive $1,500 per eligible employee per fortnight.

On 21 July the government announced an extension of Job Keeper through to 28 March 2021.

The Jobkeeper payment scheme runs for the fortnights from 30 March until 27 September 2020. 

Eligible employees need to make a business monthly declaration to the ATO. Business can enrol

any time within this period to claim JobKeeper. Sole traders may be eligible for the JobKeeper

scheme under the business participation entitlement if their business has experienced a downturn 

according to the eligibility criteria. From 20 July, approved providers of child care

servicesExternal Link cannot claim JobKeeper payments for themselves or their employees

whose ordinary duties relate principally to the operation of the child care service.

JobSeeker Payment is a financial help offered to people aged between 22 and pension age

and looking for work. JobSeeker payment is offered to sick or injured people. Jobseeker

payment is for every 2 weeks. A single with no children gets $1115.70 per fortnight as 

JobSeeker payment. A single with a dependent child gets $1,162.00 per fortnight as JobSeeker 

payment. Contact number for additional information on Job Seeker payment is 136240. 

Centrelink phone service is available 7 days a week, 24 hours a day. '''
answers = ["Coronavirus disease","Wednesday 8 July 2020","remote and flexible learning.","receive $1,500 per eligible employee","30 March until 27 September 2020.","business monthly declaration to the ATO.","Jobkeeper payment scheme","$1115.70 per fortnight","A single with a dependent child gets $1,162.00 per","Centrelink phone service","symptoms","136240"]





def beam_search_decoding (inp_ids,attn_mask):

        beam_output = model.generate(input_ids=inp_ids,

                                         attention_mask=attn_mask,

                                         max_length=50,

                                       num_beams=10,

                                       num_return_sequences=4,

                                       no_repeat_ngram_size=2,

                                       early_stopping=True

                                       )

        Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in beam_output]

        return [Question.strip().capitalize() for Question in Questions]

        

#         res = max(Questions, key = len) 

          

#         return  res





import ast 

import gc

gc.collect()
for ans in answers:

    output = t5_answer(text,ans)

    res = ast.literal_eval(output) 

    longest_string = max(res, key=len)

    print(ans)

    print(res)
text = "He ran because he was in hurry"

answer = "ran"

t5_answer(text,answer)
text = "Christmas is on 25th december every year"

answer = "25th december"

t5_answer(text,answer)
text = "New Delhi is located in India"

answer = "New Delhi"

t5_answer(text,answer)
text = "Donald Trump is the president of USA"

answer = "Donald Trump"

t5_answer(text,answer)
text = '''Tim is a skilled Software Testing professional. Tim's experience includes test team management, client and vendor 

management, automation test engineering, test infrastructure creation and maintenance. 

Tim is an expert in building testing teams from scratch. Tim puts together processes, strategies, tools and frameworks, 

skilled resources for efficient and seamless delivery. 

Expertise in designing and developing test strategy, test plan, test cases and generating test reports and defect reports.

Extensive experience in coordinating testing effort, responsible for test deliverables,

status reporting to management, issue escalations etc. Have Extensive experience in automating and 

testing enterprise web applications using Selenium 1.0 and Selenium 2.0 (WebDriver) tool using both Microsoft and JAVA technology stack.

Tim is very good at JAVA, C# and python language programming. Tim has strong experience in C# programming using MS Visual Studio and

other MS technology stack in particular MSTest unit testing framework. Tim has very good experience in setting up Continuous Integration

environment using Team Foundation Server, creating Build definitions and management etc. Domain experience includes Superannuation, 

Telco, Insurance, Banking, Finance and healthcare domains. Tim's linkedin profile is https://www.linkedin.com/in/timothy-r-alex-ai/.

Tim's contact phone number is 0470139767. Tim's email address is timothyrajan@gmail.com. 

Tim is has good exposure to gitlab and github hosting platforms.'''
!pip install git+https://github.com/boudinfl/pke.git

!python -m nltk.downloader stopwords

!python -m nltk.downloader universal_tagset

!python -m spacy download en # download the english model
from pke.unsupervised import TopicRank

def extract_keyword(text):

    extractor = TopicRank()



    # load the content of the document, here in CoreNLP XML format

    # the input language is set to English (used for the stoplist)

    # normalization is set to stemming (computed with Porter's stemming algorithm)

    extractor.load_document(input=text,

                            language="en")



    # select the keyphrase candidates, for TopicRank the longest sequences of 

    # nouns and adjectives

    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})



    # weight the candidates using a random walk. The threshold parameter sets the

    # minimum similarity for clustering, and the method parameter defines the 

    # linkage method

    extractor.candidate_weighting(threshold=0.99,

                                  method='average')

    keyphrases = []



    # print the n-highest (10) scored candidates

    for (keyphrase, score) in extractor.get_n_best(n=3):

        keyphrases.append(keyphrase)

    return keyphrases
import pke

import string

from nltk.corpus import stopwords



 # 1. create a TopicRank extractor.

extractor = pke.unsupervised.TopicRank()



 # 2. load the content of the document.

extractor.load_document(input=text)



 # 3. select the longest sequences of nouns and adjectives, that do

 #    not contain punctuation marks or stopwords as candidates.

pos = {'NOUN', 'PROPN', 'ADJ'}

stoplist = list(string.punctuation)

stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

stoplist += stopwords.words('english')

extractor.candidate_selection(pos=pos, stoplist=stoplist)



 # 4. build topics by grouping candidates with HAC (average linkage,

 #    threshold of 1/4 of shared stems). Weight the topics using random

 #    walk, and select the first occuring candidate from each topic.

extractor.candidate_weighting(threshold=0.99, method='average')



 # 5. get the 10-highest scored candidates as keyphrases

keyphrases = extractor.get_n_best(n=50)

for (keyphrase, score) in keyphrases:

    print(keyphrase)
import random

random.randint(1,4)
def Multiple_questions(keyphrases,text):

    for ans in keyphrases:

        output = t5_answer(text,ans)

        res = ast.literal_eval(output) 

        longest = min(res, key=len)

        quest = longest.lstrip('Question:')

        print(quest)

        

#         print(ans)

keyphrases = extract_keyword(text)
Multiple_questions(keyphrases,text)
import gc

gc.collect()
from nltk import tokenize
text = '''Tim is a skilled Software Testing professional. Tim's experience includes test team management, client and vendor 

management, automation test engineering, test infrastructure creation and maintenance. 

Tim is an expert in building testing teams from scratch. Tim puts together processes, strategies, tools and frameworks, 

skilled resources for efficient and seamless delivery. 

Expertise in designing and developing test strategy, test plan, test cases and generating test reports and defect reports.

Extensive experience in coordinating testing effort, responsible for test deliverables,

status reporting to management, issue escalations etc. Have Extensive experience in automating and 

testing enterprise web applications using Selenium 1.0 and Selenium 2.0 (WebDriver) tool using both Microsoft and JAVA technology stack.

Tim is very good at JAVA, C# and python language programming. Tim has strong experience in C# programming using MS Visual Studio and

other MS technology stack in particular MSTest unit testing framework. Tim has very good experience in setting up Continuous Integration

environment using Team Foundation Server, creating Build definitions and management etc. Domain experience includes Superannuation, 

Telco, Insurance, Banking, Finance and healthcare domains. Tim's linkedin profile is https://www.linkedin.com/in/timothy-r-alex-ai/.

Tim's contact phone number is 0470139767. Tim's email address is timothyrajan@gmail.com. 

Tim is has good exposure to gitlab and github hosting platforms.'''
def para_to_sent(para):

    keyphrases = tokenize.sent_tokenize(para)

    return keyphrases
def substring(whole, sub1, sub2):

    return whole[whole.index(sub1) : whole.index(sub2)]
extract_keyword(text)
def substring_before(s, delim):

    return s
def extract_span(text):

    l = max(extract_keyword(text))

    if len(l.split())>1:

        a = substring_before(text, l)

    else:

         a = substring_before(text, l)+l[0]

    return a
extract_span(text)
token_sent = para_to_sent(text)
token_sent[:]
text = "Tim's linkedin profile is https://www.linkedin.com/in/"

text.split()
def heavenspan(text,keyword):

    l = text.split()

    keyword = list(keyword.split())

    keyword = keyword[0]

    k = l.index(keyword)

    span = keyword

    if (len(l)>4):

        if k == 0 or k == 1:

            span = keyword+" "+l[k+1]+" "+l[k+2]+" "+l[k+3]

        elif k == (len(l)-1) or k== (len(l)-2):

            span =l[k-2]+" "+l[k-1]+" "+keyword

        else:

            span = l[k-1]+" "+keyword+" "+l[k+1]+" "+l[k+2]

    return span

    
heavenspan(text,"https://www.linkedin.com/in/")
def final_question_recipe(text):

    token_sent = para_to_sent(text)

    keywords = []

    i = 0

#     print(i)

    for x in token_sent:

        k = heavenspan(x,extract_span(x))

        keywords.append(k)

    print(keywords)

    Multiple_questions(keywords,text)

        

    
text = '''Tim is a skilled Software Testing professional. Tim's experience includes test team management, client and vendor 

management, automation test engineering, test infrastructure creation and maintenance. 

Tim is an expert in building testing teams from scratch. Tim puts together processes, strategies, tools and frameworks, 

skilled resources for efficient and seamless delivery. 

Expertise in designing and developing test strategy, test plan, test cases and generating test reports and defect reports.

Extensive experience in coordinating testing effort, responsible for test deliverables,

status reporting to management, issue escalations etc. Have Extensive experience in automating and 

testing enterprise web applications using Selenium 1.0 and Selenium 2.0 (WebDriver) tool using both Microsoft and JAVA technology stack.

Tim is very good at JAVA, C# and python language programming. Tim has strong experience in C# programming using MS Visual Studio and

other MS technology stack in particular MSTest unit testing framework. Tim has very good experience in setting up Continuous Integration

environment using Team Foundation Server, creating Build definitions and management etc. Domain experience includes Superannuation, 

Telco, Insurance, Banking, Finance and healthcare domains. Tim's linkedin profile is https://www.linkedin.com/in/timothy-r-alex-ai/.

Tim's contact phone number is 0470139767. Tim's email address is timothyrajan@gmail.com. 

Tim has good exposure to gitlab and github hosting platforms.'''
final_question_recipe(text)