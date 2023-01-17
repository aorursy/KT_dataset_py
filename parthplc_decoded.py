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



model = T5ForConditionalGeneration.from_pretrained('/kaggle/input/answergen/')

tokenizer = T5Tokenizer.from_pretrained('t5-base')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print ("device ",device)

model = model.to(device)
text = '''Coronaviruses were discovered in the 1960s. They are a group of viruses that cause diseases in mammals and birds. In humans, coronaviruses cause respiratory tract infections that are typically mild, such as the common cold.



The name "coronavirus" is derived from the Latin corona, meaning crown or halo. The name refers to the characteristic appearance of the infective form of the virus, which is reminiscent of a crown or a solar corona.'''

question = '''What is coronavirus?'''
con = "context:%s \n question:%s </s>" %(text,question)
max_len = 256


encoding = tokenizer.encode_plus(con, return_tensors="pt")

input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

def greedy_decoding (inp_ids,attn_mask):

    greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=10)

    Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)

    return Question.strip().capitalize()
print (con)

# print ("\nGenerated Question: ",truefalse)



output = greedy_decoding(input_ids,attention_masks)

print (output)




def beam_search_decoding (inp_ids,attn_mask):

  beam_output = model.generate(input_ids=inp_ids,

                                 attention_mask=attn_mask,

                                 max_length=256,

                               num_beams=20,

                               num_return_sequences=3,

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

                                 max_length=256,

                               do_sample=True,

                               top_k=40,

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

def t5_answer(context,question):

    con = "context:%s \n question:%s </s>" %(context,question)

    encoding = tokenizer.encode_plus(con, return_tensors="pt")

    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    output = beam_search_decoding (input_ids, attention_masks)

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
question = ["What is the name of the disease?","What disease is it caused by?","What kind of disease is it?","What is the cause of the disease?","What happens to most people who fall sick with Covid- 19?","What symptoms will most people who fall sick with Covid- 19 experience?","What is the name of the disease?","What type of sickness will most people who fall ill with Covid- 19 experience?","What type of symptoms will most people experience?","What are symptoms of Covid- 19?"]





def beam_search_decoding (inp_ids,attn_mask):

        beam_output = model.generate(input_ids=inp_ids,

                                         attention_mask=attn_mask,

                                         max_length=256,

                                       num_beams=50,

                                       num_return_sequences=2,

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
for ques in question:

    output = t5_answer(text,ques)

    res = ast.literal_eval(output) 

    longest_string = max(res, key=len)



    print(ques)

    print(longest_string)
quest2 = ["What is the permit system in place for?","Where do people need to travel to get to?","What is the name of the system that allows people to travel from Victoria into NSW?","What is in place for people who need to travel from Victoria into NSW?"]
for ques in quest2:

    output = t5_answer(text,ques)

    res = ast.literal_eval(output) 

    longest_string = max(res, key=len)



    print(ques)

    print(longest_string)
learning = ["What is the purpose of the program?","What kind of learning is that?","What kind of learning will they be returning to?","How will students go to school?","What kind of learning will be available on the new school year?"]
text = "All Year 11 and Year 12 students will go back to school for Term 3 as planned from 13 July, along with students at special schools.Prep to Year 10 students will return to school on Monday 20 July through remote and flexible learning."
for ques in learning:

    output = t5_answer(text,ques)

    res = ast.literal_eval(output) 

    longest_string = max(res, key=len)

    print(ques)

    print(longest_string)
declaration = ["What does the employer need to make a business monthly declaration to?","What type of employees need to make a business monthly declaration?","What type of declaration must employees make?","How often must employees make a business declaration?","What do employees need to make a business monthly declaration to?","What must employees make to the Ato?"]
text = '''The JobKeeper Payment scheme is a temporary subsidy for businesses significantly affected by coronavirus. Eligible employers, 

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
for ques in declaration:

    output = t5_answer(text,ques)

    res = ast.literal_eval(output) 

    longest_string = max(res, key=len)

    print(ques)

    print(longest_string)