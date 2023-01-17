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
import torch

from transformers import T5ForConditionalGeneration,T5Tokenizer





def set_seed(seed):

  torch.manual_seed(seed)

  if torch.cuda.is_available():

    torch.cuda.manual_seed_all(seed)



set_seed(42)



model = T5ForConditionalGeneration.from_pretrained('../input/t5-tuning-for-paraphrasing-questions/result/')

tokenizer = T5Tokenizer.from_pretrained('t5-base')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print ("device ",device)

model = model.to(device)
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
truefalse= 'True'
answer= ['Tim','Java','Testing professional','linkedin']
con1 = "context: %s answer: %s</s>" %(text, answer)
con = "truefalse: %s passage: %s </s>" % (text, truefalse)
con2= 'ParaphraseQuestion: %s </s>' % ('Can you tell me the right way please?')
encoding = tokenizer.encode_plus(con, return_tensors="pt")

input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
def greedy_decoding (inp_ids,attn_mask):

    greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=40)

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

                                   num_beams=30,

                                   num_return_sequences=6,

                                   no_repeat_ngram_size=2,

                                   early_stopping=True

                                   )

      Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in

                   beam_output]

      return [Question.strip().capitalize() for Question in Questions]

      #return  Questions







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
output = topkp_decoding(input_ids,attention_masks)

print ("\nTopKP decoding [Not very accurate but more variety in questions] ::\n")

for out in output:

    print (out)
import ast

import gc

gc.collect()
def t5_answer(t,a):

    con = "context: %s answer: %s</s>" % (t,a)

    encoding = tokenizer.encode_plus(con, return_tensors="pt")

    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    output = beam_search_decoding (input_ids, attention_masks)

    return str(output)
t5_answer(text, answer)
for ans in answer:

    output = t5_answer(text,ans)

    res = ast.literal_eval(output) 

    longest_string = max(res, key=len)



    print(ans)

    print(longest_string)