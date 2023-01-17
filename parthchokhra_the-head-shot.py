import numpy as np # linear algebra
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import ast

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('../input/ms-marco')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)
def beam_search_decoding (inp_ids,attn_mask):
        beam_output = model.generate(input_ids=inp_ids,
                                         attention_mask=attn_mask,
                                         max_length=500,
                                       num_beams=50,
                                       num_return_sequences=2,
                                       no_repeat_ngram_size=2,
                                       early_stopping=True
                                       )
        Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in beam_output]
        return [Question.strip().capitalize() for Question in Questions]
def t5_answer(context,question):
    con = "question: %s context: %s </s>" %(context,question)
    encoding = tokenizer.encode_plus(con, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    output = beam_search_decoding (input_ids, attention_masks)
    return str(output)
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
question1= ['Does Tim has good exposure to gitlab and github hosting platforms ?',
'what is tims profession?',
'what is tims linkedin profile?',
'what is tims phone number?',
'what is tims email address?',
'what is tims exposure to gitlab?',
'what are some tims domain experience?',
'what is tims experience?',
'does tim build testing teams from scratch?',
'what does tim do?',
'what is tims job?',
'what experience does tim have?',
'does tim have experience in automating and testing enterprise web applications?',
'Which programming languages does tim know?',
'does tim have experience in c# programming?',
'what experience does tim have in setting up continuous integration environment using team foundation server?'
]
question= ['What type of professional is Tim?',
'What type of experience does Tim have?',
'What is Tim an expert in doing?',
'What does Tim put together for efficient and seamless delivery?',
'What is Tims expertise in?',
'What does Tim have extensive experience with?',
'How does Tim automate and test enterprise web applications?',
'What language programming is Tim good at?',
'Does Tim have strong experience in C# programming?',
'What does Tim have very good experience in setting up?',
'What type of domain experience does Tim have?',
'Where is Tims linkedin profile located?',
'What is Tims contact number?',
'What is the email address of Tim?',
'Which hosting platforms does Tim have good exposure to?']
for ques in question:
    output = t5_answer(text,ques)
    res = ast.literal_eval(output) 
    longest_string = max(res, key=len)

    print(ques)
    print(longest_string)
    print('\n')
for ques in question1:
    output = t5_answer(text,ques)
    res = ast.literal_eval(output) 
    longest_string = max(res, key=len)

    print(ques)
    print(longest_string)
text = '''Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people who fall sick with COVID-19 will experience mild to moderate symptoms and recover without special treatment. Coronavirus is mainly transmitted through droplets generated when an infected person coughs, sneezes, or exhales. The Victorian Government announced Stay at Home Directions for metropolitan Melbourne and Mitchell Shire from 11:59pm on Wednesday 8 July until 11.59pm on Wednesday 19 August. There are 4 reasons people in metropolitan Melbourne and the Mitchell Shire can leave home. The reasons to leave home can be either shopping, work or study, medical treatment or for exercise. The New South Wales border with Victoria closed at 12:01am on Wednesday 8 July 2020. There is a permit system in place for people who need to travel from Victoria into NSW. the Victorian Government has announced that if you live in metropolitan Melbourne or Mitchell Shire, you must wear a face covering when you leave your home from 11.59pm on Wednesday 22 July 2020. the Victorian Government has announced that metropolitan Melbourne and Mitchell Shire will return to Stage 3 Stay at Home restrictions from 11.59pm on Wednesday 8 July 2020. All Year 11 and Year 12 students will go back to school for Term 3 as planned from 13 July, along with students at special schools. Prep to Year 10 students will return to school on Monday 20 July through remote and flexible learning. The JobKeeper Payment scheme is a temporary subsidy for businesses significantly affected by coronavirus. Eligible employers, sole traders and other entities affected by coronavirus can apply to receive $1,500 per eligible employee per fortnight.'''
question = 'What are the reasons behind people leaving their homes in Melbourne?'
con = " question: %s context: %s </s>" %(text,question)
encoding = tokenizer.encode_plus(con, return_tensors="pt")
input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
def greedy_decoding (inp_ids,attn_mask):
    greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=50)
    Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
    return Question.strip().capitalize()
output = greedy_decoding(input_ids,attention_masks)
print (question)
print (output)