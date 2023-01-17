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
import ast

def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('../input/ms-marco-the-better')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print ("device ",device)
model = model.to(device)
def beam_search_decoding (inp_ids,attn_mask):
        beam_output = model.generate(input_ids=inp_ids,
                                         attention_mask=attn_mask,
                                         max_length=400,
                                       num_beams=70,
                                       num_return_sequences=7,
                                       no_repeat_ngram_size=3,
                                       early_stopping=True
                                       )
        Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in beam_output]
        return [Question.strip().capitalize() for Question in Questions]
def t5_answer(context,question):
    con = "question: %s context: %s </s>" %(question,context)
    encoding = tokenizer.encode_plus(con, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    output = beam_search_decoding (input_ids, attention_masks)
    return (output)
text = '''Essay Topic: Learning at Home during Lockdown:
My Parents and My Teachers

Ever since the lockdown started, I feel lonely at home.
I do have a brother but soon realized that talking to a person
or doing the same thing consistently can get monotonous.
Sometimes, I even feel that it would be better to go to school,
which a month-back I could not have thought of in a million
years.At my house, both my parents are doctors. Not that they do
not have holidays, they do! Somehow, the holidays do not
seem enough.My parents are treating COVID-19 patients and often discuss their healthcare. 
At times, I nd their conversations scary and mom calms me down by saying this will end soon.
Yet, I am hardly convinced with her explanations. In the little time that I get to talk to 
my friends, we discuss the current situation due to pandemic and its advantages, especially 
on the environment, as us human beings are in lockdown.
A few days ago, when my father and I were sitting in the balcony at night I looked up in the
sky and saw a lot more stars than I usually get to see. Even my mom told me that Yamuna
river is getting cleaner amidst the lockdown.
I also feel that my friends have their parents at home, spending quality time with them and
all having fun times, together. While they have fun, my parents are at the hospital treating
patients and, of course, this is something that makes me very proud. Still, it is not the same
as having them at home.
However, the advantage of not having parents at home is that I do not have to do any work
until they are back. A few weeks ago, I panicked thinking that I would not get to celebrate my
birthday on its due date, just as it was not celebrated the previous three consecutive years on
the birthday day, since my parents were busy treating patients of either typhoid, pneumonia
or dengue. A sigh of relief, this year it does not matter that much as long as my family and I
are safe.
I am also anxious about school; I hope that they do not take away our summer holidays to
make up for the missed school days. I always enjoyed attending Bharatanatyam dance
classes but now, due to the lockdown, we have these classes on Zoom, which I can only
imagine, must be hard for the teacher as she tries to make it look eortless. These classes,
on the other hand, do us some good, as we do not get to copy someone if we need to.
On weekdays the school gives us work, which I sometimes nd overwhelming, but it is
more work on their side, so that is impressive. Another thing I like is the kind of eort the
teachers are making to teach us by newer methods like making videos of concepts and
even dance steps, so hats o to them for that!
On days when we do have homework, my parents when home check it, which is good
because after the tiring day at work they still spend time with us.
Out of the many things I have learned during the lockdown, one main thing is that my parents
keep reminding through their example that we should keep hope and stay positive. 
'''
question = "What is my parents profession?"
t5_answer(text,question)

question = "What is one main thing I have learned during lockdown?"
t5_answer(text,question)

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
question= ['What type of professional is Tim?',
'What type of experience does Tim have?',
'What is Tim an expert in doing?',
'What does Tim put together for efficient and seamless delivery?',
"What is Tim's expertise in?",
"How does Tim automate and test enterprise web applications?",
"What language programming is Tim good at?",
"Does Tim have strong experience in C# programming?",
"What does Tim have very good experience in setting up?",
"What type of domain experience does Tim have?",
"Where is Tim's linkedin profile located?",
"What is Tim's contact number?",
"What is the email address of Tim?",
"What hosting platforms does Tim have good exposure to?"]
def Qpair(text,question):
    output = t5_answer(text,question)
    print (question)
    print (str(max(output)))
for ques in question:
    Qpair(text,ques)
text = '''Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people who fall sick with COVID-19 will experience mild to moderate symptoms and recover without special treatment. Coronavirus is mainly transmitted through droplets generated when an infected person coughs, sneezes, or exhales. The Victorian Government announced Stay at Home Directions for metropolitan Melbourne and Mitchell Shire from 11:59pm on Wednesday 8 July until 11.59pm on Wednesday 19 August. There are 4 reasons people in metropolitan Melbourne and the Mitchell Shire can leave home. The reasons to leave home can be either shopping, work or study, medical treatment or for exercise. The New South Wales border with Victoria closed at 12:01am on Wednesday 8 July 2020. There is a permit system in place for people who need to travel from Victoria into NSW. the Victorian Government has announced that if you live in metropolitan Melbourne or Mitchell Shire, you must wear a face covering when you leave your home from 11.59pm on Wednesday 22 July 2020. the Victorian Government has announced that metropolitan Melbourne and Mitchell Shire will return to Stage 3 Stay at Home restrictions from 11.59pm on Wednesday 8 July 2020. All Year 11 and Year 12 students will go back to school for Term 3 as planned from 13 July, along with students at special schools. Prep to Year 10 students will return to school on Monday 20 July through remote and flexible learning. The JobKeeper Payment scheme is a temporary subsidy for businesses significantly affected by coronavirus. Eligible employers, sole traders and other entities affected by coronavirus can apply to receive $1,500 per eligible employee per fortnight.'''
question = 'What are the reasons behind people leaving their homes in Melbourne?'

t5_answer(text,question)

def Qpair(text,question):
    output = t5_answer(text,question)
    print (question)
    print (str(max(output)))
text = ''' The American Revolutionary War (1775–1783), also known as the American War of 
Independence, was initiated by the thirteen original colonies in Congress against the Kingdom 
of Great Britain over their objection to Parliament's direct taxation and its lack of colonial
representation. The overthrow of British rule established the United States of America as the 
first republic in modern history extending over a large territory. 
The primary reasons for the American revolution were 1. The Stamp Act 2. The Townshend Acts
3. The Boston Massacre 4. The Boston Tea Party 5. The Coercive Acts 6. Lexington and Concord 
7. British attacks on coastal towns'''
question = "What was the first republic in modern history?"
Qpair(text,question)
question = "What were the primary reasons were there for the American Revolution?"
Qpair(text,question)
text = "Hey I am 7 and I love to play Football."
question = "What is my age?"
Qpair(text,question)
text = ''' The American Revolutionary War (1775–1783), also known as the American War of 
Independence, was initiated by the thirteen original colonies in Congress against the Kingdom 
of Great Britain over their objection to Parliament's direct taxation and its lack of colonial
representation. The overthrow of British rule established the United States of America as the 
first republic in modern history extending over a large territory.The primary reasons for the American revolution were 1. The Stamp Act 2. The Townshend Acts
3. The Boston Massacre 4. The Boston Tea Party 5. The Coercive Acts 6. Lexington and Concord 
7. British attacks on coastal towns'''
question = "What were primary reasons for the American Revolution?"
Qpair(text,question)
