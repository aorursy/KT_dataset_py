import numpy as np # linear algebra

import os

import pandas as pd
from tqdm import tqdm
print(os.listdir("../input"))
def josn_DF():
    import json
    with open("../input/train-v1.1.json") as input_file:    
        json_data = json.load(input_file)
        
    context = pd.DataFrame(columns=['id', 'context'])
    Question = pd.DataFrame(columns=['id', 'cID', 'Question'])
    Answer = pd.DataFrame(columns=['id','qID','cID','Answer'])  
    
    context_no_tracker = 0
    Question_no_tracker = 0
    Answer_no_tracker = 0
    json_layer1 = json_data['data']
    for i in tqdm(json_layer1):
        json_layer2 = i['paragraphs']
        for context_no_tracker,j in enumerate(json_layer2,context_no_tracker):
            context.loc[context_no_tracker] = [context_no_tracker,j['context']]
            for Question_no_tracker,k in enumerate(j['qas'],Question_no_tracker):
                Question.loc[Question_no_tracker] = [Question_no_tracker,context_no_tracker,k['question']]
                for  Answer_no_tracker,l in enumerate(k['answers'],Answer_no_tracker):
                    Answer.loc[Answer_no_tracker] = [Answer_no_tracker,Question_no_tracker,context_no_tracker,l['text']]
            
    return(context,Question,Answer)
context,Question,Answer = josn_DF()
context.head()
Question.head()
Answer.head()

