MODEL_URL = "https://github.com/huggingface/neuralcoref-models/releases/" \
            "download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz"
!pip install spacy==2.0.12
!pip install {MODEL_URL}
!python -m spacy download en_core_web_md
import en_coref_md

nlp = en_coref_md.load()

test_sent = '''
What are the main breeds of goat? Tell me about boer goats. What breed is good for meat? Are angora goats good for it? What about boer goats? What are pygmies used for? What is the best for fiber production? How long do Angora goats live? Can you milk them? How many can you have per acre? Are Angora goats profitable?
'''
test_list = list(test_sent.split(" ")) 
print(test_list)
doc = nlp(test_sent)
from dataclasses import dataclass
from IPython.core.display import display, HTML
import pandas as pd
import re
i = 0
control = 0
myDict = {} 
@dataclass
class Question:
    questionId:int = 0
    title: str = ""
    questionText: str = ""
with open('../input/questions/only_questions.txt') as f:
    lines = [line.rstrip() for line in f]
ques = Question()
questionList = []
headList = []
for line_number in range(len(lines)):
    
    lineList = lines[line_number].split(":")

    ques.title = lineList[0]
    ques.questionText = lineList[1]
    headList.append(str(lineList[0]))
    #questionList.append(ques)
    #print(ques.questionText)
    doc = nlp(ques.questionText)
    test_list = list(ques.questionText.split(" "))
    
    if doc._.has_coref is True:
        for i in range(len(doc._.coref_clusters)):
            for j in range(len(doc._.coref_clusters[i])):
                for n, k in enumerate(test_list):
                    if k == str(doc._.coref_clusters[i].mentions[j]) and control<len(doc._.coref_clusters[i].mentions):
                        test_list[n] = str(doc._.coref_clusters[i].main)
                        control = control + 1
    
                control = 0
    str1 = ' '.join(test_list)

   
    res = re.split('\?', str1)
    myDict[lineList[0]] = [res] 
print(myDict)     

import json

# as requested in comment
Dict = {'exDict': myDict}

with open('file2.txt', 'w') as file:
     file.write(json.dumps(myDict))            



doc._.has_coref
doc._.coref_clusters