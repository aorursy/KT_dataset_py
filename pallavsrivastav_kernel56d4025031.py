#import the required libraries
import pandas as pd
import numpy as np
import difflib
import re
import json
import os.path
#Read the meta data into pandas dataframe
df_covid19=pd.read_csv("CORD-19-research-challenge/metadata.csv")
#As we know that not every research papers have full text, we will get rid of those from our dataset
df_covid19=df_covid19[df_covid19["has_full_text"]]
#Let's also get rid of the research papers that doesn't have sha ids
df_covid19=df_covid19[df_covid19["sha"]==df_covid19["sha"]]
#Write a method to read all the json files path
def jsonFilePath(shaid):
    for directoryname, _, files in os.walk('CORD-19-research-challenge'):
        if shaid+'.json' in files:
            return os.path.join(directoryname,shaid+".json")   
df_covid19["JsonPath"]=df_covid19.apply(lambda x: jsonFilePath(x["sha"]),axis=1)
df_covid19=df_covid19[df_covid19["JsonPath"]==df_covid19["JsonPath"]]
VirusTexts={} 
Abs_and_concl_w_punct={}
valid_id=[]
for shaid,file in zip(df_covid19["sha"],df_covid19["JsonPath"]):
    with open(file, 'r') as f:
        doc=json.load(f)
    MainText=''
    A_C_w_p=''
    for item in doc["body_text"]:
        MainText=MainText+(re.sub('[^a-zA-Z0-9]', ' ', item["text"].lower()))
        if (item["section"]=="Discussion") or (item["section"]=="Abstract") or (item["section"]=="Conclusion"):
            A_C_w_p=A_C_w_p+item["text"].lower()
    if ('vir ' in MainText) and ('corona' in MainText):
        VirusTexts[shaid]=MainText
        Abs_and_concl_w_punct[shaid]=A_C_w_p
        valid_id.append(shaid)
df_covid19=df_covid19[df_covid19["sha"].isin(valid_id)]
MIN_LENGTH=6 
drugs=[]
for shaid in valid_id:
    iterator=re.finditer('vir ',VirusTexts[shaid])
    for m in iterator:
        drugs.append(VirusTexts[shaid][VirusTexts[shaid].rfind(' ',0, m.end()-2):m.end()])
drugs=[i for i in drugs if len(i)>MIN_LENGTH]
drugs_set=list(set(drugs))
count=[]
for d in drugs_set:
    count.append(-drugs.count(d))
drugs_set=list(np.array(drugs_set)[np.array(count).argsort()]) 
import distance
THRESHOLD=2 #Threshold for the Levenshtein distance
incorrects=[]
corrects=[]
from itertools import combinations
for str1,str2 in combinations(drugs_set,2):
    if (distance.levenshtein(str1, str2)<THRESHOLD) and (drugs.count(str1)>10 or drugs.count(str2)>10):
            if drugs.count(str1)>=drugs.count(str2):
                incorrect=str2
                correct=str1
            else:
                incorrect=str1
                correct=str2
            print(str1, "(",drugs.count(str1),")", "and", str2, "(",drugs.count(str2),")", "look very similar.")
            if incorrect not in incorrects:
                incorrects.append(incorrect)
                corrects.append(correct)
for item in incorrects:
    drugs_set.remove(item)
for shaid in valid_id:
    for inc in range(0,len(incorrects)):
        VirusTexts[shaid]=re.sub(incorrects[inc],corrects[inc], VirusTexts[shaid])
antivirals=pd.DataFrame(drugs_set,columns=["AntiViral"])

def count1(drug,druglist):
    return druglist.count(drug)

def count2(drug):
    n=0
    for shaid in valid_id:
        iterator=re.finditer(drug,Abs_and_concl_w_punct[shaid])
        for m in iterator:
            n+=1 
    return n
        
antivirals['Number of times used in body'] = antivirals.apply(lambda x: count1(x["AntiViral"],drugs),axis=1) 
antivirals['Number of times used in abstract and conclusion'] = antivirals.apply(lambda x: count2(x["AntiViral"]),axis=1) 
antivirals['Weightage']=antivirals['Number of times used in abstract and conclusion']/antivirals['Number of times used in body']*100
result = antivirals.sort_values(by=['Number of times used in body','Weightage'],ascending=[0, 0])
result.to_excel("Antivirals.xlsx")
result