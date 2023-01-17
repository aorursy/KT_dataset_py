import os

import pandas as pd,numpy as np

import re

import spacy

from spacy import displacy

from collections import Counter

import en_core_web_sm

nlp = en_core_web_sm.load()

from collections import Counter

import pprint

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

bulletin_dir = '../input/cityofla/CityofLA/Job Bulletins/'

additional_data_dir = '../input/cityofla/CityofLA/Additional data/'
headings = {}

m=0

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        for line in f.readlines():

            line = line.replace("\n","").replace("\t","").replace(":","").strip()

#             m+=1

#             if m==8:

#                 break

            if line.isupper():

                if line not in headings.keys():

                    headings[line] = 1

                else:

                    count = int(headings[line])

                    headings[line] = count+1

#     break
del headings['$103,606 TO $151,484'] #This is not a heading, it's an Annual Salary component

headingsFrame = []

for i,j in (sorted(headings.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)):

    headingsFrame.append([i,j])

headingsFrame = pd.DataFrame(headingsFrame)

headingsFrame.columns = ["Heading","Count"]

#headingsFrame.head()
#Add 'FILE_NAME', 'JOB_CLASS_TITLE', 'JOB_CLASS_NO' ,'OPEN_DATE'

data_list = []

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        job_class_title = ''

        for line in f.readlines():

            #Insert code to parse job bulletins

            if "Open Date:" in line:

                job_bulletin_date = line.split("Open Date:")[1].split("(")[0].strip()

            if "Class Code:" in line:

                job_class_no = line.split("Class Code:")[1].strip()

            if len(job_class_title)<2 and len(line.strip())>1:

                job_class_title = line.strip()

        data_list.append([filename, job_bulletin_date, job_class_title, job_class_no])
df = pd.DataFrame(data_list)

df.columns = ["FILE_NAME", "OPEN_DATE", "JOB_CLASS_TITLE", "JOB_CLASS_NO"]

df.head()
#Add 'REQUIREMENT_SET_ID','REQUIREMENT_SUBSET_ID'

requirements = []

requirementHeadings = [k for k in headingsFrame['Heading'].values if 'requirement' in k.lower()]

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        readNext = 0

        isNumber=0

        prevNumber=0

        prevLine=''

        

        for line in f.readlines():

            clean_line = line.replace("\n","").replace("\t","").replace(":","").strip()   

            if readNext == 0:                         

                if clean_line in requirementHeadings:

                    readNext = 1

            elif readNext == 1:

                if clean_line in headingsFrame['Heading'].values:

                    if isNumber>0:

                        requirements.append([filename,prevNumber,'',prevLine])

                    break

                elif len(clean_line)<2:

                    continue

                else:

                    rqrmntText = clean_line.split('.')

                    if len(rqrmntText)<2:

                        requirements.append([filename,'','',clean_line])

                    else:                        

                        if rqrmntText[0].isdigit():

                            if isNumber>0:

                                requirements.append([filename,prevNumber,'',prevLine])

                            isNumber=1

                            prevNumber=rqrmntText[0]

                            prevLine=clean_line

                        elif re.match('^[a-z]$',rqrmntText[0]):

                            requirements.append([filename,prevNumber,rqrmntText[0],prevLine+'-'+clean_line])

                            isNumber=0

                        else:

                            requirements.append([filename,'','',clean_line])
df_requirements = pd.DataFrame(requirements)

df_requirements.columns = ['FILE_NAME','REQUIREMENT_SET_ID','REQUIREMENT_SUBSET_ID','REQUIREMENT_TEXT']

df_requirements.head()
#Check for one sample file 

df_requirements.loc[df_requirements['FILE_NAME']=='SYSTEMS ANALYST 1596 102717.txt']
#Check for salary components

salHeadings = [k for k in headingsFrame['Heading'].values if 'salary' in k.lower()]

sal_list = []

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        readNext = 0

        for line in f.readlines():

            clean_line = line.replace("\n","").replace("\t","").replace(":","").strip()  

            if clean_line in salHeadings:

                readNext = 1

            elif readNext == 1:

                if clean_line in headingsFrame['Heading'].values:

                    break

                elif len(clean_line)<2:

                    continue

                else:

                    sal_list.append([filename, clean_line])

df_salary = pd.DataFrame(sal_list)

df_salary.columns = ['FILE_NAME','SALARY_TEXT']

df_salary['SALARY_TEXT'][6]
files = []

for filename in os.listdir(bulletin_dir):

    files.append(filename)

    
#Add 'ENTRY_SALARY_GEN','ENTRY_SALARY_DWP'

pattern = r'\$?([0-9]{1,3},([0-9]{3},)*[0-9]{3}|[0-9]+)(.[0-9][0-9])?'

dwp_salary_list = {}

gen_salary_list = {}

for filename in files:

    for sal_text in df_salary.loc[df_salary['FILE_NAME']==filename]['SALARY_TEXT']:

        if 'department of water' in sal_text.lower():

            if filename in dwp_salary_list.keys():

                continue

            matches = re.findall(pattern+' to '+pattern, sal_text) 

            if len(matches)>0:

                salary_dwp = ' - '.join([x for x in matches[0] if x and not x.endswith(',')])

            else:

                matches = re.findall(pattern, sal_text)

                if len(matches)>0:

                    salary_dwp = matches[0][0]

                else:

                    salary_dwp = ''

            dwp_salary_list[filename]= salary_dwp

        else:

            if filename in gen_salary_list.keys():

                continue

            matches = re.findall(pattern+' to '+pattern, sal_text)

            if len(matches)>0:

                salary_gen = ' - '.join([x for x in matches[0] if x and not x.endswith(',')])

            else:

                matches = re.findall(pattern, sal_text)

                if len(matches)>0:

                    salary_gen = matches[0][0]

                else:

                    salary_gen = ''

            if len(salary_gen)>1:

                gen_salary_list[filename]= salary_gen
df_salary_dwp = pd.DataFrame(list(dwp_salary_list.items()), columns=['FILE_NAME','ENTRY_SALARY_DWP'])

df_salary_gen = pd.DataFrame(list(gen_salary_list.items()), columns=['FILE_NAME','ENTRY_SALARY_GEN'])
result = pd.merge(df, df_requirements, how='inner', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)



result = pd.merge(result, df_salary_dwp, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)



result = pd.merge(result, df_salary_gen, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)
#ADD JOB_DUTIES_1 AND JOB_DUTIES_2

duties = {}

name2=[]

d1=[]

d2=[]

m=0

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        for line in f.readlines():

            line = line.replace("\n","").replace("\t","").replace(":","").strip()

            if 'DUTIES' in line:

                m=1

    #             print(line)

            if line.isupper() and line!='DUTIES':

                m=0

    #             print(line)

            if m==1 and line!='DUTIES' and line!='':

                

                if filename not in duties:

                    duties[filename]=line

                else:

                    d3=[duties[filename],line]

                    duties[filename]=d3

                    

for i in duties:

    name2.append(i)

    if len(duties[i])==2:

        d1.append(duties[i][0])

        d2.append(duties[i][1])

    else:

        d1.append(duties[i])

        d2.append(None)



df_duties = pd.DataFrame({'FILE_NAME': name2,'JOB_DUTIES_1': d1,'JOB_DUTIES_2': d2})

result = pd.merge(result, df_duties, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)





#WITHOUT DUTIES



# APPARATUS OPERATOR 2121 071417 (1).txt

# ENGINEER OF FIRE DEPARTMENT 2131 111116.txt

# FIRE ASSISTANT CHIEF 2166 011218.txt

# FIRE BATTALION CHIEF 2152 030918.txt

# FIRE HELICOPTER PILOT 3563 081415 REV. 081815.txt

# FIRE INSPECTOR 2128 031717.txt

# Vocational Worker  DEPARTMENT OF PUBLIC WORKS.txt

### ADD the EXAM_type

exam_type = []

name1=[]

m=0

EXAM_TYPE=[]

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        for line in f.readlines():

            line = line.replace("\n","").replace("\t","").replace(":","").strip()

            if 'THIS EXAM' in line:

                m=1

    #             print(line)

            if 'discriminate' in line:

                m=0

    #             print(line)

            if m==1 and line!='' and not('THIS' in line):

                name1.append(filename)

                exam_type.append(line)

#                 print(filename,line)



for i in exam_type:

    if ('OPEN' in i) and ('INTER' in i):

        EXAM_TYPE.append('OPEN_INT_PROM')

    elif ('OPEN' in i) and ('PROMO' in i):

        EXAM_TYPE.append('OPEN_DEPT_PROM')

    elif 'OPEN' in i:

        EXAM_TYPE.append('OPEN')

    elif 'INTER' in i:

        EXAM_TYPE.append('INT_DEPT_PROM')

    else:

        EXAM_TYPE.append('DEPT_PROM')

            

df_exam_type = pd.DataFrame({'FILE_NAME': name1,'EXAM_TYPE': EXAM_TYPE})

result = pd.merge(result, df_exam_type, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)





#WITHOUT EXAMINATION



# PILE DRIVER WORKER 3553 041417.txt

# Vocational Worker  DEPARTMENT OF PUBLIC WORKS.txt
#result.drop(columns=['REQUIREMENT_TEXT'], inplace=True)

result

# result.to_csv('result_table.csv',index=False)
def getSchoolType(txt):

    txt1=''

    line = txt.lower().find('college or university')

    if line==-1: 

#         return ''

        txt1+=''

    else:

        txt1+= 'college or university '

    line1 = txt.lower().find('high school')

    if line1==-1:

        txt1+= ''

    else:

        txt1+='HIGH SCHOOL'

    

    line2 = txt.lower().find('apprenticeship')

    if line2==-1:

        txt1+= ''

    else:

        txt1+='apprenticeship'



    return txt1



result['SCHOOL_TYPE'] = result['REQUIREMENT_TEXT'].apply(lambda x: getSchoolType(x))

result
df1=result[result['SCHOOL_TYPE']=='apprenticeship'].drop(columns=['OPEN_DATE', 'JOB_CLASS_TITLE', 'JOB_CLASS_NO','ENTRY_SALARY_DWP', 'ENTRY_SALARY_GEN', 'JOB_DUTIES_1', 'JOB_DUTIES_2','EXAM_TYPE','SCHOOL_TYPE'])

df1
for i in df1['REQUIREMENT_TEXT']:

    print(i)
displacy.render(doc[175],style='dep')
import string

from spacy import displacy

from spacy.matcher import Matcher



tagger=nltk.pos_tag

tokenizer = word_tokenize

stop = stopwords.words('english')





df1['REQUIREMENT_TEXT']= df1['REQUIREMENT_TEXT'].apply(lambda x: x.lower())

# df1['REQUIREMENT_TEXT']= df1['REQUIREMENT_TEXT'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

df1['REQUIREMENT_TEXT']= df1['REQUIREMENT_TEXT'].apply(lambda x: x.translate(str.maketrans('', '', string.digits)))







# ####################################################################################################################

# text= result['REQUIREMENT_TEXT'].apply(tokenizer)

# text= text.apply(tagger)

#text=text.apply(lambda x: [item for item in x if item not in stop])

doc= df1['REQUIREMENT_TEXT'].apply(nlp)



# displacy.render(doc[19],style='dep')



matcher = Matcher(nlp.vocab)

pattern = [{'POS': 'NOUN'},

           {'POS': 'ADP'},

           {'POS': 'DET'},

           {'POS': 'NOUN'},

           {'POS': 'CCONJ', 'OP': '?'},

           {'POS': 'NOUN', 'OP': '?'},

           {'POS': 'NOUN', 'OP': '?'},

           {'POS': 'Noun', 'OP': '?'}]





# text= result['REQUIREMENT_TEXT'].apply(tokenizer)

# # text= text.apply(lambda x: [item for item in x if item not in stop])

# # text = [word for word in text if word.isalpha()]

# # text



                                                    

# # def preprocess(txt):

# #     txt = nltk.word_tokenize(txt)

# #     txt = nltk.pos_tag(txt)

# #     return txt
matcher.add("apprenticeship",None, pattern)

matches = matcher(doc[219])

matches

for match_id, start, end in matches:

# Get the matched span by slicing the Doc

    span = doc[219][start:end]

    print(span.text)
matcher.add("apprenticeship",None, pattern)

# matches =[]

for i in doc:

    matches =[]

    matches.append(matcher(i))

    if matches[0]:

        print(i[matches[0][-1][1]:matches[0][-1][2]])

# matches

# for match_id, start, end in matches:

# # Get the matched span by slicing the Doc

#     span = doc[19][start:end]

#     print(span.text)
matches
displacy.render(doc[19],style='dep')
doc[0],doc[1],doc[2]
text= result['REQUIREMENT_TEXT'].apply(tokenizer)

text= text.apply(lambda x: [item for item in x if item not in stop])
for i in text:

    for t in i:

        if len(t)==1:

            i.remove(t)

text