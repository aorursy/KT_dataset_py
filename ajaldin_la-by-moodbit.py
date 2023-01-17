import numpy as np # linear algebra

import csv as csv

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import os, sys

import numpy as np

from datetime import datetime

from collections  import Counter

from nltk import word_tokenize

import seaborn as sns

import matplotlib.pyplot as plt

import calendar

from wordcloud import WordCloud ,STOPWORDS

from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models import word2vec

from sklearn.manifold import TSNE

from nltk import pos_tag

from nltk.help import upenn_tagset

import gensim

import matplotlib.colors as mcolors

from nltk import jaccard_distance

from nltk import ngrams

plt.style.use('ggplot')





import spacy

from spacy import displacy

import nltk

import xml.etree.cElementTree as ET

from collections import OrderedDict

import json

import networkx as nx
bulletins=os.listdir("../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/")

additional=os.listdir("../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/")
files=[dir for dir in os.walk('../input/data-science-for-good-city-of-los-angeles/CityofLA/CityofLA/')]

for file in files:

    print(os.listdir(file[0]))

    print("\n")
csvfiles=[]

for file in additional:

    if file.endswith('.csv'):

        print(file)

        csvfiles.append("../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/"+file)
job_title=pd.read_csv(csvfiles[2])

sample_job=pd.read_csv(csvfiles[0])

kaggle_data=pd.read_csv(csvfiles[1])
job_title.head()
def get_headings(bulletin):       

    

    """"function to get the headings from text file

        takes a single argument

        1.takes single argument list of bulletin files"""

    

    with open("../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/"+bulletins[bulletin]) as f:    ##reading text files 

        data=f.read().replace('\t','').split('\n')

        data=[head for head in data if head.isupper()]

        return data

        

def clean_text(bulletin):      

    

    

    """function to do basic data cleaning

        takes a single argument

        1.takes single argument list of bulletin files"""

                                            

    

    with open("../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/"+bulletins[bulletin]) as f:

        data=f.read().replace('\t','').replace('\n','')

        return data
get_headings(1)
def to_dataframe(num,df):

    """"function to extract features from job bulletin text files and convert to

    pandas dataframe.

    function take two arguments 

                        1.the number of files to be read

                        2.dataframe object                                      """

    



    

    opendate=re.compile(r'(Open [D,d]ate:)(\s+)(\d+-\d\d-\d\d)')       #match open date

    

    salary=re.compile(r'\$(\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?')       #match salary

    

    requirements=re.compile(r'(REQUIREMENTS?/\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)')      #match requirements

    

    apply=re.compile(r'(WHERE TO APPLY?)(.*)(NOTE)')

    

    for no in range(0,num):

        with open("../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/"+bulletins[no],encoding="ISO-8859-1") as f:         #reading files 

                try:

                    file=f.read().replace('\t','')

                    data=file.replace('\n','')

                    headings=[heading for heading in file.split('\n') if heading.isupper()]             ##getting heading from job bulletin



                    try:

                        date=datetime.strptime(re.search(opendate,data).group(3),'%m-%d-%y')

                    except Exception as e:

                        date=np.nan

                    try:

                        req=re.search(requirements,data).group(2)

                    except Exception as e:

                        try: 

                            req=re.search('(.*)POST-Certified',re.findall(r'(REQUIREMENTS?)(.*)(POST-Certified)', 

                                                                  data)[0][1][:1200]).group(1)

                        except Exception as e:

                            req=re.search('(.*)NOTES?',re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',

                                                              data)[0][1][:1200]).group(1)

                    

                    try:

                        duties=re.search(r'(DUTIES)(.*)(REQ[A-Z])',data).group(2)

                        

                    except Exception as e:

                        duties=np.nan

                    

                    app=re.search(apply,data).group(2)

                    

                    try:

                        enddate=re.search(

                                r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s(\d{1,2},\s\d{4})'

                                ,data).group()

                    except Exception as e:

                        enddate=np.nan

                        

                    sal=re.search(salary,data)

                    selection= [z[0] for z in re.findall('([A-Z][a-z]+)((\s\.\s)+)',data)]     ##match selection criteria

                    try:

                        sal=re.search(salary,data)

                        df=df.append({'FILE_NAME':bulletins[no],'Position':headings[0].lower(), 'salary_start':sal.group(1),'salary_end':sal.group(5),

                                  "opendate":date,"requirements":req, 'duties':duties, 'apply':app,#'duties':duties, 'apply':app,

                                  'deadline':enddate},ignore_index=True)  #,'selection':selection

                    

                    except Exception as e:

                        sal=np.nan

                        df=df.append({'FILE_NAME':bulletins[no],'Position':headings[0].lower(), 'salary_start':sal,'salary_end':sal,

                                  "opendate":date, "requirements":req, 'duties':duties, 'apply':app,#"requirements":req,'duties':duties, 'apply':app,

                                  'deadline':enddate, 'selection':selection},ignore_index=True)  #,'selection':selection

                    

                    #selection= [z[0] for z in re.findall('([A-Z][a-z]+)((\s\.\s)+)',data)]     ##match selection criteria

                    

                    #df=df.append({'File Name':bulletins[no],'Position':headings[0].lower(), 'salary_start':sal.group(1),'salary_end':sal.group(5),

                    #              "opendate":date,#"requirements":req,'duties':duties, 'apply':app,

                    #              'deadline':enddate},ignore_index=True)  #,'selection':selection

                    

                    

                    reg=re.compile(r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)\s(years?)\s(of\sfull(-|\s)time)')

                    df['EXPERIENCE_LENGTH']=df['requirements'].apply(lambda x :  re.search(reg,x).group(1) if re.search(reg,x) is not None  else np.nan)

                    df['FULL_TIME_PART_TIME']=df['EXPERIENCE_LENGTH'].apply(lambda x:  'FULL_TIME' if x is not np.nan else np.nan )

                    

                    reg=re.compile(r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)(\s|-)(years?)\s(college)')

                    df['EDUCATION_YEARS']=df['requirements'].apply(lambda x :  re.search(reg,x).group(1) if re.search(reg,x) is not None  else np.nan)

                    df['SCHOOL_TYPE']=df['EDUCATION_YEARS'].apply(lambda x : 'College or University' if x is not np.nan else np.nan)

                    

                #except Exception as e:

                #    print('umatched sequence')

                #    print(f)

                except IOError:

                    print('An error occured trying to read the file.')

    

                except ValueError:

                    print('Non-numeric data found in the file.')



                except ImportError:

                    print ("NO module found")

    

                except EOFError:

                    print('Why did you do an EOF on me?')

                except KeyboardInterrupt:

                    print('You cancelled the operation.')



                except Exception as e:

                    print(e)

                    print(f)

                

                

        

           

    return df
df=pd.DataFrame(columns=['FILE_NAME','Position','salary_start','salary_end', 'opendate', 'requirements', 'duties', 'apply', 'deadline']) #,'duties', 'apply'

df=to_dataframe(len(bulletins),df)

df.shape
df.isnull().sum()
df.head()
bulletin_dir = "../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/"

additional_data_dir = "../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/"
headings = {}

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        for line in f.readlines():

            line = line.replace("\n","").replace("\t","").replace(":","").strip()

            

            if line.isupper():

                if line not in headings.keys():

                    headings[line] = 1

                else:

                    count = int(headings[line])

                    headings[line] = count+1
del headings['$103,606 TO $151,484'] #This is not a heading, it's an Annual Salary component

headingsFrame = []

for i,j in (sorted(headings.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)):

    headingsFrame.append([i,j])

headingsFrame = pd.DataFrame(headingsFrame)

headingsFrame.columns = ["Heading","Count"]

#headingsFrame.head()
#Check for note components

noteHeadings = [k for k in headingsFrame['Heading'].values if 'note' in k.lower()]

note_list = []

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        readNext = 0

        for line in f.readlines():

            clean_line = line.replace("\n","").replace("\t","").replace(":","").strip()  

            if clean_line in noteHeadings:

                readNext = 1

            elif readNext == 1:

                if clean_line in headingsFrame['Heading'].values:

                    break

                elif len(clean_line)<2:

                    continue

                else:

                    note_list.append([filename, clean_line])
df_note = pd.DataFrame(note_list)

df_note.columns = ['FILE_NAME','NOTE_TEXT']

df_note.head()
file_name = df_note['FILE_NAME'].unique()
note_list = []

for title in file_name:

    d = df_note[df_note['FILE_NAME']==title]

    context = ' '.join(list(d['NOTE_TEXT']))

    note_list.append([title, context])
df_note = pd.DataFrame(note_list)

df_note.columns = ['FILE_NAME','NOTE_TEXT']

df_note.head()
result = pd.merge(df, df_note, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)
result.head()
#Check for process components

proHeadings = [k for k in headingsFrame['Heading'].values if 'process' in k.lower()]

pro_list = []

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

        readNext = 0

        for line in f.readlines():

            clean_line = line.replace("\n","").replace("\t","").replace(":","").strip()  

            if clean_line in proHeadings:

                readNext = 1

            elif readNext == 1:

                if clean_line in headingsFrame['Heading'].values:

                    break

                elif len(clean_line)<2:

                    continue

                else:

                    pro_list.append([filename, clean_line])
df_pro = pd.DataFrame(pro_list)

df_pro.columns = ['FILE_NAME','PROCESS_TEXT']

df_pro.head()
pro_list = []

for title in file_name:

    df = df_pro[df_pro['FILE_NAME']==title]

    context = ' '.join(list(df['PROCESS_TEXT']))

    pro_list.append([title, context])
df_pro = pd.DataFrame(pro_list)

df_pro.columns = ['FILE_NAME','PROCESS_TEXT']

df_pro.head()
result = pd.merge(result, df_pro, how='left', left_on='FILE_NAME', right_on='FILE_NAME', sort=True)
result.head()
word_list = pd.read_csv('../input/genderbiasdictionary/genderbiascatalog1.csv')

word_list
masculine_list = word_list['Biased Catalog']
replace_list = word_list['Replace']
master_dic = {}



masculine_list = [w.lower() for w in masculine_list]

columns = ['requirements', 'NOTE_TEXT', 'duties', 'PROCESS_TEXT', 'apply']



for c in columns:

    for sentence in result[c]:

        for word in str(sentence).split():

            if word.lower() not in master_dic and word.lower() in masculine_list:

                master_dic[word.lower()] = 1

            elif word.lower() in master_dic and word.lower() in masculine_list:

                master_dic[word.lower()] += 1

master_dic
file_dic = {}



for name in result['FILE_NAME']:

    for c in columns:

        for sentence in result[result['FILE_NAME']==name][c]:

            for word in str(sentence).split():

                if name not in file_dic and word.lower() in masculine_list:

                    file_dic[name] = [word.lower()]

                elif name in file_dic and word.lower() in masculine_list and word.lower() not in file_dic[name]:

                    file_dic[name].append(word.lower())
file_dic
file_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in file_dic.items() ]))

file_df_T = file_df.transpose()

file_df_T.columns = ['bias_1', 'bias_2', 'bias_3', 'bias_4', 'bias_5', 'bias_6', 'bias_7']

file_df_T
def ifef(col):

    col = str(col)

    if col in masculine_list:

        i = masculine_list.index(col)

        return  replace_list[i]

    else:

        return 'NaN'
file_df_T['replace_1'] = file_df_T['bias_1'].apply(ifef)

file_df_T['replace_2'] = file_df_T['bias_2'].apply(ifef)

file_df_T['replace_3'] = file_df_T['bias_3'].apply(ifef)

file_df_T['replace_4'] = file_df_T['bias_4'].apply(ifef)

file_df_T['replace_5'] = file_df_T['bias_5'].apply(ifef)

file_df_T['replace_6'] = file_df_T['bias_6'].apply(ifef)

file_df_T['replace_7'] = file_df_T['bias_7'].apply(ifef)

file_df_T = file_df_T[['bias_1', 'replace_1', 'bias_2', 'replace_2', 'bias_3', 'replace_3', 'bias_4', 'replace_4', 'bias_5', 'replace_5', 'bias_6', 'replace_6', 'bias_7', 'replace_7']]

file_df_T
# Extract Lower Job Class

req_pos = result[['Position', 'requirements']].copy()

# regular expression for extracting job class

job_regex = r'(?:(?<=experience\swith\s)|(?<=experience\s))(.+?)(?:\.|\;)'



class_dict2 = {}



for name in req_pos['Position']:

    for sentence in req_pos[req_pos['Position']==name]["requirements"]:

            job = re.findall(job_regex, sentence)

            class_dict2[name] = job
class_dict2
degree_dict = {}

#regex for extracting degree

degree_regex = r'(?<=degree\s).+?\.'

for name in req_pos['Position']:

    for sentence in req_pos[req_pos['Position']==name]["requirements"]:

            degree = re.findall(degree_regex, sentence)

            degree_dict[name] = degree

        
degree_dict
class_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in class_dict2.items() ]))

class_df = class_df.transpose()

class_df.reset_index(inplace=True)

class_df = pd.melt(class_df, id_vars=['index'], var_name='number', value_name = 'requirements')



class_df.head()
for name in class_df.loc[0:100,'index']:

    group = class_df.groupby('index').get_group(name)

    FG = nx.from_pandas_edgelist(group, source='requirements', target='index', edge_attr=True)

    plt.figure()

    nx.draw_networkx(FG, with_labels=True)