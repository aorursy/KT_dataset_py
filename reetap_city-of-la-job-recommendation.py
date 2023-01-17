# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import os

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

print(os.listdir("../input"))

from gensim.models import word2vec

from sklearn.manifold import TSNE

from nltk import pos_tag

from nltk.help import upenn_tagset

import gensim

import matplotlib.colors as mcolors

import nltk

from nltk.tokenize import word_tokenize

from nltk.tag import pos_tag

import xml.etree.cElementTree as ET

from collections import OrderedDict

import json

from nltk import jaccard_distance

from nltk import ngrams

#import textstat

plt.style.use('ggplot')
import os

import re #search in strings.



import plotly.plotly as py

import cufflinks as cf



import seaborn as sns

import matplotlib.pyplot as plt



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from wordcloud import WordCloud

#import textstat



pd.set_option('max_colwidth', 10000)  # this is important because the requirements are sooooo long 



import warnings

warnings.filterwarnings('ignore')   # get rid of the matplotlib warnings
import os

files=[dir for dir in os.walk('../input/cityofla')]

for file in files:

    print(os.listdir(file[0]))

    print("\n")
for subfold in os.listdir("../input/cityofla/CityofLA/"):

    print(subfold)
job_dir= '../input/cityofla/CityofLA/Job Bulletins'

listOfFile = os.listdir(job_dir)

listOfFile
bulletin_dir = '../input/cityofla/CityofLA/Job Bulletins/'

additional_data_dir = '../input/cityofla/CityofLA/Additional data/'

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
import pandas as pd

import numpy as np

df = pd.DataFrame(data_list)

df.columns = ["START FILE_NAME", "OPENING_DATE", "JOB_CLASS_TITLE", "JOB_CLASS_NO"]

df.head()
import pandas as pd

import numpy as np

input_dir =  '../input/cityofla/CityofLA/Job Bulletins'

def getListOfFiles(dirName):

    

#list of file and sub directories and names in the given directory 

    listOfFile = os.listdir(dirName)

    allFiles = list()

    # Iterate all the entries

    for entry in listOfFile:

    # Create full path

        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory 

        if os.path.isdir(fullPath):

            allFiles = allFiles + getListOfFiles(fullPath)

        else:

            allFiles.append(fullPath)

    return allFiles

listOfFiles = getListOfFiles(input_dir)

df_bulletins = pd.DataFrame(listOfFiles, columns = ['job_position'])

df_bulletins.head()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import os

import numpy as np

# Clean up of the job_position name

df_positions = pd.DataFrame()

df_positions['job_position'] = (df_bulletins['job_position']

                                .str.replace(input_dir, '', regex=False)

                                .str.replace('.txt', '', regex=False)

                                .str.replace('\d+', '')

                                .str.replace(r"\s+\(.*\)","")

                                .str.replace(r"REV",""))



#Remove the numbers

df_positions['class_code'] = (df_bulletins['job_position']

                              .str.replace(input_dir, '', regex=False)

                              .str.replace('.txt', '', regex=False)

                              .str.extract('(\d+)'))



display(df_positions.head())

# Add the Text fields of Salary, Duties and Minimum REQ
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
#Convert the txt files to a table:

import glob

path = input_dir =  '../input/cityofla/CityofLA/Job Bulletins'

all_files = glob.glob(path + "/*.txt")

li = []



for filename in all_files:

    with open (filename, "r",errors='replace') as myfile:

        data=pd.DataFrame(myfile.readlines())

        #df = pd.read_csv(filename, header=0,error_bad_lines=False, encoding='latin-1')

    li.append(data)

frame = pd.concat(li, axis=1, ignore_index=True)

#pd.read_csv(listOfFiles,header = None)

frame = frame.replace('\n','', regex=True)

frame.head(16)
import tkinter

from tkinter import Frame

import pandas as pd

import numpy as np

 #Here the loop should start, for each text file do:

def getString(col_i, frame):

    try:

        filter = frame[col_i] != ""

        bulletin = frame[col_i][filter]

        #display(salary)

        isal = min(bulletin[bulletin.str.contains('SALARY',na=False)].index.values) #take the sum to convert the array to an int...TO CHANGE

        inot = min(bulletin[bulletin.str.contains('NOTES',na=False)].index.values) # NOTES

        idut = min(bulletin[bulletin.str.contains('DUTIES',na=False)].index.values) # DUTIES

        ireq = min(bulletin[bulletin.str.contains('REQUIREMENT',na=False)].index.values) #REQUIREMENTS

        ipro = min(bulletin[bulletin.str.contains('PROCESS',na=False)].index.values) # PROCESS NOTES



        #isal = sum(bulletin.loc[bulletin == 'ANNUAL SALARY'].index.values) #take the sum to convert the array to an int...TO CHANGE

        #inot = sum(bulletin.loc[bulletin == 'NOTES:'].index.values) # NOTES

        #idut = sum(bulletin.loc[bulletin == 'DUTIES'].index.values) # DUTIES

        #ireq = sum(bulletin.loc[bulletin == '(.*)REQUIREMENTS(.*)'].index.values) #REQUIREMENTS

        #ipro = sum(bulletin.loc[bulletin == '(.*)PROCESS(.*)'].index.values) # PROCESS NOTES



        icode = min(bulletin[bulletin.str.contains('Class Code',na=False)].index.values)

        class_code = sum(bulletin.str.extract('(\d+)').iloc[icode].dropna().astype('int'))

        salary = (bulletin.loc[isal+1:inot-1]).to_string()

        duties = (bulletin.loc[idut+1:ireq-1]).to_string()

        requirements = (bulletin.loc[ireq+1:ipro-1]).to_string()

        return (class_code, salary, duties, requirements)

    except:

        return (np.nan,np.nan,np.nan,np.nan)

jobsections = pd.DataFrame()

   #getString(0,bulletin)

for col_i in range(frame.shape[1]):

    #print(col_i)

    #print(list(getString(col_i,frame)))

    prop = getString(col_i,frame)

    prop = pd.DataFrame(list(prop)).T

    jobsections = jobsections.append(prop)
jobsections.head()
import pandas as pd

import numpy as np

jobsections.columns = ['class_code','salary','duties','requirements']

jobsections['class_code'] = pd.to_numeric(jobsections['class_code'],downcast='integer')

df_positions['class_code'] = pd.to_numeric(df_positions['class_code'], downcast='integer')

df_positions['class_code']

df_jobs = df_positions.merge(jobsections, left_on='class_code',right_on='class_code', how='outer')

display(df_jobs.dropna())
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
import pandas as pd

import numpy as np

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

df_salary.head()
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

def preprocess(txt):

    txt = nltk.word_tokenize(txt)

    txt = nltk.pos_tag(txt)

    return txt
import nltk

def getEducationMajor(row):

    txt = row['REQUIREMENT_TEXT']

    txtMajor = ''

    if 'major in' not in txt.lower() and ' majoring ' not in txt.lower():

        return txtMajor

    result = []

    

    istart = txt.lower().find(' major in ')

    if istart!=-1:

        txt = txt[istart+10:]

    else:

        istart = txt.lower().find(' majoring ')

        if istart==-1:

            return txtMajor

        txt = txt[istart+12:]

    

    txt = txt.replace(',',' or ').replace(' and/or ',' or ').replace(' a closely related field',' related field')

    sent = preprocess(txt)

    pattern = """

            NP: {<DT>? <JJ>* <NN.*>*}

           BR: {<W.*>|<V.*>} 

        """

    cp = nltk.RegexpParser(pattern)

    cs = cp.parse(sent)

    #print(cs)

    checkNext = 0

    for subtree in cs.subtrees():

        if subtree.label()=='NP':

            result.append(' '.join([w for w, t in subtree.leaves()]))

            checkNext=1

        elif checkNext==1 and subtree.label()=='BR':

            break

    return '|'.join(result)

#Add EDUCATION_MAJOR

df_requirements['EDUCATION_MAJOR']=df_requirements.apply(getEducationMajor, axis=1)

df_requirements.loc[df_requirements['EDUCATION_MAJOR']!=''].head()
def getValues(searchText, COL_NAME):

    data_list = []

    dataHeadings = [k for k in headingsFrame['Heading'].values if searchText in k.lower()]



    for filename in os.listdir(bulletin_dir):

        with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

            readNext = 0 

            datatxt = ''

            for line in f.readlines():

                clean_line = line.replace("\n","").replace("\t","").replace(":","").strip()   

                if readNext == 0:                         

                    if clean_line in dataHeadings:

                        readNext = 1

                elif readNext == 1:

                    if clean_line in headingsFrame['Heading'].values:

                        break

                    else:

                        datatxt = datatxt + ' ' + clean_line

            data_list.append([filename,datatxt.strip()])

    result = pd.DataFrame(data_list)

    result.columns = ['FILE_NAME',COL_NAME]

    return result

#Add JOB_DUTIES

df_duties = getValues('duties','JOB_DUTIES')
print(df_duties['JOB_DUTIES'].loc[df_duties['FILE_NAME'] == 'AIRPORT POLICE SPECIALIST 3236 063017 (2).txt'].values)
#Function to retrieve values that match with pre-defined values 

def section_value_extractor( document, section, subterms_dict, parsed_items_dict ):

    retval = OrderedDict()

    single_section_lines = document.lower()

    

    for node_tag, pattern_string in subterms_dict.items():

        pattern_list = re.split(r",|:", pattern_string[0])#.sort(key=len)

        pattern_list=sorted(pattern_list, key=len, reverse=True)

        #print (pattern_list)

        matches=[]

        for pattern in pattern_list:

            if pattern.lower() in single_section_lines:

                matches.append(pattern)

                single_section_lines = single_section_lines.replace(pattern.lower(),'')

        #print (matches)

        if len(matches):

            info_string = ", ".join(list(matches)) + " "

            retval[node_tag] = info_string

    return retval
#Function to read xml configuration to return json formatted string

def read_config( configfile ):

    root = ET.fromstring(configfile)

    config = []

    for child in root:

        term = OrderedDict()

        term["Term"] = child.get('name', "")

        for level1 in child:

            term["Method"] = level1.get('name', "")

            term["Section"] = level1.get('section', "")

            for level2 in level1:

                term[level2.tag] = term.get(level2.tag, []) + [level2.text]



        config.append(term)

    json_result = json.dumps(config, indent=4)

    return config
def parse_document(document, config):

    parsed_items_dict = OrderedDict()



    for term in config:

        term_name = term.get('Term')

        extraction_method = term.get('Method')

        extraction_method_ref = globals()[extraction_method]

        section = term.get("Section")

        subterms_dict = OrderedDict()

        

        for node_tag, pattern_list in list(term.items())[3:]:

            subterms_dict[node_tag] = pattern_list

        parsed_items_dict[term_name] = extraction_method_ref(document, section, subterms_dict, parsed_items_dict)



    return parsed_items_dict

#Read job_titles to use them to find patterns in the requirement text to extract job_class_titles

job_titles = pd.read_csv(additional_data_dir+'/job_titles.csv', header=None)



job_titles = ','.join(job_titles[0])

job_titles = job_titles.replace('\'','').replace('&','and')

configfile = r'''

<Config-Specifications>

<Term name="Requirements">

        <Method name="section_value_extractor" section="RequirementSection">

            <SchoolType>College or University,High School,Apprenticeship,Certificates</SchoolType>

            <JobTitle>'''+job_titles+'''</JobTitle>

        </Method>

    </Term>

</Config-Specifications>

'''
config = read_config(configfile)

result = df_requirements['REQUIREMENT_TEXT'].apply(lambda k: parse_document(k,config))

i=0

df_requirements['EXP_JOB_CLASS_TITLE']=''

df_requirements['SCHOOL_TYPE']=''

for item in (result.values):

    for requirement,dic in list(item.items()):        

        if 'JobTitle' in dic.keys():

            df_requirements.loc[i,'EXP_JOB_CLASS_TITLE'] = dic['JobTitle']

        if 'SchoolType' in dic.keys():

            df_requirements.loc[i,'SCHOOL_TYPE'] = dic['SchoolType']

    i=i+1
#Let's check the result for one sample file

df_requirements[df_requirements['FILE_NAME']=='SYSTEMS ANALYST 1596 102717.txt'][['FILE_NAME','EXP_JOB_CLASS_TITLE','SCHOOL_TYPE']]
result.drop(columns=['REQUIREMENT_TEXT'], inplace=True)

result[result['FILE_NAME']=='SYSTEMS ANALYST 1596 102717.txt']