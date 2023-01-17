import os

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats
sample_output_df = pd.read_csv("../input/cityofla/CityofLA/Additional data/sample job class export template.csv")

sample_output_df.T
kaggle_dic = pd.read_csv("../input/cityofla/CityofLA/Additional data/kaggle_data_dictionary.csv")

kaggle_dic
job_bulletins_path = "../input/cityofla/CityofLA/Job Bulletins/"

print("Number of Job bulletins : ",len(os.listdir(job_bulletins_path)))
os.listdir(job_bulletins_path)[0:2]
with open(job_bulletins_path + os.listdir(job_bulletins_path)[15]) as f: 

    print (f.read(5000))
#Extracting file name

jobs_list = []

for file_name in os.listdir(job_bulletins_path):

    with open(job_bulletins_path + file_name, encoding = "ISO-8859-1") as f:

        content = f.read()

        jobs_list.append([file_name, content])

jobs_df = pd.DataFrame(jobs_list)

jobs_df.columns = ["FileName", "Content"]

jobs_df.head()
#Extracting Job class title

import re



def extract_job_class_title(text):

    word1 = " ".join(re.findall("[a-zA-Z]+", text))

    return word1.rsplit(' ', 1)[0]

    

jobs_df["JOB_CLASS_TITLE"] = jobs_df["FileName"].apply(lambda x: extract_job_class_title(x))

jobs_df.head()
#Extracting Job Class No

def extract_job_class_no(text):

    try:

        return int(re.findall(r'\d+',text)[0])

    except:

        -1



jobs_df["JOB_CLASS_NO"] = jobs_df["FileName"].apply(lambda x: extract_job_class_no(x))

jobs_df.head()  
#There is only one which does not have a JobClassNo

jobs_df[jobs_df.isnull().any(axis=1)]
#Still Unclear on what these mean
#Still Unclear on what these mean
def extract_job_duties(text):

    words = 'DUTIES'.split(' ')

    sentences = re.findall(r"([^.]*\.)" ,text)  

    for sentence in sentences:

        try:

            if any(word in sentence for word in words):

                return sentence.split('\n')[4]

        except:

            if any(word in sentence for word in words):

                return sentence.split('\n')[3]



            

jobs_df["JOB_DUTIES"] = jobs_df["Content"].apply(lambda x: extract_job_duties(x))

jobs_df.head()
#Extracting data for education years and education type

def extract_edu_info(text):

    words = 'college university'.split(' ')

    sentences = re.findall(r"([^.]*\.)" ,text)  

    for sentence in sentences:

        if any(word in sentence for word in words):

            return sentence

            



jobs_df["EDUCATION_INFO"] = jobs_df["Content"].apply(lambda x: extract_edu_info(x))

jobs_df.head()



#There are a lot of job positions (62%) that do not require a college university education

jobs_df[jobs_df.isnull().any(axis=1)].shape[0]/jobs_df.shape[0] * 100
numbers = ["one","two","three", "four","five","six","seven","eight","nine"]



def extract_edu_years(text):

    try:

        y = (set(re.findall(r'\s|,|[^-\s]+', text.lower())).intersection(set(numbers)))

        return y

    except:

        "Null"

        

jobs_df["EDUCATION_YEARS"] = jobs_df["EDUCATION_INFO"].apply(lambda x: extract_edu_years(x))

jobs_df.head()
def extract_school_type(text):

    try:

        y = (set(re.findall(r'\s|,|[^-\s]+', text.lower())).intersection(set(['college','university'])))

        return y

    except:

        "Null"

        

jobs_df["SCHOOL_TYPE"] = jobs_df["EDUCATION_INFO"].apply(lambda x: extract_school_type(x))

jobs_df.head()
filelist = os.listdir("../input/cityofla/CityofLA/Additional data/City Job Paths")
#Still figuring out on how to do this

filelist = [i.split('.')[0].replace('_',' ').lower() for i in filelist]

text = jobs_df.EDUCATION_INFO[0].lower()

x = [i for i in filelist if i in text]

print(x)
#Extracting the EXPERIENCE_LENGTH



numbers = ["one","two","three", "four","five","six","seven","eight","nine"]



def extract_exp_len(text):

    words = 'full-time paid experience'.split(' ')

    sentences = re.findall(r"([^.]*\.)" ,text)  



    list = []



    for sentence in sentences:

        if any(word in sentence for word in words):

            #print(sentence)

            list.append(sentence)

            

    try:

        y = (set(list[0].lower().split()).intersection(set(numbers)))

        return(y)

    except:

        "Null"



jobs_df["EXPERIENCE_LENGTH"] = jobs_df["Content"].apply(lambda x: extract_exp_len(x))

jobs_df.head()
jobs_df[jobs_df['EXPERIENCE_LENGTH'].isnull()]
def extract_FULL_TIME_PART_TIME(text):

    words = 'full-time part-time'.split(' ')

    sentences = re.findall(r"([^.]*\.)" ,text)  



    for sentence in sentences:

        if any(word in sentence for word in words):

            x = set(sentence.split()).intersection(set(words))

            return x

        

jobs_df["FULL_TIME_PART_TIME"] = jobs_df["Content"].apply(lambda x: extract_FULL_TIME_PART_TIME(x))

jobs_df.head()

jobs_df[jobs_df['FULL_TIME_PART_TIME'].isnull()].shape[0]