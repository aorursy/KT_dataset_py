import re

import os

import string

import operator

import numpy as np 

import pandas as pd

import seaborn as sns

#from afinn import Afinn

from datetime import datetime

from collections import Counter

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

from sklearn.base import BaseEstimator, TransformerMixin

 

import gensim

from gensim.models import word2vec



import nltk

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.corpus import wordnet as wn

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

from nltk.corpus import sentiwordnet as swn

from nltk import sent_tokenize, word_tokenize, pos_tag

from nltk.sentiment.vader import SentimentIntensityAnalyzer



import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)



%matplotlib inline
print(os.listdir("../input"))
additional = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data'

job_titles=pd.read_csv(additional+'/job_titles.csv',names=["JOB_TITLE"])

sample_job=pd.read_csv(additional+'/sample job class export template.csv')

kaggle_data=pd.read_csv(additional+'/kaggle_data_dictionary.csv')
# UTILS

bulletins = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins'

# Exclude the bulletins with inappropriate filling

valid_bulletins = [i for i in os.listdir(bulletins) if i not in ('ANIMAL CARE TECHNICIAN SUPERVISOR 4313 122118.txt',

                                               'WASTEWATER COLLECTION SUPERVISOR 4113 121616.txt',

                                               'SENIOR EXAMINER OF QUESTIONED DOCUMENTS 3231 072216 REVISED 072716.txt',

                                               'SENIOR UTILITY SERVICES SPECIALIST 3753 121815 (1).txt',

                                               'CHIEF CLERK POLICE 1219 061215.txt',

                                               'Vocational Worker  DEPARTMENT OF PUBLIC WORKS.txt')]

# Requiremens - Full Text

def get_requirement(line):

    i = 1

    req = []

    while True:

        req.append(data_list[line + i])

        i+=1

        if any(x in data_list[line + i] for x in headers):

            break

    return list(filter(None, req))

                   

# Salary

# https://www.kaggle.com/ranimdewaib/city-of-la-convert-text-files-to-csv-file

def get_salary(line):

    global salary_gen

    global salary_dwp 

    salary = []

    

    if "flat" in line.lower().strip() and "Department of Water and Power" not in line.strip(): #flat and not dwp(gen)

        salary_gen = re.search(r"\$\d{2,3}\,\d{3}", line.lower()).group()

        salary_dwp = ""

    if "flat" not in line.lower().strip() and "Department of Water and Power" not in line.strip(): #not flat and not dwp(gen)

        salary_gen = re.search(r"\$\d{2,3}\,\d{3}\sto\s\$\d{2,3}\,\d{3}|\$\d{2,3}\,\d{3}[*]\sto\s\$\d{2,3}\,\d{3}|\$\d{2,3}\,\d{3}", line.lower()).group().replace("to","-")

        salary_dwp = ""

    if "flat" in line.lower().strip() and "Department of Water and Power" in line.strip(): #flat and dwp

        salary_dwp = re.search(r"\$\d{2,3}\,\d{3}", line.lower()).group()

    if "flat" not in line.lower().strip() and "Department of Water and Power" in line.strip(): #not flat and dwp

        salary_dwp = re.search(r"\$\d{2,3}\,\d{3}\sto\s\$\d{2,3}\,\d{3}|\$\d{2,3}\,\d{3}\sto\s\$\d{2,3}\,\s\d{3}", line.lower()).group().replace("to","-")

 

    salary.append(salary_gen)

    salary.append(salary_dwp)



    return salary

                   

# Examination Type

# https://www.kaggle.com/danielbecker/l-a-jobs-data-exctraction-eda

def get_exam_type(text):



    regex_dic = {'OPEN_INT_PROM':r'BOTH.*INTERDEPARTMENTAL.*PROMOTIONAL', 

                 'INT_DEPT_PROM':r'INTERDEPARTMENTAL.*PROMOTIONAL', 

                 'DEPT_PROM':r'DEPARTMENTAL.*PROMOTIONAL',

                 'OPEN':r'OPEN.*COMPETITIVE.*BASIS'

                }

    result = np.nan

    for key, value in regex_dic.items():

        regex = value

        regex_find = re.findall(regex, text, re.DOTALL|re.IGNORECASE)

        if regex_find:

            result = key

            break

    return result



# Driver's license requirement (R and P) & licence type (A, B, C)

def get_drive(text):

    global lic_req

    global lic_type

    

    req_dic = {'is required':'R', 'may require':'P'}

    

    lic_req = np.nan

    lic_type = np.nan

    

    drive_search = re.search(r'(is required|may require)', text, re.IGNORECASE)

    if drive_search:

        lic_req = drive_search.group(0)

        lic_req = req_dic[lic_req]

           

    drive_lic_type_search = re.findall(r"(Class \w,\s*\w|Class \w)", text, re.IGNORECASE)

    lic_type = ','.join(drive_lic_type_search).upper().replace("CLASS ", "")

    

    return lic_req, lic_type
# Extracting features from Full Bulletin Text

JOB = []

for file_name in valid_bulletins:

    with open(bulletins + '/' + file_name, encoding = "ISO-8859-1", errors = 'ignore') as f:

        file = f.read().replace('\t','')

        data = file.replace('\n','')

        data_list = file.split('\n')

        headers = [head for head in data_list if head.isupper()]

        class_title = headers[0].lower()  # JOB_CLASS_TITLE

        exam_type = get_exam_type(data)

        lic_req = np.nan

        lic_type = np.nan

        deadline = np.nan

        for line in range(len(data_list)):

            if "Class Code:" in data_list[line]:

                class_code = data_list[line].split("Class Code:")[1].strip()  # JOB_CLASS_NO

            if "Open Date:" in data_list[line]:                              

                open_date = data_list[line].split("Open Date:")[1].split("(")[0].strip() # OPEN_DATE  

            if "DUTIES" in data_list[line]:

                job_duties = data_list[line+2]  # JOB_DUTIES

            if "driver's license" in data_list[line]:

                function = get_drive(data_list[line])

                try:

                    lic_req = function[0]  # DRIVERS_LICENSE_REQ

                    lic_type = function[1] # DRIV_LIC_TYPE

                except TypeError:

                    lic_req = np.nan

                    lic_type = np.nan

            if bool(re.search(r"\$\d{2,3}\,\d{3}", data_list[line])): 

                salary_gen = get_salary(data_list[line])[0] 

                salary_dwp = get_salary(data_list[line])[1]

            if "REQUIRE" in data_list[line]:                                      

                req = get_requirement(line) # REQUIREMENTS_Full_Text

            if "APPLICATION DEADLINE" in data_list[line]:

                deadline_search = re.search(r'([A-Z]{1,9})\s(\d{1,2},\s\d{4})',data_list[line + 2])        

                if deadline_search:

                    deadline = deadline_search.group()  # DEADLINE



        JOB.append([file_name, class_title, class_code, open_date, 

                    exam_type, job_duties, salary_gen, salary_dwp, 

                    req, lic_req, lic_type, deadline])

        

df = pd.DataFrame(JOB)

df.columns = ["FILE_NAME", "JOB_CLASS_TITLE", 'JOB_CLASS_NO', 'OPEN_DATE', 

              "EXAM_TYPE", "JOB_DUTIES", "ENTRY_SALARY_GEN", "ENTRY_SALARY_DWP", 

              "REQ_TEXT","DRIVERS_LICENSE_REQ", "DRIV_LIC_TYPE", "DEADLINE"]
df["REQ_TEXT"][14]
# Split the Requirement Full Text into rows: each requirement in one row

def splitDataFrameList(df,target_column):

    ''' df = dataframe to split,

    target_column = the column containing the values to split

    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 

    The values in the other columns are duplicated across the newly divided rows.

    '''

    row_accumulator = []



    def splitListToRows(row):

        split_row = row[target_column]

        for s in split_row:

            new_row = row.to_dict()

            new_row[target_column] = s

            row_accumulator.append(new_row)



    df.apply(splitListToRows, axis=1)

    new_df = pd.DataFrame(row_accumulator)

    return new_df[df.columns]



df_new = splitDataFrameList(df,'REQ_TEXT')

df_new.head()
# Utils for Feature Exctraction from Requirement Text

# Requirement set and subset IDs 

def req_set_subset_id(text):

    reg_expr_set = r'^\d(?=\.)'

    reg_expr_subset = r'^[a-z](?=\.)'

    set_search = re.finditer(reg_expr_set, text)

    if set_search:

        set_search_result = ''.join([(x.group(0)) for x in set_search])

    else:

        set_search_result = ''



    subset_search = re.finditer(reg_expr_subset, text, re.MULTILINE|re.IGNORECASE)

    if subset_search:

        subset_search_result = ''.join([(x.group(0)) for x in subset_search])

    else:

        subset_search_result = ''

        

    return set_search_result, subset_search_result



# Full Time or Part Time Job?

def get_full_time_part_time(text):

    full_time_search = re.search(r'full\s*-\s*time', text, re.DOTALL|re.IGNORECASE)

    part_time_search = re.search(r'part\s*-\s*time', text, re.DOTALL|re.IGNORECASE)

    if full_time_search:

        full_time_part_time = 'FULL_TIME'

    elif part_time_search:

        full_time_part_time = 'PART_TIME'

    else:

        full_time_part_time = np.nan

    return full_time_part_time



# Required Job Experience : Job Title

def get_job_title(text):

    job_list = job_titles.values

    job_list = job_titles['JOB_TITLE'].values

    job_list = [x for x in job_list if str(x) != 'nan']

    job_list = list(filter(lambda x: re.sub("[^a-zA-Z]","", x), job_list))

    job_list = [x.lower() for x in job_list]

    jobs = []

    for job in job_list:

        if job in text.lower():

            jobs.append(job)

    return '|'.join(jobs)



# Required Job Experience Length

def get_experience_length(text):

    num_dic = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5,

                'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10}

    result = np.nan

    regex_search = re.search(r'(\w{3,5}|\d{1,2})\s*(months?|years?)\s(of\sfull\s*-\s*time|of\spart\s*-\s*time)', text, re.IGNORECASE)

    if regex_search:

        exp_len_raw = regex_search.group(1).lower()

        if exp_len_raw.isnumeric():

            exp_len = exp_len_raw

        elif exp_len_raw in num_dic:

            exp_len = num_dic[exp_len_raw]

        else:

            exp_len = ''

      

        units = regex_search.group(2).lower()

        if 'year' in units and str(exp_len).isnumeric():

            result = float(exp_len)

        if 'month' in units and str(exp_len).isnumeric():

            result = round(float(exp_len)/12.0, 2)

    return result



# Required Job Experience : Job Function

def get_job_func(text):

    functions = ''

    function_search = re.search(r'(experience in|worker in|the responsibility for|performing|performance of|working on)', text, re.IGNORECASE)

    if function_search:

        functions = text.split(function_search.group())[1].strip()

    return functions

    



# Required Education Type 

def get_school(text):    

    

    num_dic = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5,

                'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10}

    

    num_dic_1 = {'one-year':1, 'two-year':2, 'three-year':3, 'four-year':4, 'five-year':5}

    

    edu_years = np.nan

    school_type = np.nan

    school_search = re.search(r'(\w+|\w{3,5}-year?)\s*(college or university|college|university|high school|apprenticeship)', text, re.DOTALL|re.IGNORECASE)

    school_len_search = re.search(r'(\w{3,5}|\d{1,2})(\s*)(year?|month?|semester?)\s*(college or university|college|university|high school|apprenticeship)', text, re.DOTALL|re.IGNORECASE)

    

    if school_search:

        school_len_raw = school_search.group(1)

        school_type = school_search.group(2).upper()

        if school_type == 'COLLEGE' or school_type == 'UNIVERSITY':

            school_type = 'COLLEGE OR UNIVERSITY'

        if school_len_raw in num_dic_1:

            school_len = num_dic_1[school_len_raw]

        else:

            school_len = ''

        edu_years = school_len

    

    if school_len_search:

        school_len_raw = school_len_search.group(1).lower()

        units = school_len_search.group(3).lower()

        

        if school_len_raw.isnumeric():

            school_len = school_len_raw

        elif school_len_raw in num_dic:

            school_len = num_dic[school_len_raw]

        else:

            school_len = ''



        if 'year' in units and str(school_len).isnumeric():

            edu_years = float(school_len)



        if 'month' in units and str(school_len).isnumeric():

            edu_years = round(float(school_len)/12.0, 2)



        if 'semester' in units and str(school_len).isnumeric():

            edu_years = round(float(school_len)/2.0, 2)

        

    return edu_years, school_type



# Required Courses

def get_courses(text):

    courses = ''

    courses_search = re.search(r'(course in|courses in)\s*(\w+\s*and\s*\w+|\w+)', text, re.IGNORECASE)

    if courses_search:

        courses = courses_search.group(2).split(' and ')

    return '|'.join(courses).upper()





# Required Length of Courses

def get_courses_length(text):

    result = ''

    semester = ''

    quarter = ''

    semester_search = re.search(r'(\d{1,2})\s(semester?)', text, re.DOTALL|re.IGNORECASE)

    quarter_search = re.search(r'(\d{1,2})\s(quarter?)', text, re.DOTALL|re.IGNORECASE)

    courses_search = re.search(r'(course in|courses in)', text, re.IGNORECASE)

    if semester_search and courses_search:

        semester = semester_search.group(1)

    if quarter_search and courses_search:

        quarter = quarter_search.group(1)

    if semester.isnumeric() and quarter.isnumeric():

        result = str(quarter) + 'Q' + '/' + str(semester) + 'S'

    return result



# Education major

'''

Use the major list from https://github.com/fivethirtyeight/data/tree/master/college-majors

and word2Vec representing words as vectors

to calculate the cosine similarity between words and 

find the words more similar to the education majors presented in the majors-list.csv



'''

def word2vec(word):

    from collections import Counter

    from math import sqrt



    # count the characters in word

    cw = Counter(word)

    # precomputes a set of the different characters

    sw = set(cw)

    # precomputes the "length" of the word vector

    lw = sqrt(sum(c*c for c in cw.values()))



    # return a tuple

    return cw, sw, lw



def cosdis(v1, v2):

    # which characters are common to the two words?

    common = v1[1].intersection(v2[1])

    # by definition of cosine distance we have

    return sum(v1[0][ch]*v2[0][ch] for ch in common)/v1[2]/v2[2]



majors_df = pd.read_csv('../input/majorslist/majors-list.csv')

majors_list = majors_df['Major'].values

majors_list = [x for x in majors_list if str(x) != 'nan']

majors_list = list(filter(lambda x: re.sub("[^a-zA-Z]","", x), majors_list))

majors_list = [x.lower() for x in majors_list]



def get_education_major(text):

    

    single_list = text.split()

    double_list = list(map(' '.join, zip(single_list[:-1], single_list[1:])))

    text_list = single_list + double_list



    results = []

    threshold = 0.94

    for key in majors_list:

        for word in text_list:

            try:

                res = cosdis(word2vec(word), word2vec(key))

                if res > threshold:

                    #print("Found a word with cosine distance > 94 - {} : {} with original word: {}".format(res*100, word, key))

                    if key not in results:

                        results.append(key)

            except IndexError:

                pass

    return '|'.join(results)
# Extract features from Requirement Full Text

df_new['REQUIREMENT_SET_ID'], df_new['REQUIREMENT_SUBSET_ID'] = zip(*df_new['REQ_TEXT'].map(req_set_subset_id))

df_new['EDUCATION_YEARS'], df_new['SCHOOL_TYPE'] = zip(*df_new['REQ_TEXT'].map(get_school))

df_new['EDUCATION_MAJOR'] = df_new['REQ_TEXT'].map(get_education_major)

df_new['EXPERIENCE_LENGTH'] = df_new['REQ_TEXT'].map(get_experience_length)

df_new['FULL_TIME_PART_TIME'] = df_new['REQ_TEXT'].map(get_full_time_part_time)

df_new['EXP_JOB_CLASS_TITLE'] = df_new['REQ_TEXT'].map(get_job_title)

df_new['EXP_JOB_CLASS_FUNCTION'] = df_new['REQ_TEXT'].map(get_job_func)

df_new['COURSE_SUBJECT'] = df_new['REQ_TEXT'].map(get_courses)

df_new['COURSE_LENGTH'] = df_new['REQ_TEXT'].map(get_courses_length)
df_new.columns
df_new.to_csv('City of Los Angeles.csv', index=False)

#df_new = pd.read_csv('City of Los Angeles.csv')
class CleanText(BaseEstimator, TransformerMixin):

    def remove_mentions(self, input_text):

        return re.sub(r'@\w+', '', input_text)

    

    def remove_urls(self, input_text):

        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)

    

    def remove_punctuation(self, input_text):

        # Make translation table

        punct = string.punctuation

        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space

        return input_text.translate(trantab)



    def remove_digits(self, input_text):

        return re.sub('\d+', '', input_text)

    

    def to_lower(self, input_text):

        return input_text.lower()

    

    def remove_stopwords(self, input_text):

        stopwords_list = stopwords.words('english')

        # Some words which might indicate a certain sentiment are kept via a whitelist

        whitelist = ["n't", "not", "no"]

        words = input_text.split() 

        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 

        return " ".join(clean_words) 

    

    def stemming(self, input_text):

        porter = PorterStemmer()

        words = input_text.split() 

        stemmed_words = [porter.stem(word) for word in words]

        return " ".join(stemmed_words)

    

    def fit(self, X, y=None, **fit_params):

        return self

    

    def transform(self, X, **transform_params):

        clean_X = self.remove_mentions(X)

        clean_X = self.remove_urls(clean_X)

        clean_X = self.remove_punctuation(clean_X)

        clean_X = self.remove_digits(clean_X)

        clean_X = self.to_lower(clean_X)

        clean_X = self.remove_stopwords(clean_X)

        clean_X = self.stemming(clean_X)

        #clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords).apply(self.stemming)

        return clean_X
ct = CleanText()

clean_text = []

for file_name in valid_bulletins:

    with open(bulletins + '/' + file_name, encoding = "ISO-8859-1", errors = 'ignore') as f:

        file = f.read().replace('\t',' ')

        data = file.replace('\n',' ')

        data = ct.fit_transform(data)

        clean_text.append(data)

        

df_text = pd.DataFrame(clean_text)

df_text.columns = ['text']
# Now text of each bulletin is cleaned and stemmatized

list(df_text['text'].iloc[1:2])
# Sourse of words http://gender-decoder.katmatfield.com/results/083aa142-2c95-4930-b2d6-afe5033e1482



feminine_coded_words = [

    "agree",

    "affectionate",

    "child",

    "cheer",

    "collab",

    "commit",

    "communal",

    "compassion",

    "connect",

    "considerate",

    "cooperat",

    "co-operat",

    "depend",

    "emotiona",

    "empath",

    "feel",

    "flatterable",

    "gentle",

    "honest",

    "interpersonal",

    "interdependen",

    "interpersona",

    "inter-personal",

    "inter-dependen",

    "inter-persona",

    "kind",

    "kinship",

    "loyal",

    "modesty",

    "nag",

    "nurtur",

    "pleasant",

    "polite",

    "quiet",

    "respon",

    "sensitiv",

    "submissive",

    "support",

    "sympath",

    "tender",

    "together",

    "trust",

    "understand",

    "warm",

    "whin",

    "enthusias",

    "inclusive",

    "yield",

    "share",

    "sharin"

]



masculine_coded_words = [

    "active",

    "adventurous",

    "aggress",

    "ambitio",

    "analy",

    "assert",

    "athlet",

    "autonom",

    "battle",

    "boast",

    "challeng",

    "champion",

    "compet",

    "confident",

    "courag",

    "decid",

    "decision",

    "decisive",

    "defend",

    "determin",

    "domina",

    "dominant",

    "driven",

    "fearless",

    "fight",

    "force",

    "greedy",

    "head-strong",

    "headstrong",

    "hierarch",

    "hostil",

    "impulsive",

    "independen",

    "individual",

    "intellect",

    "lead",

    "logic",

    "objective",

    "opinion",

    "outspoken",

    "persist",

    "principle",

    "reckless",

    "self-confiden",

    "self-relian",

    "self-sufficien",

    "selfconfiden",

    "selfrelian",

    "selfsufficien",

    "stubborn",

    "superior",

    "unreasonab"

]
class gender_coded_words():

    

    def __init__(self, text):

        self.text = text

        

    def as_list(self):

        return self.text.split()

    

    def count_coded_words(self, gender_word_list):

        gender_biased_words = [word for word in self.as_list() 

                               for coded_word in gender_word_list 

                               if word.startswith(coded_word)]

        return (",").join(gender_biased_words), len(gender_biased_words)

    

    def gender_bias_score(self):

        masculine_words, masculine_words_count = self.count_coded_words(masculine_coded_words)

        feminine_words, feminine_words_count = self.count_coded_words(feminine_coded_words)

        coding_score = masculine_words_count - feminine_words_count

        coding = ''

        if coding_score == 0:

            if feminine_words_count:

                coding = "neutral"

            else:

                coding = ''

        elif coding_score < -3:

            coding = "strongly feminine-coded"

        elif coding_score < 0:

            coding = "feminine-coded"

        elif coding_score > 3:

            coding = "strongly masculine-coded"

        else:

            coding = "masculine-coded"

        return coding
df_text['gender_bias'] = list(map(lambda x: gender_coded_words(x).gender_bias_score(),df_text['text']))
df_text['gender_bias'].iplot(kind='hist', xTitle='Gender Tone',

                  yTitle='Count', title='Gender coded words distribution')
strict_requirements = ['must', 'requir', 'essenti', 'necess', 'need', 'expert', 'strong', 'profess']

soft_requirements = ['desir', 'familiar', 'capab', 'abl','inform', 'convers', 'practic', 'addit']
class requirement_tone():

    

    def __init__(self, text):

        self.text = text

        

    def as_list(self):

        return self.text.split()

    

    def count_words(self, req_word_list):

        words = [word for word in self.as_list() 

                               for req_word in req_word_list 

                               if word.startswith(req_word)]

        return (",").join(words), len(words)

    

    def req_tone_score(self):

        strict_words, strict_words_count = self.count_words(strict_requirements)

        soft_words, soft_words_count = self.count_words(soft_requirements)

        strict_words_score = strict_words_count - soft_words_count

        coding = ''

        if strict_words_score == 0:

            if strict_words_count:

                coding = "neutral"

            else:

                coding = ''

        elif strict_words_score < -3:

            coding = "very low demands"

        elif strict_words_score < 0:

            coding = "low demands"

        elif strict_words_score > 3:

            coding = "very high demands"

        else:

            coding = "high demands"

        return coding
df_text['req_tone'] = list(map(lambda x: requirement_tone(x).req_tone_score(),df_text['text']))
df_text['req_tone'].iplot(kind='hist', xTitle='Requirement Tone',

                  yTitle='Count', title='Requirements distribution')
def benefits(text):

    benefit_list= ['work-lif', 'flexibl', 'childcar', 

                   'parent', 'healthcar', 'matern', 'benefit', 'opportun']

    

    words = [word for word in text.split() if word in benefit_list]

    

    return len(words)



df_text['benefits'] = df_text['text'].apply(benefits)
df['DUTIES_len'] = df['JOB_DUTIES'].str.split().apply(len)

df['REQ_len'] = list(map(lambda x: len(''.join(x).split()),df['REQ_TEXT']))

df['FULL_TEXT_len'] = df_text['text'].str.split().apply(len)
df[['DUTIES_len','REQ_len','FULL_TEXT_len']].iplot(kind='box',

                  yTitle='Number of words', title='Job Description Length')
!pip install afinn

from afinn import Afinn
afn = Afinn(emoticons=False) 
# AFINN is a list of words rated for valence with an integer between minus five (negative) and plus five (positive).

# Simple examples:

print('Predicted Sentiment polarity:', afn.score('The movie was so bad'))

print('Predicted Sentiment polarity:', afn.score('The movie was so good'))
df['DUTIES_afinn_score'] = list(map(lambda x: afn.score(x), df['JOB_DUTIES']))

df['REQ_afinn_score'] = list(map(lambda x: afn.score(''.join(x)),df['REQ_TEXT']))

df['FULL_TEXT_afinn_score'] = list(map(lambda x: afn.score(x), df_text['text']))
df[['DUTIES_afinn_score', 'REQ_afinn_score','FULL_TEXT_afinn_score']].iplot(kind='box', 

                  yTitle='Sentiment polarity', title='Sentiment Scores')
print('{} % of job bulletines are classified as negative'.format(int(len(df[df['FULL_TEXT_afinn_score']<-1])*100/len(df))))
nltk.download('sentiwordnet')
# Simple example

text = list(swn.senti_synsets('awesome', 'a'))[0]

print('Positive Polarity Score:', text.pos_score())

print('Negative Polarity Score:', text.neg_score())
lemmatizer = WordNetLemmatizer()



def penn_to_wn(tag):

    """

    Convert between the PennTreebank tags to simple Wordnet tags

    """

    if tag.startswith('J'):

        return wn.ADJ

    elif tag.startswith('N'):

        return wn.NOUN

    elif tag.startswith('R'):

        return wn.ADV

    elif tag.startswith('V'):

        return wn.VERB

    return None

 

def swn_polarity(text):

    """

    Return a sentiment polarity: negative, positive or neutral

    """

 

    sentiment = 0.0

    tokens_count = 0

 

    raw_sentences = sent_tokenize(text)

    for raw_sentence in raw_sentences:

        tagged_sentence = pos_tag(word_tokenize(raw_sentence))

 

        for word, tag in tagged_sentence:

            wn_tag = penn_to_wn(tag)

            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):

                continue

 

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)

            if not lemma:

                continue

 

            synsets = wn.synsets(lemma, pos=wn_tag)

            if not synsets:

                continue

 

            # Take the first sense, the most common

            synset = synsets[0]

            swn_synset = swn.senti_synset(synset.name())

 

            sentiment += swn_synset.pos_score() - swn_synset.neg_score()

            tokens_count += 1

 

    # netral by default

    if not tokens_count:

        return 'neutral'

 

    # sum greater than 0 => positive sentiment

    if sentiment >= 0:

        return 'positive'

 

    # negative sentiment

    else:

        return 'negative'
df['DUTIES_swn'] = list(map(lambda x: swn_polarity(x), df['JOB_DUTIES']))

df['REQ_swn'] = list(map(lambda x: swn_polarity(''.join(x)),df['REQ_TEXT']))

df['FULL_TEXT_swn'] = list(map(lambda x: swn_polarity(x), df_text['text']))
df.iplot(kind='hist', barmode = 'group', histnorm = 'percent', dimensions=(1000, 300), 

         columns = ['DUTIES_swn', 'REQ_swn','FULL_TEXT_swn'], 

         yTitle='%', title='Sentiment Scores')
print('{} % of job bulletines are classified as negative'.format(int(len(df[df['FULL_TEXT_swn']=='negative'])*100/len(df))))
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

def get_vader_score(sent):

    # Polarity score returns dictionary

    ss = sid.polarity_scores(sent)

    ss.pop('compound', None)

    # return 'pos' - positive, 'neg' - negative and 'neu' - neutral

    return max(ss.items(), key=operator.itemgetter(1))[0]
df['DUTIES_sid'] = list(map(lambda x: get_vader_score(x), df['JOB_DUTIES']))

df['REQ_sid'] = list(map(lambda x: get_vader_score(''.join(x)),df['REQ_TEXT']))

df['FULL_TEXT_sid'] = list(map(lambda x: get_vader_score(x), df_text['text']))
df.iplot(kind='hist', barmode = 'group', histnorm = 'percent', dimensions=(1000, 300), 

         columns = ['DUTIES_sid', 'REQ_sid','FULL_TEXT_sid'], 

         yTitle='%', title='Sentiment Scores')