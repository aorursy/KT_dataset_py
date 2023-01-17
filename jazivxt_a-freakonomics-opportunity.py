import numpy as np

import pandas as pd

import glob, re



path = '../input/cityofla/CityofLA/Additional data/'

template = pd.read_csv(path + 'sample job class export template.csv')

job_bulletins = pd.DataFrame(glob.glob('../input/cityofla/CityofLA/Job Bulletins/**'), columns=['Path'])



job_bulletins['Text'] = job_bulletins['Path'].map(lambda x: open(x, 'r', encoding='latin-1').read())

for c in template.columns:

    job_bulletins[c] = -1

    job_bulletins[c] = job_bulletins[c].astype(template[c].dtype)

    

job_bulletins['FILE_NAME'] = job_bulletins['Path'].map(lambda x: str(x).split('/')[-1])

job_bulletins['JOB_CLASS_TITLE'] = job_bulletins['Text'].map(lambda x: str(x).strip('\n\t ').split('\n')[0].split('\t')[0])

job_bulletins.head(1)
def get_job_class_no(t):

    try:

        m = re.search(r'Class Code:[ ]*([0-9]*)', t)

        return m.group(1).zfill(4)

    except:

        return -1

job_bulletins['JOB_CLASS_NO'] = job_bulletins['Text'].map(lambda x: get_job_class_no(str(x)))

job_bulletins.head(1)
def get_requirements(t):

    try:

        t = t.replace('\n\n','~')

        r = ['REQUIREMENTS/MINIMUM QUALIFICATIONS', 'REQUIREMENTS/ MINIMUM QUALIFICATIONS', 'REQUIREMENTS/MINUMUM QUALIFICATIONS', 

             'REQUIREMENT/MINIMUM QUALIFICATION', 'REQUIREMENT/ MINIMUM QUALIFICATION', 'REQUIREMENTS', 'REQUIREMENT']

        for t1 in r:

            if t1 in t:

                t = t.replace(t1, 'REQUIREMENTS')

                break

        m = re.search(r'REQUIREMENTS~(.*?)~', t, flags=re.S | re.M)

        requirements = m.group(1).split('\n')

        return requirements

    except:

        return -1

job_bulletins['REQUIREMENT_SET_ID'] = job_bulletins['Text'].map(lambda x: get_requirements(str(x)))

job_bulletins.head(1)
def get_job_duties(t):

    try:

        t = t.replace('\n\n','~')

        m = re.search(r'DUTIES~(.*?)~', t, flags=re.S | re.M)

        return m.group(1)

    except:

        return -1

job_bulletins['JOB_DUTIES'] = job_bulletins['Text'].map(lambda x: get_job_duties(str(x)))

job_bulletins.head(1)
#EDUCATION_YEARS

#SCHOOL_TYPE
def get_major(t):

    try:

        t = t.replace('\n','~')

        m = re.search(r'[Mm]ajor in(.*?)~', t, flags=re.S | re.M)

        return m.group(1)

    except:

        return -1

job_bulletins['EDUCATION_MAJOR'] = job_bulletins['Text'].map(lambda x: get_job_duties(str(x)))

job_bulletins.head(1)
"""EXPERIENCE_LENGTH

FULL_TIME_PART_TIME

EXP_JOB_CLASS_TITLE

EXP_JOB_CLASS_ALT_RESP

EXP_JOB_CLASS_FUNCTION

COURSE_COUNT

COURSE_LENGTH

COURSE_SUBJECT

MISC_COURSE_DETAILS

DRIVERS_LICENSE_REQ

DRIV_LIC_TYPE

ADDTL_LIC"""
EXAM_TYPE = {'OPEN_INT_PROM' : ['ON AN INTERDEPARTMENTAL PROMOTIONAL AND OPEN COMPETITIVE BASIS', 

                                'ON AN INTERDEPARTMENTAL PROMOTIONAL AND AN OPEN COMPETITIVE BASIS', 

                                'INTERDEPARTMENTAL PROMOTIONAL AND OPEN COMPETITIVE BASIS',

                                'INTERDEPARTMENTAL PROMOTIONAL AND AN OPEN COMPETITIVE BASIS', 

                                'AN INTERDEPARTMENTAL PROMOTIONAL AND AN OPEN COMPETITIVE BASIS'],

             'OPEN' :['OPEN COMPETITIVE BASIS', 'ONLY ON AN OPEN COMPETITIVE BASIS', 'ON AN OPEN COMPETITIVE BASIS', 'AN OPEN COMPETITIVE BASIS'],

             'DEPT_PROM' : ['ONLY ON A DEPARTMENTAL PROMOTIONAL BASIS', 'ON A DEPARTMENTAL PROMOTIONAL BASIS'], 

             'INT_DEPT_PROM' : ['ON AN INTERDEPARTMENTAL PROMOTIONAL BASIS', 'INTERDEPARTMENTAL PROMOTIONAL BASIS']

}



def get_exam_type(t):

    global EXAM_TYPE

    for etype in EXAM_TYPE:

        for subt in EXAM_TYPE[etype]:

            if subt in t:

                return etype

                break



job_bulletins['EXAM_TYPE'] = job_bulletins['Text'].map(lambda x: get_exam_type(str(x)))

job_bulletins['EXAM_TYPE'].value_counts()
def get_salary(t):

    try:

        t = t.replace('\n\n','~')

        m = re.search(r'ANNUAL SALARY~(.*?)~', t, flags=re.S | re.M)

        return m.group(1)

    except:

        return -1

job_bulletins['ENTRY_SALARY_GEN'] = job_bulletins['Text'].map(lambda x: get_salary(str(x)))

job_bulletins.head(1)



#ENTRY_SALARY_DWP
def get_open_date(t):

    try:

        m = re.search(r'Open Date:[ ]*(\d{2}-\d{2}-\d{2})', t)

        return m.group(1)

    except:

        return -1

job_bulletins['OPEN_DATE'] = job_bulletins['Text'].map(lambda x: get_open_date(str(x)))

job_bulletins.head(1)
import PyPDF2



def getPDFText(path):

    t = ''

    pdf = PyPDF2.PdfFileReader(open(path, 'rb'))

    for p in range(pdf.getNumPages()):

        p = pdf.getPage(p)

        t += p.extractText()

    return t

    

titles = pd.read_csv('../input/cityofla/CityofLA/Additional data/job_titles.csv', header=None, names=['Titles'])

promo_files = glob.glob('../input/cityofla/CityofLA/Additional data/City Job Paths/**.pdf')

job_path = {f.split('/')[-1].split('.')[0].upper().replace('_',' '): getPDFText(f) for f in promo_files}

job_bulletins['JOB_PATH'] = job_bulletins['JOB_CLASS_TITLE'].map(job_path)
job_bulletins.to_csv('job_bulletins.csv', index=False)
data_dictionary = pd.read_csv(path + 'kaggle_data_dictionary.csv')

#Update data dictionary with any new findings

data_dictionary.head()
from wand.image import Image as Img

Img(filename='../input/cityofla/CityofLA/Additional data/PDFs/2017/july 2017/July 21/ARTS ASSOCIATE 2454 072117 REV 072817.pdf', resolution=300)