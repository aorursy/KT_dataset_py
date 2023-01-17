import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

from wordcloud import WordCloud, STOPWORDS

import os

import re
OTHER_DATA_PATH = '../input/cityofla/CityofLA/Additional data'

sample_csv = pd.read_csv(OTHER_DATA_PATH+'/'+'sample job class export template.csv')

sample_csv.head()
# Helpful Functions



def append_data(list_name, data_to_append):

    if data_to_append:

        list_name.append(data_to_append)

    else:

        list_name.append('')

    return None



def get_job_class_no(file_lines):

    job_class_no = re.search('Class Code (\d+)', file_lines)

    if job_class_no:

        job_class_nos.append(job_class_no.group(1))

    else:

        job_class_nos.append('')

    return None



def get_job_duty(file_lines):

    job_duty = re.search('DUTIES(\W+)(.*\n)', file_lines)

    if job_duty:

        try:

            job_duties.append(str(job_duty.group(0).split('\n')[2]))

        except Exception as e:

            job_duty = re.search('Duties include:[^.]*', file_lines)

            job_duties.append(job_duty.group(0).split(': ')[1])

    else:

        job_duties.append('')

    return None



def get_open_dates(file_lines):

    open_date = re.search('Open Date:[^.](.*\n)', file_lines)

    if open_date:

        open_dates.append(open_date.group(0).split(':')[1].lstrip().rstrip('\n'))

    else:

        open_dates.append('')

    return None

DATA_PATH = '../input/cityofla/CityofLA/Job Bulletins'



structured_data = pd.DataFrame()

col_names = ['FILE_NAMES', 'JOB_TITLE', 'JOB_CLASS_NO', 'JOB_DUTIES']



job_titles = []

file_names = []

job_class_nos = []

job_duties = []

open_dates = []



for index, file_name in enumerate(os.listdir(DATA_PATH)):

    with open(DATA_PATH+'/'+file_name, encoding = "ISO-8859-1") as f:

        file_lines = f.read()

        file_names.append(file_name)

        job_title = file_lines.split('\n')[0]

        append_data(job_titles, job_title)

        get_job_class_no(file_lines)       

        get_job_duty(file_lines)        

        get_open_dates(file_lines) 

#         get_annual_salary(file_lines)
def make_data():

    structured_data = pd.DataFrame()

    structured_data['FILE_NAMES'] = file_names

    structured_data['JOB_TITLE'] = job_titles

    structured_data['JOB_CLASS_NO'] = job_class_nos

    structured_data['JOB_DUTIES'] = job_duties

    structured_data['OPEN_DATES'] = open_dates

    return structured_data
structured_data = make_data()

structured_data.head()