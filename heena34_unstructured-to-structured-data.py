import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

from wordcloud import WordCloud, STOPWORDS

import os

import re

print(os.listdir("../input/cityofla/CityofLA/Additional data"))
OTHER_DATA_PATH = '../input/cityofla/CityofLA/Additional data'

sample_csv = pd.read_csv(OTHER_DATA_PATH+'/'+'sample job class export template.csv')

sample_csv.head()
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





def get_salary(file_lines, filename):

    all_salaries = re.search('ANNUAL SALARY?\n+(.*)', file_lines)

    if all_salaries:

        all_salaries = all_salaries.group(0)

        salary = re.findall('\$\d+,\s*\d+\s+(?:to)\s+\$\d+,\d+|(?:\$\d+,\s*\d+\s+(?=\(flat-\s?rated\)))|(?:\$\d+,\s*\d+\s+and\s+\$\d+,\s*\d+\s+(?=\(flat-\s?rated\)))', all_salaries, re.I)

        if salary:

            salaries.append(salary)

        else:

            salaries.append('')

#             print(filename)

    else:

        salaries.append('')

    return None





def get_salary_DWP(file_lines):

    all_salaries = re.search(r'The salary range in the Department of Water'

                             r'and Power is?\s+(.*)', file_lines)

    if all_salaries:

        all_salaries = all_salaries.group(0)

        salary = re.findall('(\$\d+,\d+\s+to\s+\$\d+,\d+)', all_salaries, re.I)

        if salary:

            salaries_DWP.append(salary)

        else:

            salaries_DWP.append('')

    else:

        salaries_DWP.append('')

    return None

        



def get_application_deadline(file_lines, file_name):

#     print(file_name)

    deadline = re.findall('(\w+\s\d{,2}\:\d{,2}\s?(?:am|pm|a.m.|p.m.)\s\w+,\s\w+\s\d+,\s\d+\sto\s\d{,2}\:\d{,2}\s?(?:am|pm|a.m.|p.m.),\s\w+,\s\w+\s\d+,\s\d+)', file_lines, re.I)

    if deadline:

        deadlines.append(' OR '.join(i for i in deadline))

    else:

        deadline_text = re.search('(DEADLINE?\n+(.*))', file_lines)

        if deadline_text:

            deadline2 = re.search('(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY),\s(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s\d+,\s\d+', deadline_text.group(0))

            if deadline2:

                deadlines.append(deadline2.group(0))

            else:

                deadlines.append('')

        else:

            deadlines.append('')

    return None



def get_where_to_apply(file_lines, filename):

    apply = re.findall('(on-line|online)', file_lines, re.I)

    if apply:

        where_to_apply.append('Online')

    elif re.findall('Applications WILL ONLY BE ACCEPTED ON THE CITY APPLICATION FORM', file_lines, re.I):

        where_to_apply.append('on the city application form')

    elif re.findall('fax|email', file_lines, re.I):

        where_to_apply.append('fax or email')

    else:

        where_to_apply.append('')

        print(filename)

        

    
DATA_PATH = '../input/cityofla/CityofLA/Job Bulletins'

structured_data = pd.DataFrame()

col_names = ['FILE_NAMES', 'JOB_TITLE', 'JOB_CLASS_NO', 'JOB_DUTIES']



job_titles = []

file_names = []

job_class_nos = []

job_duties = []

open_dates = []

salaries = []

salaries_DWP = []

deadlines = []

where_to_apply = []



for index, file_name in enumerate(os.listdir(DATA_PATH)):

    with open(DATA_PATH+'/'+file_name,encoding = "ISO-8859-1") as f:

        file_lines = f.read()

        file_names.append(file_name)

        job_title = file_lines.split('\n')[0]

        append_data(job_titles, job_title)

        get_job_class_no(file_lines)       

        get_job_duty(file_lines)        

        get_open_dates(file_lines) 

        get_salary(file_lines, file_name)

        get_application_deadline(file_lines, file_name)

        get_where_to_apply(file_lines, file_name)

        get_salary_DWP(file_lines)

#         break



        

def make_data():

    structured_data = pd.DataFrame()

    structured_data['FILE_NAMES'] = file_names

    structured_data['JOB_TITLE'] = job_titles

    structured_data['JOB_CLASS_NO'] = job_class_nos

    structured_data['JOB_DUTIES'] = job_duties

    structured_data['OPEN_DATES'] = open_dates

    structured_data['SALARY'] = salaries

    structured_data['SALARY DWP'] = salaries_DWP

    structured_data['DEADLINE'] = deadlines

    structured_data['WHERE TO APPLY'] = where_to_apply

    return structured_data
structured_data = make_data()

structured_data.head()