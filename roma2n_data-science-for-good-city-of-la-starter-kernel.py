!pip install word2number
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import re

import os

from datetime import datetime

import string

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from word2number import w2n



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input/cityofla/CityofLA"))



# Any results you write to the current directory are saved as output.
!ls ../input/cityofla/CityofLA/Job\ Bulletins
!ls ../input/cityofla/CityofLA/Additional\ data
import random

import os

bulletins_dir = "../input/cityofla/CityofLA/Job Bulletins"

bulletins = os.listdir("../input/cityofla/CityofLA/Job Bulletins")

print("{} files".format(len(bulletins)))

for i in range(1):

    rand = random.randint(0, len(bulletins))

    file = os.path.join(bulletins_dir, bulletins[rand])

    with open(file) as f:

        for x in f.readlines():

            print(x)
from wand.image import Image as Img
Img(filename='../input/cityofla/CityofLA/Additional data/PDFs/2016/April 2016/PORT POLICE OFFICER 3221 Rev 041516.pdf', resolution=300)
additional = os.listdir("../input/cityofla/CityofLA/Additional data/")

print(additional)
import pandas as pd

path_additional = "../input/cityofla/CityofLA/Additional data/"

job_titles = pd.read_csv(os.path.join(path_additional, 'job_titles.csv'))

sample_job = pd.read_csv(os.path.join(path_additional, 'sample job class export template.csv'))

kaggle_data = pd.read_csv(os.path.join(path_additional, 'kaggle_data_dictionary.csv'))
job_titles.head()
job_titles.shape
sample_job.head()
kaggle_data
kaggle_data.shape

sample_job.shape
def process_bulletins(df, _bulletins):

    

    re_open_date = re.compile(r'(open date:)(\s+)((\d)?\d-\d\d-\d\d)', re.IGNORECASE|re.MULTILINE) 

    re_class_code = re.compile(r'class code:(\s+)([0-9a-z]+)', re.IGNORECASE|re.MULTILINE)

    re_requirements = re.compile(r'(REQUIREMENTS?/\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)', re.MULTILINE)

    re_duties = re.compile(r'(DUTIES)(.*)(REQ[A-Z])', re.MULTILINE)

    re_education_years = re.compile(r'(.*)college(.*)university(.*)\s+(\d+)\s+(.*)(semester)', re.IGNORECASE|re.MULTILINE)

    re_school_type = re.compile(r'(.*)qualifying(.*)education(.*)from(.*)accredited(.*)', re.IGNORECASE|re.MULTILINE)

    re_edu_major = re.compile(r'(major|degree) in(.*)', re.IGNORECASE|re.MULTILINE)

    re_exp_length = re.compile(r'(.*)years(.*)full(.*)experience(.*)', re.IGNORECASE|re.MULTILINE)

    re_fulltime_parttime = re.compile(r'(.*)(full|part)(.*)experience(.*)', re.IGNORECASE|re.MULTILINE)

    re_exp_job_class_title = re.compile(r'(.*)experience as(.*)', re.IGNORECASE|re.MULTILINE)

    re_course_count = re.compile(r'(.*)completion of(.*)(course?)(.*)', re.IGNORECASE|re.MULTILINE)

    re_drivers_license_req = re.compile(r'(.*driver\'s(\s*)license)', re.IGNORECASE|re.MULTILINE)

    re_addtl_lic = re.compile(r'as a (licensed)(.+[;,.])', re.IGNORECASE|re.MULTILINE)

    re_exam_type = re.compile(r'THIS EXAM( |INATION )IS TO BE GIVEN(.*)BASIS', re.IGNORECASE|re.MULTILINE)

    re_entry_salary_gen = re.compile(r'(\$\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?')

    re_entry_salary_dwp = (r' (Water and Power is) (\$\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?')

    

    for filename in _bulletins:

        with open(bulletins_dir + "/" + filename, 'r', errors='ignore') as f:

            file = f.read().replace('\t', '')

            # job_class_title

            position = [x for x in file.split('\n') if x.isupper()][0]

            

            file = file.replace('\n', ' ')

            #file = ' '.join([x + '\n' for x in file.split(' ') if x.isupper()])

            #print(file)

            # open_date

            open_date_found = re.search(re_open_date, file)

            if open_date_found is not None:

                open_date = datetime.strptime(open_date_found.group(3), '%m-%d-%y')

            else:

                open_date = np.nan

              

            # job_class_no

            class_no_found = re.search(re_class_code, file)

            if class_no_found is not None:

                class_no = class_no_found.group(2)

            else:

                class_no = np.nan

                

            # requirements

            requirements_found = re.search(re_requirements, file)

            if requirements_found is not None:

                requirements = requirements_found.group(2)

            else:

                requirements = ""#re.search('(.*)NOTES?', re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)', file)[0][1][:1200]).group(1)

            #print(requirements)

            

            # job_duties

            duties_found = re.search(re_duties, file)

            if duties_found is not None:

                duties = duties_found.group(2)

            else:

                duties = ""

                

            # education_years

            education_years_found = re.search(re_education_years, requirements)

            if education_years_found is not None:

                education_years = education_years_found.group(4).strip()

            else:

                education_years = np.nan

                

            # school_type

            school_type_found = re.search(re_school_type, file)

            if school_type_found is not None:

                school_type = school_type_found.group(4)

                school_type = school_type.split('accredited')[0]

            else:

                school_type = ''

                

            # education_major

            # course_subjects

            course_subjects = []

            education_major_found = re.search(re_edu_major, requirements)

            if education_major_found is not None:

                education_major = education_major_found.group(2)

                education_major = education_major.strip().strip('.').split(';')[0]

                education_major = education_major.replace('and/or', 'or').replace('landscape', 'lnadscape').split('and')[0]

                education_major = education_major.replace('lnadscape', 'landscape').split( 'or in' )[0]

                education_major = education_major.split( 'or upon' )[0]

                education_major = education_major.split( '.' )



                education_major = string.capwords(education_major[0])



                education_major_repl = education_major.replace(' or ', ',').replace(' Or ', ',').replace('A Related Field','').strip(',').split(',')

                if 0 < len(education_major_repl):

                    for major in education_major_repl:

                        if 0 < len(major.split(' or')):

                            for submajor in major.split(' or'):

                                course_subjects.append(submajor.strip())

                        else :

                            course_subjects.append(major.strip())                   

            else:

                education_major = '-'

                

            # experience_length

            experience_length_found = re.search(re_exp_length, file)

            if experience_length_found is not None:

                experience_length = experience_length_found.group(1).split()       

                if len(experience_length) > 0:

                    experience_length = experience_length[-1]

                    if '.' in experience_length:

                        experience_length = experience_length.split('.')[-1]

                    try:

                        experience_length = w2n.word_to_num(re.sub(r'[^a-zA-Z]+', '', experience_length))

                    except:

                        pass

                else:

                    experience_length = np.nan

            else:

                experience_length = np.nan

            

            # fulltime_parttime

            fulltime_parttime_found = re.search(re_fulltime_parttime, requirements)

            if fulltime_parttime_found is not None:

                fulltime_parttime = fulltime_parttime_found.group(2)         

                fulltime_parttime = (fulltime_parttime + '_TIME').upper()

            else:

                fulltime_parttime = None

                

            # exp_job_class_title

            exp_job_class_title_found = re.search(re_exp_job_class_title, requirements)

          

            if exp_job_class_title_found is not None:



                exp_job_class_title = exp_job_class_title_found.group(2)         

                exp_job_class_title = exp_job_class_title.strip().strip('.').lstrip('a ').lstrip('an ')

                broken = 0 

                if len(exp_job_class_title.split(';')) > 0:

                    for line in exp_job_class_title.split(';'):

                        if len(line.split('with')) > 0:

                            for sub_line in line.split('with'):

                                if 'Los Angeles' not in sub_line:

                                    sub_line = sub_line.strip().strip('.').lstrip('a ').lstrip('an ').split(' or in')

                                    sub_line = sub_line[0]

                                    if '.' not in sub_line:

                                        exp_job_class_title = string.capwords(sub_line.strip().strip('.').strip(','))

                                        exp_job_class_title = exp_job_class_title.split('.')

                                        exp_job_class_title = exp_job_class_title[0]

                                        broken = 1 

                                        break

                        if broken == 1:

                            break

                else: 

                    exp_job_class_title = '-'



            # course_count

            course_count_found = re.search(re_course_count, requirements)

            course_count = 0

            if course_count_found is not None:

                course_count = course_count_found.group(2)        

                course_count = course_count.split('course')[0]

                

                if len(course_count.strip().split(' ')) > 0:

                    for line in course_count.split(' '):

                        try:

                            course_count = w2n.word_to_num(line)

                            break

                        except:

                            continue

    

            # course_subject

            course_subject = '|'.join(course_subjects)

            

            # drivers_license_req

            drivers_license = []

            drivers_license_req_found = re.search(re_drivers_license_req, file)

            if drivers_license_req_found is not None:

                drivers_license_req = drivers_license_req_found.group()

                licenses = re.findall(r'([C,c]lass ([a-zA-Z]))', drivers_license_req)

                if len(licenses) > 0:

                    for license in licenses:

                        drivers_license.append(license[1])



                if('may' not in drivers_license_req and ('valid' in drivers_license_req or 'require' in drivers_license_req)):

                    drivers_license_req = 'R'

                elif('may' in drivers_license_req and ('valid' in drivers_license_req or 'require' in drivers_license_req)):

                    drivers_license_req = 'P'

            else:

                drivers_license_req = ''

            

            # driv_lic_type

            driv_lic_type = ','.join(set(drivers_license))

            

            # addtl_lic

            addtl_lic_found = re.search(re_addtl_lic, file)

            if addtl_lic_found is not None:

                addtl_lic = addtl_lic_found.group(2)

                addtl_lic = addtl_lic.strip().split('.')[0]

                addtl_lic = addtl_lic.strip().split('issued')[0]

                addtl_lic = addtl_lic.strip().split(';')[0]

            else:

                addtl_lic = np.nan

                

            # exam_type

            exam_type = '-'

            exam_types = []

            exam_type_found = re.search(re_exam_type, file)

            if exam_type_found is not None:

                exam_type = exam_type_found.group(2).upper()

                if('OPEN COMPETITIVE' in exam_type and 'INTERDEPARTMENTAL PROMOTION' in exam_type):

                    exam_types.append('OPEN_INT_PROM')

                else:    

                    if('OPEN COMPETITIVE' in exam_type):

                        exam_types.append('OPEN')

                    if('INTERDEPARTMENTAL PROMOTION' in exam_type):

                        exam_types.append('INT_DEPT_PROM')



                if(' DEPARTMENTAL PROMOTION' in exam_type):

                    exam_types.append('DEPT_PROM')

            else:

                exam_types.append('-')

            exam_type = ','.join(exam_types)

            

            # entry_salary_gen

            salary_gen = 0

            try:

                if re.search(re_entry_salary_gen, file).group():

                    salary_gen = []

                    if re.search(re_entry_salary_gen, file).group(1):

                        salary_gen.append((re.search(re_entry_salary_gen, file).group(1)).strip())

                    if re.search(re_entry_salary_gen, file).group(5):

                        salary_gen.append(re.search(re_entry_salary_gen, file).group(5).strip())

                    salary_gen = '-'.join(salary_gen)

            except:

                salary_gen = 0

                

            # entry_salary_dwp

            salary_dwp = 0

            try:

                if re.search(re_entry_salary_dwp, file).group():

                    salary_dwp = []

                    if re.search(re_entry_salary_dwp, file).group(2):

                        salary_dwp.append((re.search(re_entry_salary_dwp, file).group(2)).strip())

                    if re.search(re_entry_salary_dwp, file).group(6):

                        salary_dwp.append((re.search(re_entry_salary_dwp, file).group(6)).strip())

                    salary_dwp = '-'.join(salary_dwp)

            except:

                salary_dwp = 0

            

            df = df.append({'FILE_NAME': filename, 'JOB_CLASS_TITLE': position, 'JOB_CLASS_NO': class_no, 'JOB_DUTIES': duties,

                            'EDUCATION_YEARS': education_years, 'SCHOOL_TYPE': school_type, 'EDUCATION_MAJOR': education_major,

                            'EXPERIENCE_LENGTH': experience_length, 'FULL_TIME_PART_TIME': fulltime_parttime, 'EXP_JOB_CLASS_TITLE': exp_job_class_title,

                            'COURSE_COUNT': course_count, 'COURSE_SUBJECT': course_subject, 'DRIVERS_LICENSE_REQ': drivers_license_req,

                            'DRIV_LIC_TYPE': driv_lic_type, 'ADDTL_LIC': addtl_lic, 'EXAM_TYPE': exam_type,

                            'ENTRY_SALARY_GEN': salary_gen, 'ENTRY_SALARY_DWP': salary_dwp,

                            'OPEN_DATE': open_date}, ignore_index=True)

            

            #break

    return df
df = pd.DataFrame(columns=['FILE_NAME', 'JOB_CLASS_TITLE', 'JOB_CLASS_NO', 'JOB_DUTIES',

                            'EDUCATION_YEARS', 'SCHOOL_TYPE', 'EDUCATION_MAJOR', 'EXPERIENCE_LENGTH', 'FULL_TIME_PART_TIME', 

                           'EXP_JOB_CLASS_TITLE', 'COURSE_COUNT', 'COURSE_SUBJECT', 'DRIVERS_LICENSE_REQ', 'DRIV_LIC_TYPE', 'ADDTL_LIC',

                           'EXAM_TYPE', 'ENTRY_SALARY_GEN', 'ENTRY_SALARY_DWP', 'OPEN_DATE'])

df = process_bulletins(df, bulletins)

df
df.to_csv("competition_output.csv")

kaggle_data.to_csv("data_dictionary.csv")
df.describe()