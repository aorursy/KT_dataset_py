# Packages

import os, re, datetime

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# directories

bulletin_dir = "../input/cityofla/CityofLA/Job Bulletins/"

titles = pd.read_csv('../input/cityofla/CityofLA/Additional data/job_titles.csv')
def find_salary(lines):

    """

        Searches a salary pattern in the array lines

        input: array of strings

        output: str

    """

    for line in lines:

        flat_rate = re.search('(?i)flat|rated', line)

        if flat_rate and flat_rate[0]:

            salary = re.search('\$\d+,\d+', line)

            if salary:

                return salary[0] + ' (flat-rated)'



        else:

            ranges = re.split('and|;|,\s', line)

            for range in ranges:

                bounds = re.findall('\$\d+,\d+', range)

                if len(bounds) == 1: # in case of a flat-rated salary which was not properly noted

                    return bounds[0] + ' (flat-rated)'

                if len(bounds) == 2: # found a range.

                    return '-'.join(bounds) 
def find_exam_type(line):

    """

        Find's the exam type of the job bulletin

        input: string

        output: string

    """

    exam_type = []

    found = re.findall(r'OPEN COMPETITIVE BASIS|INTERDEPARTMENTAL PROMOTIONAL|DEPARTMENTAL PROMOTIONAL BASIS', line, re.IGNORECASE) 



    if "OPEN COMPETITIVE BASIS" in found:

        exam_type.append("OPEN")



    if "INTERDEPARTMENTAL PROMOTIONAL" in found:

        exam_type.append("INT_DEPT_PROM")



    if "DEPARTMENTAL PROMOTIONAL BASIS" in found:

        exam_type.append("DEPT_PROM")



    exam_type = ', '.join(exam_type)

    return exam_type
def gather_data(df):

    """

    Gathers all the needed data of a job bulletin contained in a datafram

    input: pandas.DataFrame

    output: list of objects

    """

    

    data_reqs = []

    data = {}

    default = {

        'JOB_CLASS_TITLE': '',

        'ENTRY_SALARY_GEN': None,

        'ENTRY_SALARY_DWP': None,

        'JOB_DUTIES': '',

        'EXAM_TYPE': ''

    }

    

    test = df[0].str.contains('(?i)requirements')



    for idx, line in enumerate(df[0]):     

        if idx == 0:

            # Title

            title_regex = re.search('\w([^\s]{2,}[^\n|\t])+\S', line)

            if title_regex:

                data['JOB_CLASS_TITLE'] = title_regex[0].strip()

            

        # Class Code

        code_line_reg = re.search('(?i)Class\s+Code:.*', line)        

        if code_line_reg:     

            code = re.search('\d+', code_line_reg[0])

            if code:

                data['JOB_CLASS_NO'] = code[0]

                

        

        # Salary

        s1_salary_line =  re.search('ANNUAL\s?SALARY', line)

        if s1_salary_line:

            section = get_section(df, idx + 1)

            s1 = find_salary(section)

            if s1:

                data['ENTRY_SALARY_GEN'] = s1

        

        s2_salary_line = re.search('(?i).*Department.*Water.*Power.*', line)

        if s2_salary_line:

            s2 = find_salary([s2_salary_line[0]])

            if s2:

                data['ENTRY_SALARY_DWP'] = s2

                 

        # Open Date

        open_date_line = re.search('(?i)((Open Date)|(Date)):.*', line)

        if open_date_line:

            open_date = re.search('(?<=:)[^\(\)]*', open_date_line[0])

            data['OPEN_DATE'] = open_date[0].strip()

            

        # Duties

        if "DUTIES" in line:

            section = get_section(df, idx + 1)

            data['JOB_DUTIES'] = ''.join(section)

            

        if 'APPLICATION DEADLINE' in line:

            deadline_section = get_section(df, idx + 1)

            data['DEADLINE_DATE'] = find_deadline(deadline_section) 

            

        # exam type

        if 'THIS EXAMINATION IS TO BE GIVEN' in line.upper():

            exam = find_exam_type(df[0][idx + 1])

            if exam:

                data['EXAM_TYPE'] = exam



        # Requirements and Education / Courses / Experience / Extra requirements ... 

        req_regex = re.search('REQUIREMENT', line)    

        if req_regex:

            reqs, notes = get_section(df, idx + 1, notes=True)

            req_sets = extract_sets(reqs)

            requirements = extract_extra_reqs(req_sets, notes)





    for idx, req in enumerate(requirements):  #iterate over the requirement sets

        entry = { 'REQUIREMENT_SET_ID': idx + 1, 'REQUIREMENT_SUBSET_ID': 'A' }

        entry.update(default)

        entry.update(data)

        entry.update(req)

        data_reqs.append(entry)          



    return data_reqs
def find_deadline(lines):

    """

    Find the job application deadline

    input: array of strings

    output: string

    """



    for line in lines:

        date = re.search("(?i)(MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY|SUNDAY).*\d{4}", line)

        if date:

            return date[0]
def find_driver_license(text):

    """

        Searches for a driver license requirement and the type of the license required

        input: string

        output: tuple of strings

    """

    

    may_req = re.search("may[^\.]*requir[^\.]*driver[^\.]*license", text)

    req = re.search('(requir[^\.]*driver[^\.]*license)|(driver[^\.]*license[^\.]*requir)', text)

    

    if may_req:

        driver  = 'M'

    elif req:

        driver = 'R'

    else:

        driver = ''

        

    driver_types = []    

    if driver == 'M' or driver == 'R':

        driver_types = re.findall('(?i)class\s\w', text)

        driver_types = list(dict.fromkeys(driver_types)) # Removes any duplicates

        

    driver_types = ', '.join(driver_types)

    

    return driver, driver_types
def find_license(text):   

    """

    Finds additional licenses

    input: str

    output: str

    """

    text = text.lower()

    if "license" in text:

        if "driver's" not in text:

            look_behind = re.search('([A-Z]+\w*\s)+license', text)

            look_ahead = re.search('license \w+ ([A-Z]+\w*\s)+', text)

            if look_behind:

                return look_behind[0].upper()

            if look_ahead:

                return look_ahead[0].upper()

    if 'medical certificate' in text:

        return 'Medical Certificate'
def extract_sets(req):

    """Extract requirements sets and subsets (D1, D2)"""

    mandatories = []

    optionals = []

    

    previous_conditional = False

    for line in req:

        or_conditional = re.search('(?i)or\s?$', line)

        

        if previous_conditional:

            idx = len(optionals) - 1

            optionals[idx].append(line)

        elif or_conditional:

            optionals.append([line])

        else:

            mandatories.append(line)

            

        previous_conditional = True if or_conditional else False

    

    # This loop we is so we create combinatories of the optional requirements

    # e.g: we might have optionals: [[a, b],[c, d]]. We would then generate ac, ad, bc, bd.

    req_sets = []

    for idx, optional in enumerate(optionals):

        for conditional in optional:

            requirements = mandatories.copy()

            requirements.append(conditional)

            

            for idx2, optional_2 in enumerate(optionals):

                if idx == idx2:

                    continue

                for conditional2 in optional_2:

                    requirements.append(conditional2)

            req_sets.append(requirements)

    if not len(req_sets):

        req_sets = [mandatories]

    return req_sets  
def extract_extra_reqs(req_sets, notes):

    """

        Extracts requirements such as:

        Education, Driver's Liscence, Expirence, Course Info, etc.

        (F, G, H, I, J, L, M, N, O, P1, P2, Q)

        input: 

            req_sets: array of strings

            notes: array of strings

        output: list of objects            

    """

    requirements = []

    driver = ''

    course_count = 0

    licenses = []

    

    # Extract driver's license and other licenses from notes

    for line in notes:

        if not driver:

                driver, driver_types = find_driver_license(line)

                license = find_license(line)

                if license:

                    licenses.append(license)



    

    for req_set in req_sets:

        data = {}

        

        # REQUIREMENT_TEXT

        data['REQUIREMENT_TEXT'] = ' | '.join(req_set)

        

        for line in req_set:

            

            # EDUCATION_YEARS, SCHOOL TYPE, EDUCATION_MAJOR

            education = re.search('(?i).*(education)|(college)|(school)|(university).*', line)

            if education:

                edu_type = re.search('(?i)(university)|(college)|(school)', line)

                if 'college or university' in line.lower():

                    data['SCHOOL_TYPE'] = 'COLLEGE OR UNIVERSITY'

                elif edu_type:

                    data['SCHOOL_TYPE'] = edu_type[0].upper()

                    

                major = re.search('((degree)|(major)) in(\s[A-Z]+\w*)+', line)

                if major:

                    data['EDUCATION_MAJOR'] = re.search('(\s[A-Z]+\w*)+', major[0])[0].upper()

                

                years = re.search('(?i)((\d+)|(\w)+).(years?)', line)

                if years:

                    data['EDUCATION_YEARS'] = years[0].upper()

                    

                semesters = re.search('(?i)((\d+)|(\w)+).(semesters?)', line)

                if semesters:

                    numb = re.search('\d+', semesters[0])

                    if numb:

                        data['EDUCATION_YEARS'] = numb[0]



    

            # EXPERIENCE_LENGTH

            experience = re.search('(?i).*experience.*', line)

            if experience:



                length = re.search('(?i)((\d+)|(\w+)) years?', experience[0])

                if length:

                    data['EXPERIENCE_LENGTH'] = length[0].upper()



                

                time = re.search('(?i)((full)|(part))-?\s?time', experience[0])

                if time:

                    data['FULL_TIME_PART_TIME'] = time[0].upper()

                    

                title = re.search('((as an?)|(at the level of an?))(\s[A-Z]+\w*)+', experience[0])

                if title:

                    title = re.search('(\s[A-Z]+\w*)+', title[0])[0]

                    data['EXP_JOB_CLASS_TITLE'] = title.upper()



            # DRIVER'S LICENSE            

            if not driver:

                driver, driver_types = find_driver_license(line)

                

            # OTHER LICENSES

            license = find_license(line)

            if license:

                licenses.append(license)

                

            # courses

             #COURSE_LENGTH COURSE_SUBJECT MISC_COURSE_DETAILS, COURSE_COUNT

            if re.search("(?i)courses?", line):

                title = re.search('([A-Z]+\w*\s)+course', line)

                if title:

                    course_count += 1

                    data['COURSE_SUBJECT'] = title[0]

                misc = re.search("course.*is equivalent to", line)

                if misc:

                    misc = re.search("(?=equivalent to).*", line)

                    data['MISC_COURSE_DETAILS'] = misc[0]

                

                course_length = re.search(

                    '((\d+)|(\w+))\s((semesters?)|(quarters?)|(years?)(hours?))(?:.{0,50}course)', line)

                if course_length:

                    data['COURSE_LENGTH'] = course_length[0]



        licenses = list(dict.fromkeys(licenses))

        data['COURSE_COUNT'] = course_count if course_count else None

        data['ADDTL_LIC'] = ', '.join(licenses)

        data['DRIVERS_LICENSE_REQ'] = driver

        data['DRIV_LIC_TYPE'] = driver_types

        requirements.append(data)

    

    return requirements
def get_section(df, idx, notes=False):

    """Extracts the section bellow a header"""

    ln = df[0][idx]

    sentences = []

    

    while not ln.isupper(): 

        if ln:

            sentences.append(ln)

        idx += 1

        ln = df[0][idx]

    

    if notes and re.search('(?i)note', ln):

        notes = get_section(df, idx + 1)

        return sentences, notes

    elif notes:

        return sentences, []

    

    return sentences
def get_csv(directory):

    values = []



    for idx, filename in enumerate(os.listdir(directory)):

        with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as f:

            lines = f.readlines()



            df = pd.DataFrame(lines)

            df = df.replace('\n','', regex=True).replace('\t', '', regex=True)

            text = ' '.join(lines)

            

            data_reqs = gather_data(df)

            for entry in data_reqs:

                entry['FILE_NAME'] = filename      

                values.append(entry)

                

    data_df = pd.DataFrame(values)

    data_df['OPEN_DATE'] = pd.to_datetime(data_df['OPEN_DATE'])

    data_df = data_df[['FILE_NAME', 'JOB_CLASS_TITLE', 'JOB_CLASS_NO', 

                       'REQUIREMENT_SET_ID', 'REQUIREMENT_SUBSET_ID', 

                       'JOB_DUTIES', 'EDUCATION_YEARS', 'SCHOOL_TYPE', 

                       'EDUCATION_MAJOR', 'EXPERIENCE_LENGTH', 'FULL_TIME_PART_TIME',  

                       'EXP_JOB_CLASS_TITLE', 'COURSE_LENGTH', 'COURSE_SUBJECT', 

                       'MISC_COURSE_DETAILS', 'COURSE_COUNT',

                       'DRIVERS_LICENSE_REQ', 'DRIV_LIC_TYPE', 'ADDTL_LIC', 

                       'ENTRY_SALARY_GEN', 'ENTRY_SALARY_DWP', 'OPEN_DATE', 

                       'DEADLINE_DATE','EXAM_TYPE', 'REQUIREMENT_TEXT']]

    return data_df
output = get_csv(bulletin_dir)
output.head()
dictionary =  pd.read_csv('../input/cityofla/CityofLA/Additional data/kaggle_data_dictionary.csv')



columns = dictionary.columns

new_data = pd.DataFrame([['DEADLINE_DATE', 'U', 'Deadline of for the job bulletin', 'String', '', 'Yes', ''],

                      ['REQUIREMENT_TEXT', 'V', 'The requirements of a particular job', 'String', '', 'Yes', '']], 

                     columns=columns)



dictionary = dictionary.append(new_data)

dictionary.tail(2)
dictionary.to_csv('dictionary.csv', index=False)

output.to_csv('bulletins.csv', index=False)
# section packages

import matplotlib.pyplot as plt

import seaborn as sns

data = output

unique = data.drop_duplicates(['JOB_CLASS_NO'])
dates = pd.DataFrame([[date.year, 1] for date in pd.to_datetime(unique['OPEN_DATE'])])

dates = dates.groupby([0]).aggregate(sum)

sns.set(style="darkgrid")

# Plot the responses for different events and regions

plt.plot(dates[1].keys(), dates[1].values)
pos = [val for val in unique['JOB_CLASS_TITLE'] if type(val) == type('')]

d = pd.DataFrame([[val.split()[-1], 1] for val in pos if val])

d = d.groupby([0]).aggregate(sum)

counts = d.nlargest(15, 1)[1]

labels = counts.keys()



# plt.figure(figsize=(10, 8))

# plt.title('Type of Job x Bulletin Count')

plot = sns.barplot(x=counts, y=labels)
def clean_salary(text):

    sal = re.split(' |-', text)

    start = round(int(sal[0][1:].replace(',', '')), -3) // 1000

    end = round(int(sal[1][1:].replace(',','')), -3) // 1000 if sal[1][0] == '$' else start

    avg = (start + end) // 2

    diff = end - start

    return start, end, avg, diff



def get_salaries(data):

    start_sal = []

    end_sal = []

    avgs = []

    for sal in data:

        if not sal:

            continue        

        start, end, avg, _ = clean_salary(sal)

        start_sal.append(start)

        end_sal.append(end)

        avgs.append(avg)

    return pd.DataFrame(data={'start': start_sal, 'end': end_sal, 'avg': avgs })



gen = get_salaries(unique['ENTRY_SALARY_GEN'])

dwp = get_salaries(unique['ENTRY_SALARY_DWP'])

gen= gen.sort_values(by='start')

dwp = dwp.sort_values(by='start')
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(14,8))



sns.distplot(gen['start'], kde=False, ax=ax1)

sns.distplot(gen['end'], kde=False, ax=ax1)

ax1.set_title('Salary Ranges')

ax1.set(xlabel='Salary (in thousands)', ylabel='Frequency')

ax1.legend(labels=['Lower Bound', 'Upper Bound'])



sns.distplot(dwp['start'],kde=False, ax=ax2)

sns.distplot(dwp['end'], kde=False, ax=ax2)

ax2.set(xlabel='Salary (in thousands)', ylabel='Frequency')

ax2.set_title('Salary Distribution DWP')

ax2.legend(labels=['Lower Bound', 'Upper Bound'])



sns.distplot(gen['avg'], kde=False, ax=ax3, color='g')

ax3.set(xlabel='Salary (in thousands)', ylabel='Frequency')

ax3.set_title('Averaged Salary Distribution')



sns.distplot(dwp['avg'], kde=False, ax=ax4, color='g')

ax4.set(xlabel='Salary (in thousands)', ylabel='Frequency')

ax4.set_title('Averaged Salary Distribution DWP')



print('Average across salaries:', round(np.mean(gen['avg'])))

print('Average across salaries (DWP):', round(np.mean(dwp['avg'])))

res = []

for row in unique.iterrows():

    title = row[1][1]

    sal = row[1][19]

    if sal and title:

        start, end, avg, diff = clean_salary(sal)

        res.append([title, start, end, avg, diff])

res = pd.DataFrame(res)

largests = res.nlargest(10, 3)



plt.title('Top averaged salaries')

plt.xlabel('Salaries in thousands')

ax = sns.barplot(y=0, x=3, data=largests)

ax.set(xlabel='Averaged Salary (in thousands)', ylabel='Titles')

plt.show()
plt.title('Highest deviations')

plt.xlabel('Salaries in thousands')

ax = sns.barplot(y=0, x=4, data=res.nlargest(10, 4))

ax.set(xlabel='Differential in thousands', ylabel='Titles')

plt.show()
from wand.image import Image as Img

Img(filename='../input/cityofla/CityofLA/Additional data/PDFs/2018/September/Sept 28/ARTS MANAGER 2455 092818.pdf', resolution=300)
# Notice: kaggle package library doesn't have the readability package

# The rest of the post proceeds with a screenshot of the output of the code

import readability

import os

def get_metrics(directory):

    results = []

    for idx, filename in enumerate(os.listdir(directory)):

        with open(directory + "/" + filename, 'r', errors='ignore') as f:

            lines = f.readlines()

            res = readability.getmeasures(lines, lang='en')

            res = res['readability grades']

            results.append(res)

    return pd.DataFrame(results)

results = get_metrics("./Job Bulletins/")



flesch = results['FleschReadingEase']

gunning = results['GunningFogIndex']



import matplotlib.pyplot as plt

import seaborn as sns

flesch_m = np.mean(flesch)

gunning_m = np.mean(gunning)



fig, (ax1, ax2) =plt.subplots(1,2, figsize=(20, 7))

ax1.title.set_text('Average Score')

sns.barplot(x=['Flesch Reading Ease', 'Gunning Fog Index'], y=[flesch_m, gunning_m], ax=ax1)



ax2.title.set_text('Distribution')





sns.distplot(gunning, color='orange', kde=False, rug=False, ax=ax2)

sns.distplot(flesch, kde=False, rug=False, ax=ax2)

plt.legend(labels=['Gunning Fog Index', 'Flesch Reading Ease'])

plt.show()



print('Average Flesch:', flesch_m.round(2),'\nAverage Gunning:', gunning_m.round(2))

print('Min/Max Flesch:', np.min(flesch).round(2), '/',np.max(flesch).round(2))

print('Min/Max Gunning:', np.min(gunning).round(2), '/', np.max(gunning).round(2))
import re

import os

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import numpy as np

def get_word_length(directory):

    results = []

    for idx, filename in enumerate(os.listdir(directory)):

        with open(directory + "/" + filename, 'r', errors='ignore') as f:

            text = ''.join(f.readlines())

            words = re.findall('\w+', text)

            results.append((len(words), len(text)))



    return pd.DataFrame(results)



lengths = get_word_length("../input/cityofla/CityofLA/Job Bulletins/")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))



sns.distplot(lengths[0], color='blue', ax=ax1, kde=False)

ax1.title.set_text('No. Words in job postings')

sns.distplot(lengths[1], color='orange', kde=False, rug=False, ax=ax2)

ax2.title.set_text('No. Characters in job postings')

plt.show()

print('Avg. number of words of the job postings: ', int(np.mean(lengths[0])))

print('Avg. number of characters: ', int(np.mean(lengths[1])))