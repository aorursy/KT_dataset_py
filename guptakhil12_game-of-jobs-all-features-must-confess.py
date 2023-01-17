# Importing libraries

import pandas as pd

import numpy as np

import glob

import warnings

from collections import Counter

import seaborn as sns

import math

import os



import matplotlib.pyplot as plt

%matplotlib inline



os.chdir('/kaggle/input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/')



warnings.filterwarnings("ignore")
# Getting the list of Job Postings

job_bulletins = sorted(glob.glob('Job Bulletins/*.txt'))



# Master dataframe with filename added

master = pd.DataFrame(data=[job.split('/')[1] for job in job_bulletins], columns=['FILE_NAME'])



# Traversing over job postings to create a list of list of individual elements within the job posting for easy extraction

jobs_list = []

for job in job_bulletins:

    string_val = ''

    flag = 1

    for i in open(job, "r", encoding='latin1').readlines():

        if len(i.strip()) <= 1:

            continue

        else:

            j = i.strip()

            if j.isupper() == True:

                if flag == 1:

                    string_val += "***%s"%(j)

                    flag = 0

                else:

                    string_val += '\n%s'%(j)

            else:

                flag = 1

                string_val += '\n%s'%(j)

    jobs_list.append([i.strip() for i in string_val.split('***') if len(i)!=0])
# Common headings to get a sense of what the raw data content looks like!

ck = []

for j in jobs_list:

    for i in j:

        urr = []

        for k in i.replace('\n', ' ').split(' '):

            if k.isupper()==True and len(k)>1:

                urr.append(k)

            else:

                break

        ck.append(' '.join(urr))

sorted(dict(Counter(ck)).items(), key=lambda x:x[1], reverse=True)[:25]
job_info = []



for idx, job in enumerate(jobs_list):

    val = job[0]

    for old, new in {'  ': ' ', '\n': ' ', '\t': ' ', 'date': 'Date'}.items():

        val = val.replace(old, new)

    title = val.split('Class Code:')[0].strip()

    try:

        code = val[val.find('Class Code:') : val.find('Open Date')].split(':')[1].strip()

        if len(code) != 4 or code.isdigit() == False:

            print(code)

        date = val.split('Open Date:')[1].strip()

        date = date.split(' ')[0]

    except:

        code = ''

        date = np.nan

        print("No job info (class code and open date) found for ***%s***"%(master.iloc[idx]['FILE_NAME']))

    job_info.append((title, code, date))



master['JOB_CLASS_TITLE'] = [i[0] for i in job_info]

master['JOB_CLASS_NO'] = [i[1] for i in job_info]

master['OPEN_DATE'] = pd.to_datetime([i[2] for i in job_info]) # Validation of the Date



del job_info
# Rectifying title name values with the File-Name



def break_file_name(x):

    title = []

    code = []

    for i in x.split(' '):

        if len(i) == 4 and i.isdigit()==True:

            code.append(i)

            break

        title.append(i)

    return (' '.join(title).strip(), ' '.join(code).strip())



master['TITLE_CHECKER'] = master.apply(lambda x: break_file_name(x['FILE_NAME'])[0], axis=1)

master['CODE_CHECKER'] = master.apply(lambda x: break_file_name(x['FILE_NAME'])[1], axis=1)



master['TITLE_MATCH'] = np.where(master['JOB_CLASS_TITLE'] == master['TITLE_CHECKER'], 1, 0)

master['CODE_MATCH'] = np.where(master['JOB_CLASS_NO'] == master['CODE_CHECKER'], 1, 0)



# Rectifying title

for index, row in master.iterrows():

    y = row['JOB_CLASS_TITLE']

    if row['TITLE_MATCH'] == 0:

        if 'CAMPUS INTERVIEWS ONLY' in y:

            master.set_value(index, 'JOB_CLASS_TITLE', y.split('CAMPUS INTERVIEWS ONLY')[1].strip())

        elif '(' in y or ')' in y:

            st = y[y.find('('): y.find(')') + 1]

            if len(st)==0:

                st = y[y.find('('):]

            y = y.replace(st, '')

            y = y.replace('  ', ' ')

            master.set_value(index, 'JOB_CLASS_TITLE', y.strip())

        elif '\t' in y:

            y = y.split('\t')[0]

            master.set_value(index, 'JOB_CLASS_TITLE', y.strip())

        else:

            continue

            

master.drop(columns=['CODE_CHECKER', 'CODE_MATCH'], inplace=True)

master.drop(columns=['TITLE_CHECKER', 'TITLE_MATCH'], inplace=True)
revised_dates = []



for job in jobs_list:

    flag = 0

    for i in job:

        j = i.upper()

        if 'REVISED' in j:

            g = []

            g = [k.strip(':') for k in j[j.find('REVISED') : j.find('REVISED') + 100].split(' ')[1:] if len(k)>1][0].split('\n')[0][:8] # Keep maximum eight characters MM-DD-YY

            dates = g.split('-')

            if len(dates) == 3:

                revised_dates.append('%d-%d-%d'%(int(dates[0]), int(dates[1]), 2000 + int(dates[2])))

                flag = 1

                break

    if flag == 0:

        revised_dates.append(np.nan)



master['REVISED_DATE'] = pd.to_datetime(revised_dates)



del revised_dates
jobs_more_than_once = master.groupby(['JOB_CLASS_NO'])['FILE_NAME'].nunique().reset_index()

jobs_more_than_once = jobs_more_than_once[jobs_more_than_once['FILE_NAME']>1]



# Remove the job titles which have multiple occurences on the same open date

master[master['JOB_CLASS_NO'].isin(jobs_more_than_once['JOB_CLASS_NO'])].sort_values(['JOB_CLASS_NO'])
# Removing the record for Animal Care Techniciam Supervisor i.e. INDEX 27! ()

master.drop(master.index[27], inplace=True)

master.reset_index(drop=True, inplace=True)

del jobs_list[27]
master['TIME_TO_REVISION'] = (master['REVISED_DATE'] - master['OPEN_DATE']).astype('timedelta64[D]')

plt.figure(figsize=(10, 8))

sns.distplot(master[(master['TIME_TO_REVISION'].isnull()==False)&(master['TIME_TO_REVISION']<=800)]['TIME_TO_REVISION'])

plt.xticks(range(0, 800, 50))

plt.xlabel("Days taken for the Job Posting Revision")

plt.ylabel("Proportion of Jobs")

plt.show()
# Open Date Trends

def quarter(x):

    if math.isnan(x):

        return None

    else:

        return '%d_Q%d'%(x/100, ((x%100 - 1)/3) + 1)



master['OPEN_MONTH_YEAR'] = master.apply(lambda x: (x['OPEN_DATE'].month) + (x['OPEN_DATE'].year)*100, axis=1)

master['REVISED_MONTH_YEAR'] = master.apply(lambda x: (x['REVISED_DATE'].month) + (x['REVISED_DATE'].year)*100, axis=1)

date_analysis = master[((master['OPEN_MONTH_YEAR']>=201400)&(master['OPEN_MONTH_YEAR']<201900)) | ((master['REVISED_MONTH_YEAR']>=201400)&(master['REVISED_MONTH_YEAR']<201900))]

date_analysis['OPEN_MONTH_YEAR'] = np.where((date_analysis['OPEN_MONTH_YEAR']<201400) | (date_analysis['OPEN_MONTH_YEAR']>201900), date_analysis['REVISED_MONTH_YEAR'], 

                                            date_analysis['OPEN_MONTH_YEAR'])

date_analysis['OPEN_QUARTER'] = date_analysis.apply(lambda x: quarter(x['OPEN_MONTH_YEAR']), axis=1)

date_removed = master.iloc[list(set(master.index) - set(date_analysis.index)),:]

quarterly = date_analysis.groupby(['OPEN_QUARTER'])['FILE_NAME'].nunique().reset_index()

quarterly.sort_values(['OPEN_QUARTER'], ascending=True, inplace=True)

plt.figure(figsize=(15, 8))

plt.plot(quarterly['FILE_NAME'], marker='s')

plt.xlabel("Quarters")

plt.ylabel("# Jobs")

display_quarters = [i[2:] for i in list(quarterly['OPEN_QUARTER'])]

for idx, i in enumerate(display_quarters):

    if idx%4 == 0:

        plt.axvline(x=idx, color='orange', linestyle='--')

plt.xticks(range(0, len(quarterly)), display_quarters)

plt.show()



master.drop(columns=['OPEN_MONTH_YEAR', 'REVISED_MONTH_YEAR'], inplace=True)
# These 7 records are being dropped because their open date is either before 2014, or in 2019.

date_removed
# Removing these dates from the master analysis file as well (Too old or too new)

date_removed = list(date_removed.index.values)

master.drop(index=date_removed, inplace=True)

master.reset_index(drop=True, inplace=True)



jobs_list2 = []

for idx, i in enumerate(jobs_list):

    if idx not in date_removed:

        jobs_list2.append(i)

jobs_list = jobs_list2

del jobs_list2
duties = []



for job in jobs_list:

    duty = ''

    for i in job:

        for old, new in {'\n': ' ', '\t': ' '}.items():

            i = i.replace(old, new)

        for term in ['DUTIES AND RESPONSIBILITIES', 'DUTIES']: # NOT AVAILABLE FOR FIRE-RELATED JOBS

            if term in i:

                duty = i.split(term)[1].strip()

                break

    duties.append(duty)

    

master['JOB_DUTIES'] = duties

master['JOB_DUTIES'] = np.where(master['JOB_DUTIES'] == '', np.nan, master['JOB_DUTIES'])



del duties
# Wordcloud for most frequent words (TF-IDF scores could also be used for a better deep-dive into this feature)



from wordcloud import WordCloud

import string



bow = []

for i in list(master['JOB_DUTIES'].unique()):

    if type(i) == str:

        words = i.lower().split(' ')

        for w in words:

            for s in list(string.punctuation) + ["'s"]:

                if s in w:

                    w = w.strip(s)

            bow.append(w)

    else:

        pass

bow = list(sorted(dict(Counter(bow)).items(), key=lambda x:x[1], reverse=True))



st = pd.read_csv('/kaggle/input/stopwords/stopwords.csv')

bow = dict([i for i in bow if i[0] in list(set([i[0] for i in bow]) - set(st['stopwords'].unique()))])



wordcloud = WordCloud(min_font_size=10,max_font_size=60,background_color="white",width=600,height=300).generate_from_frequencies(bow)

plt.figure(figsize=(15, 9))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
driver_license_req = []



for job in jobs_list:

    text = "driver's license"

    y = ' '.join(job).lower().strip()

    y = y.replace("drivers'", "driver's")

    vicinity = y[y.find(text) -100: y.find(text) + 100]

    if text in vicinity:

        if 'may require a valid' in y or 'some positions require a valid' in y or 'license may be required' in y:

            driver_license_req.append('P')

        elif 'license is required' in y or 'possession of a valid' in y or 'is required' in y or 'who apply with a valid':

            driver_license_req.append('R')

        else:

            raise ValueError("Please check for the specified license keywords.")

    else:

        driver_license_req.append('')

        

master['DRIVERS_LICENSE_REQ'] = driver_license_req

master['DRIVERS_LICENSE_REQ'] = np.where(master['DRIVERS_LICENSE_REQ'] == '', np.nan, master['DRIVERS_LICENSE_REQ'])



del driver_license_req
master['DRIVERS_LICENSE_REQ'].value_counts()
def parse_exam_type(value): # Identifying the exam type from the text present in the job body

    if 'INTERDEPARTMENTAL' and 'OPEN' in value:

        return 'OPEN_INT_PROM'

    elif 'INTER' in value:

        return 'INT_DEPT_PROM'

    elif 'DEPARTMENT' in value:

        return 'DEPT_PROM'

    elif 'OPEN' in value:

        return 'OPEN'

    else:

        return ''

    

exam_type = []



for idx, job in enumerate(jobs_list):

    flag = 0

    for i in job:

        for old, new in {'\n': ' ', '\t': ' ', 'EXAM ': 'EXAMINATION '}.items():

            i = i.replace(old, new)

        if 'THIS EXAMINATION' in i:

            g = []

            for j in i[i.find('THIS EXAMINATION'): i.find('THIS EXAMINATION') + 150].split(' '):

                if j.isupper() == True:

                    g.append(j)

                else:

                    break

            exam_type.append(' '.join(g))

            flag = 1

            break

    if flag == 0:

        print('Exam Type not found for ***%s***'%(master.iloc[idx]['FILE_NAME']))

        exam_type.append('')

        

master['EXAM_TYPE'] = exam_type

master['EXAM_TYPE'] = master.apply(lambda x: parse_exam_type(x['EXAM_TYPE']), axis=1) ### 646 is screwed up! 



del exam_type
# Bar Chart for the Exam Type

plt.figure()

plt.bar(master["EXAM_TYPE"].value_counts().index, master['EXAM_TYPE'].value_counts().values, width=0.5)

plt.xlabel("Exam Type")

plt.ylabel("# of Jobs")

plt.show()
def salary_decompose(sal): # Identifying the first-mentioned salary components

    if '$' not in sal:

        return ''

    else:

        gen_sal = sal.split(' ')

        x = []

        initial_sal = 0

        initial_idx = -1

        for idx, s in enumerate(gen_sal):

            if '$' in s and initial_sal == 0:

                x.append(s)

                initial_idx = idx

                initial_sal = 1

            if len(x) > 0 and '$' in s and 'to' in gen_sal[idx-1] and idx - 2 == initial_idx:

                x.append('to')

                x.append(s)

                break

        if 'to' not in ' '.join(x):

            y = x[0].split('(')[0].strip(';').strip(',').strip('.')

        else:

            y = ' '.join(x).strip(';').strip(',').strip('.')

        return y



salary = []

salary_wp = []



for job in jobs_list:

    flag_salary = 0

    flag_salary_wp = 0

    for i in job:

        for old, new in {'ANNUALSALARY': 'ANNUAL SALARY', '\n': ' ', '$ ': '$'}.items():

            i = i.replace(old, new)

        if 'ANNUAL SALARY' in i and '$' in i:

            x = i.split('Water')

            salary.append(salary_decompose(x[0]))

            flag_salary = 1

            if len(x) > 1:

                salary_wp.append(salary_decompose(x[1]))

                flag_salary_wp = 1

    if flag_salary == 0:

        salary.append('')

    if flag_salary_wp == 0:

        salary_wp.append('')



master['ENTRY_SALARY_GEN'] = salary

master['ENTRY_SALARY_GEN'] = np.where(master['ENTRY_SALARY_GEN'] == '', np.nan, master['ENTRY_SALARY_GEN'])

master['ENTRY_SALARY_DWP'] = salary_wp

master['ENTRY_SALARY_DWP'] = np.where(master['ENTRY_SALARY_DWP'] == '', np.nan, master['ENTRY_SALARY_DWP'])



del salary, salary_wp
# Extracting the Average Salary for each job post



def salarized(x):

    if type(x) == float:

        return np.nan

    else:

        if 'to' in x.lower():

            a1 = int(x.split(' ')[0].strip('$').strip('*').replace(',', '').split('.')[0])

            a2 = int(x.split(' ')[-1].strip('$').strip('*').replace(',', '').split('.')[0])

            return (a1+a2)/2

        else:

            return int(x.strip('$').strip('*').replace(',', '').split('.')[0])



master['SALARY_GEN_AVG'] = master.apply(lambda x: salarized(x['ENTRY_SALARY_GEN']), axis=1)

master['SALARY_DWP_AVG'] = master.apply(lambda x: salarized(x['ENTRY_SALARY_DWP']), axis=1)
plt.figure(figsize=(10, 8))

sns.distplot(master[master['SALARY_GEN_AVG'].isnull()==False]['SALARY_GEN_AVG'], bins=50)

plt.xticks(range(0, 300001, 50000), list(range(0, 300001, 50000)))

plt.xlabel('Average Entry Salary (General)')

plt.show()
plt.figure(figsize=(10, 8))

sns.distplot(master[master['SALARY_DWP_AVG'].isnull()==False]['SALARY_DWP_AVG'], bins=50, label='DWP (All)')

sns.distplot(master[(master['SALARY_DWP_AVG'].isnull()==False)&(master['SALARY_GEN_AVG'].isnull()==False)]['SALARY_GEN_AVG'], bins=50, label='DWP (with GEN)')

plt.xticks(range(0, 300001, 50000), list(range(0, 300001, 50000)))

plt.xlabel('Average Entry Salary (DWP)')

plt.legend()

plt.show()
temp = master[(master['SALARY_DWP_AVG'].isnull()==False)&(master['SALARY_GEN_AVG'].isnull()==False)]

plt.figure(figsize=(10, 8))

plt.scatter(temp['SALARY_GEN_AVG'], temp['SALARY_DWP_AVG'], color='orange')

x = np.linspace(0, 200001)

plt.plot(x, x, 'black', linestyle='--')

plt.xticks(range(0, 200001, 50000), list(range(0, 200001, 50000)))

plt.yticks(range(0, 200001, 50000), list(range(0, 200001, 50000)))

plt.xlabel("Average Entry Salary (GEN)")

plt.ylabel("Average Entry Salary (DWP)")

plt.show()
exam_weights = []

for job in jobs_list:

    flag = 0

    for i in job:

        x = i.lower().replace('\t', ' ')

        if 'examination weight' in x:

            j = x[x.find('examination weight'):].split(':')[1].split('\n')

            curr = []

            for k in j:

                if '. . .' in k:

                    k = k.replace('.', '_')

                    k = k.replace(' ', '_')

                    k = k.split('__')

                    m = []

                    for l in k:

                        l = l.replace('_', ' ').strip()

                        if len(l)>0:

                            m.append(l)

                    curr.append(m)

            exam_weights.append(curr)

            flag = 1

            break

    if flag == 0:

        exam_weights.append([])
# Validate Exam Weights and Rectify that

exam_weights2 = []



def check_exam_weight(x):

    if (x[0] == 'advisory' and x[1] == 'essay') or (x[0] == 'advisory essay'):

        x = ['essay', 'advisory']

    if x[1] in ['qualifying(pass/fail)', 'qualifying (pass/fail)', 'pass/fail']:

        x = [x[0], 'qualifying']

    if x[0] in ['demonstration of job knowledge and evaluation of general qualifications by technical interview']:

        x = ['qualification by technical interview', x[1]]

    if x[0] in ['assessment of training & experience', 'assessment of training and experience questionnaire']:

        x = ['assessment of training and experience', x[1]]

    if x[0] in ['essay test', 'writing exercise']:

        x = ['essay', x[1]]

    if x[0] in ['interview (including city application, essay, and personnel folder)', 'interview (including city application, advisory essay, and personnel folder)',

               'technical interview', 'qualification by technical interview', 'interview (including city application, problem solving exercise, and personnel folder)',

               'training and experience questionnaire/interview']:

        x = ['interview', x[1]]

    if x[0] in ['job simulation exercises']:

        x = ['job simulation exercise', x[1]]

    if x[0] in ['multiple choice test', 'multiple-choice written test', 'multiple-choice', 'written test - multiple-choice', 'qualifying multiple-choice test']:

        x = ['multiple-choice test', x[1]]

    if x[0] in ['oral presentation and defense', 'oral presentation exercise']:

        x = ['oral presentation', x[1]]

    if x[0] in ['physical abilities test (pat)', 'physical abilities']:

        x = ['physical abilities test', x[1]]

    if x[0] in ['pre-trip safety inspection test']:

        x = ['pre-trip inspection test', x[1]]

    if x[0] in ['training & experience questionnaire', 'training and experience questionnaire evaluation', 

                'training and experience questionnaire', 'assessment of training and experience', 'evaluation of training and experience questionnaire', 

                'training and experience (t&e) questionnaire', 'forensic print specialist training and experience questionnaire']:

        x = ['training and experience evaluation', x[1]]

    if x[0] in ['knowledge, skills, and abilities written test', 'written', 'personal characteristics written test']:

        x = ['written test', x[1]]

    return x



for idx, i in enumerate(exam_weights):

    curr = []

    for j in i:

        k = j

        if len(k)==3:

            if 'advisory' in k[1] and 'interview' in k[1]:

                k = [j[0]] + j[1].split(' ') + [j[2]]

            else:

                k = [' '.join(j[:2])] + [j[2]]

        if len(k)==4:

            curr.append(check_exam_weight(k[:2]))

            curr.append(check_exam_weight(k[2:]))

        elif len(k)==2:

            curr.append(check_exam_weight(k))

        else:

            print(len(k), k)

            raise ValueError("Unknown length of array encountered.")

    exam_weights2.append(curr)

    

exam_weights3 = []

for i in exam_weights2:

    curr = []

    for j in i:

        curr.append(' | '.join(j))

    exam_weights3.append(' || '.join(curr))

    

master['SELECTION_CRITERIA'] = exam_weights3

del exam_weights, exam_weights2, exam_weights3
months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']



def check_date(date):

    flag = 0

    temp_date = date.split('_')

    if temp_date[0] in months: # Months

        flag+=1

    if temp_date[1].isdigit() == True: # Dates

        if int(temp_date[1]) >= 1 and int(temp_date[1]) <= 31:

            flag+=1

    if temp_date[2].isdigit() == True: # Years

        if int(temp_date[2]) >= 1990 and int(temp_date[2]) <= 2025:

            flag+=1

    return int(flag/3)



application_deadlines = []

prior_notice = []



for job in jobs_list:

    flag = 0

    for i in job:

        if 'APPLICATION DEADLINE' in i:

            x = i[i.find('APPLICATION DEADLINE'):].upper().split(' ')

            dates_identified = []

            dates_idx = []

            for idx, j in enumerate(x):

                if j in months:

                    check_term = x[idx] + '_' + x[idx+1].split(',')[0] + '_' + x[idx+2][:4]

                    if check_date(check_term) == 1:

                        if len(dates_idx) > 0:

                            if dates_idx[-1]<80 and idx>100:

                                continue

                            else:

                                dates_identified.append(check_term)

                                dates_idx.append(idx)

                        else:

                            if idx<100:

                                dates_identified.append(check_term)

                                dates_idx.append(idx)

            if len(dates_identified) == 3:

                    dates_identified = [dates_identified[0], dates_identified[1]]

            if len(dates_identified) > 0:

                flag = 1

                if 'may close without prior notice' in i:

                    prior_notice.append(1)

                else:

                    prior_notice.append(0)

                if len(dates_identified) == 1:

                    dates_identified = dates_identified[0].split('_')

                    application_deadlines.append('%s %s, %s'%(dates_identified[0], dates_identified[1], dates_identified[2]))

                else:

                    dates_combos = []

                    idx_marker = 0

                    while 2*len(dates_combos) != len(dates_identified):

                        st = dates_identified[idx_marker].split('_')

                        en = dates_identified[idx_marker + 1].split('_')

                        idx_marker += 2

                        dates_combos.append("%s %s, %s to %s %s, %s"%(st[0], st[1], st[2], en[0], en[1], en[2]))

                    application_deadlines.append(" | ".join(dates_combos))

    if flag == 0:

        if 'prior notice' in ' '.join(job):

            prior_notice.append(1)

        else:

            prior_notice.append(0)

        application_deadlines.append('')



master['APPLICATION_DEADLINE'] = application_deadlines

master['CLOSE_W/O_PRIOR_NOTICE'] = prior_notice



del application_deadlines, prior_notice
applicable_civil_service_rules = []



for job in jobs_list:

    curr = []

    for i in job:

        if 'Civil Service Rule' in i:

            curr_text = i[i.find('Civil Service Rule'):i.find('Civil Service Rule')+100]

            curr_text = [t.strip().strip(',').strip('.') for t in curr_text.split(' ')]

            curr_text = [t for t in curr_text if len(t.split('.'))==2 and t.split('.')[0].isdigit()==True and t.split('.')[1].isdigit()==True]

            if len(curr_text)>0:

                curr.append(', '.join(curr_text))

    if len(curr)>0:

        applicable_civil_service_rules.append(' | '.join(set(curr)))

    else:

        applicable_civil_service_rules.append('')

        

master['APPL_CIVIL_SERV_RULES'] = applicable_civil_service_rules



del applicable_civil_service_rules
master.sample(5, random_state=42)