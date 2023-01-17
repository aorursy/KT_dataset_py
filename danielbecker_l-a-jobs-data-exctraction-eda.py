import os

import re

import math

import random

import warnings



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import spacy

nlp = spacy.load('en')

nlp.remove_pipe('parser')

nlp.remove_pipe('ner')



from wordcloud import WordCloud

from collections import Counter
warnings.filterwarnings('ignore')

pd.set_option('display.max_colwidth', -1)



SEED = 13

random.seed(SEED)

np.random.seed(SEED)
file_job_titles = '../input/cityofla/CityofLA/Additional data/job_titles.csv'

job_titles = pd.read_csv(file_job_titles, header=None, names=['job_title'])

job_titles = job_titles['job_title'].tolist()
file_data_dic = '../input/cityofla/CityofLA/Additional data/kaggle_data_dictionary.csv'

data_dictionary = pd.read_csv(file_data_dic)
dir_job_bulletins = '../input/cityofla/CityofLA/Job Bulletins'

data_list = []

for filename in os.listdir(dir_job_bulletins):

    with open(os.path.join(dir_job_bulletins, filename), 'r', errors='ignore') as f:

        data_list.append([filename, ''.join(f.readlines())])

jobs = pd.DataFrame(data_list, columns=['file', 'job_description'])
jobs = jobs[jobs['file'] != 'Vocational Worker  DEPARTMENT OF PUBLIC WORKS.txt']
def merge_jobs_data(jobs, extracted_data):

    """ Add the extracted_data to the current jobs DataFrame



        param jobs: Current jobs DataFrame

        param extracted_data: Series with DataFrame inside to extract

        return jobs: Merged DataFrame

    """ 

    jobs['temp'] = extracted_data

    for index, row in jobs.iterrows():

        extracted_data = row['temp']

        if isinstance(extracted_data, pd.DataFrame):

            for c in extracted_data.columns:

                jobs.loc[index, c] = extracted_data[c][0]

    jobs = jobs.drop('temp', axis=1) 

    return jobs



def extract_text_by_regex(text, regex_dictionary, flags=re.DOTALL):

    """ Extract values by regular expressions



        param text: String to extract the values

        param regex_dictionary: Dictionary with the names and regular expressions to extract

        return result: Series with the first extracted values

    """ 

    regex_dictionary = pd.DataFrame(regex_dictionary, columns=['name', 'regexpr'])

    result = regex_dictionary.copy()

    result['text'] = np.NaN

    for index,row in regex_dictionary.iterrows():

        find_reg = re.findall(row['regexpr'], text, flags)

        extracted_text = find_reg[0].strip() if find_reg else np.NaN

        result.loc[index, 'text'] = extracted_text

    return result.set_index('name')[['text']].T 



def extract_text_by_regex_index(text, regex_dictionary):

    """ Extract values by regular expressions

    

        Search for the index of the first occurrence of the regular expression 

        and extract the text to the next regular expression.



        param text: String to extract the values

        param regex_dictionary: Dictionary with the names and regular expressions to extract

        return result: Series with the first extracted values

    """ 

    regex_dictionary = pd.DataFrame(regex_dictionary, columns=['name', 'regexpr'])



    result = regex_dictionary.copy()

    result['text'] = np.NaN

    for index,row in regex_dictionary.iterrows():

        find_text = re.search(row['regexpr'], text)

        find_text = find_text.span(0)[0] if find_text else np.nan

        result.loc[index, 'start'] = find_text

    result.dropna(subset=['start'], inplace=True)

    result['end'] = result['start'].apply(lambda x: np.min(result[result['start'] > x]['start'])).fillna(len(text))

    

    for index,row in result.iterrows():

        extracted_text = text[int(row['start']):int(row['end'])]

        find_reg = re.findall(row['regexpr']+'(.*)', extracted_text, re.DOTALL)

        extracted_text = find_reg[0].strip() if find_reg else np.NaN

        result.loc[index, 'text'] = extracted_text

    return result.set_index('name')[['text']].T 



def nlp_transformation(data, token_pos=None):

    """ Use NLP to transform the text corpus to cleaned sentences and word tokens



        param data: List with sentences, which should be processed.

        param token_pos: List with the POS-Tags to filter (Default: None = All POS-Tags)

        return processed_tokens: List with the cleaned and tokenized sentences

    """    

    def token_filter(token):

        """ Keep tokens who are alphapetic, in the pos (part-of-speech) list and not in stop list

            

        """    

        if token_pos:

            return not token.is_stop and token.is_alpha and token.pos_ in token_pos

        else:

            return not token.is_stop and token.is_alpha

    

    data = [re.compile(r'<[^>]+>').sub('', x) for x in data] #Remove HTML-tags

    processed_tokens = []

    data_pipe = nlp.pipe(data)

    for doc in data_pipe:

        filtered_tokens = [token.lemma_.lower() for token in doc if token_filter(token)]

        processed_tokens.append(filtered_tokens)

    return processed_tokens
regex_dictionary = [('metadata', r''), 

                      ('salary', r'(?:ANNUAL SALARY|ANNUALSALARY)'),

                      ('duties', r'(?:DUTIES)'),

                      ('requirements', r'(?:REQUIREMENTS/MINIMUM QUALIFICATIONS|REQUIREMENT/MINIMUM QUALIFICATION|REQUIREMENT|REQUIREMENTS|REQUIREMENT/MIMINUMUM QUALIFICATION)'),

                      ('where_to_apply', r'(?:WHERE TO APPLY|HOW TO APPLY)'),

                      ('application_deadline', r'(?:APPLICATION DEADLINE|APPLICATION PROCESS)'),

                      ('selection_process', r'(?:SELECTION PROCESS|SELELCTION PROCESS)'),

                      ]

extracted_data = jobs['job_description'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))

jobs = merge_jobs_data(jobs, extracted_data)
regex_dictionary = [('job_title', r'(.*?)(?=\n)'), 

                      ('class_code', r'(?:Class Code:|Class  Code:)\s*(\d\d\d\d)'),

                      ('open_date', r'(?:Open Date:|Open date:)\s*(\d\d-\d\d-\d\d)'),

                      ('revised', r'(?:Revised:|Revised|REVISED:)\s*(\d\d-\d\d-\d\d)')

                      ]

extracted_data = jobs['metadata'].dropna().apply(lambda x: extract_text_by_regex(x, regex_dictionary))

jobs = merge_jobs_data(jobs, extracted_data)

jobs['open_date'] = pd.to_datetime(jobs['open_date'], infer_datetime_format=True)

jobs['revised'] = pd.to_datetime(jobs['revised'], infer_datetime_format=True)
# Extract the first salary

regex_dictionary = [('salary_first', r'(\$(?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})* to \$(?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})*|\$(?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})* \(flat-rated\))'), 

                      ('salary_additional', r'(?:\n)(.*)(?:NOTES)'),

                      ('salary_notes', r'(?:NOTES:)(.*)'),

                      ]

extracted_data = jobs['salary'].dropna().apply(lambda x: extract_text_by_regex(x, regex_dictionary, re.DOTALL|re.IGNORECASE))

jobs = merge_jobs_data(jobs, extracted_data)

# Extract from the first salary the values

regex_dictionary = [('salary_from', r'\$((?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})*).*'), 

                      ('salary_to', r'(?:to \$)((?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})*).*'),

                      ('salary_flatrated', r'(flat-rated)')

                      ]

extracted_data = jobs['salary_first'].dropna().apply(lambda x: extract_text_by_regex(x, regex_dictionary, re.DOTALL|re.IGNORECASE))

jobs = merge_jobs_data(jobs, extracted_data)

jobs['salary_from'] = jobs['salary_from'].dropna().apply(lambda x: float(x.replace(',', '')))

jobs['salary_to'] = jobs['salary_to'].dropna().apply(lambda x: float(x.replace(',', '')))

jobs['salary_flatrated'] = jobs['salary_flatrated'].dropna().apply(lambda x: True)

jobs.drop('salary_first', axis=1, inplace=True)
regex_dictionary = [('duties_text', r''), 

                      ('duties_notes', r'(?:NOTE:|NOTES:)'),

                      ]

extracted_data = jobs['duties'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))

jobs = merge_jobs_data(jobs, extracted_data)
regex_dictionary = [('requirements_text', r''), 

                         ('requirements_notes', r'(?:PROCESS NOTES|NOTES:|NOTE:|PROCESS NOTE)'),

                         ('requirements_certifications', r'(?:SELECTIVE CERTIFICATION|SELECTIVE CERTIFICATION:)'),

                      ]

extracted_data = jobs['requirements'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))

jobs = merge_jobs_data(jobs, extracted_data)
regex_dictionary = [('where_to_apply_text', r''), 

                         ('where_to_apply_notes', r'(?:NOTE:)'),

                      ]

extracted_data = jobs['where_to_apply'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))

jobs = merge_jobs_data(jobs, extracted_data)
regex_dictionary = [('application_deadline_text', r''), 

                         ('application_deadline_notes', r'(?:NOTE:)'),

                         ('application_deadline_review', r'(?:QUALIFICATIONS REVIEW|EXPERT REVIEW COMMITTEE)'),

                      ]

extracted_data = jobs['application_deadline'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))

jobs = merge_jobs_data(jobs, extracted_data)
regex_dictionary = [('selection_process_text', r''), 

                         ('selection_process_notes', r'(?:NOTES:)'),

                         ('selection_process_notice', r'(?:NOTICE:|Notice:)'),

                      ]

extracted_data = jobs['selection_process'].dropna().apply(lambda x: extract_text_by_regex_index(x, regex_dictionary))

jobs = merge_jobs_data(jobs, extracted_data)
def get_exam_type(text):

    """ Extract the exam type from the text



        param text: String to extract the values

        return result: String with the exam_type code

    """ 

    regex_dic = {'OPEN_INT_PROM':r'BOTH.*INTERDEPARTMENTAL.*PROMOTIONAL', 

                 'INT_DEPT_PROM':r'INTERDEPARTMENTAL.*PROMOTIONAL', 

                 'DEPT_PROM':r'DEPARTMENTAL.*PROMOTIONAL',

                 'OPEN':r'OPEN.*COMPETITIVE.*BASIS'

                }

    result = np.nan

    for key, value in regex_dic.items():

        regex = value

        regex_find = re.findall(regex, text, re.DOTALL)

        if regex_find:

            result = key

            break

    return result



jobs['exam_type'] = jobs['selection_process'].dropna().apply(get_exam_type)
def get_driver_license(text):

    """ Extract the driver license from the text



        param text: String to extract the values

        return result: String if a driver license is needed

    """ 

    regex_dic = {'P':r'(may[^\.]*requir[^\.]*driver[^\.]*license)', 

                 'R':r'(requir[^\.]*driver[^\.]*license)|(driver[^\.]*license[^\.]*requir)', 

                }

    result = np.nan

    for key, value in regex_dic.items():

        regex = value

        regex_find = re.findall(regex, text, re.IGNORECASE)

        if regex_find:

            result = key

            break

    return result



jobs['driver_license'] = jobs['job_description'].dropna().apply(get_driver_license)
def extrac_salary(text):

    reg_expr = r'(\$(?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})* to \$(?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})*|\$(?:\d{1,3})(?:\,\d{3})*(?:\.\d{2})* \(flat-rated\))'

    result = re.findall(reg_expr, text, re.DOTALL|re.IGNORECASE)

    return pd.Series(result)



salary = jobs['salary'].dropna().apply(extrac_salary)

salary.columns = ['salary_{}'.format(x) for x in salary.columns]

for col in salary.columns:

    jobs[col] = salary[col]
def extract_requirements(text):

    """ Extract the main and seconday requirements from the text



        param text: String to extract the values

        return result: DataFrame with the main and seconday requirements

    """ 

    reg_expr_set = r'^\d(?=\.)'

    reg_expr_subset = r'^[a-z](?=\.)'

    find = re.finditer(reg_expr_set, text, re.MULTILINE|re.IGNORECASE)

    result = [(x.group(0), x.span(0)[0]) for x in find]

    result = pd.DataFrame(result, columns=['set', 'start'])

    result['end'] = result['start'].apply(lambda x: np.min(result[result['start'] > x]['start'])).fillna(len(text))

    for index,row in result.iterrows():

        extracted_text = text[int(row['start']):int(row['end'])].strip()

        result.loc[index, 'set_text'] = extracted_text



        find_subset = re.finditer(reg_expr_subset, extracted_text, re.MULTILINE|re.IGNORECASE)

        result_subset = [(x.group(0).lower(), x.span(0)[0]) for x in find_subset]

        result_subset = pd.DataFrame(result_subset, columns=['subset', 'start'])

        result_subset['end'] = result_subset['start'].apply(lambda x: np.min(result_subset[result_subset['start'] > x]['start'])).fillna(len(extracted_text))

        for index_sub, row_sub in result_subset.iterrows():

            extracted_sub_text = extracted_text[int(row_sub['start']):int(row_sub['end'])].strip()

            result.loc[index, row_sub['subset']] = extracted_sub_text

            result.loc[index, 'set_text'] = result.loc[index, 'set_text'].replace(extracted_sub_text, '').strip()

    result.drop(['start', 'end'], axis=1, inplace=True) 

    return result



temp = jobs['requirements_text'].dropna().apply(extract_requirements)



requirements = pd.DataFrame()

for i, value in temp.iteritems(): 

    if isinstance(value, pd.DataFrame):

        value['file'] = jobs.loc[i, 'file']

        requirements = requirements.append(value, ignore_index=True)

        

requirements = requirements.melt(id_vars=['file', 'set', 'set_text'], var_name='subset', value_name='subset_text')

requirements = requirements[(requirements['subset'] == 'a') | ((requirements['subset'] != 'a') & (~requirements['subset_text'].isnull()))]  

requirements['subset'] = requirements['subset'].apply(lambda x: x.upper())
def get_school_type(text):

    """ Extract the school type from the text



        param text: String to extract the values

        return result: String with the schol_type code

    """ 

    regex_dic = {'COLLEGE OR UNIVERSITY':'college or university', 

                 'HIGH SCHOOL':'high school', 

                 'APPRENTICESHIP':'apprenticeship'

                }

    result = np.nan

    for key, value in regex_dic.items():

        regex = value

        regex_find = re.findall(regex, text, re.DOTALL|re.IGNORECASE)

        if regex_find:

            result = key

            break

    return result



requirements['school_type'] = requirements['set_text'].dropna().apply(get_school_type)
def get_education_years(text):

    """ Extract the education years from the text



        param text: String to extract the values

        return result: String with the schol_type code

    """ 

    regex_dic = {1:'one', 2:'two', 3:'three', 4:'four', 5:'five',

                6:'six', 7:'seven', 8:'eight', 9:'nine'}

    result = np.nan

    for key, value in regex_dic.items():

        regex = value+'[/s-]year.*(college or university|high school|apprenticeship)'

        regex_find = re.findall(regex, text, re.DOTALL|re.IGNORECASE)

        if regex_find:

            result = key

            break

    return result



requirements['education_years'] = requirements['set_text'].dropna().apply(get_education_years)
def get_full_part_time(text):

    """ Extract the full time / part time from the text



        param text: String to extract the values

        return result: String with the full time / part time

    """ 

    regex_dic = {'FULL_TIME':'full-time', 

                 'PART_TIME':'part-time', 

                }

    result = np.nan

    for key, value in regex_dic.items():

        regex = value

        regex_find = re.findall(regex, text, re.DOTALL|re.IGNORECASE)

        if regex_find:

            result = key

            break

    return result



requirements['full_time_part_time'] = requirements['set_text'].dropna().apply(get_full_part_time)
def get_experience_length(text):

    """ Extract the experience length from the text



        param text: String to extract the values

        return result: String with the experience length

    """ 

    regex_dic = {1:'one', 2:'two', 3:'three', 4:'four', 5:'five',

                6:'six', 7:'seven', 8:'eight', 9:'nine'}

    result = np.nan

    for key, value in regex_dic.items():

        regex = r'(?!.*(college or university|high school|apprenticeship)).+'+value+'[\s-]year.*(full[\s-]time|part[\s-]time)'

        regex_find = re.findall(regex, text, re.DOTALL|re.IGNORECASE)

        if regex_find:

            result = key

            break

    return result



requirements['experience_length'] = requirements['set_text'].dropna().apply(get_experience_length)
def get_exp_job_class_title(text, job_titles=job_titles):

    """ Check if the text has a job from the job_titles list



        param text: String to extract the values

        param job_titles: List with possible job titles

        return result: String with the job title

    """ 

    result = np.nan

    for x in job_titles:

        if x.lower() in text.lower():

            result = x

            break

    return result



requirements['exp_job_class_title'] = requirements['set_text'].dropna().apply(get_exp_job_class_title)
drop_cols = ['job_description', 'metadata', 'salary', 'duties', 

        'requirements', 'where_to_apply', 

        'application_deadline', 'selection_process']

example = jobs[jobs['file'] == 'SYSTEMS ANALYST 1596 102717.txt'].drop(drop_cols, axis=1).iloc[0,:].dropna()

for idx in example.index:

    print('\033[42m'+idx+':'+'\033[0m')

    print(example[idx])
data_dictionary
# Initialize empty DataFrame with 'Field Name' as Columns

submission = pd.DataFrame(columns=data_dictionary['Field Name'].values)



# FILE_NAME

submission['FILE_NAME'] = jobs['file']

# JOB_CLASS_TITLE

submission['JOB_CLASS_TITLE'] = jobs['job_title']

# JOB_CLASS_NO

submission['JOB_CLASS_NO'] = jobs['class_code']

# EXAM_TYPE

submission['EXAM_TYPE'] = jobs['exam_type']

# OPEN_DATE

submission['OPEN_DATE'] = jobs['open_date']

# ENTRY_SALARY_GEN

submission['ENTRY_SALARY_GEN'] = jobs['salary_0']

# ENTRY_SALARY_DWP

submission['ENTRY_SALARY_DWP'] = jobs['salary_1']

# DRIVERS_LICENSE_REQ

submission['DRIVERS_LICENSE_REQ'] = jobs['driver_license']

# JOB_DUTIES

submission['JOB_DUTIES'] = jobs['duties_text']



# Merge with Requirements

submission = pd.merge(submission, requirements, left_on='FILE_NAME', right_on='file', how='left')

submission['REQUIREMENT_SET_ID'] = submission['set']

submission['REQUIREMENT_SUBSET_ID'] = submission['subset']

submission['EDUCATION_YEARS'] = submission['education_years']

submission['SCHOOL_TYPE'] = submission['school_type']

submission['EXP_JOB_CLASS_FUNCTION'] = submission['subset_text']

submission['FULL_TIME_PART_TIME'] = submission['full_time_part_time']

submission['EXPERIENCE_LENGTH'] = submission['experience_length']

submission['EXP_JOB_CLASS_TITLE'] = submission['exp_job_class_title']

submission.drop(requirements.columns, axis=1, inplace=True)
submission[submission['FILE_NAME'] == 'SYSTEMS ANALYST 1596 102717.txt'].T
def validate_submission(submission, data_dictionary):

    """ Makes some validations and creates a summary of the submission file



        param submission: DataFrame ob the Submission file

        param data_dictionary: DataFrame with the data dictionary to validate the submission file

        return result: DataFrame with a summary of the validation

    """ 

    result = pd.DataFrame(index=data_dictionary['Field Name'].values)

    for col in data_dictionary['Field Name'].values:

        result.loc[col, 'Unique Values'] = len(submission[col].dropna().unique())

        result.loc[col, 'Values'] = len(submission[col].dropna())

        result.loc[col, 'Null Values'] = len(submission[submission[col].isnull()][col])

        result.loc[col, 'Accept Null Values'] = data_dictionary[data_dictionary['Field Name'] == col]['Accepts Null Values?'].values[0]

        result.loc[col, 'Data Type'] = submission[col].dtype

        result.loc[col, 'Expected Data Type'] = data_dictionary[data_dictionary['Field Name'] == col]['Data Type'].values[0]

        result.loc[col, 'Check Values'] = ('Okay' if (result.loc[col, 'Accept Null Values'] == 'Yes') or (result.loc[col, 'Null Values'] == 0) else 'No Null Values allowed')

        result.loc[col, 'Check Data Type'] = ('Okay' 

                                              if ((result.loc[col, 'Data Type'] == 'object') and (result.loc[col, 'Expected Data Type'] == 'String'))

                                              or ((result.loc[col, 'Data Type'] == 'float64') and (result.loc[col, 'Expected Data Type'] == 'Float'))

                                              or ((result.loc[col, 'Data Type'] == 'datetime64[ns]') and (result.loc[col, 'Expected Data Type'] == 'Date'))

                                              or ((result.loc[col, 'Data Type'] == 'int64') and (result.loc[col, 'Expected Data Type'] == 'Integer'))

                                              else 'Check Data Type')

    return result



validate_submission(submission, data_dictionary)
temp = jobs.fillna('Missing')

temp = temp.applymap(lambda x: x if x == 'Missing' else 'Available')

figsize_width = 12

figsize_height = len(temp.columns)*0.5

plt_data = pd.DataFrame()

for col in temp.columns:

    temp_col = temp.groupby(col).size()/len(temp.index)

    temp_col = pd.DataFrame({col:temp_col})

    plt_data = pd.concat([plt_data, temp_col], axis=1)

    

ax = plt_data.T.plot(kind='barh', stacked=True, figsize=(figsize_width, figsize_height))



# Annotations

labels = []

for i in plt_data.index:

    for j in plt_data.columns:

        label = '{:.2%}'.format(plt_data.loc[i][j])

        labels.append(label)

patches = ax.patches

for label, rect in zip(labels, patches):

    width = rect.get_width()

    if width > 0:

        x = rect.get_x()

        y = rect.get_y()

        height = rect.get_height()

        ax.text(x + width/2., y + height/2., label, ha='center', va='center')



plt.xlabel('Frequency')

plt.title('Missing values')

plt.xticks(np.arange(0, 1.05, 0.1))

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
plt_data = (jobs.groupby(jobs['open_date'].dropna().dt.strftime('%Y-%m')).size())

plt_data = pd.DataFrame(plt_data)

plt_data.plot(kind='bar', figsize=(15, 5))

plt.xlabel('Month')

plt.ylabel('Quantity')

plt.title('Distribution over time')

plt.legend('')

plt.show()
plt_data = jobs[['salary_from', 'salary_to']]

plt_data.plot(kind='box', showfliers=True, vert=False, figsize=(12, 3), grid=True)

plt.xticks(range(0, 300001, 25000))

plt.xlabel('Salary')

plt.title('Salary Distribution')

plt.show()

plt_data.describe()
plt_data = jobs[['salary_from', 'salary_to']]

plt_data.plot(kind='hist', bins=1000, density=True, histtype='step', cumulative=True, figsize=(15, 7), lw=2, grid=True)

plt.xlabel('Salary')

plt.ylabel('Cumulative')

plt.title('Cumulative histogram for salary')

plt.legend(loc='upper left')

plt.xlim([25000, 200000])

plt.xticks(range(25000, 200001, 10000))

plt.yticks(np.arange(0, 1.05, 0.05))

plt.show()
plt_data_duties = jobs['duties_text'].astype(str).apply(lambda x: len(x.split()))

plt_data_requirements = jobs['requirements_text'].astype(str).apply(lambda x: len(x.split()))

plt_data = pd.DataFrame([plt_data_duties, plt_data_requirements]).T



plt_data.plot(kind='box', showfliers=False, vert=False, figsize=(12, 3), grid=True)

plt.xticks(range(0, 201, 10))

plt.xlabel('Words')

plt.title('Word count')

plt.show()
plt_data = jobs['exam_type'].fillna('None')

plt_data = plt_data.groupby(plt_data).size()

plt_data.plot(kind='pie', figsize=(10, 5), autopct='%.2f')

plt.title('Exam Type')

plt.ylabel('')

plt.show()
plt_data = jobs['driver_license'].fillna('None')

plt_data = plt_data.groupby(plt_data).size()

plt_data.plot(kind='pie', figsize=(10, 5), autopct='%.2f')

plt.title('Driver License')

plt.ylabel('')

plt.show()
pos_tags = ['NOUN', 'VERB', 'PROPN', 'ADJ']

plot_cols = 2

plot_rows = math.ceil(len(pos_tags) / plot_cols)

axisNum = 0

plt.figure(figsize=(7*plot_cols, 4*plot_rows))

for pos_tag in pos_tags:

    plt_data = nlp_transformation(jobs['duties_text'].dropna(), [pos_tag])

    plt_data = [j for i in plt_data for j in i]

    plt_data=Counter(plt_data)

    wordcloud = WordCloud(margin=0, max_words= 40, random_state=SEED).generate_from_frequencies(plt_data)

    axisNum += 1

    ax = plt.subplot(plot_rows, plot_cols, axisNum)

    plt.imshow(wordcloud, interpolation='bilinear')

    title = 'Duties ({})'.format(pos_tag)

    plt.title(title)

    plt.axis("off")

plt.show()    
pos_tags = ['NOUN', 'VERB', 'PROPN', 'ADJ']

plot_cols = 2

plot_rows = math.ceil(len(pos_tags) / plot_cols)

axisNum = 0

plt.figure(figsize=(7*plot_cols, 4*plot_rows))

for pos_tag in pos_tags:

    plt_data = nlp_transformation(jobs['requirements_text'].dropna(), [pos_tag])

    plt_data = [j for i in plt_data for j in i]

    plt_data=Counter(plt_data)

    wordcloud = WordCloud(margin=0, max_words= 40, random_state=SEED).generate_from_frequencies(plt_data)

    axisNum += 1

    ax = plt.subplot(plot_rows, plot_cols, axisNum)

    plt.imshow(wordcloud, interpolation='bilinear')

    title = 'Requirements ({})'.format(pos_tag)

    plt.title(title)

    plt.axis("off")

plt.show()    
plt_data = requirements['exp_job_class_title'].dropna().tolist()

plt_data=Counter(plt_data)

plt.figure(figsize=(10, 10))



wordcloud = WordCloud(margin=0, random_state=SEED).generate_from_frequencies(plt_data)

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Requirements (Job class title)')

plt.axis("off")

plt.show() 