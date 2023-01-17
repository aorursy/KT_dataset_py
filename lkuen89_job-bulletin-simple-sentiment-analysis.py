# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/cityofla/CityofLA"))



def get_second_part(kw):

    sp = kw.split(':')

    if len(sp) > 1:

        return sp[1].replace(' ','')

    else:

        return 0



# Any results you write to the current directory are saved as output.



bulletins_prefix = '../input/cityofla/CityofLA/Job Bulletins/'



all_bulletins = os.listdir(bulletins_prefix)



df_array = []

# construct the data frame

next_keyword = ''

keywords = ['ANNUAL SALARY', 

            'NOTE', 

            'DUTIES',

            'PROCESS NOTES',

            'REQUIREMENT',

            'REQUIREMENTS1',

            'REQUIREMENT/MINIMUM QUALIFICATION',

            'PROCESS NOTES',

            'WHERE TO APPLY',

            'APPLICATION DEADLINE',

            'SELECTION PROCESS',

            'NOTICE']

for bulletin in all_bulletins:

    full_path = os.path.join(bulletins_prefix, bulletin)

    f = open(full_path, encoding = "ISO-8859-1")

    # FILE_NAME,

    new_item = { 'FILE_NAME': bulletin }

    for kw in keywords:

        new_item[kw] = ''

    for i, line in enumerate(f.readlines()):

        if i == 0:

            # JOB_CLASS_TITLE

            new_item['JOB_CLASS_TITLE'] = line

            continue

        if 'Class Code' in line:

            # JOB_CLASS_NO

            new_item['JOB_CLASS_NO'] = get_second_part(line)

            continue

        if 'Open Date' in line:

            new_item['OPEN_DATE'] = get_second_part(line)

            next_keyword = 'REQUIREMENTS' # the req line seems to come immediatly after to OPEN DATE Line

            continue

        # REQUIREMENT_SET_ID,REQUIREMENT_SUBSET_ID,JOB_DUTIES,EDUCATION_YEARS,SCHOOL_TYPE,EDUCATION_MAJOR,EXPERIENCE_LENGTH,FULL_TIME_PART_TIME,EXP_JOB_CLASS_TITLE,EXP_JOB_CLASS_ALT_RESP,EXP_JOB_CLASS_FUNCTION,COURSE_COUNT,COURSE_LENGTH,COURSE_SUBJECT,MISC_COURSE_DETAILS,DRIVERS_LICENSE_REQ,DRIV_LIC_TYPE,ADDTL_LIC,EXAM_TYPE,ENTRY_SALARY_GEN,ENTRY_SALARY_DWP,OPEN_DATE

        if next_keyword == 'REQUIREMENTS':

            new_item['REQUIREMENTS'] = line

            next_keyword = ''

            continue

        new_keyword = False

        for kw in keywords:

            if kw in line:

                next_keyword = kw

                new_keyword = True

                break

        if new_keyword:

            continue

        if next_keyword == '':

            continue

        new_item[next_keyword] += line

    df_array.append(new_item)

                

df = pd.DataFrame(df_array)

df.head()
# lets do some cleaning, to replace newlines and all that



def clean_line(row):

    for col in df.columns:

        row[col] = str(row[col]).replace('\n','').replace('\n52','')

    return row



df = df.apply(clean_line, axis=1)

df.head()
import textblob



blob_columns = ['DUTIES', 'NOTE', 'SELECTION PROCESS']



# whole_text = dict()

positive_jobs = dict()

negative_jobs = dict()



def get_blob_sentiment(row):

    row['polarity'] = 0

    row['subjectivity'] = 0

    for col in blob_columns:

        row[col] = row[col].replace('.','. ').replace('/',' / ') # in some occasions there is not space after .

        polarity_col = col + '_Polarity'

        subjectivity_col = col + '_Subjectivity'

        blob = textblob.TextBlob(row[col])

        # vader_sentiment = sid.polarity_scores(row[col])['compound']

        row[polarity_col] = blob.sentiment.polarity

        row[subjectivity_col] = blob.sentiment.subjectivity

        row['polarity'] = row['polarity'] + blob.sentiment.polarity

        # row['polarity'] = row['polarity'] + vader_sentiment 

        row['subjectivity'] = row['subjectivity'] + blob.sentiment.subjectivity

    if row['polarity'] > 0:

        if row['JOB_CLASS_TITLE'] in positive_jobs.keys():

            positive_jobs[row['JOB_CLASS_TITLE']].append(row['polarity'])

        else:

            positive_jobs[row['JOB_CLASS_TITLE']] = [ row['polarity'] ]

    else:

        if row['JOB_CLASS_TITLE'] in negative_jobs.keys():

            negative_jobs[row['JOB_CLASS_TITLE']].append(-row['polarity'])

        else:

            negative_jobs[row['JOB_CLASS_TITLE']] = [ -row['polarity'] ]

        # for tag in blob.tags:

        #     if tag[1] in ('JJ',

        #                   'JJR',

        #                   'JJS',

        #                   'RB',

        #                   'RBR',

        #                   'RBS',

        #                   'VB',

        #                   'VBD',

        #                   'VBG',

        #                   'VBN',

        #                   'VBP',

        #                   'VBZ'):

        #         if tag[0] in whole_text.keys():

        #             whole_text[tag[0]].append(blob.sentiment.polarity)

        #         else:

        #             whole_text[tag[0]] = [ blob.sentiment.polarity ]

    return row





df = df.apply(get_blob_sentiment, axis=1)



            

for key in positive_jobs.keys():

    positive_jobs[key] = np.mean(positive_jobs[key]) * 1000

    



for key in negative_jobs.keys():

    negative_jobs[key] = np.mean(negative_jobs[key]) * 1000



df.head()
# lets see the histograms for a starter

%matplotlib notebook

import seaborn as sns

import matplotlib.pyplot as plt

plt.axis("on")

from scipy import stats



sns.set(color_codes=True)

# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.5)

fig, ax = plt.subplots(nrows=1, ncols=2)



sns.distplot(df['polarity'], ax=ax[0])

sns.distplot(df['subjectivity'], ax=ax[1])
plt.figure()



g = sns.jointplot(df['polarity'], df['subjectivity'], kind="scatter", height=7, space=0)
def display_file(filename, from_tag=None):

    f = open(filename)

    if from_tag:

        write = False

    else:

        write = True

    print(filename)

    for line in f.readlines():

        if write == True:

            for col in df.columns:

                if col in line:

                    write = False

        if from_tag:

            if from_tag in line:

                write = True

        if write:

            print(line)



# for i, row in df[(df['subjectivity']>1.4) & (df['polarity']<-0.6)].iterrows():

#    fname = os.path.join(bulletins_prefix, row['FILE_NAME'])

#     display_file(fname, from_tag='DUTIES')

print('Most negative')

print('---------------------------------------------------------------')

print(df[df['polarity'] == df['polarity'].min()]['DUTIES'].iloc[0])

print('---------------------------------------------------------------')

print('Most positive')

print('---------------------------------------------------------------')

print(df[df['polarity'] == df['polarity'].max()]['DUTIES'].iloc[0])
from wordcloud import WordCloud



wc = WordCloud(relative_scaling=1, background_color='white',

        max_words=250,

        max_font_size=40, 

        scale=3)



wordcloud = wc.generate_from_frequencies(positive_jobs)

# fig = plt.figure(1, figsize=(12, 12))

plt.figure()

plt.axis('off')

fig.suptitle('Most negative Jobs', fontsize=20)

fig.subplots_adjust(top=2.3)

plt.imshow(wordcloud)

plt.show()

wordcloud = wc.generate_from_frequencies(negative_jobs)

# fig = plt.figure(1, figsize=(12, 12))

plt.figure()

plt.axis('off')

fig.suptitle('Most negative Jobs', fontsize=20)

fig.subplots_adjust(top=2.3)

plt.imshow(wordcloud)

plt.show()
# the annual salaray column is a bit messy, so some more cleaning is required here. 

# We will use everything that can be parsed to a number and take the median



def take_messy_median(row):

    spl = row['ANNUAL SALARY'].replace('.',' ').replace(';',' ').replace('(',' ').split(' ')

    nums = []

    for sp in spl:

        number = sp.replace('$','').replace(',','')

        try:

            number = float(number)

            nums.append(number)

        except:

            pass # we dont care right now about the exceptions

    if len(nums) == 0:

        print(row['ANNUAL SALARY']) # this line should be never printed. If it happens it must be fixed

        row['ANNUAL SALARY'] = None

    row['ANNUAL SALARY'] = np.median(nums)

    return row



df = df.apply(take_messy_median, axis=1)



# df['ANNUAL SALARY']
mean_salary = df[df['ANNUAL SALARY'].notnull()]['ANNUAL SALARY'].mean()



print('Mean Salary: ' + str(mean_salary))



df = df[df['ANNUAL SALARY'].notnull()]
# df['ANNUAL SALARY']



plt.figure()



g = sns.jointplot(df['polarity'], df['ANNUAL SALARY'], kind="scatter", height=7, space=0)
duties = df[df['ANNUAL SALARY']>250000]['DUTIES'].iloc[0]

print(duties)

polarity = df[df['ANNUAL SALARY']>250000]['polarity'].iloc[0]

print('Polarity: ' + str(polarity))