import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import glob

import os

import re

from matplotlib import style

from datetime import datetime

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



%matplotlib inline



style.use('fivethirtyeight')
opening_date = []

annual_salary = []

job_title = []

file_list = []

class_code = []

duties  = []

driver_license = []

full_time_part_time =[]

experience_year =[]



dir_path = "../input/cityofla/CityofLA/Job Bulletins"

for file_name in os.listdir(dir_path):

    with open(dir_path+'/'+file_name, 'r',encoding = "ISO-8859-1") as f:

        contents = f.read()

        pattern = re.compile(r'open?[\s]*date?[\s]*:?[\s]*[\d-]*',re.I)

        match = pattern.findall(contents)

        file_list.append(file_name)

        #Get Opening Date

        try:

            match[0] = match[0].replace('Open Date: ','')

            match[0] = match[0].replace('Open date: ','')

            match[0] = match[0].strip()

            try:

                match[0] = datetime.strptime(match[0], '%m-%d-%y')

                opening_date.append(match[0])

            except :

                opening_date.append(match[0])

        except IndexError:

            opening_date.append(np.nan)

        #get annual salary

        annual_pattern = re.compile(r'annual?[\s]*salary?[\s]*?[$][\d,\stoand$;]*[\d]', re.I)

        annual_salary_match = annual_pattern.findall(contents)

        try:

            annual_salary_string = re.findall(r'\$[\d$,\s;toand]*', annual_salary_match[0])

            try:

                annual_salary.append(annual_salary_string[0])

            except:

                annual_salary.append(np.nan)

            #annual_salary.append(re.sub('ANNUAL SALARY\s+',' ',annual_salary_match[0]))

            #annual_salary.append(annual_salary_match[0].encode('ascii', "ignore"))

        except IndexError:

            annual_salary.append(np.nan)

        

        #Job Title

        jobtitle_pattern = re.compile(r'[A-Z\d\s-]*');

        jobtitle_match = jobtitle_pattern.findall(contents)

        try:

            job_title.append(re.sub('\s+',' ', jobtitle_match[0][:-1]))

        except IndexError:

            job_title.append(np.nan)

        #Class Code 

        classcode_pattern = re.compile(r'Class[\s]*Code?[\s]*?:?[\s]*[\d]*')

        classcode_match = classcode_pattern.findall(contents)

        try:

            class_code.append(classcode_match[0].replace('Class Code: ',''))

        except IndexError:

            class_code.append(np.nan)

        #Get Duties

        duties_pattern = re.compile(r'DUTIES?[\s\w\d;./]*', re.I)

        duties_match = duties_pattern.findall(contents)

        try:

            duties.append(duties_match[0])

        except IndexError:

            duties.append(np.nan)

        #Driver's License

        driver_license_pattern = re.compile(r'driver\'s license is required', re.I)

        driver_license_match = driver_license_pattern.findall(contents)

        try:

            driver_license.append(driver_license_match[0])

        except IndexError:

            driver_license.append('P')

        #Full time part time job

        job_type_pattern = re.compile(r'full[\s\W]*time[\s\W]*paid', re.I)

        job_type_match = job_type_pattern.findall(contents)

        try:

            full_time_part_time.append(job_type_match[0])

        except IndexError:

            full_time_part_time.append('PART-TIME')

            

        #Experience

        experience_pattern = re.compile(r'(one|two|three|four|five|six|seven|eight|nine)[\s]years[\s]*of[\s]*(full|part)-time[\s]*paid[\s]*experience', re.I)

        experience_match = experience_pattern.findall(contents)

        

        try:

            experience_year.append(experience_match[0][0].title())

        except IndexError:

            experience_year.append('0')
job_bulletins = pd.DataFrame({'file_name':file_list, 'job_title':job_title,'class_code':class_code,'opening_date':opening_date,'annual_salary':annual_salary,'job_duties':duties,'driver_license':driver_license,

                              'full_time_part_time':full_time_part_time,'experience_year':experience_year})
len(job_bulletins.index)
job_bulletins.head(5)
job_bulletins.dtypes
job_bulletins['class_code'] =pd.to_numeric(job_bulletins['class_code'],errors='coerce').fillna(0).astype(int)
job_bulletins.dtypes
job_bulletins.loc[job_bulletins['annual_salary'].isnull() == True, ]
job_bulletins.loc[job_bulletins['class_code']  == 0 , ]
job_bulletins.loc[job_bulletins['opening_date'].isnull() == True, ]
job_bulletins.loc[job_bulletins['job_duties'].isnull() == True, ]
pd.value_counts(job_bulletins['driver_license'])
job_bulletins.loc[job_bulletins['full_time_part_time'] !="PART-TIME",'full_time_part_time'] = 'FULL-TIME' 
job_bulletins.loc[job_bulletins['driver_license'] !='P','driver_license'] = 'R'
pd.value_counts(job_bulletins['driver_license'])
pd.value_counts(job_bulletins['driver_license']).plot(kind = 'bar', rot=0)

plt.show()
pd.set_option('display.max_colwidth', -1)

#print(job_bulletins['annual_salary'])



def isnan(value):

  try:

      import math

      return math.isnan(float(value))

  except:

      return False



def annual_salary_split(annual_salary):

    max_salary = []

    min_salary = []

    annual_salary_split_pattern = re.compile(r'[\dto,$\s]*')    

    for salary_text  in annual_salary:

        if isnan(salary_text) == False:

            annual_salary_split__match = annual_salary_split_pattern.findall(salary_text)

            split_max_salary = []

            split_min_salary = []

            for annual_salary_text in annual_salary_split__match:        

                if bool(re.search(r'\d', annual_salary_text)) == True:

                    annual_salary_text = re.sub(r'[,$]','', annual_salary_text)

                    salary_amount_pattern = re.compile(r'[\d]+')

                    #print(annual_salary_text)

                    salary_amount_match = salary_amount_pattern.findall(annual_salary_text)

                    print(salary_amount_match)

                    try:

                        split_min_salary.append(salary_amount_match[0])

                    except IndexError:

                        split_min_salary.append('0')

                    try:

                        split_max_salary.append(salary_amount_match[1])

                    except IndexError:

                        split_max_salary.append('0')

            try:

                min_salary.append(max(split_min_salary))

            except IndexError:

                min_salary.append('0')



            try:

                max_salary.append(max(split_max_salary))

            except IndexError:

                max_salary.append('0')

        else:

            #print(salary_text)

            min_salary.append('0')

            max_salary.append('0')

            

    return min_salary, max_salary

    #print(annual_salary[0])

    



min_salary=[]

max_salary=[]

 

min_salary, max_salary = annual_salary_split(job_bulletins['annual_salary'])



job_bulletins['min_salary'] = min_salary

job_bulletins['max_salary'] = max_salary
job_bulletins.head(5)
pd.value_counts(job_bulletins['full_time_part_time'])
pd.value_counts(job_bulletins['full_time_part_time']).plot(kind = 'bar', rot=0)

plt.show()
job_bulletins.head(5)
job_bulletins['max_salary'] =pd.to_numeric(job_bulletins['max_salary'],errors='coerce').fillna(0)

job_bulletins['min_salary'] =pd.to_numeric(job_bulletins['min_salary'],errors='coerce').fillna(0)

job_bulletins.dtypes
job_bulletins['max_salary'].mean()
job_bulletins['min_salary'].mean()
plt.hist(job_bulletins['min_salary'], bins=20)

plt.show()
job_bulletins.loc[job_bulletins['max_salary'] == 0,]
job_bulletins.loc[job_bulletins['max_salary'] == 0,].index
for i in job_bulletins.loc[job_bulletins['max_salary'] == 0,].index:

    job_bulletins.loc[i,'max_salary']  = job_bulletins.loc[i,'min_salary'] 
job_bulletins['max_salary'].max()
job_bulletins['min_salary'].max()
job_bulletins.dtypes
plt.hist(job_bulletins['max_salary'], bins=20)

plt.show()
pd.value_counts(job_bulletins['experience_year'])
pd.value_counts(job_bulletins['experience_year']).plot(kind = 'bar', rot=45)

plt.show()
job_bulletins['opening_date'] = pd.to_datetime(job_bulletins['opening_date'])
pd.value_counts(job_bulletins['opening_date'].dt.to_period('Y'))
pd.value_counts(job_bulletins['opening_date'].dt.to_period('Y')).plot(kind = 'bar', rot=45)

plt.show()
pd.value_counts(job_bulletins['opening_date'].dt.strftime('%b')).plot(kind = 'bar', rot=45)

plt.show()
pd.value_counts(job_bulletins['opening_date'].dt.strftime('%a')).plot(kind = 'bar', rot=45)

plt.show()
all_job_title = " ".join(jtitle for jtitle in job_bulletins.job_title)
print("There are {} job titile words in the combinationof all job title".format(len(all_job_title)))
#Generate a wordcloud image

wordcloud = WordCloud(background_color="white").generate(all_job_title);

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
job_bulletins.to_csv('Job_Bulletinsjob_bulletins_df.csv')