# Configuration and package installation

import pandas as pd

import numpy as np

import datetime as dt

import re

import csv

import os

import sys

import matplotlib.pyplot as plt

import seaborn as sns

!pip install textstat

import textstat

import nltk

import nltk

from nltk import word_tokenize

from nltk import pos_tag

from collections  import Counter

!pip install find_job_titles

!pip install graphviz

from find_job_titles import FinderAcora

from graphviz import Digraph

!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



# Prevent warning message, making report more user-friendly

if not sys.warnoptions:

    import warnings

    warnings.simplefilter("ignore")
# Open files directory 

files=[dir for dir in os.walk('../input/cityofla/CityofLA')]



bulletins=os.listdir("../input/cityofla/CityofLA/Job Bulletins/")

additional=os.listdir("../input/cityofla/CityofLA/Additional data/")
# Create a list of job positions: each element of list is one job description

job = []



for bullet in bulletins:

    

    f = open("../input/cityofla/CityofLA/Job Bulletins/"+ bullet,encoding='latin-1')

    position = f.read()

    pos = position.replace('\t','').replace('\n','')

    f.close()

    

    job.append(pos)
# Write regular patterns of each feature

date = '(\d\d\d\d\d\d)'

code = '(\d\d\d\d)'

salary = '\$(\d+,\d+)(\s(to|and)\s)(\$\d+,\d+)'

require1 = '(REQUIREMENTS?/\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)' 

require2 = '(REQUIREMENTS?)(.*)(NOTES?)'

educate = 'College or University|College|University|High School|college or university|college|university|high school|special degree program|degree program|undergraduate|Undergraduate|Graduate|graduate|Bachelor’s Degree|Bachelor|Master|Master’s Degree|Doctoral or Professional Degree'

experience = 'paid experience|volunteer experience|unpaid experience|full-time experience|part-time experience|full time experience|part time experience|work experience|experiences|experience'



# Check regular patterns function 

def check_re(pat, S):

    return bool(re.search(pat, S))



# Extract Title to a list

Title = []

for i in bulletins:

    Title.append(re.split("\d{4}", i)[0].lower())



Title = pd.Series(Title)



# Extract OpenDate to a list

OpenDate = []

for i in range(len(bulletins)):

    if check_re(date,bulletins[i]) == True:

        OpenDate.append(re.search(date,bulletins[i]).group())

    else:

        OpenDate.append('')

        

OpenDate = pd.Series(OpenDate)



# Extract ClassNode to a list

ClassCode = []

for i in range(len(job)):

    if check_re(code,job[i]) == True:

        ClassCode.append(re.search(code,job[i]).group())

    else:

        ClassCode.append('')

    

ClassCode = pd.Series(ClassCode)



# Extract Salary to a list

Salary = []

for i in range(len(job)):

    if check_re(salary,job[i]) == True:

        Salary.append(re.search(salary,job[i]).group())

    else:

        Salary.append('')



Salary = pd.Series(Salary)



SalaryStart = []

SalaryEnd = []

for i in range(len(Salary)):

    if Salary[i] != '':

        SalaryStart.append(re.split('(to|and)',Salary[i])[0])

        SalaryEnd.append(re.split('(to|and)',Salary[i])[2])

    else:

        SalaryStart.append('')

        SalaryEnd.append('')



SalaryStart = pd.Series(SalaryStart)

SalaryEnd = pd.Series(SalaryEnd)



# Check regular patterns function 

def check_re(pat, S):

    return bool(re.search(pat, S))



# Extract Requirement to a list

Requirement = []

for i in range(len(job)):

    if check_re(require1,job[i]) == True:

        Requirement.append(re.search(require1,job[i]).group(2))

    elif check_re(require2,job[i]) == True:

        try:

            Requirement.append(re.search('(.*)NOTES?',re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',job[i])[0][1][:1200]).group(0))

        except:

            Requirement.append('NA')

    else:

        Requirement.append('')



# Clean requirement:       

for r in range(len(Requirement)):

    if 'NOTE' in Requirement[r]:

        Requirement[r] = Requirement[r].split("NOTE",1)[0]    

        

Requirement = pd.Series(Requirement)



# Extract FileName to a list

FileName = pd.Series(bulletins)



# Extract Length to a list

Length = []

for i in range(len(job)):

    Length.append(len(job[i].split()))

    

Length = pd.Series(Length)



# Extract Education Level to a list

Education = []

for i in range(len(Requirement)):

    if check_re(educate,Requirement[i]) == True:

        Education.append(re.search(educate,Requirement[i]).group(0))

    else:

        Education.append('no education required')



for i in range(len(Education)):

    if check_re('College or University|College|University|college or university|college|university|undergraduate|Undergraduate|Bachelor’s Degree|Bachelor',Education[i]) == True:

        Education[i] = 'college or university'

    elif check_re('High School|high school',Education[i]) == True:

        Education[i] = 'high school'

    elif check_re('Graduate|graduate|Master|Master’s Degree|Doctoral or Professional Degree', Education[i]) == True:

        Education[i] = 'graduate'

    elif check_re('special degree program|degree program',Education[i]):

        Education[i] = 'special degree program'

    else:

        Education[i] = Education[i]





Education = pd.Series(Education)



# Extract Experience Level to a list

Experience = []

for i in range(len(Requirement)):

    if check_re(experience,Requirement[i]) == True:

        Experience.append('require experience')

    else:

        Experience.append('no experience required')

           

Experience = pd.Series(Experience)
# Extract Duties to a list

Duties = []

duty = '(DUTIES)(\s*)'



for bullet in bulletins:

    f = open("../input/cityofla/CityofLA/Job Bulletins/"+ bullet,encoding='latin-1')

    jobList = f.readlines()

    for i in range(len(jobList)):

        jobList[i] = jobList[i].replace('\n','')

    

    jobList = [elem for elem in jobList if elem != '']

          

    if sum(1 for i in range(len(jobList)) if check_re(duty,jobList[i]) == True) == 1:

        for i in range(len(jobList)):

            if check_re(duty,jobList[i]) == True:

                Duties.append(jobList[i+1])

    else:

        Duties.append('') # some job description doesn't contains duties



Duties = pd.Series(Duties)



# EXtract Deadlines of applications

EndDate = []

for i in job:

    try:

        EndDate.append(re.search(r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s(\d{1,2},\s\d{4})',i).group())

    except Exception as e:

        EndDate.append(np.nan)

        

EndDate = pd.Series(EndDate)
# Create dataframe for further analysis

df = pd.concat([FileName, Title, OpenDate, ClassCode, Salary,SalaryStart, SalaryEnd, 

                Requirement, Experience, Education, Duties, Length,EndDate], axis=1)

df.columns = ['FileName','Title', 'OpenDate', 'ClassCode', 'Salary','SalaryStart', 'SalaryEnd', 

              'Requirement','Experience', 'Education', 'Duties','Length','EndDate']
# Correct the column types: SalaryStart and SalaryEnd

df['SalaryStart'] = df['SalaryStart'].str.replace('$','')

df['SalaryStart'] = df['SalaryStart'].str.replace(',','')

df['SalaryStart'] = pd.to_numeric(df['SalaryStart'])



df['SalaryEnd'] = df['SalaryEnd'].str.replace('$','')

df['SalaryEnd'] = df['SalaryEnd'].str.replace(',','')

df['SalaryEnd'] = pd.to_numeric(df['SalaryEnd'])



# Fill the missing values with mean

df['SalaryStart'] = df['SalaryStart'].fillna(df['SalaryStart'].mean())

df['SalaryEnd'] = df['SalaryEnd'].fillna(df['SalaryEnd'].mean())



# Delete unnecessary column

del df['Salary']



# Set class code as index

df = df.set_index('ClassCode')
# Assign the null value a new value: '010100'

for i in range(len(df['OpenDate'])):

    if df['OpenDate'][i] == '':

        df['OpenDate'][i] = '010100'



# Transform column type into datetime

df['OpenDate'] = df['OpenDate'].apply(lambda x : dt.datetime.strptime(x, '%m%d%y'))
df.head()
# Describe the statistics of length of words

df['Length'].describe()
# Plot histogram 

fig, ax = plt.subplots(figsize=(12, 8))

ax.hist(df['Length'], bins=10, color="#d1ae45")



# Set title label legend and xtickers

plt.title('Words Count',size = 18,alpha=0.8)

plt.xlabel('Number Of Words',size = 18,alpha=0.8)

plt.ylabel('Frequency',size = 18,alpha=0.8)





# Change background color inside the axes

ax.set_facecolor("#2E2E2E")



plt.show();
# Count the number of characters in Title and number of words in Requirements and Duties

df['TitleLen'] = df['Title'].apply(lambda x: len(x))

df['RequireLen'] = df['Requirement'].apply(lambda x: len(x.split(' ')))

df['DutiesLen'] = df['Duties'].apply(lambda x: len(x.split(' ')))



fig = plt.figure(2,(12,8))

fig.subplots_adjust(hspace=0.4, wspace=0.4)



# Subplot the length of job title

fig.add_subplot(3,1,1)

sns.boxplot(x=df['TitleLen'],color = 'lightcoral')

plt.title('Job Title Length(in characters)')

plt.xlabel('number of characters',size = 12,alpha=0.8)



# Subplot the length of job requirements

fig.add_subplot(3,1,2)

sns.boxplot(x=df['RequireLen'],color = 'firebrick')

plt.title('Job Requirement Length(in words)')

plt.xlabel('number of words',size = 12,alpha=0.8)



# Subplot the length of job duties

fig.add_subplot(3,1,3)

sns.boxplot(x=df['DutiesLen'],color = 'mistyrose')

plt.title('Job Duty Length(in words)')

plt.xlabel('number of words',size = 12,alpha=0.8)





plt.tight_layout()

plt.show()
# Plot starting salary distplot

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

sns.distplot(df['SalaryStart'], rug=True, rug_kws={"color": "darkgreen"},

             kde_kws={"color": "olive", "lw": 3, "label": "KDE"},

             hist_kws={"histtype": "step", "linewidth": 3,

                       "alpha": 1, "color": "darkgreen"})



plt.title('Starting Salary Distribution')

plt.ylabel('Probability',size = 12,alpha=0.8)

plt.xlabel('Starting Salary',size = 12,alpha=0.8)



# Plot ending salary distplot

plt.subplot(1,2,2)

sns.distplot(df['SalaryEnd'], rug=True, rug_kws={"color": "firebrick"},

             kde_kws={"color": "orange", "lw": 3, "label": "KDE"},

             hist_kws={"histtype": "step", "linewidth": 3,

                       "alpha": 1, "color": "firebrick"})

plt.title('Ending Salary Distribution')

plt.ylabel('Probability',size = 12,alpha=0.8)

plt.xlabel('Ending Salary',size = 12,alpha=0.8)



plt.tight_layout()



plt.show()
# Calculating the difference of salary start and salary end 

df['SalaryDiff']=abs(df['SalaryEnd']-df['SalaryStart'])



# Filter the top 10 job position with largest difference of salary

ranges=df[['Title','SalaryDiff']].sort_values(by='SalaryDiff',ascending=False)[:10]

ranges
# Count the occurance of education requirements 

education = pd.DataFrame(df.groupby(by='Education').agg({'Title': 'count'}))

education.columns = ['count']

education = education[education['count'] > 10]

education.sort_values(by = 'count',ascending=False)





# Create donut plot to visualize the occurance of education requirements 

fig, ax = plt.subplots(figsize=(14, 8), subplot_kw=dict(aspect="equal"))



wedges, texts = ax.pie(education['count'], wedgeprops=dict(width=0.5), startangle=-40,colors=['chocolate','salmon','gold'])



bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)

kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),

          bbox=bbox_props, zorder=0, va="center")



# Demonstrate the value of each category instead of percentage as the default value

for i, p in enumerate(wedges):

    ang = (p.theta2 - p.theta1)/2. + p.theta1

    y = np.sin(np.deg2rad(ang))

    x = np.cos(np.deg2rad(ang))

    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]

    connectionstyle = "angle,angleA=0,angleB={}".format(ang)

    kw["arrowprops"].update({"connectionstyle": connectionstyle})

    ax.annotate(str(education.index[i])+' : '+str(education['count'][i]), xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),

                 horizontalalignment=horizontalalignment, **kw)



ax.set_title("The number of education requirements")



plt.show()
# Count the occurance of experience requirements 

experience = pd.DataFrame(df.groupby(by='Experience').agg({'Title': 'count'}))

experience.columns = ['count']

experience.sort_values(by = 'count',ascending=False)



# Plot the frequency of experience requirements

plt.figure(figsize=(10, 6))

sns.barplot(y=experience['count'],x=experience.index,palette="Set2")

plt.title('Experience requirement')



plt.show()
# Subplot quarter 

plt.figure(figsize=(16,8))

plt.subplot(2,2,1)

df['OpenQuarter']=[z.quarter for z in df['OpenDate']]

count=df['OpenQuarter'].value_counts(sort=False)

sns.barplot(y=count.values,x=count.index,palette='Spectral')

plt.title('Quarter')



# Subplot year

plt.subplot(2,2,2)

df['OpenYear']=[z.year for z in df['OpenDate']]

count=df['OpenYear'].value_counts(sort=False)

sns.barplot(y=count.values,x=count.index,palette='Spectral')

plt.title('Year')



# Subplot month

plt.subplot(2,2,3)

df['OpenMonth']=[z.month for z in df['OpenDate']]

count=df['OpenMonth'].value_counts(sort=False)

sns.barplot(y=count.values,x=count.index,palette='Spectral')

plt.title('Month')



# Subplot weekday

plt.subplot(2,2,4)

df['OpenDay']=[z.weekday() for z in df['OpenDate']]

count=df['OpenDay'].value_counts(sort=False)

sns.barplot(y=count.values,x=count.index,palette='Spectral')

plt.title('Weekday')





plt.show()
# Get the redability score of requirement for each job post

readability_score = []

for i in range(len(df)):

    temp = df['Requirement'][i]

    readability_score.append(textstat.text_standard(temp)[:2])

readability_score = pd.Series(readability_score).str.replace('[t]', '')
# Visualize the readability_score

readability_score = pd.to_numeric(readability_score)

readability_score.hist(bins=40, figsize = (10,8));
req=' '.join(text for text in df['Requirement'])

duties= ' '.join(d for d in df['Duties'])
def pronoun(data):

    

    '''function to tokenize data and perform pos_tagging.Returns tokens having "PRP" tag'''

    

    prn=[]

    vrb=[]

    token=word_tokenize(data)

    pos=pos_tag(token)

   

    vrb=Counter([x[0] for x in pos if x[1]=='PRP'])

    

    return vrb

    





req_prn=pronoun(req)

duties_prn=pronoun(duties)

print('pronouns used in requirement section are')

print(req_prn.keys())

print('\npronouns used in duties section are')

print(duties_prn.keys())
analyzer = SentimentIntensityAnalyzer()



# Write the function get_sentiment to get the sentiment information

def get_sentiment(column_name):

    sentiment = []

    for i in range(len(df)):

        temp = list(analyzer.polarity_scores(df[column_name][i]).values())[3]

        sentiment.append(temp)

        

    #print(sentiment)

    return sentiment
# Visualize the analysis for responsibility

plt.hist(get_sentiment('Requirement'), bins=30)

plt.title('Sentiment Distribution for Requeirment')

plt.xlabel('Sentiment')

plt.ylabel('count')

plt.show()
# Visualize the analysis for Duties

plt.hist(get_sentiment('Duties'), bins=30)

plt.title('Sentiment Distribution for Duties')

plt.xlabel('Sentiment')

plt.ylabel('count')

plt.show()
# Visualize the analysis for Title

plt.hist(get_sentiment('Title'), bins=30)

plt.title('Sentiment Distribution for Title')

plt.xlabel('Sentiment')

plt.ylabel('count')

plt.show()
dot = Digraph(comment='Promotions')



finder=FinderAcora()

strReq = "Requirement"

arrRelation=[]

arrFinal=[]



def get_promotion(row):

    strLine = str(row [strReq]) #only check Requirement/minimum qualification section to get the promotion path

    # usually the promotion sentences start with "as a" and finish with "with"

    if strLine.find("as a") > 0:

        objItem = dict() 

        # only one previous position required

        #if strLine.find("or") < 0:

        strTemp = strLine[strLine.find("as a"):strLine.find("with")-1]

        strTemp = strTemp.replace("as an", "as a").replace("as a ","")

        if strTemp != "":

            for m in finder.findall(strTemp):

                objItem['Job1'] = row["Title"].lstrip().rstrip().upper()

                objItem['Job2'] = m[2].lstrip().rstrip().upper()

                arrRelation.append(objItem)

    return row

    

df = df.apply(get_promotion, axis=1)



pro = pd.DataFrame(arrRelation)

#pro.head()

#pro.to_csv('C:/Work/Promotion.csv')

pro = pro.drop_duplicates()

#Job1 is higher role and Job1 is the role which could promote to Job1

for index, row in pro.iterrows():

# as the image is too large, here just take police related jobs as an example

    if row["Job2"].startswith("ANIMAL"):

        dot.edge(row["Job2"], row["Job1"], label='Promote')

dot
# Extract data frame into csv file

df.to_csv('./jobs.csv', index=None)