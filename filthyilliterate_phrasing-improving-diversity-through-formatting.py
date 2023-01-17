# Importing all the required libraries mentioned above and also establishing the file paths we'll be checking for files from

import re

import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from itertools import islice

import os



# Notably, if you want to automate this script, simply point it to some directory where new job postings are aggregated

# You can then perform this operation as a batch job and process the collected data from there

bulletin_dir = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins'

additional_data_dir = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/'



# countWords should be defined as a function of the data-cleaner class for readability

def countWords(string):

    word_count = 0

    string = re.sub("\d+", " ", string)

    for char in '-.,\n':

        string=string.replace(char,' ')

    wordsList = string.split()

    word_count += len(wordsList)

    return word_count



data_dictionary = pd.read_csv(additional_data_dir + 'kaggle_data_dictionary.csv', header=None)

data_dictionary.at[11, 0] = "EXPERIENCE_TYPE"

data_dictionary = data_dictionary.drop([20, 21], axis=0)

data_dictionary = data_dictionary.drop([18], axis=0)

data_dictionary.head(25)

# Loading and prepping the data of private sector jobs obtained from the Online Job Postings dataset

privateSectorJobs = pd.read_csv('../input/jobposts/data job posts.csv')

privateSectorJobs['WORD_COUNT'] = privateSectorJobs['jobpost'].apply(countWords)



# Loading a list of gender-biased words based on the following paper: 

# http://gender-decoder.katmatfield.com/static/documents/Gaucher-Friesen-Kay-JPSP-Gendered-Wording-in-Job-ads.pdf

maleBiasWords = {'active', 'adventurous', 'aggressive', 'ambitious', 'analyze'

                'assertive', 'athletic', 'autonomous', 'battle', 'boast', 'challenge',

                'champion', 'competitive', 'confident', 'courageous', 'decision', 'decisive',

                'defend', 'determined', 'dominant', 'driven', 'fearless', 'fight', 'greedy'

                'head-strong', 'headstrong', 'hierarchical', 'hierarchy', 'hostile', 'impulsive',

                'independent', 'independence', 'individual', 'intellect', 'lead', 'logic',

                'objective', 'opinion', 'outspoken', 'persist', 'principle', 'reckless', 'self-confident',

                'self', 'stubborn', 'superior', 'unreasonable'}

femaleBiasWords = {'agree', 'affectionate', 'child', 'cheerful', 'collaborate', 'commit', 'communal'

                   'compassionate', 'connect', 'considerate', 'cooperative', 'dependent', 

                   'emotional', 'empathetic', 'feel', 'flatterable', 'gentle', 'honest', 'inclusive',

                   'interpersonal', 'interdependent', 'interpersonal', 'kind', 'kinship', 'loyal'

                   'modest', 'nag', 'nurturing', 'pleasant', 'polite', 'quiet', 'responsive', 'submissive', 

                   'supportive', 'sympathetic', 'sharing', 'tender', 'together', 'trustworthy', 'understanding',

                   'warm', 'yield', }



numberWords = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen"]



# Creating a HashMap (Dictionary) for counting which gender-biased words occur most frequently

maleBiasCount = {}

for word in maleBiasWords:

    maleBiasCount[word] = 0

femaleBiasCount = {}

for word in femaleBiasWords:

    femaleBiasCount[word] = 0

numberDict = {}

# Create a HashMap for converting numbered words later on

for idx, num in enumerate(numberWords):

    numberDict[num] = (idx+1)



# Compendium of all the helper methods for the main data-mining loop

class DataCleaner:

    # Returns index where next nonempty line begins

    # Ensure to initialize to currentIndex+1 to index if trying to move from a line with text on it

    def skipWhiteLines(index, fileLines):

        index+=1

        line = fileLines[index].strip()

        while(len(line)==0):

            line = fileLines[index+1].strip()

            index+=1

        return index



    # Updates index value and skips (tempIndex-index) iterations of loop

    def skipIterations(tempIndex, index):

        n = tempIndex - index

        next(islice(it, n, n), None)

        return tempIndex



    # Detects whether salary is in range or flat-rate format and outputs cleaned string

    def salaryFormatDetect(line):

        line = line.lower()

        salariesRange = re.search("\$*\d+,\d+ *t", line)

        salariesFlat = re.search("\$*\d+,\d+;*\.*\(*", line)

        if(salariesRange):

            salaries = re.sub(" to ", "-", salariesRange.string)

            salaries = re.sub("\$ +", "$", salaries)

            salaries = re.findall("\$\d+,*\d+-\$,*\d+,*\d+", salaries)[0]

            salaries = salaries.split(';')[0]

            salaries = salaries.split('.')[0]

        elif(salariesFlat):

            salaries = salariesFlat.string.split(" ")[0]

            salaries = re.sub("\$ +", "$", salaries)

            salaries = re.split(" *\(", salaries)[0] + " (flat-rated)"

            salaries = salaries.split(';')[0]

            salaries = salaries.split('.')[0]

            if(" (flat-rated)" not in salaries):

                salaries = salaries + " (flat-rated)"

        else:

            salaries = "$62,118"

        salaries = re.sub(",", "", salaries)

        return salaries

    

    def dwpSalaryDetect(line):

        salariesDWP = re.search("Department of Water and Power is \$\d+", line) 

        if(salariesDWP):

            salaryDWP = re.split("is *", salariesDWP.string)[1]

            entry_salary_dwp = dc.salaryFormatDetect(salaryDWP)

        else:

            entry_salary_dwp = None  

        return entry_salary_dwp

    

    def openDateCleaner(line):

        job_bulletin_date = line.split("Open Date:")[1].split("(")[0].strip()

        # Clean dates that lack a starting 0 e.g. 4-13-18 -> 04-13-18

        if(len(job_bulletin_date) < 8):

            job_bulletin_date = '0' + job_bulletin_date

        # Other dates have YYYY which should be cleaned to YY e.g. 4-13-2018 -> 4-13-18

        elif(len(job_bulletin_date) > 8):

            job_bulletin_date = job_bulletin_date[:6] + job_bulletin_date[8:]

        return job_bulletin_date

    

    # Checks both line where EXAM begins and next line for exam typing

    def examTypeDetect(line1, line2):

        op = (re.search("OPEN", line1) or re.search("OPEN", line2))

        inter = (re.search("INTER", line) or re.search("INTER", line2))

        depar = (re.search(" DEPARTMENTAL", line) or re.search(" DEPARTMENTAL", line2))

        if(op and inter):

            exam = "OPEN_INTER_PROM"

        elif(op):

            exam = "OPEN"

        elif(inter):

            exam = "INT_DEPT_PROM"

        elif(depar):

            exam = "DEPT_PROM"

        else:

            exam = "Not Found"

        return exam

    

    # Takes in requirements block of job postings and returns a list containing all the possibly null values in a job posting e.g. education_major

    # All helper functions for parseRequirements are below its code block

    def parseRequirements(reqList, reqData):

        reqLines = iter(enumerate(reqList))

        reqNumber = 1

        education_type, education_length, education_major, experience_type, = None, None, None, None

        experience_length, course_count, course_length, course_subject = None, None, None, None

        experience_title, experience_alt_title, experience_job_function = None, None, None

        

        for i, line in reqLines:

            requirement_set_id = reqNumber

            requirement_subset_id = 'a'

            

            education_type = dc.educationTypeDetect(line)

            if(education_type):

                education_length = dc.educationLengthDetect(line)

                if(education_type == 'COLLEGE OR UNIVERSITY'):

                    education_major = dc.educationMajorDetect(line)

            else:

                education_length = None

                education_major = None

                

            experience_type = dc.experienceTypeDetect(line)

            if(experience_type):

                # Length tends to come before experience while actual titles come afterwards

                # RegEx also gets tricky with titles that are more than 2 words long. Can create a Set for all job titles and search the string for each

                # But that can possibly end up being more inefficient in both space/time complexity.

                length = re.split("experience", line)[0]

                if(len(re.split("experience", line)) > 1):

                    title = re.split("experience", line)[1]

                else:

                    title = length

                if(re.search("college|school|apprenticeship", length)):

                    length=re.split("college|school|apprenticeship", length)[1]

                experience_length = dc.experienceLengthDetect(length)

                experience_job_function = length

                titles = re.findall("[A-Z][a-z]+ [[A-Z][a-z]+]*", title)

                if(len(titles) > 1):

                    experience_title = titles[0]

                    experience_alt_title = titles[1]

                elif(titles):

                    experience_title = titles[0]

                else:

                    experience_title = None

            else:

                experience_length = None

                experience_title = None

                if(education_type):

                    experience_job_function = None

                else:

                    experience_job_function = line

            

            course_count = dc.courseCountDetect(line)

            if(course_count):

                course_subject = dc.educationMajorDetect(line)

            else:

                course_subject = None

            if(re.search("semester|quarter", line)):

                course_length = dc.courseLengthDetect(line)

            else:

                course_length = None

                

            # Check for requirement subset

            firstElement = line.split(" ")[0].strip()

            if(re.search("\(*[a-z]\)|\(*[A-Z]\)|[a-z]\.|[A-Z]\.", firstElement)):

                reqNumber = reqNumber - 1

                requirement_set_id = reqNumber

                firstElement = firstElement.split("\t")[0].lower()

                firstElement = re.sub('[^a-zA-Z]+', '', firstElement)

                # Concatenate first info on subset with set

                if(firstElement == 'a'):

                    reqNumber += 1

                    reqData[len(reqData)-1][7] = experience_job_function

                    reqData[len(reqData)-1][8] = course_count

                    reqData[len(reqData)-1][9] = course_length

                    reqData[len(reqData)-1][10] = course_subject

                    continue

                requirement_subset_id = firstElement

                

                #Pull the appropriate data from the previous entry when a subset

                education_type = reqData[len(reqData)-1][4]

                education_length = reqData[len(reqData)-1][3]

            

            reqNumber += 1

            reqData.append([job_class_title, requirement_set_id, requirement_subset_id, education_length, education_type, education_major,

                           experience_length, experience_type, experience_title, experience_alt_title, experience_job_function, course_count, course_length, course_subject])

            

        return reqData

    

    def educationTypeDetect(line):

        if(re.search("college|university", line)):

            education_type = "COLLEGE OR UNIVERSITY"

        elif(re.search("school", line)):

            education_type = "HIGH SCHOOL"

        elif(re.search("apprenticeship", line)):

            education_type = "APPRENTICESHIP"

        else:

            education_type = None     

        return education_type

    

    def experienceTypeDetect(line):

        if(re.search("full-time", line)):

            exp_type = "FULL_TIME"

        elif(re.search("part-time", line)):

            exp_type = "PART_TIME"

        else:

            exp_type = None     

        return exp_type

    

    def educationMajorDetect(line):

        education_major = ""

        majors = re.findall("[A-Z][a-z]+ *[a-zA-Z]*,", line)

        for m in majors:

            m = re.sub(",", "", m)

            education_major = education_major + m.upper()

            if(len(majors) > 1):

                education_major = education_major + "|"

        if(education_major == ""):

            education_major = None

        return education_major

    

    # Currently operates under the presumption that # of years/months comes before type e.g. 4 years of college vs attended college for 4 years

    # This is due to ambiguity when both education and experience length are within the same line e.g. 4 years of college and 2 years of related work 

    # (How does one determine with regular expressions which belongs to which? Closest proximity would not work, etc. What about handling other cases?)

    # Ultimately requires more advanced NLP knowledge that I do not have currently.

    def educationLengthDetect(line):

        line = re.split("college|school|apprenticeship", line)[0]

        line = dc.numberConversion(line)

        education_length = dc.lengthCalculator(line)

        # A master's degree is typically 6 years of combined schooling, hence the 6

        if(re.search("[mM]aster", line)):

            education_length = 6

        # Similarly, while most PhD's take longer than 8 years, 8 is a fairly reasonable number to place as required schooling

        if(re.search("[dD]octorate|PhD", line)):

            education_length = 8

        if(education_length == 0):

            education_length = None

        return education_length

    

    def experienceLengthDetect(line):

        line = dc.numberConversion(line)

        exp_length = dc.lengthCalculator(line)

        if(exp_length == 0):

            exp_length = None

        return exp_length    

    

    def courseCountDetect(line):

        count = None

        line = dc.numberConversion(line)

        countMatch = re.search("\d+ course", line)

        if(countMatch):

            countMatch = re.sub(" course", "", countMatch.group())

            count = int(countMatch)

        return count

    

    def courseLengthDetect(line):

        line = dc.numberConversion(line)

        totalLength = ""

        semesterMatch = re.search("\d+ semester", line)

        quarterMatch = re.search("\d+ quarter|\d+ units", line)

        if(semesterMatch):

            semesterMatch = re.sub(" semester", "", semesterMatch.group())

            totalLength = totalLength + semesterMatch + "M"

        if(quarterMatch):

            quarterMatch = re.sub(" quarter", "", quarterMatch.group())

            quarterMatch = re.sub(" units", "", quarterMatch)

            totalLength = totalLength + "|" +quarterMatch + "Q"

        if(totalLength == ""):

            totalLength = None

        return totalLength

    

     # Converts written versions of numbers to integers for easier retrieval (e.g. two -> 2)

    def numberConversion(line):

        line = re.sub("-", " ", line)

        for word in line.split():

            if(word.lower() in numberDict):

                line = re.sub(word, str(numberDict[word.lower()]), line)

        return line

    

    def lengthCalculator(line):

        totalLength = 0

        yearMatch = re.search("\d year", line)

        monthMatch = re.search("\d month", line)

        if(yearMatch):

            yearMatch = re.sub(" year", "", yearMatch.group())

            totalLength += int(yearMatch)

        if(monthMatch):

            monthMatch = re.sub(" month", "", monthMatch.group())

            totalLength += (0.1 * int(monthMatch))

        return totalLength

        
dc = DataCleaner

data = []

reqData = []

diversityData = []



# Main Data-Mining Loop

for filename in os.listdir(bulletin_dir):

    with open(bulletin_dir + "/" + filename, 'r', errors='ignore') as file:

        lines = file.readlines()

        it = iter(enumerate(lines))

        job_class_title, job_duties, requirements, license_required = "", "", "", None

        

        for index, line in it:

            # Since job titles are always the first *written* line (some docs have leading white space), we can apply this condition

            if(len(job_class_title) < 2):

                job_class_title = line.split("Class")[0].strip().upper()

            if "Open Date:" in line:

                job_bulletin_date = dc.openDateCleaner(line)                 

            if "Class Code:" in line:

                job_class_no = line.split("Class Code:")[1].strip()[:4]      

                

            if "ANNUAL SALARY" in line:

                tempIndex = dc.skipWhiteLines(index, lines)

                line = lines[tempIndex].strip()               

                # Check for the three types of salary formats in the next non-empty line after ANNUAL SALARY

                entry_salary_gen = dc.salaryFormatDetect(line)

                entry_salary_dwp = dc.dwpSalaryDetect(line)

                # Check if the position is only offered within the DWP and should therefore be put as _gen

                if(entry_salary_dwp and line.strip()[0] == 'T'):

                    salaryDWP = re.split("is *", line)[1]

                    entry_salary_gen = dc.salaryFormatDetect(salaryDWP)

                # If DWP salary specification was not in same line as general salary, check if it exists below

                tempIndex = dc.skipWhiteLines(tempIndex, lines)

                line = lines[tempIndex].strip()

                entry_salary_dwp = dc.dwpSalaryDetect(line)

                index = dc.skipIterations(tempIndex, index)

                

            if "DUTIES" in line:

                tempIndex = dc.skipWhiteLines(index, lines)

                line = lines[tempIndex].strip()

                job_duties = line

                while(re.search("[A-Z]{4}", line) is None):

                    tempIndex = dc.skipWhiteLines(tempIndex, lines)

                    line = lines[tempIndex].strip()

                    job_duties = job_duties + line

                index = dc.skipIterations(tempIndex, index)

            

            # Requirements are taken in bulk to be processed later to ensure appropriate number of entries due to how data is listed in the graph

            if re.search("REQUIREMENT", line):

                tempIndex = dc.skipWhiteLines(index, lines)

                line = lines[tempIndex]

                requirement = ""

                while(re.search("[A-Z]{3}", line) is None):

                    requirement = requirement + line

                    tempIndex = dc.skipWhiteLines(tempIndex, lines)

                    line = lines[tempIndex]

                index = dc.skipIterations(tempIndex, index) 

                

            if re.search("EXAMINATION", line):

                exam = dc.examTypeDetect(line, lines[dc.skipWhiteLines(index, lines)])

                

            if re.search("may require a valid California driver's license", line):

                license_required = "P"

            elif re.search("license is required", line):

                license_required = "R"

        

        # Currently implemented by parsing the entire document again separately. Should be more neatly implemented within said loop. 

        # O(lf) still holds but is not optimized. 

        word_count = 0

        male_bias = 0

        female_bias = 0

        for line in lines:

            word_count+= countWords(line)

            line = line.split()

            for word in line:

                if(word in maleBiasWords):

                    male_bias += 1

                    maleBiasCount[word] = maleBiasCount[word] + 1

                elif(word in femaleBiasWords):

                    female_bias += 1

                    femaleBiasCount[word] = femaleBiasCount[word] + 1

        

        # Parse collected requirement block separately as certain blocks of information e.g. exam_type come after requirements in a document

        requirementsList = requirement.splitlines()

        reqData = dc.parseRequirements(requirementsList, reqData)

    

        data.append([filename, job_bulletin_date, job_class_title, job_duties, job_class_no, exam, 

                     entry_salary_gen, entry_salary_dwp, license_required])

        diversityData.append([job_class_title, word_count, male_bias, female_bias])



reqDF = pd.DataFrame(reqData) 

reqDF.columns = ["JOB_CLASS_TITLE", "REQUIREMENT_SET_ID", "REQUIREMENT_SUBSET_ID", "EDUCATION_LENGTH", "EDUCATION_TYPE", "EDUCATION_MAJOR",

                "EXPERIENCE_LENGTH", "EXPERIENCE_TYPE", "EXP_JOB_CLASS_TITLE", "EXP_JOB_CLASS_ALT_RESP", "EXP_JOB_CLASS_FUNCTION", "COURSE_COUNT", "COURSE_LENGTH", "COURSE_SUBJECT"]

df = pd.DataFrame(data)

df.columns = ["FILE_NAME", "OPEN_DATE", "JOB_CLASS_TITLE", "JOB_DUTIES", 

              "JOB_CLASS_NO","EXAM_TYPE", 'ENTRY_SALARY_GEN', 'ENTRY_SALARY_DWP', "DRIVERS_LICENSE_REQ"]



# Separate DF used for later analysis

diversityDF = pd.DataFrame(diversityData)

diversityDF.columns = ["JOB_CLASS_TITLE", "WORD_COUNT", "MALE_BIASED_WORDS", "FEMALE_BIASED_WORDS"]



# Join DF's based on job title to ensure jobs with multiple requirements have their entries grouped together and then order based on data dictionary

combinedDF = df.join(reqDF.set_index('JOB_CLASS_TITLE'), on='JOB_CLASS_TITLE')

colShift = ["FILE_NAME", "OPEN_DATE", "JOB_CLASS_TITLE", "JOB_DUTIES", "JOB_CLASS_NO", "REQUIREMENT_SET_ID", "REQUIREMENT_SUBSET_ID",

            "EDUCATION_LENGTH", "EDUCATION_TYPE", "EDUCATION_MAJOR","EXPERIENCE_LENGTH", "EXPERIENCE_TYPE", "EXP_JOB_CLASS_TITLE", "EXP_JOB_CLASS_ALT_RESP", 

            "EXP_JOB_CLASS_FUNCTION", "COURSE_COUNT", "COURSE_LENGTH","COURSE_SUBJECT", "DRIVERS_LICENSE_REQ", "EXAM_TYPE", 'ENTRY_SALARY_GEN', 'ENTRY_SALARY_DWP']

combinedDF = combinedDF[colShift]

combinedDF.head(10)



# Replace with preferred path for output

#combinedDF.to_csv('job_data.csv')
combinedDF[combinedDF['JOB_CLASS_TITLE'] == 'SYSTEMS ANALYST']
def salaryToInt(salary):

    salary = re.findall("\$\d+", salary)[0]

    salary = re.sub("\$", "", salary)

    return int(salary)



def dwpSalaryDiff(gen, dwp):

    gen = re.findall("\$\d+", gen)[0]

    gen = re.sub("\$", "", gen)

    gen = int(gen)

    dwp = re.findall("\$\d+", dwp)[0]

    dwp = re.sub("\$", "", dwp)

    dwp = int(dwp)

    return (dwp-gen)



plt.figure(figsize= (9, 4))

plt.subplot(1, 2, 1)

s = df[['ENTRY_SALARY_GEN']].copy()

s.loc[:,'ENTRY_SALARY_GEN'] = s['ENTRY_SALARY_GEN'].apply(salaryToInt)

citySal = sns.distplot(s['ENTRY_SALARY_GEN'], kde=False)

citySal.set(xlabel='Base Salaries ($)', ylabel='Number of Jobs')

citySal.set_title("City Salary Distribution")



plt.subplot(1, 2, 2)

diff = df[['ENTRY_SALARY_GEN', 'ENTRY_SALARY_DWP']].copy().dropna()

diff['Difference'] = diff.apply(lambda row: dwpSalaryDiff(row['ENTRY_SALARY_GEN'], row['ENTRY_SALARY_DWP']), axis=1)

diff = sns.distplot(diff['Difference'], kde=False, color='m')

diff.set(xlabel='Difference in Salary ($)')

diff.set_title("DWP Salary Differential")
plt.figure(figsize= (9, 4))

plt.subplot(1, 2, 1)

cityLength = sns.distplot(diversityDF['WORD_COUNT'], kde=False)

cityLength.set(xlabel='Number of Words', ylabel='Number of Jobs')

cityLength.set_title("City Post Length")

#df['WORD_COUNT'].median()



plt.subplot(1, 2, 2)

privateLength = sns.distplot(privateSectorJobs['WORD_COUNT'], kde=False, color='r')

privateLength.set(xlabel='Number of Words')

privateLength.set_title("Private Sector Post Length")

#privateSectorJobs['WORD_COUNT'].mean()

diversityDF['WORD_COUNT'].mean()
fig = plt.figure(figsize= (9, 8))

plt.subplot(2, 2, 1)

mbw = sns.distplot(diversityDF['MALE_BIASED_WORDS'], bins=8, kde=False)

mbw.set(xlabel='Word Count', ylabel='Frequency')

mbw.set_title("# of Male-Biased Words per Posting")



plt.subplot(2, 2, 2)

commonMaleWords = pd.DataFrame.from_dict(maleBiasCount, orient='index', columns=['Count'])

commonMaleWords = commonMaleWords[(commonMaleWords.T != 0).any()]

commonMaleWords = commonMaleWords.sort_values(by=['Count'], ascending=False)

cmw = sns.barplot(data=commonMaleWords.reset_index(), y='Count', x='index')

cmw.set(xlabel='Male-Biased Words')

cmw.set_title('Frequency of Specific Male-Biased Words')

plt.xticks(rotation=90)



plt.subplot(2, 2, 3)

mbw = sns.distplot(diversityDF['FEMALE_BIASED_WORDS'], kde=False)

mbw.set(xlabel='Word Count', ylabel='Frequency')

mbw.set_title("# of Female-Biased Words per Posting")



plt.subplot(2, 2, 4)

commonFemaleWords = pd.DataFrame.from_dict(femaleBiasCount, orient='index', columns=['Count'])

commonFemaleWords = commonFemaleWords[(commonFemaleWords.T != 0).any()]

commonFemaleWords = commonFemaleWords.sort_values(by=['Count'], ascending=False)

cfw = cmw = sns.barplot(data=commonFemaleWords.reset_index(), y='Count', x='index')

cmw.set(xlabel='Female-Biased Words')

cmw.set_title('Frequency of Specific Female-Biased Words')

plt.xticks(rotation=90)



fig.tight_layout()
maleBiasedJobs = diversityDF[['JOB_CLASS_TITLE', 'MALE_BIASED_WORDS']].copy()

maleBiasedJobs = maleBiasedJobs.sort_values(by=['MALE_BIASED_WORDS'], ascending=False)

maleBiasedJobs.head(10)
maleBiasedJobs.tail(10)
femaleBiasedJobs = diversityDF[['JOB_CLASS_TITLE', 'FEMALE_BIASED_WORDS']].copy()

femaleBiasedJobs = femaleBiasedJobs.sort_values(by=['FEMALE_BIASED_WORDS'], ascending=False)

femaleBiasedJobs.head(10)