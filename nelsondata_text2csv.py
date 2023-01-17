



import pandas as pd

import re

import os

import nltk



def salaryParser(verboseLog, row,line,file):

    """ Find the lowest and highest salary in the salary range



    row - dictionary for each Job Bulletin

    line - file input



    """

    # remove special characters and letters

    line = re.sub("\$" , "", line)



    # store information about salary

    salaryNotes = re.sub('[0-9;]','',line)

    salaryNotes = re.sub(' and ','',salaryNotes)

    salaryNotes = re.sub(' to','',salaryNotes)

    row["SALARY_NOTES"] = re.sub(",", " ", salaryNotes)



    # store salaries

    line = re.sub('[a-zA-Z():;\-\%\&\*\/\_\,\"\']','',line)



    # is dot the end of the sentence or a decimal indicator

    decimalSalary = re.findall('\d{1}\.\d{2}',line)

    m = re.search('\d{1}\.\d{2}', line)

    if m:

        f = line[:m.start()+1]

        e = line[m.end():]

        line = f+e



    m = re.search('\d{1}\.\d{1}\.', line)

    if m:

        f = line[:m.start()+1]

        e = line[m.end():]

        line = f+e



    line = re.sub("\.","",line)



    salaryRange = line.split()

    # convert to integer

    try:

        salaryRange = list(map(int,salaryRange))

    except:

        if verboseLog:

            print("CHECK THIS ", file, " at line: ", line)

        

    # remove numbers that are not in a normal salary range

    outliers = []

    outliers[:] = (value for value in salaryRange if value < 10000)

    try:

        for outlier in outliers:

            if verboseLog:

                print("SALARY OUTLIER: ", file, line, outlier)

                salaryRange.remove(outlier)

        if len(salaryRange) > 0:

            row["LOW_SALARY"] = salaryRange[0]

            row["HIGH_SALARY"] = salaryRange[-1]

    except:

        if verboseLog:

            print("Outliers: ", outlier)



def dutiesParser(verboseLog,row,line):

    """ Adds job duties



    row - dictionary for each Job Bulletin

    line - file input

    

    """

    row["JOB_DUTIES"] = line





def experienceParser(verboseLog, degrees,schoolTypes,jobTypes,years,row,line):

    """ Finds education and experience needed for each job



    degrees - list of degrees

    schoolTypes - list of types of schools, such as college, trade school...

    jobTypes - list of job types, such as full-time, part-time...

    years - list translating text to number, for example "one year" translates to 1

    row - dictionary for each Job Bulletin

    line - file input

    

    """

    wordHist = wordHistogram(line)



    # EDUCATION

    for degree in degrees:

        if degree[0] in line.lower():

            row["DEGREE"] = degree[0]

            row["EDUCATION_YEARS"] = degree[1]

            break



    for schoolType in schoolTypes:

        if schoolType in line.lower():

            row["SCHOOL_TYPE"] = schoolType

            break



    if 'semester' in line.lower():

        row["COURSE_LENGTH_SEMESTER"] = 6

        line = re.sub("," , " ", line)

        row['MISC_COURSE_DETAILS'] = line

    if 'quarter' in line.lower():             # Assumes semester and quarter are used in same sentence

        row["COURSE_LENGTH_QUARTER"] = 9



    # EXPERIENCE

    for jobType in jobTypes:

        if jobType in line.lower():

            row["FULL_TIME_PART_TIME"] = jobType

            break



    # Years or months of required experience 

    partsOfSpeech = nltk.pos_tag(line.split())

    reg_exChunk = r"""NumYrs: {<CD>*<NNS>*}"""

    regexParser = nltk.RegexpParser(reg_exChunk)

    chunkd = regexParser.parse(partsOfSpeech)

    for subtree in chunkd.subtrees(filter=lambda t: t.label() == 'NumYrs'):

        s = str(subtree)

        if 'year' in s:

            m = re.search("/",s)

            m = m.start()

            textYr = s[7:m].lower().strip(' ')

            for year in years:

                y = re.sub(' year', '', year[0])

                if y in textYr:

                    row["EXPERIENCE_LENGTH"] = str(year[1])

                    break



        if 'month' in s:

            m = re.search("/",s)

            m = m.start()

            if s[7:m].lower().strip(' ') == 'six':

                row["EXPERIENCE_LENGTH"] = 0.5                



    # type of experience required

    reg_exChunk = r"""expType: {<JJ>*<VBD>*<NN>*}"""

    regexParser = nltk.RegexpParser(reg_exChunk)

    chunkd = regexParser.parse(partsOfSpeech)

    for subtree in chunkd.subtrees(filter=lambda t: t.label() == 'expType'):

        s = str(subtree)

        if 'experience' in s:

            s = re.sub("/NN","",s)

            s = re.sub("/JJ","",s)

            s = re.sub("/VBD","",s)

            s = re.sub("\n", "", s)

            row["EXP_JOB_CLASS_FUNCTION"] = s[9:-1]



    # job titles from prior experience

    reg_exChunk = r"""expTitle: {<NNP>*}"""

    regexParser = nltk.RegexpParser(reg_exChunk)

    chunkd = regexParser.parse(partsOfSpeech)

    for subtree in chunkd.subtrees(filter=lambda t: t.label() == 'expTitle'):

        s = str(subtree)

        s = re.sub("/NNP","", s)

        s = re.sub("\n", "", s)

        s = re.sub(";#", "", s)

        if s[10:-1] not in ("Note: Supplemental Forms", "Los Angeles", "Requirement", "Graduation", "Bachelor's"):

            row["EXP_JOB_CLASS_TITLE"] = s[10:-1]



def notesParser(verboseLog, row,line):

    """ Finds more information about job requirements



    row - dictionary for each Job Bulletin

    line - file input

    

    """

    wordHist = wordHistogram(line)

    

    commonNotes = mostCommon(wordHist,'most')

    common = ''

    for c in commonNotes:  common += c[1] + ' '

    row["NOTES_MOST_COMMON"] = common





    # LICENSE REQUIRED

    if "license" in line.lower():

        

        if "valid california driver's license" in line.lower() or "valid california's driver's license" in line.lower():

            row["DRIVER_LICENSE_REQ"] = "Y"

            row["DRIV_LIC_TYPE"] = "C"

            

        elif "valid class a california driver's license" in line.lower():

            row["DRIVER_LICENSE_REQ"] = "Y"

            row["DRIV_LIC_TYPE"] = "A"



        elif "valid california class b driver's license" in line.lower():

            row["DRIVER_LICENSE_REQ"] = "Y"

            row["DRIV_LIC_TYPE"] = "B"



        elif "Class 1/a or 2/b california driver's license" in line.lower() or "valid class a or b driver's license" in line.lower():

            row["DRIVER_LICENSE_REQ"] = "Y"

            row["DRIV_LIC_TYPE"] = "A or B"

        

        elif "valid medical certificate" in line.lower():

            row["ADDTL_LIC"] = "Medical Certificate"



def mostCommon(wordHist,type,):

    """Makes a list of how frequently a word is used and a list of title case words



    wordHist: map frequency of word

    type:   most - sort by most frequent

            least - sort by least frequent



    returns: list of (frequency, word) pairs and a list of title case words

    """

    w = []

    grammar_words = ['a','an','as','at','be','by','go','if','in','is','of','on','or','to','all','and','are','for','job','may','not','the','who','iii','iv','vi','los','angeles','with','that','this','which','who','what','when','where','are','list','time','open','civil','meet','interview','hiring','must','required','received','process','position','minimum','maximum','experience','least','work','year', 'years','relating','City','Los','Angeles', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'will']

    for wordkey, idxvalue in wordHist.items():

        if wordkey not in grammar_words:

            if len(wordkey) > 3:

                w.append((idxvalue,wordkey))

    if len(w) > 0:

        w.sort()

        if type == 'most':

            w.reverse()    

    return w[0:5]



def wordHistogram(line):

    wordHist = {}

    for word in line.split():

        if word.startswith('http://') or word.startswith('https://'):

            continue

        else:

            wordHist[word] = wordHist.get(word, 0) + 1

    return wordHist





def main():

    """ Reads job bulletins """



    verboseLog = False  # True - show parsing messages, False - do not show parsing messages

    

    JobBulletins = [os.path.join(root, file) for root, folder, JobBulletin in os.walk('../input/jobbulletindata/JobBulletins') for file in JobBulletin]

    print(len(JobBulletins), " Job Bulletins\n")

    if not verboseLog:

        print('Messages muted.  Set "verboseLog = True" to see parsing messages')



    degrees = [['bachelor',4], ['master',6], ['phd',10],['associate',2],['degree',4],['graduat',4],['apprenticeship',0]]

    schoolTypes = ['college or university','college', 'university','trade school', 'high school']

    jobTypes = ['full','part','contract','intern']

    years = [['one year',1.0], ['two year',2.0], ['three year',3.0], ['four year',4.0], ['five year',5.0], ['six year',6.0], ['seven year',7.0], ['eight year',8.0], ['nine year',9.0]]

    verifyText = []     # save original text in a DataFrame

    csvList = []        # store DataFrame fields for creating a Pandas DataFrame for further analysis



    idx = 0   # for dataframe index

    for JobBulletin in JobBulletins:

        filename = re.sub('[\']','',JobBulletin)  # strip the single quote so the file will open

        f = open(filename)



        row = {}

        verifyTextRow = {}

        verifyTextRow["FILE_NAME"] = filename



        salary = ''

        experience = ''

        notes = ''

        readIT = 0



        row["FILE_NAME"] = filename

        r = re.findall('[0-9]{4}',filename)

        JobClass = r[0]

        row["JOB_CLASS_NO"] = JobClass[0:4]

        

        k=0     # counts each line in the file

        try:

            for line in f:

                line = line.strip('\n')

                line = re.sub("," , "", line)

                lineWithText = re.search('[a-zA-Z]',line)



                if lineWithText:



                    if k==0:

                        # get the job class title from the first line of the Job Bulletin

                        line = re.sub('[^A-Z ]','',line)

                        line = re.sub('  ','',line)     # some job bulletins have blank spaces before the "Class Code" at the end of the first line

                        line = re.sub('C C','',line)    # some job bulletins have "Class Code" at the end of the first line so it is necessary to remove the remaining 'C C'

                        row["JOB_CLASS_TITLE"] = line.strip(' ')



                    if "Class Code:" in line and "." not in line:

                        verifyJobClass = line[len(line)-4:len(line)]

                        

                        # if the job class numbers don't match then print messages giving a human the data to review

                        if JobClass[0:4] != verifyJobClass:

                            if verboseLog:

                                 print("\nJOB_CLASS_NO extracted from filename: ", JobClass)

                                 print("JOB_CLASS_NO inside the file: ", verifyJobClass)

                                 print("Line where job class number was found: ", line)



                    elif "open date" in line.lower():

                        row["OPEN_DATE"] = line[len(line)-8:len(line)]



                    elif "$" in line:

                        readIT = 1



                    elif "DUTIES" in line:

                        readIT = 2

                        continue



                    elif "REQUIREMENT" in line or "ADDITIONAL JOB INFORMATION" in line:

                        readIT = 3

                        continue



                    elif "NOT" in line :

                        readIT = 4

                        continue



                    elif "WHERE TO APPLY" in line:

                        continue



                    elif "APPLICATION DEADLINE" in line:

                        continue



                    elif "SELECTION PROCESS" in line:

                        continue



                    elif "EQUAL OPPORTUNITY EMPLOYER" in line:

                        break

                    

                    if readIT == 1:

                        # add a space to prevent $100,0006. Commodit... where $100,000 is the salary and 6. Commodit is a second line containing a $

                        salary += " " + line



                    if readIT == 2:

                        dutiesParser(verboseLog, row,line)



                    if readIT == 3:

                        experience += line



                    if readIT == 4:

                        notes += line

                    k+=1

        except:

            if verboseLog:

                print("\nDOUBLE CHECK THIS FILE: ", filename)



        # Parse salary, experience and notes

        verifyTextRow['SALARY'] = salary

        verifyTextRow['EXPERIENCE'] = experience

        verifyTextRow['NOTES'] = notes

        salaryParser(verboseLog, row,salary,filename)

        experienceParser(verboseLog, degrees,schoolTypes,jobTypes,years,row,experience)

        notesParser(verboseLog, row,notes)



        # Store each row

        row["IDX"] = idx

        csvList.append(row)

        verifyText.append(verifyTextRow)

        idx += 1



    

    # RESULTS

    num2View = 10



    dfVerify = pd.DataFrame(verifyText)

    dfVerify.index.name = 'IDX'



    df = pd.DataFrame(csvList)

    df.index.name = 'IDX'



    pd.options.display.max_columns=len(df)

    

    print("\nCOLUMNS in the JOB BULLETIN CSV FILE:\n")

    print(df.info())



    print("\n\nSAMPLE DATA in JOB BULLETIN CSV FILE:")

    print("\nJob Class, open date, and job title:\n")

    print(df.loc[:,['JOB_CLASS_NO', 'OPEN_DATE', 'JOB_CLASS_TITLE']].head(num2View)) 

    

    print("\nLow salary, high salary, and notes about salaries in other departments")

    print(df.loc[:,['JOB_CLASS_NO', 'LOW_SALARY', 'HIGH_SALARY', 'SALARY_NOTES']].head(num2View))



    print("\nEducation")

    print(df.loc[:,['JOB_CLASS_NO', 'SCHOOL_TYPE', 'DEGREE', 'COURSE_LENGTH_QUARTER', 'COURSE_LENGTH_SEMESTER', 'EDUCATION_YEARS']].head(num2View))

    print(df.loc[:,['MISC_COURSE_DETAILS']].head(num2View))



    print("\nExperience")

    print(df.loc[:,['JOB_CLASS_NO', 'EXPERIENCE_LENGTH', 'EXP_JOB_CLASS_FUNCTION', 'EXP_JOB_CLASS_TITLE']].head(num2View))

    print(df.loc[:,['JOB_CLASS_NO', 'ADDTL_LIC', 'DRIVER_LICENSE_REQ', 'DRIV_LIC_TYPE']].head(num2View))

    print(df.loc[:,['JOB_CLASS_NO', 'FULL_TIME_PART_TIME']].head(num2View))



    print("\nJob Duties")

    print(df.loc[:,['JOB_CLASS_NO', 'JOB_DUTIES']].head(num2View))



    print("\nTowards text comprehension - finding the most common words other than grammar words or stopwords in the Notes")

    print(df.loc[:,['JOB_CLASS_NO', 'NOTES_MOST_COMMON']].head(num2View))



    # csv file

    

    df.to_csv("JobBulletin.csv")

    dfVerify.to_csv("VerifyDataJobBulletin.csv")



if __name__ == '__main__':

    main()

# Analysis of Missing Data



df = pd.read_csv("JobBulletin.csv")



print("\n\nANALYSIS of MISSING DATA:\n")

print("Number of null entries in each column:")

print(df.isnull().sum())



print("\n\nMissing JOB_CLASS_TITLE")

jc = df[df['JOB_CLASS_TITLE'].isnull()]

print(jc['FILE_NAME'])



print("\nMissing OPEN_DATE")

jc = df[df['OPEN_DATE'].isnull()]

print(jc['FILE_NAME'])



print("\nMissing JOB_DUTIES")

jc = df[df['JOB_DUTIES'].isnull()]

print(jc['FILE_NAME'])



df = pd.read_csv("JobBulletin.csv")



import matplotlib.pyplot as plt

df['LOW_SALARY'].plot(figsize=(10,5), y='Low Salary', kind='hist', title='Low Salary Range')

plt.show()    



df['HIGH_SALARY'].plot(figsize=(10,5), y='High Salary', kind='hist', title='High Salary Range')

plt.show()    



print("\nMissing LOW_SALARY")

ls = df[df['LOW_SALARY'].isnull()]

print(ls['FILE_NAME'])



print("\nMissing HIGH_SALARY")

hs = df[df['HIGH_SALARY'].isnull()]

print(hs['FILE_NAME'])
df = pd.read_csv("JobBulletin.csv")



print("\nUNIQUE EXP_JOB_CLASS_FUNCTION")

print(df.EXP_JOB_CLASS_FUNCTION.unique())



print("\nUNIQUE EXP_JOB_CLASS_TITLE")

print(df.EXP_JOB_CLASS_TITLE.unique())