import re

from os import listdir

from os.path import isfile, join

import pandas as pd

#Get the list of files under Job Bulletins

foldername='../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/'

filelist = [f for f in listdir(foldername) if isfile(join(foldername, f))]



##PROCESSING the available Job Tiles provided in the Kaggle Competition Dataset 



JobTitleFoldername = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/'

JobTitleFilename = 'job_titles.csv'

JobTitleentirepath= JobTitleFoldername + JobTitleFilename



#Setting an empty JobTile list

JobTitle = []



#Read the Jobtitles into a dataframe and convert it to list

JobTitleDF = pd.read_csv(JobTitleentirepath, header=None)

JobTitle = JobTitleDF[0].values.tolist()



#Sort and reverse the Jobtile list based on the length of the element

#  so that SENIOR CARPENTER is in the top of list be for CARPENTER 

#  this is will help in sorting the list based on seniority as currently we dont have any such alignment data available  

JobTitle.sort(key=len)

JobTitle.reverse()



#Converting the JobTile list to a pattern so that the element are seperated by or '|

#  This pattern will be used further in regex search function

JobtilePattern = "'abc|"+'|'.join(JobTitle) + "|xyz'"



##END: PROCESSING the available Job Tiles provided in the Kaggle Competition Dataset 

        



#Initialize a list which will collect JobClass and Missing JobClass info

JobClassALL = []

MissingJobClassALL = []

MismatchJobClassnFileNameALL = []



#Process the files in Job Bulletins folder

for filename in filelist:



    entirepath= foldername + filename

    

    textfile = open(entirepath,'r',encoding='latin-1')

    filetext = textfile.read()

    textfile.close()



#initializing the List for current file as EMPTY

    JOBClass =[0,0,0,0,0,0,0,0,0,0,0]

    MissingJobClass = [0]

    MismatchJobClassnFileName = [0]



#Populating the filename as a first element of list

    JOBClass[0] = filename



#EXTRACTING Job Class Title from the files

# Few files have 'CAMPUS INTERVIEWS ONLY' tag in the beginning. Omitting that

# Few files have acrynom in Job tiles enclosed () . Omitting that

    Refinefiletext = re.sub('CAMPUS INTERVIEWS ONLY','',filetext.strip()).strip()

    JobClassExtract = Refinefiletext[:re.search('\w*\n',Refinefiletext.strip()).end()-1]

    JobClassTransform = re.sub('[ ]*\(.*\)[ ]*',' ',re.split('\s{2,}|\t',JobClassExtract)[0]).strip()

#END: EXTRACTING Job Class Title from the files



#Populating the JobTile in the list

    JOBClass[1] = JobClassTransform



#BEGIN :  Matching JobTile of the plain text file with the available JobTile list

# Checking if the JobTile fetched from the job bulletin files are matching with the available Job Tile list

# If the Job Titles are not available, writting them in seperate file

#    

    JobClassSrch = re.search(JobtilePattern.upper().strip(), JobClassTransform.upper())

    if JobClassSrch is not None:

        JOBClass[2] = JobClassTransform[JobClassSrch.start():JobClassSrch.end()]

#       Check if the Job Class in the plain text file matches with Job Class in the Filename

        if re.search(JOBClass[2].strip(), filename) is None:

            MismatchJobClassnFileName[0] = filename

            MismatchJobClassnFileNameALL.append(MismatchJobClassnFileName)



    if JOBClass[1] != JOBClass[2] :

        MissingJobClass[0] = JOBClass[1]

        MissingJobClassALL.append(MissingJobClass)

#END :  Matching JobTile of the plain text file with the available JobTile list



#Appending the current files outcome to the main list

    JobClassALL.append(JOBClass)





#Converting into panda dataframe

JobClassALLDF = pd.DataFrame(JobClassALL)

MissingJobClassDF = pd.DataFrame(MissingJobClassALL)

MismatchJobClassnFileNameDF = pd.DataFrame(MismatchJobClassnFileNameALL)



#Adding identified missing JobTile to the dataframe that has the existing JobTitle list

JobTitleDF = JobTitleDF.append(pd.DataFrame(MissingJobClassALL),ignore_index=True)



print("Job Tiles missing from the Job Tile csv available in the Comptetion data")

print(MissingJobClassDF)



print("Job Tiles Mismatch from the Job Tile  in plain text Filename and it's content")

print(MismatchJobClassnFileNameDF)



#Writing into csv file for the ease of analysis

JobClassALLDF.to_csv("JOBClassExtract.csv",sep=',',index=None)

MissingJobClassDF.to_csv("MissingJobClass.csv",sep=',',index=None,header=None)

JobTitleDF.to_csv("NewJobTitles.csv",index=None,header=None)

MismatchJobClassnFileNameDF.to_csv("MismatchJobClassnFileName.csv",index=None,header=None)

import re

from os import listdir

from os.path import isfile, join

import pandas as pd

import matplotlib.pyplot as plt

#Get the list of files under Job Bulletins

foldername='../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/'

filelist = [f for f in listdir(foldername) if isfile(join(foldername, f))]



#Initialize a list which will collect Salary info

SalaryALL = []

LosAnglesSalaryManyALL = [] 

DWPSalaryCleanManyALL = []

#Process the files under Job Bulletins

for filename in filelist:



    entirepath= foldername + filename

    

    textfile = open(entirepath,'r',encoding='latin-1')

    filetext = textfile.read()

    textfile.close()



#initializing the List for current file as EMPTY

    SalOut =[0,0,0,0,0,0,0,0,0,0,0]



#Populating the filename as a first element of list

    SalOut[0] = filename



#Spliting the file text into two parts, Second part one contains text after salary

    FromsalaryData = re.split('ANNUAL\s?SALARY',filetext)

    DWPCount = re.findall('DepartmentofWaterandPower',re.sub('\n|\s+|\,|\*','',filetext))

#Check the lenght of the list to see if any file doesnt contain ANNUAL SALARY details

#    Identified one file which doesnt have the ANNUAL SALARY details - Vocational Worker  DEPARTMENT OF PUBLIC WORKS.txt

#    Reviewed the file and it doesnt contain salary details so setting salary to NA   

    if len(FromsalaryData) > 1:



#Spliting the file to extract the salary section seperately

        SalaryData = re.split('[A-Z]{3,}',FromsalaryData[1])

        if len(SalaryData) > 1:

            SalaryDataStrip = SalaryData[0].strip()



#Removing special charcter from the data

            SalaryDataClean = re.sub('\n|\s+|\,|\*','',SalaryDataStrip)

            

#Splitting the data to get both the salary details- City of Los Angles and  Department of Water and Power Salary

            DWPSplit = re.split('DepartmentofWaterandPower',SalaryDataClean)



#Extracting the section to get the City of Los Angles Salary details

            LosAnglesSalary = DWPSplit[0]



#Extracting the section to get the Department of Water and Power Salary details

            if len(DWPSplit) > 1:

               DWPSalary = DWPSplit[1]

            else:

               DWPSalary = 'NA'



            LosAnglesSalaryClean = re.findall('\$\d{3,}\-\$\d{3,}|\$\d{3,}[a-z\W]*',re.sub('to','-',LosAnglesSalary))              

            if len(LosAnglesSalaryClean) > 0:    

                if re.search('\$\d{3,}\-\$\d{3,}',LosAnglesSalaryClean[0]) : 

                    SalOut[1] = re.sub('\$','',LosAnglesSalaryClean[0])

                else:

                    SalOut[1] = re.sub('[a-z\W\,\s]','',LosAnglesSalaryClean[0])

            else:

                SalOut[1] = 'N/A'

                

            DWPSalaryClean = re.findall('\$\d{3,}\-\$\d{3,}|\$\d{3,}[a-z\W]*',re.sub('to','-',DWPSalary))              

            if len(DWPSalaryClean) > 0:    

                if re.search('\$\d{3,}\-\$\d{3,}',DWPSalaryClean[0]) : 

                    SalOut[2] = re.sub('\$','',DWPSalaryClean[0])

                else:

                    str2 = re.sub('[a-z\W\,\s]','',DWPSalaryClean[0])

                    SalOut[2] = str2

            else:

                SalOut[2] = 'N/A'

            

#Extracting the details when more than one salary range is specified in Job Bulletins

            if len(LosAnglesSalaryClean) >1: 

                LosAnglesSalaryMany = [x for x in LosAnglesSalaryClean]

                SalOut[3] = LosAnglesSalaryMany

                LosAnglesSalaryManyALL.append([SalOut[0],LosAnglesSalaryMany,len(LosAnglesSalaryMany)])

            if len(DWPSalaryClean) >1:

                DWPSalaryCleanMany = [x for x in DWPSalaryClean ]

                SalOut[4] = DWPSalaryCleanMany

                DWPSalaryCleanManyALL.append([SalOut[0],DWPSalaryCleanMany,len(DWPSalaryCleanMany),len(DWPCount)])

 

    SalOut[5] = DWPCount

    SalOut[6] = len(DWPCount)

       

#Appending the current files outcome to the main list

    SalaryALL.append(SalOut)

#Converting into panda dataframe

SalaryALLDF = pd.DataFrame(SalaryALL)

LosAnglesSalaryManyALLDF = pd.DataFrame(LosAnglesSalaryManyALL,columns=['filename','Salary Range','Count'])

DWPSalaryCleanManyALLDF = pd.DataFrame(DWPSalaryCleanManyALL,columns=['filename','Salary Range','Count','DWPCount'])



#Bar plot to show how many salary range are available in the Job Bulletins

fig = plt.figure(figsize=(12,6))

fig.suptitle('Multiple Salary Range Mentioned In Job Bulletins', fontsize=20)

plt.subplot(1,2,1)

LosAnglesSalaryManyALLDF['Count'].value_counts().plot.bar()

plt.title('Multiple Salary figures for LosAngles Salary')

plt.xlabel('Count of Salary Range Available')

plt.ylabel('Count of JobBulletins')



plt.subplot(1,2,2)

DWPSalaryCleanManyALLDF['Count'].value_counts().plot.bar()

plt.title('Multiple Salary figures for DWP Salary')

plt.xlabel('Count of Salary Range Available')

plt.ylabel('Count of JobBulletins')



plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=None,hspace=None)

plt.show()



fig = plt.figure(figsize=(12,6))

fig.suptitle('Missing criteria to apply for Department of Water and Power', fontsize=20)

DWPSalaryCleanManyALLDF['DWPCount'].value_counts().plot.bar()

plt.title('Department of Water and Power')

plt.xlabel('# of times \'Department of Water and Power\' is used in Job Bulletin')

plt.ylabel('Count of JobBulletins')



plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=None,hspace=None)

plt.show()



#Writing into csv file for the ease of analysis

SalaryALLDF.to_csv("SalaryDF.csv",sep=',',index=None)

LosAnglesSalaryManyALLDF.to_csv("LosAnglesSalaryManyDF.csv",sep=',',index=None)

DWPSalaryCleanManyALLDF.to_csv("DWPSalaryCleanALLDF.csv",sep=',',index=None)

import re

from os import listdir

from os.path import isfile, join

import pandas as pd

import matplotlib.pyplot as plt

from datetime import date

#Get the list of files under Job Bulletins

foldername='../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/'

filelist = [f for f in listdir(foldername) if isfile(join(foldername, f))]



#Initialize a list which will collect Open Date info

OpenDateALL = []



#Get the current year. It will be used for further processing

Curryear = date.today().strftime("%Y")



#Process the files under Job Bulletins

for filename in filelist:



    entirepath= foldername + filename

    

    textfile = open(entirepath,'r',encoding='latin-1')

    filetext = textfile.read()

    textfile.close()



#initializing the List and variable for current file as EMPTY

    OpenDate =['','','','','']

    OrgOpenDate = ''

    RevDate = ''

    OpenDateValue = ''

    OpenDateYear  = ''

    OpenDateYearInt = ''

    TextNoREQ = ''



#Populating the filename as a first element of list

    OpenDate[0] = filename



#Fetching the Open Date related data from the plain text file

    OpenDateSrch = re.search('OPEN DATE(\W+)(\d{1,2}\-\d{1,2}\-\d{4})|OPEN DATE(\W+)(\d{1,2}\-\d{1,2}\-\d{2})',filetext.upper())



#Fetching the Revised Open Date related data from the plain text file

    RevisedSrch = re.search('REVISED(\W+)(\d{1,2}\-\d{1,2}\-\d{4})|REVISED(\W+)(\d{1,2}\-\d{1,2}\-\d{2})',filetext.upper())



#The below logic check is the Revised Date is available. In case it is available it will the Open Date of the Job Bulletin

#    In case Only Open Date is available it will retained as Open Date

    if OpenDateSrch is not None :

        if RevisedSrch is not None :

            OpenDateValue = filetext[RevisedSrch.start()+8:RevisedSrch.end()].strip()

            RevDate = filetext[RevisedSrch.start()+8:RevisedSrch.end()].strip()

        else:    

            OpenDateValue = filetext[OpenDateSrch.start()+10:OpenDateSrch.end()].strip()

        OrgOpenDate = filetext[OpenDateSrch.start()+10:OpenDateSrch.end()].strip()

    else:

        OpenDate[1] = 'N/A'



#Populate the evaluated open date based on availibility of Revised Date

    OpenDate[1] = OpenDateValue

#Populate the Open date availibile in the Job Bulletin

    OpenDate[2] = OrgOpenDate

#Populate the Revised Open date availibile in the Job Bulletin

    OpenDate[3] = RevDate



### BEGIN : Refernce to Old Date Analysis

# The below logic is based on the hypothesis that Job Bulletins may contains reference to older date( Older than Open Date of Job Bulletins)

# From the filetext we will remove the Requirment Details(as there are instance where it refers to old dates for specifying experience)

# We will capture the content before the REQUIREMENT DETAILS and after WHERE TO APPLY sections

# We will evaluate this new text to check for any reference to older dates

# Please note: There can still be few exceptions where older dates are guinely referred     

    BFRREQText=re.search('REQ[A-Z]{4}',filetext)

    if BFRREQText is not None:

        TextNoREQ = filetext[:BFRREQText.start()]

        AFTWTAText=re.search('WHERE TO APPLY',filetext)

        if AFTWTAText is not None:

            TextNoREQ = TextNoREQ + filetext[AFTWTAText.start():]

        else:

            TextNoREQ = filetext

    else:

        TextNoREQ = filetext

        

#The below logic formates the year of Open Date to YYYY format

#  When the year is in YY format it check if it is lesser than current year's YY format

#        In case it is less, it will be prefix by 20 for tweenth century else it will be 19

    if len(OpenDateValue) <= 8:

        if (OpenDateValue[-2:] <= Curryear[2:]):

            OpenDateYear = "20" + OpenDateValue[-2:]

        else:

            OpenDateYear = "19" + OpenDateValue[-2:]

    else:

        OpenDateYear = OpenDateValue[-4:]



#The below logic search for occurance of year older than the Open Date year

#  It check for any reference of last four year

#  If Open Date year is 2018 , it will search for any reference for 2017,2016,2015 and 2014 

    OpenDateYearInt = list(set(re.findall(str(int(OpenDateYear[0:4]) -1) + "|" + str(int(OpenDateYear[0:4]) -2) + "|" + str(int(OpenDateYear[0:4]) -3) + "|" + str(int(OpenDateYear[0:4]) -4),TextNoREQ)))



    if OpenDateYearInt == []:

        OpenDate[4] = ''

    else:

        OpenDate[4] = OpenDateYearInt

### END : Refernce to Old Date Analysis

       

#Appending the current files outcome to the main list

    OpenDateALL.append(OpenDate)

#Converting into panda dataframe

OpenDateALLDF = pd.DataFrame(OpenDateALL,columns=['Filename','Open Date','Original Open Date','Revised Open Date','Old Year Reference'])



print("Job Bulletins contains reference to older date( Older than Open Date of Job Bulletins)")

print('Count of Job Bulletines with Revised Open Date' , OpenDateALLDF['Revised Open Date'][OpenDateALLDF['Revised Open Date']!=''].count())

print('Count of Job Bulletines with Old Year Reference', OpenDateALLDF['Old Year Reference'][OpenDateALLDF['Old Year Reference']!=''].count())

print('Total Count of Job Bulletines',len(OpenDateALLDF))







#Writing into csv file for the ease of analysis

OpenDateALLDF.to_csv("OpenDateALLDF.csv",sep=',',index=None)

import re

from os import listdir

from os.path import isfile, join

import pandas as pd

#Get the list of files under Job Bulletins

foldername='../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/'

filelist = [f for f in listdir(foldername) if isfile(join(foldername, f))]



#Initialize a list which will collect Driving License info

LicALL = []



#Process the files in Job Bulletins folder

for filename in filelist:



    entirepath= foldername + filename



    textfile = open(entirepath,'r',encoding='latin-1')

    filetext = textfile.read()

    textfile.close()



#initializing the List for current file as EMPTY

    LicOut =[0,0,0,0]



#Populating the filename as a first element of list

    LicOut[0] = filename



#BEGIN : License required , possibly required or Not required



# Check if any driving License information is available in the file 

    LicBasicLicSrch  = re.findall('CLASS [A-Z]{1} .*LICENSE|DRIVER.*LICENSE',filetext.upper())



# Check if driving License is required 

    LicRQRDSrch = re.findall('DRIVER[S\']{2} LICENSE IS REQUIRED|POSSESSION OF A VALID CALIFORNIA DRIVER\'S LICENSE|DRIVER\'S LICENSE[A-W ]*REQUIRED|DRIVER\'S LICENSE[A-W ]*BY[A-W ]*REQUIRED|SOME POSITIONS REQUIRE A VALID CALIFORNIA DRIVER\'S LICENSE|THESE POSITIONS ARE OPEN TO ALL WHO APPLY WITH A VALID CALIFORNIA DRIVER\'S LICENSE',filetext.upper())



# Check if driving License is possibly required 

    LicPOSSSrch = re.findall('MAY ALSO REQUIRE A VALID CALIFORNIA DRIVER\'S LICENSE|POSITIONS MAY REQUIRE.*DRIVER\'S LICENSE|DRIVER\'S LICENSE MAY BE REQUIRED',filetext.upper())



# Check if driving License is required with special licence class

    LicSpecificSrch  = re.findall('CALIFORNIA CLASS [A-Z]{*} LICENSE IS REQUIRED|CLASS A OR B DRIVER\'S LICENSE|CLASS A DRIVER\'S LICENSE|CLASS B DRIVER\'S LICENSE|CLASS A OR CLASS B DRIVER\'S LICENSE|CALIFORNIA B DRIVER\'S LICENSE|CALIFORNIA A DRIVER\'S LICENSE',filetext.upper())



# Set the flag as 'N' for Not required, 'P' for possibly required and 'R' for required

    if LicBasicLicSrch == []:

        LicOut[1] = 'N'

    elif LicRQRDSrch != []:

        LicOut[1] = 'R'

    elif LicPOSSSrch !=[]:

        LicOut[1] = 'P'

    elif LicSpecificSrch!=[]:

        LicOut[1] = 'R'

    else:

        LicOut[1] = 'P'



#END : License required , possibly required or Not required

        

#BEGIN : Specific driving License requirement        

#  Merge all the Driving License information extracted earlier in one list

    AllSrchData = list(set(LicBasicLicSrch +LicRQRDSrch+LicPOSSSrch+LicSpecificSrch))

#  Join the merge list in one text for easy processing, manipulation and data extraction

    AllSrchDatajoin= '|'.join(AllSrchData)

#  Clean the data by removing puncuations

    All = re.sub('\"|\(\)|\'','',AllSrchDatajoin)



#  Extract the information pertaining to specific driving licence

#  Normalize the data to use proper CLASS tag before the driving licence Class information.

#    There are instances where we see tag like CLASS A or B, the idea is the make them CLASS A or CLASS B

    CleanClass  = list(set([re.sub('CLASS CLASS| CLASS\/','CLASS',re.sub('OR','OR CLASS',re.sub('CALIFORNIA ','CLASS ',x.strip()))) for x in list(set(re.findall('CLASS [A-Z]{1} OR [A-Z]{1} |CLASS [A-Z]{1} OR CLASS [A-Z]{1} |CLASS [A-Z]{1} AND [A-Z]{1} |CLASS [A-Z]{1} AND CLASS [A-Z]{1} |CLASS [A-Z]{1} AND\/OR CLASS [A-Z]{1} |CLASS [A-Z]{1} OR\/AND CLASS [A-Z]{1} |CALIFORNIA [A-Z]{1} |CLASS [A-Z]{1} ',All)))]))

#  In file where we have multiple licence classes ,we will seperate them with | to indicate OR

#    This step is pre step to that.Replace any cocurance of AND/OR or OR/AND with AND~ or ~AND to ease further processing

#    We will further split the data by 'OR' 

    RemoveOR  = list(set(re.split(' OR ',re.sub('OR\/|\/OR','~',' OR '.join(CleanClass)))))



#Extracting the Class with AND clause in it in one list and other list contains the remaining data 

    ExtractANDClass = [x for x in RemoveOR if re.search('AND',x) is not None ]

    ExtractSinClass = [x for x in RemoveOR if re.search('AND',x) is None ]

    

#initializing the List as EMPTY to contain class data

    LicCategory = []



#Identifying distinct classes 

#  Since we collected based on 'class' tag, we will fetch all the combinations.

#  Like CLASS B, CLASS A AND/OR CLASS B, CLASS A 

#  the following logic cleans up this pattern to identify distinct CLASSess

    if ExtractANDClass != []:

        for x in ExtractANDClass:

            for y in ExtractSinClass:

                if re.search(y,x) is None:

                    LicCategory = LicCategory + [y]

        LicCategory = LicCategory + [x]

    else:

        LicCategory = ExtractSinClass



#After  processing of data replace AND~ with the correct clause mentioned in the job bulletin

    LicOut[2] =re.sub('AND~|~AND','AND/OR','|'.join(LicCategory))

    LicOut[3] = All

#END : Specific driving License requirement        



    LicALL.append(LicOut)



#Converting into panda dataframe

promoDF = pd.DataFrame(LicALL)

print(promoDF[3].head())

#Writing into csv file for the ease of analysis

promoDF.to_csv("DrivingLicDetails.csv",sep=',',index=None)

#####################################################################################

#BEGIN :Function for generating list of requirement sets and subset in a JobBulletins



def RequirementSetDetails(REQ_Text):

# Summary:

#  This function extract the requirement set and sub set from the Requirements

#  sample job class export template is the basis for this logic

# Logic:

#   If the requirements are seperated by OR conjunction, the requirement details are presented in seperate rows in the output csv

#   If the requirements are seperated by AND conjunction, the requirement details are presented in current rows in the output csv

#Output:

#   The output of this function is a list of different requirement  available in a Job Bulletins

#Example 1:

#   If a Job Bulletin has requirement as:

#    1. ReQ Detail1; and 2. ReQ Detail2       ; and 3. ReQ Detail3       ; and 4. ReQ Detail4

#                        a. ReQ Detail2; or         a. ReQ Detail3; or 

#                        b. ReQ Detail2;            b. ReQ Detail3;  

#   Here the Requirments are seperated  by and conjuction but ReQ 2 and ReQ3 has two sub requirement

#   The output set for this  requirement is presented as below 

#    1. ReQ Detail1; and 2. ReQ Detail2  a. ReQ Detail2;  and 3. ReQ Detail3   a. ReQ Detail3; and 4. ReQ Detail4

#    1. ReQ Detail1; and 2. ReQ Detail2  a. ReQ Detail2;  and 3. ReQ Detail3   b. ReQ Detail3; and 4. ReQ Detail4

#    1. ReQ Detail1; and 2. ReQ Detail2  b. ReQ Detail2;  and 3. ReQ Detail3   a. ReQ Detail3; and 4. ReQ Detail4

#    1. ReQ Detail1; and 2. ReQ Detail2  b. ReQ Detail2;  and 3. ReQ Detail3   b. ReQ Detail3; and 4. ReQ Detail4

#

#Example 2:

#   If a Job Bulletin has requirement as:

#    1. ReQ Detail1; and 2. ReQ Detail2 ; and 3. ReQ Detail3 ; and 4. ReQ Detail4

#   Here the Requirments are seperated  by and conjuction 

#   The output set for this  requirement is presented as below 

#    1. ReQ Detail1; and 2. ReQ Detail2 ; and 3. ReQ Detail3 ; and 4. ReQ Detail4

#    

#Example 3:

#   If a Job Bulletin has requirement as:

#    1. ReQ Detail1; or 2. ReQ Detail2 ; or 3. ReQ Detail3 ; or 4. ReQ Detail4

#   Here the Requirments are seperated  by and conjuction 

#   The output set for this  requirement is presented as below 

#    1. ReQ Detail1; 

#    2. ReQ Detail2;

#    3. ReQ Detail3;

#    4. ReQ Detail4

#

#    

#Set the Requirement ID to 1

    REQ_SET_ID = 1

#Empth List for data processing

    ReqListNew =[]

    ReqListFinal =[]

#    print(REQ_Text)

#    print('^^^^^')



#Splitting the  requirement set pattern  on Conjunction 'OR' to get distinct requirement set

    ReqList = re.split('^\d{1,2}[\. ]{1}|[\W|\s]{1}or\s*\n\d{1,2}\W|[\W|\s]{1}OR\s*\n\d{1,2}\W',REQ_Text)

#    ReqList = re.split('^\d{1,2}\W',REQ_Text)

    

#Clean the list of requriment after spilt on basis of Conjunction 'OR'

#     to Omit Empty elements and to attach Requriment Set ID to each requirement set

    for x in ReqList:

        if x != '':

            ReqListNew = ReqListNew + [str(REQ_SET_ID) + '. ' + x]

            REQ_SET_ID = REQ_SET_ID + 1

#Follwing with Process the requirement sets obtained from previous steps 

    for y in ReqListNew:

#       Cleaning the requirement set data

        rowstrip  = y.strip().strip('\W').strip('\s').strip('(')



#       Check if both 'AND' and 'OR' conjuction 

#       This will primaraily cover the requriment where Main Requirement are seperated by 'AND' conjunction and the sub requirement have 'OR' conjuction

#         To state few example; the files are:

#          ADVANCE PRACTICE PROVIDER CORRECTIONAL CARE 2325 020808 REV 111214.txt

#          EMS ADVANCED PROVIDER 2341 111618 REV 122018.txt

        if re.findall('\;[ ]{1,}or\s*\n|\;[ ]{1,}OR\s*\n',rowstrip) != [] and re.findall('\;[ ]{1,}and\s*\n\d|\;[ ]{1,}AND\s*\n\d',rowstrip) != []:

# The if part deals with requirement that has both AND and OR conjuntion

# Splitting the  requirement text  on Conjunction 'AND' for further processing

            ReqListANDSplit = re.split('\;[ ]{1,}and\s*\n|\;[ ]{1,}AND\s*\n',rowstrip)



# Initializing list and few variables

            ANDPartText = []

            ORPartText =[]

            ORPartSrch =[]

#Looping over the set after spliting the requirement by AND conjunction            

            for x in ReqListANDSplit:

# Since the requirement seperated by AND conjuction are part of one set, renaming the Requirment identification number to letter for ease of proceessing in later stage                

                if ReqListFinal == []:

                    x = re.sub('^\d\.','a.',x)

# The below statement search from the main requirement statement in the and requirement before the sub requirement starts

#Example :

#   If a Job Bulletin has requirement as:

#    2. ReQ Detail2  

#      a. ReQ Detail2; or 

#      b. ReQ Detail2;     

#   The output set for this  statement will be as below 

#    2. ReQ Detail2 

                ANDPartSrch = re.search('\W[A-Za-z]{1}\.',x)

                if ANDPartSrch is not None:

                    ANDPartText1 =  x[:ANDPartSrch.start()]

#                   The below statement search from the main requirement statement in the and requirement before the sub requirement starts

#                   Considering the above example the out will be :

#                       [a. ReQ Detail2,b. ReQ Detail2]    

                    ORPartSrch = re.split('\;[ ]{1,}or\s*\n|\;[ ]{1,}OR\s*\n',x[ANDPartSrch.end()-2:])

#                   Appending the extracted AND part to each OR part. The output will look like:

#                       [2. ReQ Detail2 a. ReQ Detail2,2. ReQ Detail2 b. ReQ Detail2]    

                    ORPartSrch = [ ANDPartText1 + ' ' + x for x in re.split('\;[ ]{1,}or\s*\n|\;[ ]{1,}OR\s*\n',x[ANDPartSrch.end()-2:])]

                else:

                    ANDPartText = ANDPartText + [x]

                    

# Creating a list of list for all the OR clause in the requirement. For Example1 mentioned above, it will look like

#   [[2. ReQ Detail2 a. ReQ Detail2,2. ReQ Detail2 b. ReQ Detail2],[3. ReQ Detail 3 a. ReQ Detail3,3. ReQ Detail3 b. ReQ Detail3]]    

                if ORPartSrch != []:

                    ORPartText.append(ORPartSrch)



                ORPartSrch =[]



# create a single list for list of list created above

#   [2. ReQ Detail2 a. ReQ Detail2,2. ReQ Detail2 b. ReQ Detail2,3. ReQ Detail 3 a. ReQ Detail3,3. ReQ Detail3 b. ReQ Detail3]]    

            ORPartFlat = list(itertools.chain(*ORPartText))

# Get all the Combination of requirement for the above list:

#  For Example 1 above the len(ORPartText) = 2

#  Combinations will look like:

#             [2. ReQ Detail2 a. ReQ Detail2,2. ReQ Detail2 b. ReQ Detail2]

#             [2. ReQ Detail2 a. ReQ Detail2,3. ReQ Detail 3 a. ReQ Detail3]

#             :

#             :

            ORComb = list(itertools.combinations(ORPartFlat,len(ORPartText)))

            FinalList = ['~'.join(x) for x in ORPartText]

            CombList = ['~'.join(x) for x in ORComb]

# Remove the irrevant combinations

#  For Example 1 above the combination

#             [2. ReQ Detail2 a. ReQ Detail2,2. ReQ Detail2 b. ReQ Detail2]

#  is part of 

#    2. ReQ Detail2  

#      a. ReQ Detail2; or 

#      b. ReQ Detail2;   

#  which is seperated by Or conjuctions and hence omitting it

#  the below combination is valid

#           [2. ReQ Detail2 a. ReQ Detail2,3. ReQ Detail 3 a. ReQ Detail3]

#  as it represents :

#     2. ReQ Detail2       ; and 3. ReQ Detail3   

#       a. ReQ Detail2;          a. ReQ Detail3; 

            CombiFinal2 = [re.split('\~',x) for x in CombList if re.search('|'.join([re.sub('\W','',x) for x in FinalList]),re.sub('\W','',x)) is None]

#  For Example 1 above 

#    1. ReQ Detail1; and 2. ReQ Detail2       ; and 3. ReQ Detail3       ; and 4. ReQ Detail4

#                        a. ReQ Detail2; or         a. ReQ Detail3; or 

#                        b. ReQ Detail2;            b. ReQ Detail3;  

# the below logic with add the below details to all the valid combinations

#    1. ReQ Detail1; and 4. ReQ Detail4

            for x in CombiFinal2:

                ReqListFinal = ReqListFinal + [';and '.join(ANDPartText) + ';and '+ ';and '.join(x)]

        else:

# This part deals with  requirement whic has only OR conjuction or AND conjuction in requirement details

# Splitting the  requirement text  on Conjunction 'OR' for further processing

            ReqListOrSplit = re.split('\;[ ]{1,}or\s*\n|\;[ ]{1,}OR\s*\n',rowstrip)

            ReqListFinal= ReqListFinal + [z for z in ReqListOrSplit]

#            print(ReqListFinal)

    return ReqListFinal



#END :Function for generating list of requirement sets and subset in a JobBulletins

#####################################################################################

#Import Libaries 

import re

from os import listdir

from os.path import isfile, join

import pandas as pd

import itertools



#####################################################################################

#BEGIN :Function for generating list of requirement sets and subset in a JobBulletins



def RequirementSetDetails(REQ_Text):

# Summary:

#  This function extract the requirement set and sub set from the Requirements

#  sample job class export template is the basis for this logic

# Logic:

#   If the requirements are seperated by OR conjunction, the requirement details are presented in seperate rows in the output csv

#   If the requirements are seperated by AND conjunction, the requirement details are presented in current rows in the output csv

#Output:

#   The output of this function is a list of different requirement  available in a Job Bulletins

#Example 1:

#   If a Job Bulletin has requirement as:

#    1. ReQ Detail1; and 2. ReQ Detail2       ; and 3. ReQ Detail3       ; and 4. ReQ Detail4

#                        a. ReQ Detail2; or         a. ReQ Detail3; or 

#                        b. ReQ Detail2;            b. ReQ Detail3;  

#   Here the Requirments are seperated  by and conjuction but ReQ 2 and ReQ3 has two sub requirement

#   The output set for this  requirement is presented as below 

#    1. ReQ Detail1; and 2. ReQ Detail2  a. ReQ Detail2;  and 3. ReQ Detail3   a. ReQ Detail3; and 4. ReQ Detail4

#    1. ReQ Detail1; and 2. ReQ Detail2  a. ReQ Detail2;  and 3. ReQ Detail3   b. ReQ Detail3; and 4. ReQ Detail4

#    1. ReQ Detail1; and 2. ReQ Detail2  b. ReQ Detail2;  and 3. ReQ Detail3   a. ReQ Detail3; and 4. ReQ Detail4

#    1. ReQ Detail1; and 2. ReQ Detail2  b. ReQ Detail2;  and 3. ReQ Detail3   b. ReQ Detail3; and 4. ReQ Detail4

#

#Example 2:

#   If a Job Bulletin has requirement as:

#    1. ReQ Detail1; and 2. ReQ Detail2 ; and 3. ReQ Detail3 ; and 4. ReQ Detail4

#   Here the Requirments are seperated  by and conjuction 

#   The output set for this  requirement is presented as below 

#    1. ReQ Detail1; and 2. ReQ Detail2 ; and 3. ReQ Detail3 ; and 4. ReQ Detail4

#    

#Example 3:

#   If a Job Bulletin has requirement as:

#    1. ReQ Detail1; or 2. ReQ Detail2 ; or 3. ReQ Detail3 ; or 4. ReQ Detail4

#   Here the Requirments are seperated  by and conjuction 

#   The output set for this  requirement is presented as below 

#    1. ReQ Detail1; 

#    2. ReQ Detail2;

#    3. ReQ Detail3;

#    4. ReQ Detail4

#

#    

#Set the Requirement ID to 1

    REQ_SET_ID = 1

#Empth List for data processing

    ReqListNew =[]

    ReqListFinal =[]





#Splitting the  requirement set pattern  on Conjunction 'OR' to get distinct requirement set

    ReqList = re.split('^\d{1,2}[\. ]{1}|[\W|\s]{1}or\s*\n\d{1,2}\W|[\W|\s]{1}OR\s*\n\d{1,2}\W',REQ_Text)



#Clean the list of requriment after spilt on basis of Conjunction 'OR'

#     to Omit Empty elements and to attach Requriment Set ID to each requirement set

    for x in ReqList:

        if x != '':

            ReqListNew = ReqListNew + [str(REQ_SET_ID) + '. ' + x]

            REQ_SET_ID = REQ_SET_ID + 1

#Follwing with Process the requirement sets obtained from previous steps 

    for y in ReqListNew:

#       Cleaning the requirement set data

        rowstrip  = y.strip().strip('\W').strip('\s').strip('(')



#       Check if both 'AND' and 'OR' conjuction 

#       This will primaraily cover the requriment where Main Requirement are seperated by 'AND' conjunction and the sub requirement have 'OR' conjuction

#         To state few example; the files are:

#          ADVANCE PRACTICE PROVIDER CORRECTIONAL CARE 2325 020808 REV 111214.txt

#          EMS ADVANCED PROVIDER 2341 111618 REV 122018.txt

        if re.findall('\;[ ]{1,}or\s*\n|\;[ ]{1,}OR\s*\n',rowstrip) != [] and re.findall('\;[ ]{1,}and\s*\n\d|\;[ ]{1,}AND\s*\n\d',rowstrip) != []:

# The if part deals with requirement that has both AND and OR conjuntion

# Splitting the  requirement text  on Conjunction 'AND' for further processing

            ReqListANDSplit = re.split('\;[ ]{1,}and\s*\n|\;[ ]{1,}AND\s*\n',rowstrip)



# Initializing list and few variables

            ANDPartText = []

            ORPartText =[]

            ORPartSrch =[]

#Looping over the set after spliting the requirement by AND conjunction            

            for x in ReqListANDSplit:

# Since the requirement seperated by AND conjuction are part of one set, renaming the Requirment identification number to letter for ease of proceessing in later stage                

                if ReqListFinal == []:

                    x = re.sub('^\d\.','a.',x)

# The below statement search from the main requirement statement in the and requirement before the sub requirement starts

#Example :

#   If a Job Bulletin has requirement as:

#    2. ReQ Detail2  

#      a. ReQ Detail2; or 

#      b. ReQ Detail2;     

#   The output set for this  statement will be as below 

#    2. ReQ Detail2 

                ANDPartSrch = re.search('\W[A-Za-z]{1}\.',x)

                if ANDPartSrch is not None:

                    ANDPartText1 =  x[:ANDPartSrch.start()]

#                   The below statement search from the main requirement statement in the and requirement before the sub requirement starts

#                   Considering the above example the out will be :

#                       [a. ReQ Detail2,b. ReQ Detail2]    

                    ORPartSrch = re.split('\;[ ]{1,}or\s*\n|\;[ ]{1,}OR\s*\n',x[ANDPartSrch.end()-2:])

#                   Appending the extracted AND part to each OR part. The output will look like:

#                       [2. ReQ Detail2 a. ReQ Detail2,2. ReQ Detail2 b. ReQ Detail2]    

                    ORPartSrch = [ ANDPartText1 + ' ' + x for x in re.split('\;[ ]{1,}or\s*\n|\;[ ]{1,}OR\s*\n',x[ANDPartSrch.end()-2:])]

                else:

                    ANDPartText = ANDPartText + [x]

                    

# Creating a list of list for all the OR clause in the requirement. For Example1 mentioned above, it will look like

#   [[2. ReQ Detail2 a. ReQ Detail2,2. ReQ Detail2 b. ReQ Detail2],[3. ReQ Detail 3 a. ReQ Detail3,3. ReQ Detail3 b. ReQ Detail3]]    

                if ORPartSrch != []:

                    ORPartText.append(ORPartSrch)



                ORPartSrch =[]



# create a single list for list of list created above

#   [2. ReQ Detail2 a. ReQ Detail2,2. ReQ Detail2 b. ReQ Detail2,3. ReQ Detail 3 a. ReQ Detail3,3. ReQ Detail3 b. ReQ Detail3]]    

            ORPartFlat = list(itertools.chain(*ORPartText))

# Get all the Combination of requirement for the above list:

#  For Example 1 above the len(ORPartText) = 2

#  Combinations will look like:

#             [2. ReQ Detail2 a. ReQ Detail2,2. ReQ Detail2 b. ReQ Detail2]

#             [2. ReQ Detail2 a. ReQ Detail2,3. ReQ Detail 3 a. ReQ Detail3]

#             :

#             :

            ORComb = list(itertools.combinations(ORPartFlat,len(ORPartText)))

            FinalList = ['~'.join(x) for x in ORPartText]

            CombList = ['~'.join(x) for x in ORComb]

# Remove the irrevant combinations

#  For Example 1 above the combination

#             [2. ReQ Detail2 a. ReQ Detail2,2. ReQ Detail2 b. ReQ Detail2]

#  is part of 

#    2. ReQ Detail2  

#      a. ReQ Detail2; or 

#      b. ReQ Detail2;   

#  which is seperated by Or conjuctions and hence omitting it

#  the below combination is valid

#           [2. ReQ Detail2 a. ReQ Detail2,3. ReQ Detail 3 a. ReQ Detail3]

#  as it represents :

#     2. ReQ Detail2       ; and 3. ReQ Detail3   

#       a. ReQ Detail2;          a. ReQ Detail3; 

            CombiFinal2 = [re.split('\~',x) for x in CombList if re.search('|'.join([re.sub('\W','',x) for x in FinalList]),re.sub('\W','',x)) is None]

#  For Example 1 above 

#    1. ReQ Detail1; and 2. ReQ Detail2       ; and 3. ReQ Detail3       ; and 4. ReQ Detail4

#                        a. ReQ Detail2; or         a. ReQ Detail3; or 

#                        b. ReQ Detail2;            b. ReQ Detail3;  

# the below logic with add the below details to all the valid combinations

#    1. ReQ Detail1; and 4. ReQ Detail4

            for x in CombiFinal2:

                ReqListFinal = ReqListFinal + [';and '.join(ANDPartText) + ';and '+ ';and '.join(x)]

        else:

# This part deals with  requirement whic has only OR conjuction or AND conjuction in requirement details

# Splitting the  requirement text  on Conjunction 'OR' for further processing

            ReqListOrSplit = re.split('\;[ ]{1,}or\s*\n|\;[ ]{1,}OR\s*\n',rowstrip)

            ReqListFinal= ReqListFinal + [z for z in ReqListOrSplit]

#            print(ReqListFinal)

    return ReqListFinal



#END :Function for generating list of requirement sets and subset in a JobBulletins

#####################################################################################



#Function for generating REQUIREMENT_SUBSET_ID - A-Z represents 1-26. In the case of 27th, 28th, ... sub-requirement, use AA, BB, ...

def next_alpha_sequence(alp):

    lpart = alp.rstrip('Z')

    if not lpart:  # s contains only 'Z'

        alp = 'A' * (len(alp) + 1)

    else:

        num_replacements = len(alp) - len(lpart)

        alp = lpart[:-1] + (chr(ord(lpart[-1]) + 1) if lpart[-1] != 'Z' else 'A')

        alp += 'A' * num_replacements

    return alp



#The below function converts number names to number for numbers at thousands positions

# It is called in conjunction with names_to_number(which is the main function)

def thousand_conv(in_num):

    in_num = in_num.group()

    findthousandand = re.findall('[\d]{1}\~1000\~[\d]{1,}',in_num)

    findthousand = re.findall('[\d]{1}\~1000[ ]+',in_num)

    if findthousandand !=[]:

        numberlist = re.split('~',in_num)

        newpattern = int(numberlist[0])*int(numberlist[1]) + int(numberlist[2])

    elif findthousand !=[]:

        numberlist = re.split('~',in_num)

        newpattern = int(numberlist[0])*int(numberlist[1]) 

    else:

        newpattern =''

    in_num = str(newpattern)

    return in_num



#The below function converts number names to number for numbers at hundred positions

# It is called in conjunction with names_to_number(which is the main function)

def hundred_conv(in_num):

    in_num = in_num.group()

    findhundredand = re.findall('[\d]{1}\~?100\~?[\d]{1,}',in_num)

    findhundred = re.findall('[\d]{1}\~?100[ ]+',in_num)

    if findhundredand !=[]:

        numberlist = re.split('~',in_num)

        newpattern = int(numberlist[0])*int(numberlist[1]) + int(numberlist[2])

    elif findhundred !=[]:

        numberlist = re.split('~',in_num)

        newpattern = int(numberlist[0])*int(numberlist[1]) 

    else:

        newpattern =''

    in_num = str(newpattern)

    return in_num



#The below function converts number names to number for numbers at tens positions

# It is called in conjunction with names_to_number(which is the main function)

def tens_conv(in_num):

    in_num = in_num.group()

    findoneand = re.findall('\~[\d]{1}0\~[\- ]*[\d]{1}',in_num)

    findone = re.findall('\~[\d]{1}0\~',in_num)

    if findoneand !=[]:

        numberlist = re.split('~',re.sub('\-','',in_num))

        newpattern = int(numberlist[1]) + int(numberlist[2])

    elif findone !=[]:

        numberlist = re.split('~',in_num)

        newpattern = int(numberlist[1]) 

    else:

        newpattern =''

    in_num = str(newpattern)

    return in_num



#The below function converts number names to number in a given text

def names_to_number(para):

    npara = re.sub('[\W]*THOUSAND','~1000~',re.sub('[\W]*THOUSAND[\W]*AND[\W]*','~1000~',para.upper()))

    npara = re.sub('[\W]*HUNDRED','~100',re.sub('[\W]*HUNDRED[\W]*AND[\W]*','~100~',npara.upper()))

    npara = npara.replace('THIRTY','~30~').replace('FOURTY','~40~').replace('FORTY','~40~').replace('FIFTY','~50~').replace('SIXTY','~60~').replace('SEVENTY','~70~').replace('EIGHTY','~80~').replace('NINETY','~90~')

    npara = npara.replace('ELEVEN','11').replace('TWELVE','12').replace('THRITEEN','13').replace('FOURTEEN','14').replace('FIFTEEN','15').replace('SIXTEEN','16').replace('SEVENTEEN','17').replace('EIGHTEEN','18').replace('NINETEEN','19').replace('TWENTY','~20~')

    npara = npara.replace('ONE','1').replace('TWO','2').replace('THREE','3').replace('FOUR','4').replace('FIVE','5').replace('SIX','6').replace('SEVEN','7').replace('EIGHT','8').replace('NINE','9').replace('TEN','10')

    npara = re.sub('\~[\d]{1}0\~[\- ]*[\d]{1}|\~[\d]{1}0\~',tens_conv,npara)

    npara = re.sub('[\d]{1}\~100\~[\d]{1,}|[\d]{1}\~100[ ]+',hundred_conv,npara)

    npara = re.sub('\~[\W ]*','~',npara)

    npara = re.sub('[\d]{1}\~1000\~[\d]{1,}|[\d]{1}\~1000[]*[\d]{1,}|[\d]{1}\~1000[ ]+',thousand_conv,npara)

    return npara



#  The below function calculates the experiennce length when Experience length comes first like in the input text  '5 years of Experience'

def years_of_experience(YearExp):

    yearsExpOP = ''

    if YearExp == []:

        yearsExpOP = ''

    elif re.search('HALF',YearExp[0]) is not None:

        HalfEndSrch = re.search('\DAND',YearExp[0].strip())

        HalfEndNoANDSrch = re.search('\d+',YearExp[0].strip())

        if HalfEndSrch is not None:

            yearsExpOP = int(YearExp[0].strip()[:HalfEndSrch.start()]) + 0.5

        elif HalfEndNoANDSrch  is not None:

            yearsExpOP = int(YearExp[0].strip()[:HalfEndNoANDSrch.end()]) * 0.5

        else:

            yearsExpOP = 0.5

            

    elif re.search('YEAR',YearExp[0]) is not None:

        YearEndSrch = re.search('\D',YearExp[0].strip())

        if YearEndSrch is not None:

            yearsExpOP = YearExp[0].strip()[:YearEndSrch.start()]

    elif re.search('MONTH',YearExp[0]) is not None:

        MonEndSrch = re.search('\D',YearExp[0].strip())

        if MonEndSrch is not None:

            yearsExpOP = round(int(YearExp[0].strip()[:MonEndSrch.start()])/12 ,2)

    elif re.search('HOURS',YearExp[0]) is not None:

        HourEndSrch = re.search('\D',re.sub('\,','',YearExp[0]).strip())

        if HourEndSrch is not None:

            yearsExpOP = round(int(re.sub('\,','',YearExp[0]).strip()[:HourEndSrch.start()])/(365*24),2)

    return yearsExpOP



#  The below function calculates the experiennce length when Experience length comes first like in the input text  '5 years of Experience'

def years_of_experiencelater(YearExp):

    yearsExpOP = ''

    if YearExp == []:

        yearsExpOP = ''

    elif re.search('HALF',YearExp[0]) is not None:

        HalfEndSrch = re.search('\d+ AND',YearExp[0].strip())

        HalfEndNoANDSrch = re.search('\d+',YearExp[0].strip())

        if HalfEndSrch is not None:

            yearsExpOP = int(YearExp[0].strip()[HalfEndSrch.start():HalfEndSrch.end()-3]) + 0.5

        elif HalfEndNoANDSrch is not None:

            yearsExpOP = int(YearExp[0].strip()[HalfEndNoANDSrch.start():HalfEndNoANDSrch.end()]) * 0.5

        else:

            yearsExpOP =  0.5

    elif re.search('YEAR',YearExp[0]) is not None:

        YearSrch = re.search('\d+',YearExp[0].strip())

        if YearSrch is not None:

            yearsExpOP = YearExp[0].strip()[YearSrch.start():YearSrch.end()]

    elif re.search('MONTH',YearExp[0]) is not None:

        MonEndSrch = re.search('\d+',YearExp[0].strip())

        if MonEndSrch is not None:

            yearsExpOP = round(int(YearExp[0].strip()[MonEndSrch.start():MonEndSrch.end()])/12 ,2)

    elif re.search('HOURS',YearExp[0]) is not None:

        HourEndSrch = re.search('\d+',re.sub('\,','',YearExp[0]).strip())

        if HourEndSrch is not None:

            yearsExpOP = round(int(re.sub('\,','',YearExp[0]).strip()[HourEndSrch.start():HourEndSrch.end()])/(365*24),2)

    return yearsExpOP



##PROCESSING the available Job Tiles provided in the Kaggle Competition Dataset 



JobTitleFoldername = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Additional data/'

JobTitleFilename = 'job_titles.csv'

JobTitleentirepath= JobTitleFoldername + JobTitleFilename



#Setting an empty JobTile list

JobTitle = []



#Read the Jobtitles into a dataframe and convert it to list

JobTitleDF = pd.read_csv(JobTitleentirepath, header=None)

JobTitle = JobTitleDF[0].values.tolist()



#Sort and reverse the Jobtile list based on the length of the element

#  so that SENIOR CARPENTER is in the top of list be for CARPENTER 

#  this is will help in sorting the list based on seniority as currently we dont have any such alignment data available  

JobTitle.sort(key=len)

JobTitle.reverse()



#Converting the JobTile list to a pattern so that the element are seperated by or '|

#  This pattern will be used further in regex search function

JobtilePattern = "'abc|\W"+'\W|\W'.join(JobTitle) + "\W|xyz'"



#Get the list of files under Job Bulletins

foldername='../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/'

filelist = [f for f in listdir(foldername) if isfile(join(foldername, f))]



#Initialize a list which will File error 

FilesWithMissingInfo = []



#Initialize a list which will contain the data collected from all the csv

# It will contain list of lists that will contain the data from csv

frmtAll = []



#Initialize the header of the output dataset 

fileheader =["FILE_NAME","JOB_CLASS_TITLE","JOB_CLASS_NO","REQUIREMENT_SET_ID","REQUIREMENT_SUBSET_ID","JOB_DUTIES","EDUCATION_YEARS","SCHOOL_TYPE","EDUCATION_MAJOR","EXPERIENCE_LENGTH","FULL_TIME_PART_TIME","EXP_JOB_CLASS_TITLE","EXP_JOB_CLASS_ALT_RESP","EXP_JOB_CLASS_FUNCTION","COURSE_COUNT","COURSE_LENGTH","COURSE_SUBJECT","MISC_COURSE_DETAILS","DRIVERS_LICENSE_REQ","DRIV_LIC_TYPE","ADDTL_LIC","EXAM_TYPE","ENTRY_SALARY_GEN","ENTRY_SALARY_DWP","OPEN_DATE"]



#Process the files to create a token list

for filename in filelist:



# Intialize the filepath by appending filename to the folder path where the Job Bulletins are available 

    entirepath= foldername + filename



# Read the content of plaintext file in filetext variable    

    textfile = open(entirepath,'r',encoding='latin-1')

    filetext = textfile.read()

    textfile.close()



#Initialize a list which will contain the data collected from the csv being read

    frmtOut = ['','','','','','','','','','','','','','','','','','','','','','','','','']



#BEGIN: FILENAME INFORMATION 

    frmtOut[0]=filename

#END: FILENAME INFORMATION 



#BEGIN: JOB CLASS TITLE INFORMATION 

#EXTRACTING Job Class Title from the files

# Few files have 'CAMPUS INTERVIEWS ONLY' tag in the beginning. Omitting that

# Few files have acrynom in Job tiles enclosed () . Omitting that

    Refinefiletext = re.sub('CAMPUS INTERVIEWS ONLY','',filetext.strip()).strip()

    JobClassExtract = Refinefiletext[:re.search('\w*\n',Refinefiletext.strip()).end()-1]

    JobClassTransform = re.sub('[ ]*\(.*\)[ ]*',' ',re.split('\s{2,}|\t',JobClassExtract)[0]).strip()



    frmtOut[1] = JobClassTransform

#END: JOB CLASS TITLE INFORMATION 

    

#BEGIN: CLASS CODE INFORMATION 

    ClassCodeSrch = re.search('Class Code((\W+)|(\w+))[0-9]{4}',filetext)

    if ClassCodeSrch is not None :

        ClassCode = filetext[ClassCodeSrch.end()-4:ClassCodeSrch.end()]

        if len(ClassCode) == 3:

            frmtOut[2] = '0' + ClassCode

        else:

            frmtOut[2] =  str(ClassCode)

    else:

        FilesWithMissingInfo.append(["Missing ClassCode",filename])

        frmtOut[2] = 'N/A'

#END: CLASS CODE INFORMATION 

 



#BEGIN: DUTIES INFORMATION 

    DutiesSrch_StartPos = re.search('DUTIES(\W+)(.)',filetext)

    DutiesSrch_EndPos = re.search('DUTIES(\W+)(.*\n)',filetext)

    if DutiesSrch_StartPos is not None and DutiesSrch_EndPos is not None:

        frmtOut[5] =filetext[DutiesSrch_StartPos.end()-1:DutiesSrch_EndPos.end()-1]

    else:

        FilesWithMissingInfo.append(["Missing Duties Information",filename])

        frmtOut[5] = 'N/A'

#END: DUTIES INFORMATION 



#BEGIN: OPEN DATE INFORMATION 

# Search for Open Date text 

    OpenDateSrch = re.search('OPEN DATE(\W+)(\d{1,2}\-\d{1,2}\-\d{4})|OPEN DATE(\W+)(\d{1,2}\-\d{1,2}\-\d{2})',filetext.upper())

# Few Open Date have been revised 

    RevisedSrch = re.search('REVISED(\W+)(\d{1,2}\-\d{1,2}\-\d{4})|REVISED(\W+)(\d{1,2}\-\d{1,2}\-\d{2})',filetext.upper())

    if OpenDateSrch is not None :

        if RevisedSrch is not None :

            frmtOut[24] = filetext[RevisedSrch.start()+8:RevisedSrch.end()].strip()

        else:    

            frmtOut[24] = filetext[OpenDateSrch.start()+10:OpenDateSrch.end()].strip()

    else:

        FilesWithMissingInfo.append(["Missing Open Date",filename])

        frmtOut[24] = 'N/A'

#END: OPEN DATE INFORMATION 



#BEGIN: EXAM TYPE INFORMATION 

    EXAM_TYPE = re.findall('INTER[A-Z]{2,}\sPROMO[A-Z]{2,}|OPEN\sCOMPET[A-Z]{2,}|DEPART[A-Z]{2,}\sPROMO[A-Z]{2,}|open to all',filetext)

    if len(EXAM_TYPE)==1 : 

        if EXAM_TYPE[0][0].upper() == 'O':

            frmtOut[21] =  "OPEN"

        if EXAM_TYPE[0][0].upper() == 'I':

            frmtOut[21] =  "INT_DEPT_PROM"

        if EXAM_TYPE[0][0].upper() == 'D':

            frmtOut[21] =  "DEPT_PROM"



    if len(EXAM_TYPE)==2 : 

        if re.sub("OI","IO",EXAM_TYPE[0][0].upper() + EXAM_TYPE[1][0].upper()) == 'IO':

            frmtOut[21] =  "OPEN_INT_PROM"  

        else:

            frmtOut[21] =  "N/A" 

#END: EXAM TYPE INFORMATION 

 

#BEGIN: SALARY INFORMATION 

    FromsalaryData = re.split('ANNUAL\s?SALARY',filetext)

    if len(FromsalaryData) > 1:

        SalaryData = re.split('[A-Z]{3,}',FromsalaryData[1])

        if len(SalaryData) > 1:

            SalaryDataStrip = SalaryData[0].strip()

            SalaryDataClean = re.sub('\n|\s+|\,|\*','',SalaryDataStrip)



            DWPSplit = re.split('DepartmentofWaterandPower',SalaryDataClean)

            LosAnglesSalary = DWPSplit[0]



            if len(DWPSplit) > 1:

               DWPSalary = DWPSplit[1]

            else:

               DWPSalary = 'N/A'



            LosAnglesSalaryClean = re.findall('\$\d{3,}\-\$\d{3,}|\$\d{3,}[a-z\W]*',re.sub('to','-',LosAnglesSalary))              

            if len(LosAnglesSalaryClean) > 0:    

                if re.search('\$\d{3,}\-\$\d{3,}',LosAnglesSalaryClean[0]) : 

                    frmtOut[22] = re.sub('\$','',LosAnglesSalaryClean[0])

                else:

                    frmtOut[22] = re.sub('[a-z\W\,\s]','',LosAnglesSalaryClean[0])

            else:

                frmtOut[22] = 'N/A'

                

            DWPSalaryClean = re.findall('\$\d{3,}\-\$\d{3,}|\$\d{3,}[a-z\W]*',re.sub('to','-',DWPSalary))              

            if len(DWPSalaryClean) > 0:    

                if re.search('\$\d{3,}\-\$\d{3,}',DWPSalaryClean[0]) : 

                    frmtOut[23] = re.sub('\$','',DWPSalaryClean[0])

                else:

                    frmtOut[23] = re.sub('[a-z\W\,\s]','',DWPSalaryClean[0])

            else:

                frmtOut[23] = 'N/A'

#END: SALARY INFORMATION 



#BEGIN: DRIVING LICENSE INFORMATION

    LicBasicLicSrch  = re.findall('CLASS [A-Z]{1} .*LICENSE|DRIVER.*LICENSE',filetext.upper())



    LicRQRDSrch = re.findall('DRIVER[S\']{3} LICENSE IS REQUIRED|POSSESSION OF A VALID CALIFORNIA DRIVER\'S LICENSE|DRIVER\'S LICENSE[A-W ]*REQUIRED|DRIVER\'S LICENSE[A-W ]*BY[A-W ]*REQUIRED|SOME POSITIONS REQUIRE A VALID CALIFORNIA DRIVER\'S LICENSE|THESE POSITIONS ARE OPEN TO ALL WHO APPLY WITH A VALID CALIFORNIA DRIVER\'S LICENSE',filetext.upper())



    LicPOSSSrch = re.findall('MAY ALSO REQUIRE A VALID CALIFORNIA DRIVER\'S LICENSE|POSITIONS MAY REQUIRE.*DRIVER\'S LICENSE|DRIVER\'S LICENSE MAY BE REQUIRED',filetext.upper())



    LicSpecificSrch  = re.findall('CALIFORNIA CLASS [A-Z]{*} LICENSE IS REQUIRED|CLASS A OR B DRIVER\'S LICENSE|CLASS A DRIVER\'S LICENSE|CLASS B DRIVER\'S LICENSE|CLASS A OR CLASS B DRIVER\'S LICENSE|CALIFORNIA B DRIVER\'S LICENSE|CALIFORNIA A DRIVER\'S LICENSE',filetext.upper())



    if LicBasicLicSrch == []:

        frmtOut[18] = 'N'

    elif LicRQRDSrch != []:

        frmtOut[18] = 'R'

    elif LicPOSSSrch !=[]:

        frmtOut[18] = 'P'

    elif LicSpecificSrch!=[]:

        frmtOut[18] = 'R'

    else:

        frmtOut[18] = 'P'



    AllSrchData = list(set(LicBasicLicSrch +LicRQRDSrch+LicPOSSSrch+LicSpecificSrch))

    AllSrchDatajoin= '|'.join(AllSrchData)

    All = re.sub('\"|\(\)|\'','',AllSrchDatajoin)



    CleanClass  = list(set([re.sub('CLASS CLASS| CLASS\/','CLASS',re.sub('OR','OR CLASS',re.sub('CALIFORNIA ','CLASS ',x.strip()))) for x in list(set(re.findall('CLASS [A-Z]{1} OR [A-Z]{1} |CLASS [A-Z]{1} OR CLASS [A-Z]{1} |CLASS [A-Z]{1} AND [A-Z]{1} |CLASS [A-Z]{1} AND CLASS [A-Z]{1} |CLASS [A-Z]{1} AND\/OR CLASS [A-Z]{1} |CLASS [A-Z]{1} OR\/AND CLASS [A-Z]{1} |CALIFORNIA [A-Z]{1} |CLASS [A-Z]{1} ',All)))]))

    RemoveOR  = list(set(re.split(' OR ',re.sub('OR\/|\/OR','~',' OR '.join(CleanClass)))))

    ExtractANDClass = [x for x in RemoveOR if re.search('AND',x) is not None ]

    ExtractSinClass = [x for x in RemoveOR if re.search('AND',x) is None ]

    

    LicCategory = []



    if ExtractANDClass != []:

        for x in ExtractANDClass:

            for y in ExtractSinClass:

                if re.search(y,x) is None:

                    LicCategory = LicCategory + [y]

        LicCategory = LicCategory + [x]

    else:

        LicCategory = ExtractSinClass



    frmtOut[19] =re.sub('AND~|~AND','AND/OR','|'.join(LicCategory))

#END: DRIVING LICENSE INFORMATION



###################################################################################

###################################################################################

###################################################################################

###################################################################################

#BEGIN: REQUIREMENT DETAILS 

# The following elements defined in the Data Dictonary are being retrived from the REQUIREMENT/MINIMUM QUALIFICATION section for the Job Bulletins     

#   EDUCATION_YEARS

#   SCHOOL_TYPE

#   EDUCATION_MAJOR

#   EXPERIENCE_LENGTH

#   FULL_TIME_PART_TIME

#   EXP_JOB_CLASS_TITLE

#   EXP_JOB_CLASS_ALT_RESP

#   EXP_JOB_CLASS_FUNCTION

#   COURSE_COUNT

#   COURSE_LENGTH

#   COURSE_SUBJECT

#   MISC_COURSE_DETAILS

#   ADDTL_LIC

    

#Intializing the Update Flag

    updateflag = 0

#  Extract the REQUIREMENT/MINIMUM QUALIFICATION section from the Job Bulletin

    REQsrch=re.search('REQUI[A-Z]{4,}\W*MI[A-Z]{4,}\W*REQ[A-Z]{4,}\W*|REQUI[A-Z]{4,}\W*MI[A-Z]{4,}\W*QUA[A-Z]{4,}\W*|REQUI[A-Z]{4,}\W*MI[A-Z]{4,}\W*|REQUI[A-Z]{4,}\W*',filetext)

    if REQsrch is not None:

        # The below variable contains text from REQUIREMENT/MINIMUM QUALIFICATION section onwards till the end of file

        Newfiletext = filetext[REQsrch.end():]



        # Finding the end of the REQUIREMENT/MINIMUM QUALIFICATION section

        REQsrchEND=re.search('\n\s*[A-Z]{4,}',Newfiletext)

        if REQsrchEND is not None:

            #Extracting only the REQUIREMENT/MINIMUM QUALIFICATION section

            REQ_Text = Newfiletext[:REQsrchEND.start()].strip()

            

            #Calling the function to get list of distinct REQUIREMENT SET based on the AND/OR conjunction

            REQDetails  = RequirementSetDetails(REQ_Text)



            # Intializing Variable

            REQCounter = 0

            updateflag = 0

            prevID = 0



            # Intializing REQUIREMENT SET ID and REQUIREMENT SET SUB_ID

            REQ_SET_ID = 1

            REQ_SET_SUB_ID = 'A'



            # Processing each REQUIREMENT SET ID  available in the Job Bulletin

            for r in REQDetails:

                rowstrip  = r.strip().strip('\W').strip('\s').strip('(')



#################################################################################

# BEGIN OF LOGIC to set REQUIREMENT SET ID and REQUIREMENT SET SUB_ID for the Job Bulletin



                REQCounter = REQCounter + 1

                if REQCounter == 1:

                    REQ_SET_SUB_ID = 'A'

                    # Initializing the variables for the first Requirement Set for a JoBBulletin else it will contain stale data

                    EduType =''

                    EduYear =''

                    EduMajor =''

                    AdditionalLicType =''

                    SemQuartUnitsValue = ''

                    CourseCount = ''

                    SemQuartSub = ''

                    SemQuartSubRaw = ''

                    frmtOut[6] =''

                    frmtOut[7] =''

                    frmtOut[8] =''



                    frmtOut[9] =''

                    frmtOut[10] =''

                    frmtOut[11] =''

                    frmtOut[12] =''

                    frmtOut[13] =''



                # If the requirement start with digit it is the begining of a REQ SET data                    

                if re.match('\d',rowstrip[:1]):

                    if (re.match('\d\W',rowstrip[:2]) or re.match('\d ',rowstrip[:2])) and rowstrip[:1] != prevID :

                        frmtOut[3] = REQ_SET_ID

                        REQ_SET_ID = REQ_SET_ID + 1

                        REQ_SET_SUB_ID = 'A'



                        # Initializing the variables for the each Requirement Set for a JoBBulletin else data of one Requirement set will move in the later REquuirement set

                        EduType =''

                        EduYear =''

                        EduMajor =''

                        AdditionalLicType =''

                        SemQuartUnitsValue = ''

                        CourseCount = ''

                        SemQuartSub = ''

                        SemQuartSubRaw = ''

                        frmtOut[6] =''

                        frmtOut[7] =''

                        frmtOut[8] =''



                        frmtOut[9] =''

                        frmtOut[10] =''

                        frmtOut[11] =''

                        frmtOut[12] =''

                        frmtOut[13] =''



#                   The below logic is to handle the situation in the below example

#                   REQuirement in Job Bulletins is a below:                         

#                       1 REQ 1  Details   or                          

#                       2 REQ 2 Details   and                          

#                       3 REQ 3 Details                             

#                       a REQ 3a Details   or                          

#                       b REQ 3b Details  

#                   RequirementSetDetails function will Split it as the following REQUIREMENT SET]

#                       1 REQ 1  Details                             

#                       2 REQ 2 Details  and 3 REQ 3 Details a REQ 3a Details              

#                       2 REQ 2 Details  and 3 REQ 3 Details b REQ 3b Details              

#                   This is a special case and hence REQ_SET_ID which was incremented in previous loop run will be set back                          

                    elif rowstrip[:1] == prevID:

                        frmtOut[3] = REQ_SET_ID - 1

                    else:

                        frmtOut[3] = REQ_SET_ID

                    prevID = rowstrip[:1]

                    frmtOut[4] = REQ_SET_SUB_ID

                    updateflag = 1                



                # If the requirement start with letter it is the begining of a REQ SUB SET data                    

                if re.match('[A-Z]{1}',rowstrip[:1].upper()) and REQCounter > 1:

                    frmtOut[4] = REQ_SET_SUB_ID

                    updateflag = 1

                else: 

                    frmtOut[4] = REQ_SET_SUB_ID



                # If below logic takes care of the situation when the requirement doesnot contain there own REQ ID And SUB ID 

                if not(re.match('1',rowstrip[:1])) and REQCounter == 1 :

                    frmtOut[3] = REQ_SET_ID

                    frmtOut[4] = REQ_SET_SUB_ID

                    REQ_SET_ID = REQ_SET_ID + 1

                    updateflag = 1



                REQ_SET_SUB_ID = next_alpha_sequence(REQ_SET_SUB_ID)



# END OF LOGIC to set REQUIREMENT SET ID and REQUIREMENT SET SUB_ID for the Job Bulletin

#################################################################################



#######################################################################################

#BEGIN Education Details



#Converting number names to number for easire processing which searching of how many years information

                rnumtoname = re.sub('[ ]*\(.*?\)[ ]*|\)|\d+\.|\W[A-Z]{1}[\.\:]',' ',names_to_number(rowstrip))

#Correcting the conversion above for words like writTEN, mainTENance,wEIGHT...

                CorrectNumRep = list(set(re.findall('[A-Z]+8+[A-Z]+|[A-Z]+8+|[A-Z]+10+[A-Z]+|[A-Z]+10+|[A-Z]+1+[A-Z]+|[A-Z]+1+',rnumtoname)))

                if CorrectNumRep !=[]:

                    for x in CorrectNumRep:

                        repla = re.sub('1','ONE',re.sub('8','EIGHT',re.sub('10','TEN',x)))

                        rnumtoname = re.sub(x,repla,rnumtoname)



#Searching for school Type Information

                SchoolTypefindall = re.findall('GRADUATE|GRADUATION|BACHELOR|PH\.D|MASTER\'S|HIGH SCHOOL|CERTIFICATE PROGRAM|CERTIFICATES|CERTIFICATE|CERTIFICATION|DEGREE|ASSOCIATE[A-Z\' ]{1,15}DEGREE',rnumtoname.upper())

                if SchoolTypefindall !=[]:

# Not considering the details for CERTIFICATES

#   CERTIFICATES is refered in other context(like allocation, permits) in the set of data studied 

                    if ('CERTIFICATES' in SchoolTypefindall) and not('GRADUATION' in SchoolTypefindall  or 'HIGH SCHOOL' not in SchoolTypefindall  or 'GRADUATE' in SchoolTypefindall  or 'BACHELOR' in SchoolTypefindall or 'PH.D' in SchoolTypefindall or 'MASTER\'S' in SchoolTypefindall or 'DEGREE' in SchoolTypefindall or 'ASSOCIATE[A-Z\' ]{1,15}DEGREE' in SchoolTypefindall):

                        EduType = ''

                    else:

# Setting up the accurate Education Type 

# After studying the data, this logic is categorizing Education Type as

#        COLLEGE OR UNIVERSITY : if the education is attained from College or University

#        HIGH SCHOOL :For High School education

#        CERTIFICATE : For any certificate

#        ASSOCIATE DEGREE : For Associate Degree

#        ACCREDITED INSTITUTE : When College or University is not refered and it is mentioned that the education should be from Accrediated Institute

#        DEGREE WITHOUT SPECIFICATION : For instance where degree is referred but it is not mentioned whether it should be from Collegeor University or Any accrediated institute 

                        

                        if ('GRADUATION' in SchoolTypefindall  and 'HIGH SCHOOL' not in SchoolTypefindall)  or 'CERTIFICATE PROGRAM' in SchoolTypefindall  or 'GRADUATE' in SchoolTypefindall  or 'BACHELOR' in SchoolTypefindall or 'PH.D' in SchoolTypefindall or 'MASTER\'S' in SchoolTypefindall or 'DEGREE' in SchoolTypefindall or re.search('ASSOCIATE[A-Z\' ]{1,15}DEGREE','|'.join(SchoolTypefindall)) is not None :

                            if re.search('COLLEGE|UNIVERSITY',rnumtoname.upper()) and re.search('ASSOCIATE[A-Z\' ]{1,15}DEGREE','|'.join(SchoolTypefindall)) is None:

                                EduType = 'COLLEGE OR UNIVERSITY'

                            elif re.search('ACCREDITED',rnumtoname.upper()) and re.search('ASSOCIATE[A-Z\' ]{1,15}DEGREE','|'.join(SchoolTypefindall)) is None:

                                EduType = 'ACCREDITED INSTITUTE'

                            elif re.search('ASSOCIATE[A-Z\' ]{1,15}DEGREE','|'.join(SchoolTypefindall)) is not None: 

                                EduType = 'ASSOCIATE DEGREE'

                            elif re.search('DEGREE',rnumtoname.upper()) and len(list(set(SchoolTypefindall))) >1:

                                EduType = 'DEGREE WITHOUT SPECIFICATION'

                            elif re.search('CERTIFICATE PROGRAM',rnumtoname.upper()) and len(list(set(SchoolTypefindall))) >1:

                                EduType = 'CERTIFICATE PROGRAM'

                            elif not('GRADUATION' in SchoolTypefindall or 'GRADUATE' in SchoolTypefindall or 'DEGREE' in SchoolTypefindall):

                                EduType = 'OTHERS'



# Setting up the accurate Education Year 

                            EducationYearSrch = re.findall('\d*[- ]*YEAR[A-Z\W ]{1,5}COLLEGE|\d*[- ]*YEAR[A-Z\W ]{1,5}UNIVERSITY|\d*[- ]*YEAR[A-Z\W ]{1,25}DEGREE|\d*[- ]*YEAR[A-Z\W ]{1,35}ACCREDITED',rnumtoname)

                            if EducationYearSrch !=[]:

                                EduYear = EducationYearSrch[0][0]

                            else:

                                EduYear = ''

# Setting up the accurate Education Major

# For Associate degree : The logic looks for the details like is Associate of Art/Science or any specifiction like Associate Degree IN

# For College or University, Accrediated Institute and Degree:

#             The logic looks for the details like is MAJOR or IN

#             The below logic omits unnecessary preposition, adjective... words but this filteration is very specfic to the Job Bulletins that were available

#                                             

                            MajorInSrch  =  re.findall('COLLEGE[A-Z\, ]*MAJOR[A-Z ]* IN [A-Z\,\- ]*|UNIVERSITY[A-Z\, ]*MAJOR[A-Z ]* IN [A-Z\,\- ]*|COLLEGE IN [A-Z\,\- ]*|UNIVERSITY IN [A-Z\,\- ]*|DEGREE IN [A-Z\,\- ]*|DEGREE PROGRAM IN [A-Z\,\- ]*|ACCREDITED[A-Z ]* IN [A-Z\,\- ]*|CERTIFICATE PROGRAM IN [A-Z\,\- ]*',re.sub('\:| [A-Z]?\.','',rnumtoname))  

                            EduMajor = ''

                            if EduType == 'ASSOCIATE DEGREE'and MajorInSrch == []:

                                MajorInSrch  =  re.findall('ASSOCIATE OF [A-Z]* DEGREE|ASSOCIATE[\'S]* DEGREE[A-Z ]* IN [A-Z\,\- ]*',rnumtoname)  

                                if MajorInSrch!= [] and re.search('ASSOCIATE OF [A-Z]* DEGREE',MajorInSrch[0])is not None:

                                    EduMajor = re.sub('ASSOCIATE OF | DEGREE','',MajorInSrch[0])

                            if MajorInSrch != []:    

                                MajorInSrchSplit = re.split(' IN ', MajorInSrch[0]) 

                                if len(MajorInSrchSplit) > 1:

                                    MajorInSrchSplitEnd = re.split('DESIGNATION| WITH |MAY|SUCCESSFUL|INCLUDING |WHICH |UPON |FROM |AT | ANY|ANY |OR OTHER |OR A |OR SPECIAL|OR [A ]*CLOSELY[\- ]?RELATED FIELD|OR [A ]*RELATED|[ ]*AND$',MajorInSrchSplit[1].strip())

                                    MajorSuchSub =re.sub('OR VARIOUS [A-Z ]*DISCIPLINES SUCH AS ','',MajorInSrchSplitEnd[0])

                                    if re.search('SUCH AS',MajorSuchSub) is None:

                                        EduMajor = '|'.join([re.sub('^[A-Z]{1} | AND','',x).strip() for x in re.split('\,|OR | OR',MajorSuchSub) if x.strip() != '' ])

                                    else:

                                        SuchSplit = re.split('SUCH AS',MajorInSrchSplitEnd[0])

                                        EduMajor = '|'.join([re.sub('^[A-Z]{1} | AND','',x).strip() for x in re.split('\,|OR | OR',SuchSplit[1]) if x.strip() != '' ])



                        elif 'CERTIFICATE' in SchoolTypefindall or 'CERTIFICATION' in SchoolTypefindall:

                            EduType = 'CERTIFICATE'

                        elif 'HIGH SCHOOL' in SchoolTypefindall:

                            EduType = 'HIGH SCHOOL'

                        frmtOut[7] = EduType

                        frmtOut[6] = EduYear

                        frmtOut[8] = EduMajor

#END Education Details

#######################################################################################

#BEGIN : Additional License Information Processing

#               Look for the presence of License information in the REQuirement text                          

                LicenseTypeFindAll = re.findall('LICENSE',rnumtoname.upper())

#               Initialize variables                          

                AddLicTypeList=[]

#               Clean up the extracted data to get the additional licence details

#               Fetch  License details

                LicTypeDetailSrch  =  re.findall('LICENSE[D]? AS .*|.*LICENSED BY.*|A VALID.*LICENSE.*|AS A LICENSED.*|AS A.*LICENSED.*|LICENSE.*AS A.*|.*A STATE OF CALIFORNIA PARAMEDIC LICENSE.*|POSSESSION OF.*LICENSE?.*',rnumtoname.upper())

                if LicTypeDetailSrch !=[]:

#                  Process the data if it has license details other that Driver License

                    LicTypeDetailSrchFilter  =  [x for x in LicTypeDetailSrch if re.search('CLASS [A-Z]{1} .*LICENSE|DRIVER.*LICENSE',x) is None]

                    if LicTypeDetailSrchFilter !=[]:

                        LicTypeDetailSplit  =   re.split('\n|\.|\;',LicTypeDetailSrchFilter[0])

#                       Remove unecessary text for the extracted details

                        for x in LicTypeDetailSplit:

                            if re.search('LICENSE',x) is not None:

                                AddLicTypeList  = AddLicTypeList + [x]

                        AdditionalLicType = re.sub('^AS[\W]*|^AND[\W]*|[\W]*AND$|\s{2,}',' ',AddLicTypeList[0].strip())

                frmtOut[20]= AdditionalLicType    



#END : Additional License Information Processing                        



################################################################

# BEGIN : Course Details

#Looking for the Course  related details in the requirement section               

                CourseFindAll = re.findall('COURSE|SEMESTER|QUARTER',rnumtoname.upper())

#Extract for the Semester and Quarter unit count

                SemQuartUnits = [re.sub('\W|\s','',re.sub('QUARTER','Q',re.sub('SEMESTER','S',x))) for x in re.findall('\W*\d+ SEMESTER|\W*\d+ QUARTER',rnumtoname.upper())]

                if len(SemQuartUnits) == 1:

                    SemQuartUnitsValue = SemQuartUnits[0]

                elif len(SemQuartUnits) >=2:

                    SemQuartUnitsValue = SemQuartUnits[0] + '|' + SemQuartUnits[1]

#Extract for the number of courses required

                if re.findall('\d+ COURSE',rnumtoname) !=[]:

                    CourseCount = re.findall('\d+',re.findall('\d+ COURSE',rnumtoname)[0])[0]

                

                frmtOut[14] = CourseCount

                frmtOut[15] = SemQuartUnitsValue



# Extracting the Course subjects

#             The below logic omits unnecessary preposition, adjective... words but this filteration is very specfic to the Job Bulletins that were available

#             It also omiting the details after are refered in some context like SUCH AS or RELATED                                

#             Anything apart from this will come as a dirty data                

                SemQurtDataExtract = re.findall('SEMESTER.* IN [A-Z\,\-\: ]*|QUARTER.* IN [A-Z\,\-\: ]*|SEMESTER.* OF [A-Z\,\-\: ]*|QUARTER.* OF [A-Z\,\-\: ]*|SEMESTER.* ON [A-Z\,\-\: ]*|QUARTER.* ON [A-Z\,\-\: ]*',rnumtoname)    

                SemQurtSrchSplitEnd = []

                if len(SemQurtDataExtract)>=1:

                    SemQurtDataSplit = re.split(' IN ',SemQurtDataExtract[0])

                    if (len(SemQurtDataSplit) == 1):

                        SemQurtDataSplit = re.split(' UNITS OF ',SemQurtDataExtract[0])

                    if (len(SemQurtDataSplit) == 1):

                        SemQurtDataSplit = re.split(' ON ',SemQurtDataExtract[0])

                    if len(SemQurtDataSplit) > 1:

                        SemQurtSrchSplitEnd = re.split(' WITH | FIELDS|\d|COMPLETION|REQUIREMENT|AN ACCREDITED|MAY|INCLUDING |WHICH |UPON |FROM |AT |OR OTHER |OR A | RELATED|SUCH AS |OF AREAS',SemQurtDataSplit[1].strip())

                        SemQuartSubRaw = '|'.join([re.sub('^[A-Z]{1} |[\W]*THE [\W]*|^THE$|COURSEWORK|COURSE WORK|EACH |OF THE FOLLOWING AREAS|COMBINATION OF|EITHER|FOLLOWING|[\W]+AND[ ]*$|COURSES| ANY|ANY |','',x).strip() for x in re.split('\,|OR | OR|\:',SemQurtSrchSplitEnd[0]) if x.strip() != '' ])

                        SemQuartSub = re.sub('^\||\|$','',SemQuartSubRaw)

                frmtOut[16] =SemQuartSub

 

# Extracting the Misc Course details:                   

                CourseDataExtract = [re.sub('^[A-Za-z]{1}[\. ]{1}','',x).strip() for x in re.findall('.*course.*|.*Course.*',re.sub('[ ]*\(.*?\)[ ]*|\)|\d+\.','',rowstrip)) if re.search('Course Waiver|course waiver',x) is None]   

                SemnQuartExtract = [re.sub('^[A-Za-z]{1}[\. ]{1}','',x).strip() for x in re.findall('.*semester.*|.*Semester.*|.*quarter.*|.*Quarter.*',re.sub('[ ]*\(.*?\)[ ]*|\)|\d+\.','',rowstrip)) if re.search('Course Waiver|course waiver',x) is None]   



# REfining  the Misc Course details.

#               Merging the data extracted above for Course and Semeter            

#               Omitting the details that are already in the other course related fields:                   

                DetailFilter = [x for x in list(set(SemnQuartExtract + CourseDataExtract)) if re.search('[Ss]{1}emester.* in [A-Za-z\,\-\: ]*|[Qq]{1}uarter.* in [A-Za-z\,\-\: ]*|[Ss]{1}emester.* on [A-Za-z\,\-\: ]*|[Qq]{1}uarter.* on [A-Za-z\,\-\: ]*|[Ss]{1}emester.* on [A-Za-z\,\-\: ]*|[Qq]{1}uarter.* on [A-Za-z\,\-\: ]*',x) is None]

                frmtOut[17] = ';and '.join(DetailFilter)

# END : Course Details

################################################################





#######################################################################################

#BEGIN Experience Details

# The below  aims to categorise data in the following Experience Category

#    a) Full Time

#    b) Part Time

#    c) Paid Or Vocational Experience 

#    d) Apprenticship  Experience 

#    e) Hours Of : When the experience is mentioned in hours not in Full Time, Part Time ...                

#    f) Others : When the experience details doesnt falls in any of the mentioned category

                

                #Searching for Experience Type Information

                ExpTypeFindAll = re.findall('FULL-TIME|FULL TIME|PART-TIME|PART TIME|PAID OR VOLUNTEER|VOLUNTEER OR PAID|HOURS OF[A-Z ]*EXPERIENCE|EXPERIENCE|CURRENT EMPLOY|YEARS AS A|APPRENTICESHIP',rnumtoname)

                if ExpTypeFindAll !=[]:

#                   Processing the Full Time Experience Data

                    if 'FULL TIME' in ExpTypeFindAll or 'FULL-TIME' in ExpTypeFindAll:

                        frmtOut[10] = 'FULL-TIME'

                        # Check if any Experience Length is mentioned for Full Time experience

                        FullTimeLen = re.findall('\d+.*FULL[- ]{1}TIME.*',rnumtoname)

                        # Only first occurance of Full Time will be consider to evaluate experience length

                        if FullTimeLen != []:

                            #In the event when multiple Full Time experience length is mentioned, the logic considers the first occurance of of the experience lenght

                            FullTimeLenTxt = re.findall('\d+[A-Z\-\,\:\#\;\d ]{1,10}HALF[A-Z\-\,\:\#\; ]YEAR[A-Z\-\,\:\#\; ]{1,15}FULL[- ]{1}TIME?|\d+[A-Z\-\,\:\#\; ]*YEAR[A-Z\-\,\:\#\; ]*FULL[- ]{1}TIME?|\d+[A-Z\-\,\:\#\; ]*MONTH[A-Z\-\,\:\#\; ]*FULL[- ]{1}TIME?|[\d\,]+[A-Z\-\,\:\#\; ]*HOUR[A-Z\-\,\:\#\; ]*FULL[- ]{1}TIME?',FullTimeLen[0])

                            # Caculate the exact lenght by calling years_of_experience function

                            frmtOut[9] = years_of_experience(FullTimeLenTxt)



                        # Below is the logic the Extract the correponding Job Class if anything has been mentioned 

                        #   The REQuirement Details contains details  of Experience which can act as substitute 

                        #   Since the following logic identifies the Primary Job Class, the details pertaining to substitute experience has been omitted

                        FullTimeTextAll =[x for x in re.findall('.*FULL[- ]{1}TIME.*',rnumtoname) if re.findall('FULL[- ]{1}TIME.*SUBSTIT.*',x)==[] ]

                        FullTimeJBC = re.findall(JobtilePattern,'|'.join(FullTimeTextAll))



                        # Below is the logic the Extract the correponding alternate Job Class if anything has been mentioned 

                        #   The REQuirement Details contains details  of Experience which can act as substitute 

                        #   Since the following logic identifies the Alternative Job Class, the details pertaining to substitute experience has been considered

                        #       Please note that, from data analysis I identified that requirement can have a Full Time experience details along with any other experience

                        #       For alternate Job class we are considerring any experience be it be Full Time or not

                        #       Hence I am filtering data on EXPERIENCE

                        FULLExpTextAltAll = re.findall('.*EXPERIENCE.*',rnumtoname)

                        

                        # Intialize the list which will contain the Job Class details for Alternate Job Class for Full Time experience

                        FULLExpTxt =[]

                        for x in FULLExpTextAltAll:

                            # The below logic is for the senario when we have two set of Full Time experience details in the requirement

                            #    One that is for the Primary Job Class and Second for the alternative Job Class

                            #    Since the Primary Job Class has been evaluated earlier, the logic filters out the data for alternative Job Class(which is normally specified as the word Substitute)

                            if re.findall('FULL[- ]{1}TIME.*SUBSTIT.*',x)!=[]:

                                # Omitting the word Full Time in case of Alternate Job Class

                                # This will be helpful in the next steps and avoid filtering of data which contains alternate Job Class for Full Time

                                x = re.sub('FULL[- ]{1}TIME','',x)

                            FULLExpTxt = FULLExpTxt  + [x]    

                        #This logic extracts the final filtered text that contains alternate Job Class

                        #  In many requirement Education/Certification/Course/Program are considered as Substitute for an experience

                        #  Below logic Omits those details as we need only expericence details to compute alternate Job Class

                        #  Since the Requirement text already has Full Time experience details which was used to evaluate the Primary Job Class. We are omitting that details to avoid reconsidering the same for alternate Job Class

                        #  The logic Splits on 'SUBSTIT' as many of the requirement mentions that the experience is a substitute for the Primary JobClass and if we dont split the data we will get the Primary Job Class again in Alternate Job Class 

                        FULLExpTextAlt = [re.split('SUBSTIT',x)[0] for x in FULLExpTxt if re.findall('FULL[- ]{1}TIME|RECOMMENDATION|PROGRAM.*SUBSTIT.*|COURSE.*SUBSTIT.*|DEGREE.*SUBSTIT.*|CERTIFI.*SUBSTIT.*',x)==[]]

                        FULLExpAltJBC = re.findall(JobtilePattern,'|'.join(FULLExpTextAlt))



                        # Below Logic formats and cleans the Job Class in final format 

                        #  Multiple Job Class is seperated by |  

                        #  Omitting any unnecessary puncuation mark around the Job Class

                        frmtOut[11] = '|'.join([x.strip('\.|\,|\;|\"| ') for x in FullTimeJBC if x.strip('\.|\,|\;|\"| ') != ''])

                        frmtOut[12] = '|'.join([x.strip('\.|\,|\;|\"| ') for x in FULLExpAltJBC if x.strip('\.|\,|\;|\"| ') != ''])



#                   Processing the Part Time Experience Data

                    elif 'PART TIME' in ExpTypeFindAll or 'PART-TIME' in ExpTypeFindAll:

                        frmtOut[10] = 'PART-TIME'

                        # Check if any Experience Length is mentioned for Part Time experience

                        PARTTimeLen = re.findall('\d+.*PART[- ]{1}TIME.*',rnumtoname)



                        # Only first occurance of PART Time will be consider to evaluate experience length

                        if PARTTimeLen != []:

                            #In the event when multiple Part Time experience length is mentioned, the logic considers the first occurance of of the experience lenght

                            PARTTimeLenTxt = re.findall('\d+[A-Z\-\,\:\#\;\d ]{1,10}HALF[A-Z\-\,\:\#\; ]YEAR[A-Z\-\,\:\#\; ]{1,15}PART[- ]{1}TIME?|\d+[A-Z\-\,\:\#\; ]*YEAR[A-Z\-\,\:\#\; ]*PART[- ]{1}TIME?|\d+[A-Z\-\,\:\#\; ]*MONTH[A-Z\-\,\:\#\; ]*PART[- ]{1}TIME?|[\d\,]+[A-Z\-\,\:\#\; ]*HOUR[A-Z\-\,\:\#\; ]*PART[- ]{1}TIME?',PARTTimeLen[0])

                            # Caculate the exact length by calling years_of_experience function

                            frmtOut[9] = years_of_experience(PARTTimeLenTxt)



                        # Below is the logic the Extract the corresponding Job Class if anything has been mentioned 

                        #   The REQuirement Details contains details  of Experience which can act as substitute 

                        #   Since the following logic identifies the Primary Job Class, the details pertaining to substitute experience has been omitted

                        PARTTimeTextAll =[x for x in re.findall('.*PART[- ]{1}TIME.*',rnumtoname) if re.findall('PART[- ]{1}TIME.*SUBSTIT.*',x)==[] ]

                        PARTTimeJBC = re.findall(JobtilePattern,'|'.join(PARTTimeTextAll))



                        # Below is the logic the Extract the correponding alternate Job Class if anything has been mentioned 

                        #   The REQuirement Details contains details  of Experience which can act as substitute 

                        #   Since the following logic identifies the Alternative Job Class, the details pertaining to substitute experience has been considered

                        #       Please note that, from data analysis I identified that requirement can have a Part Time experience details along with any other experience

                        #       For alternate Job class we are considerring any experience be it be Part Time or not

                        #       Hence I am filtering data on EXPERIENCE

                        PARTExpTextAltAll = re.findall('.*EXPERIENCE.*',rnumtoname)



                        # Intialize the list which will contain the Job Class details for Alternate Job Class for Part Time experience

                        PARTExpTxt =[]

                        for x in PARTExpTextAltAll:

                            # The below logic is for the senario when we have two set of Part Time experience details in the requirement

                            #    One that is for the Primary Job Class and Second for the alternative Job Class

                            #    Since the Primary Job Class has been evaluated earlier, the logic filters out the data for alternative Job Class(which is normally specified as the word Substitute)

                            if re.findall('PART[- ]{1}TIME.*SUBSTIT.*',x)!=[]:

                                # Omitting the word Part Time in case of Alternate Job Class

                                # This will be helpful in the next steps and avoid filtering of data which contains alternate Job Class for Part Time

                                x = re.sub('PART[- ]{1}TIME','',x)

                            PARTExpTxt = PARTExpTxt  + [x]    



                        #This logic extracts the final filtered text that contains alternate Job Class

                        #  In many requirement Education/Certification/Course/Program are considered as Substitute for an experience

                        #  Below logic Omits those details as we need only expericence details to compute alternate Job Class

                        #  Since the Requirement text already has Part Time experience details which was used to evaluate the Primary Job Class. We are omitting that details to avoid reconsidering the same for alternate Job Class

                        #  The logic Splits on 'SUBSTIT' as many of the requirement mentions that the experience is a substitute for the Primary JobClass and if we dont split the data we will get the Primary Job Class again in Alternate Job Class 

                        PARTExpTextAlt = [re.split('SUBSTIT',x)[0] for x in PARTExpTxt if re.findall('PART[- ]{1}TIME|RECOMMENDATION|PROGRAM.*SUBSTIT.*|COURSE.*SUBSTIT.*|DEGREE.*SUBSTIT.*|CERTIFI.*SUBSTIT.*',x)==[]]

                        PARTExpAltJBC = re.findall(JobtilePattern,'|'.join(PARTExpTextAlt))



                        # Below Logic formats and cleans the Job Class in final format 

                        #  Multiple Job Class is seperated by |  

                        #  Omitting any unnecessary puncuation mark around the Job Class

                        frmtOut[11] = '|'.join([x.strip('\.|\,|\;|\"| ') for x in PARTTimeJBC if x.strip('\.|\,|\;|\"| ') != ''])

                        frmtOut[12] = '|'.join([x.strip('\.|\,|\;|\"| ') for x in PARTExpAltJBC if x.strip('\.|\,|\;|\"| ') != ''])



#                   Processing the Paid or Volunteer Experience Data

                    elif 'PAID OR VOLUNTEER' in ExpTypeFindAll or 'VOLUNTEER OR PAID' in ExpTypeFindAll:

                        frmtOut[10] = 'PAID OR VOLUNTEER'



                        # Check if any Experience Length is mentioned for PAID OR VOLUNTEER experience

                        PVTimeLen = re.findall('\d+.*PAID OR VOLUNTEER.*|\d+.*VOLUNTEER OR PAID.*',rnumtoname)

                        # Only first occurance of PAID OR VOLUNTEER Time will be consider to evaluate experience length

                        if PVTimeLen != []:

                            #In the event when multiple Paid or Volunteer Time experience length is mentioned, the logic considers the first occurance of of the experience lenght

                            PVTimeLenTxt = re.findall('\d+[A-Z\-\,\:\#\;\d ]{1,10}HALF[A-Z\-\,\:\#\; ]YEAR[A-Z\-\,\:\#\; ]{1,15}VOLUNTEER OR PAID?|\d+[A-Z\-\,\:\#\; ]*YEAR[A-Z\-\,\:\#\; ]*VOLUNTEER OR PAID?|\d+[A-Z\-\,\:\#\; ]*MONTH[A-Z\-\,\:\#\; ]*VOLUNTEER OR PAID?|[\d\,]+[A-Z\-\,\:\#\; ]*HOUR[A-Z\-\,\:\#\; ]*VOLUNTEER OR PAID?|\d+[A-Z\-\,\:\#\;\d ]{1,10}HALF[A-Z\-\,\:\#\; ]YEAR[A-Z\-\,\:\#\; ]{1,15}PAID OR VOLUNTEER?|\d+[A-Z\-\,\:\#\; ]*YEAR[A-Z\-\,\:\#\; ]*PAID OR VOLUNTEER?|\d+[A-Z\-\,\:\#\; ]*MONTH[A-Z\-\,\:\#\; ]*PAID OR VOLUNTEER?|[\d\,]+[A-Z\-\,\:\#\; ]*HOUR[A-Z\-\,\:\#\; ]*PAID OR VOLUNTEER?',PVTimeLen[0])

                            # Caculate the exact length by calling years_of_experience function

                            frmtOut[9] = years_of_experience(PVTimeLenTxt)



                        # Below is the logic the Extract the corresponding Job Class if anything has been mentioned 

                        #   The REQuirement Details contains details  of Experience which can act as substitute 

                        #   Since the following logic identifies the Primary Job Class, the details pertaining to substitute experience has been omitted

                        PVTimeTextAll =[x for x in re.findall('.*PAID OR VOLUNTEER.*|.*VOLUNTEER OR PAID.*',rnumtoname) if re.findall('PAID OR VOLUNTEER.*SUBSTIT.*|VOLUNTEER OR PAID.*SUBSTIT.*',x)==[] ]

                        PVTimeJBC = re.findall(JobtilePattern,'|'.join(PVTimeTextAll))



                        # Below is the logic the Extract the correponding alternate Job Class if anything has been mentioned 

                        #   The REQuirement Details contains details  of Experience which can act as substitute 

                        #   Since the following logic identifies the Alternative Job Class, the details pertaining to substitute experience has been considered

                        #       Please note that, from data analysis I identified that requirement can have a Paid or Volunteer Time experience details along with any other experience

                        #       For alternate Job class we are considerring any experience be it be Paid or Volunteer Time or not

                        #       Hence I am filtering data on EXPERIENCE

                        PVExpTextAltAll = re.findall('.*EXPERIENCE.*',rnumtoname)



                        # Intialize the list which will contain the Job Class details for Alternate Job Class for Paid or Volunteer Time experience

                        PVExpTxt =[]

                        for x in PVExpTextAltAll:

                            # The below logic is for the senario when we have two set of Paid or Volunteer Time experience details in the requirement

                            #    One that is for the Primary Job Class and Second for the alternative Job Class

                            #    Since the Primary Job Class has been evaluated earlier, the logic filters out the data for alternative Job Class(which is normally specified as the word Substitute)

                            if re.findall('PAID OR VOLUNTEER.*SUBSTIT.*|VOLUNTEER OR PAID.*SUBSTIT.*',x)!=[]:

                                # Omitting the word Paid or Volunteer Time in case of Alternate Job Class

                                # This will be helpful in the next steps and avoid filtering of data which contains alternate Job Class for Paid or Volunteer Time

                                x = re.sub('PAID OR VOLUNTEER|VOLUNTEER OR PAID','',x)

                            PVExpTxt = PVExpTxt  + [x]    



                        #This logic extracts the final filtered text that contains alternate Job Class

                        #  In many requirement Education/Certification/Course/Program are considered as Substitute for an experience

                        #  Below logic Omits those details as we need only expericence details to compute alternate Job Class

                        #  Since the Requirement text already has Paid or Volunteer Time experience details which was used to evaluate the Primary Job Class. We are omitting that details to avoid reconsidering the same for alternate Job Class

                        #  The logic Splits on 'SUBSTIT' as many of the requirement mentions that the experience is a substitute for the Primary JobClass and if we dont split the data we will get the Primary Job Class again in Alternate Job Class 

                        PVExpTextAlt = [re.split('SUBSTIT',x)[0] for x in PVExpTxt if re.findall('PAID OR VOLUNTEER|VOLUNTEER OR PAID|RECOMMENDATION|PROGRAM.*SUBSTIT.*|COURSE.*SUBSTIT.*|DEGREE.*SUBSTIT.*|CERTIFI.*SUBSTIT.*',x)==[]]

                        PVExpAltJBC = re.findall(JobtilePattern,'|'.join(PVExpTextAlt))



                        # Below Logic formats and cleans the Job Class in final format 

                        #  Multiple Job Class is seperated by |  

                        #  Omitting any unnecessary puncuation mark around the Job Class

                        frmtOut[11] = '|'.join([x.strip('\.|\,|\;|\"| ') for x in PVTimeJBC if x.strip('\.|\,|\;|\"| ') != ''])

                        frmtOut[12] = '|'.join([x.strip('\.|\,|\;|\"| ') for x in PVExpAltJBC if x.strip('\.|\,|\;|\"| ') != ''])



#                   Processing the Apprenticeship Experience Data

                    elif 'APPRENTICESHIP' in ExpTypeFindAll:

                        frmtOut[10] = 'APPRENTICESHIP'

                       

                        # Check if any Experience Length is mentioned for APPRENTICESHIP experience

                        APPTimeLen = re.findall('\d+.*APPRENTICESHIP.*',rnumtoname)

                        # Only first occurance of APPRENTICESHIP Time will be consider to evaluate experience length

                        if APPTimeLen != []:

                            APPTimeLenTxt = re.findall('\d+[A-Z\-\,\:\#\;\d ]{1,10}HALF[A-Z\-\,\:\#\; ]YEAR[A-Z\-\,\:\#\; ]{1,15}APPRENTICESHIP?|\d+[A-Z\-\,\:\#\; ]*YEAR[A-Z\-\,\:\#\; ]*APPRENTICESHIP?|\d+[A-Z\-\,\:\#\; ]*MONTH[A-Z\-\,\:\#\; ]*APPRENTICESHIP?|[\d\,]+[A-Z\-\,\:\#\; ]*HOUR[A-Z\-\,\:\#\; ]*APPRENTICESHIP?',APPTimeLen[0])

                            frmtOut[9] = years_of_experience(APPTimeLenTxt)



                        # Since this section considers Apprenticeship, it cant have Job Classes

                        # Considered Apprenticeship as just a pre requisite Job Class internship for Job Class Applicant will apply for

                        frmtOut[11] = ''

                        frmtOut[12] = ''



#                   Processing the Hours Of Experience Data

                    elif re.search('HOURS OF','|'.join(ExpTypeFindAll)) is not None:

                        frmtOut[10] = 'HOURS OF'

                        # Check if any Experience Length is mentioned for HOURS OF experience

                        HoursOfLen = re.findall('\d+.*HOURS OF.*',rnumtoname)

                        # Only first occurance of HOURS OF will be consider to evaluate experience length

                        if HoursOfLen != []:

                            #In the event when multiple Hours Of  Time experience length is mentioned, the logic considers the first occurance of of the experience lenght

                            HoursOfLenTxt = re.findall('[\d\,]+[A-Z\-\,\:\#\; ]*HOURS OF?',HoursOfLen[0])

                            # Caculate the exact length by calling years_of_experience function

                            frmtOut[9] = years_of_experience(HoursOfLenTxt)



                        # Below is the logic the Extract the corresponding Job Class if anything has been mentioned 

                        #   The REQuirement Details contains details  of Experience which can act as substitute 

                        #   Since the following logic identifies the Primary Job Class, the details pertaining to substitute experience has been omitted

                        HoursOfTextAll =[x for x in re.findall('.*HOURS OF.*',rnumtoname) if re.findall('HOURS OF.*SUBSTIT.*',x)==[] ]

                        HoursOfJBC = re.findall(JobtilePattern,'|'.join(HoursOfTextAll))



                        # Below is the logic the Extract the correponding alternate Job Class if anything has been mentioned 

                        #   The REQuirement Details contains details  of Experience which can act as substitute 

                        #   Since the following logic identifies the Alternative Job Class, the details pertaining to substitute experience has been considered

                        #       Please note that, from data analysis I identified that requirement can have a Hours Of  Time experience details along with any other experience

                        #       For alternate Job class we are considerring any experience be it be Hours Of  Time or not

                        #       Hence I am filtering data on EXPERIENCE

                        HoursOfTextAltAll = re.findall('.*EXPERIENCE.*',rnumtoname)

                        # Intialize the list which will contain the Job Class details for Alternate Job Class for Hours Of  Time experience

                        HoursOfTxt =[]

                        for x in HoursOfTextAltAll:

                            # The below logic is for the senario when we have two set of Hours Of  Time experience details in the requirement

                            #    One that is for the Primary Job Class and Second for the alternative Job Class

                            #    Since the Primary Job Class has been evaluated earlier, the logic filters out the data for alternative Job Class(which is normally specified as the word Substitute)

                            if re.findall('HOURS OF.*SUBSTIT.*',x)!=[]:

                                # Omitting the word Hours Of  Time in case of Alternate Job Class

                                # This will be helpful in the next steps and avoid filtering of data which contains alternate Job Class for Hours Of  Time

                                x = re.sub('HOURS OF','',x)

                            HoursOfTxt = HoursOfTxt  + [x]    



                        #This logic extracts the final filtered text that contains alternate Job Class

                        #  In many requirement Education/Certification/Course/Program are considered as Substitute for an experience

                        #  Below logic Omits those details as we need only expericence details to compute alternate Job Class

                        #  Since the Requirement text already has Hours Of  Time experience details which was used to evaluate the Primary Job Class. We are omitting that details to avoid reconsidering the same for alternate Job Class

                        #  The logic Splits on 'SUBSTIT' as many of the requirement mentions that the experience is a substitute for the Primary JobClass and if we dont split the data we will get the Primary Job Class again in Alternate Job Class 

                        HoursOfTextAlt = [re.split('SUBSTIT',x)[0] for x in HoursOfTxt if re.findall('HOURS OF|RECOMMENDATION|PROGRAM.*SUBSTIT.*|COURSE.*SUBSTIT.*|DEGREE.*SUBSTIT.*|CERTIFI.*SUBSTIT.*',x)==[]]

                        HoursOfAltJBC = re.findall(JobtilePattern,'|'.join(HoursOfTextAlt))



                        # Below Logic formats and cleans the Job Class in final format 

                        #  Multiple Job Class is seperated by |  

                        #  Omitting any unnecessary puncuation mark around the Job Class

                        frmtOut[11] = '|'.join([x.strip('\.|\,|\;|\"| ') for x in HoursOfJBC if x.strip('\.|\,|\;|\"| ') != ''])

                        frmtOut[12] = '|'.join([x.strip('\.|\,|\;|\"| ') for x in HoursOfAltJBC if x.strip('\.|\,|\;|\"| ') != ''])



#                   Processing the Other Experience Data

                    else:

                        # The Below logic checks for any Education/Certification/Program/Course/ Certificate related inforamation is present in the requirement that is substitute for experience

                        StudySubExpFindAll = re.findall('GRADUAT[A-Z0-9\W ]*SUBSTIT[A-Z0-9\W ]*EXPERIENCE[\.]?|DEGREE[A-Z0-9\W ]*SUBSTIT[A-Z0-9\W ]*EXPERIENCE|SEMESTER[A-Z0-9\W ]*SUBSTIT[A-Z0-9\W ]*EXPERIENCE|QUARTER[A-Z0-9\W ]*SUBSTIT[A-Z0-9\W ]*EXPERIENCE|PROGRAM[A-Z0-9\W ]*SUBSTIT[A-Z0-9\W ]*EXPERIENCE|COURSE[A-Z0-9\W ]*SUBSTIT[A-Z0-9\W ]*EXPERIENCE',rnumtoname)



                        # The further logic is executed if

                        #    No Education/Certification/Program/Course/ Certificate related inforamation is present as we are bothered only for the experience details

                        #    Or 

                        #    In case, any Education/Certification/Program/Course/ Certificate related inforamation is present

                        #      It check any additonal Experience data is available. So the word EXPERIENCE has to appear more than Study related Data

                        #          If the requirement says that XYZ education is substitute for Experience then there must be additional Experience data

                        if StudySubExpFindAll == [] or (StudySubExpFindAll != [] and ExpTypeFindAll.count('EXPERIENCE') > len(StudySubExpFindAll)):

                            frmtOut[10] = 'OTHERS'



                            # Check if any Experience Length is mentioned for OTHERS experience

                            OthersLen =   re.findall('\d+.*EXPERIENCE.*|CURRENT EMPLOY.*|\d+.*YEARS AS A.*|EXPERIENCE EQU.*\d+.*',rnumtoname)

                            # Only first occurance of Other Time will be consider to evaluate experience length

                            if OthersLen != []:



                                #In the event when multiple Other experience length is mentioned, the logic considers the first occurance of of the experience lenght

                                #  The below consider the text where Experience length comes first like  '5 years of Experience'

                                OthersLenTxt = re.findall('\d+[A-Z\-\,\:\#\;\d ]{1,10}HALF[A-Z\-\,\:\#\; ]YEAR[A-Z\-\,\:\#\; ]{1,15}EXPERIENCE?|\d+[A-Z\-\,\:\#\; ]*YEAR[A-Z\-\,\:\#\; ]*EXPERIENCE?|\d+[A-Z\-\,\:\#\; ]*MONTH[A-Z\-\,\:\#\; ]*EXPERIENCE?|[\d\,]+[A-Z\-\,\:\#\; ]*HOUR[A-Z\-\,\:\#\; ]*EXPERIENCE?|\d+[A-Z\-\,\:\#\;\d ]{1,10}HALF[A-Z\-\,\:\#\; ]YEAR[A-Z\-\,\:\#\; ]{1,15}CURRENT EMPLOY?|\d+[A-Z\-\,\:\#\; ]*YEAR[A-Z\-\,\:\#\; ]*CURRENT EMPLOY?|\d+[A-Z\-\,\:\#\; ]*MONTH[A-Z\-\,\:\#\; ]*CURRENT EMPLOY?|[\d\,]+[A-Z\-\,\:\#\; ]*HOUR[A-Z\-\,\:\#\; ]*CURRENT EMPLOY?|\d+[A-Z\-\,\:\#\; ]*YEARS AS A?',OthersLen[0])

                                #  The below consider the text where Experience length comes later  like 'Experience Equivlant to 5 years'

                                OthersLenEquTxt = re.findall('EXPERIENCE EQU[A-Z\W ]*TO [\d]+[A-Z\W ]*HALF[A-Z\-\,\:\#\; ]YEAR|EXPERIENCE EQU[A-Z\W ]*TO [\d]+[A-Z\W ]*YEAR|EXPERIENCE EQU[A-Z\W ]*TO [\d]+[A-Z\W ]*MONTH|EXPERIENCE EQU[A-Z\W ]*TO [\d\,]*[A-Z\W ]*HOUR',OthersLen[0])

                                

                                # Calling the functions that calculates the actual experience lenght based on the above two conditions                             

                                if OthersLenEquTxt !=[]:

                                    frmtOut[9] = years_of_experiencelater(OthersLenEquTxt)

                                else:

                                    frmtOut[9] = years_of_experience(OthersLenTxt)



                                # Below is the logic the Extract the corresponding Job Class if anything has been mentioned 

                                #   The REQuirement Details contains details  of Experience which can act as substitute 

                                OthersTextAll =[x for x in re.findall('.*EXPERIENCE.*|.*CURRENT EMPLOY.*|.*YEARS AS A.*|.*EXPERIENCE EQU.*',rnumtoname) if re.findall('EXPERIENCE.*SUBSTIT.*|CURRENT EMPLOY.*SUBSTIT.*|YEARS AS A.*SUBSTIT.*|EXPERIENCE EQU.*SUBSTIT.*',x)==[] ]

                                OthersJBC = re.findall(JobtilePattern,'|'.join(OthersTextAll))



                                # Below Logic formats and cleans the Job Class in final format 

                                #  Multiple Job Class is seperated by |  

                                #  Omitting any unnecessary puncuation mark around the Job 

                                #  For Experience in Other Category, the logic does not  evaluate alternate Job Class as it this category considers all the experience that were not considered earlier 

                                frmtOut[11] = '|'.join([x.strip('\.|\,|\;|\"| ') for x in OthersJBC if x.strip('\.|\,|\;|\"| ') != ''])

                                frmtOut[12] = ''

                                        



#               The Below logic extracts the Job Class function if the requirement has contains experience details as per the category mentioned above

                if  frmtOut[10] != '':

                    JobFunction= re.findall('.*',rowstrip)

                    # Omitting other details that are already captured in other elements and extracting the Job Class function details    

                    JobFunctionDetail= [x for x in re.findall('.*',rowstrip) if re.findall('FULL[- ]{1}TIME|PART[- ]{1}TIME|PAID OR VOLUNTEER|VOLUNTEER OR PAID|HOURS OF|APPRENTICESHIP|EXPERIENCE|CURRENT.*EMPLOY|VALID.*CARD|YEAR|MONTH|HOUR|PROGRAM|COURSE|DEGREE|HIGH SCHOOL|CERTIFI|REGIST|QUARTER|SEMETER|LICENSE|COMPLETION OF|ATTAINMENT OF|RECOMMENDATION|U.S. CITIZENSHIP|GRADUAT',x,flags=re.IGNORECASE) == [] and x.strip()!='']

                    JobFunctionDetailClean = [re.sub('^[A-Za-z0-9]{1}[\.\) ]{1}|\s',' ',x).strip('\-\* ') for x in JobFunctionDetail if re.findall('JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER',x,flags=re.IGNORECASE)==[]]

                    frmtOut[13] = ';and '.join(JobFunctionDetailClean)

#END Experience Details

#######################################################################################





                if updateflag == 1 :

                    frmtOut_CP = frmtOut.copy()

                    frmtAll.append(frmtOut_CP)

                updateflag = 0

            



#END: REQUIREMENT DETAILS 

###################################################################################

###################################################################################

###################################################################################

###################################################################################







pandaDF = pd.DataFrame(frmtAll,columns=fileheader)

FilesWithMissingInfoDF = pd.DataFrame(FilesWithMissingInfo,columns=['Error Details','File Name'])



#print(pandaDF)



pandaDF.to_csv("JobBullentinExtractCSV.csv",sep=',',index=None)

FilesWithMissingInfoDF.to_csv("FilesWithMissingInfo.csv",sep=',',index=None)

print('Structured CSV file have been created as JobBullentinExtractCSV.csv' )

print(FilesWithMissingInfoDF)

print('Count Of Files with Missing Data:',len(FilesWithMissingInfo))

#pandaDF.to_csv()
import re

from os import listdir

#import readability

from wordcloud import WordCloud

from os.path import isfile, join

import pandas as pd

import matplotlib.pyplot as plt



#Get the list of files under Job Bulletins

foldername='../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/'

filelist = [f for f in listdir(foldername) if isfile(join(foldername, f))]



#Initialize  list

AnalysisAll = []

JobHeaderAll = []





#Process the files in Job Bulletins folder

for filename in filelist:



    entirepath= foldername + filename

    textfile = open(entirepath,'r',encoding='latin-1')

    filetext = textfile.read()

    textfile.close()



#initializing the List for current file as EMPTY

    AnalysisOut =[0,0,0,0,0,0,0,0,0,0]



#Populating the filename as a first element of list

    AnalysisOut[0] = filename



# Calculating the length of file

    AnalysisOut[1] = len(filetext)

    

# Calculating the Number of sentences of file    

    AnalysisOut[2] = len(re.findall('\n',filetext))



# Calculating the Number of Section(like Annual Salary, Duties) and their length in the file 

    HeaderListRaw = re.findall('\n[A-Z\d\W]{4,}\n',filetext)

    HeaderList = [re.sub('\n*|\s{2,}|\t|\:','',x) for x in HeaderListRaw]

    AnalysisOut[3] = HeaderList

    AnalysisOut[4] = len(HeaderList)

    

# Generating a list containing length of each section 

    StartPoint = 0

    NewText = filetext[StartPoint:]

    SectionLen=[]

    for x in HeaderListRaw:

        NewText = NewText[StartPoint:]

        Srchx = re.search(x.replace('+','\+'),NewText)

        if Srchx is not None:

            SectionText = NewText[:Srchx.start()]

            StartPoint = Srchx.start()

        SectionLen = SectionLen + [len(SectionText)]

    SectionLen = SectionLen + [len(NewText[StartPoint:])]

    AnalysisOut[5] = SectionLen

    

# Calculating the Minimum length of sections in file 

    AnalysisOut[6] = min(SectionLen)

# Calculating the Maximum length of sections in file 

    AnalysisOut[7] = max(SectionLen)

# Calculating the Average length of sections in file 

    AnalysisOut[8] = round(sum(SectionLen)/len(SectionLen))

# Calculating the FleschReadingScale of the file 

#    AnalysisOut[9] = readability.getmeasures(filetext, lang='en')['readability grades']['FleschReadingEase']

    AnalysisOut[9] = 0

    

    AnalysisAll.append(AnalysisOut)

    JobHeaderAll = JobHeaderAll + AnalysisOut[3]



#Converting into panda dataframe

promoDF = pd.DataFrame(AnalysisAll,columns=['File Name','File Length','Number of sentences','Section in file','Number of Sections','Length Of Sections','Section Min Length','Section Max Length','Section Avg Length','FleschReadingScale'])

print("Average of various count showing the statistics on the readibility of Job Bulletins")

print(promoDF.mean())

# Create and generate a word cloud image:

wordcloud = WordCloud().generate(' '.join(JobHeaderAll))



fig = plt.figure(figsize=(35,9))



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()



#Writing into csv file for the ease of analysis

promoDF.to_csv("JBAnalysisDetails.csv",sep=',',index=None)

from os import listdir

from os.path import isfile, join

import pandas as pd

import matplotlib.pyplot as plt



#Get the list of files under Job Bulletins

foldername='../input/kaggle-cityofla-competition/'

filename = 'JobBullentinExtractCSV.csv'



entirepath= foldername + filename

#Setting an empty JobTile list

JobBull = []



#Read the Job Bulletins CSV into a dataframe and convert it to list

JobBullDF = pd.read_csv(entirepath)



#Plotting Graphs

JobBullDF['SCHOOL_TYPE'].value_counts().plot.bar()

plt.title('JOB Bulletins volume based on EDUCATION TYPE')

plt.xlabel('EDUCATION TYPE')

plt.ylabel('Count of JobBulletins')

plt.show()





JobBullDF['FULL_TIME_PART_TIME'].value_counts().plot.bar()

plt.title('JOB Bulletins volume based on EXPERIENCE TYPE')

plt.xlabel('EXPERIENCE TYPE')

plt.ylabel('Count of JobBulletins')



plt.show()



JobBullDF['EXPERIENCE_LENGTH'].hist()

plt.title('JOB Bulletins volume based on JOB EXPERIENCE TENURE')

plt.xlabel('JOB EXPERIENCE TENURE(In Years)')

plt.ylabel('Count of JobBulletins')



plt.show()
import re

from os import listdir

#import readability

from wordcloud import WordCloud

from os.path import isfile, join

import pandas as pd

import matplotlib.pyplot as plt



#Get the list of files under Job Bulletins

foldername='../input/kaggle-cityofla-competition'

filelist = [f for f in listdir(foldername)]

print(filelist)

promoDF = pd.DataFrame(filelist)

print(promoDF)
