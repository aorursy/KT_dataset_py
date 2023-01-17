# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from random import seed

from random import choice

import os

import csv

import re

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
##           BASE EXTRACT METHOD                        ##

##########################################################

## def note: For the Extract_... functions we simply 

## 1. Define a Regex to match the file text - re.compile...

## 2. Apply the Regex to the file text - pattern...findall()

## 3. Equate the first item in the match list to our match

## 4. Return the match value

##########################################################



def Extract_Open_Date(file_text):

    pattern=re.compile(r'(?<=en\s[dD]ate:\s)([,\d\s-]*)',re.I)

    match = pattern.findall(file_text)

    match=match[0]

    return match.strip()



def Extract_Class_Code(file_text):

    pattern=re.compile(r'(?<=[Cc]ode[:])\s*(.*)',re.I)

    match = pattern.findall(file_text)

    match=match[0]

    return match.strip()	



def If_Drivers_License(file_text):

    pattern=re.compile(r'PROCESS\sNOTES?\s\s([\S\s]*?)[A-Z]{4,5}')

    pattern3=re.compile(r'may\srequire', re.I)

    pattern4=re.compile(r'possible', re.I)

    

    match = pattern.findall(file_text)

    if (match):

        match=match[0]

    else:

        match=''

        

    pattern2=re.compile(r'driver', re.I)

    if(pattern2.findall(match)):

        match=pattern2.findall(match)

        match=match[0]

        if(pattern3.findall(match) or pattern4.findall(match)):

            match='P'

        else:

            match='R'

    else:

        match='NaN'

    



        

            

    return match
def Extract_Drivers_License_Type(file_text):

    pattern=re.compile(r'(?<=[Cc]lass)([\w\s]{1,8})', re.I)

    pattern1=re.compile(r'\b[A-Z]\b')

    if(pattern.findall(file_text)):

        match = pattern.findall(file_text)

        match=match[0]

    if (pattern1.findall(match)):

        match=pattern1.findall(match)

        for i in range(0, len(match)):

            match=match[0]

            if (len(match)==2):

                match=match[0]+', '+match[1]

    else:

        match='NaN'

    

    return match





#######  M  ############	

def Extract_Course_Length(file_text):

    pattern=re.compile(r'(\d\d?\ssemester\sunits\sor\s\d\d?\squarter\sunits)')

    if(pattern.findall(file_text)):

        match=pattern.findall(file_text)

        pattern2=re.compile(r'\d\d?')

        match=match[0]

        SQ=pattern2.findall(match)

        semester=SQ[0]+'S'

        quarter=SQ[1]+'Q'

        match=semester+'|'+quarter

    else: 

        match='NaN'

    

    return match





def Extract_Duties(file_text):

    pattern=re.compile(r'DUTIES?\s([\S\s]*?)[A-Z]{4,5}')

    if(pattern.findall(file_text)):

        match = pattern.findall(file_text)

    else:

        match='NaN'



    match=match[0]



    return match



##   On the second iteration of the match procedure, a descending

##   specificity is used to filter the results.



def Extract_Exam_Type(file_text):

    pattern=re.compile(r'(?<=THIS\sEXAMINATION)([\S\s]*?)BASIS')

    if(pattern.findall(file_text)):

        match = pattern.findall(file_text)

        match=match[0]

    else:

        match=''

    pattern2=re.compile(r'OPEN\sINTERDEPARTMENTAL\sPROMOTION')

    pattern3=re.compile(r'INTERDEPARTMENTAL\sPROMOTION')

    pattern4=re.compile(r'DEPARTMENTAL\sPROMOTION')

    pattern5=re.compile(r'OPEN')

    

  

    if(pattern2.findall(match)):

        match="OPEN_INT_PROM"

    elif(pattern3.findall(match)):

        match="INT_DEPT_PROM"

    elif(pattern4.findall(match)):

        match="DEPT_PROM"

    elif(pattern5.findall(match)):

        match="OPEN"

    else:

        match="NaN"

    

    return match

    

    

def Extract_Exam_Distro(file_text):

    if(re.compile(r'(?<=\sPROCESS)([\S\s]*)ADDI')):

        pattern=re.compile(r'(?<=\sPROCESS)([\S\s]*)ADDI')

        match = pattern.findall(file_text)

    elif(re.compile(r'(?<=\sPROCESS)([\S\s]*)NOT')):

        pattern2=re.compile(r'(?<=\sPROCESS)([\S\s]*)NOT')

        match2 = pattern2.findall(file_text)

    

        

    pattern3=re.compile(r'written\stest',re.I)

    pattern4=re.compile(r'interview',re.I)

    pattern5=re.compile(r'Performance\sTest',re.I)

    pattern6=re.compile(r'Multiple-?Choice\sTest',re.I)

    pattern7=re.compile(r'Easy',re.I)

    

    pattern8=re.compile(r'[A-Z]+[A-Z]{3,4}\s*')

    match_f=''

    match_w=''

    match_f2=''

    match_w2=''

    match=''

    

    if(pattern3.findall(match)):

        match_f = pattern3.findall(match)

        match_w+="Written Test"

    if(pattern4.findall(match)):

        match_f = pattern4.findall(match)

        match_w+="Interview"

    if(pattern5.findall(match)):

        match_f = pattern5.findall(match)

        match_w+="Performance Test"

    if(pattern6.findall(match)):

        match_f = pattern6.findall(match)

        match_w+="Multiple-Choice Test"

    if(pattern7.findall(match)):

        match_f = pattern7.findall(match)

        match_w+="Easy"

    if(match_w!=''):

        match=match_w

    

    if(pattern8.findall(match)):

        for i in len(pattern8.findall(match)):

            match_f = pattern8.findall(match)

            match_w2+=match_f[i]

        match=match_w2    

    

    return match
def Extract_Job_Class(job_title):

    

    pattern=re.compile(r'(.*)\s(?=I|II|III|VII|V|IV|VI|V)')

    ## If it is graded, then chop the Roman Numerals

    if(pattern.findall(job_title)):

        match=pattern.findall(job_title)

        match=match[0]

    else:

        match=job_title

    

    

    return match

    
def Extract_School_Type(file_text):

    match1='NaN'

    match2='NaN'

    match3='NaN'

    match4='NaN'

    

    pattern_case1=re.compile(r'QUALIFICATIONS?\s\s([\S\s]*?)[A-Z]{4,5}')

    pattern_case2=re.compile(r'REQUIREMENTS?\s\s([\S\s]*?)[A-Z]{4,5}')

    

    if(pattern_case1.findall(file_text)):

        match = pattern_case1.findall(file_text)

        match=match[0]

    elif(pattern_case2.findall(file_text)):

        match = pattern_case2.findall(file_text)

        match=match[0]

    else:

        match=''



    pattern_case3=re.compile(r'high\sschool',re.I)

    pattern_case4=re.compile(r'college', re.I)

    pattern_case5=re.compile(r'university', re.I)

    pattern_case6=re.compile(r'apprenticeship', re.I)

    

    

    if(pattern_case3.findall(match)):

        match1 = 'HIGH SCHOOL'

    if(pattern_case4.findall(match)):

        match2 = 'COLLEGE or UNIVERSITY'

    if(pattern_case5.findall(match)):

        match2 = 'COLLEGE or UNIVERSITY'

    elif(pattern_case6.findall(match)):

        match4 = 'APPRENTICESHIP'

    else:

        match='NaN'

        

    match=match1+', '+match2+', '+match4

        

    return match
def Count_Commas(file_text):

    match1=''

    match2=''

    match3=''

    match4=''

    

    pattern_case1=re.compile(r',')

    

    if(pattern_case1.findall(file_text)):

        match=pattern_case1.findall(file_text)

        size=len(match)+1

    else:

        size='NaN'

        

    return size
def Extract_General_Skills(file_text):

    match1=''

    match2=''

    match3=''

    match4=''

    

    pattern_case1=re.compile(r'QUALIFICATIONS?\s\s([\S\s]*?)[A-Z]{4,5}')

    pattern_case2=re.compile(r'REQUIREMENTS?\s\s([\S\s]*?)[A-Z]{4,5}')

    

    if(pattern_case1.findall(file_text)):

        match = pattern_case1.findall(file_text)

        

    elif(pattern_case2.findall(file_text)):

        match = pattern_case2.findall(file_text)

    else:

        match=''



    if(len(match)>1):

        match=match[0]

    elif(len(match)==1):

        match=match[0]

    else:

        match=''

        

    pattern_case3=re.compile(r'(?<=experience\sin\s)(.*[;.])')

 

    #print(match[0])

    if(pattern_case3.findall(match)):

        match = pattern_case3.findall(match)

        match=match[0]

    else:

        match=''

        

    #Exclude the matches with an "ing"    

    pattern_case4=re.compile(r'ing')

    if(pattern_case4.findall(match)):

        match = match[0]

    else:

        match=''

        

    pattern_case5=re.compile(r'[\w]*,\sa?n?d?\s?[\w]*')

    if(pattern_case5.findall(match)):

        match = match

    else:

        match='NaN'      

        

    return match
def Word_to_Number(word):

    num=0

    

    if(word=='one' or word=='One'):

        num=1

    elif(word=='two' or word=='Two'):

        num=2

    elif(word=='three' or word=='Three'):

        num=3

    elif(word=='four' or word=='Four'):

        num=4

    elif(word=='five' or word=='Five'):

        num=5

    elif(word=='six' or word=='Six'):

        num=6

    #print(num)

    return num
def Extract_Education_Years(file_text):

    match1=''

    match2=''

    match3=''

    match4=''

    

    pattern_case1=re.compile(r'QUALIFICATIONS?\s\s([\S\s]*?)[A-Z]{4,5}')

    pattern_case2=re.compile(r'REQUIREMENTS?\s\s([\S\s]*?)[A-Z]{4,5}')

    

    if(pattern_case1.findall(file_text)):

        match = pattern_case1.findall(file_text)

        match=match[0]

    elif(pattern_case2.findall(file_text)):

        match = pattern_case2.findall(file_text)

        match=match[0]

    else:

        match=''





    pattern_case3=re.compile(r'([tT]wo|f[F]our|[oO]ne)-?\s?(?=year\s)', re.I)



    if(pattern_case3.findall(match)):

        match = pattern_case3.findall(match)

        match=match[0]

        num=Word_to_Number(match)

    else:

        match='NaN'

                    

    

        

    return match
def Extract_Board_Certs(file_text):

    match1=''

    match2=''

    match3=''

    match4=''

    

    pattern_case1=re.compile(r'QUALIFICATIONS?\s\s([\S\s]*?)[A-Z]{4,5}')

    pattern_case2=re.compile(r'REQUIREMENTS?\s\s([\S\s]*?)[A-Z]{4,5}')

    

    if(pattern_case1.findall(file_text)):

        match = pattern_case1.findall(file_text)

        

    elif(pattern_case2.findall(file_text)):

        match = pattern_case2.findall(file_text)

    else:

        match=''



    if(len(match)>1):

        match=match[0]

    elif(len(match)==1):

        match=match[0]

    else:

        match=''

        

    pattern_case3=re.compile(r'(?<=by\sthe\s)([\w+\s]+)[;,.]')



    if(pattern_case3.findall(match)):

        match = pattern_case3.findall(match)

        match=match[0]

    else:

        match=''

        

    ## A filtering procedure: Let's exclude some things

    pattern_case4=re.compile(r'(Los\sAngeles)')

    pattern_case5=re.compile(r'(Motor)')

    pattern_case6=re.compile(r'(Water)')                         

                             

    if(pattern_case4.findall(match) or pattern_case5.findall(match) or pattern_case6.findall(match) ):

        match = ''

        

    

        

    return match
def Extract_Salary(file_text):

    

    pattern_case1=re.compile(r'SALARY?\s\s([\S\s]*?)[A-Z]{4,5}')



    

    if(pattern_case1.findall(file_text)):

        match = pattern_case1.findall(file_text)

    else:

        match=''



    if(len(match)>1 or len(match)==1):

        match=match[0]

    else:

        match=''

        

    ## Pattern to match ###,### or ##,###, 

    ## i.e. $10,000 - $999,999

    pattern_case3=re.compile(r'(\d\d\d?,\d\d\d)')

 

    if(pattern_case3.findall(match)):

        match = pattern_case3.findall(match)

    else:

        match=''

    

    ## Convert the strings to int's because the max and min do not work

    ## well on string numerical values

    for i in range(0, len(match)):

        match[i]=int(match[i].replace(',', ''))

        

    mino=min(match)

    maxo=max(match)

    

    match_final='$'+str(mino)+' to '+'$'+str(maxo)

        

    return match_final
def Extract_DWP_Salary(file_text):

    

    pattern_case1=re.compile(r'SALARY?\s\s([\S\s]*?)[A-Z]{4,5}')

    pattern_case2=re.compile(r'Water\sand\sPower.*')

    

    if(pattern_case1.findall(file_text)):

        match = pattern_case1.findall(file_text)

        match=match[0]

    else:

        match=''





    if(pattern_case2.findall(match)):

        match = pattern_case2.findall(match)

        match=match[0]

    else:

        match=''

    

    pattern_case3=re.compile(r'(\d\d\d?,\d\d\d)')

 

    if(pattern_case3.findall(match)):

        match = pattern_case3.findall(match)

    else:

        match=''

    

    ## Convert the strings to int's because the max and min do not work

    ## well on string numerical values

    for i in range(0, len(match)):

        match[i]=int(match[i].replace(',', ''))

    

    if (match):    

        mino=min(match)

        maxo=max(match)

    

        match_final='$'+str(mino)+' to '+'$'+str(maxo)

    else:    

        match_final='NaN'

        

    return match_final
def Extract_EEOC(file_text):

    match1=''

    

    pattern_case1=re.compile(r'BASIS?\s\s([\S\s]*?)For')

  

    

    if(pattern_case1.findall(file_text)):

        match = pattern_case1.findall(file_text)

        match=match[0]

    else:

        match=''



        

    pattern_case3=re.compile(r'discriminate')



    if(pattern_case3.findall(match)):

        match1 = pattern_case3.findall(match)

        match1=match[0]

    else:

        match='NaN'

        

        

    return match
def Extract_Major(file_text):

    match1=''

    

    pattern_case1=re.compile(r'QUALIFICATIONS?\s\s([\S\s]*?)[A-Z]{4,5}')

    pattern_case2=re.compile(r'REQUIREMENTS?\s\s([\S\s]*?)[A-Z]{4,5}')

    pattern_case3=re.compile(r'(?<=major)i?n?g?\sin\s(.*)', re.I)

    pattern_case4=re.compile(r'\w?[A-Z]?[a-z]*?\s[\w]+,\sa?n?d?o?r?\s?[\w]*', re.I) 

    pattern_case5=re.compile(r'(?<=degree\sin\s)(.*)', re.I)

    

    if(pattern_case1.findall(file_text)):

        match = pattern_case1.findall(file_text)

        match=match[0]

    elif(pattern_case2.findall(file_text)):

        match = pattern_case2.findall(file_text)

        match=match[0]

    else:

        match=''



  

    if(pattern_case3.findall(match)):

        match1 = pattern_case3.findall(match)

        match=match1[0]

    elif(pattern_case5.findall(match)):

        match1 = pattern_case5.findall(match)

        match=match1[0]

    else:

        match=''

  

    if(pattern_case4.findall(match)):

        match1 = pattern_case4.findall(match)

        match=match1[0]

    else:

        match='NaN'

        

        

    return match
def Extract_Language(file_text):

    

    pattern_case1=re.compile(r'QUALIFICATIONS?\s\s([\S\s]*?)[A-Z]{4,5}')

    pattern_case2=re.compile(r'REQUIREMENTS?\s\s([\S\s]*?)[A-Z]{4,5}')

    

    if(pattern_case1.findall(file_text)):

        match = pattern_case1.findall(file_text)

        

    elif(pattern_case2.findall(file_text)):

        match = pattern_case2.findall(file_text)

    else:

        match=''



    if(len(match)>1):

        match=match[0]

    elif(len(match)==1):

        match=match[0]

    else:

        match=''

        

    pattern_case3=re.compile(r'foreign\slanguage')

 

    #print(match[0])

    if(pattern_case3.findall(match)):

        match = pattern_case3.findall(match)

        match=match[0]

    else:

        match='NaN'

        

        

    return match
def Extract_Job_Title(file_text):

## Job Titles can occur in either of three different formats, below are the regexes for all three ##

    pattern_case1=re.compile(r'([A-Z][A-Z\s]*(ER|ATE|OR|IAN|NT|YST|IST|SMITH|NURSE|PILOT|CHIEF|AIRPORTS|EXPERT|HAND|VE|RK|IDE|NIC))',re.I)

    pattern_case2=re.compile(r'((?<=ONLY\s)[A-Z\s].*)',re.I)

    pattern_case3=re.compile(r'((CHIEF|DIRECTOR|AIRPORTS|CORRECTIONAL|ASSISTANT|ADVANCED|APPRENTICE)[\w\s\d]*$)',re.I)



    ## Discover which Job Class format is used ##

    if(pattern_case1.findall(file_text)):

        match=pattern_case1.findall(file_text)

    elif(pattern_case2.findall(file_text)):

        match=pattern_case2.findall(file_text)

    elif(pattern_case3.findall(file_text)):

        match=pattern_case3.findall(file_text)

    else:

        match='NaN'



## Clean and return the Job Title

    match=match[0]

    return match[0].strip()
def Itemize(num, item):

    num1=num+1

    line=str(num) + '. ' + str(item)

    return line
def Extract_Annual_Salaries(file_text, job_title):

    pattern_case1=re.compile(r'(?<=Department\sof\sW).*',re.I)

    pattern_case2=re.compile(r'(?<=flat-rated\sat\s\$)(\d\d\d?,\d\d\d)', re.I)

    pattern_case3=re.compile(r'flat-rated',re.I)

    pattern_case4=re.compile(r'(\d?\d\d,\d\d\d)')

    pattern_case5=re.compile(r'is\s\$(.*)\s\(')

    

    no_salary={}

    salary={}

    flat=0	

    grade=''

    job_title_array={}

    years_reqd={}

    job_reqd={}

    

    ##Case 1: Department of Water

    if(pattern_case1.findall(file_text)):

        match=pattern_case1.findall(file_text)

        #Case 1a: "is $34,000 (flat-rated)"

        if(pattern_case5.findall(file_text)):

            if(pattern_case4.findall(file_text)):

                match=pattern_case4.findall(file_text)

                flat=1 

            else:

                flat=0

        #Case 1b: The salary is flat-rated at $54,202

        elif(pattern_case2.findall(file_text)):

            match=pattern_case2.findall(file_text)  

            flat=1

        #Case 1c: The salary at the Department of Water is $43,199 to $54,010

        elif(pattern_case4.findall(file_text)):

            match=pattern_case4.findall(file_text)

            flat=0

    ##Case 2: Outside the Department of Water and Power and flat rated

    else:

    ## What if the annual salary is flat rated, use: ##

        if(pattern_case3.findall(file_text)):

            match=pattern_case3.findall(file_text)

            flat=1

         

            #flat-rated salaries

            if(pattern_case4.findall(file_text)):

                match=pattern_case4.findall(file_text)

                flat=1  

        

        elif(pattern_case4.findall(file_text)):

            match=pattern_case4.findall(file_text)

            flat=0  

    

        ## If there is an abnormal salary like ....pending then:

        else:

            no_salary[0]=40000

            match=no_salary

            flat=0

 

    ## ok, now if we have:

## one value	- take the single value as the average 

## two values   - find a value within the range

## three values - find a value within the first two values, 1 iteration

## four values - find two values wthin the two ranges, 2 iterations, then save iterations

    #print('begin')

## To create the algorithm, let's find the iterations first)

    if(len(match)):

        elements=len(match)

        if(flat==1):

            iterations=0

            salary[0]=int(match[0].replace(',',''))

        elif((elements%2)==0):

            iterations=elements/2

        else:



            iterations=(elements-1)/2

    else:

      

        iterations=0

        job_title_array[0]=job_title

        salary[0]=int(match[0].replace(',',''))

        

    iterations=int(iterations)	

    

  

    # iterate through the salaries

    if(iterations>0):

        for i in range(0, iterations):

            maxo=int(match[(i*2)+1].replace(',', ''))

            mino=int(match[i*2].replace(',', ''))

            x=maxo-mino

            offset=0.6*x

            salary[i]=mino+offset



        if(iterations==1):

            years_reqd[0]=2

            job_reqd[0]=''

        elif(iterations>1):

   

            # This will assign grades: Position I, Position II, Position III

            for j in salary:

                if(j<3):

                    grade+='I'

                elif(j==3):

                    grade='IV'

                elif(j==4):

                    grade='V'

                elif(j==5):

                    grade='VI'

                elif(j==6):

                    grade='VII'



                job_title_array[j]=job_title+' '+grade

                #2 years to earn eligibility at job with salary $xx,xxx

                if(j<len(salary)-1):

                    years_reqd[j]=2

                elif(j==len(salary)):

                    years_reqd[j]=0

                    #Job with salary $xx,xxx needs what job requirement 

              

                if(j>0):    

                    job_reqd[j]=job_title_array[j-1]

                    years_reqd[j]=2

                elif(j==0):     

                    job_reqd[j]=''

                    years_reqd[j]=0

    ##       END FOR LOOP ##

    

    if(iterations==0 or iterations==1):

        job_title_array[0]=job_title

    if(iterations==0):

        job_reqd[0]=''

        years_reqd[0]=2

    

        

## The approach is to divide the general positions into graded positions ##

## For example: An Auditor has one job application form for multiple positions ##

## these positions correspond to the salary ranges	##

## Auditor I, Auditor II, Auditor III, etc	##

###########################################################################



## In the array we will have:	###########################################	

## Job Title: Auditor I	|	Auditor II	|	Auditor III	##

## Salary	: 97888	|	105933	|	118220	##

###########################################################################



## Compute the real salary for a given profession based on a publicly-   ##

## released payroll for 2017	##

###########################################################################

    file_n='transparent_ca.csv'

    dir_path='../input/transparent-california-los-angeles'



    with open(dir_path+'/'+file_n, 'r', encoding = "ISO-8859-1") as csv_file:

        data = csv.reader(csv_file)

        header_count = 0

        found_employees={}

        total_salary={}

        payroll_data=[]

        total=0

        found=0

        rows=17

        cols=5

        count=0

## Total Salary will be the accumulator for our running salary total     ##

## Remember, we have to compute salaries for all graded poisitions	##

## If the graded positions are not found, then we must revert to the ##

## approximated value	##

        ## Graded Positions Loop: Auditor I, Auditor II, Auditor III	##

        for k in range(0, len(job_title_array)):

            header_count=0

            found=0

            total_salary[k]=1

            found_employees[k]=0

            ## Occurences in CSV File (Transparent_CA)	##

            for row in data:

                  

             ## Filter the Header

                if (header_count == 0):

                    header_count += 1

                    ## the employee data case

                elif (row[1].lower()==job_title_array[k].lower()):

                    if(row[11]=='FT'):

                        found += 1

                        other=float(row[4])

                        overtime=float(row[3])

                        regular=float(row[2])

                        total_salary[k]+=regular+overtime+other

                ###   END FOR LOOP   ###

            ## Replace the advertised salaries with the computed salaries	##

                found_employees[k]=found

        

        for i in range(0,len(salary)):

            #if total salary exists then check and replace

            if (total_salary[i]):

                if(total_salary[i]>salary[i]):

                    salary[i]=total_salary[i]/found_employees[i]

           

        ##  Number of Job Grades that will be columns number long

        for row in range(len(salary)): payroll_data += [[]*(cols)]



        

        #print(job_reqd[0])

        for j in range(0, len(salary)):

            

            payroll_data[j].append(job_title_array[j])

            payroll_data[j].append(salary[j])

            payroll_data[j].append(found_employees[j])

            payroll_data[j].append(job_reqd[j])

            payroll_data[j].append(years_reqd[j])

     

    return payroll_data            

def Extract_Qualifications(file_text, job_list):

            i=0

            s=0

            line=[]

            job_requirements=[]

            no_match={}

            rows=6

            job_reqd=[]

            cols=1

            b='\\b'

            temp=''

            tmp={}

            for row in range(rows): line += [[]*cols]

            

#######     READ IN QUALIFICATIONS AND REQUIREMENTS      #########

            pattern_years=re.compile(r'(one|two|three|four|five|six)(?=\s?-?years?)', re.I)  

            pattern_case1=re.compile(r'QUALIFICATIONS?\s\s([\S\s]*?)[A-Z]{4,5}')

            pattern_case2=re.compile(r'REQUIREMENTS?\s\s([\S\s]*?)[A-Z]{4,5}') 

            pattern_hours=re.compile(r'(?<=\d[.]\s)(\d?\d,\d\d\d)')

            pattern_hours_begin=re.compile(r'(\d?\d,\d\d\d)')

            pattern_jobs=re.compile(r'(?<=level\sof\s)(([A-Z][a-z]+\s(of|and|for)?\s?){1,4})')

            pattern_jobs2=re.compile(r'(?<=as\san\s)(([A-Z][a-z]+\s(of|and|for)?\s?){1,4})')

            pattern_jobs3=re.compile(r'(?<=as\sa\s)(([A-Z][a-z]+\s(of|and|for)?\s?){1,4})')

     

            ## Read in the Entire Section of QUALIFICATIONS/REQUIREMENTS ##

            if(pattern_case2.findall(file_text)):

                match=pattern_case2.findall(file_text)

              

            elif(pattern_case1.findall(file_text)):

                match=pattern_case1.findall(file_text)

            else:

                no_match[0]='a'

                match=no_match

               

            #print(match[0])

            match_splitter=re.compile(r'([\d][.][,\w\d\s;][-\w\d,\'\s(]+)')   ## In the form 1. Requirement 1

            segments=match_splitter.findall(match[0]) 

            size=len(segments)

            #print(segments)

            #print(size)

            

            ##  REQUIREMENT SEGMENTS

            ## Split the entire section into Segments by Number

            ## Case #1: There is more than one requirement

            if(size>1):

                  ##only grab the first group

                ## Case #1: 1. Maintenance of complex electrical systems ...

                ## Case #2: 2. Development of new processes to address ... (ignored as '')

                line[2]=segments[0].rstrip()

                line[5]=''

                for i in range(0,len(segments)-1):

                        line[5]+=segments[i+1].rstrip()

            ## Case #2: If there is only one requirement

            else:    

                #print(match)

                line[2]=match[0].rstrip()

   



            ## The variable segments is either equal to 0 or a Natural Number (1,2,3,etc.)

    

            ## Find the referenced job titles and years of experience in the requirements ##

            ## Execute all three searches below ##

                        ## I. First the Job Title Search



                        ## check to see the current segment matches a saved job from job list

            ## job_list[]       -   segment[0]

            ## ---------------------------------------------------------------------

            ## AIRPORT GUIDE I  -   Two years as a Couselor III with the City of ...

            ## COUNSELOR III

            ## ART DIRECTOR

            ## etc

            

            ## TIME DURATIONS

            ## Case 1: There is only one requirement

            if (size==0): 

                match=match[0]

                if(pattern_years.findall(match)):

                    #if(pattern_jobs.findall(match)):

                     #   job_reqd=pattern_jobs.findall(match)

                    #elif(pattern_jobs2.findall(match)):

                     #   job_reqd=pattern_jobs.findall(match)

                    #elif(pattern_jobs3.findall(match)):

                     #   job_reqd=pattern_jobs.findall(match)

                        

                    match_time=pattern_years.findall(match)

                    match_time=Word_to_Number(match_time) 

                elif(pattern_hours.findall(match)):

                    #if(pattern_jobs.findall(match)):

                     #   job_reqd=pattern_jobs.findall(match)

                    #elif(pattern_jobs2.findall(match)):

                     #   job_reqd=pattern_jobs.findall(match)

                    #elif(pattern_jobs3.findall(match)):

                     #   job_reqd=pattern_jobs.findall(match)

                        

                    match_time=pattern_hours.findall(match)

                    match_replace=[item.replace(",", "") for item in match_time]

                    match_time = int(match_replace[0])

                    match_time=match_time/2080  ## There are 2080 hours in a year

            

            job_reqd=''        

            ## Case 2: There is more than one time requirement

            for t in range(0,size):

                

                if(pattern_years.findall(segments[t])):

                    ## Set the Jobs Requirement for this requirement

                    if(pattern_jobs.findall(segments[t])):

                        job_reqd=pattern_jobs.findall(segments[t])

                    elif(pattern_jobs2.findall(segments[t])):

                        job_reqd=pattern_jobs.findall(segments[t])

                    elif(pattern_jobs3.findall(segments[t])):

                        job_reqd=pattern_jobs.findall(segments[t])

                        

                    match_time=pattern_years.findall(segments[t])

                    match_time=Word_to_Number(match_time[0])     

                elif(pattern_hours.findall(segments[t]) ):

                    if(pattern_jobs.findall(segments[t])):

                        job_reqd=pattern_jobs.findall(segments[t])

                    elif(pattern_jobs2.findall(segments[t])):

                        job_reqd=pattern_jobs.findall(segments[t])

                    elif(pattern_jobs3.findall(segments[t])):

                        job_reqd=pattern_jobs.findall(segments[t])

                        

                    match_time=pattern_hours.findall(segments[t])

                    match_replace=[item.replace(",", "") for item in match_time]

                    match_time = int(match_replace[0])

                    #print(match_time)

                    match_time=match_time/2080  ## There are 2080 hours in a year 

                    #print(match_time)

                else:

                    match_time=0

                    

                if(len(job_reqd)):

                    job_reqd=job_reqd[0]

                    #print(len(job_reqd))

                    #print(job_reqd[0])

            

                        #We now have the Required Number of Years

                        #Jobs_Required and the Time Required

               



                        

                        # Store the Job Title and the Number of Years #

                        ##

                        ##  Actuary III (0,0)  |   Garage Mechanic IV  (0,1)

                        ##      3     (1,0)    |     8                (1,1)

                        ###############################################

                        #if(s==0):

                           # line[0][0]=job_reqd

                            #line[1][0]=match_time

                        #elif(s>0):

                if(t==0):

                    line[0].append(job_reqd)

                    line[1].append(match_time)

                            

                elif(t>0):

                    line[3].append(job_reqd)    

                    line[4].append(match_time)

              

                #print(segments)

                    ######   END FOUND JOB SECTION   #######             

                       

            s+=1

               

                ######  END JOB LIST ITERATION SECTION   ######    

           

            ########  END ITERATE THROUGH REQUIREMENTS SECTIONS

            #print(line)

            return line

dir_path = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA/Job Bulletins/'

t=0

header=['JOB_TITLE','AVG_SALARY','FOUND','EXP_JOB_CLASS_TITLE', 'YEARS_REQ_PRIMARY', 

        'REQUIREMENT_SET_ID', 'EXP_JOB_CLASS_ALT_RESP','YEARS_REQ_SECONDARY', 'REQUIREMENT_SUBSET_ID',

        'COURSE_LENGTH', 'OPEN_DATE', 'JOB_CLASS_NO', 'DRIVERS_LIC_REQ', 'DRIV_LIC_TYPE',

        'ADDTL_LIC', 'COURSE_SUBJECT', 'JOB_DUTIES', 'SCHOOL_TYPE', 'Other Certs', 'EXAM_TYPE', 'FILE_NAME', 'Exam Distro',

        'ENTRY_SALARY_GEN', 'HOURLY_WAGES','EXP_JOB_CLASS_FUNCTION', 'LANGUAGE', 'EDUCATION_MAJOR', 'ENTRY_SALARY_DWP',

       'COURSE_COUNT', 'EDUCATION_YEARS', 'JOB_CLASS_TITLE']



##Job Title - conforms the grade of Job Class to payroll information

##4##AVG_SALARY - The average salary computed for each position calculated from TransparentCalifornia

##8##ADDTL-LIC - The Board from which a license needs to be approved

##13#Other Certs - n/a

##21#JOB_TITLE - The actual jobs to which people apply

##27#

#####



partial_list=[]

annuals=[]

all_jobs=[]

board_certs=[]

cols=5

header_cols=30

job_list=[]

language=[]


for file_name in os.listdir(dir_path):



    with open(dir_path+'/'+file_name, 'r', encoding = "ISO-8859-1") as f :

            file_text=f.read()



        ## Extract Job Title will expand each Job Category 

        ## For example: [0] Airport Guide - [1] Auditor - [2] Barrister

            job_title=Extract_Job_Title(file_text) ##ok

            

            ## Filter out the "VOCATIONAL WORKER" Job Bulletin

            if(len(job_title)<43):

                ## Extract Annual Salaries will create all Job Positions beyond the Job Bulletins

                ## Job Class - Airport Guide - provided in the job bulletin 

                ## Extract Annual Salaries - [0] - Airport Guide I, Airport Guide II, Airport Guide III ##

                ##+++++++++++++ also save the ranges



                annuals=Extract_Annual_Salaries(file_text, job_title)# class_code, open_date##ok

                #for row in range(len(annuals)): partial_list +=[[]*cols] 

                #len(annuals)= the number of job grades under each job category

                #e.g. a Street Lighting Engineer may have len(annuals)=3

                # Street Lighting Engineer I, Street Lighting Engineer II, Street Lighting Engineer III

                if(len(annuals)):

                    if(t==0):

                        #partial_list is our target list which is fully contructed with no values

                        

                        #Initialize - Add len(annuals)+1 rows - Below is how our list looks:

                        #

                        #|x  |x  |x  |__ |x  |

                        #|x  |x  |x  |x  |x  |

                        #................

                        #|x  |x  |x  |x  |__ |

                        for row in range(len(annuals)+1):

                            partial_list +=[[]*cols]

                        #Add columns to the header row

                        for i in range(header_cols):

                            #print(i)

                            partial_list[0].append(header[i])

                        #Add columns to the 1st Job Position row

                        for row in range(len(annuals)):

                            for k in range(cols):

                                partial_list[row+1].append(annuals[row][k])

                    else:

                        size=len(partial_list)

                        for row in range(len(annuals)):partial_list+=[[]*cols]

                        #Now fill the rows in partial_list

                        for row in range(len(annuals)):

                            for j in range(cols):

                                partial_list[row+size].append(annuals[row][j])

                               

                t+=1

     

                ## Extract the general qualifications for each Job Category

                qualifications=Extract_Qualifications(file_text, job_list) ##full_list[0] = all the jobs

                #Index [end-addition] is the beginning of the Job Category

                #Index [end] is the end of the Job Category

                end=len(partial_list)

                addition=len(annuals)

                

                beginning=end-addition

                

                ## Add more columns

                ## How many times should we dessiminate the qualifications?

                ## We should dessiminate to the last end-addition list spots

                

                

                for i in range(addition):

                    if (i==0):

                        partial_list[beginning][3]=(qualifications[0]) ##jobs required - primary

                        for k in range(4):

                            partial_list[beginning].append('n/a')

                        partial_list[beginning][6]=(qualifications[3]) ##jobs required - secondary

                    elif ((i<addition-1) and i>0):

                        for k in range(4):

                            partial_list[beginning+i].append('n/a')

        

                        partial_list[beginning+i][6]=(qualifications[3]) ##jobs required - secondary

                        partial_list[beginning+i][7]=(qualifications[4]) ##years required - secondary                      

                    elif(i==addition-1):

                        for k in range(4):

                            partial_list[beginning+i].append('n/a')

                            

                        partial_list[beginning+i][7]=(qualifications[4]) ##years required - secondary

                        partial_list[beginning+i][4]=(qualifications[1]) ##years required - primary

                        partial_list[beginning+i][6]=(qualifications[3]) ##jobs required - secondary

                    for v in range(23): #(16) General Skills equals [26]

                        partial_list[beginning+i].append(0)

                    

                    partial_list[beginning+i][5]=(qualifications[2]) ##requiremnets (segment) - primary

                    partial_list[beginning+i][8]=(qualifications[5]) ##requirements required - secondary

                    partial_list[beginning+i][9]=(Extract_Course_Length(file_text))

                    partial_list[beginning+i][10]=(Extract_Open_Date(file_text))

                    partial_list[beginning+i][11]=(Extract_Class_Code(file_text))

                    partial_list[beginning+i][12]=(If_Drivers_License(file_text))

                    partial_list[beginning+i][13]=(Extract_Drivers_License_Type(file_text))

                    partial_list[beginning+i][14]=(Extract_Board_Certs(file_text))

                    ##partial_list[beginning+i][15]=(Extract_Course_List(file_text))

                    partial_list[beginning+i][16]=(Extract_Duties(file_text))

                    partial_list[beginning+i][17]=(Extract_School_Type(file_text))

                    ##partial_list[beginning+i][18]=(Extract_Other_Certs(file_text))

                    partial_list[beginning+i][19]=(Extract_Exam_Type(file_text))

                    #print(beginning)

                    partial_list[beginning+i][20]=file_name

                    #print(partial_list[beginning+i][20])

                    partial_list[beginning+i][21]=(Extract_Exam_Distro(file_text))

                    partial_list[beginning+i][22]=(Extract_Salary(file_text))

                    partial_list[beginning+i][23]=('$'+str(partial_list[beginning+i][1]/2080))   #WAGES

                    partial_list[beginning+i][24]=(Extract_General_Skills(file_text))

                    partial_list[beginning+i][25]=(Extract_Language(file_text)) 

                    #28

                    partial_list[beginning+i][26]=Extract_Major(file_text)

                    partial_list[beginning+i][27]=Extract_DWP_Salary(file_text)

                    partial_list[beginning+i][28]=Count_Commas(Extract_Major(file_text))

                    partial_list[beginning+i][29]=Extract_Education_Years(file_text)

                    partial_list[beginning+i][30]=Extract_Job_Class(job_title)

                    
##Extract all the job titles

for t in range(1,len(partial_list)):

    all_jobs.append(partial_list[t][0])

print(all_jobs[0:24])

##Extract all the job titles

#for t in range(1,len(partial_list)):

for t in range(1,len(partial_list)):

    if (len(partial_list[t][14])>20):

        print(partial_list[t][14])

        
full_list=partial_list

print(full_list[:][75])

  
print(full_list[:][571])
#df=pd.DataFrame(full_list)

brics = pd.DataFrame(full_list)

brics.to_csv('output.csv', sep=',', encoding='utf-8')



print(brics)     


size=range(len(partial_list))

rnum=choice(size)

file_name=full_list[11][20]



with open(dir_path+'/'+file_name, 'r', encoding = "ISO-8859-1") as f :

            file_text=f.read()

        

EEOC=(Extract_EEOC(file_text)  )

print(EEOC)