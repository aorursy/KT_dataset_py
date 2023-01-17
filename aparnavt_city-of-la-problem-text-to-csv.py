# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import csv

import re



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# print(os.listdir("../input/cityofla/CityofLA/Job Bulletins"))



# Defining dictionary of the REGEX for each category. Still pending to extract salaries for Department of Water and Power.



rx_dict = {

    'TITLE': re.compile(r'(?P<title>.*)(\n+.*Class Code:)'),

    'Class Code': re.compile(r'(Class Code:)(?P<classcode>.*)(\n)'),

    'Open Date': re.compile(r'(Open Date:)(?P<open_date>.*)(\n)'),

    'Annual Salary': re.compile(r'ANNUAL\s*SALARY\s*\n*\s*(?P<Lowerend>\$\d*,\d*)(\s*to\s*(?P<Upperend>\$\d*,\d*)|.*)'),

    'Salary DWP': re.compile(r'ANNUAL\s*SALARY(.|\n)*?.*Department of Water and Power.*(?P<Lowerend>\$\d*,\d*)(\s*to\s*(?P<Upperend>\$\d*,\d*))'),

    'DUTIES': re.compile(r'(DUTIES(?P<Duties>(.|\n)*?))(SELECTION PROCESS|APPLICATION DEADLINE|WHERE TO APPLY|MINIMUM\s*QUALIFICATIONS|QUALIFICATIONS|REQUIREMENTS|QUALIFICATION|REQUIREMENT|\Z)'),

    'REQUIREMENTS': re.compile(r'((REQUIREMENT|REQUIREMENT/MINIMUM QUALIFICATION|QUALIFICATION|QUALIFICATIONS)(?P<Requirement>(.|\n)*?))(SELECTION PROCESS|APPLICATION DEADLINE|WHERE TO APPLY|\Z)'),

    'WHERE TO APPLY': re.compile(r'(WHERE TO APPLY(?P<WheretoApply>(.|\n)*?))(SELECTION PROCESS|APPLICATION DEADLINE|\Z)'),

    'APPLICATION DEADLINE': re.compile(r'(APPLICATION DEADLINE(?P<ApplicationDeadline>(.|\n)*?))(SELECTION PROCESS|\Z)'),

    'SELECTION PROCESS': re.compile(r'(SELECTION PROCESS(?P<SelectionProcess>(.|\n)*?))(\Z)'),

    'ALL':re.compile(r'(DUTIES(?P<Duties>(.|\n)*))((REQUIREMENTS|REQUIREMENTS\\MINIMUM QUALIFICATIONS|QUALIFICATIONS|REQUIREMENT|QUALIFICATION)(?P<Requirements>(.|\n)*))(WHERE TO APPLY(?P<Wheretoapply>(.|\n)*))(APPLICATION DEADLINE(?P<ApplicationDeadline>(.|\n)*))(SELECTION PROCESS(?P<SelectionProcess>(.|\n)*))')

}

                             

                            



       

 

def _parse_file(filepath,filename):

    data = []  # create an empty list to collect the data

    # open the file and read through it line by line

    Salary_lower_gen = ""

    Salary_upper_gen = ""

    Salary_lower_DWP = ""

    Salary_upper_DWP = ""

    Title = ""

    Title_final = ""

    Class_code = ""

    Class_code_final = ""

    Duties = ""

    Duties1 = ""

    Duties_final = ""

    Requirements = ""

    Requirements1 = ""

    Requirement_final =""

    Where_to_apply = ""

    Where_to_apply1 = ""

    Where_to_apply_final = ""

    Application_deadline = ""

    Application_deadline1 = ""

    Application_deadline_final = ""

    Selection_process = ""

    Selection_process1= ""

    Selection_process_final = ""

    

    with open(filepath, 'r', encoding="latin-1") as file_object:

        line = file_object.read()

        for key, rx in rx_dict.items():

            match = rx.search(line)

            if match:

                #print("Enter")

                if key == 'TITLE':

                    Title = "" if not match.group('title') else match.group('title').strip() 

                    #print(Title)

           

                if key == 'Class Code':

                    Class_code = "" if not match.group('classcode') else match.group('classcode').strip() 

                    #print(Class_code)

                

            

                if key == 'Open Date':

                    Open_date = "" if not match.group('open_date') else match.group('open_date').strip()

                    #print(Open_date)

                    

                if key == 'Annual Salary':

                    Salary_lower_gen = "" if not match.group('Lowerend') else match.group('Lowerend').strip()

                    Salary_upper_gen = "" if not match.group('Upperend') else match.group('Upperend').strip()

                    #print(Salary_lower_gen, Salary_upper_gen)

                if key == 'Salary DWP':

                    Salary_lower_DWP = "" if not match.group('Lowerend') else match.group('Lowerend').strip()

                    Salary_upper_DWP = "" if not match.group('Upperend') else match.group('Upperend').strip()

                    

                

                #if key == 'DUTIES':

                    #Duties = 'default' if not match.group('Duties') else match.group('Duties')

                    #print(Salary_lower_gen, Salary_upper_gen)

                if key == 'ALL':

                    Duties = "" if not match.group('Duties') else match.group('Duties')

                    Requirements = "" if not match.group('Requirements') else match.group('Requirements')

                    Where_to_apply = "" if not match.group('Wheretoapply') else match.group('Wheretoapply')

                    Application_deadline = "" if not match.group('ApplicationDeadline') else match.group('ApplicationDeadline')

                    Selection_process = "" if not match.group('SelectionProcess') else match.group('SelectionProcess')

                    #print(Salary_lower_gen, Salary_upper_gen)   

                if key == 'DUTIES':

                    Duties1 = "" if not match.group('Duties') else match.group('Duties')

                if key == 'REQUIREMENTS':

                    Requirements1 = "" if not match.group('Requirement') else match.group('Requirement')

                if key == 'WHERE TO APPLY':

                    Where_to_apply1 = "" if not match.group('WheretoApply') else match.group('WheretoApply')

                if key == 'APPLICATION DEADLINE':

                    Application_deadline1 = "" if not match.group('ApplicationDeadline') else match.group('ApplicationDeadline')

                if key == 'SELECTION PROCESS':

                    Selection_process1 = "" if not match.group('SelectionProcess') else match.group('SelectionProcess')

           

    # Extract each field using two different regexes and obtain the best possible result.

    Duties_final = Duties1.strip() if not Duties else Duties.strip()                

    Requirements_final = Requirements1.strip() if not Requirements else Requirements.strip()

    Where_to_apply_final = Where_to_apply1.strip() if not Where_to_apply else Where_to_apply.strip()

    Application_deadline_final = Application_deadline1.strip() if not Application_deadline else Application_deadline.strip()

    Selection_process_final = Selection_process1.strip() if not Selection_process else Selection_process.strip()

    Title_final = re.split('[0-9]',filename)[0].strip() if not Title else Title

    #print(re.split('[0-9]',filename)[0].strip())

    try:

        Class_code_final = re.split('(\s\d\d\d\d\s)',filename)[1].strip() if not Class_code else Class_code

    except: 

        Class_code_final =  Class_code

    row = {

                'FILENAME': filename,

                'JOB_CLASS_TITLE': Title_final,

                'JOB_CLASS_NO': Class_code_final,

                'ENTRY_SALARY_GEN': Salary_lower_gen,

                'HIGHEST_SALARY_GEN': Salary_upper_gen,

                'ENTRY_SALARY_DWP': Salary_lower_DWP,

                'HIGHEST_SALARY_DWP': Salary_upper_DWP,

                'DUTIES': Duties_final,

                'REQUIREMENTS': Requirements_final,

                'WHERE_TO_APPLY': Where_to_apply_final,

                'APPLICATION_DEADLINE': Application_deadline_final,

                'SELECTION_PROCESS': Selection_process_final

              }



    data.append(row)

    return data
#for file in JD_Filename:

filepath = "../input/cityofla/CityofLA/Job Bulletins/" 

df = pd.DataFrame(columns = ['FILENAME','JOB_CLASS_TITLE','JOB_CLASS_NO', 'ENTRY_SALARY_GEN' , 'HIGHEST_SALARY_GEN', 'ENTRY_SALARY_DWP','HIGHEST_SALARY_DWP','DUTIES','REQUIREMENTS',                'WHERE_TO_APPLY',

                'APPLICATION_DEADLINE',

                'SELECTION_PROCESS'])



#Parsing through all the files and calling the parser function

for file in os.listdir(filepath):

    #print(file)

    data =_parse_file(filepath + file,file)

    #print(data)

    df = df.append(data, ignore_index = True)

#Writing output to CSV

output = df

output.to_csv('submission.csv', index=False)