!pip install datefinder


import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import re,glob

import datefinder

print(os.listdir("../input"))

job_path = "../input/cityofla/CityofLA/Job Bulletins/"

job_files = os.listdir(job_path)

print("No of files in Job Bulletins Folder ",len(job_files))
from wand.image import Image as Img

pdf = '../input/cityofla/CityofLA/Additional data/PDFs/2014/April 2014/040414/PORT POLICE SERGEANT 3222.pdf'

Img(filename=pdf, resolution=200)
with open(job_path+job_files[0]) as file:

    print("File name: ",file.name)

    print("******************")

    print(file.read())
requirment_tag = ['REQUIREMENTS/MINIMUM QUALIFICATIONS', 'REQUIREMENTS/ MINIMUM QUALIFICATIONS', 'REQUIREMENTS/MINUMUM QUALIFICATIONS', 

             'REQUIREMENT/MINIMUM QUALIFICATION', 'REQUIREMENT/ MINIMUM QUALIFICATION', 'REQUIREMENTS', 'REQUIREMENT']

with open(job_path+job_files[0]) as file:

    

    text=file.read()

    

    for i in requirment_tag:

        if i in text:

            text = text.replace('\n\n','~')

            text = text.replace(i,"Requirment")

#             break

            

            

    title          = text.split("Class Code:")[0].strip()

    

    class_code     = text.split("Class Code:")[1].split("Open Date")[0].strip()

    

    open_date      = text.split("Open Date:")[1].split("(")[0].strip()

    open_date      = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{2})", open_date).group()

    

    annual_salary  = text.split("ANNUAL SALARY")[1].split("NOTES")[0].strip()

    annual_salary  = re .split(r'\r\n|\r|\n', annual_salary, flags=re.M)[0].strip()

    

    find_requirment     = re.search(r'Requirment~(.*?)~', text, flags=re.S | re.M)

    if find_requirment:

        requirements   = find_requirment.group(1)

    else:

        requirements= ""

    

    where_to_apply     = text.split("WHERE TO APPLY")[1].split("NOTE:")[0].strip()

    

    application_deadine = " ".join(text.split("APPLICATION DEADLINE")[1].split("~")[:2])

    for i in datefinder.find_dates(application_deadine):

        application_deadine = i

    

    duties = " ".join(text.split("DUTIES")[1].split("~")[:2])

    

    

    print("Title: ", title)

    print("***********************************************************************")

    print("Class code: ", class_code)

    print("***********************************************************************")

    print("Open date: ",open_date)

    print("***********************************************************************")

    print("Annual salary: ",annual_salary)

    print("***********************************************************************")

    print("Requirment: ", requirements)

    print("***********************************************************************")

    print("Where to Apply: ", where_to_apply)

    print("***********************************************************************")

    print("Application DeadLine: ", application_deadine)

    print("***********************************************************************")

    print("Duties: ",duties)

    print("***********************************************************************")
data=[]

for i in job_files:

    with open(job_path+i) as file:

        filename=str(file.name).split("/")[-1]

        try:

            text=file.read()

            for i in requirment_tag:

                if i in text:

                    text = text.replace('\n\n','~')

                    text = text.replace(i,"Requirment")

            title      = text.split("Class Code:")[0].strip()

            

            class_code = text.split("Class Code:")[1].split("Open Date")[0].strip()

            

            open_date  = text.split("Open Date:")[1].split("(")[0].strip()

            open_date  = re.search("([0-9]{2}\-[0-9]{2}\-[0-9]{2})", open_date).group()

            

            annual_salary  = text.split("ANNUAL SALARY")[1].split("NOTES")[0].strip()

            annual_salary  =  re .split(r'\r\n|\r|\n', annual_salary, flags=re.M)[0].strip()

            

            

            find_requirment     = re.search(r'Requirment~(.*?)~', text, flags=re.S | re.M)

            if find_requirment:

                requirement   = find_requirment.group(1)

            else:

                requirement  = ""



            where_to_apply  = text.split("WHERE TO APPLY")[1].split("NOTE:")[0].strip()

            

            application_deadine = " ".join(text.split("APPLICATION DEADLINE")[1].split("~")[:5])

            for i in datefinder.find_dates(application_deadine):

                application_deadine = str(i)

#                 print(application_deadine)

            

            duties = " ".join(text.split("DUTIES")[1].split("~")[:2])

            

            data_list=[]

            for i in [filename,title,class_code,open_date,annual_salary,requirement,where_to_apply,application_deadine,duties]:

                data_list.append(i.replace("~",""))

            data.append(data_list)

            

        except Exception as e:

            print(filename)



df = pd.DataFrame(data)

df.columns = ["File_name", "Title", "Class_code","Open_date","Annual_salary","Requirment","Where_to_Apply",

              "Application_deadine","Duties"]

df.drop(['File_name'],inplace=True,axis=1)

df.head()
df.shape
df.to_csv("job_bulletins.csv")
job_df = pd.read_csv("job_bulletins.csv")

job_df.drop(['Unnamed: 0'],inplace=True,axis=1)

job_df.to_csv("job_bulletins.csv")

job_df.head()
job_df.columns