import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

import glob

from datetime import datetime

import re
input_path="../input/cityofla/CityofLA/"
job_bulletein_path=input_path+"Job Bulletins/"

job_bulletein_path
files=glob.glob(job_bulletein_path+"/*.txt")

print("Number of files in Job Bulletein path ",len(files))
data=pd.DataFrame()

job_class_title=[]

job_class_number=[]

open_date=[]

revised_date=[]

file_name=[]



for __file__ in files:



    file_name.append(__file__.replace(job_bulletein_path,""))

    jp=__file__.replace(job_bulletein_path,"").upper()

    jp=jp.replace("DRAFT.TXT","")

    jp=jp.replace(".TXT","")

    jp=re.sub(r"\d+","",jp)

    jp=jp.replace("REVISED","")

    jp=jp.replace("REV","")

    jp=jp.replace("()","").strip()

    jp=jp.replace("BULLETIN FINAL","")

    

    

    job_class_title.append(jp)

    

    ## Get job_class_number - it is either length 3 or length 4

    

    class_regex = ' \d{3,4} '

    class_num = re.findall(class_regex, __file__)

    #print(class_num)

    job_class_number.append(class_num)



    ### Get open date and Revised Date

    

    date_regex='\d{5,6}'

    dates_search=re.findall(date_regex,__file__)

    if len(dates_search)>1:

        revised_date.append(dates_search[1])

    else:

        revised_date.append("")

    

    if len(dates_search)>0:

        open_date.append(dates_search[0])

    else:

        open_date.append("")
data['JOB_CLASS_TITLE']=job_class_title

data['JOB_CLASS_NO']=job_class_number

data['FILE_NAME']=file_name

data['OPEN_DATE']=open_date

data['REVISED_DATE']=revised_date
data.to_excel("JobBulleteins_version1.xlsx")