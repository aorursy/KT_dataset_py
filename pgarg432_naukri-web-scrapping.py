# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy
import pandas as pd
from bs4 import BeautifulSoup
import requests
import csv
from datetime import datetime
headers = {"User-Agent":"Mozilla/5.0"} 
for i in range(1, 10): #scrap data of 10 pages
    page = requests.get("https://www.naukri.com/html-jobs-in-delhi--%s" %i, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    data = soup.find_all("div",{"itemtype":"http://schema.org/JobPosting"}) #get all job divs
    item = data[0] 
    alldata = []   
    desig = [] #Job title
    org =[] #Recruiter
    exp = [] #Work Exp required
    loc = [] #Job Location
    skill = [] #Skill(s) required for the job opening
    salary = [] #Salary offered
    date = [] #Date of post
    store = [] #time of post
    
    for item in data:
        try:
            desig = item.find(class_="desig").get_text()
            str(desig + ",")
            org = item.find(class_="org").get_text()	
            str(org + ",")	
            exp = item.find(class_="exp").get_text()
            str(exp + ",")
            loc = item.find(class_="loc").get_text()
            str(loc + ",")
        except:
            pass
        try:
            skill = item.find(class_="skill").get_text()
            str(skill + ",")
        except:
            pass
        try:
            salary = item.find(class_="salary").get_text()
            str(salary + ",")
        except:
            pass
        try:
            date = item.find(class_="date").get_text()
            str(date)
        except:
            pass

        for alldata in  range (1):
            with open('#file_name.csv', 'a') as csv_file:  
                writer = csv.writer(csv_file)
                writer.writerow([desig, org,exp,loc,skill,salary,date, datetime.now()])