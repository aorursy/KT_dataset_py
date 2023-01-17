from wand.image import Image as Img

Img(filename='../input/cityofla/CityofLA/Additional data/PDFs/2018/September/Sept 28/ARTS MANAGER 2455 092818.pdf', 

    resolution=250)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns



import re #Regular Expressions

import nltk#Natural Language

from nltk.corpus import stopwords



from collections  import Counter

from nltk import word_tokenize



from sklearn.preprocessing import Imputer



# Standard plotly imports

import plotly.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)





import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
JobBulletinsDir = "../input/cityofla/CityofLA/Job Bulletins"

Data = []

for file in os.listdir(JobBulletinsDir):

  FileName = file  

  with open (JobBulletinsDir + '/' + file, encoding = "ISO-8859-1") as f:

   FileContent = f.read()

   Data.append([FileName, FileContent]) 

   

    
df = pd.DataFrame(Data)

df.columns = ["FILE_NAME", "FILE_CONTENT"]

def OpenDate(text):

  OpenDate = ""

  start = 0

  for line in text.strip().split("\n"):

    #Check if Open Date is available in file

    if "Open Date" in line or "Open date" in line:

      start = 1

    elif start and line.isupper():

      start = 0

    #If Open date is available in file

    if start:

      OpenDate += line + "\n"

  OpenDate = OpenDate.replace("Open Date","").replace("Open date","").replace(":","").strip().split("\n")[0] 

  return OpenDate[0:8]

#Assign Open Date to Data Frame

df['OPEN_DATE']=df['FILE_CONTENT'].apply(lambda x : OpenDate(x))  

def AnnualSalary(text):

  AnnualSalary = ""

  start = 0

  for line in text.strip().split("\n"):

    if "ANNUAL SALARY" in line or "ANNUALSALARY" in line:

      start = 1

    elif start and line.isupper():

      start = 0

    #If Annual Salary is available in file

    if start:

      AnnualSalary += line + "\n"

  AnnualSalary = AnnualSalary.replace("ANNUAL SALARY","").replace("ANNUALSALARY","").strip().split("\n")[0]  

  #If multiple salaries are available

  FinalSalary = []

  SalarySplit = AnnualSalary.split()

  for i, v in enumerate(SalarySplit):

    if v.lower() == "to":

      salary = SalarySplit[i-1] + " to " + SalarySplit[i+1]

      FinalSalary.append(salary)

  if len(FinalSalary) == 0:

    salary = AnnualSalary.split("and")

    FinalSalary.extend(salary)

  return FinalSalary

#Assign Annual Salary to Dataframe

df['ANNUAL_SALARY']=df['FILE_CONTENT'].apply(lambda x : AnnualSalary(x))  

def ClassTitle(text):

    return text.strip().split("\n")[0].split("\t")[0]

#Assign Class Title to Dataframe

df["CLASS_TITLE"] = df["FILE_CONTENT"].apply(lambda x: ClassTitle(x))

def ClassCode(text):

  for line in text.strip().split("\n"):

    if "Class Code:" in line:

      return line.replace("Class Code:","").strip().split("\t")[0]

    elif "Class  Code:" in line:

      return line.replace("Class  Code:","").strip().split("\t")[0]

#Assign Class Code to Dataframe

df["CLASS_CODE"] = df["FILE_CONTENT"].apply(lambda x: ClassCode(x))



def Duties(text):

  Duties = ""

  start = 0

  for line in text.strip().split("\n"):

    if "DUTIES" in line:

      start = 1

    elif start and line.isupper():

      start = 0

    #If Duties present in file

    if start:

      Duties += line + "\n"

  Duties = Duties.replace("DUTIES","").strip().split("\n")[0] 

  return Duties

#Assign Duties to Dataframe

df['DUTIES']=df['FILE_CONTENT'].apply(lambda x : Duties(x))  

def Requirements(text):

  Requirements = ""

  start = 0

  for line in text.strip().split("\n"):

    if "REQUIREMENTS" in line or "REQUIREMENT" in line:

      start = 1

    elif start and line.isupper():

      start = 0

    #If Requirements present in file

    if start:

      Requirements += line + "\n"

  Requirements = Requirements.replace("REQUIREMENTS","").replace("REQUIREMENT","").replace("/MINIMUM QUALIFICATIONS","").replace("/MINIMUM QUALIFICATION","").strip().split("\n")[0]

  return Requirements

#Assign Requirements to Dataframe

df['REQUIREMENTS']=df['FILE_CONTENT'].apply(lambda x : Requirements(x))  

def Notes(text):

  Notes = ""

  start = 0

  for line in text.strip().split("\n"):

    if "NOTES" in line:

      start = 1

    elif start and line.isupper():

      start = 0

    #If Notes present in file

    if start:

      Notes += line + "\n"

  Notes = Notes.replace("NOTES","").replace(":","").strip().split("\n")[0] 

  return Notes

#Assign Notes to Dataframe       

df['NOTES']=df['FILE_CONTENT'].apply(lambda x : Notes(x))  

def ProcessNotes(text):

  ProcessNotes = ""

  start = 0

  for line in text.strip().split("\n"):

    if "PROCESS NOTES" in line:

      start = 1

    elif start and line.isupper():

      start = 0

    #If Process Notes present in file

    if start:

      ProcessNotes += line + "\n"

  ProcessNotes = ProcessNotes.replace("PROCESS NOTES","").replace(":","").strip().split("\n")[0] 

  return ProcessNotes

#Assign Process Notes to Dataframe        

df['PROCESS_NOTES']=df['FILE_CONTENT'].apply(lambda x : ProcessNotes(x))  

def WhereToApply(text):

  WhereToApply = ""

  start = 0

  for line in text.strip().split("\n"):

    if "WHERE TO APPLY" in line:

      start = 1

    elif start and line.isupper():

      start = 0

    #If Where To Apply present in file

    if start:

      WhereToApply += line + "\n"

  WhereToApply = WhereToApply.replace("WHERE TO APPLY","").replace(":","").strip().split("\n")[0] 

  return WhereToApply

#Assign Where To Apply to Dataframe        

df['WHERE_TO_APPLY']=df['FILE_CONTENT'].apply(lambda x : WhereToApply(x))  

def Note(text):

  Note = ""

  start = 0

  for line in text.strip().split("\n"):

    if "NOTE" in line and "NOTES" not in line:

      start = 1

    elif start and line.isupper():

      start = 0

    #If Note present in file

    if start:

      Note += line + "\n"

  Note = Note.replace("NOTE","").replace(":","").strip().split("\n")[0] 

  return Note

#Assign Note to DataFrame      

df['NOTE']=df['FILE_CONTENT'].apply(lambda x : Note(x))  

def ApplicationDeadline(text):

  ApplicationDeadline = ""

  start = 0

  for line in text.strip().split("\n"):

    if "APPLICATION DEADLINE" in line:

      start = 1

    elif start and line.isupper():

      start = 0

    #If Application Deadline is present in file

    if start:

      ApplicationDeadline += line + "\n"

  ApplicationDeadline = ApplicationDeadline.replace("APPLICATION DEADLINE","").replace(":","").strip().split("\n")[0] 

  return ApplicationDeadline

#Assign Application Deadline to Dataframe     

df['APPLICATION_DEADLINE']=df['FILE_CONTENT'].apply(lambda x : ApplicationDeadline(x))  

def SelectionProcess(text):

  SelectionProcess = ""

  start = 0

  for line in text.strip().split("\n"):

    if "SELECTION PROCESS" in line:

      start = 1

    elif start and line.isupper():

      start = 0

    #If Selection process present in file

    if start:

      SelectionProcess += line + "\n"

  SelectionProcess = SelectionProcess.replace("SELECTION PROCESS","").replace(":","").strip().split("\n")[0] 

  return SelectionProcess

#Assign Selection Process to Dataframe       

df['SELECTION_PROCESS']=df['FILE_CONTENT'].apply(lambda x : SelectionProcess(x))  

def Notice(text):

  Notice = ""

  start = 0

  for line in text.strip().split("\n"):

    if "Notice" in line:

      start = 1

    elif start and line.isupper():

      start = 0

    #If Notice present in file

    if start:

      Notice += line + "\n"

  Notice = Notice.replace("Notice","").replace(":","").strip().split("\n")[0] 

  return Notice

#Assign Notice to Dataframe       

df['NOTICE']=df['FILE_CONTENT'].apply(lambda x : Notice(x))  

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(df['DUTIES'][0])

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
#Get Start Salary from Annual Salary

START_SALARY=[]

for i in range(len(df)):

  try:

    START_SALARY.append(int(re.findall('\d+',df['ANNUAL_SALARY'][i][0].replace(",",""))[0]))

  except:

    #If Annual Salary is flat

    try:

      START_SALARY.append(int(re.findall('\d+',df['ANNUAL_SALARY'][i][1].replace(",",""))[0]))

    except:

      START_SALARY.append('NaN')

#Assign Start Salary to Dataframe    

df['START_SALARY']=START_SALARY  



#Get End Salary from Annual Salary

END_SALARY=[]

for i in range(len(df)):

  try:

    END_SALARY.append(int(re.findall('\d+',df['ANNUAL_SALARY'][i][0].replace(",",""))[1]))

  except:

    #If Annual Salary is flat

    try:

      END_SALARY.append(int(re.findall('\d+',df['ANNUAL_SALARY'][i][1].replace(",",""))[1]))

    except:

      END_SALARY.append('x')

#Assign Start Salary to Dataframe     

df['END_SALARY']=END_SALARY



#Make Start Salary Same as End Salary is salary is flat

for i in range(len(df)):

  if df['END_SALARY'][i] is 'x':

    df['END_SALARY'][i]=df['START_SALARY'][i]

    
#Plot to show Start salary range by Job Class

#Impute missing values for numerical varibales

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

df_num_col = ["START_SALARY"]

data_num=df[df_num_col]

imputer=imputer.fit(data_num)

df["START_SALARY"]=imputer.transform(data_num)

df['START_SALARY'].iplot(kind='hist', xTitle='Start Salary',

                  yTitle='count', title='Start Salary Distribution')
#Plot to show End salary range by Job Class

#Impute missing values for numerical varibales

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

df_num_col = ["END_SALARY"]

data_num=df[df_num_col]

imputer=imputer.fit(data_num)

df["END_SALARY"]=imputer.transform(data_num)

df['END_SALARY'].iplot(kind='hist', xTitle='End Salary',

                  yTitle='count', title='End Salary Distribution')
#Job Openings by Date

plt_data = df['OPEN_DATE'].value_counts()

plt.figure(figsize=(12,15))

sns.distplot(plt_data.values)

plt.title('Job Openings by Date')

plt.legend('')

plt.show()

#Number of Openings by Job Class

title=''.join(job for job in df['CLASS_TITLE'])    

title=word_tokenize(title)

title=Counter(title)

#Select 1o most common Job Class

class_title=[job for job in title.most_common(10) if len(job[0])>3] 

a,b=map(list, zip(*class_title))



plt.figure(figsize=(12,15))

sns.barplot(a,b)

plt.xlabel('Job Count')

plt.ylabel('Class')

plt.title('Number Of Openings by Job Class')

plt.legend('')

plt.show()