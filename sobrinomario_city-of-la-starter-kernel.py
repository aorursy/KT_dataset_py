%matplotlib inline





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import re #search in strings.



import plotly.plotly as py

import cufflinks as cf



import seaborn as sns

import matplotlib.pyplot as plt



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from wordcloud import WordCloud
input_dir = '../input/cityofla/CityofLA/Job Bulletins/'



def getListOfFiles(dirName):

# create a list of file and sub directories 

# names in the given directory 

    listOfFile = os.listdir(dirName)

    allFiles = list()

    # Iterate over all the entries

    for entry in listOfFile:

    # Create full path

        fullPath = os.path.join(dirName, entry)

        # If entry is a directory then get the list of files in this directory 

        if os.path.isdir(fullPath):

            allFiles = allFiles + getListOfFiles(fullPath)

        else:

            allFiles.append(fullPath)

    return allFiles

listOfFiles = getListOfFiles(input_dir)

df_bulletins = pd.DataFrame(listOfFiles, columns = ['job_position'])
# Clean up of the job_position name

df_positions = pd.DataFrame()

df_positions['job_position'] = (df_bulletins['job_position']

                                .str.replace(input_dir, '', regex=False)

                                .str.replace('.txt', '', regex=False)

                                .str.replace('\d+', '')

                                .str.replace(r"\s+\(.*\)","")

                                .str.replace(r"REV",""))



#Remove the numbers

df_positions['class_code'] = (df_bulletins['job_position']

                              .str.replace(input_dir, '', regex=False)

                              .str.replace('.txt', '', regex=False)

                              .str.extract('(\d+)'))



display(df_positions.head())

# Add the Text fields of Salary, Duties and Minimum REQ

#Convert the txt files to a table:

import glob

path = input_dir # use your path

all_files = glob.glob(path + "/*.txt")

li = []



for filename in all_files:

    with open (filename, "r",errors='replace') as myfile:

        data=pd.DataFrame(myfile.readlines())

        #df = pd.read_csv(filename, header=0,error_bad_lines=False, encoding='latin-1')

    li.append(data)

frame = pd.concat(li, axis=1, ignore_index=True)

#pd.read_csv(listOfFiles,header = None)

frame = frame.replace('\n','', regex=True)

# Here the loop should start, for each text file do:

def getString(col_i, frame):

    try:

        filter = frame[col_i] != ""

        bulletin = frame[col_i][filter]

        #display(salary)

        isal = min(bulletin[bulletin.str.contains('SALARY',na=False)].index.values) #take the sum to convert the array to an int...TO CHANGE

        inot = min(bulletin[bulletin.str.contains('NOTES',na=False)].index.values) # NOTES

        idut = min(bulletin[bulletin.str.contains('DUTIES',na=False)].index.values) # DUTIES

        ireq = min(bulletin[bulletin.str.contains('REQUIREMENT',na=False)].index.values) #REQUIREMENTS

        ipro = min(bulletin[bulletin.str.contains('PROCESS',na=False)].index.values) # PROCESS NOTES



        #isal = sum(bulletin.loc[bulletin == 'ANNUAL SALARY'].index.values) #take the sum to convert the array to an int...TO CHANGE

        #inot = sum(bulletin.loc[bulletin == 'NOTES:'].index.values) # NOTES

        #idut = sum(bulletin.loc[bulletin == 'DUTIES'].index.values) # DUTIES

        #ireq = sum(bulletin.loc[bulletin == '(.*)REQUIREMENTS(.*)'].index.values) #REQUIREMENTS

        #ipro = sum(bulletin.loc[bulletin == '(.*)PROCESS(.*)'].index.values) # PROCESS NOTES



        icode = min(bulletin[bulletin.str.contains('Class Code',na=False)].index.values)

        class_code = sum(bulletin.str.extract('(\d+)').iloc[icode].dropna().astype('int'))

        salary = (bulletin.loc[isal+1:inot-1]).to_string()

        duties = (bulletin.loc[idut+1:ireq-1]).to_string()

        requirements = (bulletin.loc[ireq+1:ipro-1]).to_string()

        return (class_code, salary, duties, requirements)

    except:

        return (np.nan,np.nan,np.nan,np.nan)

    

jobsections = pd.DataFrame()

#getString(0,bulletin)

for col_i in range(frame.shape[1]):

    #print(col_i)

    #print(list(getString(col_i,frame)))

    prop = getString(col_i,frame)

    prop = pd.DataFrame(list(prop)).T

    jobsections = jobsections.append(prop)
jobsections.head()
jobsections.columns = ['class_code','salary','duties','requirements']

jobsections['class_code'] = pd.to_numeric(jobsections['class_code'],downcast='integer')

df_positions['class_code'] = pd.to_numeric(df_positions['class_code'], downcast='integer')

#df_positions['class_code']

df_jobs = df_positions.merge(jobsections, left_on='class_code',right_on='class_code', how='outer')

display(df_jobs.dropna())
