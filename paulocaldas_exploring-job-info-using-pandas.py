import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os

import glob

import re

%matplotlib inline
def Extract_Job_Info(job_bulletin):

    

    # read file as a pandas dataframe

    file = pd.read_table(job_bulletin, sep='\n', header = None, nrows=10)

    

    # extract job_title by cleaning the txt file name

    

    job_title = pd.Series(job_bulletin)

    job_title = job_title.str.replace(input_dir, '') # remove dir name

    job_title = job_title.str.replace('\\', '')      # remove \\ 

    job_title = job_title.str.replace('.txt','')     # remove file ext

    job_title = job_title.str.replace(r'\d+', '')    # remove all numbers 

    job_title = job_title.str.replace('rev', '', flags = re.IGNORECASE) 

    job_title = job_title.str.replace(r"\s+\(.*\)","")[0]   # remove all symbols

    

    # if the job title (in caps) occupies 2 lines 

    # drop that line and reset the index for the next steps

    if file.iloc[1][0].isupper() == True:                                 

        file = file.drop(file.index[1]).reset_index()                  



    # extract class_code from second line of the file

    class_code = file.iloc[1].str.replace(r'\D+', '')[0]                



    # extract open_date from thrid line of the file

    open_date = file.iloc[2].str.replace(r'\D+', '')[0] 

    

    # adjust the format to DD-MM-YY

    if len(open_date) < 6:

        open_date = '0'+ open_date                                      

    if len(open_date) > 6:

        open_date = open_date.replace(open_date[-4:],open_date[-2:])      

    

    # introduce dashes to set a pretty format like dd-mm-yy format

    open_date = '-'.join([open_date[:2], open_date[2:4], open_date[4:]])



    # extract minimum and max salary 

    # add exception to set 'NaN' forjobs with no salary info

    

    try:   

        # find the line before the one containing salary info

        # clean the line to obtain only the numbers

        # find the max and min value

        

        for lines in file.iterrows():

            if lines[1][0][:6] == 'ANNUAL': 

                possible_salaries = file.iloc[lines[0] + 1].str.replace(r',', '').str.replace(r'\D+',' ')



        salary_max = np.array(possible_salaries[0].split()).astype(int).max() # finds the max value inline 

        salary_min = np.array(possible_salaries[0].split()).astype(int).min() # finds the min value inline



    except:                                                                

        # set max and min salary to NaN if no information is given

        salary_max = np.nan

        salary_min = np.nan



    return pd.Series(data = [job_title, open_date, salary_min, salary_max, class_code])
# read folder containing the txt files

input_dir = "../input/cityofla/CityofLA/Job Bulletins/"

job_bulletins_list = glob.glob(os.path.join(input_dir,'*txt')) # a list containing all the txt files



# apply the function to each job file in the folder

# save all info as a Pandas dataframe

Job_Info_Table = pd.DataFrame([Extract_Job_Info(job) for job in job_bulletins_list])

Job_Info_Table.columns = ['Job Title','Open Date','Salary Min ($)','Salary Max ($)', 'Class Code']

Job_Info_Table.sort_values('Job Title')