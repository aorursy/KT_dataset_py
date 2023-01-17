# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import re

# Any results you write to the current directory are saved as output.
data_dictionary = pd.read_csv('../input/cityofla/CityofLA/Additional data/kaggle_data_dictionary.csv')

data_dictionary[data_dictionary['Field Name'] == 'ENTRY_SALARY_GEN']
sample_output = pd.read_csv('../input/cityofla/CityofLA/Additional data/sample job class export template.csv')

sample_output.head()
bulletins_dir = 'cityofla/CityofLA/Job Bulletins/'
all_files = glob.glob('../input/' + bulletins_dir + "/*.txt")
rows = []

for filename in all_files:

    row = {}

    fileValue = filename.split('/')[5]

    values = [x for x in re.split('(-?\d+\.?\d*)', fileValue) if x != '']

    try:

        # Hardcoding alert! Swapping the data specific parsing

        if 'DIRECTOR' in values[1]:

            values[0] = values[0]+values[1]

            values[1] = values[2]

            values[3] = values[4]

        row['FILE_NAME'] = fileValue

        row['JOB_CLASS_TITLE'] = values[0]

        row['JOB_CLASS_NO'] = values[1]

        row['OPEN_DATE'] = values[3].replace('.', '')

        rows.append(row)

        with open(filename, encoding = "ISO-8859-1") as bulletinFile:

            bulletinData = bulletinFile.read()

            # Read annual salary, this will need to be processed further to map to the sample output

            row['ANNUAL_SALARY'] = re.search('ANNUAL SALARY(\W+)(.*\n)', bulletinData).group(2)

            # Open data is embedded both as file name and in the bulletin, change here based on content within the file

            row['OPEN_DATE'] = re.search('Open Date(\W+)(.*\n)', bulletinData).group(2).split('\n')[0]

            # Get job duties

            row['JOB_DUTIES'] = re.search('DUTIES(\W+)(.*\n)', bulletinData).group(0).split('\n\n')[1]

            # Processing of educational qualifications needs population of a section. There is no common header for this,

            # processing various flavors of this header

            requirements = re.search('(REQUIREMENTS/MINIMUM QUALIFICATIONS|REQUIREMENTS|REQUIREMENT/MINIMUM QUALIFICATION)(\W+)(.*\n)', bulletinData).group(0).split('\n\n')[1]

            row['QUALIFICATIONS'] = requirements

            # Process notes looks to be the section related to driver's license, so lets pull that in

            processNotes = re.search('PROCESS NOTES(\W+)(.*\n)', bulletinData).group(0).split('\n\n')[1]

            row['PROCESS_NOTES'] = processNotes

    except IndexError as e:

        pass

    except AttributeError as e1:

        pass

df = pd.DataFrame(rows)

df.head()
df['QUALIFICATIONS'] = df['QUALIFICATIONS'].fillna('')

df['PROCESS_NOTES'] = df['PROCESS_NOTES'].fillna('')

# Set the open_data as date time (there is one bulletin that has some data issue which causes a failure in setting the data type)

df['OPEN_DATE'] = pd.to_datetime(df['OPEN_DATE'], errors='ignore')

df['DEGREE_NEEDED'] = ['Y' if 'degree' in x else 'N' for x in df['QUALIFICATIONS']]

#df['DRIVERS_LICENSE_REQ'] = ['P' if 'driver\'s license' in x else 'N' for x in df['PROCESS_NOTES']]

df['DRIVERS_LICENSE_REQ'] = ['R' if 'driver\'s license is required' in x else 'N' for x in df['PROCESS_NOTES']]

print(df[df['DRIVERS_LICENSE_REQ'] == 'R'][['PROCESS_NOTES', 'DRIVERS_LICENSE_REQ']])