# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import unicodecsv

print(os.listdir("../input"))

def read_csv(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)

enrollments = read_csv('../input/enrollments.csv')
daily_engagement = read_csv('../input/daily_engagement.csv')
project_submissions = read_csv('../input/project_submissions.csv')
    
### For each of these three tables, find the number of rows in the table and
### the number of unique students in the table. To find the number of unique
### students, you might want to create a set of the account keys in each table.

print ('This is the account key data: ')
print (enrollments['account_key'])

def num_unique(data, act):
    unique = []
    for d in data:
        unique.append(d[act])
    print('The length of unique {} values: '.format(str(data)) + str(len(unique)))
    
num_unique(enrollments, 'account_key')
        

enrollment_num_rows = len(enrollments)             # Replace this with your code
enrollment_num_unique_students = 0  # Replace this with your code
print ('The num enrollment rows are: ' + str(enrollment_num_rows))

engagement_num_rows = len(enrollments)             # Replace this with your code
engagement_num_unique_students = 0  # Replace this with your code

submission_num_rows = len(enrollments)             # Replace this with your code
submission_num_unique_students = 0  # Replace this with your code

# Any results you write to the current directory are saved as output.
