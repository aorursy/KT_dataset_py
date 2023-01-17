# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

raw_school_data=pd.read_csv("../input/data-science-for-good/2016 School Explorer.csv")

# list all of the column headers
raw_school_data.keys()

print ('The dataset has %d categories.' % len(raw_school_data.columns))
print ('The dataset has %d schools.' % len(raw_school_data.values))

# Any results you write to the current directory are saved as output.

##### data formatting #########

# fix an inconsistent title for one of the columns
cols = raw_school_data.columns.values
cols[51] = 'Grade 3 Math - All Students Tested'
raw_school_data.columns = cols.tolist()

# remove all the % signs
raw_school_data = raw_school_data.replace(to_replace='%', value='', regex=True)

################################



admin = []
common_data  = []
demographics_data  = []
grades_data = []

# admin params
admin.append(raw_school_data['School Name'])
admin.append(raw_school_data['District'])
admin.append(raw_school_data['Location Code'])
admin.append(raw_school_data['Latitude'])
admin.append(raw_school_data['Longitude'])
admin.append(raw_school_data['Grades'])
admin.append(raw_school_data['Grade Low'])
admin.append(raw_school_data['Grade High'])
admin.append(raw_school_data['Community School?'])
admin.append(raw_school_data['Zip'])

# common data params
common_data.append(raw_school_data['Economic Need Index'].astype('float'))
common_data.append(raw_school_data['Student Attendance Rate'].astype('float'))
common_data.append(raw_school_data['Percent of Students Chronically Absent'].astype('float'))
common_data.append(raw_school_data['Rigorous Instruction %'].astype('float'))
common_data.append(raw_school_data['Collaborative Teachers %'].astype('float'))
common_data.append(raw_school_data['Supportive Environment %'].astype('float'))
common_data.append(raw_school_data['Effective School Leadership %'].astype('float'))
common_data.append(raw_school_data['Strong Family-Community Ties %'].astype('float'))
common_data.append(raw_school_data['Trust %'].astype('float'))
common_data.append(raw_school_data['Average ELA Proficiency'].astype('float'))
common_data.append(raw_school_data['Average Math Proficiency'].astype('float'))

# demographics data params
demographics_data.append(raw_school_data['Percent ELL'].astype('float'))
demographics_data.append(raw_school_data['Percent Asian'].astype('float'))
demographics_data.append(raw_school_data['Percent Black'].astype('float'))
demographics_data.append(raw_school_data['Percent Hispanic'].astype('float'))
demographics_data.append(raw_school_data['Percent White'].astype('float'))
# get all the grade-specific data

def get_grades_data (data, grades_list):
    grade_data = []
    for i in range(len(grades_list)):
        key = 'Grade %d ELA - All Students Tested' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d Math - All Students Tested' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d ELA 4s - All Students' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d Math 4s - All Students' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d ELA 4s - Limited English Proficient' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d Math 4s - Limited English Proficient' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d ELA 4s - Economically Disadvantaged' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d Math 4s - Economically Disadvantaged' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d ELA 4s - Asian or Pacific Islander' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d Math 4s - Asian or Pacific Islander' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d ELA 4s - Black or African American' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d Math 4s - Black or African American' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d ELA 4s - Hispanic or Latino' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d Math 4s - Hispanic or Latino' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d ELA 4s - White' % grades_list[i]
        grade_data.append(data[key].astype('float'))
        key = 'Grade %d Math 4s - White' % grades_list[i]
        grade_data.append(data[key].astype('float'))
    return grade_data


# get all of the individual grades params
grades_data = get_grades_data (raw_school_data, [3,4,5,6,7,8])




# remove all schools with missing common data

keep_list = (np.ones(len(common_data[0]))>0)
for i in range(len(common_data)):
    for j in range(len(common_data[i])):
        if (math.isnan(common_data[i][j])):
            keep_list[j] = False
            
for i in range(len(demographics_data)):
    for j in range(len(demographics_data[i])):
        if (math.isnan(demographics_data[i][j])):
            keep_list[j] = False
            
print ('Removed %d schools with incomplete data' % np.count_nonzero(~keep_list))


# remove all the schools with missing grade 7 and 8 data

n_schools = len(grades_data[0])
for i in range(n_schools):
    zero_cnt = 0
    for j in range(len(grades_data)):
        if (grades_data[j].name == 'Grade 7 ELA - All Students Tested'  or
            grades_data[j].name == 'Grade 7 Math - All Students Tested' or
            grades_data[j].name == 'Grade 8 ELA - All Students Tested'  or
            grades_data[j].name == 'Grade 8 Math - All Students Tested'):
            if (grades_data[j][i] == 0.0):
                zero_cnt += 1
    if (zero_cnt==4):
         keep_list[i] = False

print ('Removed %d schools with no grade 7 or 8 students' % np.count_nonzero(~keep_list))


# use the keep list to refine the schools list

for i in range(len(admin)):
    admin[i] = admin[i][keep_list]

for i in range(len(common_data)):
    common_data[i] = common_data[i][keep_list]

for i in range(len(demographics_data)):
    demographics_data[i] = demographics_data[i][keep_list]

for i in range(len(grades_data)):
    grades_data[i] = grades_data[i][keep_list]


print ('After pruning, there are %d schools remaining.' % np.count_nonzero(keep_list))


# create a combined list

nyc_school_data = admin + common_data + demographics_data + grades_data
nyc_school_data_keys = []
for i in range(len(nyc_school_data)):
    nyc_school_data_keys.append(nyc_school_data[i].name)

# convert to DataFrame object and save as .csv

df = pd.DataFrame(data=np.array(nyc_school_data).transpose(),columns=nyc_school_data_keys)
df.to_csv('nyc_school_explorer_refined.csv', index=False)