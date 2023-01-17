# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# Import files

# NOTE: Some of the imported packages are not used, they are just used as usual imports

import os

import sys

import numpy as np

import matplotlib as plt

import tensorflow as tf

import time

import random

import math

import pandas as pd

import sklearn

from scipy import misc

import glob

import pickle

import re

%matplotlib inline

plt.pyplot.style.use('ggplot')
dataSet = pd.read_csv('../input/naukri_com-job_sample.csv')
# Dataset headers or column names

heads = list(dataSet)

print("Number of columns : "+str(len(heads)))

dataSet.head()
# Dissecting the dataset

#This dataset has following fields: (Ref: https://www.kaggle.com/PromptCloudHQ/jobs-on-naukricom)



#    company

#    education

#    experience

#    industry

#    job description

#    jobid

#    joblocation_address

#    job title

#    number of positions

#    pay rate

#    postdate

#    site_name

#    skills

#    uniq_id
# When we try to do Pattern Recognition or Corelation Analysis, we have to consider what fields to use in them

# Also what kind of changes we have to make in a field for it to be used to in the data analysis

# Some fields might require parsing to convert the human language into a statistical value or even a category



# NOTE:

### We will be looking at each field individually to understand the techniques for data processing

### This will help to learn how a particular type of data is turned into an input for Machine Learning Algorithms
# Field-1

# Company Name

print(heads[0])

print(type(heads[0]))



# An example name looks like

print(dataSet[heads[0]][0])





# Suggested Use:

### We can use it for individual company analytics as there is no industry mentioned for the company as well

### To classify it into some kind of sector e.g. Steel Sector, IT Sector

### Hence, for Analysis of trends in payrates, or salaries based on experience, this field can be skipped

### Principal Component Analysis (PCA) is used for the dimensionality-reduction in statistics, but as this field is a string

### We either have to declare numerical counterparts for such fields before including them in statistical analysis

### or We can do company wise analysis as mentioned earlier



# If you check out some data mining books, such data without any order is  'categorical' data
# Field-2

# Education

print(heads[1])

print(type(heads[1]))



# An example name looks like

print(dataSet[heads[1]][0])





# Suggested Use:

### This is a string as well, but this needs to be 'parsed' into different skills and can be useful for the dataset

### Education skills for Under Graduate(UG) and Post Graduate(PG) can help form a trend analysis to see

### which skills have the highest pay, or are in demand

### This would be a 'categorical' data as well,

### Because, e.g. Jobs for Computers and Media Technology may be different but they don't bear a rank among themselves





# We take 'any' as a catergory as well, but its semantic meaning implies to the requirements mentioned in the particular job post

# Parsing the data to extract the skills for both UG and PG

# 'UG' and 'PG' are both followed by ':' which makes it a good first candidate for splitting between the Two parts of strings

# As they are different levels of qualification



# NOTE: The Analysis and preprocessing for this field will be released in the upcoming kernel very soon
# Field-3

# Experience

print(heads[2])

print(type(heads[2]))



# An example name looks like

print("\nBefore Processing: ")

print(dataSet[heads[2]][10])



# We process this field and now get the splits based on "-" and replace everything not a digit

exp = []

for i in range(len(dataSet)):

    exp.append(((str(dataSet[heads[2]][i])).replace(" ","")).split("-"))

    

for i in range(len(exp)):

    for j in range(len(exp[i])):

        exp[i][j] = re.sub(r'[^0-9]','',exp[i][j])

# An Example after processing

print("\nAfter Processing: ")

print(exp[10])
# Field-4

# Industry

print(heads[3])

print(type(heads[3]))



# An example name looks like

print(dataSet[heads[3]][0])



# Storing the Industries for different jobs, splitting with "/" and " " and removing all spaces using 'replace', otherwise they would occur in the list

### NOTE: The dataSet in this field had one object being recognized as float, hence we have to str() the field

### inspite of it being a 'str' type field, we will be able to filter out that number when we try to see,

### the categories of industries involved, as we will produce different industries and their number of occurences

Industry=[]

for i in range(len(dataSet)):

    Industry.append(((str(dataSet[heads[3]][i])).replace(" ","")).split("/"))

print(len(Industry))



# an example of how the industries currently look like

print(Industry[10])
# Field-5

# Job Description

print(heads[4])

print(type(heads[4]))



# An example name looks like

print(dataSet[heads[4]][0])



# This is a whole description about the job post,which may have relevant information

# By first look ,  : - == > , may seem like a separator for parsing, but if we look closely, the field description before it is 

# mixed with the next field name, e.g. Knowledge Job Requirement : - == > System or Laptop Type of job: - == > Full Time or Part time

# Here, Job Requirement seems like a possible field and Type of Job, but they are both together

# To clean the data, we would require the NLP(Natural Language Processing) after initial cleaning

# which may be implemented in the upcoming versions of this kernel
# Field-6

# Job Id

print(heads[5])

print(type(heads[5]))



# An example name looks like

print(dataSet[heads[5]][0])



# Job Id is useful to search for the job using this number, but for analysis, it can serve the same purpose, as locating

# a particular job column

# An example of finding the index of a job based on the 'jobid' and this index can be used in other columns to find this record

# Although we know the index would be one, but may be due to duplicacy in data, with some different field values for only one or two columns

# The jobs would not be seen as identical by duplicacy removal programs everytime, and hence it may be more than 1 index too

# So we safely turn it into a list so that there is no disurption in the running of the program

idx = dataSet.jobid[dataSet.jobid == 210516002263].index.tolist()

print(idx)
# Field-7

# Job Location Address

print(heads[6])

print(type(heads[6]))



# An example name looks like

print(dataSet[heads[6]][0])

print(dataSet[heads[6]][10]) # It can contain multiple locations too, so we have to use the splitting again using ','



# Locations list

Locations=[]

for i in range(len(dataSet)):

    Locations.append(((str(dataSet[heads[6]][i])).replace(" ","")).split(","))

print(len(Locations))

# An Example of Locations list

print(Locations[10])
# Field-8

# Job Title

print(heads[7])

print(type(heads[7]))



# An example name looks like

print(dataSet[heads[7]][0])

print(dataSet[heads[7]][200])



# Another field, requiring cleaning and understanding the structure, hence skipped for current analysis

### As can be seen in the two examples printed, it varies heavily on the job poster and sometimes is full of information

### not belonging to this field

### e.g. Day Shift is the type of Job, and it is mentioned twice in this example as dayshift, Day Shift
# Field-9

# Number of Positions

print(heads[8])

print(type(heads[8]))



# An example name looks like

print(dataSet[heads[8]][0])





# Number of disclosed job postings

# Although the class is string, when passed to np.nansum, it will be parsed to its equivalent float value

# nansum treats NaN as zero

print("Number of Disclosed Positions: "+str(np.nansum(dataSet['numberofpositions'])))
# Field-10

# Pay Rate

print(heads[9])

print(type(heads[9]))



# An example name looks like

print(dataSet[heads[9]][0])



# Lets split the salaries first with '-' later, we can replace everything except numbers with ""

payrates=[]

for i in range(len(dataSet)):

    payrates.append(((str(dataSet[heads[9]][i]))).split("-"))



print(len(payrates))

# An Example of payrates list

print("\nBefore Processing: ")

print(payrates[0])

print(payrates[10])



# Checking if the string has a digit, because some categories are "Not Disclosed by Recruiter"

# If we process the strings for numbers these will cause problems

# Hence, we will run one more loop to clean it, we could have done it in the previous loop too,

# But to demonstrate the problems such as the example having '2,25,000 P.A' as one of the values

# We will clean it in another loop, where we will replace all values except for digits in a string 

# having at least one digit by "" which will remove whitespaces and "," and "P.A" or any other strings

for i in range(len(payrates)):

    for j in range(len(payrates[i])):

        payrates[i][j] = re.sub(r'[^0-9]','',payrates[i][j])

# An Example after processing

print("\nAfter Processing: ")

print(payrates[0])

print(payrates[10])
# Field-11

# Post Date

print(heads[10])

print(type(heads[10]))



# An example name looks like

print(dataSet[heads[10]][40])



# We can split this field using " " and keep the first element as it is the one containing date

# The values seem timestamped and hence in a uniform format

dates=[]

for i in range(len(dataSet)):

    dates.append(str(dataSet[heads[10]][i]).split(" ")[0])

    

# An example of dates, some examples may have 'nan' as well which was found during analysis

print(dates[0])



# nan value existing in dates

print(max(dates))
# Field-12

# Site Name

print(heads[11])

print(type(heads[11]))



# A lot of values in this field were nan(Not a Number)

# An example name looks like

print(dataSet[heads[11]][90])
# Field-13

# Skills

print(heads[12])

print(type(heads[12]))



# An example name looks like

print(dataSet[heads[12]][100])
# Field-14

# Unique Id

print(heads[13])

print(type(heads[13]))



# An example name looks like

print(dataSet[heads[13]][0])

# This can be used in a similar manner to the 'jobid' field