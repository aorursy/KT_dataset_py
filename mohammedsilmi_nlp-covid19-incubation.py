import pandas as pd 

from pandas import DataFrame

import re

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import os

import numpy as np

import urllib.request
#read csv file into a sentences dataframe

df_sentences = pd.read_csv("/kaggle/input/covid-articles-csv/export_dataframe.csv")

#creating a new dataframe contains only the sought after sentences (which contain 'incubation')

df_incubation_sentences= df_sentences[df_sentences[" col 1"].str.contains('incubation',na=False)]
#assign incubation sentences to a list called text 

#creating empty lists for later numerical processing 

text = df_incubation_sentences[' col 1'].values

incubation_times=[]

float_times=[]

int_times=[]
#finding all numerical values within the "incubation" sentences

for t in text:

    for sentence in t.split(". "):

        if "incubation" in sentence:

            iday=re.findall(r" \d{1,2} day| \d{1,2}.\d{1,2} day",sentence)

            if len(iday)==1:

                num=iday[0].split(" ")

                incubation_times.append(num[1])
#processing ranged data, example: (6-10 days) by taking the average, 8 days

for row in incubation_times:

    if "-" in row:

        day=row.split('-')

       

        num1=int(day[0])

        num2=int(day[1])

        num3=(num1+num2)/2

        back=str(num3)

        

        new=row.replace(row, back)

        float_times.append(new)
#processing one unit non-range values in the sentences

for row in incubation_times:

    if(len(row)<3):

        int_times.append(row)

    
# combining both float and int data 

incubation_int_float=[]

incubation_int_float=int_times+float_times

df_incubation=pd.DataFrame(incubation_int_float, columns=["duration"])
#converting type of data to numerical 

df_incubation["duration"] = pd.to_numeric(df_incubation["duration"])

df_incubation=df_incubation[df_incubation["duration"]<30]
#average incubation period

df_incubation.mean()