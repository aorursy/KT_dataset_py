# use Python3 and urllib.request to scrape data directly from http://www.cisn.org/shakemap/sc/shake/archive

import urllib.request



# use regex to parse data

import re



# use pandas to format data

import pandas as pd



# here's a sample file that runs in this notebook

html = open("/kaggle/input/shakemap-archive/sampleData.txt","r")

# uncomment this line in order to scrape data from the website

#html = urllib.request.urlopen("http://www.cisn.org/shakemap/sc/shake/archive")



def stripTags(line):

    # strip the HTML tags to extract the data

    if "td>" in line:

        line = line.strip('\n')

        line = re.sub('<td>','',line)

        line = re.sub('</td>','',line)

        line = re.sub('<b>','',line)

        line = re.sub('</b>','',line)

        line = re.sub('</a>','',line)

        line = re.sub('<td align="center">','',line)

        line = re.sub("b'         ",'',line)

        if "href" in line:

            l = line.split(">")

            line=l[1]

        return line[:-3]



# extract data from the HTML table

flag=False

k=0

row=[]

earthquakes=[]

for line in html:

    line = str(line)

    # if 'intensity.html' is in the line then begin extracting data

    if 'intensity.html' in line:

        flag=True

        k=1

    elif '</table>' in line:

        flag=False

        

    if flag is True:

        line = stripTags(line)

        if line is not None:

            row.append(line)



    # every 7 lines in the table describes a single earthquake

    if k==7:

        if len(row) > 0:

            earthquakes.append(row)

        k=0

        row=[]

    k+=1



# set the column names for the pandas dataframe

cols = ['EQ_EVENT','EQ_EPICENTER','EQ_DATE','EQ_TIME','EQ_LAT','EQ_LNG','EQ_MAG']

# import data from the earthquakes list into your pandas dataframe

df = pd.DataFrame(data=earthquakes,columns=cols)

pd.set_option('display.max_columns',len(cols))

display(df)



# uncomment this line to save your data in a csv file

#df.to_csv("shakemap.csv",index=False)

# FINDING DATA USED IN THIS NOTEBOOK

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.