# must for data analysis

% matplotlib inline

import numpy as np

import pandas as pd

from matplotlib.pyplot import *



# useful for data wrangling

import io, os, re, subprocess



# for sanity

from pprint import pprint
data_files = os.listdir('../input')

pprint(data_files)
filename = 'ca_law_enforcement_by_campus.csv'

filewpath = "../input/"+filename



with open(filewpath) as f:

    lines = f.readlines()



# First 6 lines are part of the header

header = ' '.join(lines[:6])

header = re.sub('\n','',header)

data = lines[6:]



pprint([p.strip() for p in data[:10]])
number_of_commas = [len(re.findall(',',p)) for p in data]

print(number_of_commas)
# Parse each line with a regular expression

newlines = []

for p in data:

    if( len(re.findall(',,,,',p))==0):

        newlines.append(p)



pprint(newlines[:10])
one_string = '\n'.join(newlines)

sio = io.StringIO(one_string)



columnstr = header



# Get rid of \r stuff

columnstr = re.sub('\r',' ',columnstr)

columnstr = re.sub('\s+',' ',columnstr)



# Fix what can ONLY have been a typo, making this file un-parsable without superhuman regex abilities

columnstr = re.sub(',Campus','Campus',columnstr)



columns = columnstr.split(",")



df = pd.read_csv(sio,quotechar='"',header=None,  names=columns, thousands=',')

df.head()
import seaborn as sns
sns.pairplot(df)
sns.distplot(df['Student enrollment'],bins=15,kde=False)
# Divide the schools into three size bins using quantiles

slice1 = np.percentile(df['Student enrollment'],q=33)

slice2 = np.percentile(df['Student enrollment'],q=66)



def school_size(enrollment):

    if enrollment < slice1:

        return 'Small'

    elif enrollment < slice2:

        return 'Medium'

    else:

        return 'Large'



df['Size'] = df['Student enrollment'].map(lambda x : school_size(x))
sns.pairplot(df, hue="Size")
def ca_law_enforcement_by_agency(data_directory):

    filename = 'ca_law_enforcement_by_agency.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        content = f.read()



    content = re.sub('\r',' ',content)

    [header,data] = content.split("civilians\"")

    header += "civilians\""

    

    data = data.strip()

    agencies = re.findall('\w+ Agencies', data)

    all_but_agencies = re.split('\w+ Agencies',data)

    del all_but_agencies[0]

    

    newlines = []

    for (a,aba) in zip(agencies,all_but_agencies):

        newlines.append(''.join([a,aba]))

    

    # Combine into one long string, and do more processing

    one_string = '\n'.join(newlines)

    sio = io.StringIO(one_string)

    

    # Process column names

    columnstr = header.strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columns = columnstr.split(",")

    columns = [s.strip() for s in columns]



    # Load the whole thing into Pandas

    df = pd.read_csv(sio,quotechar='"',header=None,names=columns)



    return df





df1 = ca_law_enforcement_by_agency('../input/')

df1.head()
def ca_law_enforcement_by_campus(data_directory):

    filename = 'ca_law_enforcement_by_campus.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        lines = f.readlines()

    

    header = ' '.join(lines[:6])

    header = re.sub('\n','',header)

    data = lines[6:]

    

    # Process each string in the list

    newlines = []

    for p in data:

        if( len(re.findall(',,,,',p))==0):

            newlines.append(re.sub(r'^([^"]{1,})(,"[0-9])' ,  r'"\1"\2', p))



    # Combine into one long string, and do more processing

    one_string = '\n'.join(newlines)

    sio = io.StringIO(one_string)



    columnstr = header



    # Get rid of \r stuff

    columnstr = re.sub('\r',' ',columnstr)

    columnstr = re.sub('\s+',' ',columnstr)



    # Fix what can ONLY have been a typo, making this file un-parsable without superhuman regex abilities

    columnstr = re.sub(',Campus','Campus',columnstr)



    columns = columnstr.split(",")



    df = pd.read_csv(sio,quotechar='"',header=None,  names=columns, thousands=',')



    return df





df2 = ca_law_enforcement_by_campus('../input/')

df2.head()
def ca_law_enforcement_by_city(data_directory):

    filename = 'ca_law_enforcement_by_city.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        content = f.read()



    content = re.sub('\r',' ',content)

    [header,data] = content.split("civilians\"")

    header += "civilians\""

    

    data = data.strip()

        

    # Combine into one long string, and do more processing

    one_string = re.sub(r'([0-9]) ([A-Za-z])',r'\1\n\2',data)

    sio = io.StringIO(one_string)

    

    # Process column names

    columnstr = header.strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columns = columnstr.split(",")



    # Load the whole thing into Pandas

    df = pd.read_csv(sio,quotechar='"', header=None, names=columns, thousands=',')



    return df





df3 = ca_law_enforcement_by_city('../input/')

df3.head()
def ca_law_enforcement_by_county(data_directory):

    filename = 'ca_law_enforcement_by_county.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        content = f.read()



    content = re.sub('\r',' ',content)

    [header,data] = content.split("civilians\"")

    header += "civilians\""

    

    data = data.strip()

        

    # Combine into one long string, and do more processing

    one_string = re.sub(r'([0-9]) ([A-Za-z])',r'\1\n\2',data)

    sio = io.StringIO(one_string)

    

    # Process column names

    columnstr = header.strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columns = columnstr.split(",")



    # Load the whole thing into Pandas

    df = pd.read_csv(sio,quotechar='"',header=None,names=columns,thousands=',')



    return df





df4 = ca_law_enforcement_by_county('../input/')

df4.head()
def ca_offenses_by_agency(data_directory):

    filename = 'ca_offenses_by_agency.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        lines = f.readlines()

    

    one_line = '\n'.join(lines[1:])

    sio = io.StringIO(one_line)

    

    # Process column names

    columnstr = lines[0].strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columns = columnstr.split(",")

    

    # Load the whole thing into Pandas

    df = pd.read_csv(sio,quotechar='"',names=columns, thousands=',')



    return df



df5 = ca_offenses_by_agency('../input/')

df5.head()
def ca_offenses_by_campus(data_directory):

    filename = 'ca_offenses_by_campus.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        lines = f.readlines()

    

    # Process each string in the list

    newlines = []

    for p in lines[1:]:

        if( len(re.findall(',,,,',p))==0):

            # This is a weird/senseless/badly formatted line

            if( len(re.findall('Medical Center, Sacramento5',p))==0):

                newlines.append(re.sub(r'^([^"]{1,})(,"[0-9])' ,  r'"\1"\2', p))



    one_line = '\n'.join(newlines)

    sio = io.StringIO(one_line)

    

    # Process column names

    columnstr = lines[0].strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columnstr = re.sub(',Campus','Campus',columnstr)

    columns = columnstr.split(",")

    

    # Load the whole thing into Pandas

    df = pd.read_csv(sio, quotechar='"', thousands=',', names=columns)

    

    return df



df6 = ca_offenses_by_campus('../input/')

df6.head()
def ca_offenses_by_city(data_directory):

    filename = 'ca_offenses_by_city.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        content = f.read()

    

    lines = content.split('\n')

    one_line = '\n'.join(lines[1:])

    sio = io.StringIO(one_line)

    

    # Process column names

    columnstr = lines[0].strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columns = columnstr.split(",")

    

    # Load the whole thing into Pandas

    df = pd.read_csv(sio,quotechar='"',names=columns,thousands=',')



    return df



df7 = ca_offenses_by_city('../input/')

df7.head()
def ca_offenses_by_county(data_directory):

    filename = 'ca_offenses_by_county.csv'



    # Load file into list of strings

    with open(data_directory + '/' + filename) as f:

        lines = f.readlines()

    

    one_line = '\n'.join(lines[1:])

    sio = io.StringIO(one_line)

    

    # Process column names

    columnstr = lines[0].strip()

    columnstr = re.sub('\s+',' ',columnstr)

    columnstr = re.sub('"','',columnstr)

    columns = columnstr.split(",")

    

    # Load the whole thing into Pandas

    df = pd.read_csv(sio,quotechar='"',names=columns,thousands=',')



    return df



df8 = ca_offenses_by_county('../input/')

df8.head()