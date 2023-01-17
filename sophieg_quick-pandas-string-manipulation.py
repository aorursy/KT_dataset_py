import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

fresh_df  = pd.read_csv('../input/train.csv', #provide file name "relative to the notebook location"

                  index_col=0) # use the first column as an index

fresh_df.info()
fresh_df.head()
#replace all zeros by -1

#create a copy

df = fresh_df.copy()

df = df.replace(0,-1)

df.head()
#refresh

df = fresh_df.copy()

def lookup(string):

    """

    input  a string 

    output is the string  if string not in dictionary dic

    d[string] if string in dictionary dic

    """

    if string in dic:

        return dic[string]

    return string

dic = {'Braund, Mr. Owen Harris': 'Mr Harris',

       'Allen, Mr. William Henry':'Mr Allen'}

df.Name = df.Name.apply(lookup)

df.head()
#fresh

df = fresh_df.copy()

dic = {'Braund, Mr. Owen Harris': 'Mr Harris',

       'Allen, Mr. William Henry':'Mr Allen'}

#syntax {'column name': dictionary of desired replacements}

df.replace({'Name':dic},inplace=True)

df.head()
#fresh

df = fresh_df.copy()

dic = {'Braund, Mr. Owen Harris': 'Mr Harris',

       'Allen, Mr. William Henry':'Mr Allen'}



df['Name'] = df['Name'].map(lambda x: dic[x] if x in dic.keys() else x)

df.head()
#fresh

df = fresh_df.copy()

def removedigit(string):

    """

    input is a string 

    output is a string with no digits

    """

    return ''.join(ch for ch in string if not ch.isdigit())

df.Ticket = df.Ticket.apply(removedigit)

df.head()
#refresh df

df = fresh_df.copy()

df.Ticket = df.Ticket.str.replace('[0-9]','')

df.head()
#refresh df

df = fresh_df.copy()

def removeAfterComma(string):

    """

    input is a string 

    output is a string with everything after comma removed

    """

    return string.split(',')[0].strip()

df.Name = df.Name.apply(removeAfterComma)

df.head()
#refresh df

df = fresh_df.copy()

df.Name = df.Name.str.split(',').str[0].str.strip()

df.Name.head()
#refresh df

df = fresh_df.copy()

df.Name = df.Name.str.replace('\,.*','').str.strip()

df.Name.head()