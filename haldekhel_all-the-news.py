# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from textblob import TextBlob

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import glob

print(os.listdir("../input"))



def process(url, out):

    df =  pd.read_csv(url)

    cpd = df['publication'].value_counts()

    print(cpd)



    col_names =  ['subject', 'publication', 'date', 'polarity', 'subjectivity']

    nf  = pd.DataFrame(columns = col_names)



    isDateBefore = '2016-11-08'

    isDateAfter = '2015-03-18'

    df = df[(df['date'] > isDateAfter) & (df['date'] <= isDateBefore)]



    tf = df[~df['title'].str.contains("Hillary", "Clinton",  na=False)]

    tf = tf[tf['title'].str.contains("Trump", "Donald",  na=False)]



    hf = df[~df['title'].str.contains("Trump", "Donald",  na=False)]

    hf = hf[hf['title'].str.contains("Hillary", "Clinton",  na=False)]



    print(list(df))

    # print(hf['title'].head())

    # print(hf.count())



    # print(tf['title'].head())

    # print(tf.count())



    for row in tf.iterrows():

        subject = 'T'

        publication = row[1][3]

        score = TextBlob(row[1][9]).sentiment

        polarity = score[0]

        subjectivity = score[1]

        date = row[1][5]

        nfc = pd.DataFrame([[subject, publication, date, polarity, subjectivity]], columns=col_names)        

        nf = nf.append(nfc)

    

    

    for row in hf.iterrows():

        subject = 'H'

        publication = row[1][3]

        score = TextBlob(row[1][9]).sentiment

        polarity = score[0]

        subjectivity = score[1]

        date = row[1][5]

        nfc = pd.DataFrame([[subject, publication, date, polarity, subjectivity]], columns=col_names)        

        nf = nf.append(nfc)



    nf.to_csv(out, index=True)

    

    return cpd





k = process('../input/articles1.csv', 'part1.csv')

l = process('../input/articles2.csv', 'part2.csv')

m = process('../input/articles3.csv', 'part3.csv')



interesting_files = glob.glob("*.csv")

print(interesting_files)



df_list = []

for filename in sorted(interesting_files):

    df_list.append(pd.read_csv(filename))

full_df = pd.concat(df_list)

full_df.to_csv('output.csv')





spd = full_df.groupby(['subject', 'publication'])['polarity'].mean().to_frame()

ssd = full_df.groupby(['subject', 'publication'])['subjectivity'].mean().to_frame()

spd.to_csv('spd.csv')

ssd.to_csv('ssd.csv')















# Any results you write to the current directory are saved as output.