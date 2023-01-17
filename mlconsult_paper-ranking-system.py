import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# load the meta data from the CSV file using 3 columns (abstract, title, authors),

df=pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv', usecols=['title','abstract','authors','doi','publish_time'])

print (df.shape)

#drop duplicate abstracts

df = df.drop_duplicates(subset='title', keep="first")

#drop NANs 

df=df.dropna()

# convert abstracts to lowercase

df["abstract"] = df["abstract"].str.lower()



#show 5 lines of the new dataframe

print (df.shape)

df.head()
def search_focus(df):

    dfa = df[df['abstract'].str.contains('covid')]

    dfb = df[df['abstract'].str.contains('-cov-2')]

    dfc = df[df['abstract'].str.contains('hcov')]

    frames=[dfa,dfb,dfc]

    df = pd.concat(frames)

    df=df.drop_duplicates(subset='title', keep="first")

    return df



df=search_focus(df)

i=1

for index, row in df.iterrows():

    author_list=str(row['authors']).split(',')

    code=''

    if 'systematic review' in str(row['abstract']):

        code='I. Systematic Review'

    if 'randomized control' in str(row['abstract']):

        code='II. Randomized Controlled Trial'

    if 'pseudo randomized' in str(row['abstract']):

        code='III-1. Pseudo-Randomized Controlled Trial'

    if 'retrospective cohort' in str(row['abstract']):

        code='III-2. Retrospective Cohort'

    if 'matched case control' in str(row['abstract']):

        code='III-2. Matched Case Control'

    if 'matched case control' in str(row['abstract']):

        code='III-2. Matched Case Control'

    if 'cross sectional' in str(row['abstract']):

        code='III-2. Cross Sectional Control'

    if 'cross-sectional' in str(row['abstract']):

        code='III-2. Cross Sectional Control'

    if 'time series analysis' in str(row['abstract']):

        code='III-3. Time Series Analysis'

    if 'prevalence study' in str(row['abstract']):

        code='IV. Prevalence Study'

    if code=='':

        code='N/A'

    if code!='N/A':

        print (i,row['title'])

        print ("Number of Authors:", len(author_list), 'Study Code: ',code)

        print ("---------")

        i=i+1