# import packages

import numpy as np 

import pandas as pd 

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
# import data

df=pd.read_csv("../input/h1b_kaggle.csv")
df.shape



# as is seen it is a dataset of over 3 million records of H1B visa application. Job titles include

# a large number of different positions they are hired for. Among them I sm only interested in

# postdoctoral fellows
# show header

df.head()
# show column names (see data description for details)

df.columns
# create postdoctoral dataset I am interested in.

# postdocs are often writtent in official job titles as "Postdoctoral", " Post doctoral" or

# simply "Postdoc" fellows. I used all 3 terms to filter rows and create the dataset containing

# information only about postdocs.



# filter job title containing "POSTDOCTORAL"

post1 = df[df['JOB_TITLE'].str.match('POSTDOCTORAL', na=False)]

# filter job title containing "POST DOCTORAL"

post2 = df[df['JOB_TITLE'].str.match('POST DOCTORAL', na=False)]

# filter job title containing "POSTDOC"

post3 = df[df['JOB_TITLE'].str.match('POSTDOC ', na=False)]

# join the three dataframes

postdoc = post1.append([post2, post3])
# show summary stats

print('mean postdoc salary is:', postdoc['PREVAILING_WAGE'].mean())

print('meadian postdoc salary is:', postdoc['PREVAILING_WAGE'].median())

print('minimum salary:', postdoc['PREVAILING_WAGE'].min())

print('maximum postdoc salary:', postdoc['PREVAILING_WAGE'].max())



# The mean ($ 117K) looks very high, so I looked at median, which is reasonable. So there must be some 

# very large numbers driving the mean to become so large?

# Looked at the max value, which is absurd for a postdoc salary!
# so to detect outliers/potential flaws in data by I decided to see salaries more than 140K.

(postdoc[postdoc['PREVAILING_WAGE']>140000]).shape



# there are 44 out of 37793 with a postdoc salary more than $140K. These are definitely problem rows, 

# visual inspection of rows also showed that all of those cases are either denied or withdrawm

# so decided to removed them
# drop 44 rows with salaries more than 140k

postdoc = postdoc[postdoc['PREVAILING_WAGE']<140000]
# now let's see some summary stats

postdoc['PREVAILING_WAGE'].describe()
# ploting salary distribution to see how it looks like in visuals

sns.distplot(postdoc['PREVAILING_WAGE'])
# Now I want to create another summary viaual for different years to see the trend. 

# But before that I want to inspect data types



postdoc.info()

# looks like we need to change 'YEAR' data type, which definitely isn't floating type
# change YEAR column from floating to integer

postdoc['YEAR'] = postdoc['YEAR'].astype(int)



# now see how many entries in each year

postdoc['YEAR'].value_counts()
sns.boxplot(x='YEAR' , y='PREVAILING_WAGE', data = postdoc)