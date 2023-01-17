import pandas as pd

from IPython.display import display



scoresdf = pd.read_csv("../input/scores.csv")

sleepdf= pd.read_csv("../input/sleep.csv")



display(scoresdf.head())

display(sleepdf.head())
#Striping whitespaces in ID columns

sleepdf['ID1'] = sleepdf.apply(lambda row: row['ID'].strip(),axis=1)

scoresdf['ID2'] = scoresdf.apply(lambda row: row['ID'].strip(),axis=1)
#Merging 2 files into 1 df

df = pd.merge(scoresdf,sleepdf,how='inner',left_on='ID2',right_on='ID1')

df.head()
#Dropping extra columns

df = df.drop(['ID_x','ID_y','ID2'],axis=1)



#Converting columns types

df['Scores'] = df['Scores'].astype('float')

df['Sleep'] = df['Sleep'].astype('int')
df.head()
#Removing outliers

from scipy import stats

import numpy as np

df = df[(np.abs(stats.zscore(df[['Sleep','Scores']])) < 3).all(axis=1)]
df.head()
#For quantitatively measuring the relationship between 2 continuous variables, we take correlation which ranges between -1 to 1.

#Here, -1 indicates inversely correlated and 1 indicates highly correlated. Also, 0 indicates that 2 variables are completely independent of each other.
df['Sleep'].corr(df['Scores'])
#As we can see, we are getting correlation close to 1 which indicates strong relationship between number of hours of sleep and their exam scores.

# Hence the result are in line with the expectation. It is thus proven that better sleep results in higher score.