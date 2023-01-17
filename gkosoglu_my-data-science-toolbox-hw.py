# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #visualization tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')

df.info()
df.columns = ['gender', 'race', 'parentDegree', 'lunch', 'course', 'mathScore', 'readingScore', 'writingScore'] 
#LAMBDA FUNCTION

# This part creates a lambda function which is used for averaging Math,Reading and Writing Scores for all students. Then adds additional column to the current data.

average = lambda x,y,z : (x+ y + z)/3

#print(average(1,2,3))



average_all = average(df.mathScore,df.readingScore,df.writingScore)

print(type(average_all))



df = df.assign(averageScore = pd.Series(average_all).values )

print(df)

#This part gives us the comparison of Math grades to the average score of all students. The ones above the average are tagged as "high grade" others "low grade".

math_threshold = sum(df.mathScore)/len(df.mathScore) # finds average of mathScore

df["threshold_math"] = ["high grade" if i>math_threshold else "low grade" for i in df.mathScore]

df.drop(columns=['gender','race','parentDegree','lunch','course'],inplace = True)

#df = df.assign(math_grades = pd.Series(threshold_math).values )

cols = df.columns.tolist()

cols = cols[-1:] + cols[:-2]

df = df[cols]

print('threshold value of math scores is:',math_threshold)

print(df)
#LIST COMPHERENSIONs

# Below chart shows the student's mathscores and its comparison to the average of writing and Reading Scores. M_over_WR gives us the average of Reading and Writing

# then this value is compared to the math scores of each students. IF the average of W&R is higher than student's math score it is written HIGH, else LOW.

scores1 = df.writingScore

scores2 = df.readingScore

scores3 = df.mathScore



avr_WR = (scores1 + scores2)/2

df["avr_WR"] = avr_WR

df = df.assign(avr_WR = pd.Series(avr_WR).values )



df["M_over_WR"] = ["high" if i>avr_WR[i] else "low" for i in scores3]

df.loc[:10,["avr_WR","mathScore","M_over_WR"]]
zipped_scores = zip(scores1,scores2,scores3)

print(zipped_scores)

z_list = list(zipped_scores)

print(z_list[:10])
un_zip = zip(*z_list)

W_score,R_score,M_score = list(un_zip)

print(W_score[:10])

print(R_score[:10])

print(M_score[:10])




