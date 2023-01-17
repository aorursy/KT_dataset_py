# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-

"""

Created on Sun Feb 26 01:15:50 2017



@author: HITESH VIJAY SOMANI

"""

import pandas as pd



df_exams=pd.read_csv("../input/exams.csv")

df_exams

df_students=pd.read_csv("../input/students.csv")

df_students



df_exams=df_exams.drop(df_exams[df_exams.loc[:,'Score']=='All'].index)

df_exams=df_exams.drop(df_exams[df_exams.loc[:,'Score']=='Average'].index)

df_exams=df_exams.dropna(axis=0)

df_exams=df_exams.reset_index(drop=True)

df_exams



df_students=df_students.dropna(axis=0)

df_students=df_students.reset_index(drop=True)

df_students



import matplotlib.pyplot as plt

plt.style.use('ggplot')

fig = plt.figure()





#Visualization of hich subjects are taken more by males or females.

df_students1=df_students.loc[:,['Exam Subject','Students (Male)','Students (Female)']]

df_students1

ax=df_students1.plot.bar( figsize=(9,6), title='Male vs Female wrt Exam Subjects')

ax.set_xticklabels(df_students1['Exam Subject'])

plt.show()







#Visualization of popularity of subjects among various races.

df_students2=df_students.iloc[:,[0,11,12,13,14,15,16,17,18,19]]

df_students2

ax=df_students2.plot.line(xticks=range(36), figsize=(20,6), title='Distribution of Students of various Races wrt Exam Subject')

ax.set_xticklabels(df_students2['Exam Subject'], rotation='vertical')

plt.show()





#Visualization of which subject is taught more in schools 

df_students3=df_students.iloc[:,[0,1]]

df_students3

ax=df_students3.plot.bar( figsize=(9,6), title='Popularity of Exam Subject in Schools')

ax.set_xticklabels(df_students3['Exam Subject'])

plt.show()