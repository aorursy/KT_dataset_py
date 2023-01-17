# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import init_notebook_mode, iplot, download_plotlyjs

import plotly.graph_objs as go

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Reading dataset in this form, so that I can use it on my laptop also.

try:

    df_mcr = pd.read_csv('../input/multipleChoiceResponses.csv')

    df_schema = pd.read_csv('../input/SurveySchema.csv')

except Exception as e:

    pass
# Basic information about Dataset.

print(df_mcr.shape)

print("As one can see there are 395 columns.")
# Let's see the details about the null values.

null = df_mcr.isnull().sum()

print(null[:30])

df_mcr.head()
gender = df_mcr['Q1'].value_counts().reset_index()

gender.iplot(kind='pie', labels='index', values='Q1',title='Ration of Gender', pull=0.2, hole=0.2 )
age = df_mcr['Q2'].value_counts().reset_index()

age.iplot(kind='bar', x='index', y='Q2', title='Age of respondants', xTitle='Age',

          yTitle='Number of responses', colors='deepskyblue')
country = df_mcr['Q3'].value_counts().reset_index()[:10]

country.iplot(kind='bar', x='index', y='Q3', 

              title='Top 10 Countries participated in survey', xTitle='Country', yTitle='Number of Respondants')

country.drop([3], axis=0, inplace=True)

country

values = country['Q3'].values

#print(values)

name = country['index'].values

#print(name)

code = ['USA','IND','CHN','RUS','BRA','DEU','GBR','CAN','FRA']

data = dict(

        type = 'choropleth',

        locations = code,

        z = values,

        text = name,

        colorbar = {'title' : 'Number of Participants'},

      ) 

layout = dict(

    title = 'Country Wise Users',

    geo = dict(

        showframe = False,

        projection = {'type':'natural earth'}

    )

)

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
degree = df_mcr['Q4'].value_counts().reset_index()

degree.iplot(kind='bar',x='index', y='Q4', title='Top formal educations', xTitle='Degree', 

             yTitle='Frequency', colors='deepskyblue')
titles = df_mcr['Q6'].value_counts().reset_index()[:10]

titles.iplot(kind='bar',x='index', y='Q6', title='Top Current Titles', 

             xTitle='Title', yTitle='Frequency',colors='green')
industry = df_mcr['Q7'].value_counts().reset_index()[:10]

industry.iplot(kind='bar',x='index', y='Q7', title='Top 10 Current industry', 

             xTitle='Industry', yTitle='Frequency',colors='indigo')
experience = df_mcr['Q8'].value_counts().reset_index()[:10]

experience.iplot(kind='bar',x='index', y='Q8', title='Average Years of experience', 

             xTitle='Years', yTitle='Frequency',colors='indianred')
compensation = df_mcr['Q9'].value_counts().reset_index()[:10]

compensation.iplot(kind='bar',x='index', y='Q9', title='Average compensation', 

             xTitle='Amount of compensation', yTitle='Frequency',colors='indianred')

temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2,p3 = temp_df.loc[:,'Q11_Part_1'],temp_df.loc[:,'Q11_Part_2'],temp_df.loc[:,'Q11_Part_3'],

p4,p5  = temp_df.loc[:,'Q11_Part_4'],temp_df.loc[:,'Q11_Part_5'],

p6,p7 = temp_df.loc[:,'Q11_Part_6'],temp_df.loc[:,'Q11_Part_7']



new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7]).reset_index()      # Concating 7 columns of Q7.

new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()

new_df.columns = ['a','b']

new_df.iplot(kind='bar', x='a', y='b', title='Important Activities', xTitle='Activity', colors='brown')    

ml = df_mcr['Q10'].value_counts().reset_index()[:10]

ml.iplot(kind='bar',x='index', y='Q10', title='Use of ML ', 

             xTitle='Whether Ml is used or not', yTitle='Frequency',colors='indigo')

temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2,p3 = temp_df.loc[:,'Q13_Part_1'],temp_df.loc[:,'Q13_Part_2'],temp_df.loc[:,'Q13_Part_3'],

p4,p5  = temp_df.loc[:,'Q13_Part_4'],temp_df.loc[:,'Q13_Part_5'],

p6,p7,p8 = temp_df.loc[:,'Q13_Part_6'],temp_df.loc[:,'Q13_Part_7'],temp_df.loc[:,'Q13_Part_8']

p9,p10,p11 = temp_df.loc[:,'Q13_Part_9'],temp_df.loc[:,'Q13_Part_10'],temp_df.loc[:,'Q13_Part_11']

p12,p13,p14 = temp_df.loc[:,'Q13_Part_12'],temp_df.loc[:,'Q13_Part_13'],temp_df.loc[:,'Q13_Part_14']

p15 = temp_df.loc[:,'Q13_Part_15']



new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15]).reset_index()      # Concating 7 columns of Q7.

new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()

new_df.columns = ['a','b']

new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    

temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2,p3 = temp_df.loc[:,'Q14_Part_1'],temp_df.loc[:,'Q14_Part_2'],temp_df.loc[:,'Q14_Part_3'],

p4,p5  = temp_df.loc[:,'Q14_Part_4'],temp_df.loc[:,'Q14_Part_5'],

p6,p7,p8 = temp_df.loc[:,'Q14_Part_6'],temp_df.loc[:,'Q14_Part_7'],temp_df.loc[:,'Q14_Part_8']

p9,p10,p11 = temp_df.loc[:,'Q14_Part_9'],temp_df.loc[:,'Q14_Part_10'],temp_df.loc[:,'Q14_Part_11']

#p12,p13,p14 = temp_df.loc[:,'Q13_Part_12'],temp_df.loc[:,'Q13_Part_13'],temp_df.loc[:,'Q13_Part_14']

#p15 = temp_df.loc[:,'Q13_Part_15']



new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11]).reset_index()      # Concating 7 columns of Q7.

new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()

new_df.columns = ['a','b']

#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    

new_df.iplot(kind='pie', labels='a', values='b', title='Top ONLINE NOTEBOOKS', pull=0.2, hole=0.2)
temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2,p3 = temp_df.loc[:,'Q16_Part_1'],temp_df.loc[:,'Q16_Part_2'],temp_df.loc[:,'Q16_Part_3'],

p4,p5  = temp_df.loc[:,'Q16_Part_4'],temp_df.loc[:,'Q16_Part_5'],

p6,p7,p8 = temp_df.loc[:,'Q16_Part_6'],temp_df.loc[:,'Q16_Part_7'],temp_df.loc[:,'Q16_Part_8']

p9,p10,p11 = temp_df.loc[:,'Q16_Part_9'],temp_df.loc[:,'Q16_Part_10'],temp_df.loc[:,'Q16_Part_11']

p12,p13,p14 = temp_df.loc[:,'Q16_Part_12'],temp_df.loc[:,'Q16_Part_13'],temp_df.loc[:,'Q16_Part_14']

p15,p16,p17 = temp_df.loc[:,'Q16_Part_15'], temp_df.loc[:,'Q16_Part_16'], temp_df.loc[:,'Q16_Part_17']

p18 = temp_df.loc[:,'Q16_Part_18']



new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]).reset_index()      



new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()[:10]

new_df.columns = ['a','b']

#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    

new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Programming Languages', pull=0.2, hole=0.2)
temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2,p3 = temp_df.loc[:,'Q19_Part_1'],temp_df.loc[:,'Q19_Part_2'],temp_df.loc[:,'Q19_Part_3'],

p4,p5  = temp_df.loc[:,'Q19_Part_4'],temp_df.loc[:,'Q19_Part_5'],

p6,p7,p8 = temp_df.loc[:,'Q19_Part_6'],temp_df.loc[:,'Q19_Part_7'],temp_df.loc[:,'Q19_Part_8']

p9,p10,p11 = temp_df.loc[:,'Q19_Part_9'],temp_df.loc[:,'Q19_Part_10'],temp_df.loc[:,'Q19_Part_11']

p12,p13,p14 = temp_df.loc[:,'Q19_Part_12'],temp_df.loc[:,'Q19_Part_13'],temp_df.loc[:,'Q19_Part_14']

p15,p16,p17 = temp_df.loc[:,'Q19_Part_15'], temp_df.loc[:,'Q19_Part_16'], temp_df.loc[:,'Q19_Part_17']

p18 = temp_df.loc[:,'Q19_Part_18']



new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]).reset_index()      



new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()[:10]

new_df.columns = ['a','b']

#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    

new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Machine Learning Libraries', pull=0.2, hole=0.2)
temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2,p3 = temp_df.loc[:,'Q21_Part_1'],temp_df.loc[:,'Q21_Part_2'],temp_df.loc[:,'Q21_Part_3'],

p4,p5  = temp_df.loc[:,'Q21_Part_4'],temp_df.loc[:,'Q21_Part_5'],

p6,p7,p8 = temp_df.loc[:,'Q21_Part_6'],temp_df.loc[:,'Q21_Part_7'],temp_df.loc[:,'Q21_Part_8']

p9,p10,p11 = temp_df.loc[:,'Q21_Part_9'],temp_df.loc[:,'Q21_Part_10'],temp_df.loc[:,'Q21_Part_11']

p12,p13 = temp_df.loc[:,'Q21_Part_12'],temp_df.loc[:,'Q21_Part_13']

#p15,p16,p17 = temp_df.loc[:,'Q16_Part_15'], temp_df.loc[:,'Q16_Part_16'], temp_df.loc[:,'Q16_Part_17']

#p18 = temp_df.loc[:,'Q16_Part_18']



new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,]).reset_index()      



new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()[:10]

new_df.columns = ['a','b']

#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    

new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Libraries for Visualization', pull=0.2, hole=0.2)
time = df_mcr['Q24'].value_counts().reset_index()

time.iplot(kind='pie', labels='index', values='Q24',title='Time in Year ', pull=0.2, hole=0.2 )
temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2,p3 = temp_df.loc[:,'Q28_Part_1'],temp_df.loc[:,'Q28_Part_2'],temp_df.loc[:,'Q28_Part_3'],

p4,p5  = temp_df.loc[:,'Q28_Part_4'],temp_df.loc[:,'Q28_Part_5'],

p6,p7,p8 = temp_df.loc[:,'Q28_Part_6'],temp_df.loc[:,'Q28_Part_7'],temp_df.loc[:,'Q28_Part_8']

p9,p10,p11 = temp_df.loc[:,'Q28_Part_9'],temp_df.loc[:,'Q28_Part_10'],temp_df.loc[:,'Q28_Part_11']

p12,p13,p14 = temp_df.loc[:,'Q28_Part_12'],temp_df.loc[:,'Q28_Part_13'],temp_df.loc[:,'Q28_Part_14']

p15,p16,p17 = temp_df.loc[:,'Q28_Part_15'], temp_df.loc[:,'Q28_Part_16'], temp_df.loc[:,'Q28_Part_17']

p18 = temp_df.loc[:,'Q28_Part_18']



new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]).reset_index()      



new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()[:10]

new_df.columns = ['a','b']

#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    

new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Machine Learning Products', pull=0.2, hole=0.2)
temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2,p3 = temp_df.loc[:,'Q30_Part_1'],temp_df.loc[:,'Q30_Part_2'],temp_df.loc[:,'Q30_Part_3'],

p4,p5  = temp_df.loc[:,'Q30_Part_4'],temp_df.loc[:,'Q30_Part_5'],

p6,p7,p8 = temp_df.loc[:,'Q30_Part_6'],temp_df.loc[:,'Q30_Part_7'],temp_df.loc[:,'Q30_Part_8']

p9,p10,p11 = temp_df.loc[:,'Q30_Part_9'],temp_df.loc[:,'Q30_Part_10'],temp_df.loc[:,'Q30_Part_11']

p12,p13,p14 = temp_df.loc[:,'Q30_Part_12'],temp_df.loc[:,'Q30_Part_13'],temp_df.loc[:,'Q30_Part_14']

p15,p16,p17 = temp_df.loc[:,'Q30_Part_15'], temp_df.loc[:,'Q30_Part_16'], temp_df.loc[:,'Q30_Part_17']

p18 = temp_df.loc[:,'Q30_Part_18']



new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]).reset_index()      



new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()[:10]

new_df.columns = ['a','b']

#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    

new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Big Data and Analytic Products', pull=0.2, hole=0.2)
temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2,p3 = temp_df.loc[:,'Q31_Part_1'],temp_df.loc[:,'Q31_Part_2'],temp_df.loc[:,'Q31_Part_3'],

p4,p5  = temp_df.loc[:,'Q31_Part_4'],temp_df.loc[:,'Q31_Part_5'],

p6,p7,p8 = temp_df.loc[:,'Q31_Part_6'],temp_df.loc[:,'Q31_Part_7'],temp_df.loc[:,'Q31_Part_8']

p9,p10,p11 = temp_df.loc[:,'Q31_Part_9'],temp_df.loc[:,'Q31_Part_10'],temp_df.loc[:,'Q31_Part_11']

p12 = temp_df.loc[:,'Q31_Part_12']

#p15,p16,p17 = temp_df.loc[:,'Q30_Part_15'], temp_df.loc[:,'Q30_Part_16'], temp_df.loc[:,'Q30_Part_17']

#p18 = temp_df.loc[:,'Q30_Part_18']



new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12]).reset_index()      



new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()[:10]

new_df.columns = ['a','b']

#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    

new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Type of Data', pull=0.2, hole=0.2)
temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2,p3 = temp_df.loc[:,'Q33_Part_1'],temp_df.loc[:,'Q33_Part_2'],temp_df.loc[:,'Q33_Part_3'],

p4,p5  = temp_df.loc[:,'Q33_Part_4'],temp_df.loc[:,'Q33_Part_5'],

p6,p7,p8 = temp_df.loc[:,'Q33_Part_6'],temp_df.loc[:,'Q33_Part_7'],temp_df.loc[:,'Q33_Part_8']

p9,p10,p11 = temp_df.loc[:,'Q33_Part_9'],temp_df.loc[:,'Q33_Part_10'],temp_df.loc[:,'Q33_Part_11']

#p12 = temp_df.loc[:,'Q31_Part_12']

#p15,p16,p17 = temp_df.loc[:,'Q30_Part_15'], temp_df.loc[:,'Q30_Part_16'], temp_df.loc[:,'Q30_Part_17']

#p18 = temp_df.loc[:,'Q30_Part_18']



new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11]).reset_index()      



new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()[:10]

new_df.columns = ['a','b']

new_df.iplot(kind='bar', x='a', y='b', title='TOP Places for Dataset', xTitle='Place', colors='deepskyblue')    

#new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Type of Data', pull=0.2, hole=0.2)
online = df_mcr['Q37'].value_counts().reset_index()[:10]

online.iplot(kind='pie', labels='index', values='Q37',title='Top 10 Online Platforms to learn Data Science ',

             pull=0.2, hole=0.2 )
temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2,p3 = temp_df.loc[:,'Q38_Part_1'],temp_df.loc[:,'Q38_Part_2'],temp_df.loc[:,'Q38_Part_3'],

p4,p5  = temp_df.loc[:,'Q38_Part_4'],temp_df.loc[:,'Q38_Part_5'],

p6,p7,p8 = temp_df.loc[:,'Q38_Part_6'],temp_df.loc[:,'Q38_Part_7'],temp_df.loc[:,'Q38_Part_8']

p9,p10,p11 = temp_df.loc[:,'Q38_Part_9'],temp_df.loc[:,'Q38_Part_10'],temp_df.loc[:,'Q38_Part_11']

p12,p13,p14 = temp_df.loc[:,'Q38_Part_12'],temp_df.loc[:,'Q38_Part_13'],temp_df.loc[:,'Q38_Part_14']

p15,p16,p17 = temp_df.loc[:,'Q38_Part_15'], temp_df.loc[:,'Q38_Part_16'], temp_df.loc[:,'Q38_Part_17']

p18 = temp_df.loc[:,'Q38_Part_18']



new_df = pd.concat([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]).reset_index()      



new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()[:10]

new_df.columns = ['a','b']

#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    

new_df.iplot(kind='pie', labels='a', values='b', title='Top 10 Websites for Data Science News', pull=0.2, hole=0.2)
temp_df = df_mcr.drop([0],axis=0)

temp_df.head()



p1,p2 = temp_df.loc[:,'Q39_Part_1'],temp_df.loc[:,'Q39_Part_2']





new_df = pd.concat([p1,p2]).reset_index()      



new_df.dropna(inplace=True)

new_df = new_df[0].value_counts().reset_index()[:10]

new_df.columns = ['a','b']

#new_df.iplot(kind='bar', x='a', y='b', title='TOP IDE', xTitle='IDE', colors='deepskyblue')    

new_df.iplot(kind='pie', labels='a', values='b', title='View on quality of OnlineLearning and Bootcamp',

             pull=0.2, hole=0.2)
online = df_mcr['Q40'].value_counts().reset_index()[:10]

online.iplot(kind='bar', x='index', y='Q40',title='Academic Achievement VS Independent projects for Data Science ',

             xTitle='Comparision', colors='deepskyblue' )