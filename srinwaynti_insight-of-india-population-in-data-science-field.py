# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
MCR=pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv")

questions=pd.read_csv("/kaggle/input/kaggle-survey-2019/questions_only.csv")

survey_schema=pd.read_csv("/kaggle/input/kaggle-survey-2019/survey_schema.csv")

other_text_response=pd.read_csv("/kaggle/input/kaggle-survey-2019/other_text_responses.csv")
MCR.head()
MCR.shape
MCR_description=MCR.loc[0]

MCR_columns=MCR.columns
for _ in range(0,len(MCR_description)-1):

    print(MCR_columns[_],"==>",MCR_description[_])
MCR.rename(columns = {"Q1": "Age", 

                     "Q2":"Gender",

                     "Q3":"Country",

                     "Q4":"Educational Qualification",

                     "Q5":"Current Role",

                     "Q6":"Size of Company",

                      "Q7":"Persons responsible for Data Science",

                      "Q8":"Does your current employer incorporate machine learning methods into their business?",

                      "Q10":"Salary",

                      "Q11": "Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?",

            

                     }, 

                                 inplace = True) 
MCR.drop(0,inplace=True)

MCR.head()
MCR.drop(['Q2_OTHER_TEXT','Q5_OTHER_TEXT'],axis=1,inplace=True)
MCR.drop('Time from Start to Finish (seconds)',axis=1,inplace=True)

MCR.head()
MCR.Age.value_counts()
sns.countplot(MCR.Age)
MCR[MCR['Age']=='25-29'].Country.value_counts().head(10).plot(kind='barh')
MCR['Educational Qualification'].value_counts()
MCR[MCR['Country']=='India']['Educational Qualification'].value_counts().plot(kind='barh')
MCR.Gender.value_counts()
MCR[MCR['Country']=='India'].Gender.value_counts().plot(kind='barh')
MCR['Current Role'].value_counts()
MCR[MCR['Country']=='India']['Current Role'].value_counts().plot(kind='barh')
MCR['Size of Company'].value_counts()
MCR[MCR['Country']=='India']['Size of Company'].value_counts().plot(kind='barh')