# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

Washington_Health_Workforce_Survey_Data = pd.read_csv("../input/washington-health-workforce-survey-data/Washington_Health_Workforce_Survey_Data.csv")
df=Washington_Health_Workforce_Survey_Data["CredentialType"].value_counts()

df


df=Washington_Health_Workforce_Survey_Data

df=df[['CredentialType','Sex']]

cred_filter=df['CredentialType']=='Dentist License'

lang_filter=df['Sex']=="Female"

df.loc[cred_filter & lang_filter, :]

#pc=Washington_Health_Workforce_Survey_Data["PrimaryPracticeCity"].value_counts()

df=Washington_Health_Workforce_Survey_Data

df=df[['CredentialType','PrimaryPracticeCity']]

city_filter=df['PrimaryPracticeCity']=="Yakima"

df=df.loc[cred_filter & city_filter, :]



df['PrimaryPracticeCity'].value_counts()
df=Washington_Health_Workforce_Survey_Data



df['HighestEducationOnline'].value_counts().plot(kind="pie")
df=Washington_Health_Workforce_Survey_Data



df['CommunicateOtherLanguage'].value_counts().plot(kind="barh")
df=Washington_Health_Workforce_Survey_Data



def age (year): 

    if year <1964:

        return "Age 56-74"

    elif year >1965 and year <=1979:

        return "Age 41-55"

    elif year >1975 and year <=1985:

        return "Age 35-45"

    elif year >1986 and year <=1994:

        return "Age 26-34"



    

df['BirthYear'].apply(age).value_counts().plot(kind="bar")
df=Washington_Health_Workforce_Survey_Data



df['WorkStatus'].value_counts().plot(kind="bar")