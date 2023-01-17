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



import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

from matplotlib import cm



import numpy as np

import seaborn as sns



import plotly.express as px
data_mentalhealth = pd.read_csv("../input/mental-health-in-tech-survey/survey.csv")

data_mentalhealth
# Assign default values for each data type

defaultInt = 0

defaultString = 'NaN'

defaultFloat = 0.0



intColumn = ['Age']

stringColumn = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',

                 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',

                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',

                 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',

                 'seek_help']

floatColumn = []



# Clean the NaN's

for feature in data_mentalhealth:

    if feature in intColumn:

        data_mentalhealth[feature] = data_mentalhealth[feature].fillna(defaultInt)

    elif feature in stringColumn:

        data_mentalhealth[feature] = data_mentalhealth[feature].fillna(defaultString)

    elif feature in floatColumn:

        data_mentalhealth[feature] = data_mentalhealth[feature].fillna(defaultFloat)

    else:

        print('Error: Feature %s not recognized.' % feature)

        

data_mentalhealth.head(5)
# pra processing gender variable

male = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]

transgender = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           

female = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]



for (row, col) in data_mentalhealth.iterrows():



    if str.lower(col.Gender) in male:

        data_mentalhealth['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)



    if str.lower(col.Gender) in female:

        data_mentalhealth['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)



    if str.lower(col.Gender) in transgender:

        data_mentalhealth['Gender'].replace(to_replace=col.Gender, value='transgender', inplace=True)



#Get rid of bullshit

stk_list = ['A little about you', 'p']

data_mentalhealth = data_mentalhealth[~data_mentalhealth['Gender'].isin(stk_list)]
# handle age missing value

data_mentalhealth['Age'].fillna(data_mentalhealth['Age'].median(), inplace = True)
# handle self_employed missing value

data_mentalhealth['self_employed'] = data_mentalhealth['self_employed'].replace([defaultString], 'No')
# handle work_interfere missing value

data_mentalhealth['work_interfere'] = data_mentalhealth['work_interfere'].replace([defaultString], 'Don\'t know' )
# drop some column

data_mentalhealth = data_mentalhealth.drop(['state', 'Timestamp', 'comments'], axis=1)

data_mentalhealth
# mengelompokkan responden berdasarkan background

respondent_data_bg = data_mentalhealth.groupby('tech_company').count()

respondent_data_bg = respondent_data_bg.rename(index={'No': 'Perusahan Non IT', 'Yes': 'Perusahaan IT'})
# visualtization

plt.pie(

      respondent_data_bg['treatment'],  # data

      labels=respondent_data_bg.index, # give label

      startangle=90, 

      autopct='%.2f%%', # set format value of label

      explode=(0.2, 0.1) # supaya mencar satu

)

plt.title('Background Pekerjaan Responden')

plt.show()
# mengelompokkan data berdasar butuh treatment ato nggak

respondent_data_needtreatment = data_mentalhealth.groupby('treatment').count()

respondent_data_needtreatment = respondent_data_needtreatment.rename(index={'No': 'Tidak butuh \'Treatment\'', 'Yes': 'Butuh \'Treatment\''})
worker_data = data_mentalhealth.groupby('treatment').apply(

                              lambda x: pd.Series(

                                  dict(

                                    non_techcompany_worker=(x.tech_company == 'Yes').sum(),

                                    techcompany_worker=(x.tech_company == 'No').sum()

                                )

                              )

                          )

worker_data = worker_data.rename(index={'No': 'Tidak butuh', 'Yes': 'Butuh'})
plt.pie(

      respondent_data_needtreatment['tech_company'],  # data

      labels=respondent_data_needtreatment.index, # give label

      startangle=90, 

      autopct='%.2f%%', # set format value of label

)

plt.title('Kebutuhan Perawatan Menurut Responden')

plt.show()
plt.pie(

      worker_data['non_techcompany_worker'],  # data

      labels=worker_data.index, # give label

      startangle=90, 

      autopct='%.2f%%', # set format value of label

)

plt.title('Kebutuhan Perawatan Kesehatan Mental di Pekerja Perusahaan Non IT')

plt.show()
plt.pie(

      worker_data['techcompany_worker'],  # data

      labels=worker_data.index, # give label

      startangle=90, 

      autopct='%.2f%%', # set format value of label

)

plt.title('Kebutuhan Perawatan Kesehatan Mental di Pekerja Perusahaan IT')

plt.show()
treatment_need_data = data_mentalhealth.groupby('tech_company').apply(

                              lambda x: pd.Series(

                                  dict(

                                    need_treatment=(x.treatment == 'Yes').sum(),

                                    dont_need_treatment=(x.treatment == 'No').sum()

                                )

                              )

                          )

treatment_need_data = treatment_need_data.rename(index={'No': 'Pekerja di Perusahaan Non IT', 'Yes': 'Pekerja di Perusahaan IT'})
plt.pie(

      treatment_need_data['need_treatment'],  # data

      labels=treatment_need_data.index, # give label

      startangle=90, 

      autopct='%.2f%%', # set format value of label,

      explode=(0.1, 0.05) # supaya mencar satu

)

plt.title('Kebutuhan \'Treatment\' Untuk Kesehatan Mental')

plt.show()
plt.subplots(figsize=(12,6)) # set plot size

sns.countplot(data = data_mentalhealth, x = 'no_employees', hue ='care_options' )

#ticks = plt.setp(ax.get_xticklabels(),rotation=45)

plt.title('Jumlah Karyawan vs Layanan Perawatan')
data_mentalhealth['Age'] = pd.to_numeric(data_mentalhealth['Age'],errors='coerce')

def age_process(Age):

    if Age>=0 and Age<=100:

        return Age

    else:

        return np.nan

data_mentalhealth['Age'] = data_mentalhealth['Age'].apply(age_process)
plt.subplots(figsize=(10,6))

sns.distplot(

    data_mentalhealth['Age'].dropna(),

    kde=True,

    color='#0d47a1'

    )

plt.title('Distribusi Berdasar Usia Responden')

plt.ylabel('Frekuensi')
g = sns.FacetGrid(data_mentalhealth, col='treatment', height=7,margin_titles=True)

g = g.map(sns.distplot, "Age")
sns.catplot(x='seek_help', hue='treatment', row='family_history', kind='count', data=data_mentalhealth, orient = 'horizontal')
# make color for barplot

color = cm.inferno_r(np.linspace(.4,.8, 10))

data_mentalhealth.groupby('Country').treatment.count().nlargest(10).plot(

                                                                                    kind = "barh",

                                                                                    figsize=(12,6), 

                                                                                    color= color,

                                                                                    title = "Negara Dengan Kebutuhan Treatment Tertinggi",

                                                                                    )
# make color for barplot

color = cm.inferno_r(np.linspace(.4,.8, 30))

data_mentalhealth.groupby(['Country','Gender']).treatment.count().nlargest(10).plot(

                                                                                    kind = "barh",

                                                                                    figsize=(12,6), 

                                                                                    color= color,

                                                                                    title = "Negara Dengan Kebutuhan Treatment Tertinggi Berdasarkan Jenis Kelamin",

                                                                                    )