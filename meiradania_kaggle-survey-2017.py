# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
multiple_choice_df = pd.read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding='latin-1', low_memory=False)

multiple_choice_df.head()
countries = pd.read_csv("../input/countrieseurope/Countries-Europe.csv", encoding='latin-1')

countries.head()
worklife_questions = ["EmployerSize", "EmployerIndustry", "CurrentJobTitleSelect", "EmploymentStatus", "CodeWriter"]



europe_df = multiple_choice_df.loc[multiple_choice_df.Country.isin(countries.name),

                                   multiple_choice_df.columns.isin(worklife_questions+["Country"]) | (multiple_choice_df.columns.str.contains('WorkChallenge')) ] 

europe_df.shape
europe_df.CodeWriter.value_counts()
questions = pd.read_csv("../input/kaggle-survey-2017/schema.csv", encoding='latin-1', low_memory=False)

questions.loc[questions.Column.str.contains('WorkChallenge'),:]
europe_df_coding_workers = europe_df.loc[(europe_df.EmploymentStatus.isin(["Employed full-time", "Employed part-time", "Independent contractor, freelancer, or self-employed"])) &

                                         (europe_df.CodeWriter=='Yes')

                                         ,:]

print(europe_df_coding_workers.shape)

europe_df_coding_workers.EmploymentStatus.value_counts()

# CodingWorker: Respondents who indicated that they were "Employed full-time", "Employed part-time", or an "Independent contractor, freelancer, or self-employed" AND that they write code to analyze data in their current job
europe_df_coding_workers.Country.value_counts()
europe_df_coding_workers.CurrentJobTitleSelect.value_counts()
europe_df_coding_workers.EmployerIndustry.value_counts()
europe_df_coding_workers.columns[europe_df_coding_workers.columns.str.contains('WorkChallenge')]
for challenge in europe_df_coding_workers.columns[europe_df_coding_workers.columns.str.contains('WorkChallenge')]:

    if challenge not in ['WorkChallengesSelect', 'WorkChallengeFrequencyOtherSelect']:

        display(europe_df_coding_workers.loc[:,challenge].value_counts())
pd.crosstab(europe_df_coding_workers.Country, europe_df_coding_workers.WorkChallengeFrequencyExpectations, margins='row')
europe_df_coding_workers.loc[:,'WorkChallengesSelectSplit'] = europe_df_coding_workers.WorkChallengesSelect.apply(lambda x: str(x).split(','))

europe_df_coding_workers[['WorkChallengesSelect','WorkChallengesSelectSplit']].head()
a = pd.Series([item for sublist in europe_df_coding_workers.WorkChallengesSelectSplit for item in sublist])

total = europe_df_coding_workers.shape[0]

df_workchallenges = a.value_counts().reset_index().rename(columns={'index':'Work Challenge', 0:'Respondents'})

df_workchallenges['Share'] = df_workchallenges.loc[:,'Respondents'] / total

df_workchallenges
europe_df_coding_workers.loc[europe_df_coding_workers.WorkChallengesSelect.isnull()].EmploymentStatus.value_counts()