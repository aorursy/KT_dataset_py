# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



directory = '/kaggle/input/kaggle-survey-2019/'



df_mcr = pd.read_csv(directory+'multiple_choice_responses.csv', low_memory=False)

df_questions = pd.read_csv(directory+'questions_only.csv', low_memory=False)

cols_to_use = ['Time from Start to Finish (seconds)', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8',

              'Q10', 'Q11', 'Q14', 'Q15', 'Q18_Part_1', 'Q18_Part_2', 'Q18_Part_3', 'Q18_Part_4', 'Q18_Part_5',

              'Q18_Part_6', 'Q18_Part_7', 'Q18_Part_8', 'Q18_Part_9', 'Q18_Part_10', 'Q18_Part_11', 'Q18_Part_12',

              'Q19', 'Q22', 'Q23']

df_mcr = df_mcr[cols_to_use]

df_mcr = df_mcr.drop([0], axis=0)

# Overall opinion of what language to learn first

language_one = df_mcr.loc[(df_mcr.Q19=='Python')|(df_mcr.Q19=='R')|(df_mcr.Q19=='SQL')|(df_mcr.Q19=='C++')|(df_mcr.Q19=='MATLAB')|(df_mcr.Q19=='C')|(df_mcr.Q19=='Java')|(df_mcr.Q19=='Javascript')|(df_mcr.Q19=='Bash')|(df_mcr.Q19=='Typescript')]['Q19'].value_counts(normalize=True)

fig, ax = plt.subplots()

ax.pie(language_one)

ax.legend(labels=language_one.index)

plt.show()



language_one
# Share of 'expert' first language recommendation

experts = df_mcr.loc[((df_mcr.Q19=='Python')|(df_mcr.Q19=='R')|(df_mcr.Q19=='SQL')|(df_mcr.Q19=='C++')|(df_mcr.Q19=='MATLAB')|(df_mcr.Q19=='C')|(df_mcr.Q19=='Java')|(df_mcr.Q19=='Javascript')|(df_mcr.Q19=='Bash')|(df_mcr.Q19=='Typescript'))&((df_mcr.Q23=='5-10 years')|(df_mcr.Q23=='10-15 years')|(df_mcr.Q23=='20+ years'))]['Q19'].value_counts(normalize=True)

fig, ax = plt.subplots()

ax.pie(experts)

ax.legend(labels=experts.index)

plt.show()



experts
for title in df_mcr.Q5.value_counts().index:

    title_vc = df_mcr.loc[((df_mcr.Q19=='Python')|(df_mcr.Q19=='R')|(df_mcr.Q19=='SQL')|(df_mcr.Q19=='C++')|(df_mcr.Q19=='MATLAB')|(df_mcr.Q19=='C')|(df_mcr.Q19=='Java')|(df_mcr.Q19=='Javascript')|(df_mcr.Q19=='Bash')|(df_mcr.Q19=='Typescript'))&((df_mcr.Q23=='5-10 years')|(df_mcr.Q23=='10-15 years')|(df_mcr.Q23=='20+ years'))&((df_mcr.Q5==title))]['Q19'].value_counts(normalize=True)

    print('\n\n'+title + ' W/ 5 or more years ML experience')

    fig, ax = plt.subplots()

    ax.pie(title_vc)

    ax.legend(title_vc.index)

    plt.show()

    print('% share of languages recommended by: '+title)

    print(title_vc)
learned = df_mcr.Q4.value_counts()

fig, ax = plt.subplots()

ax.barh(learned.index, learned)



plt.show()
for level in df_mcr.Q4.value_counts().index:

    ed_vc = df_mcr.loc[((df_mcr.Q19=='Python')|(df_mcr.Q19=='R')|(df_mcr.Q19=='SQL')|(df_mcr.Q19=='C++')|(df_mcr.Q19=='MATLAB')|(df_mcr.Q19=='C')|(df_mcr.Q19=='Java')|(df_mcr.Q19=='Javascript')|(df_mcr.Q19=='Bash')|(df_mcr.Q19=='Typescript'))&((df_mcr.Q23=='5-10 years')|(df_mcr.Q23=='10-15 years')|(df_mcr.Q23=='20+ years'))&((df_mcr.Q4==level))]['Q19'].value_counts(normalize=True)

    print('\n\n'+level + ' W/ 5 or more years ML experience:')

    fig, ax = plt.subplots()

    ax.pie(ed_vc)

    ax.legend(ed_vc.index)

    plt.show()

    print('% share of languages recommended by: '+level)

    print(ed_vc)