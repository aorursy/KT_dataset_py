# titl_1: Nothing Can Stop You From Success

# titl_2: Beat by Compete

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from bokeh.plotting import figure

pd.set_option('display.max_columns', None)  

pd.set_option('display.max_rows', None)  

PATH = '/kaggle/input/kaggle-survey-2019/'

data = pd.read_csv(f'{PATH}multiple_choice_responses.csv', low_memory = False)

data.columns = data.iloc[0]

data = data.drop([0], axis=0)

plt.style.use("seaborn")

sns.set(font_scale=1)

gender = 'What is your gender? - Selected Choice'

country = 'In which country do you currently reside?'

salary = 'What is your current yearly compensation (approximate $USD)?'

age = 'What is your age (# years)?'

education = 'What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'

job_title = 'Select the title most similar to your current role (or most recent title if retired): - Selected Choice'



order = ['$0-999', '1,000-1,999', '2,000-2,999', '3,000-3,999', '4,000-4,999', '5,000-7,499',

         '7,500-9,999', '10,000-14,999', '15,000-19,999', '20,000-24,999', '25,000-29,999',

         '30,000-39,999', '40,000-49,999', '50,000-59,999', '60,000-69,999', '70,000-79,999',

         '80,000-89,999', '90,000-99,999', '100,000-124,999', '125,000-149,999', '150,000-199,999',

         '200,000-249,999', '250,000-299,999', '300,000-500,000', '> $500,000']



data_gender = data[(data[gender]=='Male') | (data[gender]=='Female')]



data[gender].replace(to_replace={"Prefer to self-describe":"Others",

                               "Prefer not to say" : "Others"}, inplace=True)

data_degree = data[(data[education]=='Master’s degree') |(data[education]=='Bachelor’s degree') |(data[education]=='Doctoral degree')];

data_1 = data_degree[(data_degree[country]=='India') |(data_degree[country]=='United States of America') | (data_degree[country]=='China')  | (data_degree[country]=='Russia')];



data.head()
# Plotting

plt.figure(figsize=(10,10))

ax = sns.countplot(x=gender, data=data,

                   linewidth=3,

                   edgecolor=sns.color_palette("dark", 1),

                   color = sns.color_palette()[0])

plt.xlabel('Gender', fontsize = 24);

plt.ylabel('Respondent Count', fontsize = 24);

plt.xticks(fontsize=18);

plt.yticks(fontsize=18);

tick_props = np.arange(0,1,0.10);

tick_names = ['{:0.2f}'.format(v) for v in tick_props];

plt.yticks(tick_props * data.shape[0], tick_names);
plt.figure(figsize=(20,10))



ax = sns.countplot(x=salary, hue=gender, data=data_gender,

                   linewidth=3,

                   edgecolor=sns.color_palette("dark", 1),

                   color = sns.color_palette()[0],

                  order = order)



plt.xlabel('Gender to Salary', fontsize = 24);

plt.ylabel('Respondent Count', fontsize = 24);

plt.xticks(fontsize=18, rotation=90);

plt.yticks(fontsize=18);

plt.legend(fontsize = 22);
plt.figure(figsize=(30,10))

ax = sns.countplot(x=job_title, hue=gender, data=data_gender,

                   linewidth=3,

                   edgecolor=sns.color_palette("dark", 1),

                   color = sns.color_palette()[0])



plt.xlabel('Job Title', fontsize = 24);

plt.ylabel('Respondent Count', fontsize = 24);

plt.xticks(fontsize=18, rotation=90);

plt.yticks(fontsize=18);

plt.legend(fontsize = 22);
plt.figure(figsize=(30,30))

ax = sns.countplot(y=country, data=data,

                   linewidth=3,

                   edgecolor=sns.color_palette("dark", 1),

                   color = sns.color_palette()[0],

                   order = data[country].value_counts().index)

plt.ylabel('Country Reside', fontsize = 24);

plt.xlabel('Respondent Count', fontsize = 24);

plt.xticks(fontsize=18);

plt.yticks(fontsize=18);
plt.figure(figsize=(30,10));

ax = sns.countplot(x=education, hue=country, data=data_1,

                   linewidth=3,

                   edgecolor=sns.color_palette("dark", 1),

                   color = sns.color_palette()[0]);

plt.xlabel('Country to Degree', fontsize = 24);

plt.ylabel('Respondent Count', fontsize = 24);

plt.legend(fontsize = 22);

plt.xticks(fontsize=18);

plt.yticks(fontsize=18);
# Plotting

plt.figure(figsize=(10,10))



ax = sns.countplot(x=age, data=data,

                   linewidth=3,

                   edgecolor=sns.color_palette("dark", 1),

                   color = sns.color_palette()[0],

                  order = data.groupby(age)[age].unique().index)

plt.xlabel('Age Range', fontsize = 24);

plt.ylabel('Respondent Count', fontsize = 24);

plt.xticks(fontsize=18, rotation = 90);

plt.yticks(fontsize=18);
plt.figure(figsize=(70,20))

ax = sns.countplot(x=salary, hue=age, data=data,

                   linewidth=3,

                   edgecolor=sns.color_palette("dark", 1),

                  order = order,

                  hue_order = data.groupby(age)[age].unique().index)

plt.xlabel('Age to Salary', fontsize = 30);

plt.ylabel('Respondent Count', fontsize = 30);

plt.xticks(fontsize=30, rotation = 90);

plt.yticks(fontsize=30);

plt.legend(fontsize = 30);
plt.figure(figsize=(20,10))



ax = sns.countplot(x=education,data=data,

                   linewidth=3,

                   edgecolor=sns.color_palette("dark", 1),

                   color = sns.color_palette()[0])

plt.xlabel('Education', fontsize = 24);

plt.ylabel('Respondent Count', fontsize = 24);

plt.xticks(fontsize=18, rotation = 90);

plt.yticks(fontsize=18);
plt.figure(figsize=(30,10))

ax = sns.countplot(x=job_title, hue=education, data=data,

                   linewidth=3,

                   edgecolor=sns.color_palette("dark", 1),

                   color = sns.color_palette()[0])

plt.xlabel('Education to Profession', fontsize = 24);

plt.ylabel('Respondent Count', fontsize = 24);

plt.xticks(fontsize=18, rotation = 90);

plt.yticks(fontsize=18);

plt.legend(fontsize = 18);
plt.figure(figsize=(30,10))

ax = sns.countplot(x=salary, hue=education, data=data_degree,

                   linewidth=3,

                   edgecolor=sns.color_palette("dark", 1),

                   color = sns.color_palette()[0], order=order)



plt.xlabel('Education to Salary', fontsize = 24);

plt.ylabel('Respondent Count', fontsize = 24);

plt.xticks(fontsize=18, rotation = 90);

plt.yticks(fontsize=18);

plt.legend(fontsize = 22);