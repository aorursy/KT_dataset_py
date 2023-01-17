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

import matplotlib.pyplot as plt # Data Visualisation

import seaborn as sns # Data Visualisation

import re # For Text finding



import warnings

warnings.filterwarnings('ignore')
# To see all the columns display max columns bby 500

pd.set_option("display.max_columns",500)



# Uploading the datasets

mcr = pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv")

ocr = pd.read_csv("/kaggle/input/kaggle-survey-2019/other_text_responses.csv")

# Reading the data sets to get feel of it:

mcr.head()
ocr.head()
# Dropping the _OTHER_TEXT columns in the mcr dataset as these columns are not required for analysis

to_drop = []

for i in mcr.columns:

    if re.search("_OTHER_TEXT",i):

        to_drop.append(i)



mcr.drop(to_drop, axis=1, inplace=True)
# Concatinating the two datasets



df = pd.concat([mcr, ocr], axis=1)



#Reading the final Dataset



df.head()
# Removing the first row of the dataset 

df = df.iloc[1:,:]
# Let's filter the data out and remove the othe countries

df_india = df.loc[(df.Q3=="India"),:]
df_india.head()
plt.figure(figsize=(10,10))

df_india.Q1.value_counts().plot(kind='bar')

plt.ylabel("Counts")

plt.xlabel("Age Groups")

plt.title("Pressence of Age Group")

plt.show()
plt.figure(figsize=(15,8))



plt.subplot(1,2,1)

df_india.Q2.value_counts().plot(kind="bar")

plt.ylabel('Counts')

plt.xlabel("Gender")

plt.title("Gender Distribution of Kagglers in India")



df = df.loc[(df.Q2=="Female") & ((df.Q3=="India") | (df.Q3=="United States of America") | (df.Q3=="China") | (df.Q3=="Japan")), :]

plt.subplot(1,2,2)

df.Q3.value_counts().plot(kind='bar')

plt.ylabel("Counts")

plt.xlabel("Countries")

plt.title("India Vs Technology Giants in Women Participation")



plt.show()
plt.figure(figsize=(14,8))



plt.subplot(1,2,1)

df_india.Q4.value_counts().plot(kind='bar')

plt.xticks(rotation=90)

plt.ylabel("Counts")

plt.xlabel("Formal Education")

plt.title("Education")



df_degree = pd.DataFrame(df_india.Q4.value_counts())

def func(pct, allval):

    absolute= int((pct/np.sum(allval))*100)

    return "{:.1f}".format(pct, absolute)

plt.subplot(1,2,2)

plt.pie(df_degree['Q4'], autopct= lambda pct: func(pct, df_degree["Q4"]), labels=df_degree.index)

plt.title("Education Distribution")

plt.show()

# Replacing some column names with the Max occuring term for further analysis

for idx in df_india.columns:

    for jdx in re.findall("_Part", idx):

        try:

            df_india.fillna("NA", inplace=True)

            question = idx[:idx.index("_")+1]

            for kdx in set(df_india[idx]):

                if (kdx!="NA"):

                    col_new = kdx.split("(")[0].strip()

            df_india[idx] = df_india[idx].apply(lambda x: 0 if x=="NA" else 1)

            df_india.rename(columns={idx:question + col_new},inplace=True)

        except:

            pass

# Checking the new Dataset

df_india.head()
#Favorite media source

dict_media_source=dict()

cols=["Q12_Twitter","Q12_Hacker News","Q12_Reddit","Q12_Kaggle","Q12_Course Forums","Q12_YouTube","Q12_Podcasts","Q12_Blogs","Q12_Journal Publications","Q12_Slack Communities","Q12_None","Q12_Other"]

for col in cols:

      dict_media_source[col.split("_")[1]] = [df_india[col].value_counts()[0], df_india[col].value_counts()[1]]

df_media_source=pd.DataFrame(dict_media_source)

df_media_source=df_media_source.T

df_media_source.sort_values(by=(df_media_source.columns[1]), ascending=False, inplace=True)



# Let's see the distribution of the media sources

ax = df_media_source[[1]].plot(kind='bar', title ="Media Source", figsize=(10, 10), legend=False, fontsize=12)

plt.ylabel("Counts")

plt.xlabel("Media Sources")



rects = ax.patches

labels = (df_media_source[1] * 100/df_media_source[1].sum()).round(2).values.tolist()

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, str(label) + " %",

            ha='center', va='bottom')

    

plt.show()

#Most Used Platform

dict_platform=dict()

cols=["Q13_Udacity","Q13_Coursera","Q13_edX","Q13_DataCamp","Q13_DataQuest","Q13_Kaggle Courses","Q13_Fast.ai","Q13_Udemy","Q13_LinkedIn Learning","Q13_University Courses","Q13_None","Q13_Other"]

for col in cols:

      dict_platform[col.split("_")[1]] = [df_india[col].value_counts()[0], df_india[col].value_counts()[1]]

df_platform=pd.DataFrame(dict_platform)

df_platform=df_platform.T

df_platform.sort_values(by=(df_platform.columns[1]), ascending=False, inplace=True)



# Distribution of Platform from which individuals starts learning Data Science, Machine Learning

ax = df_platform[[1]].plot(kind='bar', figsize=(10, 10), legend=False, fontsize=12)

plt.ylabel("Counts")

plt.xlabel("Platforms")

plt.title("Learning Platforms")



rects = ax.patches

labels = (df_platform[1] * 100/df_platform[1].sum()).round(2).values.tolist()

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, str(label) + " %",

            ha='center', va='bottom')

    

plt.show()
#Most Used notebooks 

dict_notebooks=dict()

cols=["Q16_Jupyter","Q16_RStudio","Q16_PyCharm","Q16_Atom","Q16_MATLAB","Q16_Visual Studio / Visual Studio Code","Q16_Spyder","Q16_Vim / Emacs","Q16_Notepad++","Q16_Sublime Text","Q16_None","Q16_Other","Q17_Kaggle Notebooks","Q17_Google Colab","Q17_Microsoft Azure Notebooks","Q17_Google Cloud Notebook Products","Q17_Paperspace / Gradient","Q17_FloydHub","Q17_Binder / JupyterHub","Q17_IBM Watson Studio","Q17_Code Ocean","Q17_AWS Notebook Products"]

for col in cols:

      dict_notebooks[col.split("_")[1]] = [df_india[col].value_counts()[0], df_india[col].value_counts()[1]]

df_notebooks=pd.DataFrame(dict_notebooks)

df_notebooks=df_notebooks.T

df_notebooks.sort_values(by=(df_notebooks.columns[1]), ascending=False, inplace=True)



# Most Used Programming languages/tools

dict_tools=dict()

cols=["Q18_Python","Q18_R","Q18_SQL","Q18_C","Q18_C++","Q18_Java","Q18_Javascript","Q18_TypeScript","Q18_Bash","Q18_MATLAB","Q18_None","Q18_Other"]

for col in cols:

      dict_tools[col.split("_")[1]] = [df_india[col].value_counts()[0], df_india[col].value_counts()[1]]

df_tools=pd.DataFrame(dict_tools)

df_tools = df_tools.T

df_tools.sort_values(by=(df_tools.columns[1]), ascending=False, inplace=True)



# Distribution of tools and notebooks 

plt.figure(figsize=(14,8))



plt.subplot(1,2,1)

ax = df_notebooks[1].plot(kind='bar')

plt.ylabel("Counts")

plt.xlabel("Notebook")

plt.title("Different Notebooks")

rects = ax.patches

labels = (df_notebooks[1] * 100/df_notebooks[1].sum()).round(2).values.tolist()

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, str(label) + " %",

            ha='center', va='bottom',rotation='vertical')

    

plt.subplot(1,2,2)

ax = df_tools[1].plot(kind='bar')

plt.ylabel("Counts")

plt.xlabel("Tools")

plt.title("Different Programming Languages")

rects = ax.patches

labels = (df_tools[1] * 100/df_tools[1].sum()).round(2).values.tolist()

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width() / 2, height + 5, str(label) + " %",

            ha='center', va='bottom',rotation='vertical')

plt.show()
# Current role or profile 

plt.figure(figsize=(14,14))



plt.subplot(2,1,1)

df.Q5.value_counts().plot(kind='barh')

plt.xticks(rotation=90)

plt.ylabel("Counts")

plt.xlabel("Profiles")

plt.title("Current Profile of Responders")



df_role = pd.DataFrame(df_india.Q5.value_counts())

def func1(pct, allval):

    absolute= int((pct/np.sum(allval))*100)

    return "{:.1f}".format(pct, absolute)



plt.subplot(2,1,2)

plt.pie(df_role['Q5'], autopct=lambda pct: func1(pct, df_role["Q5"]), labels=df_role.index)



plt.show()
