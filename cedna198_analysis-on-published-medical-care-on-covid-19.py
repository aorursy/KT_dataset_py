#Import libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import string

import xgboost as xgb





plt.rcParams.update({'font.size': 14})

# Load data



meta_dt = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

print (meta_dt.shape)
meta_dt.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    meta_dt['has_pdf_parse'], 

    meta_dt['title'], 

    random_state = 1

)



print("Training dataset: ", X_train.shape[0])

print("Test dataset: ", X_test.shape[0])
meta_dt.duplicated().sum()

meta_dt = meta_dt.drop_duplicates().reset_index(drop=True)

sns.countplot(y=meta_dt.has_pdf_parse);
meta_dt.has_pdf_parse.value_counts()

meta_dt.isnull().sum()
# Checking all the full text files on the metadata.csv files for uniques keywords



print (meta_dt.has_pmc_xml_parse.nunique())

print (meta_dt.has_pdf_parse.nunique())

# Most common keywords found



plt.figure(figsize=(9,6))

sns.countplot(y=meta_dt.has_pmc_xml_parse, order = meta_dt.has_pmc_xml_parse.value_counts().iloc[:15].index)

plt.title('Files containing the keywords')

plt.show()



# meta_dt.keyword.value_counts().head(10)
# Check number of unique keywords and journals



print (meta_dt.journal.nunique())
# Most common journals we have



plt.figure(figsize=(9,6))

sns.countplot(y=meta_dt.journal, order = meta_dt.journal.value_counts().iloc[:15].index)

plt.title('Top 15 journals')

plt.show()
# Most common titles of the articles we have



plt.figure(figsize=(9,6))

sns.countplot(y=meta_dt.title, order = meta_dt.title.value_counts().iloc[:15].index)

plt.title('Top 15 Titles')

plt.show()
raw_loc = meta_dt.title.value_counts()

top_loc = list(raw_loc[raw_loc>=10].index)

top_only = meta_dt[meta_dt.title.isin(top_loc)]



top_l = top_only.groupby('title').mean()['has_pdf_parse'].sort_values(ascending=False)

plt.figure(figsize=(6,6))

sns.barplot(x=top_l.index, y=top_l)

plt.axhline(np.mean(meta_dt.has_pdf_parse))

plt.xticks(rotation=80)

plt.show()
for col in ['title','has_pdf_parse']:

    meta_dt[col] = meta_dt[col].fillna('None')

   

def clean_loc(x):

    if x == 'None':

        return 'None'

    elif x == 'Corona' or x =='Corona Virus' or x == 'Covid-19':

        return 'Covid-19'

    elif 'Virus' in x or 'Viral' in x:

        return 'Virus'    

    elif 'Viruses' in x:

        return 'Viruses'

    elif 'Virology' in x:

        return 'Virology'

    elif 'Vaccine' in x and 'Vaccines' in x and 'Vaccinantion' in x:

        return 'Vaccine'

    elif x in top_loc:

        return x

    else: return 'Others'

    

meta_dt['title_clean'] = meta_dt['title'].apply(lambda x: clean_loc(str(x)))
top_l2 = meta_dt.groupby('title_clean').mean()['has_pdf_parse'].sort_values(ascending=False)

plt.figure(figsize=(14,10))

sns.barplot(x=top_l2.index, y=top_l2)

plt.axhline(np.mean(meta_dt.has_pdf_parse))

plt.xticks(rotation=80)

plt.show()
meta_dt.to_csv('submission.csv', index = False)