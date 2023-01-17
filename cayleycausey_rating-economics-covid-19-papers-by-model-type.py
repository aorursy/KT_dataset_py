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
metadata = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")

metadata.head(5)
metadata.title

count_econ = 0

title_list = []

for title in metadata.title:

    title = str(title)

    if title.find("economics") >= 0:

        title_list.append(title)

        count_econ += 1

    elif title.find("economy") >= 0:

        title_list.append(title)

        count_econ +=1

    elif title.find("economic") >= 0:

        title_list.append(title)

        count_econ +=1

print(f"There are {count_econ} papers that have to do with coronavirus and the economy.")
economics_metadata = metadata.loc[metadata['title'].isin(title_list)]

economics_metadata.head(5)
economics_metadata['full_text_file'].value_counts()
file_ids = economics_metadata.sha

file_pmcids = economics_metadata.pmcid

economics_file_names = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        for file_id in file_ids:

            file_id = str(file_id)

            if filename.find(file_id) >= 0:

                economics_file_names.append(f'{dirname}/{filename}')

        for file_pmcid in file_pmcids:

            file_pmcid = str(file_pmcid)

            if filename.find(file_pmcid) >= 0:

                economics_file_names.append(f'{dirname}/{filename}')
import json
economics_file_names

paper_rating = {}

paragraph_list = []

for file_path in economics_file_names:

    with open(file_path) as file:

        paper_content = json.load(file)

        body = paper_content['body_text']

        for paragraph in body:

            paragraph = str(paragraph)

            if paragraph.find('heterogen') > -1:

                paper_rating[file_path] = 1

                paragraph_list.append(paragraph)

            elif paragraph.find('random') > -1 or paragraph.find('experimental') > -1:

                paper_rating[file_path] = 2

            elif paragraph.find('non-random') > -1:

                paper_rating[file_path] = 3

            elif paragraph.find('cohort') > -1:

                paper_rating[file_path] = 3

            elif paragraph.find('cross-section') > -1:

                paper_rating[file_path] = 3

            elif paragraph.find('case study') > -1:

                paper_rating[file_path] = 4

print(f"This function was able to rate {len(paper_rating)} papers.")
paper_rating.keys()

df_sha = pd.DataFrame()

df_pmc = pd.DataFrame()

sha_list = []

pmc_list = []

for key in paper_rating.keys():

    key = str(key)

    key = key.split('/')

    key = key[-1]

    key = key.split('.')

    key = key[0]

    if key[0:3] == 'PMC':

        pmc_list.append(key)

        sha_list.append('NaN')

    else:

        sha_list.append(key)

        pmc_list.append('Nan')

df_sha['sha'] = sha_list

df_pmc['pmcid'] = pmc_list

df_sha['rating'] = paper_rating.values()

df_pmc['rating'] = paper_rating.values()
df1 = pd.DataFrame.merge(economics_metadata,df_sha, on = 'sha', how = 'left')

df_final = pd.DataFrame.merge(df1, df_pmc, on = 'pmcid', how = 'left')

df_final.head(5)
rating = []

for x, y in zip(df_final.rating_x, df_final.rating_y):

    x = str(x)

    y = str(y)

    z = ''

    if x == 'nan':

        if y == 'nan':

            z = 'NaN'

        else:

            z = y

    else:

        z = x

    rating.append(z)

len(rating)

df_final['rating'] = rating
df_final = df_final.drop(columns = ['rating_x','rating_y'])

df_final.head(5)