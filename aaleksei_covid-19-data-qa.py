# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json
df = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
isempty = df[(df['has_full_text'] == True) & (df['sha'] == 'NaN')].empty

print('Is the DataFrame empty :', isempty)
df_verified = df.dropna(subset=['sha'])

uniq_count = df_verified['sha'].drop_duplicates().count()

total_count = df_verified['sha'].count()

print('Uniq sha: ', uniq_count)

print('Total sha: ', total_count)

print(total_count - uniq_count)

df_verified[df_verified.duplicated(subset=['sha'],keep=False)].sort_values('sha')
import os.path

def full_path(sha,full_text_file):

    data_path_pattern = "/kaggle/input/CORD-19-research-challenge/{full_text_file}/{full_text_file}/{sha}.json"

    path = data_path_pattern.format(

            full_text_file=full_text_file,

            sha=sha

            )

    return path
def is_file_exists(shas, full_text_file):

        result = all(os.path.exists(full_path(sha,full_text_file)) for sha in shas.split("; "))    

        return result



df_uniq = df_verified.drop_duplicates(subset = ["sha"])

df_uniq['is_file_exists'] = df_uniq.apply(lambda x: is_file_exists(x['sha'], x['full_text_file']), axis=1)

    
len(df_uniq[df_uniq['is_file_exists'] == False])
# Example of wp's data



data_path_pattern = "/kaggle/input/CORD-19-research-challenge/{full_text_file}/{full_text_file}/{sha}.json"



path = data_path_pattern.format(

    full_text_file="custom_license",

    sha="be7e8df88e63d2579e8d61e2c3d716d57d347676"

)



with open(path, "r") as f:

    data = json.load(f)

def object_size(name, sha, full_text_file):

    path = full_path(sha,full_text_file)

    with open(path, "r") as f:

        data = json.load(f)

        return len(data[name])

    

def abstract_size(sha, full_text_file):

    return object_size('abstract', sha, full_text_file)



    

def body_size(sha, full_text_file):

    return object_size('body_text', sha, full_text_file)



def authors_size(sha, full_text_file):

    path = full_path(sha,full_text_file)

    with open(path, "r") as f:

        data = json.load(f)

        return len(data['metadata']['authors'])

    

from tqdm.notebook import trange, tqdm



df_uniq = df_verified.drop_duplicates(subset = ["sha"])[['sha','full_text_file']]

rows = []



for row in tqdm(df_uniq.iterrows()):

    _,(shas,full_text_file) = row

    for sha in shas.split("; "):

        new_row = {'sha': sha, 'full_text_file': full_text_file}

        rows.append(new_row)

        

df_test = pd.DataFrame(rows)
len(df_test)
df_test['is_file_exists'] = df_test.apply(lambda x: is_file_exists(x['sha'], x['full_text_file']), axis=1)
df_test['abstarct_size'] = df_test.apply(lambda x: abstract_size(x['sha'], x['full_text_file']), axis=1)
df_test['authors_size'] = df_test.apply(lambda x: authors_size(x['sha'], x['full_text_file']), axis=1)
df_test['body_size'] = df_test.apply(lambda x: body_size(x['sha'], x['full_text_file']), axis=1)
df_test['bib_entries_size'] = df_test.apply(lambda x: object_size('bib_entries', x['sha'], x['full_text_file']), axis=1)
df_test['authors_size'].describe()
from tqdm.notebook import trange, tqdm

df_uniq = df_verified.drop_duplicates(subset = ["sha"]).head(100)[['sha','full_text_file']]

for row in tqdm(df_uniq.iterrows()):

    _,(shas,full_text_file) = row

    for sha in shas.split("; "):

        path = data_path_pattern.format(

        full_text_file=full_text_file,

        sha=sha

        )

        with open(path, "r") as f:

            data = json.load(f)

            if (len(data['body_text']) == 1):

                print(data['body_text'])
duplicate_title_df = df_verified[df_verified.duplicated(subset=['title'],keep=False)].sort_values('title')

duplicate_title_df
duplicate_title_df['title'].value_counts()
df_verified['journal'].value_counts()
#Article should has a bib_entries > 3 items



from tqdm.notebook import trange, tqdm

df_uniq = df_verified.drop_duplicates(subset = ["sha"]).head(10)[['sha','full_text_file']]

for row in tqdm(df_uniq.iterrows()):

    _,(shas,full_text_file) = row

    for sha in shas.split("; "):

        path = data_path_pattern.format(

        full_text_file=full_text_file,

        sha=sha

        )

        with open(path, "r") as f:

            data = json.load(f)

            if (len(data['bib_entries']) < 3):

                print(data['bib_entries'])

                print(sha)
#Article should has a **Cred-Test-2****: article should have at least 3 authors > 3 items



from tqdm.notebook import trange, tqdm

i = 0

df_uniq = df_verified.drop_duplicates(subset = ["sha"])[['sha','full_text_file']]

for row in tqdm(df_uniq.iterrows()):

    _,(shas,full_text_file) = row

    for sha in shas.split("; "):

        path = data_path_pattern.format(

        full_text_file=full_text_file,

        sha=sha

        )



        with open(path, "r") as f:

            data = json.load(f)

            if (len(data['metadata']['authors']) < 2):

                i += 1



print(i)