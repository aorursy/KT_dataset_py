# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

counter = 0

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
meta_data = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')



#drop duplicates based on sha code

meta_data.drop_duplicates(subset=['sha', 'doi', 'pubmed_id'], inplace=True)



print(meta_data.shape)

meta_data.head()
def doi_url(d):

    if d.startswith('http://'):

        return d

    elif d.startswith('doi.org'):

        return f'http://{d}'

    else:

        return f'http://doi.org/{d}'
# Long process, run locally and uploaded file manually

def is_valid_doi_url(doi_url):

    print(f'Checking {doi_url}', end='\t')

    ret_doi = None

    try:

        resp = requests.get(doi_url, timeout=5)

        if resp.status_code == 200:

            ret_doi = doi_url

            print('OK')

        else:

            print('No OK')

    except:

        print('No OK')

    return ret_doi
meta_data.doi = meta_data.doi.fillna('').apply(doi_url)
meta_data.to_csv('clean_metadata.csv')