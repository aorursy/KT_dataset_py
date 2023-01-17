# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json

from tqdm import tqdm, tqdm_notebook



from pandas import option_context
def get_v1_date(versions):

    # Use this function if you plan to use the first submitted date to filter the data instead of update date

    return versions[0]['created'].split(",")[1][1:11]
data = []

with open("../input/arxiv/arxiv-metadata-oai-snapshot.json", "r") as i:

    for line in tqdm(i):

        data.append(json.loads(line))

        

data = pd.DataFrame(data)

data.head()
data['update_date'] = pd.to_datetime(data['update_date'])

data['update_date'].dt.year.value_counts()
data_cy = data.loc[data['update_date'].dt.year == 2020,:].reset_index(drop=True)

data_cy.shape
# data['first_submitted_date'] = data_cy['versions'].apply(lambda x: get_v1_date(x)).apply(lambda p: pd.to_datetime(p, "coerce"))
data_cy_phish = data_cy.loc[data_cy['abstract'].str.contains("phish|Phish|social engineering"), :].reset_index(drop=True)

print(data_cy_phish.shape)

data_cy_phish.head()
data_cy_phish_out = (data_cy_phish

                     .pipe(lambda x: x[['id', 'title', 'update_date']]

                     .assign(url = lambda df: "https://arxiv.org/pdf/" + df['id']))

                     .pipe(lambda x: x[['update_date', 'url', 'title']])

                     .sort_values("update_date", ascending=False)

                     .reset_index(drop=True)

                    )





with option_context('display.max_colwidth', 400):

    display(data_cy_phish_out)