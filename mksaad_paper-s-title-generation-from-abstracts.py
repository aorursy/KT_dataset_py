# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os, psutil  



def cpu_stats():

    pid = os.getpid()

    py = psutil.Process(pid)

    memory_use = py.memory_info()[0] / 2. ** 30

    return 'memory GB:' + str(np.round(memory_use, 2))
cpu_stats()
data_file = '../input/arxiv/arxiv-metadata-oai-snapshot.json'
def get_metadata():

    with open(data_file, 'r') as f:

        for line in f:

            yield line
import json
metadata = get_metadata()

for paper in metadata:

    paper_dict = json.loads(paper)

    print('Title: {}\n\nAbstract: {}\nRef: {}'.format(paper_dict.get('title'), paper_dict.get('abstract'), paper_dict.get('journal-ref')))

#     print(paper)

    break
titles = []

abstracts = []

years = []

metadata = get_metadata()

for paper in metadata:

    paper_dict = json.loads(paper)

    ref = paper_dict.get('journal-ref')

    try:

        year = int(ref[-4:]) 

        if 2010 < year < 2021:

            years.append(year)

            titles.append(paper_dict.get('title'))

            abstracts.append(paper_dict.get('abstract'))

    except:

        pass 



len(titles), len(abstracts), len(years)
cpu_stats()
papers = pd.DataFrame({

    'title': titles,

    'abstract': abstracts,

    'year': years

})

papers
del titles
del abstracts
del years
cpu_stats()
papers.to_json('arXiv_title_abstract_20200809_2011_2020.json')
val_df = papers.sample(frac=0.1, random_state=1007)

train_df = papers.drop(val_df.index)

test_df = train_df.sample(frac=0.1, random_state=1007)

train_df.drop(test_df.index, inplace=True)
del papers
cpu_stats()
train_df
val_df
test_df
train_df.shape[0], val_df.shape[0], test_df.shape[0]
train_df.info()
cpu_stats()