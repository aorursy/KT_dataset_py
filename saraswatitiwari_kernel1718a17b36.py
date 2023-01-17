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
from IPython.display import YouTubeVideo



YouTubeVideo('HctunZLmc10', width=800/1.2, height=450/1.25)
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import re

import os

import string

import pickle





import nltk

import gensim



from nltk.tokenize import word_tokenize, sent_tokenize

from gensim.parsing.preprocessing import remove_stopwords

from nltk.stem import PorterStemmer

from nltk.stem import LancasterStemmer



from gensim.test.utils import common_corpus, common_dictionary

from gensim.similarities import MatrixSimilarity



from gensim.test.utils import datapath, get_tmpfile

from gensim.similarities import Similarity



from IPython.display import display, Markdown, Math, Latex, HTML





import pandas as pd

import seaborn as sns



!pip install webdriverdownloader

from webdriverdownloader import GeckoDriverDownloader



!pip install selenium

from selenium.webdriver.common.by  import By as selenium_By

from selenium.webdriver.support.ui import Select as selenium_Select

from selenium.webdriver.support.ui import WebDriverWait as selenium_WebDriverWait

from selenium.webdriver.support    import expected_conditions as selenium_ec

from IPython.display import Image







from selenium import webdriver as selenium_webdriver

from selenium.webdriver.firefox.options import Options as selenium_options

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities as selenium_DesiredCapabilities



from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()





# YOU MUST ADD YOUR USERNAME AND PASSWORD OF RESEARCH GATE TO THE SECRET CREDENTIALS TO BE ABLE TO GET THE SCRAPED DATA

# email = user_secrets.get_secret("email")

# password = user_secrets.get_secret("pass")

email = user_secrets.get_secret("email")

password = user_secrets.get_secret("pass")
meta = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

print("Cols names: {}".format(meta.columns))

meta.head(7)
plt.figure(figsize=(20,10))

meta.isna().sum().plot(kind='bar', stacked=True)
meta.columns
meta_dropped = meta.drop(['who_covidence_id'], axis = 1)

plt.figure(figsize=(20,10))



meta_dropped.isna().sum().plot(kind='bar', stacked=True)
miss = meta['abstract'].isna().sum()

print("The number of papers without abstracts is {:0.0f} which represents {:.2f}% of the total number of papers".format(miss, 100* (miss/meta.shape[0])))
abstracts_papers = meta[meta['abstract'].notna()]

print("The total number of papers is {:0.0f}".format(abstracts_papers.shape[0]))

missing_doi = abstracts_papers['doi'].isna().sum()

print("The number of papers without doi is {:0.0f}".format(missing_doi))

missing_url = abstracts_papers['url'].isna().sum()

print("The number of papers without url is {:0.0f}".format(missing_url))
abstracts_papers = abstracts_papers[abstracts_papers['publish_time'].notna()]

abstracts_papers['year'] = pd.DatetimeIndex(abstracts_papers['publish_time']).year
missing_url_data = abstracts_papers[abstracts_papers["url"].notna()]

print("The total number of papers with abstracts, urls, but missing doi = {:.0f}".format( missing_url_data.doi.isna().sum()))
abstracts_papers = abstracts_papers[abstracts_papers["url"].notna()]