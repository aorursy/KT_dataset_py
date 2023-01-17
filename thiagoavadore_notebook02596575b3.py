# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import csv

import io

import requests

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
url_train = "https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3136/train.csv?sv=2015-12-11&sr=b&sig=GjaocykT0oDMUbq1X8gCMsiJrbCoepjT1kz2ZVSpAfs%3D&se=2016-12-15T16%3A56%3A32Z&sp=r"

s = requests.get(url_train).content

titanic_df = pd.read_csv(io.StringIO(s.decode('utf-8')))



titanic_df.head()
url_test = "https://kaggle2.blob.core.windows.net/competitions-data/kaggle/3136/test.csv?sv=2015-12-11&sr=b&sig=ktq5H%2BAUhThZCmRSvlTuqZJpyDsKhsRYDNPVnCfJSqo%3D&se=2016-12-15T17%3A01%3A21Z&sp=r"

s = requests.get(url_test).content

test_df = pd.read_csv(io.StringIO(s.decode('utf-8')))



test_df.head()