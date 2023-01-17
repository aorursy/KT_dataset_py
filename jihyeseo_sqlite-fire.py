# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
%matplotlib inline
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
filenames = check_output(["ls", "../input"]).decode("utf8").strip()
# helpful character encoding module
import chardet

# set seed for reproducibility
np.random.seed(0)
import sqlite3
conn = sqlite3.connect('../input/FPA_FOD_20170508.sqlite')
# getting help from https://www.dataquest.io/blog/python-pandas-databases/
conn
cur = conn.cursor()
# https://stackoverflow.com/questions/34570260/how-to-get-table-names-using-sqlite3-through-python 
# get table names from ...
res = conn.execute("select name from sqlite_master where type ='table';")
for name in res:
    print(name[0])
cur.execute("select * from Fires limit 5;")
results = cur.fetchall()
print(results)
# https://www.dataquest.io/blog/python-pandas-databases/
df = pd.read_sql_query("select * from Fires;", conn)
df
