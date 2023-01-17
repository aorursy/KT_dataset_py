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
for size in range(1,10):
    length = 10 ** size
    with open("../input/2012_unemployment.csv", 'rb') as rawdata:
        result = chardet.detect(rawdata.read(length))

    # check what the character encoding might be
    print(size, length, result)



listOfDf = []

for year in range(2012,2017):
    mydf = pd.read_csv("../input/" + str(year) + "_unemployment.csv", encoding = 'ISO-8859-1',   thousands='.', decimal = ',')
    mydf['Year'] = year
    listOfDf.append(mydf)

df = pd.concat(listOfDf)
df.head()

# https://stackoverflow.com/questions/11763204/how-to-efficiently-handle-european-decimal-separators-using-the-pandas-read-csv

df = df[df.Desembre != '-']

for colname in df.columns[3:15]:
    df[colname] = df[colname].astype(str)
    df[colname] = [x.replace(',', '.') for x in df[colname]]
    df[colname] = df[colname].astype(float)

    print(colname) 
df.head()
df.dtypes
df['Població\n16-64 anys']
df.Barris.unique()
df.groupby('Barris').Gener.sum().sort_values(ascending = False)
